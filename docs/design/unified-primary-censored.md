# Unified `primary_censored` engine: a grammar of delay graphs

This note synthesises the in-flight delay-graph work
(`SequentialDistribution` #309, `ParallelPrimaryCensored` #317,
`EventTree` #298/#318, CFR/completeness #300) into a single design.
The idea, from the maintainer, is that every member of the family is the
same object: primary censoring of an origin event, followed by a structured
graph of delays.
One constructor, `primary_censored`, dispatches on the *shape* of its first
argument and on a small set of tagged node kinds, and the whole family
becomes one composable grammar.

This is a design synthesis with a light prototype, not a finished
refactor.
It records the recommendation, the exact dispatch, how the existing types
map on, the maths, and the decisions that need the maintainer's call.

## The three node kinds

A delay graph is built from three kinds of node.
All three hang off the same `primary_censored` engine and compose
recursively.

### 1. Conjunctive nodes (shape-dispatched, every branch happens)

A conjunctive node is one where all of its outgoing delays occur.
It is selected purely by the *shape* of the container passed to
`primary_censored(container, primary_event)`:

| Container shape | Meaning | Existing type |
|---|---|---|
| single `UnivariateDistribution` | one delay (today's univariate case) | `PrimaryCensored` |
| `Vector` of delays | **sequential chain** `E0 -> E1 -> ... -> Ek` | `SequentialDistribution` |
| one-row `Matrix` (`1 x n`) | **parallel** branches from a shared origin | `ParallelPrimaryCensored` |
| `Matrix` (`m x n`) | **grid**: `m` sequential stages, each `n`-way parallel | (grid of the two) |
| **nested container** (container inside a container) | **arbitrary branching tree**: recursion depth = branching depth | `EventTree` |

The recursion is the key: a `Vector` whose entries are themselves `Vector`s
or `Matrix`es is an arbitrary tree, because each level of nesting is one
level of branching.
A regular `Matrix` is just the special case of a tree that happens to be a
full grid.

The maths of a conjunctive node is a product of independent factors over the
edges, with the missingness rule (below) deciding, per edge, whether the
factor is a single-delay density or a convolution over a marginalised run.
For the chain this is exactly what `SequentialDistribution.logpdf` already
does; for the tree it is the same sum over edges with the segment grouping
following the tree topology instead of a single path.

### 2. Disjunctive nodes (weighted mixture, exactly one branch happens)

A disjunctive node is the CFR / split node (#300).
Unlike conjunctive nodes, only **one** outgoing branch happens, chosen with
probability `p_i`.
This carries data (the branch probabilities), so it is *not* pure-shape: it
is a tagged node, not a bare container.

Proposed spelling:

```julia
primary_censored(
    Disjunctive([p1 => branch1, p2 => branch2, ...]),
    primary_event)
```

where `sum(pi) == 1` (validated at construction) and each `branch_i` is
itself any node (single delay, chain, parallel, tree, or another
disjunctive node).

Semantics:

- `logpdf` is a log-sum-exp mixture over the branches:
  `logpdf(d, x) = logsumexp_i (log p_i + logpdf(branch_i, x))`.
  When the realised branch is *known* (e.g. the outcome death-vs-discharge
  is recorded) the mixture collapses to the single observed branch and the
  log-density gains the `log p_i` factor for that branch.
- `rand` samples a branch index `i ~ Categorical(p)` then draws the path
  from `branch_i`, returning both the branch label and the path.

The CFR is the two-branch case: `[cfr => fatal_chain, 1 - cfr => recovery_chain]`.
This is `MixtureModel` arithmetic, but tagged so it sits inside the same tree
grammar and so `rand` can report which branch fired.

**Do not implement this node yet** — it is specified here; the existing #300
branch ships the completeness/thinning helpers (`completeness_probability`,
`thin_by_completeness`) but not the mixture node.

### 3. Observation nodes (design-B: what is seen, per record)

Design-B keeps the distribution data-free and decides per record, from the
observation vector's missingness, what to do at each event.
Three behaviours, all already present in pieces across the branches:

- **Marginalise an unobserved intermediate.**
  When the event between two delays is `Missing`, the run of delays it spans
  is convolved and evaluated at the observed gap.
  Marginalising a continuous unobserved intermediate **is** the convolution:
  if `E1` is unobserved, `f(E2 - E0) = (f_{D1} * f_{D2})(E2 - E0)`, the
  density of the sum `D1 + D2`.
  This is `SequentialDistribution._segment_distribution` building a
  `Convolved` over the run, and `truncation.jl._collapse_to_observation`
  doing the same for the truncation denominator.
- **Condition an observed intermediate.**
  When the event is recorded, it cuts the chain there: each adjacent delay
  becomes an independent factor evaluated at its observed gap.
  Conditioning an observed intermediate is a *factor* in the product, not a
  convolution.
- **Ascertainment / completeness = thinning.**
  An event observed only with probability `q` is design-B missingness
  carrying a completeness weight.
  This is the #300 completeness factor: `q = cdf(delay, window)` is the
  probability the chain completed (and so was ascertained) by the horizon,
  and the observed quantity is thinned by `q`
  (`thin_by_completeness`).
  As a node it is a multiplicative weight on the density/count, distinct from
  re-weighting branch probabilities in a disjunctive node.

The reason the tree needs **named** events (the `EventTree` front-end) is
precisely this: design-B has to know *which* event is observed versus missing
to pick marginalise-versus-condition, and a named edge set (`onset -> admit`,
`admit -> death`) is the natural way to address events in an irregular tree
where positional indexing into a vector is ambiguous.

## How the existing types map on

| Today | Under the unified engine |
|---|---|
| `primary_censored(dist, pe)` (single delay) | unchanged: the single-`UnivariateDistribution` overload |
| `sequential_distribution(delays)` | becomes the `primary_censored(::Vector, pe)` overload (delegates to `SequentialDistribution`) |
| `primary_censored(::Vector, pe)` = parallel (**#317 today**) | **flips** to sequential; parallel moves to one-row `Matrix` |
| `ParallelPrimaryCensored` | `primary_censored(::Matrix (1 x n), pe)` overload, type kept |
| grid | `primary_censored(::Matrix (m x n), pe)`, built from chain-of-parallels |
| `EventTree` / `event_tree(root, edges)` | stays as the **named front-end**; lowers to the nested-container engine. Named edges are how irregular trees address events for the design-B marginalise/condition choice |
| CFR / `Disjunctive` | new tagged mixture node (spec only); composes inside any of the above |
| completeness / `thin_by_completeness` | observation-node thinning weight; stays as a helper, usable at any node |

The existing concrete types (`SequentialDistribution`,
`ParallelPrimaryCensored`, `EventTree`) are **kept as the implementation**.
The unification is at the *constructor* layer: `primary_censored` overloads
dispatch on shape and delegate to these types.
`event_tree` and the disjunctive constructor stay as named front-ends because
they carry information (names, probabilities) that pure shape cannot express.

## The #317 vector-flip

This is the one breaking decision and needs the maintainer's explicit call.

- **#317 as it stands:** `primary_censored(::AbstractVector, pe)` returns a
  `ParallelPrimaryCensored` (shared origin, all branches happen).
- **This proposal:** `primary_censored(::AbstractVector, pe)` returns a
  sequential chain; **parallel becomes a one-row `Matrix`**
  (`primary_censored([d1 d2 d3], pe)`, note no commas = a `1 x 3` Matrix
  literal).

Both are defensible. The argument for the flip:

- A `Vector` reads as an ordered sequence, which matches a chain
  (`E0 -> E1 -> E2`) more naturally than a set of parallel branches.
- It makes the `Matrix` dimensions carry real meaning: rows = sequential
  stages, columns = parallel branches, so `m x n` is a grid with no extra
  syntax.
- Nesting then gives arbitrary trees for free.

The cost: #317's vector overload must change before it merges, or ship as
parallel and be flipped later (a breaking change to a public overload).
**Recommendation: decide the vector meaning before #317 merges**, so the
public surface ships consistent. Given the grid/tree pay-off, the flip to
`Vector = sequential` is the better long-run choice.

## Recommendation

Adopt the unification, at the constructor layer, incrementally:

1. Keep the concrete types as the engine.
2. Add `primary_censored` shape overloads that delegate: `Vector` ->
   sequential, one-row `Matrix` -> parallel, `m x n Matrix` -> grid, nested
   -> tree.
3. **Flip #317's vector overload to sequential before it merges** and move
   parallel to the one-row `Matrix`.
4. Keep `event_tree` as the named front-end for irregular trees (bdbv:
   `onset -> {admit, notif}`, `admit -> {death, disch}` is not a regular
   grid, so the shape overloads do not cover it; the named constructor does).
5. Add the `Disjunctive` mixture node (spec above) as a tagged node, not a
   shape; build CFR on it; keep completeness/thinning as an observation-node
   weight.

The win is one mental model and one constructor for the whole family, with
shape doing the easy cases and named/tagged constructors doing the cases
that need extra data.

### Decisions that need the maintainer

1. **`Vector` = sequential or parallel?** (the #317 flip). Recommend
   sequential.
2. **Parallel spelling** once vector is sequential: one-row `Matrix`
   (`[d1 d2 d3]`) versus a tagged `Parallel([...])`. Matrix is terser;
   tagged is more discoverable. Recommend keeping `ParallelPrimaryCensored`'s
   own constructor *and* adding the one-row `Matrix` sugar.
3. **Disjunctive spelling**: `Disjunctive([p => branch, ...])` versus reusing
   `MixtureModel`. Recommend a thin tagged wrapper so `rand` can report the
   fired branch and so it composes in the tree.
4. **Whether shape overloads should subsume `event_tree`** for the regular
   grid case, or whether `event_tree` is the single entry for all multi-event
   trees. Recommend shape for grid, named for irregular.

## Maths check

- **Marginalising a continuous unobserved intermediate = convolution.**
  `E1` unobserved between `D1` (`E0->E1`) and `D2` (`E1->E2`):
  the observed gap `E2 - E0 = D1 + D2`, whose density is
  `(f_{D1} * f_{D2})`, the convolution. Verified by
  `SequentialDistribution` building a `Convolved` over the run and by the
  prototype test below. Right-truncation uses the convolution CDF
  (`truncate_chain`'s convolved-denominator branch).
- **Conditioning an observed intermediate = factor.**
  `E1` observed: the joint factorises into `f_{D1}(E1 - E0) * f_{D2}(E2 - E1)`,
  independent factors, each its own single-delay term (and single-delay
  truncation denominator).
- **Disjunctive = mixture.**
  Exactly one branch with probability `p_i`:
  `f(x) = sum_i p_i f_i(x)`, i.e. `logsumexp_i(log p_i + logpdf_i(x))`;
  collapses to one branch with a `log p_i` factor when the branch is known.
- **Ascertainment = weighted thinning.**
  Observed with probability `q = cdf(delay, window)`; the observed
  count/rate is `q` times the latent one (`thin_by_completeness`), a
  multiplicative observation weight, not a branch re-weighting.

## Prototype

`docs/design/prototype_unified_primary_censored.jl` is a runnable, dependency-free
prototype proving the `primary_censored(::Vector, pe)` = sequential overload
composes by delegating to the existing `SequentialDistribution` machinery.
It shows:

- the overload returns a `SequentialDistribution`;
- `logpdf` with an observed intermediate factorises (conditioning);
- `logpdf` with a `Missing` intermediate marginalises by convolution and
  equals `logpdf(convolve_distributions(D1, D2), gap)` for the marginalised
  run;
- `rand` produces the full event-time path including the unobserved
  intermediate.

It deliberately does **not** rip out the existing types — it adds one method
on top of them.

## Latent vs marginal under composition

The grammar above fixes the *shape* of a delay graph (chain, parallel, grid,
tree, disjunctive) and the design-B observation rule (marginalise unobserved,
condition observed, thin by completeness).
It does not yet say *how* each primary-censored node is computed: by
integrating the unobserved primary out (the marginal formulation) or by
carrying the primary as a sampler-owned latent variable (the latent
formulation).
That choice is the `method` of #316
([`feat/marginal-latent-redesign`](https://github.com/EpiAware/CensoredDistributions.jl/tree/feat/marginal-latent-redesign)):
`method = Auto()` (default) / `Marginal()` / `Latent()` on each
`primary_censored`.
This section specifies how `method` interacts with the composition grammar.

Recall the single-node semantics from #316:

- `Marginal()` is **univariate** over the scalar observed delay.
  Its CDF is the convolution of the delay with the primary event
  distribution, `F(q) = \int_0^w F_\mathrm{delay}(q - p) f_\mathrm{primary}(p)
  \, dp`, computed by `primarycensored_cdf` (analytical pair or quadrature).
- `Latent()` is **multivariate** over `[primary, observed]`.
  `rand` draws a fresh primary then the observed time; `logpdf([p, y]) =
  logpdf(primary_event, p) + logpdf(delay, y - p)` is the joint, prior
  included.
- `Auto()` is **univariate** and dispatches *from the observation*:
  `logpdf(d, [missing, y])` marginalises (quadrature path),
  `logpdf(d, [p, y])` conditions on the concrete primary, and a scalar
  `logpdf(d, y)` takes the marginal path.
- `conditional_logpdf(d, p, y) = logpdf(delay, y - p)` is the prior-excluding
  delay factor used in the decomposed workflow, where the prior is scored once
  by the sampler via `p ~ get_primary_event(d)` rather than twice.

### 1. Scope of `method`: per-node

**`method` is per-node, not whole-object.**
Each `primary_censored` in a composed graph carries its own
`Auto`/`Marginal`/`Latent`.
A single tree may legitimately mix formulations: a latent onset (so the
infection time is a sampled variable that other parts of a joint model can
read) feeding a marginalised onset-to-admission delay (integrated out because
the intermediate is never observed and need not be augmented).

The argument for per-node over whole-object:

- The marginalise-versus-condition choice is already per-edge in design-B
  (it follows that edge's observation), so forcing one formulation across the
  whole object would re-impose a global rule the grammar deliberately
  removed.
- Latent augmentation has a cost (it adds sampler dimensions) that is only
  worth paying where the latent event is shared with the rest of a model.
  An intermediate that no other term reads should marginalise, even if its
  parent onset is latent.
- It composes cleanly with `Auto`: each node resolves its own formulation
  per record from its own slice of the observation, so different nodes in the
  same draw can take different paths.

A node's `method` is therefore a property of that `primary_censored`
constructor call, set when the node is built and stored as the type parameter
`M` on its `PrimaryCensored`.
The container (`Vector`, `Matrix`, nested, `Disjunctive`) carries no `method`
of its own; it only aggregates the per-node densities.

`Auto` is the recommended default at every node.
`Marginal` and `Latent` are the explicit force overrides, used where a node
must be integrated (e.g. an analytical-pair speed-up) or must be augmented
(e.g. a shared onset), respectively.

### 2. Variate form of a composed object

The variate form of the whole composition follows from the per-node forms:

> **A composition is univariate if and only if every node is `Marginal`, or
> `Auto` that resolves to its marginal path for the record at hand.
> It is multivariate the moment any node is `Latent` (or an `Auto` node
> resolves to its conditioning path with a concrete primary supplied).**

The latent (and concretely-conditioned) events are the extra dimensions.
Precisely, the multivariate event vector of a composed object is the
concatenation, in graph order, of:

- for each `Latent` node, its `[primary, observed]` pair (the two latent
  event times of that node);
- for each `Auto` node fed a concrete primary, the same pair (it is being
  conditioned, so the primary is a carried dimension for that record);
- for each marginalised node (`Marginal`, or `Auto` fed `missing`/scalar),
  **no** extra dimension — the primary is integrated away and only the
  observed gap enters, supplied by the design-B observation rather than added
  to the variate.

So the dimensionality is data-dependent only through `Auto`: a tree of all
`Auto` nodes is univariate on a fully-`missing`-primary record and gains two
dimensions per concretely-supplied primary.
A tree with any `Latent` node is multivariate regardless of the record,
because `Latent` always carries its pair.

This matches the single-node rule (`_variate_form`: `Auto`/`Marginal` ->
`Univariate`, `Latent` -> `Multivariate`) lifted to the graph: take the
join over nodes, where `Multivariate` dominates.

### 3. `logpdf` semantics across the composition

The composed log density is built from the per-node densities by the
shape rules already in the grammar, with each node contributing its
`method`-appropriate term.

**Conjunctive nodes (chain / tree / parallel / grid).**
A conjunctive node is a product of independent edge factors, so its log
density is a **sum over edges**:

```math
\log f_\mathrm{node}(\mathbf{x}) =
  \sum_{e \in \mathrm{edges}} \log f_e(\mathbf{x}_e),
```

where each edge term `\log f_e` is:

- a **marginal** factor when that edge's destination event is unobserved and
  the node is marginalising: the convolution density over the marginalised
  run (a `Convolved` evaluated at the observed gap), exactly the
  `_segment_distribution` term;
- a **conditional** factor when that edge's destination event is observed (or
  is a latent draw): the single-delay density `logpdf(delay, gap)` at the
  observed/realised gap.

The latent and marginal cases differ only in whether the gap is a carried
variable (latent / conditioned) or an integrated one (marginal); the edge
*term* is the same convolution-or-single-delay density either way.

**Latent nodes add a prior term.**
A `Latent` (or concretely-conditioned `Auto`) node contributes, in addition
to its delay factor, the primary prior `logpdf(primary_event, p)` for its
carried primary — this is the `[p, y]` joint of #316.
The composed joint over a mixed tree is therefore

```math
\log f(\mathbf{x}) =
  \underbrace{\sum_{n \in \mathrm{latent}}
    \log f_{\mathrm{primary},n}(p_n)}_{\text{priors of carried primaries}}
  + \sum_{e \in \mathrm{edges}} \log f_e(\mathbf{x}_e),
```

i.e. the priors of the latent (data-augmented) primaries plus the edge sum,
where each edge factor is marginal or conditional per design-B.
Marginalised nodes contribute no prior term (the prior is inside the
convolution integral, not a separate factor).

**Disjunctive nodes mix.**
A `Disjunctive`/`MixtureModel` node is a log-sum-exp over its branches,

```math
\log f_\mathrm{disj}(\mathbf{x}) =
  \operatorname{logsumexp}_i \bigl(\log p_i + \log f_{\mathrm{branch}_i}
  (\mathbf{x})\bigr),
```

and collapses to the single observed branch (gaining only that branch's
`\log p_i`) when the fired branch is recorded.
Each branch is itself any node, so its `\log f_{\mathrm{branch}_i}` is built
by these same rules and may contain its own latent and marginal sub-nodes.
A latent dimension that lives *inside* one branch only exists on the draws
where that branch fired; in the marginalised (unknown-branch) mixture it is
integrated within that branch's term.

**Keeping the prior single-counted (the double-count warning).**
The decomposed (latent) workflow scores each latent primary's prior **once**,
via `p ~ get_primary_event(d)` for that node, and then scores the delay with
`conditional_logpdf(node, p, y)` (which excludes the prior).
Do **not** also feed `[p, y]` to that node's joint `logpdf`, and do not sample
`p ~ get_primary_event(d)` *and* score the `[p, y]` joint — either pairing
counts the primary prior twice.
The rule generalises to the tree: for every latent node, the prior appears in
exactly one place — either the joint `[p, y]` term **or** the explicit
`get_primary_event` sample plus `conditional_logpdf` delay, never both.
A composed object built for a sampler should pick one convention per node and
hold it across the tree; mixing the joint form on one node with the
decomposed form on another is fine, but mixing them *on the same node* is the
double-count bug.

### 4. `rand` semantics across the composition

`rand` of a composed object walks the graph and, at each node, does what that
node's `method` dictates:

- a **latent** node draws and returns its realised path `[primary,
  observed]` (a fresh primary, then the observed time), so the latent events
  appear in the output;
- a **marginal** node draws-then-integrates: it samples an observed gap from
  the marginal (convolution) distribution and returns only that gap, with no
  primary in the output (the primary was integrated, so there is nothing to
  realise);
- a **disjunctive** node samples a branch index `i ~ Categorical(p)`, draws
  the path from `branch_i`, and returns the branch label together with that
  path.

The returned structure of a mixed tree is the graph with each node replaced
by its realisation: latent nodes contribute `[primary, observed]` pairs,
marginal nodes contribute scalar observed gaps, disjunctive nodes contribute
`(label, path)`.
This **lines up with the `logpdf` argument layout**: the per-node
`[primary, observed, ...]` slots that `logpdf` reads are exactly the slots
`rand` fills, so a draw round-trips into `logpdf` and scores its own density.
Design-B missingness round-trips the same way: a marginalised node's `rand`
returns no primary, which is exactly the `[missing, y]` (or scalar `y`)
observation its `logpdf` expects; a latent node's `rand` returns `[p, y]`,
which is exactly the conditioned observation its `logpdf` expects.
So `logpdf(d, rand(d))` is well-formed for any mixed tree, and replacing a
latent node's drawn `p` with `missing` switches that node from the
conditioned to the marginalised score without touching the rest of the draw.

### 5. Interaction with truncation, ascertainment and weight

- **Truncation.**
  Right-truncation normalises by the CDF at the truncation horizon.
  Under `Marginal` (and the marginal `Auto` path) the denominator is the
  marginal convolution CDF — for a marginalised run this is the
  convolved-denominator branch of the chain truncation, the CDF of the sum
  over the run.
  Under `Latent` the truncation is conditional on the carried primary: the
  denominator is the delay CDF evaluated at `horizon - p`, so the
  normalisation is a per-draw deterministic function of the sampled primary.
  Truncation is therefore itself per-node and per-formulation: a latent node
  truncates conditionally, a marginal node truncates on the convolution.
- **Ascertainment / completeness.**
  Completeness thinning is a multiplicative observation weight `q` on the
  node's density/count and is **orthogonal** to the marginal/latent choice:
  `q = cdf(delay, window)` (the probability the chain completed by the
  horizon) multiplies whichever density the node produced.
  For a latent node `q` may be evaluated at the carried primary
  (`cdf(delay, horizon - p)`); for a marginal node it is the marginal
  completeness probability.
  It is the #300 `thin_by_completeness` factor and stacks on top of, not
  inside, the formulation term.
- **Weight.**
  `weight(d, w)` raises the node's density to the power `w` (a likelihood
  tempering / replication weight).
  It applies to whatever log density the node produced — marginal scalar,
  latent joint, or mixture — as a final multiplicative factor on the log
  density, so it composes with both formulations unchanged.

Order of application at a node: formulation density, then truncation
normalisation, then completeness thinning, then weight.

### 6. Automatic-differentiation safety

The two formulations stress automatic differentiation differently.

- **Latent** adds sampler-owned dimensions and a **deterministic,
  differentiable joint logpdf**: `logpdf(primary_event, p) + logpdf(delay,
  y - p)` is plain arithmetic in `p` and `y` with no integral, so it
  differentiates on every backend.
  The primary draw is owned by the sampler (it is a value passed in, not
  differentiated through), so no quadrature enters the gradient.
  This is the AD-cheap path and is preferred where a backend struggles with
  the integral.
- **Marginal** adds a nested integral: the convolution CDF goes through the
  `gl_integrate` (Gauss–Legendre) quadrature path (or an analytical pair
  where available).
  The gradient must therefore flow through the quadrature nodes/weights and
  the (possibly central-difference) `logpdf`-of-`logcdf` step.

What must hold across backends:

- `Auto` keeps missingness in **control flow only** — the `missing` is
  inspected with `===`, and only concrete values enter the differentiated
  arithmetic — so a `Union{Missing}` observation never reaches the tape.
  This must remain true through composition: a composed `Auto` tree must
  branch on missingness per node *before* any differentiated term, never
  inside it.
- The quadrature path must use the fixed-node `GaussLegendre` solver (not
  adaptive `QuadGKJL`) for reverse-mode and Enzyme/Mooncake, because adaptive
  node counts are not differentiable.

Known-good / known-broken (carry forward from the single-node #316 status,
to be re-checked under composition):

- ForwardDiff: both formulations expected to work (latent trivially; marginal
  through the fixed-node quadrature).
- ReverseDiff: latent expected to work; marginal works through the
  fixed-node quadrature, with the central-difference `logpdf` the fragile
  step.
- Mooncake: latent expected to work; marginal needs the fixed-node path and
  is the most likely to need a custom rule — **flag as to-verify**.
- Enzyme: latent expected to work; marginal through quadrature is the
  historically brittle combination — **flag as known-fragile / to-verify**.

The composition rule: a mixed tree is AD-safe iff every node is AD-safe under
its own formulation, because the composed log density is a sum (and
log-sum-exp for disjunctive) of the per-node terms, and sums preserve
differentiability.
So preferring `Latent` at nodes a fragile backend cannot integrate is a valid
escape hatch: it trades a sampler dimension for an integral-free gradient at
that node without changing the rest of the tree.

### 7. Worked examples

**(a) Fully-marginal chain.**
An onset-to-admission-to-death chain where neither intermediate is augmented
(everything integrated out).

```julia
using CensoredDistributions, Distributions

# onset -> admission -> death, all Auto, observation marginalises
chain = primary_censored(
    [LogNormal(1.5, 0.5),   # onset -> admission
     Gamma(2.0, 1.5)],      # admission -> death
    Uniform(0, 1))          # daily primary (onset) window

# admission unobserved (missing), death observed at 9.0 days from onset
lp = logpdf(chain, [missing, missing, 9.0])
```

Every node is on its marginal path, so the object is **univariate**: the
density is the convolution `LogNormal * Gamma` evaluated at the observed
onset-to-death gap, normalised by the primary window — one scalar term, no
carried primaries, AD through the quadrature path.

**(b) Latent-onset tree with a marginalised intermediate.**
The onset (infection time) is latent because a wider transmission model reads
it; the onset-to-admission intermediate is marginalised; admission-to-death
is conditioned on an observed admission.

```julia
# onset node is Latent (its primary is a sampled, shared variable);
# the downstream admission->death delay is an Auto node conditioned on the
# observed admission, with onset->admission marginalised between them.
onset = primary_censored(
    LogNormal(1.5, 0.5), Uniform(0, 1); method = Latent())
admit_to_death = primary_censored(Gamma(2.0, 1.5), Uniform(0, 1))

# decomposed (no double-count): score the onset prior once via the sampler,
# the onset delay via the prior-excluding conditional, then the downstream
# factor on the observed admission and death.
# p_onset ~ get_primary_event(onset)
# y_admit observed, y_death observed
# lp = conditional_logpdf(onset, p_onset, y_admit) +
#      logpdf(admit_to_death, [missing, y_death - y_admit])
```

This object is **multivariate**: the latent onset contributes its
`[primary, observed]` pair as carried dimensions; the marginalised
intermediate contributes none; the conditioned admission-to-death contributes
its observed gap.
The onset prior is scored exactly once (via `get_primary_event`), and the
delay uses `conditional_logpdf` so the prior is not double-counted.

**(c) `Competing` CFR node combined with a latent parent.**
A latent onset feeds a disjunctive outcome node (death vs discharge with
case-fatality probability), each branch its own delay.

```julia
# latent onset, then a disjunctive (CFR) node: with probability cfr the
# fatal branch fires, otherwise the recovery branch.
onset = primary_censored(
    LogNormal(1.5, 0.5), Uniform(0, 1); method = Latent())

# outcome = Competing/Disjunctive over the two delays from admission
# (spec-only node; shown for the design language):
# outcome = primary_censored(
#     Competing([cfr        => primary_censored(Gamma(2.0, 1.5)),
#                1 - cfr    => primary_censored(Gamma(3.0, 2.0))]),
#     Uniform(0, 1))
#
# logpdf when the outcome is RECORDED as death:
#   logpdf(primary_event_onset, p) +          # latent onset prior
#   logpdf(delay_onset, y_admit - p) +        # onset -> admission factor
#   log(cfr) +                                # fired-branch probability
#   logpdf(fatal_delay, y_death - y_admit)    # admission -> death factor
#
# logpdf when the outcome is UNKNOWN: log-sum-exp over the two branches,
# each with its own delay factor, sharing the same latent-onset prefix.
```

The latent-onset prefix (prior plus onset delay) is shared by both branches;
the disjunctive node mixes the two outcome delays (collapsing to one branch
with its `log p_i` when the outcome is recorded).
The object is **multivariate** because of the latent onset; the branch label
is an additional reported coordinate from `rand`, and `logpdf` reads it back
as the conditioning that selects the fired branch.
