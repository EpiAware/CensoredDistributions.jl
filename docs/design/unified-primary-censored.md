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
