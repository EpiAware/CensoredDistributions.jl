# [Interface contracts: valid leaves, composers and modifiers](@id interface-contracts)

[Extending the composer toolkit](@ref extending-composer) walks through writing
a custom leaf and a custom node end to end.
This page is the reference behind it: what makes a type a _valid_ member of each
family, stated as the exact method contract the package checks.

Every family has a machine-checkable conformance suite in
[`CensoredDistributions.TestUtils`](@ref).
Each suite is the canonical definition of its family's contract: the prose here
follows the suite, and the suite is what runs in the package's own tests, so the
documentation and the checks stay in sync.
To add a valid member of a family, subtype the right abstract, implement the
methods listed, and run the matching `test_*_interface` over an instance.

## The abstract hierarchy

Related families share one supertype each.
The concrete types subtype the abstract; the shared behaviour and the documented
contract hang off the abstract.

```text
Distribution{F, S}
├── AbstractComposedDistribution{F, S}   named children → an event tree
│   ├── AbstractMultiChild{S}            positional, tree-walked together
│   │   ├── Sequential                   a chain (steps in series)
│   │   └── Parallel                     a fan-out (branches off one origin)
│   ├── Choose                           disjoint alternatives (data picks one)
│   └── AbstractOneOf                    one univariate time-to-event marginal
│       ├── Resolve                      mixture over competing outcomes
│       └── Compete                      racing hazards (soonest cause fires)
│
├── AbstractModifiedDistribution{F, S}   wrap ONE inner base and modify it
│   ├── Affine            TimeChange      Modified
│   └── Transformed       Weighted        Shared
│
├── AbstractCombinedDistribution{F, S}   combine TWO+ bases algebraically
│   ├── Convolved                        the sum of independent components
│   └── Difference                       Z = X - Y
│
└── AbstractPrimaryCensored              the primary-censored family
    ├── PrimaryCensored                  the primary-event-censored delay
    └── PrimaryConditional               the latent secondary | a realised primary

plain UnivariateDistributions (no shared abstract):
    IntervalCensored        standalone interval censoring
    Reparameterised         a (mean, ...)-reparameterised leaf
    ExponentiallyTilted     a bounded exponentially-tilted base family
```

The composed and modified abstracts are parametric on variate form `F`
(`Univariate` / `Multivariate`), so one supertype spans the univariate and
multivariate members while preserving `Distribution{F, S}`.
`AbstractPrimaryCensored` is univariate-continuous only, so it is non-parametric.

`AbstractMultiChild` is an intermediate: it groups the two positional multi-child
composers (`Sequential`, `Parallel`) the tree walkers dispatch over together.
`Choose` is a sibling, not a multi-child node, and `AbstractOneOf` re-roots the
univariate one_of family under the composed supertype.

The membership relations are themselves a test:
[`test_abstract_membership`](@ref CensoredDistributions.TestUtils.test_abstract_membership)
asserts every built-in type sits under the right family (and that the standalone
leaves sit under none), so a type filed under the wrong supertype fails.

## Composed distributions: `AbstractComposedDistribution`

A composer combines named child distributions into an event tree.
A realisation is one flat vector of leaf values laid out depth-first, and the
composer walks it by an offset (the `child_*` node interface).
The members are the multivariate `Sequential` / `Parallel` / `Choose` and the
univariate one_of family (`AbstractOneOf`: `Resolve` / `Compete`).

A valid composer node implements, per
[`test_composed_interface`](@ref CensoredDistributions.TestUtils.test_composed_interface)
(which wraps both
[`test_node_interface`](@ref CensoredDistributions.TestUtils.test_node_interface)
and
[`test_interface`](@ref CensoredDistributions.TestUtils.test_interface)):

- subtypes [`AbstractComposedDistribution`](@ref CensoredDistributions.AbstractComposedDistribution) (or, for a positional
  multi-child node, [`AbstractMultiChild`](@ref CensoredDistributions.AbstractMultiChild));
- `component_names(c)` returns a `Tuple` of the child names;
- the node interface walks the flat event vector:
  - `CensoredDistributions.child_nleaves(c)` — a positive `Int`, the flat-slot
    count (one per leaf below it);
  - `CensoredDistributions.child_logpdf(c, x, offset, n)` — a finite scalar over
    the node's `n`-wide slice `x[offset + 1 : offset + n]`, independent of the
    surrounding padding;
  - `CensoredDistributions.child_rand!(out, offset, rng, c)` — fills exactly that
    slice in place and returns `nothing`, leaving the padding either side
    untouched;
- `params(c)` and `params_table(c)` (a Tables.jl table);
- `event_names(c)` (the flat path) and `event_tree(c)` (the nested record) agree
  in leaf count;
- `event(c, path...)` round-trips a known name path;
- `Base.show(io, c)` (and the `MIME"text/plain"` form).

### Adding a valid composer node

1. Subtype [`AbstractComposedDistribution`](@ref CensoredDistributions.AbstractComposedDistribution) — or
   [`AbstractMultiChild`](@ref CensoredDistributions.AbstractMultiChild) if the node stores `.components` / `.names` and
   is walked positionally like `Sequential` / `Parallel`.
2. Implement the three `child_*` methods so they read and write only the node's
   own slice; a node delegates to each child by the same methods, passing each
   child its own offset, so it nests inside any other node with no extra work.
3. Implement `component_names`, `params`, `event_names` / `event_tree`, and
   `show`.
4. Verify against the suite:

```julia
using CensoredDistributions, Distributions
using CensoredDistributions.TestUtils: test_composed_interface

node = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
test_composed_interface(node; draw = rand(node), path = (:onset_admit,),
    overall = :vector, latent_moments = true, has_endpoint = false)
```

[Writing a new composer node](@ref new-composer-node) builds a minimal `Both`
node through the `child_*` contract in full.

## Single-base modifiers: `AbstractModifiedDistribution`

A modifier wraps one inner base distribution and modifies it: `Affine`,
`TimeChange`, `Modified`, `Transformed`, `Weighted`, `Shared`.
This family is slated to move to ModifiedDistributions.jl (issue #726), so the
core censoring wrappers deliberately do not live here ([`PrimaryCensored`](@ref CensoredDistributions.PrimaryCensored)
is under `AbstractPrimaryCensored`; [`IntervalCensored`](@ref CensoredDistributions.IntervalCensored) is standalone).

A valid modifier implements, per
[`test_modified_interface`](@ref CensoredDistributions.TestUtils.test_modified_interface):

- subtypes [`AbstractModifiedDistribution`](@ref CensoredDistributions.AbstractModifiedDistribution);
- an inner base reachable as `.dist` (the default `show` accessor; a subtype
  that stores it elsewhere overrides `CensoredDistributions._modified_inner`);
- `CensoredDistributions.free_leaf(d)` returns the free inner leaf (a
  `Distribution`), and `CensoredDistributions.rewrap_leaf(d, free_leaf(d))`
  reconstructs an equivalent node whose `logpdf` matches `d`;
- `get_dist(d)` returns the underlying distribution;
- the univariate interface (`pdf` / `logpdf` / `cdf` / `quantile` / `minimum` /
  `maximum` / `insupport` / `params`), forwarded or specialised;
- `params(d)` is a `Tuple`, and `Base.show` is non-empty (the default prints
  `Name(inner)`).

### Adding a valid modifier

1. Subtype [`AbstractModifiedDistribution`](@ref CensoredDistributions.AbstractModifiedDistribution) and store the inner base as
   `.dist` (or override `_modified_inner`).
2. Implement `free_leaf` / `rewrap_leaf` so peeling to the inner leaf and
   rebuilding round-trips the density, and `get_dist`.
3. Forward or specialise the univariate interface and `params`.
4. Verify against the suite:

```julia
using CensoredDistributions, Distributions
using CensoredDistributions.TestUtils: test_modified_interface

d = affine(Gamma(2.0, 1.0); scale = 2.0, shift = 1.0)
test_modified_interface(d; x = 5.0)
```

## Primary censoring: `AbstractPrimaryCensored`

The primary-censored family the package dispatches on: [`PrimaryCensored`](@ref CensoredDistributions.PrimaryCensored)
(the primary-event-censored delay) and the latent `PrimaryConditional` (the
secondary conditioned on a realised primary).
This is core to the package, distinct from interval censoring, and stays in it.
It is univariate and continuous, so non-parametric.

A valid member implements, per
[`test_primary_censored_interface`](@ref CensoredDistributions.TestUtils.test_primary_censored_interface):

- subtypes `AbstractPrimaryCensored`;
- `get_dist(d)` returns the underlying delay distribution;
- `params(d)` is a `Tuple`;
- `logpdf(d, x)` is finite on its support;
- `Base.show(io, d)` is non-empty.

`double_interval_censored` is a constructor function, not a type: it returns a
pipeline whose object type is the outer wrapper, so there is no
`DoubleIntervalCensored` type under this supertype.

### Adding a valid primary-censored type

```julia
using CensoredDistributions, Distributions
using CensoredDistributions.TestUtils: test_primary_censored_interface

d = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
test_primary_censored_interface(d; x = 2.0)
```

## Multi-base combinations: `AbstractCombinedDistribution`

A combination joins two or more base distributions by an algebraic operation:
[`Convolved`](@ref CensoredDistributions.Convolved) (the sum of independent components) and `Difference`
(`Z = X - Y`).
This is distinct from the single-base modifier leaves and from the named-child
event-tree composers.

A valid member implements, per
[`test_combined_interface`](@ref CensoredDistributions.TestUtils.test_combined_interface):

- subtypes [`AbstractCombinedDistribution`](@ref CensoredDistributions.AbstractCombinedDistribution);
- `params(d)` is a `Tuple`;
- `logpdf(d, x)` is finite on its support;
- `Base.show(io, d)` is non-empty.

### Adding a valid combination

```julia
using CensoredDistributions, Distributions
using CensoredDistributions.TestUtils: test_combined_interface

d = convolved(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
test_combined_interface(d; x = 3.0)
```

## Plain leaves: no shared abstract

Three core types are standalone `UnivariateDistribution`s under no shared
family abstract, each for a documented reason:

- [`IntervalCensored`](@ref CensoredDistributions.IntervalCensored) — interval censoring is distinct from primary
  censoring and stays in the package independently;
- `Reparameterised` — a `(mean, ...)`-reparameterised single-base leaf, owned by
  ReparameterisedDistributions and re-exported here, a reparameterisation rather
  than a combination;
- `ExponentiallyTilted` — a bounded exponentially-tilted base family.

Any plain univariate distribution (these, or any `Distributions.jl` leaf) is a
valid leaf with no package-specific hooks: implement the standard univariate
interface and `params`.
Verify it directly with
[`test_interface`](@ref CensoredDistributions.TestUtils.test_interface)
(`univariate = true`), as
[Extending the composer toolkit](@ref extending-composer) shows.
`test_abstract_membership` pins these three under no family supertype, so filing
one under a family by mistake fails.

## Keeping the hierarchy honest

[`test_abstract_membership`](@ref CensoredDistributions.TestUtils.test_abstract_membership)
is the meta-test for the whole hierarchy: every composer subtypes
`AbstractComposedDistribution`, every single-base modifier subtypes
`AbstractModifiedDistribution`, the primary-censored family subtypes
`AbstractPrimaryCensored`, the multi-base combinations subtype
`AbstractCombinedDistribution`, and `IntervalCensored` / `Reparameterised` /
`ExponentiallyTilted` stay standalone.
Run it after adding a type to a family, alongside that family's
`test_*_interface`.

```julia
using CensoredDistributions.TestUtils: test_abstract_membership

test_abstract_membership()
```

## Per-family suite reference

The per-family conformance suites are the canonical contract for each family.

```@docs
CensoredDistributions.TestUtils.test_composed_interface
CensoredDistributions.TestUtils.test_modified_interface
CensoredDistributions.TestUtils.test_primary_censored_interface
CensoredDistributions.TestUtils.test_combined_interface
CensoredDistributions.TestUtils.test_abstract_membership
```
