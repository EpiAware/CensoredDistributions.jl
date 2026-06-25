# [Extending the composer toolkit](@id extending-composer)

[Composing censored distributions](@ref composer-toolkit) shows how to build
records from the distributions the package ships.
This page is the next step: writing your own leaf distribution so it plugs into
[`compose`](@ref), the composers, [`params_table`](@ref) and the Turing models.

## The one-line check: `test_interface`

The package ships its whole interface checklist as the public
[`CensoredDistributions.TestUtils`](@ref) submodule.
Add a leaf or node, run [`test_interface`](@ref CensoredDistributions.TestUtils.test_interface) on it, and the same harness that guards the package's own types in CI proves your type conforms.
That is the headline: write a type, run one function, read the result.

```@example extending
using CensoredDistributions, Distributions, Random
using CensoredDistributions.TestUtils: test_interface

# Any Distributions.jl univariate distribution is already a valid leaf.
test_interface(Gamma(2.0, 1.0); name = "gamma leaf", draw = 3.0,
    univariate = true, overall = :scalar)
nothing # hide
```

The checklist asserts the moments, a finite `logpdf` on an in-support draw, a monotone CDF in `[0, 1]`, that [`params_table`](@ref) is a Tables.jl table, the event-name layout, and — when an `ad` probe is supplied with an injected backend — that `logpdf` differentiates with a finite gradient.
A defective (sub-stochastic) leaf, a missing-sentinel record and the deep-nesting matrix have their own checks too; [`CensoredDistributions.TestUtils.example_fixtures`](@ref) is the package's own fixture set and doubles as worked examples of the metadata each shape needs.
A new composer node has the companion [`test_node_interface`](@ref CensoredDistributions.TestUtils.test_node_interface) (see [Writing a new composer node](@ref new-composer-node)).

The rest of this page builds a real custom leaf and node and verifies each with these checks.

## The contract

A composer treats every branch as a univariate distribution, so any type
satisfying the standard `Distributions.jl` interface composes with no extra
hooks.
The package-specific hooks are optional refinements: they make a censored or
reparameterised leaf transparent to parameter introspection.

## The extension points

### The Distributions.jl contract (required)

A custom leaf is a `ContinuousUnivariateDistribution` (or discrete).
To take part in fitting, simulation and moments it implements the standard
methods the composers call:

- `Distributions.logpdf(d, x)` and `Distributions.cdf(d, x)`: scoring a record
  and the monotone CDF the harness checks.
- `Base.rand(rng, d)`: a draw, used when a composer simulates a record.
- `Distributions.quantile(d, q)`: used by truncation and discretisation helpers.
- `Base.minimum(d)` / `Base.maximum(d)` and `Distributions.insupport(d, x)`: the
  variate support, which [`params_table`](@ref) reports per edge as the domain a
  prior must respect.
- `Distributions.mean(d)` / `Distributions.var(d)`: the per-edge moments
  `mean(d)` and `mean(latent(d))` read off the tree.
- `Distributions.params(d)`: the scalar parameters, flattened by
  [`params_table`](@ref) into one row each.

That is the whole requirement for composition.
A composer does not introduce a new leaf type or a registration step; it slots
your distribution in wherever a delay is expected.

### The package hooks (optional)

These are reached by the qualified internal name (they are not exported), and
exist only to keep a wrapped leaf transparent to introspection:

- `CensoredDistributions._param_names(d)`: the scalar parameter names, matched
  positionally to `params(d)`, so [`params_table`](@ref) labels your rows (e.g.
  `:shift`, `:scale`) instead of the positional fallback `:param_1`, `:param_2`.
- `CensoredDistributions.free_leaf(d)` / `CensoredDistributions.rewrap_leaf(d,
  inner)`: peel a wrapper off to its inner free delay and rebuild it. The
  censoring wrappers ([`primary_censored`](@ref), [`interval_censored`](@ref),
  truncation) define these so a censored leaf shows only its inner delay's
  parameters in [`params_table`](@ref) and round-trips through [`update`](@ref).
  A plain leaf is the identity for both and needs neither; only a new wrapper
  type around another distribution implements them.

- `CensoredDistributions.child_nleaves(node)` /
  `CensoredDistributions.child_logpdf(node, x, offset, n)` /
  `CensoredDistributions.child_rand!(out, offset, rng, node)`: the contract for a
  new composer node, a new way to combine branches rather than a new leaf.
  They are how the composers walk the flat event vector, and you implement them
  only when writing a node; a leaf reaches them through the existing
  `UnivariateDistribution` methods.
  [Writing a new composer node](@ref new-composer-node) below works through them.

## A worked custom leaf

We write a shifted exponential: an `Exponential(scale)` delay that cannot occur
before a `shift`.
It implements the Distributions.jl contract and nothing else.

```@example extending
using CensoredDistributions, Distributions, Random

struct ShiftedExponential{T <: Real} <: ContinuousUnivariateDistribution
    shift::T
    scale::T
end

Distributions.params(d::ShiftedExponential) = (d.shift, d.scale)
Base.minimum(d::ShiftedExponential) = d.shift
Base.maximum(::ShiftedExponential) = Inf
Distributions.insupport(d::ShiftedExponential, x::Real) = x >= d.shift

function Distributions.logpdf(d::ShiftedExponential, x::Real)
    x < d.shift && return -Inf
    return logpdf(Exponential(d.scale), x - d.shift)
end

function Distributions.cdf(d::ShiftedExponential, x::Real)
    x < d.shift && return 0.0
    return cdf(Exponential(d.scale), x - d.shift)
end

function Distributions.quantile(d::ShiftedExponential, q::Real)
    return d.shift + quantile(Exponential(d.scale), q)
end

Distributions.mean(d::ShiftedExponential) = d.shift + mean(Exponential(d.scale))
Distributions.var(d::ShiftedExponential) = var(Exponential(d.scale))

function Base.rand(rng::AbstractRNG, d::ShiftedExponential)
    return d.shift + rand(rng, Exponential(d.scale))
end

leaf = ShiftedExponential(1.0, 2.0)
(mean = mean(leaf), logpdf_at_3 = logpdf(leaf, 3.0))
```

### Composing it into a tree

The custom leaf drops into [`compose`](@ref) wherever a delay is expected, here
as one branch of a two-branch record.

```@example extending
tree = compose((onset_admit = leaf,
    admit_death = Gamma(2.0, 1.0)))

event_names(tree)
```

It fits and simulates like any other branch.

```@example extending
rand(Xoshiro(1), tree)
```

### Labelling its parameters

[`params_table`](@ref) finds the leaf's parameters automatically.
Without a name hook the rows fall back to positional names.

```@example extending
params_table(tree)
```

Extending the `_param_names` hook gives them readable labels.

```@example extending
CensoredDistributions._param_names(::ShiftedExponential) = (:shift, :scale)

params_table(tree)
```

With named rows, [`build_priors`](@ref) derives a default prior per parameter
from its support, the same as for a built-in leaf.

```@example extending
build_priors(params_table(tree))
```

## [Writing a new composer node](@id new-composer-node)

A custom leaf is one delay.
A custom node is a new way to combine branches, alongside the built-in
[`Sequential`](@ref), [`Parallel`](@ref), [`Resolve`](@ref) and
[`choose`](@ref).
Where a leaf takes part through the `Distributions.jl` methods, a node plugs in
through three public methods.

A realisation of any composer is one flat vector of leaf values laid out
depth-first, and a nested child contributes its own contiguous sub-vector, so
combining branches is concatenation.
A node walks that flat vector by an offset into it.
The three methods are the whole contract.

- `CensoredDistributions.child_nleaves(node)`: how many flat slots the node
  occupies (one per leaf below it).
- `CensoredDistributions.child_logpdf(node, x, offset, n)`: the node's
  contribution to the joint log density, reading its `n`-wide slice
  `x[offset + 1 : offset + n]`.
- `CensoredDistributions.child_rand!(out, offset, rng, node)`: draw the node in
  place into that same slice.

They are reached by the qualified name, the same way the leaf hooks `free_leaf` /
`rewrap_leaf` are.
A node delegates to its children by the same methods, passing each child its own
offset, so a node nests inside any other node without extra work.

We write a minimal `Both` node: two independent branches scored side by side,
the essence of how [`Parallel`](@ref) lays out a flat vector.

```@example extending
import CensoredDistributions: child_nleaves, child_logpdf, child_rand!

struct Both{A, B}
    first::A
    second::B
end
```

The node's width is the sum of its children's widths, found through the same
`child_nleaves` the composers use, so a child that is itself a node counts its
whole subtree.

```@example extending
child_nleaves(b::Both) = child_nleaves(b.first) + child_nleaves(b.second)
```

Scoring walks the flat vector left to right.
The first child reads the slice at `offset`; the second starts where the first
ends, at `offset + n1`.
Both children are scored by `child_logpdf`, so a leaf reads one scalar and a
nested node recurses.

```@example extending
function child_logpdf(b::Both, x, offset, ::Int)
    n1 = child_nleaves(b.first)
    n2 = child_nleaves(b.second)
    return child_logpdf(b.first, x, offset, n1) +
           child_logpdf(b.second, x, offset + n1, n2)
end
```

Drawing fills the same two slices in place.

```@example extending
function child_rand!(out, offset, rng::AbstractRNG, b::Both)
    n1 = child_nleaves(b.first)
    child_rand!(out, offset, rng, b.first)
    child_rand!(out, offset + n1, rng, b.second)
    return nothing
end
```

The node now combines any two branches, leaves or nested nodes.

```@example extending
node = Both(ShiftedExponential(1.0, 2.0), Gamma(2.0, 1.0))

out = fill(0.0, child_nleaves(node))
child_rand!(out, 0, Xoshiro(3), node)
(draw = out, logpdf = child_logpdf(node, out, 0, length(out)))
```

### Verifying the node

[`test_node_interface`](@ref CensoredDistributions.TestUtils.test_node_interface)
is the node companion to the leaf `test_interface`.
It asserts the three methods round-trip on a flat event vector: `child_nleaves`
is a positive count, `child_rand!` fills exactly the node's slot and leaves the
surrounding vector untouched, and `child_logpdf` is finite and reads only that
slot.

```@example extending
using CensoredDistributions.TestUtils: test_node_interface

test_node_interface(node; name = "Both")
nothing # hide
```

Because the methods are plain arithmetic over a vector, the node differentiates
on the same backends a leaf does.

```@example extending
using ForwardDiff

ForwardDiff.gradient([1.0, 2.0, 2.0]) do p
    n = Both(ShiftedExponential(p[1], p[2]), Gamma(p[3], 1.0))
    child_logpdf(n, out, 0, length(out))
end
```

## Editing a composed tree

The edit verbs ([`update`](@ref), [`prune`](@ref), [`splice`](@ref)) and the
path-addressing rule are tabulated in the syntax reference of
[Composing censored distributions](@ref composer-toolkit); this section covers
only the internals behind them.

A composed tree is immutable, so an edit returns a fresh tree rather than
mutating in place.
Each verb walks the tree by name path and rebuilds only the touched spine, so
the result is still a valid composed distribution that scores and `rand`s.
The shape-preserving edits ([`update`](@ref)) share one recursive
reconstruction, dispatching on whether the second argument is a parameter
`NamedTuple` or a `path => new_node` replacement.
The topology edits ([`prune`](@ref), [`splice`](@ref)) rebuild the spine around
the touched node, `prune` renormalising the remaining [`Resolve`](@ref)
probabilities and `splice` wrapping the node in a [`Sequential`](@ref).

```@example extending
base = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))

# Counterfactual: a different onset-to-admission delay (same shape).
faster = update(base, :onset_admit => Gamma(1.0, 1.0))

# Insert a reporting step after the admission-to-death delay (changes shape).
reported = splice(base, :admit_death;
    after = :death_report => Gamma(1.0, 2.0))

(event_names(base), event_names(event(reported, :admit_death)))
```

## Verifying conformance

The package ships its interface checklist as the public
[`CensoredDistributions.TestUtils`](@ref) submodule, so you can verify a new
leaf or tree against the same checks the package runs on its own fixtures.
Drop [`test_interface`](@ref CensoredDistributions.TestUtils.test_interface)
into your own `@testset`: it asserts the moments,
a finite `logpdf` on an in-support draw, a monotone CDF, that
[`params_table`](@ref) is a Tables.jl table, and that the names line up.

```@example extending
using CensoredDistributions.TestUtils: test_interface, test_rejects_invalid

test_interface(leaf; name = "shifted exponential", draw = 3.0,
    univariate = true, overall = :scalar)
nothing # hide
```

The same call verifies the composed tree.
Pass the metadata the checklist cannot recover from the object alone: an
in-support `draw`, a known `event` `path`, the overall-moment shape (`:vector`
for a `Parallel` of independent endpoints) and whether the per-event
`mean(latent(d))` view applies.

```@example extending
test_interface(tree; name = "shifted-exponential tree",
    draw = rand(Xoshiro(2), tree), path = (:onset_admit,),
    overall = :vector, latent_moments = true, has_endpoint = false)
nothing # hide
```

[`test_rejects_invalid`](@ref CensoredDistributions.TestUtils.test_rejects_invalid)
is the companion check that the standard
composers reject malformed construction (too few branches, out-of-range
probabilities, duplicate names).

```@example extending
test_rejects_invalid()
nothing # hide
```

A new composer node has its own positive conformance check,
[`test_node_interface`](@ref CensoredDistributions.TestUtils.test_node_interface),
shown above in [Writing a new composer node](@ref new-composer-node): it asserts
the node's `child_nleaves` / `child_logpdf` / `child_rand!` methods round-trip on
a flat event vector, the way the composers walk one.

See [`CensoredDistributions.TestUtils.example_fixtures`](@ref) for the package's
own fixture set, which doubles as a set of worked examples of the metadata each
composer shape needs.

## Summary

- Any `Distributions.jl` univariate distribution composes with no
  package-specific hooks: implement `logpdf`, `cdf`, `rand`, `quantile`,
  support, moments and `params`.
- The optional `_param_names` hook labels a custom leaf's [`params_table`](@ref)
  rows; `free_leaf` / `rewrap_leaf` keep a new wrapper leaf transparent to
  introspection.
- A new composer node implements the public `child_nleaves` / `child_logpdf` /
  `child_rand!` contract to combine branches into the flat event vector, the same
  way the built-in composers do.
- Verify a new leaf or tree with the public
  [`test_interface`](@ref CensoredDistributions.TestUtils.test_interface) /
  [`test_rejects_invalid`](@ref CensoredDistributions.TestUtils.test_rejects_invalid)
  harness, and a new node with
  [`test_node_interface`](@ref CensoredDistributions.TestUtils.test_node_interface),
  the same checks the package runs on its own fixtures.

## Conformance harness reference

The conformance harness is the public `CensoredDistributions.TestUtils`
submodule.

```@docs
CensoredDistributions.TestUtils
CensoredDistributions.TestUtils.test_interface
CensoredDistributions.TestUtils.test_rejects_invalid
CensoredDistributions.TestUtils.test_node_interface
CensoredDistributions.TestUtils.test_ad_safety
CensoredDistributions.TestUtils.test_registry_coverage
CensoredDistributions.TestUtils.example_fixtures
```
