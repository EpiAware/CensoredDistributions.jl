# [Extending the composer toolkit](@id extending-composer)

[Composing censored distributions](@ref composer-toolkit) shows how to build
records from the distributions the package ships.
This page is the next step: writing your own leaf distribution so it plugs into
[`compose`](@ref), the composers, [`params_table`](@ref) and the Turing models,
and checking it conforms with the public test harness.

The headline is short.
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

The composer-internal `_child_nleaves` / `_child_logpdf` / `_child_rand!`
methods (in `src/composers/nesting.jl`) are how the composers walk the flat
event vector. You implement these only when writing a new composer node (a new
way to combine branches), not a new leaf; a leaf reaches them through the
existing `UnivariateDistribution` methods.

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

## Editing a composed tree

A composed tree is immutable, so an edit returns a fresh tree rather than
mutating in place.
Each verb walks the tree by name path and rebuilds only the touched spine, so
the result is still a valid composed distribution that scores and `rand`s.
The edits split into two kinds: those that keep the tree shape and those that
change it.

| Verb | Edit | Changes shape? |
|---|---|---|
| [`update`](@ref)`(d, params::NamedTuple)` | replace free parameter values | no |
| [`update`](@ref)`(d, path => new_node)` | replace whole nodes | no |
| [`prune`](@ref)`(d, path)` | drop a branch (renormalise a `Competing` arm) | yes |
| [`splice`](@ref)`(d, path; before, after)` | insert a before/after step | yes |

[`update`](@ref) is the single verb for both shape-preserving edits.
A nested `NamedTuple` replaces free parameter values; `path => new_node` pairs
replace whole nodes.
Both dispatch on the second argument, sharing the recursive reconstruction.

[`prune`](@ref) and [`splice`](@ref) are the two topology edits.
`prune` drops a branch, renormalising the remaining [`Competing`](@ref)
probabilities.
`splice` wraps a node in a [`Sequential`](@ref) with a `before` and/or `after`
step, inserting an extra delay without rebuilding the rest of the tree.

A path is addressed the same way [`event`](@ref) reads it: a `Symbol` (a
top-level child), a dotted `Symbol` (`:admit_path.admit_death`), or a tuple of
edge names from the root.
So the address `event` reads is the one `update` / `prune` / `splice` write.

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

`intervene`, `swap_child`, and `cut_branch` are deprecated aliases of `update`
and `prune`, kept during the deprecation window.

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
probabilities, duplicate names); run it if you add a new composer node.

```@example extending
test_rejects_invalid()
nothing # hide
```

See [`CensoredDistributions.TestUtils.example_fixtures`](@ref) for the package's
own fixture set, which doubles as a set of worked examples of the metadata each
composer shape needs.

## Summary

- Any `Distributions.jl` univariate distribution composes with no
  package-specific hooks: implement `logpdf`, `cdf`, `rand`, `quantile`,
  support, moments and `params`.
- The optional `_param_names` hook labels a custom leaf's [`params_table`](@ref)
  rows; `free_leaf` / `rewrap_leaf` keep a new wrapper leaf transparent to
  introspection. The `_child_*` methods are for new composer nodes, not leaves.
- Verify a new leaf or tree with the public
  [`test_interface`](@ref CensoredDistributions.TestUtils.test_interface) /
  [`test_rejects_invalid`](@ref CensoredDistributions.TestUtils.test_rejects_invalid)
  harness, the same checks the package runs on its own fixtures.

## Conformance harness reference

The conformance harness is the public `CensoredDistributions.TestUtils`
submodule.

```@docs
CensoredDistributions.TestUtils
CensoredDistributions.TestUtils.test_interface
CensoredDistributions.TestUtils.test_rejects_invalid
CensoredDistributions.TestUtils.example_fixtures
```
