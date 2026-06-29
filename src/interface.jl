# ============================================================================
# Abstract type hierarchy: composed nodes and modified-distribution leaves
# ============================================================================
#
# Two related families share one supertype each, following the `AbstractOneOf`
# model (concrete types subtype the abstract; shared behaviour and the
# documented interface contract hang off the abstract):
#
#   AbstractComposedDistribution{F, S} — combine named child distributions into
#     an event tree (the `child_*` node interface). Spans both variate forms:
#     the multivariate event-tree composers (`Sequential`, `Parallel`, `Choose`)
#     and the univariate marginal one_of family (`AbstractOneOf`: `Resolve`,
#     `Compete`).
#
#   AbstractModifiedDistribution{F, S} — wrap ONE inner base distribution and
#     modify it (the `free_leaf` / `rewrap_leaf` leaf interface). `Affine`,
#     `TimeChange`, `Modified`, `Transformed`, `Weighted`, `Shared` and the
#     censoring wrappers `IntervalCensored` / `PrimaryCensored`.
#
# Both are parametric on variate form `F` (`Univariate` / `Multivariate`) so one
# abstract spans the univariate and multivariate members while preserving
# `Distribution{F, S}` — and so the `UnivariateDistribution{S}` alias for the
# univariate members, leaving existing dispatch unchanged. The intermediate
# `AbstractMultiChild` groups the two positional multi-child composers
# (`Sequential`, `Parallel`) the tree walkers dispatch over together.

"""
    AbstractComposedDistribution{F<:VariateForm, S<:ValueSupport}

Supertype of the composer nodes that combine named child distributions into an
event tree: the multivariate `Sequential` / `Parallel` / `Choose` and the
univariate one_of family (`AbstractOneOf`: `Resolve` / `Compete`). Parametric on
variate form so the one supertype spans both.

Required methods a concrete subtype implements (the node interface):

- `child_nleaves(c)`, `child_logpdf(c, x, offset, n)`,
  `child_rand!(out, offset, rng, c)` — walk the flat event vector;
- `component_names(c)` — the child names;
- `params(c)` and `params_table(c)`;
- `event_names(c)` (flat) and `event_tree(c)` (nested);
- `Base.show(io, c)` (and the `MIME"text/plain"` form).

Verify a subtype with
`CensoredDistributions.TestUtils.test_composed_interface`.
"""
abstract type AbstractComposedDistribution{F <: VariateForm,
    S <: ValueSupport} <: Distribution{F, S} end

"""
    AbstractMultiChild{S<:ValueSupport}

Supertype of the positional multi-child composers `Sequential` and `Parallel`
(subtype of `AbstractComposedDistribution{Multivariate, S}`). These two store
`.components` / `.names` and are walked positionally by the tree machinery, so
they share dispatch wherever a `Union{Sequential, Parallel}` is spelt today.
`Choose` (disjoint alternatives) is a sibling, not a multi-child node.
"""
abstract type AbstractMultiChild{S <: ValueSupport} <:
              AbstractComposedDistribution{Multivariate, S} end

"""
    AbstractModifiedDistribution{F<:VariateForm, S<:ValueSupport}

Supertype of the single-base modifier leaves that wrap one inner distribution
and modify it: `Affine`, `TimeChange`, `Modified`, `Transformed`, `Weighted`,
`Shared` and the censoring wrappers `IntervalCensored` / `PrimaryCensored`.
Parametric on variate form for symmetry with the composed family.

Required methods a concrete subtype implements (the leaf interface):

- an inner base reachable as `.dist` (the default `show` accessor; override
  `CensoredDistributions._modified_inner` if stored elsewhere);
- `free_leaf(d)` and `rewrap_leaf(d, inner)`, with
  `rewrap_leaf(d, free_leaf(d))` reconstructing an equivalent node;
- the univariate interface (`pdf` / `logpdf` / `cdf` / `quantile` / `minimum` /
  `maximum` / `insupport` / `params`), forwarded or specialised;
- optionally `Base.show`; the default below prints `Name(inner)`.

Verify a subtype with
`CensoredDistributions.TestUtils.test_modified_interface`.
"""
abstract type AbstractModifiedDistribution{F <: VariateForm,
    S <: ValueSupport} <: Distribution{F, S} end

# The inner base a modifier wraps, used by the default `show`. Every current
# subtype stores it as `.dist`; a subtype that does not overrides this accessor.
_modified_inner(d::AbstractModifiedDistribution) = d.dist

# Default one-line show for a modifier leaf: `Name(inner)`. Concrete subtypes
# with their own `show` (`Modified`, `IntervalCensored`, `PrimaryCensored`,
# `Shared`) override it; the transform leaves (`Affine`, `TimeChange`,
# `Transformed`, `Weighted`) use this and so no longer fall back to the bare
# Distributions default. More specific than the concrete subtypes' own `show`
# and than `Distributions`' `show(::IO, ::Distribution)`, so no ambiguity.
function Base.show(io::IO, d::AbstractModifiedDistribution)
    print(io, nameof(typeof(d)), "(", _modified_inner(d), ")")
end
