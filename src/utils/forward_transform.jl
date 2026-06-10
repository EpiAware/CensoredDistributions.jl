# ============================================================================
# Forward-transform leaves: a generic transform with thin / cumulative
# ============================================================================
#
# A forward transform applies a DETERMINISTIC operation to the count series that
# `convolve_distributions(stack, series)` produces. It is the forward dual of
# `Weighted` (which touches `logpdf` only): a forward transform is TRANSPARENT to
# `logpdf` and to introspection, and materialises only when a stack is convolved
# through a timeseries.
#
# One node type, `Transformed`, carries an OP. `thin` and `cumulative` are
# specialised constructors over it with TYPED ops (so the factor stays
# introspectable and validated); the generic `transform(d, f)` accepts any
# callable `series -> series` as an escape hatch.
#
# The boundary (per the in/out test for generic node ops): a forward transform
# is a DETERMINISTIC function of the series. A scale that is itself a sampled
# latent process stays user/model-side.

# --- the ops --------------------------------------------------------------

# Multiply the series by a fixed factor (thinning / rescaling of expected
# counts). `cumulative` accumulates the series. Typed so the factor is readable.
struct ThinOp{T <: Real}
    factor::T
end
struct CumulativeOp end

# Apply one op to a series. A bare callable is its own op (the escape hatch).
_apply_op(op::ThinOp, series) = op.factor .* series
_apply_op(::CumulativeOp, series) = cumsum(series)
_apply_op(op, series) = op(series)

# The op's display parameters (only a numeric factor surfaces; others are pure).
_op_params(op::ThinOp) = (op.factor,)
_op_params(op) = ()

# --- the generic node -----------------------------------------------------

@doc "

A delay carrying a forward-transform op, applied by
[`convolve_distributions`](@ref) to the branch's output count series.
Transparent to `logpdf` (an individual delay use ignores the op) and to
introspection (`free_leaf` peels to the inner delay). Construct with the generic
[`transform`](@ref) or the specialised [`thin`](@ref) / [`cumulative`](@ref).

# See also
- [`transform`](@ref), [`thin`](@ref), [`cumulative`](@ref): constructors
"
struct Transformed{D <: UnivariateDistribution, Op} <:
       UnivariateDistribution{ValueSupport}
    "The inner delay distribution."
    dist::D
    "The forward op applied to the convolved series."
    op::Op
end

@doc "

Apply a forward-transform op to a delay's convolved count series.

`transform(d, op)` carries `op` (a [`thin`](@ref)/[`cumulative`](@ref) op or any
callable `series -> series`) that [`convolve_distributions`](@ref) applies to the
branch's output series. Transparent to `logpdf`. Prefer [`thin`](@ref) /
[`cumulative`](@ref) for the common cases; use `transform` for an arbitrary
deterministic series map.

# Arguments
- `d`: the inner delay distribution.
- `op`: a forward op or a callable `series -> series`.

# Examples
```@example
using CensoredDistributions, Distributions

d = transform(Gamma(2.0, 1.0), s -> 0.5 .* s)
logpdf(d, 2.0) == logpdf(Gamma(2.0, 1.0), 2.0)
```

# See also
- [`thin`](@ref), [`cumulative`](@ref): specialised forward transforms
"
transform(d::UnivariateDistribution, op) = Transformed(d, op)

@doc "

Thin a delay's forward count by a probability `p`.

`thin(d, p)` is [`transform`](@ref) with a fixed factor `p ∈ [0, 1]` that
[`convolve_distributions`](@ref) multiplies into the branch's count series (e.g.
ascertainment of cases, the infection fatality ratio for deaths). Transparent to
`logpdf`. `thin(d, nothing)` returns `d` unchanged.

# Arguments
- `d`: the inner delay distribution.
- `p`: the thinning probability in ``[0, 1]``.

# Examples
```@example
using CensoredDistributions, Distributions

d = thin(LogNormal(1.5, 0.5), 0.3)
logpdf(d, 2.0) == logpdf(LogNormal(1.5, 0.5), 2.0)
```

# See also
- [`transform`](@ref): the generic forward transform
- [`cumulative`](@ref): cumulative-sum a branch's series
"
function thin(dist::UnivariateDistribution, p::Real)
    (zero(p) <= p <= one(p)) ||
        throw(ArgumentError("thin probability must be in [0, 1]"))
    return Transformed(dist, ThinOp(p))
end

thin(dist::UnivariateDistribution, ::Nothing) = dist

@doc "

Accumulate a delay's forward count series.

`cumulative(d)` is [`transform`](@ref) with a running-sum op that
[`convolve_distributions`](@ref) applies to the branch's count series, giving
cumulative counts (cumulative incidence, cumulative deaths). Transparent to
`logpdf`.

# Arguments
- `d`: the inner delay distribution.

# Examples
```@example
using CensoredDistributions, Distributions

d = cumulative(Gamma(2.0, 1.0))
logpdf(d, 2.0) == logpdf(Gamma(2.0, 1.0), 2.0)
```

# See also
- [`transform`](@ref): the generic forward transform
- [`thin`](@ref): a fixed forward factor
"
cumulative(dist::UnivariateDistribution) = Transformed(dist, CumulativeOp())

# --- transparency: delegate every distribution method to the inner delay -----

# The inner delay a forward transform wraps.
get_dist(d::Transformed) = d.dist

for f in (:minimum, :maximum, :mean, :var, :std, :mode, :median,
    :skewness, :kurtosis)
    @eval Distributions.$f(d::Transformed) = Distributions.$f(d.dist)
end

for f in (:pdf, :logpdf, :cdf, :logcdf, :ccdf, :logccdf, :quantile)
    @eval Distributions.$f(d::Transformed, x::Real) = Distributions.$f(d.dist, x)
end

Distributions.insupport(d::Transformed, x::Real) = insupport(d.dist, x)

Base.rand(rng::AbstractRNG, d::Transformed) = rand(rng, d.dist)

Distributions.params(d::Transformed) = (params(d.dist)..., _op_params(d.op)...)

# --- introspection transparency (free_leaf / rewrap_leaf) --------------------

# Peel to the inner free delay so introspection and the tree scorer see the
# delay, not the forward op. `rewrap_leaf` rebuilds the same op around a new
# inner delay.
free_leaf(d::Transformed) = free_leaf(d.dist)
rewrap_leaf(d::Transformed, inner) = Transformed(rewrap_leaf(d.dist, inner), d.op)

# --- the forward op the convolve layer applies -------------------------------

# Peel forward-transform wrappers to the underlying (possibly censored) delay,
# returning `(delay, ops)` where `ops` is the ordered tuple of ops to apply to
# the branch's convolved series. A non-transform leaf has no ops.
_peel_forward(d) = (d, ())
function _peel_forward(d::Transformed)
    inner, ops = _peel_forward(d.dist)
    return inner, (ops..., d.op)
end

# Apply a tuple of forward ops to a series, in order.
function _apply_forward_ops(series, ops)
    foldl((s, op) -> _apply_op(op, s), ops; init = series)
end
