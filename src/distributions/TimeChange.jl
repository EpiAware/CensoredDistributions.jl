@doc """

The distribution of a sojourn delay observed in calendar time under a monotone
operational-time warp `Λ`, the continuous generalisation of [`affine`](@ref).

An inner delay `X` is drawn on an operational-time clock that runs at a
calendar-time-varying intensity `λ(t)`. The observed calendar delay `Y`
satisfies `Λ(Y) = X`, where `Λ(t) = ∫₀ᵗ λ(s) ds` is the accumulated
operational time. With `Λ` strictly increasing and differentiable the change
of variables is exact:

```math
F_Y(y) = F_X(Λ(y)), \\qquad
f_Y(y) = f_X(Λ(y))\\, λ(y), \\qquad
λ(y) = Λ'(y).
```

This first form uses a log-linear intensity `λ(t) = scale · exp(rate · t)`,
covering a monotone trend or the rising/falling limb of a seasonal forcing, with
warp `Λ(t) = scale · (exp(rate · t) − 1) / rate`. At `rate = 0` the warp is
linear, `Λ(t) = scale · t`, and the node reduces to the affine rescale
`affine(X; scale = 1/scale)` (a pure clock-speed change, no shift). `scale > 0`
keeps the intensity positive and `Λ` strictly increasing for any real `rate`.

Because it is a `UnivariateDistribution`, a `TimeChange` nests as a leaf in
[`compose`](@ref), [`choose`](@ref) and [`Resolve`](@ref) automatically.

# See also
- [`timechange`](@ref): constructor function
- [`affine`](@ref): the linear special case
"""
struct TimeChange{D <: UnivariateDistribution, T <: Real} <:
       AbstractModifiedDistribution{Univariate, Continuous}
    "The inner sojourn delay drawn on the operational-time clock."
    dist::D
    "The positive base clock speed (the intensity at calendar time zero)."
    scale::T
    "The log-linear intensity growth rate; zero gives a constant clock."
    rate::T

    function TimeChange{D, T}(dist::D, scale::T, rate::T) where {
            D <: UnivariateDistribution, T <: Real}
        scale > zero(scale) ||
            throw(ArgumentError("scale must be positive"))
        new{D, T}(dist, scale, rate)
    end
end

@doc "

Create a time-changed delay observed in calendar time under a log-linear
intensity `λ(t) = scale * exp(rate * t)`.

# Arguments
- `dist`: the inner sojourn delay `X` drawn on the operational-time clock.

# Keyword Arguments
- `scale`: positive base clock speed, the intensity at `t = 0` (default `1`).
- `rate`: log-linear growth rate of the intensity (default `0`, a constant
  clock that recovers the affine rescale `affine(X; scale = 1/scale)`).

# Examples
```@example
using CensoredDistributions, Distributions

d = timechange(LogNormal(1.5, 0.5); scale = 1.2, rate = 0.1)
logpdf(d, 5.0)
```

# See also
- [`TimeChange`](@ref): the wrapped type
- [`affine`](@ref): the linear special case
"
function timechange(
        dist::UnivariateDistribution; scale::Real = 1, rate::Real = 0)
    s, r = promote(float(scale), float(rate))
    return TimeChange{typeof(dist), typeof(s)}(dist, s, r)
end

# Accumulated operational time `Λ(y)`. The `rate → 0` limit is linear; the
# expm1/divide form stays finite and AD-stable across that limit.
function _timechange_warp(d::TimeChange, y::Real)
    r = d.rate
    return iszero(r) ? d.scale * y : d.scale * expm1(r * y) / r
end

# The calendar intensity `λ(y) = Λ'(y) = scale * exp(rate * y)`.
_timechange_rate(d::TimeChange, y::Real) = d.scale * exp(d.rate * y)

# Parameter extraction: inner params followed by the warp pair.
params(d::TimeChange) = (params(d.dist)..., d.scale, d.rate)

function Base.eltype(::Type{<:TimeChange{D, T}}) where {D, T}
    return promote_type(eltype(D), T)
end

# The warp fixes `Λ(0) = 0`, so the calendar support starts where the inner
# operational-time support does (zero for a non-negative delay).
function minimum(d::TimeChange)
    m = minimum(d.dist)
    return isinf(m) ? oftype(d.scale * m, m) : zero(d.scale * m)
end

# `Λ` maps `[0, ∞)` onto `[0, ∞)`, so an unbounded inner delay stays
# unbounded in calendar time.
function maximum(d::TimeChange)
    m = maximum(d.dist)
    return oftype(d.scale * m, m)
end

function insupport(d::TimeChange, y::Real)
    y >= zero(y) &&
        insupport(d.dist, _timechange_warp(d, y))
end

@doc "

Compute the probability density function.

See also: [`logpdf`](@ref)
"
pdf(d::TimeChange, y::Real) = exp(logpdf(d, y))

@doc "

Compute the log probability density function via change of variables.

See also: [`pdf`](@ref), [`cdf`](@ref)
"
function logpdf(d::TimeChange, y::Real)
    return logpdf(d.dist, _timechange_warp(d, y)) +
           log(_timechange_rate(d, y))
end

@doc "

Compute the cumulative distribution function.

See also: [`logcdf`](@ref), [`quantile`](@ref)
"
cdf(d::TimeChange, y::Real) = cdf(d.dist, _timechange_warp(d, y))

@doc "

Compute the log cumulative distribution function.

See also: [`cdf`](@ref)
"
logcdf(d::TimeChange, y::Real) = logcdf(d.dist, _timechange_warp(d, y))

@doc "

Compute the quantile function (inverse CDF) by inverting the warp.

See also: [`cdf`](@ref)
"
function quantile(d::TimeChange, p::Real)
    x = quantile(d.dist, p)
    return _timechange_invwarp(d, x)
end

# Invert the warp, mapping an operational time `x` back to calendar time:
# `y = log1p(rate * x / scale) / rate`, with the linear `rate → 0` limit.
function _timechange_invwarp(d::TimeChange, x::Real)
    r = d.rate
    return iszero(r) ? x / d.scale : log1p(r * x / d.scale) / r
end

@doc "

Generate a random sample by inverting the warp on an inner operational draw.

See also: [`quantile`](@ref)
"
function Base.rand(rng::AbstractRNG, d::TimeChange)
    return _timechange_invwarp(d, rand(rng, d.dist))
end

# Composer introspection: a TimeChange is transparent to its inner free delay,
# so `params_table`/`update` see the inner distribution's parameters and the
# fixed scale/rate structure round-trips. Mirrors the affine wrapper.
free_leaf(d::TimeChange) = free_leaf(d.dist)
function rewrap_leaf(d::TimeChange, inner)
    rebuilt = rewrap_leaf(d.dist, inner)
    return TimeChange{typeof(rebuilt), typeof(d.scale)}(
        rebuilt, d.scale, d.rate)
end
