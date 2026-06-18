@doc """

The distribution of a deterministic affine transform `Y = scale * X + shift`
of an inner delay `X`, with `scale > 0` and any real `shift`.

Computed by the change-of-variables for a strictly increasing affine map: with
`x = (y - shift) / scale`,

```math
f_Y(y) = f_X\\!\\left(\\frac{y - \\text{shift}}{\\text{scale}}\\right)
         \\frac{1}{\\text{scale}}, \\qquad
F_Y(y) = F_X\\!\\left(\\frac{y - \\text{shift}}{\\text{scale}}\\right).
```

Because it is a `UnivariateDistribution`, an `Affine` nests as a leaf in
[`compose`](@ref), [`choose`](@ref), and [`Resolve`](@ref)
automatically.

# See also
- [`affine`](@ref): constructor function
"""
struct Affine{D <: UnivariateDistribution, T <: Real} <:
       UnivariateDistribution{Continuous}
    "The inner delay distribution being transformed."
    dist::D
    "The positive multiplicative scale."
    scale::T
    "The additive shift."
    shift::T

    function Affine{D, T}(dist::D, scale::T, shift::T) where {
            D <: UnivariateDistribution, T <: Real}
        scale > zero(scale) ||
            throw(ArgumentError("scale must be positive"))
        new{D, T}(dist, scale, shift)
    end
end

@doc "

Create an affine-transformed distribution `Y = scale * X + shift`.

# Arguments
- `dist`: the inner delay distribution `X`.

# Keyword Arguments
- `scale`: positive multiplicative factor (default `1`).
- `shift`: additive offset (default `0`).

# Examples
```@example
using CensoredDistributions, Distributions

d = affine(LogNormal(1.5, 0.5); scale = 2.0, shift = 1.0)
logpdf(d, 5.0)
```

# See also
- [`Affine`](@ref): the wrapped type
"
function affine(dist::UnivariateDistribution; scale::Real = 1, shift::Real = 0)
    s, h = promote(float(scale), float(shift))
    return Affine{typeof(dist), typeof(s)}(dist, s, h)
end

# Map an observed `y` back to the inner variate `x = (y - shift) / scale`.
_affine_inv(d::Affine, y::Real) = (y - d.shift) / d.scale

# Parameter extraction: inner params followed by the affine pair.
params(d::Affine) = (params(d.dist)..., d.scale, d.shift)

Base.eltype(::Type{<:Affine{D, T}}) where {D, T} = promote_type(eltype(D), T)

function minimum(d::Affine)
    m = minimum(d.dist)
    return isinf(m) ? oftype(d.scale * m, m) : d.scale * m + d.shift
end

function maximum(d::Affine)
    m = maximum(d.dist)
    return isinf(m) ? oftype(d.scale * m, m) : d.scale * m + d.shift
end

insupport(d::Affine, y::Real) = insupport(d.dist, _affine_inv(d, y))

@doc "

Compute the probability density function.

See also: [`logpdf`](@ref)
"
pdf(d::Affine, y::Real) = exp(logpdf(d, y))

@doc "

Compute the log probability density function via change of variables.

See also: [`pdf`](@ref), [`cdf`](@ref)
"
logpdf(d::Affine, y::Real) = logpdf(d.dist, _affine_inv(d, y)) - log(d.scale)

@doc "

Compute the cumulative distribution function.

See also: [`logcdf`](@ref), [`quantile`](@ref)
"
cdf(d::Affine, y::Real) = cdf(d.dist, _affine_inv(d, y))

@doc "

Compute the log cumulative distribution function.

See also: [`cdf`](@ref)
"
logcdf(d::Affine, y::Real) = logcdf(d.dist, _affine_inv(d, y))

@doc "

Compute the quantile function (inverse CDF).

See also: [`cdf`](@ref)
"
quantile(d::Affine, p::Real) = d.scale * quantile(d.dist, p) + d.shift

@doc "

Generate a random sample by transforming an inner draw.

See also: [`quantile`](@ref)
"
function Base.rand(rng::AbstractRNG, d::Affine)
    return d.scale * rand(rng, d.dist) + d.shift
end

@doc "

Compute the mean via the affine transform of the inner mean.

See also: [`var`](@ref)
"
mean(d::Affine) = d.scale * mean(d.dist) + d.shift

@doc "

Compute the variance via the affine transform of the inner variance.

See also: [`mean`](@ref)
"
var(d::Affine) = d.scale^2 * var(d.dist)

# Composer introspection: an Affine is transparent to its inner free delay,
# so `params_table`/`update` see the inner distribution's parameters and the
# fixed scale/shift structure round-trips. Mirrors the censoring wrappers.
free_leaf(d::Affine) = free_leaf(d.dist)
function rewrap_leaf(d::Affine, inner)
    rebuilt = rewrap_leaf(d.dist, inner)
    return Affine{typeof(rebuilt), typeof(d.scale)}(rebuilt, d.scale, d.shift)
end
