@doc raw"
Exponentially tilted distribution.

A continuous distribution on interval [min, max] with exponential tilting
controlled by parameter r. This distribution generalises the uniform
distribution by allowing exponential weighting of values within the interval.

# Fields
- `min`: Lower bound of the support
- `max`: Upper bound of the support
- `r`: Growth rate parameter (tilting parameter)
"
struct ExponentiallyTilted{T <: Real} <:
       Distributions.UnivariateDistribution{Distributions.Continuous}
    "Lower bound of the distribution support."
    min::T
    "Upper bound of the distribution support."
    max::T
    "Growth rate parameter (tilting parameter)."
    r::T

    function ExponentiallyTilted{T}(min::T, max::T, r::T) where {T <: Real}
        max > min || throw(ArgumentError("max must be greater than min"))
        isfinite(min) || throw(ArgumentError("min must be finite"))
        isfinite(max) || throw(ArgumentError("max must be finite"))
        isfinite(r) || throw(ArgumentError("r must be finite"))
        new{T}(min, max, r)
    end
end

@doc raw"
Create an exponentially tilted distribution.

An exponentially tilted distribution on interval [min, max] with growth rate r.
For r > 0, the distribution is tilted towards higher values; for r < 0, towards
lower values. When r ≈ 0, it approaches a uniform distribution.

The distribution has probability density function:
```math
f(x) = \frac{\exp(r(x - \text{min}))}{\int_{\text{min}}^{\text{max}}
\exp(r(t - \text{min})) dt}
```
for r ≠ 0, and f(x) = 1/(max - min) for r ≈ 0.

# Arguments
- `min::Real`: Lower bound of the distribution support
- `max::Real`: Upper bound of the distribution support
- `r::Real`: Growth rate parameter (tilting parameter)

# Examples
```@example
using CensoredDistributions, Distributions

# Exponentially increasing distribution
d1 = ExponentiallyTilted(0.0, 1.0, 2.0)
pdf(d1, 0.5)

# Exponentially decreasing distribution
d2 = ExponentiallyTilted(0.0, 1.0, -1.5)
cdf(d2, 0.3)

# Nearly uniform distribution (small r)
d3 = ExponentiallyTilted(0.0, 1.0, 1e-10)
rand(d3, 5)
```
"
function ExponentiallyTilted(min::Real, max::Real, r::Real)
    promoted_values = promote(min, max, r)
    return ExponentiallyTilted{typeof(promoted_values[1])}(promoted_values...)
end

# Parameter extraction
function Distributions.params(d::ExponentiallyTilted)
    return (d.min, d.max, d.r)
end

# Support and type methods
Base.eltype(::Type{<:ExponentiallyTilted{T}}) where {T} = T
Distributions.minimum(d::ExponentiallyTilted) = d.min
Distributions.maximum(d::ExponentiallyTilted) = d.max
Distributions.insupport(d::ExponentiallyTilted, x::Real) = d.min ≤ x ≤ d.max

# Helper function to check if r is effectively zero for numerical stability
function _is_r_small(r::Real, tolerance::Real = 1e-10)
    return abs(r) < tolerance
end

# Helper function to compute normalisation constant
function _normalisation_constant(min::Real, max::Real, r::Real)
    if _is_r_small(r)
        return max - min
    else
        # Integral of exp(r*(x-min)) from min to max
        # With substitution u = x - min: integral becomes exp(r*u)
        # from 0 to (max-min)
        # = (exp(r*(max-min)) - 1)/r
        r_range = r * (max - min)
        if abs(r_range) < 1e-8
            # Use Taylor expansion: (exp(r*(max-min)) - 1) ≈ r*(max-min)
            return max - min
        else
            return (exp(r_range) - 1.0) / r
        end
    end
end

# Probability density function
function Distributions.pdf(d::ExponentiallyTilted, x::Real)
    return exp(logpdf(d, x))
end

function Distributions.logpdf(d::ExponentiallyTilted, x::Real)
    if !insupport(d, x)
        return -Inf
    end

    if _is_r_small(d.r)
        # Uniform distribution case
        return -log(d.max - d.min)
    else
        # Exponentially tilted case
        # PDF: exp(r*(x-min)) / normalisation_constant
        log_numerator = d.r * (x - d.min)
        log_denominator = log(_normalisation_constant(d.min, d.max, d.r))
        return log_numerator - log_denominator
    end
end

# Cumulative distribution function
function Distributions.cdf(d::ExponentiallyTilted, x::Real)
    if x ≤ d.min
        return 0.0
    elseif x ≥ d.max
        return 1.0
    end

    if _is_r_small(d.r)
        # Uniform distribution case
        return (x - d.min) / (d.max - d.min)
    else
        # Exponentially tilted case
        numerator = exp(d.r * x) - exp(d.r * d.min)
        denominator = exp(d.r * d.max) - exp(d.r * d.min)
        return numerator / denominator
    end
end

function Distributions.logcdf(d::ExponentiallyTilted, x::Real)
    if x ≤ d.min
        return -Inf
    elseif x ≥ d.max
        return 0.0
    end

    cdf_val = cdf(d, x)
    if cdf_val ≤ 0.0
        return -Inf
    else
        return log(cdf_val)
    end
end

# Quantile function (inverse CDF)
function Distributions.quantile(d::ExponentiallyTilted, p::Real)
    if p < 0.0 || p > 1.0
        throw(ArgumentError("p must be in [0, 1]"))
    end

    if p == 0.0
        return d.min
    elseif p == 1.0
        return d.max
    end

    if _is_r_small(d.r)
        # Uniform distribution case
        return d.min + p * (d.max - d.min)
    else
        # Exponentially tilted case - solve p = F(x) for x
        # p = (exp(r*x) - exp(r*min)) / (exp(r*max) - exp(r*min))
        # Rearranging: exp(r*x) = exp(r*min) + p*(exp(r*max) - exp(r*min))
        exp_rx = exp(d.r * d.min) + p * (exp(d.r * d.max) - exp(d.r * d.min))
        return log(exp_rx) / d.r
    end
end

# Random number generation
function Base.rand(rng::AbstractRNG, d::ExponentiallyTilted)
    # Use inverse transform sampling
    u = rand(rng)
    return quantile(d, u)
end
