@doc """

A continuous distribution on interval [min, max] with exponential tilting
controlled by parameter r. This distribution generalises the uniform
distribution by allowing exponential weighting of values within the interval.

# Mathematical Definition

The probability density function is:
```math
f(x) = \\frac{r \\exp(r(x - \\text{min}))}{(\\exp(r(\\text{max} - \\text{min})) - 1)}
```
for x ∈ [min, max] and r ≠ 0.

The cumulative distribution function is:
```math
F(x) = \\frac{\\exp(r(x - \\text{min})) - 1}{\\exp(r(\\text{max} - \\text{min})) - 1}
```
for x ∈ [min, max].

The quantile function (inverse CDF) is:
```math
F^{-1}(p) = \\text{min} +
\\frac{1}{r} \\log\\left(1 + p(\\exp(r(\\text{max} - \\text{min})) - 1)\\right)
```
for p ∈ [0, 1].

When r → 0, all functions reduce to the uniform distribution on [min, max].
- For r > 0: distribution is tilted towards higher values (increasing density)
- For r < 0: distribution is tilted towards lower values (decreasing density)

"""
struct ExponentiallyTilted{T <: Real} <:
       UnivariateDistribution{Continuous}
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

@doc "
$(TYPEDSIGNATURES)

Create an exponentially tilted distribution.

An exponentially tilted distribution on interval [min, max] with growth rate r.
For r > 0, the distribution is tilted towards higher values; for r < 0, towards
lower values. When r ≈ 0, it approaches a uniform distribution.

# Examples
```@example
using CensoredDistributions, Distributions

# Exponentially increasing distribution
d1 = ExponentiallyTilted(0.0, 1.0, 2.0)
pdf(d1, 0.5)

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
function params(d::ExponentiallyTilted)
    return (d.min, d.max, d.r)
end

# Support and type methods
Base.eltype(::Type{<:ExponentiallyTilted{T}}) where {T} = T
minimum(d::ExponentiallyTilted) = d.min
maximum(d::ExponentiallyTilted) = d.max
insupport(d::ExponentiallyTilted, x::Real) = d.min ≤ x ≤ d.max

# Helper function to compute log normalisation constant
function _log_normalisation_constant(min::Real, max::Real, r::Real)
    if abs(r) < 1e-10
        # For r ≈ 0, reduces to uniform: log(max - min)
        return log(max - min)
    else
        # For r ≠ 0: log((exp(r*(max-min)) - 1)/r)
        r_range = r * (max - min)
        normalisation_constant = expm1(r_range) / r
        return log(normalisation_constant)
    end
end

# Probability density function
@doc "
$(TYPEDSIGNATURES)

Compute the probability density function.

See also: [`logpdf`](@ref)
"
function pdf(d::ExponentiallyTilted, x::Real)
    return exp(logpdf(d, x))
end

@doc "
$(TYPEDSIGNATURES)

Compute the log probability density function.

See also: [`pdf`](@ref), [`cdf`](@ref)
"
function logpdf(d::ExponentiallyTilted, x::Real)
    if !insupport(d, x)
        return -Inf
    end
    if abs(d.r) < 1e-10
        # For r ≈ 0, uniform distribution: log(1/(max-min))
        return -log(d.max - d.min)
    else
        log_numerator = d.r * (x - d.min)
        log_denominator = _log_normalisation_constant(d.min, d.max, d.r)
        return log_numerator - log_denominator
    end
end

# Cumulative distribution function
@doc "
$(TYPEDSIGNATURES)

Compute the cumulative distribution function.

See also: [`logcdf`](@ref), [`quantile`](@ref)
"
function cdf(d::ExponentiallyTilted, x::Real)
    if x ≤ d.min
        return 0.0
    elseif x ≥ d.max
        return 1.0
    end

    if abs(d.r) < 1e-10
        # For r ≈ 0, uniform distribution: (x - min) / (max - min)
        return (x - d.min) / (d.max - d.min)
    else
        # CDF(x) = (exp(r*(x-min)) - 1) / (exp(r*(max-min)) - 1)
        r_x_rel = d.r * (x - d.min)
        r_range = d.r * (d.max - d.min)

        numerator = expm1(r_x_rel)
        denominator = expm1(r_range)

        return numerator / denominator
    end
end

# Log cumulative distribution function
@doc "
$(TYPEDSIGNATURES)

Compute the log cumulative distribution function.

See also: [`cdf`](@ref)
"
function logcdf(d::ExponentiallyTilted, x::Real)
    if x ≤ d.min
        return -Inf
    elseif x ≥ d.max
        return 0.0
    end
    if abs(d.r) < 1e-10
        # For r ≈ 0, uniform distribution: log((x - min) / (max - min))
        return log((x - d.min) / (d.max - d.min))
    else
        return log(cdf(d, x))
    end
end

# Quantile function (inverse CDF)
@doc "
$(TYPEDSIGNATURES)

Compute the quantile function (inverse CDF) using analytical formula.

See also: [`cdf`](@ref)
"
function quantile(d::ExponentiallyTilted, p::Real)
    if p < 0.0 || p > 1.0
        throw(ArgumentError("p must be in [0, 1]"))
    end

    if p == 0.0
        return d.min
    elseif p == 1.0
        return d.max
    end

    if abs(d.r) < 1e-10
        # For r ≈ 0, uniform distribution: min + p * (max - min)
        return d.min + p * (d.max - d.min)
    else
        # Quantile formulation for r ≠ 0
        # p = (exp(r*(x-min)) - 1) / (exp(r*(max-min)) - 1)
        # Solve for x: exp(r*(x-min)) = 1 + p * (exp(r*(max-min)) - 1)

        r_range = d.r * (d.max - d.min)
        exp_r_range_minus_1 = expm1(r_range)
        inner_term = 1.0 + p * exp_r_range_minus_1

        return d.min + log(inner_term) / d.r
    end
end

# Random number generation
@doc "
$(TYPEDSIGNATURES)

Generate a random sample using inverse transform sampling.

See also: [`quantile`](@ref)
"
function Base.rand(rng::AbstractRNG, d::ExponentiallyTilted)
    # Use inverse transform sampling
    u = rand(rng)
    return quantile(d, u)
end

# Mean calculation
@doc "
$(TYPEDSIGNATURES)

Compute the mean of the distribution.

See also: [`var`](@ref), [`std`](@ref)
"
function mean(d::ExponentiallyTilted)
    if abs(d.r) < 1e-10
        # For r ≈ 0, uniform distribution: (min + max) / 2
        return (d.min + d.max) / 2
    else
        r_range = d.r * (d.max - d.min)
        exp_r_range = exp(r_range)
        return d.min + (d.max - d.min) *
                       (exp_r_range / (exp_r_range - 1) - 1 / r_range)
    end
end

# Variance calculation
@doc "
$(TYPEDSIGNATURES)

Compute the variance of the distribution.

See also: [`mean`](@ref), [`std`](@ref)
"
function var(d::ExponentiallyTilted)
    if abs(d.r) < 1e-10
        # For r ≈ 0, uniform distribution: (max - min)² / 12
        range = d.max - d.min
        return range^2 / 12
    else
        r_range = d.r * (d.max - d.min)
        exp_r_range = exp(r_range)
        range = d.max - d.min

        mean_y = range * exp_r_range / (exp_r_range - 1) - 1 / d.r

        second_moment_y = (range^2 * exp_r_range - 2 * range * exp_r_range / d.r +
                           2 * exp_r_range / d.r^2 - 2 / d.r^2) /
                          (exp_r_range - 1)

        second_moment = d.min^2 + 2 * d.min * mean_y + second_moment_y

        # Compute variance = E[X²] - (E[X])²
        mean_val = mean(d)
        return second_moment - mean_val^2
    end
end

# Standard deviation calculation
@doc "
$(TYPEDSIGNATURES)

Compute the standard deviation of the distribution.

See also: [`mean`](@ref), [`var`](@ref)
"
function std(d::ExponentiallyTilted)
    return sqrt(var(d))
end

# Median calculation
@doc "
$(TYPEDSIGNATURES)

Compute the median of the distribution.

See also: [`quantile`](@ref), [`mean`](@ref)
"
function median(d::ExponentiallyTilted)
    return quantile(d, 0.5)
end
