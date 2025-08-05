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

# Mean calculation
function Distributions.mean(d::ExponentiallyTilted)
    if _is_r_small(d.r)
        # Uniform case
        return (d.min + d.max) / 2
    else
        # For exponentially tilted distribution:
        # E[X] = min + (1/r) - (max-min)*exp(r*(max-min)) / (exp(r*(max-min))-1)
        r_range = d.r * (d.max - d.min)
        if abs(r_range) < 1e-6
            # Use approximation for small r_range to avoid numerical issues
            return (d.min + d.max) / 2
        else
            exp_r_range = exp(r_range)
            return d.min + (d.max - d.min) * (exp_r_range / (exp_r_range - 1) - 1 / r_range)
        end
    end
end

# Variance calculation
function Distributions.var(d::ExponentiallyTilted)
    if _is_r_small(d.r)
        # Uniform case: var = (b-a)²/12
        return (d.max - d.min)^2 / 12
    else
        # For exponentially tilted distribution, we need E[X²] - (E[X])²
        r_range = d.r * (d.max - d.min)
        if abs(r_range) < 1e-6
            # Use uniform approximation for small r_range
            return (d.max - d.min)^2 / 12
        else
            # For distribution f(x) = r*exp(r*(x-min))/(exp(r*(max-min))-1) on [min, max],
            # we compute the second moment analytically:
            # E[X²] = ∫_{min}^{max} x² * r*exp(r*(x-min))/(exp(r*(max-min))-1) dx

            exp_r_range = exp(r_range)
            range = d.max - d.min

            # Using integration by parts twice for the second moment:
            # After changing variables to y = x - min (so x = y + min, dx = dy)
            # E[X²] = ∫_0^range (y + min)² * r*exp(ry)/(exp(r*range)-1) dy
            #       = ∫_0^range (y² + 2*min*y + min²) * r*exp(ry)/(exp(r*range)-1) dy
            #       = min² + 2*min*E[Y] + E[Y²]
            # where Y has PDF r*exp(ry)/(exp(r*range)-1) on [0, range]

            # For Y on [0, L] with PDF r*exp(ry)/(exp(rL)-1):
            # E[Y] = (L*exp(rL) - (exp(rL)-1)/r) / (exp(rL)-1) = L - (exp(rL)-1)/(r*(exp(rL)-1)) = L - 1/(r*(exp(rL)-1)/(exp(rL)-1)) = (L*r*exp(rL) - L*r + 1)/(r*(exp(rL)-1))
            # Simplifying: E[Y] = L*r*exp(rL)/(r*(exp(rL)-1)) - L*r/(r*(exp(rL)-1)) + 1/(r*(exp(rL)-1))
            #                    = L*exp(rL)/(exp(rL)-1) - L/(exp(rL)-1) + 1/(r*(exp(rL)-1))
            #                    = L*(exp(rL)-1)/(exp(rL)-1) + 1/(r*(exp(rL)-1))
            #                    = L + 1/(r*(exp(rL)-1))
            # Wait, this doesn't look right. Let me recalculate...

            # For exponential distribution with parameter r on [0, L], truncated:
            # E[Y] = ∫_0^L y * r*exp(ry)/(exp(rL)-1) dy
            # Using integration by parts: ∫ y*exp(ry) dy = y*exp(ry)/r - exp(ry)/r²
            # E[Y] = [y*exp(ry)/r - exp(ry)/r²]_0^L * r/(exp(rL)-1)
            #      = [L*exp(rL)/r - exp(rL)/r² + 1/r²] * r/(exp(rL)-1)
            #      = [L*exp(rL) - exp(rL)/r + 1/r] / (exp(rL)-1)
            #      = L*exp(rL)/(exp(rL)-1) - exp(rL)/(r*(exp(rL)-1)) + 1/(r*(exp(rL)-1))
            #      = L*exp(rL)/(exp(rL)-1) - (exp(rL)-1)/(r*(exp(rL)-1))
            #      = L*exp(rL)/(exp(rL)-1) - 1/r

            mean_y = range * exp_r_range / (exp_r_range - 1) - 1 / d.r

            # E[Y²] = ∫_0^L y² * r*exp(ry)/(exp(rL)-1) dy
            # Using integration by parts twice: ∫ y²*exp(ry) dy = y²*exp(ry)/r - 2y*exp(ry)/r² + 2*exp(ry)/r³
            # E[Y²] = [y²*exp(ry)/r - 2y*exp(ry)/r² + 2*exp(ry)/r³]_0^L * r/(exp(rL)-1)
            #       = [L²*exp(rL)/r - 2L*exp(rL)/r² + 2*exp(rL)/r³ - 2/r³] * r/(exp(rL)-1)
            #       = [L²*exp(rL) - 2L*exp(rL)/r + 2*exp(rL)/r² - 2/r²] / (exp(rL)-1)

            second_moment_y = (range^2 * exp_r_range - 2 * range * exp_r_range / d.r +
                               2 * exp_r_range / d.r^2 - 2 / d.r^2) / (exp_r_range - 1)

            # Now E[X²] = min² + 2*min*E[Y] + E[Y²]
            second_moment = d.min^2 + 2 * d.min * mean_y + second_moment_y

            # Compute variance = E[X²] - (E[X])²
            mean_val = mean(d)
            return second_moment - mean_val^2
        end
    end
end

# Standard deviation calculation
function Distributions.std(d::ExponentiallyTilted)
    return sqrt(var(d))
end

# Median calculation
function Distributions.median(d::ExponentiallyTilted)
    return quantile(d, 0.5)
end
