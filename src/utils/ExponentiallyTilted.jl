"""
    ExponentiallyTilted{T<:Real} <: ContinuousUnivariateDistribution

Exponentially tilted distribution on interval [min, max] with growth rate r.

The exponentially tilted distribution provides a principled approach to model primary event
timing when events occur at exponentially changing rates. This is particularly useful in
epidemiological contexts where infection timing within censoring windows is biased by
epidemic growth or decay patterns.

# Mathematical Background

The probability density function is:
```math
f_P(t) = \\frac{r \\cdot \\exp(r \\cdot (t - \\text{min}))}{\\exp(r \\cdot \\text{max}) - \\exp(r \\cdot \\text{min})}
```

The cumulative distribution function is:
```math
F_P(x) = \\frac{\\exp(r \\cdot (x - \\text{min})) - \\exp(r \\cdot \\text{min})}{\\exp(r \\cdot \\text{max}) - \\exp(r \\cdot \\text{min})}
```

# Parameters
- `min::T`: Lower bound of support (default: 0.0)
- `max::T`: Upper bound of support (default: 1.0)
- `r::T`: Growth rate parameter (default: 0.0)
  - `r > 0`: exponential growth (events more likely near end of window)
  - `r < 0`: exponential decay (events more likely near start of window)
  - `r → 0`: approaches uniform distribution

# Examples
```julia
using CensoredDistributions, Distributions

# Uniform-like (no growth)
d1 = ExponentiallyTilted(0.0, 1.0, 0.0)

# Exponential growth (r > 0)
d2 = ExponentiallyTilted(0.0, 1.0, 2.0)

# Exponential decay (r < 0)
d3 = ExponentiallyTilted(0.0, 1.0, -1.5)

# Integration with PrimaryCensored
delay_dist = LogNormal(1.5, 0.75)
growth_prior = ExponentiallyTilted(0.0, 1.0, 0.8)  # Growing epidemic
censored_dist = primary_censored(delay_dist, growth_prior)
```

# References
- Park, S.W., Akhmetzhanov, A.R., Charniga, K. et al. (2024). "Estimating epidemiological delay distributions for infectious diseases." medRxiv preprint.
- Charniga, K., Park, S.W., Akhmetzhanov, A.R. et al. (2024). "Best practices for estimating and reporting epidemiological delay distributions of infectious diseases." PLoS Computational Biology 20(10): e1012520.
"""
struct ExponentiallyTilted{T<:Real} <: ContinuousUnivariateDistribution
    min::T
    max::T
    r::T

    function ExponentiallyTilted{T}(min::T, max::T, r::T) where {T<:Real}
        min < max || throw(ArgumentError("min must be less than max"))
        isfinite(min) && isfinite(max) || throw(ArgumentError("min and max must be finite"))
        isfinite(r) || throw(ArgumentError("r must be finite"))
        new{T}(min, max, r)
    end
end

# Constructor with type promotion
ExponentiallyTilted(min::Real=0.0, max::Real=1.0, r::Real=0.0) =
    ExponentiallyTilted{promote_type(typeof(min), typeof(max), typeof(r))}(min, max, r)

# Parameters interface
Distributions.params(d::ExponentiallyTilted) = (d.min, d.max, d.r)

# Support definition
Distributions.@distr_support ExponentiallyTilted d.min d.max

# Numerical stability threshold
const EXPONENTIALLY_TILTED_EPSILON = 1e-10

"""
    pdf(d::ExponentiallyTilted, x::Real)

Probability density function of the exponentially tilted distribution.

For numerical stability, when |r| < ε (ε ≈ 10⁻¹⁰), the distribution reduces to uniform.
"""
function Distributions.pdf(d::ExponentiallyTilted, x::Real)
    x_min, x_max, r = d.min, d.max, d.r

    # Check support
    if x < x_min || x > x_max
        return zero(typeof(x))
    end

    # Uniform case for numerical stability
    if abs(r) < EXPONENTIALLY_TILTED_EPSILON
        return one(typeof(x)) / (x_max - x_min)
    end

    # Exponentially tilted case
    exp_min = exp(r * x_min)
    exp_max = exp(r * x_max)
    exp_x = exp(r * x)

    return r * exp_x / (exp_max - exp_min)
end

"""
    logpdf(d::ExponentiallyTilted, x::Real)

Log probability density function of the exponentially tilted distribution.
"""
function Distributions.logpdf(d::ExponentiallyTilted, x::Real)
    x_min, x_max, r = d.min, d.max, d.r

    # Check support
    if x < x_min || x > x_max
        return -Inf
    end

    # Uniform case for numerical stability
    if abs(r) < EXPONENTIALLY_TILTED_EPSILON
        return -log(x_max - x_min)
    end

    # Exponentially tilted case
    return log(abs(r)) + r * x - LogExpFunctions.logsumexp([r * x_min, r * x_max])
end

"""
    cdf(d::ExponentiallyTilted, x::Real)

Cumulative distribution function of the exponentially tilted distribution.
"""
function Distributions.cdf(d::ExponentiallyTilted, x::Real)
    x_min, x_max, r = d.min, d.max, d.r

    # Handle bounds
    if x <= x_min
        return zero(typeof(x))
    elseif x >= x_max
        return one(typeof(x))
    end

    # Uniform case for numerical stability
    if abs(r) < EXPONENTIALLY_TILTED_EPSILON
        return (x - x_min) / (x_max - x_min)
    end

    # Exponentially tilted case
    exp_min = exp(r * x_min)
    exp_max = exp(r * x_max)
    exp_x = exp(r * x)

    return (exp_x - exp_min) / (exp_max - exp_min)
end

"""
    logcdf(d::ExponentiallyTilted, x::Real)

Log cumulative distribution function of the exponentially tilted distribution.
"""
function Distributions.logcdf(d::ExponentiallyTilted, x::Real)
    x_min, x_max, r = d.min, d.max, d.r

    # Handle bounds
    if x <= x_min
        return -Inf
    elseif x >= x_max
        return zero(typeof(x))
    end

    # Uniform case for numerical stability
    if abs(r) < EXPONENTIALLY_TILTED_EPSILON
        return log(x - x_min) - log(x_max - x_min)
    end

    # Exponentially tilted case using LogExpFunctions for stability
    log_numerator = LogExpFunctions.logsubexp(r * x, r * x_min)
    log_denominator = LogExpFunctions.logsubexp(r * x_max, r * x_min)

    return log_numerator - log_denominator
end

"""
    quantile(d::ExponentiallyTilted, p::Real)

Quantile function (inverse CDF) of the exponentially tilted distribution.
"""
function Distributions.quantile(d::ExponentiallyTilted, p::Real)
    x_min, x_max, r = d.min, d.max, d.r

    # Validate probability
    0 <= p <= 1 || throw(DomainError(p, "p must be in [0, 1]"))

    # Handle boundary cases
    if p == 0
        return x_min
    elseif p == 1
        return x_max
    end

    # Uniform case for numerical stability
    if abs(r) < EXPONENTIALLY_TILTED_EPSILON
        return x_min + p * (x_max - x_min)
    end

    # Exponentially tilted case
    exp_min = exp(r * x_min)
    exp_max = exp(r * x_max)
    exp_term = p * (exp_max - exp_min) + exp_min

    return x_min + log(exp_term) / r
end

"""
    rand(rng::AbstractRNG, d::ExponentiallyTilted)

Generate a random sample from the exponentially tilted distribution using inverse transform sampling.
"""
function Base.rand(rng::AbstractRNG, d::ExponentiallyTilted)
    return quantile(d, rand(rng))
end

"""
    mean(d::ExponentiallyTilted)

Mean of the exponentially tilted distribution.
"""
function Distributions.mean(d::ExponentiallyTilted)
    x_min, x_max, r = d.min, d.max, d.r

    # Uniform case for numerical stability
    if abs(r) < EXPONENTIALLY_TILTED_EPSILON
        return (x_min + x_max) / 2
    end

    # Exponentially tilted case
    exp_min = exp(r * x_min)
    exp_max = exp(r * x_max)

    numerator = r * x_max * exp_max - r * x_min * exp_min - exp_max + exp_min
    denominator = r * (exp_max - exp_min)

    return numerator / denominator
end

"""
    var(d::ExponentiallyTilted)

Variance of the exponentially tilted distribution.
"""
function Distributions.var(d::ExponentiallyTilted)
    x_min, x_max, r = d.min, d.max, d.r

    # Uniform case for numerical stability
    if abs(r) < EXPONENTIALLY_TILTED_EPSILON
        return (x_max - x_min)^2 / 12
    end

    # Exponentially tilted case - compute second moment then variance
    exp_min = exp(r * x_min)
    exp_max = exp(r * x_max)

    # Second moment: E[X²]
    r2 = r^2
    second_moment_num = r2 * x_max^2 * exp_max - r2 * x_min^2 * exp_min -
                       2 * r * x_max * exp_max + 2 * r * x_min * exp_min +
                       2 * exp_max - 2 * exp_min
    second_moment = second_moment_num / (r2 * (exp_max - exp_min))

    # Variance = E[X²] - (E[X])²
    μ = mean(d)
    return second_moment - μ^2
end

"""
    std(d::ExponentiallyTilted)

Standard deviation of the exponentially tilted distribution.
"""
Distributions.std(d::ExponentiallyTilted) = sqrt(var(d))

"""
    mode(d::ExponentiallyTilted)

Mode of the exponentially tilted distribution.
"""
function Distributions.mode(d::ExponentiallyTilted)
    x_min, x_max, r = d.min, d.max, d.r

    # For uniform case or r ≈ 0, any point in the interval is a mode
    if abs(r) < EXPONENTIALLY_TILTED_EPSILON
        return (x_min + x_max) / 2  # Return midpoint as convention
    end

    # For r > 0, mode is at x_max; for r < 0, mode is at x_min
    return r > 0 ? x_max : x_min
end

"""
    entropy(d::ExponentiallyTilted)

Differential entropy of the exponentially tilted distribution.
"""
function Distributions.entropy(d::ExponentiallyTilted)
    x_min, x_max, r = d.min, d.max, d.r

    # Uniform case for numerical stability
    if abs(r) < EXPONENTIALLY_TILTED_EPSILON
        return log(x_max - x_min)
    end

    # Exponentially tilted case
    exp_min = exp(r * x_min)
    exp_max = exp(r * x_max)

    # H = log(Z) - r * E[X] where Z is the normalization constant
    log_z = log(exp_max - exp_min) - log(abs(r))
    expected_x = mean(d)

    return log_z - r * expected_x
end