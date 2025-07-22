@doc raw"""
    Weighted{D <: UnivariateDistribution, T <: Real} <: UnivariateDistribution{ValueSupport}

A distribution wrapper that applies a weight to the log-probability of an underlying distribution.
This is primarily used where observations have associated counts or weights.

Only the `logpdf` method is affected by the weight - all other methods (pdf, cdf, sampling, etc.)
delegate directly to the underlying distribution.

# Fields
- `dist::D`: The underlying distribution
- `weight::T`: The weight factor (must be non-negative)

# Examples
```julia
using CensoredDistributions, Distributions, Turing

# Single weighted observation
d = LogNormal(1.5, 0.5)
wd = weight(d, 10.0)  # Observation with weight/count of 10

# In a Turing model
@model function example_model(y, n)
    μ ~ Normal(0, 1)
    σ ~ truncated(Normal(0.5, 0.5); lower = 0)
    d = LogNormal(μ, σ)
    
    # Instead of: Turing.@addlogprob! n * logpdf(d, y)
    # You can use:
    y ~ weight(d, n)
end
```
"""
struct Weighted{D <: UnivariateDistribution, T <: Real} <:
       Distributions.UnivariateDistribution{Distributions.ValueSupport}
    dist::D
    weight::T
    
    function Weighted(dist::D, weight::T) where {D <: UnivariateDistribution, T <: Real}
        weight >= 0 || throw(ArgumentError("Weight must be non-negative"))
        new{D, T}(dist, weight)
    end
end

@doc raw"""
    weight(dist::UnivariateDistribution, w::Real)

Create a weighted distribution where the log-probability is scaled by `w`.

# Arguments
- `dist`: The underlying distribution
- `w`: The weight factor (must be non-negative)

# Returns
A `Weight` distribution that when used in Turing.jl will contribute `w * logpdf(dist, x)` to the 
log-probability.

# Examples
```julia
# For aggregated count data
y_obs = 3.5  # Observed value
n_count = 25  # Number of times this value was observed

@model function count_model(y_obs, n_count)
    μ ~ Normal(2, 1)
    σ ~ Exponential(1)
    d = Normal(μ, σ)
    
    y_obs ~ weight(d, n_count)
end
```
"""
function weight(dist::UnivariateDistribution, w::Real)
    return Weighted(dist, w)
end

# For creating an array of weighted distributions with different weights
@doc raw"""
    weight(dist::UnivariateDistribution, weights::AbstractVector{<:Real})

Create a product distribution of weighted distributions, each with a different weight.

# Arguments
- `dist`: The underlying distribution (same for all)
- `weights`: Vector of weights for each observation

# Returns
A `Product` distribution of `Weighted` distributions suitable for vectorized observations.

# Examples
```julia
y_obs = [3.5, 4.2, 3.8]  # Observed values
n_counts = [25, 10, 15]  # Counts for each observation

@model function vectorized_model(y_obs, n_counts)
    μ ~ Normal(2, 1)
    σ ~ Exponential(1)
    d = Normal(μ, σ)
    
    y_obs ~ weight(d, n_counts)
end
```
"""
function weight(dist::UnivariateDistribution, weights::AbstractVector{<:Real})
    return product_distribution([Weighted(dist, w) for w in weights])
end

# Distributions.jl interface implementation

# Basic properties
Base.eltype(::Type{<:Weighted{D, T}}) where {D, T} = promote_type(eltype(D), T)
Distributions.minimum(d::Weighted) = minimum(d.dist)
Distributions.maximum(d::Weighted) = maximum(d.dist)
Distributions.insupport(d::Weighted, x::Real) = insupport(d.dist, x)
Distributions.params(d::Weighted) = (params(d.dist)..., d.weight)

# Probability functions
@doc raw"""
    pdf(d::Weight, x::Real)

Returns the probability density/mass at `x` from the underlying distribution (unweighted).

The PDF is not affected by weights as weights only apply to log-likelihood contributions.
"""
function Distributions.pdf(d::Weighted, x::Real)
    return pdf(d.dist, x)
end

@doc raw"""
    logpdf(d::Weight, x::Real)

Returns the weighted log-probability at `x`. This is the key method used by Turing.jl.

Returns `weight * logpdf(dist, x)`.
"""
function Distributions.logpdf(d::Weighted, x::Real)
    # If weight is zero, return -Inf to avoid 0 * -Inf = NaN
    d.weight == 0 && return -Inf
    return d.weight * logpdf(d.dist, x)
end

# CDF-based methods - delegate to underlying distribution
function Distributions.cdf(d::Weighted, x::Real)
    return cdf(d.dist, x)
end

function Distributions.logcdf(d::Weighted, x::Real)
    return logcdf(d.dist, x)
end

function Distributions.ccdf(d::Weighted, x::Real)
    return ccdf(d.dist, x)
end

function Distributions.logccdf(d::Weighted, x::Real)
    return logccdf(d.dist, x)
end

# Quantile function
function Distributions.quantile(d::Weighted, p::Real)
    return quantile(d.dist, p)
end

# Sampling - delegates to underlying distribution
Base.rand(rng::AbstractRNG, d::Weighted) = rand(rng, d.dist)