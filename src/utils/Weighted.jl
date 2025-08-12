@doc "

A distribution wrapper that applies a weight to the log-probability of an
underlying distribution.
This is primarily used where observations have associated counts or weights.

Only the `logpdf` method is affected by the weight - all other methods (pdf,
cdf, sampling, etc.) delegate directly to the underlying distribution.


# Examples
```@example
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
"
struct Weighted{D <: UnivariateDistribution, T <: Union{Real, Missing}} <:
       UnivariateDistribution{ValueSupport}
    "The underlying distribution being weighted."
    dist::D
    "The weight to apply to log-probabilities."
    weight::T

    function Weighted(
            dist::D, weight::T) where {
            D <: UnivariateDistribution, T <: Union{Real, Missing}}
        if !ismissing(weight) && weight < 0
            throw(ArgumentError("Weight must be non-negative"))
        end
        new{D, T}(dist, weight)
    end
end

@doc "

Create a weighted distribution where the log-probability is scaled by `w`.

A `Weighted` distribution that when used in Turing.jl will contribute
`w * logpdf(dist, x)` to the log-probability.

# Examples
```@example
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
"
function weight(dist::UnivariateDistribution, w::Real)
    return Weighted(dist, w)
end

# For creating an array of weighted distributions with different weights
@doc "

Create a product distribution of weighted distributions, each with a different
weight.

A `Product` distribution of `Weighted` distributions suitable for vectorized
observations.

# Examples
```@example
y_obs = [3.5, 4.2, 3.8]  # Observed values
n_counts = [25, 10, 15]  # Counts for each observation

@model function vectorized_model(y_obs, n_counts)
    μ ~ Normal(2, 1)
    σ ~ Exponential(1)
    d = Normal(μ, σ)

    y_obs ~ weight(d, n_counts)
end
```
"
function weight(dist::UnivariateDistribution, weights::AbstractVector{<:Real})
    return product_distribution([Weighted(dist, w) for w in weights])
end

# For creating weighted distributions from a vector of distributions and weights
@doc "

Create a product distribution of weighted distributions, where each
distribution has its own weight.

A `Product` distribution of `Weighted` distributions suitable for vectorized
observations with different distributions.

# Examples
```@example
using CensoredDistributions, Distributions

y_obs = [3.5, 4.2, 3.8]  # Observed values
dists = [Normal(2.0, 0.5), Normal(2.5, 0.8), Normal(1.8, 0.6)]
n_counts = [25, 10, 15]  # Counts for each observation
weighted_dists = weight(dists, n_counts)
```
"
function weight(
        dists::AbstractVector{<:UnivariateDistribution},
        weights::AbstractVector{<:Real})
    length(dists) == length(weights) ||
        throw(
            ArgumentError("Number of distributions must equal number of weights")
        )
    return product_distribution(
        [Weighted(d, w) for (d, w) in zip(dists, weights)]
    )
end

@doc "

Create a product distribution of weighted distributions with missing constructor weights.

Useful for creating distributions where weights will be provided at observation time.
Each distribution uses `missing` as constructor weight, enabling observation weight
to be used directly.

# Examples
```@example
using CensoredDistributions, Distributions

y_obs = [3.5, 4.2, 3.8]  # Observed values
dists = [Normal(2.0, 0.5), Normal(2.5, 0.8), Normal(1.8, 0.6)]

# Create weighted distributions with missing constructor weights
weighted_dists = weight(dists)

# Weights provided at observation time via joint observations
logpdf(weighted_dists, (y_obs, [25, 10, 15]))
```
"
function weight(dists::AbstractVector{<:UnivariateDistribution})
    return product_distribution(
        [Weighted(d, missing) for d in dists]
    )
end

# Distributions.jl interface implementation

# Basic properties
Base.eltype(::Type{<:Weighted{D, T}}) where {D, T} = eltype(D)  # Weight doesn't affect element type
minimum(d::Weighted) = minimum(get_dist(d))
maximum(d::Weighted) = maximum(get_dist(d))
insupport(d::Weighted, x::Real) = insupport(get_dist(d), x)
params(d::Weighted) = (params(get_dist(d))..., d.weight)

# Probability functions
@doc "

Return the probability density from the underlying distribution (unweighted).

See also: [`logpdf`](@ref)
"
function pdf(d::Weighted, x::Real)
    return pdf(get_dist(d), x)
end

# Helper functions to compute weighted logpdf with proper weight validation
@doc "

Compute weighted logpdf with proper weight validation and edge case handling.

Returns -Inf for missing or zero weights to avoid NaN from 0 * -Inf operations.
"
function _weighted_logpdf(weight, base_logpdf::Real)
    # Handle missing weight
    if ismissing(weight)
        return -Inf
    end

    # Handle zero weights to avoid 0 * -Inf = NaN
    if weight == 0
        return -Inf
    end

    return weight * base_logpdf
end

@doc "

Compute vectorised weighted logpdf with proper weight validation.

Returns -Inf if any weights are missing or zero to avoid NaN operations.
"
function _weighted_logpdf_vectorised(weights::AbstractVector, base_logpdfs::AbstractVector)
    # Handle missing weights - if any final weight is missing, return -Inf
    if any(ismissing, weights)
        return -Inf
    end

    # Handle zero weights - if any final weight is zero, return -Inf
    if any(w -> w == 0, weights)
        return -Inf
    end

    # Dot product for final result
    return sum(weights .* base_logpdfs)
end

@doc "

Combine constructor weights with observation weights for Product distributions.

Handles vectorised, scalar, and missing observation weights appropriately.
"
function _combine_product_weights(constructor_weights::AbstractVector, obs_weights)
    # Handle vectorised weight combination
    if obs_weights isa AbstractVector
        # Vectorised weight combination via broadcasting
        return combine_weights.(constructor_weights, obs_weights)
    elseif ismissing(obs_weights)
        # obs_weight is missing - just use constructor weights
        return constructor_weights
    else
        # Scalar obs_weight - broadcast to all components
        return [combine_weights(cw, obs_weights) for cw in constructor_weights]
    end
end

# More specific method signatures to avoid ambiguities with Distributions.jl

@doc "

Return the weighted log-probability for scalar observations.

See also: [`pdf`](@ref)
"
function logpdf(d::Weighted, x::Real)
    return _weighted_logpdf(d.weight, logpdf(get_dist(d), x))
end

@doc "

Return the weighted log-probability for joint observations `(value, weight)`.

Combines constructor weight with observation weight via multiplication.

See also: [`pdf`](@ref)
"
function logpdf(d::Weighted, obs::Tuple{T, S}) where {T, S}
    value, obs_weight = obs
    final_weight = combine_weights(d.weight, obs_weight)
    return _weighted_logpdf(final_weight, logpdf(get_dist(d), value))
end

@doc "

Return the weighted log-probability for vector observations.

See also: [`pdf`](@ref)
"
function logpdf(d::Weighted, x::AbstractVector{<:Real})
    return _weighted_logpdf(d.weight, logpdf(get_dist(d), x))
end

# CDF-based methods - delegate to underlying distribution
@doc "

Compute the cumulative distribution function (delegates to underlying
distribution).

See also: [`logcdf`](@ref)
"
function cdf(d::Weighted, x::Real)
    return cdf(get_dist(d), x)
end

function logcdf(d::Weighted, x::Real)
    return logcdf(get_dist(d), x)
end

function ccdf(d::Weighted, x::Real)
    return ccdf(get_dist(d), x)
end

function logccdf(d::Weighted, x::Real)
    return logccdf(get_dist(d), x)
end

# Quantile function
@doc "

Compute the quantile function (delegates to underlying distribution).

See also: [`cdf`](@ref)
"
function quantile(d::Weighted, p::Real)
    return quantile(get_dist(d), p)
end

# Sampling - delegates to underlying distribution
@doc "

Generate a random sample (delegates to underlying distribution).

See also: [`quantile`](@ref)
"
Base.rand(rng::AbstractRNG, d::Weighted) = rand(rng, get_dist(d))

# Sampler method for efficient sampling
sampler(d::Weighted) = Weighted(sampler(get_dist(d)), d.weight)

# Efficient vectorised logpdf for Product of Weighted distributions
@doc "

Efficient vectorised log-probability computation for Product{<:ValueSupport, <:Weighted}.

Handles joint observations and weight stacking without extraction loops.

See also: [`logpdf`](@ref)
"
function logpdf(d::Product{<:ValueSupport, <:Weighted, <:AbstractVector{<:Weighted}},
        obs::Tuple{T, S}) where {T, S}
    values, obs_weights = extract_obs(obs)

    # Compute base logpdfs and extract constructor weights
    logpdfs = [logpdf(wd.dist, v) for (wd, v) in zip(d.v, values)]
    constructor_weights = [wd.weight for wd in d.v]

    # Combine weights and compute final result
    final_weights = _combine_product_weights(constructor_weights, obs_weights)
    return _weighted_logpdf_vectorised(final_weights, logpdfs)
end

# More specific method for AbstractVector observations to avoid ambiguity
function logpdf(d::Product{<:ValueSupport, <:Weighted, <:AbstractVector{<:Weighted}},
        x::AbstractVector{<:Real})
    values, obs_weights = extract_obs(x)

    # Compute base logpdfs and extract constructor weights
    logpdfs = [logpdf(wd.dist, v) for (wd, v) in zip(d.v, values)]
    constructor_weights = [wd.weight for wd in d.v]

    # Combine weights and compute final result
    final_weights = _combine_product_weights(constructor_weights, obs_weights)
    return _weighted_logpdf_vectorised(final_weights, logpdfs)
end

# Specific method for AbstractArray{<:AbstractVector{<:Real}} to avoid ambiguity
function logpdf(
        d::Product{<:ValueSupport, <:Weighted, <:AbstractVector{<:Weighted}},
        x::AbstractArray{<:AbstractVector{<:Real}})
    values, obs_weights = extract_obs(x)

    # Compute base logpdfs and extract constructor weights
    logpdfs = [logpdf(wd.dist, v) for (wd, v) in zip(d.v, values)]
    constructor_weights = [wd.weight for wd in d.v]

    # Combine weights and compute final result
    final_weights = _combine_product_weights(constructor_weights, obs_weights)
    return _weighted_logpdf_vectorised(final_weights, logpdfs)
end

# Helper functions for observation handling and weight operations

@doc "

Extract observation values and weights from different input formats.

Supports:
- Scalar observations: returns `(obs, missing)`
- Vector observations: returns `(obs, missing)`
- Tuple observations `(value, weight)`: returns the tuple as-is
"
function extract_obs(obs::Real)
    return (obs, missing)
end

function extract_obs(obs::AbstractVector)
    return (obs, missing)
end

function extract_obs(obs::Tuple{<:Any, <:Any})
    return obs  # Joint observation (value, weight)
end

# Note: StatsBase.AbstractWeights support would be added here:
# function extract_obs(wv::AbstractWeights)
#     return (wv.values, wv.weights)
# end

@doc "

Combine constructor weight with observation weight using dispatch-based rules.

Weight combination rules:
- `missing, missing → missing` (both missing means no weight)
- `w1, missing → w1` (use constructor weight)
- `missing, w2 → w2` (use observation weight)
- `w1, w2 → w1 * w2` (multiply weights)

"
function combine_weights(::Missing, ::Missing)
    return missing
end

function combine_weights(w1, ::Missing)
    return w1
end

function combine_weights(::Missing, w2)
    return w2
end

function combine_weights(w1, w2)
    # Handle zero weights to avoid NaN from 0 * Inf
    w1 == 0 && return zero(typeof(w1))
    w2 == 0 && return zero(typeof(w2))
    return w1 * w2
end
