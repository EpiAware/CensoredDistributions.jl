@doc "

A distribution wrapper that applies a weight to the log-probability of an
underlying distribution.
This is primarily used where observations have associated counts or weights.

Only the `logpdf` method is affected by the weight - all other methods (pdf,
cdf, sampling, etc.) delegate directly to the underlying distribution.

# Weight Types Supported

The `Weighted` struct supports three different weight scenarios:

1. **Real weights**: Constructor weight is a specific value (e.g., `2.5`)
2. **Missing weights**: Constructor weight is `missing`, allowing weights to be
   provided at observation time via joint observations `(value = x, weight = w)`
3. **Zero weights**: Handled specially to return `-Inf` and avoid NaN from
   `0 * -Inf`

# Examples
```@example
using CensoredDistributions, Distributions

# Single weighted observation
d = LogNormal(1.5, 0.5)
wd = weight(d, 10.0)  # Observation with weight/count of 10

# Weighted log-probability calculation
observed_value = 2.0
weighted_logpdf = logpdf(wd, observed_value)

# Compare with manual calculation
manual_logpdf = 10.0 * logpdf(d, observed_value)
# weighted_logpdf ≈ manual_logpdf
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

# ============================================================================
# Constructor Functions
# ============================================================================

@doc "

Create a weighted distribution where the log-probability is scaled by `w`.

A `Weighted` distribution will contribute `w * logpdf(dist, x)` to the
log-probability when evaluating `logpdf(weighted_dist, x)`.

# Examples
```@example
# For aggregated count data
y_obs = 3.5  # Observed value
n_count = 25  # Number of times this value was observed

d = Normal(2.0, 1.0)
weighted_d = weight(d, n_count)

# Weighted log-probability calculation
weighted_logpdf = logpdf(weighted_d, y_obs)
# equivalent to: n_count * logpdf(d, y_obs)
```
"
function weight(dist::UnivariateDistribution, w::Real)
    return Weighted(dist, w)
end

@doc "

Create a product distribution of weighted distributions, each with a different
weight.

A `Product` distribution of `Weighted` distributions suitable for vectorized
observations.

# Arguments
- `dist`: The univariate distribution to be replicated and weighted for each observation
- `weights`: Vector of weights to apply to each copy of the distribution

# Examples
```@example
y_obs = [3.5, 4.2, 3.8]  # Observed values
n_counts = [25, 10, 15]  # Counts for each observation

d = Normal(2.0, 1.0)
weighted_dists = weight(d, n_counts)

# Weighted log-probability calculation
weighted_logpdf = logpdf(weighted_dists, y_obs)
# equivalent to: sum(n_counts .* logpdf.(d, y_obs))
```

# See also
- [`Weighted`](@ref): The underlying weighted distribution type
"
function weight(dist::UnivariateDistribution, weights::AbstractVector{<:Real})
    return product_distribution([Weighted(dist, w) for w in weights])
end

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
            ArgumentError(
            "Number of distributions must equal number of weights"
        )
        )
    return product_distribution(
        [Weighted(d, w) for (d, w) in zip(dists, weights)]
    )
end

@doc "

Create a weighted distribution with missing constructor weight.

Useful for creating distributions where weights will be provided at observation
time. Uses `missing` as constructor weight, enabling observation weight to be
used directly.

# Examples
```@example
using CensoredDistributions, Distributions

d = Normal(2.0, 0.5)

# Create weighted distribution with missing constructor weight
weighted_dist = weight(d)

# Weight provided at observation time via joint observations
logpdf(weighted_dist, (value = 3.5, weight = 25))
```
"
function weight(dist::UnivariateDistribution)
    return Weighted(dist, missing)
end

@doc "

Create a product distribution of weighted distributions with missing constructor
weights.

Useful for creating distributions where weights will be provided at observation
time. Each distribution uses `missing` as constructor weight, enabling
observation weight to be used directly.

# Examples
```@example
using CensoredDistributions, Distributions

y_obs = [3.5, 4.2, 3.8]  # Observed values
dists = [Normal(2.0, 0.5), Normal(2.5, 0.8), Normal(1.8, 0.6)]

# Create weighted distributions with missing constructor weights
weighted_dists = weight(dists)

# Weights provided at observation time via joint observations
logpdf(weighted_dists, (values = y_obs, weights = [25, 10, 15]))
```
"
function weight(dists::AbstractVector{<:UnivariateDistribution})
    return product_distribution(
        [Weighted(d, missing) for d in dists]
    )
end

# ============================================================================
# Distributions.jl Interface - Basic Properties
# ============================================================================

# Weight doesn't affect element type
Base.eltype(::Type{<:Weighted{D, T}}) where {D, T} = eltype(D)
minimum(d::Weighted) = minimum(get_dist(d))
maximum(d::Weighted) = maximum(get_dist(d))
insupport(d::Weighted, x::Real) = insupport(get_dist(d), x)
params(d::Weighted) = (params(get_dist(d))..., d.weight)

# ============================================================================
# Probability Density Functions
# ============================================================================

@doc "

Return the probability density from the underlying distribution (unweighted).

See also: [`logpdf`](@ref)
"
function pdf(d::Weighted, x::Real)
    return pdf(get_dist(d), x)
end

# Helper function for weighted logpdf computation with proper validation
function _logpdf(dist, value, weight)
    # Handle missing or zero weights to avoid NaN from 0 * -Inf operations
    if ismissing(weight) || weight == 0
        return -Inf
    end
    return weight * logpdf(dist, value)
end

@doc "

Return the weighted log-probability for scalar observations.

See also: [`pdf`](@ref)
"
function logpdf(d::Weighted, x::Real)
    return _logpdf(get_dist(d), x, d.weight)
end

@doc "

Return the weighted log-probability for joint observations as NamedTuple.

Combines constructor weight with observation weight via multiplication.
Expected format: `(value = x, weight = w)`.

See also: [`pdf`](@ref)
"
function logpdf(d::Weighted, obs::NamedTuple{(:value, :weight)})
    final_weight = combine_weights(d.weight, obs.weight)
    return _logpdf(get_dist(d), obs.value, final_weight)
end

# ============================================================================
# Product Distribution logpdf Methods
# ============================================================================

# Helper function for Product logpdf computation
function _logpdf_product(
        d::Product{<:ValueSupport, <:Weighted, <:AbstractVector{<:Weighted}},
        values, obs_weights)
    # Compute base logpdfs and extract constructor weights
    logpdfs = [logpdf(wd.dist, v) for (wd, v) in zip(d.v, values)]
    constructor_weights = [wd.weight for wd in d.v]

    # Combine weights and compute final result
    final_weights = combine_weights(constructor_weights, obs_weights)
    # Handle missing or zero weights - return -Inf if any found
    if any(ismissing, final_weights) || any(w -> w == 0, final_weights)
        return -Inf
    end
    return sum(final_weights .* logpdfs)
end

@doc "

Efficient vectorised log-probability computation for
Product{<:ValueSupport, <:Weighted} with joint observations.

Handles joint observations and weight stacking.
Expected format: `(values = [...], weights = [...])`.

See also: [`logpdf`](@ref)
"
function logpdf(
        d::Product{<:ValueSupport, <:Weighted, <:AbstractVector{<:Weighted}},
        obs::NamedTuple{(:values, :weights)})
    return _logpdf_product(d, obs.values, obs.weights)
end

@doc "

Efficient vectorised log-probability computation for
Product{<:ValueSupport, <:Weighted} with vector observations.

See also: [`logpdf`](@ref)
"
function logpdf(
        d::Product{<:ValueSupport, <:Weighted, <:AbstractVector{<:Weighted}},
        x::AbstractVector{<:Real})
    return _logpdf_product(d, x, missing)  # No observation weights
end

# ============================================================================
# Log-likelihood Methods for Weighted Distributions
# ============================================================================

@doc "

Compute log-likelihood for single Weighted distribution with joint observations.

Handles joint observations as NamedTuple: `(value = x, weight = w)`.

See also: [`logpdf`](@ref)
"
function loglikelihood(d::Weighted, obs::NamedTuple{(:value, :weight)})
    return logpdf(d, obs)
end

@doc "

Compute log-likelihood for single Weighted distribution with vectorized joint
observations.

Handles joint observations as NamedTuple: `(values = [...], weights = [...])`.
This is useful when a single weighted distribution is used with multiple
observations.

See also: [`logpdf`](@ref)
"
function loglikelihood(d::Weighted, obs::NamedTuple{(:values, :weights)})
    # For single distribution with multiple observations, sum logpdf results
    return sum(logpdf(d, (value = v, weight = w))
    for (v, w) in zip(obs.values, obs.weights))
end

@doc "

Compute log-likelihood for Product{<:ValueSupport, <:Weighted} with joint
observations.

Handles joint observations as NamedTuple: `(values = [...], weights = [...])`.

See also: [`logpdf`](@ref)
"
function loglikelihood(
        d::Product{<:ValueSupport, <:Weighted, <:AbstractVector{<:Weighted}},
        obs::NamedTuple{(:values, :weights)})
    return logpdf(d, obs)
end

# ============================================================================
# Other Distribution Interface Methods (delegate to underlying distribution)
# ============================================================================

@doc "

Compute the cumulative distribution function (delegates to underlying
distribution).

See also: [`logcdf`](@ref)
"
function cdf(d::Weighted, x::Real)
    return cdf(get_dist(d), x)
end

@doc "

Compute the log cumulative distribution function (delegates to underlying
distribution).

See also: [`cdf`](@ref)
"
function logcdf(d::Weighted, x::Real)
    return logcdf(get_dist(d), x)
end

@doc "

Compute the complementary cumulative distribution function (delegates to
underlying distribution).

See also: [`cdf`](@ref)
"
function ccdf(d::Weighted, x::Real)
    return ccdf(get_dist(d), x)
end

@doc "

Compute the log complementary cumulative distribution function (delegates to
underlying distribution).

See also: [`logcdf`](@ref)
"
function logccdf(d::Weighted, x::Real)
    return logccdf(get_dist(d), x)
end

@doc "

Compute the quantile function (delegates to underlying distribution).

See also: [`cdf`](@ref)
"
function quantile(d::Weighted, p::Real)
    return quantile(get_dist(d), p)
end

@doc "

Generate a random sample (delegates to underlying distribution).

See also: [`quantile`](@ref)
"
Base.rand(rng::AbstractRNG, d::Weighted) = rand(rng, get_dist(d))

@doc "

Create a sampler for efficient sampling (delegates to underlying distribution).

See also: [`rand`](@ref)
"
sampler(d::Weighted) = Weighted(sampler(get_dist(d)), d.weight)

# ============================================================================
# Helper Functions for Observation and Weight Processing
# ============================================================================

@doc "

Combine constructor weight with observation weight using dispatch-based rules.

Weight combination rules:
- `missing, missing → missing` (both missing means no weight)
- `w1, missing → w1` (use constructor weight)
- `missing, w2 → w2` (use observation weight)
- `w1, w2 → w1 * w2` (multiply weights)

# Vector Extensions

For Product distributions, additional methods handle vectorised weight
combinations:
- `Vector, Vector → combine_weights.(vector1, vector2)` (element-wise
  combination)
- `Vector, missing → Vector` (keep constructor weights)
- `Vector, scalar → [combine_weights(w, scalar) for w in Vector]` (broadcast
  scalar)

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

# Vector extensions for Product distributions
function combine_weights(
        constructor_weights::AbstractVector, obs_weights::AbstractVector)
    return combine_weights.(constructor_weights, obs_weights)
end

function combine_weights(constructor_weights::AbstractVector, ::Missing)
    return constructor_weights
end

function combine_weights(constructor_weights::AbstractVector, obs_weight::Real)
    return [combine_weights(w, obs_weight) for w in constructor_weights]
end
