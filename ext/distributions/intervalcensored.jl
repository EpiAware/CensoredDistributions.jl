"""
MLE fitting for IntervalCensored distributions
"""

using Bijectors

# Internal specialised constructor functions using multiple dispatch

# Single interval - creates one distribution
_dist_constructor(
    ::Type{IntervalCensored{D, T}}, 
    params::AbstractVector{<:Real}, 
    interval::Real
) where {D <: ContinuousUnivariateDistribution, T <: Real} = begin
    underlying_dist = D(params...)
    return interval_censored(underlying_dist, interval)
end

# Vector of intervals - creates product distribution
_dist_constructor(
    ::Type{IntervalCensored{D, T}}, 
    params::AbstractVector{<:Real}, 
    intervals::AbstractVector{<:Real}
) where {D <: ContinuousUnivariateDistribution, T <: Real} = begin
    underlying_dist = D(params...)
    individual_dists = [interval_censored(underlying_dist, int) for int in intervals]
    return product_distribution(individual_dists)
end

# Single boundary vector - creates one distribution
_dist_constructor(
    ::Type{IntervalCensored{D, T}}, 
    params::AbstractVector{<:Real}, 
    boundaries::AbstractVector{<:Real}
) where {D <: ContinuousUnivariateDistribution, T <: AbstractVector} = begin
    underlying_dist = D(params...)
    return interval_censored(underlying_dist, boundaries)
end

# Vector of boundary vectors - creates product distribution
_dist_constructor(
    ::Type{IntervalCensored{D, T}}, 
    params::AbstractVector{<:Real}, 
    boundaries::AbstractVector{<:AbstractVector{<:Real}}
) where {D <: ContinuousUnivariateDistribution, T <: AbstractVector} = begin
    underlying_dist = D(params...)
    individual_dists = [interval_censored(underlying_dist, bound) for bound in boundaries]
    return product_distribution(individual_dists)
end


"""
    Distributions.fit_mle(dist::IntervalCensored,
                         data::AbstractVector{<:Real};
                         intervals=nothing, init_params=nothing, 
                         weights=nothing, optimizer=OptimizationOptimJL.BFGS())

Fit an interval-censored distribution to data using maximum likelihood estimation.
Uses the distribution type from `dist` for dispatch, eliminating the need for `dist_type`.

# Arguments
- `dist`: An `IntervalCensored` distribution instance (used for type dispatch)
- `data`: Vector of observed interval-censored values (interval left boundaries)
- `intervals`: Interval specification - can be:
  - `nothing`: uses `dist.boundaries` (default)
  - `Real`: interval width for regular intervals
  - `AbstractVector{<:Real}`: custom interval boundaries
  - `AbstractVector{<:AbstractVector{<:Real}}`: heterogeneous intervals per observation
- `init_params`: Initial parameters for underlying distribution (optional)
- `weights`: Optional weights for observations
- `optimizer`: SciML optimizer (default: OptimizationOptimJL.BFGS())

# Returns
A fitted `IntervalCensored` distribution with estimated parameters.

# Examples
```julia
# Fit Normal distribution with regular intervals
template_dist = interval_censored(Normal(0, 1), 1.0)
fitted_dist = fit_mle(template_dist, data)

# Fit with custom intervals
fitted_dist = fit_mle(template_dist, data; intervals=2.0)

# Fit with vector of intervals for heterogeneous data
fitted_dist = fit_mle(template_dist, data; intervals=[1.0, 2.0, 1.5, 1.0])
```
"""
function Distributions.fit_mle(
        dist::IntervalCensored{D, T},
        data::AbstractVector{<:Real};
        intervals::Union{Nothing, Real, AbstractVector{<:Real}, AbstractVector{<:AbstractVector{<:Real}}} = nothing,
        init_params::Union{Nothing, AbstractVector{<:Real}} = nothing,
        weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
        optimizer = OptimizationOptimJL.BFGS()
) where {D <: ContinuousUnivariateDistribution, T}
    # Input validation
    _validate_data(data)
    _validate_weights(weights, data)

    # Determine interval structure
    interval_spec = intervals === nothing ? dist.boundaries : intervals
    
    # Initialize parameters if not provided - get defaults from input distribution
    if init_params === nothing
        init_params = collect(params(dist.dist))  # Get params from the input distribution
    end

    # Get bijector for the distribution type using DistributionsAD if available
    bijector = _get_bijector(D, init_params)

    # Create distribution constructor using dispatch - handles product distribution internally
    dist_constructor = params -> _dist_constructor(typeof(dist), params, interval_spec)

    # Optimize using the generic function
    fitted_params = _optimize_censored_distribution(
        data, init_params, dist_constructor, bijector, weights, optimizer
    )

    # Create fitted distribution
    return _dist_constructor(typeof(dist), fitted_params, interval_spec)
end


