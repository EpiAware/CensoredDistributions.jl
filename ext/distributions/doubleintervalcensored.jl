"""
MLE fitting for double interval censored distributions (IntervalCensored{<:PrimaryCensored})
"""

using Bijectors

# Internal specialised constructor functions using multiple dispatch for double interval censored

# Generic fallback method for any PrimaryCensored type
function _dist_constructor(
        ::Type{T},
        delay_params::AbstractVector{<:Real},
        primary_params::AbstractVector{<:Real},
        interval::Real,
        force_numeric::Bool = true,
        lower = nothing,
        upper = nothing
) where {T <: IntervalCensored}
    # Extract the distribution types from the type parameter
    # Handle both concrete types and UnionAll types
    if isa(T, UnionAll)
        # This shouldn't happen in practice but handle it gracefully
        throw(ArgumentError("Cannot extract distribution types from UnionAll type $T"))
    end

    inner_type = T.parameters[1] # Could be PrimaryCensored or Truncated{PrimaryCensored}

    # Handle truncated case
    if inner_type <: Truncated
        pc_type = inner_type.parameters[1] # Extract PrimaryCensored from Truncated
    else
        pc_type = inner_type # Already PrimaryCensored
    end

    D = pc_type.parameters[1] # Delay distribution type
    P = pc_type.parameters[2] # Primary distribution type

    delay_dist = D(delay_params...)
    primary_dist = P(primary_params...)

    if lower === nothing && upper === nothing
        return double_interval_censored(
            delay_dist, primary_dist; interval = interval, force_numeric = force_numeric)
    else
        return double_interval_censored(delay_dist, primary_dist; interval = interval,
            lower = lower, upper = upper, force_numeric = force_numeric)
    end
end

# Single parameters - creates one distribution (kept for backwards compatibility)
function _dist_constructor(
        ::Type{IntervalCensored{<:PrimaryCensored{<:D, <:P, <:M}, <:Real}},
        delay_params::AbstractVector{<:Real},
        primary_params::AbstractVector{<:Real},
        interval::Real,
        force_numeric::Bool = true,
        lower = nothing,
        upper = nothing
) where {D <: ContinuousUnivariateDistribution, P <: ContinuousUnivariateDistribution,
        M <: CensoredDistributions.AbstractSolverMethod}
    delay_dist = D(delay_params...)
    primary_dist = P(primary_params...)

    if lower === nothing && upper === nothing
        return double_interval_censored(
            delay_dist, primary_dist; interval = interval, force_numeric = force_numeric)
    else
        return double_interval_censored(delay_dist, primary_dist; interval = interval,
            lower = lower, upper = upper, force_numeric = force_numeric)
    end
end

# Generic vector interval constructor
function _dist_constructor(
        ::Type{T},
        delay_params::AbstractVector{<:Real},
        primary_params::AbstractVector{<:Real},
        intervals::AbstractVector{<:Real},
        force_numeric::Bool = true,
        lowers = nothing,
        uppers = nothing
) where {T <: IntervalCensored}
    # Extract the distribution types from the type parameter
    # Handle both concrete types and UnionAll types
    if isa(T, UnionAll)
        # This shouldn't happen in practice but handle it gracefully
        throw(ArgumentError("Cannot extract distribution types from UnionAll type $T"))
    end

    inner_type = T.parameters[1] # Could be PrimaryCensored or Truncated{PrimaryCensored}

    # Handle truncated case
    if inner_type <: Truncated
        pc_type = inner_type.parameters[1] # Extract PrimaryCensored from Truncated
    else
        pc_type = inner_type # Already PrimaryCensored
    end

    D = pc_type.parameters[1] # Delay distribution type
    P = pc_type.parameters[2] # Primary distribution type

    delay_dist = D(delay_params...)
    primary_dist = P(primary_params...)

    individual_dists = if lowers === nothing && uppers === nothing
        [double_interval_censored(
             delay_dist, primary_dist; interval = int, force_numeric = force_numeric)
         for int in intervals]
    else
        # Handle vector of bounds
        lowers_vec = lowers === nothing ? fill(nothing, length(intervals)) :
                     (lowers isa AbstractVector ? lowers : fill(lowers, length(intervals)))
        uppers_vec = uppers === nothing ? fill(nothing, length(intervals)) :
                     (uppers isa AbstractVector ? uppers : fill(uppers, length(intervals)))

        [double_interval_censored(delay_dist, primary_dist; interval = int,
             lower = low, upper = upp, force_numeric = force_numeric)
         for (int, low, upp) in zip(intervals, lowers_vec, uppers_vec)]
    end

    return product_distribution(individual_dists)
end

# Vector of intervals - creates product distribution (kept for backwards compatibility)
function _dist_constructor(
        ::Type{IntervalCensored{<:PrimaryCensored{<:D, <:P, <:M}, <:Real}},
        delay_params::AbstractVector{<:Real},
        primary_params::AbstractVector{<:Real},
        intervals::AbstractVector{<:Real},
        force_numeric::Bool = true,
        lowers = nothing,
        uppers = nothing
) where {D <: ContinuousUnivariateDistribution, P <: ContinuousUnivariateDistribution,
        M <: CensoredDistributions.AbstractSolverMethod}
    delay_dist = D(delay_params...)
    primary_dist = P(primary_params...)

    individual_dists = if lowers === nothing && uppers === nothing
        [double_interval_censored(
             delay_dist, primary_dist; interval = int, force_numeric = force_numeric)
         for int in intervals]
    else
        # Handle vector of bounds
        lowers_vec = lowers === nothing ? fill(nothing, length(intervals)) :
                     (lowers isa AbstractVector ? lowers : fill(lowers, length(intervals)))
        uppers_vec = uppers === nothing ? fill(nothing, length(intervals)) :
                     (uppers isa AbstractVector ? uppers : fill(uppers, length(intervals)))

        [double_interval_censored(delay_dist, primary_dist; interval = int,
             lower = low, upper = upp, force_numeric = force_numeric)
         for (int, low, upp) in zip(intervals, lowers_vec, uppers_vec)]
    end

    return product_distribution(individual_dists)
end

# Heterogeneous primary distributions - creates product distribution with different primary events per observation
function _dist_constructor(
        ::Type{T},
        delay_params::AbstractVector{<:Real},
        primary_dists::AbstractVector{<:UnivariateDistribution},
        intervals::Union{Real, AbstractVector{<:Real}},
        force_numeric::Bool = true,
        lowers = nothing,
        uppers = nothing
) where {T <: IntervalCensored}
    delay_dist_type = T.parameters[1].parameters[1] # Extract delay distribution type from IntervalCensored{PrimaryCensored{D, P}}
    delay_dist = delay_dist_type(delay_params...)
    
    # Handle scalar interval case - broadcast to match primary_dists length
    intervals_vec = intervals isa Real ? fill(intervals, length(primary_dists)) : intervals
    
    # Handle bounds
    lowers_vec = lowers === nothing ? fill(nothing, length(primary_dists)) :
                 (lowers isa AbstractVector ? lowers : fill(lowers, length(primary_dists)))
    uppers_vec = uppers === nothing ? fill(nothing, length(primary_dists)) :
                 (uppers isa AbstractVector ? uppers : fill(uppers, length(primary_dists)))

    # Create individual distributions with heterogeneous primary events
    individual_dists = [
        double_interval_censored(delay_dist, primary_dist; 
            interval = int, lower = low, upper = upp, force_numeric = force_numeric)
        for (primary_dist, int, low, upp) in zip(primary_dists, intervals_vec, lowers_vec, uppers_vec)
    ]

    return product_distribution(individual_dists)
end

"""
    Distributions.fit_mle(dist::IntervalCensored{<:PrimaryCensored},
                         data::AbstractVector{<:Real};
                         intervals=nothing, lowers=nothing, uppers=nothing,
                         delay_init=nothing, primary_init=nothing,
                         primary_dists=nothing,
                         weights=nothing, optimizer=OptimizationOptimJL.BFGS())

Fit a double interval censored distribution to data using maximum likelihood estimation.
Uses the distribution types from `dist` for dispatch, eliminating manual type specification.

# Arguments
- `dist`: An `IntervalCensored{<:PrimaryCensored}` distribution instance (used for type dispatch)
- `data`: Vector of observed interval-censored values
- `intervals`: Interval specification - can be:
  - `nothing`: uses `dist.boundaries` (default)
  - `Real`: interval width for regular intervals
  - `AbstractVector{<:Real}`: heterogeneous intervals per observation
- `lowers`: Lower bounds - can be scalar or vector (optional)
- `uppers`: Upper bounds - can be scalar or vector (optional)
- `delay_init`: Initial parameters for delay distribution (optional, defaults from dist)
- `primary_init`: Initial parameters for primary distribution (optional, defaults from dist)
- `primary_dists`: Vector of primary event distributions for heterogeneous fitting (optional, parameters extracted automatically)
- `weights`: Optional observation weights
- `optimizer`: SciML optimizer (default: OptimizationOptimJL.BFGS())

# Returns
A fitted distribution with the same structure as the input type.

# Examples
```julia
# Fit with template distribution
template_dist = double_interval_censored(LogNormal(0, 1), Uniform(0, 7); interval=1.0, force_numeric=true)
fitted_dist = fit_mle(template_dist, data)

# Fit with heterogeneous intervals
fitted_dist = fit_mle(template_dist, data; intervals=[1.0, 2.0, 1.5, 1.0])

# Fit with heterogeneous primary event distributions
primary_dists = [Uniform(0, 1), Exponential(1.0), Uniform(0, 2)]
fitted_dist = fit_mle(template_dist, data[1:3]; primary_dists=primary_dists, intervals=[1.0, 1.5, 2.0])
```
"""
function Distributions.fit_mle(
        dist::IntervalCensored{<:PrimaryCensored{D, P}},
        data::AbstractVector{<:Real};
        intervals::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
        lowers::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
        uppers::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
        delay_init::Union{Nothing, AbstractVector{<:Real}} = nothing,
        primary_init::Union{Nothing, AbstractVector{<:Real}} = nothing,
        # Support for heterogeneous primary events
        primary_dists::Union{Nothing, AbstractVector{<:UnivariateDistribution}} = nothing,
        weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
        optimizer = OptimizationOptimJL.BFGS()
) where {D <: ContinuousUnivariateDistribution, P <: ContinuousUnivariateDistribution}

    # Solver types are supported for fitting - both analytical and numerical

    # Input validation
    _validate_data(data)
    _validate_weights(weights, data)

    # Determine interval structure
    interval_spec = intervals === nothing ? dist.boundaries : intervals

    # Handle heterogeneous primary distributions case
    if primary_dists !== nothing
        # Validate that primary_dists length matches data length
        length(primary_dists) == length(data) || 
            throw(ArgumentError("primary_dists length must match data length"))
        
        # Default delay parameter initialization from input distribution
        if delay_init === nothing
            delay_init = collect(params(dist.dist.dist))
        end
        
        # Create bijector only for delay parameters (primary parameters come from provided distributions)
        delay_bijector = _get_bijector(D, delay_init)
        
        # Create distribution constructor using heterogeneous dispatch
        function dist_constructor(params)
            delay_params = params  # Only delay parameters to optimize
            force_numeric = dist.dist.method isa NumericSolver
            return _dist_constructor(typeof(dist), delay_params, primary_dists,
                interval_spec, force_numeric, lowers, uppers)
        end
        
        # Optimize using the generic function
        fitted_params = _optimize_censored_distribution(
            data, delay_init, dist_constructor, delay_bijector, weights, optimizer
        )
        
        # Return fitted distribution with heterogeneous primary distributions
        force_numeric = dist.dist.method isa NumericSolver
        return _dist_constructor(typeof(dist), fitted_params, primary_dists,
            interval_spec, force_numeric, lowers, uppers)
    else
        # Homogeneous case (original logic)
        # Default parameter initialization from input distribution
        if delay_init === nothing
            delay_init = collect(params(dist.dist.dist))  # Get delay distribution params
        end
        if primary_init === nothing
            primary_init = collect(params(dist.dist.primary_event))  # Get primary distribution params
        end

        # Combine parameters and create combined bijector
        combined_params = vcat(delay_init, primary_init)
        delay_bijector = _get_bijector(D, delay_init)
        primary_bijector = _get_bijector(P, primary_init)
        # Create parameter ranges for Stacked bijector
        n_delay = length(delay_init)
        n_primary = length(primary_init)
        delay_range = 1:n_delay
        primary_range = (n_delay + 1):(n_delay + n_primary)

        combined_bijector = Stacked([delay_bijector, primary_bijector],
            [delay_range, primary_range])

        # Create distribution constructor using dispatch
        function dist_constructor(params)
            n_delay = length(delay_init)
            delay_params = params[1:n_delay]
            primary_params = params[(n_delay + 1):end]

            force_numeric = dist.dist.method isa NumericSolver
            return _dist_constructor(typeof(dist), delay_params, primary_params,
                interval_spec, force_numeric, lowers, uppers)
        end

        # Optimize using the generic function
        fitted_params = _optimize_censored_distribution(
            data, combined_params, dist_constructor, combined_bijector, weights, optimizer
        )

        # Extract fitted parameters and create final distribution
        n_delay = length(delay_init)
        fitted_delay_params = fitted_params[1:n_delay]
        fitted_primary_params = fitted_params[(n_delay + 1):end]

        # Return fitted distribution
        force_numeric = dist.dist.method isa NumericSolver
        return _dist_constructor(typeof(dist), fitted_delay_params, fitted_primary_params,
            interval_spec, force_numeric, lowers, uppers)
    end
end

"""
    Distributions.fit_mle(dist::IntervalCensored{<:Truncated{<:PrimaryCensored}},
                         data::AbstractVector{<:Real};
                         intervals=nothing, delay_init=nothing, primary_init=nothing,
                         weights=nothing, optimizer=OptimizationOptimJL.BFGS())

Fit a truncated double interval censored distribution to data.
Uses the distribution types from `dist` for dispatch.
"""
function Distributions.fit_mle(
        dist::IntervalCensored{<:Truncated{<:PrimaryCensored{D, P}}},
        data::AbstractVector{<:Real};
        intervals::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
        delay_init::Union{Nothing, AbstractVector{<:Real}} = nothing,
        primary_init::Union{Nothing, AbstractVector{<:Real}} = nothing,
        weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
        optimizer = OptimizationOptimJL.BFGS()
) where {D <: ContinuousUnivariateDistribution, P <: ContinuousUnivariateDistribution}

    # Solver types are supported for fitting - both analytical and numerical

    # Input validation
    _validate_data(data)
    _validate_weights(weights, data)

    # Extract information from truncated distribution
    truncated_dist = dist.dist
    interval_spec = intervals === nothing ? dist.boundaries : intervals
    lower_bound = truncated_dist.lower
    upper_bound = truncated_dist.upper

    # Default parameter initialization from input distribution
    if delay_init === nothing
        delay_init = collect(params(truncated_dist.untruncated.dist))
    end
    if primary_init === nothing
        primary_init = collect(params(truncated_dist.untruncated.primary_event))
    end

    # Combine parameters and create combined bijector
    combined_params = vcat(delay_init, primary_init)
    delay_bijector = _get_bijector(D, delay_init)
    primary_bijector = _get_bijector(P, primary_init)
    # Create parameter ranges for Stacked bijector
    n_delay = length(delay_init)
    n_primary = length(primary_init)
    delay_range = 1:n_delay
    primary_range = (n_delay + 1):(n_delay + n_primary)

    combined_bijector = Stacked([delay_bijector, primary_bijector],
        [delay_range, primary_range])

    # Create distribution constructor using dispatch
    function dist_constructor(params)
        n_delay = length(delay_init)
        delay_params = params[1:n_delay]
        primary_params = params[(n_delay + 1):end]

        force_numeric = dist.dist.untruncated.method isa NumericSolver
        return _dist_constructor(
            IntervalCensored{PrimaryCensored{D, P}, typeof(interval_spec)},
            delay_params, primary_params, interval_spec, force_numeric,
            lower_bound, upper_bound)
    end

    # Optimize using the generic function
    fitted_params = _optimize_censored_distribution(
        data, combined_params, dist_constructor, combined_bijector, weights, optimizer
    )

    # Extract fitted parameters and create final distribution
    n_delay = length(delay_init)
    fitted_delay_params = fitted_params[1:n_delay]
    fitted_primary_params = fitted_params[(n_delay + 1):end]

    # Return fitted distribution
    force_numeric = dist.dist.untruncated.method isa NumericSolver
    return _dist_constructor(
        IntervalCensored{PrimaryCensored{D, P}, typeof(interval_spec)},
        fitted_delay_params, fitted_primary_params, interval_spec, force_numeric,
        lower_bound, upper_bound)
end

# Convenience wrappers for fit()
function Distributions.fit(
        dist::IntervalCensored{<:PrimaryCensored},
        data::AbstractVector{<:Real};
        kwargs...
)
    return fit_mle(dist, data; kwargs...)
end

function Distributions.fit(
        dist::IntervalCensored{<:Truncated{<:PrimaryCensored}},
        data::AbstractVector{<:Real};
        kwargs...
)
    return fit_mle(dist, data; kwargs...)
end

