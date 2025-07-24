"""
Maximum Likelihood Estimation for CensoredDistributions.jl

This module provides MLE fitting functionality for censored distributions with proper PDF implementations,
following SciML Optimization.jl best practices and the Distributions.jl interface.
"""

using Optimization
using OptimizationOptimJL
using Statistics

"""
    Distributions.fit_mle(::Type{IntervalCensored}, data::AbstractVector{<:Real};
                         dist_type=Normal, interval=1.0, init_params=nothing,
                         weights=nothing, optimizer=OptimizationOptimJL.BFGS())

Fit an interval-censored distribution to data using maximum likelihood estimation.

This extends the Distributions.jl `fit_mle` method to work with `IntervalCensored` distributions.
The function estimates the parameters of the underlying continuous distribution
from interval-censored observations.

# Arguments
- `::Type{IntervalCensored}`: The distribution type to fit
- `data`: Vector of observed interval-censored values (interval left boundaries)
- `dist_type`: Type of underlying continuous distribution to fit (default: Normal)
- `interval`: Interval width for regular intervals (default: 1.0)
- `boundaries`: Custom interval boundaries (alternative to `interval`)
- `init_params`: Initial parameters for underlying distribution (optional)
- `weights`: Optional weights for observations (passed to underlying optimization)
- `optimizer`: SciML optimizer (default: OptimizationOptimJL.BFGS())

# Returns
A fitted `IntervalCensored` distribution with estimated parameters.

# Examples
```@example
using CensoredDistributions, Distributions

# Generate synthetic interval-censored data
true_underlying = Normal(2.0, 1.5)
true_censored = interval_censored(true_underlying, 0.5)
data = rand(true_censored, 1000)  # Returns interval left boundaries

# Fit the model
fitted_dist = fit_mle(IntervalCensored, data; interval=0.5)

# Compare parameters
println("True parameters: ", params(true_underlying))
println("Fitted parameters: ", params(fitted_dist.dist))
```
"""
function Distributions.fit_mle(::Type{IntervalCensored}, data::AbstractVector{<:Real};
        dist_type = Normal,
        interval = 1.0,
        boundaries = nothing,
        init_params = nothing,
        weights = nothing,
        optimizer = OptimizationOptimJL.BFGS())

    # Input validation
    _validate_data(data)
    _validate_weights(weights, data)

    # Determine interval structure
    interval_spec = boundaries === nothing ? interval : boundaries

    # Initialize parameters if not provided
    if init_params === nothing
        init_params = _get_default_init_params(dist_type, data, interval_spec)
    end

    # Transform to unconstrained space
    x0 = _transform_to_unconstrained(dist_type, init_params)

    # Set up optimization parameters
    optimization_params = (
        data = data,
        dist_type = dist_type,
        interval_spec = interval_spec,
        weights = weights
    )

    # Define objective function (negative log-likelihood)
    function objective(x, p)
        try
            # Transform parameters back to constrained space
            params = _transform_to_constrained(p.dist_type, x)

            # Create underlying distribution
            underlying_dist = p.dist_type(params...)

            # Create interval-censored distribution
            censored_dist = interval_censored(underlying_dist, p.interval_spec)

            # Compute negative log-likelihood (with optional weights)
            if p.weights === nothing
                loglik = sum(logpdf(censored_dist, datum) for datum in p.data)
            else
                loglik = sum(w * logpdf(censored_dist, datum)
                for (w, datum) in zip(p.weights, p.data))
            end
            return -loglik

        catch e
            # Return large value for invalid parameters
            return 1e10
        end
    end

    # Set up and solve optimization problem with automatic differentiation
    optfun = OptimizationFunction(objective, Optimization.AutoForwardDiff())
    optprob = OptimizationProblem(optfun, x0, optimization_params)
    result = solve(optprob, optimizer)

    # Create fitted distribution
    fitted_params = _transform_to_constrained(dist_type, result.u)
    fitted_underlying = dist_type(fitted_params...)

    return interval_censored(fitted_underlying, interval_spec)
end

"""
    Distributions.fit(::Type{IntervalCensored}, data::AbstractVector{<:Real}; kwargs...)

Extend Distributions.jl fit method for IntervalCensored distributions.
This is a convenience wrapper around `fit_mle`.
"""
function Distributions.fit(
        ::Type{IntervalCensored}, data::AbstractVector{<:Real}; kwargs...)
    return fit_mle(IntervalCensored, data; kwargs...)
end

# Internal utility functions

function _validate_data(data)
    length(data) > 0 || throw(ArgumentError("Data cannot be empty"))
    all(isfinite, data) || throw(ArgumentError("All data values must be finite"))
    # Note: Interval boundaries can be negative for distributions with negative support
end

function _validate_weights(weights, data)
    if weights !== nothing
        length(weights) == length(data) ||
            throw(ArgumentError("Weights must have same length as data"))
        all(w -> w >= 0, weights) ||
            throw(ArgumentError("All weights must be non-negative"))
        all(isfinite, weights) || throw(ArgumentError("All weights must be finite"))
        sum(weights) > 0 || throw(ArgumentError("At least one weight must be positive"))
    end
end

# Parameter initialization functions

function _get_default_init_params(::Type{Normal}, data, interval_spec)
    # Convert interval left boundaries to approximate continuous values
    if isa(interval_spec, Real)
        # Regular intervals: data contains left boundaries, add half interval width for midpoint
        continuous_approx = data .+ (interval_spec / 2)
    else
        # Arbitrary intervals: find midpoints
        continuous_approx = similar(data)
        for (i, left_boundary) in enumerate(data)
            # Find which interval this boundary corresponds to
            idx = findfirst(x -> x == left_boundary, interval_spec[1:(end - 1)])
            if idx !== nothing
                continuous_approx[i] = (interval_spec[idx] + interval_spec[idx + 1]) / 2
            else
                # Fallback: assume midpoint
                continuous_approx[i] = left_boundary + 0.5
            end
        end
    end

    μ_init = mean(continuous_approx)
    σ_init = std(continuous_approx)
    return [μ_init, σ_init]
end

function _get_default_init_params(::Type{Exponential}, data, interval_spec)
    # Convert interval left boundaries to approximate continuous values
    if isa(interval_spec, Real)
        # Regular intervals: data contains left boundaries, add half interval width for midpoint
        continuous_approx = data .+ (interval_spec / 2)
    else
        # Arbitrary intervals: find midpoints (same logic as Normal case)
        continuous_approx = similar(data)
        for (i, left_boundary) in enumerate(data)
            idx = findfirst(x -> x == left_boundary, interval_spec[1:(end - 1)])
            if idx !== nothing
                continuous_approx[i] = (interval_spec[idx] + interval_spec[idx + 1]) / 2
            else
                continuous_approx[i] = left_boundary + 0.5
            end
        end
    end

    θ_init = mean(continuous_approx)  # Rate parameter
    return [θ_init]
end

function _get_default_init_params(dist_type, data, interval_spec)
    throw(ArgumentError("Default initialization not implemented for $(dist_type). Please provide init_params."))
end

# Parameter transformation functions

function _transform_to_unconstrained(::Type{Normal}, params)
    μ, σ = params
    return [μ, log(σ)]  # σ must be positive
end

function _transform_to_constrained(::Type{Normal}, unconstrained)
    μ, log_σ = unconstrained
    return [μ, exp(log_σ)]
end

function _transform_to_unconstrained(::Type{Exponential}, params)
    θ, = params
    return [log(θ)]  # θ must be positive
end

function _transform_to_constrained(::Type{Exponential}, unconstrained)
    log_θ, = unconstrained
    return [exp(log_θ)]
end

function _transform_to_unconstrained(::Type{LogNormal}, params)
    μ, σ = params
    return [μ, log(σ)]  # μ can be any real, σ must be positive
end

function _transform_to_constrained(::Type{LogNormal}, unconstrained)
    μ, log_σ = unconstrained
    return [μ, exp(log_σ)]
end

function _transform_to_unconstrained(::Type{Uniform}, params)
    a, b = params
    # For Uniform(a,b), we transform to ensure a < b
    # We use log(b-a) to ensure positive width
    return [a, log(b - a)]
end

function _transform_to_constrained(::Type{Uniform}, unconstrained)
    a, log_width = unconstrained
    width = exp(log_width)
    return [a, a + width]  # Ensure b = a + width > a
end

function _transform_to_unconstrained(dist_type, params)
    throw(ArgumentError("Parameter transformation not implemented for $(dist_type)."))
end

function _transform_to_constrained(dist_type, unconstrained)
    throw(ArgumentError("Parameter transformation not implemented for $(dist_type)."))
end

# Fitting support for Weighted distributions

"""
    Distributions.fit_mle(::Type{Weighted}, data::AbstractVector{<:Real};
                         underlying_dist_type=Normal, weight_value=1.0, kwargs...)

Fit a weighted distribution to data using maximum likelihood estimation.

This extends the Distributions.jl `fit_mle` method to work with `Weighted` distributions.
The weighted distribution is just a wrapper around an underlying distribution,
so this fits the underlying distribution and then wraps it with the specified weight.

# Arguments
- `::Type{Weighted}`: The weighted distribution type to fit
- `data`: Vector of observed values
- `underlying_dist_type`: Type of underlying distribution to fit (default: Normal)
- `weight_value`: Weight value for the fitted distribution (default: 1.0)
- `kwargs...`: Additional arguments passed to the underlying distribution's fit method

# Returns
A fitted `Weighted` distribution wrapping the underlying fitted distribution.

# Examples
```@example
using CensoredDistributions, Distributions

# Generate synthetic data
data = rand(Normal(2.0, 1.5), 1000)

# Fit weighted distribution
fitted_weighted = fit_mle(Weighted, data; weight_value=10.0)

# This is equivalent to:
fitted_underlying = fit(Normal, data)
fitted_weighted = weight(fitted_underlying, 10.0)
```
"""
function Distributions.fit_mle(::Type{Weighted}, data::AbstractVector{<:Real};
        underlying_dist_type = Normal,
        weight_value = 1.0,
        kwargs...)

    # Input validation
    _validate_data(data)
    weight_value >= 0 || throw(ArgumentError("Weight value must be non-negative"))

    # Fit the underlying distribution using standard Distributions.jl fit method
    fitted_underlying = fit(underlying_dist_type, data; kwargs...)

    # Wrap with weight
    return weight(fitted_underlying, weight_value)
end

"""
    Distributions.fit(::Type{Weighted}, data::AbstractVector{<:Real}; kwargs...)

Extend Distributions.jl fit method for Weighted distributions.
This is a convenience wrapper around `fit_mle`.
"""
function Distributions.fit(::Type{Weighted}, data::AbstractVector{<:Real}; kwargs...)
    return fit_mle(Weighted, data; kwargs...)
end

# Fitting support for Double Interval Censored distributions (IntervalCensored{PrimaryCensored})

"""
    Distributions.fit_mle(::Type{IntervalCensored{PrimaryCensored{D,P,S}}}, data::AbstractVector{<:Real}; kwargs...)

Fit a double interval censored distribution (IntervalCensored{PrimaryCensored}) to data using maximum likelihood estimation.

This method handles the case where `double_interval_censored(delay_dist, primary_dist)` returns an
`IntervalCensored{PrimaryCensored{...}, ...}` type. It simultaneously estimates parameters for both
the delay distribution and primary event distribution.

# Arguments
- `data`: Vector of observed interval-censored values (interval left boundaries)
- `delay_dist_type`: Type of delay distribution (default: LogNormal)
- `primary_dist_type`: Type of primary event distribution (default: Uniform)
- `delay_init`: Initial parameters for delay distribution (optional)
- `primary_init`: Initial parameters for primary distribution (optional)
- `interval`: Interval width for secondary censoring
- `upper`: Upper truncation bound (if applicable)
- `lower`: Lower truncation bound (if applicable)
- `weights`: Optional observation weights
- `optimizer`: SciML optimizer (default: OptimizationOptimJL.BFGS())

# Returns
A fitted `IntervalCensored{PrimaryCensored{...}, ...}` distribution.

# Examples
```@example
using CensoredDistributions, Distributions

# Generate synthetic double interval censored data
true_delay = LogNormal(1.0, 0.8)
true_primary = Uniform(0, 2)
true_dist = double_interval_censored(true_delay, true_primary; interval=0.5)
data = rand(true_dist, 1000)

# Fit using standard Distributions.jl interface
fitted_dist = fit_mle(typeof(true_dist), data;
    delay_dist_type=LogNormal, primary_dist_type=Uniform)
```
"""
function Distributions.fit_mle(::Type{IntervalCensored{PrimaryCensored{D, P, S}}},
        data::AbstractVector{<:Real};
        delay_dist_type = D,
        primary_dist_type = P,
        delay_init = nothing,
        primary_init = nothing,
        interval = 1.0,
        upper = nothing,
        lower = nothing,
        weights = nothing,
        optimizer = OptimizationOptimJL.BFGS()) where {D, P, S}
    return fit_double_interval_censored(data;
        delay_dist_type = delay_dist_type,
        primary_dist_type = primary_dist_type,
        delay_init = delay_init,
        primary_init = primary_init,
        interval = interval,
        upper = upper,
        lower = lower,
        weights = weights,
        optimizer = optimizer)
end

# Generic method for IntervalCensored{PrimaryCensored} types when type parameters aren't known
function Distributions.fit_mle(::Type{<:IntervalCensored{<:PrimaryCensored}},
        data::AbstractVector{<:Real}; kwargs...)
    return fit_double_interval_censored(data; kwargs...)
end

"""
    fit_double_interval_censored(data::AbstractVector{<:Real};
                                delay_dist_type=LogNormal, primary_dist_type=Uniform,
                                interval, upper=nothing, lower=nothing,
                                delay_init=nothing, primary_init=nothing,
                                optimizer=OptimizationOptimJL.BFGS())

Fit a double interval censored distribution to data using maximum likelihood estimation.

This function simultaneously estimates parameters for both the delay distribution and
primary event distribution from observations that underwent both primary event censoring
and interval censoring.

# Arguments
- `data`: Vector of observed interval-censored values (interval left boundaries)
- `delay_dist_type`: Type of delay distribution (default: LogNormal)
- `primary_dist_type`: Type of primary event distribution (default: Uniform)
- `interval`: Interval width for secondary censoring
- `upper`: Upper truncation bound (if applicable)
- `lower`: Lower truncation bound (if applicable)
- `delay_init`: Initial parameters for delay distribution (optional)
- `primary_init`: Initial parameters for primary distribution (optional)
- `optimizer`: SciML optimizer (default: OptimizationOptimJL.BFGS())

# Returns
A fitted double interval censored distribution (via `double_interval_censored`).

# Examples
```@example
using CensoredDistributions, Distributions

# Generate synthetic double interval censored data
true_delay = LogNormal(1.0, 0.8)
true_primary = Uniform(0, 2)
true_dist = double_interval_censored(true_delay, true_primary; interval=0.5)
data = rand(true_dist, 1000)

# Fit the model
fitted_dist = fit_double_interval_censored(data; interval=0.5)

# Compare parameters
println("True delay: ", params(true_delay))
println("Fitted delay: ", params(fitted_dist.dist.dist))  # Note: nested structure
```
"""
function fit_double_interval_censored(data::AbstractVector{<:Real};
        delay_dist_type = LogNormal,
        primary_dist_type = Uniform,
        interval = 1.0,
        upper = nothing,
        lower = nothing,
        delay_init = nothing,
        primary_init = nothing,
        weights = nothing,
        optimizer = OptimizationOptimJL.BFGS())

    # Input validation
    _validate_data(data)
    _validate_weights(weights, data)

    # Default parameter initialization
    if delay_init === nothing
        delay_init = _get_default_init_params_double(delay_dist_type, data, interval)
    end

    if primary_init === nothing
        primary_init = _get_default_init_params_double(primary_dist_type, data, interval)
    end

    # Transform to unconstrained space
    delay_unconstrained = _transform_to_unconstrained(delay_dist_type, delay_init)
    primary_unconstrained = _transform_to_unconstrained(primary_dist_type, primary_init)
    x0 = vcat(delay_unconstrained, primary_unconstrained)

    # Parameter organization
    n_delay_params = length(delay_init)
    n_primary_params = length(primary_init)

    # Set up optimization parameters
    optimization_params = (
        data = data,
        delay_dist_type = delay_dist_type,
        primary_dist_type = primary_dist_type,
        interval = interval,
        upper = upper,
        lower = lower,
        n_delay_params = n_delay_params,
        n_primary_params = n_primary_params,
        weights = weights
    )

    # Define objective function (negative log-likelihood)
    function objective(x, p)
        try
            # Extract and transform parameters
            delay_params = _transform_to_constrained(
                p.delay_dist_type, x[1:p.n_delay_params])
            primary_params = _transform_to_constrained(
                p.primary_dist_type, x[(p.n_delay_params + 1):end])

            # Create component distributions
            delay_dist = p.delay_dist_type(delay_params...)
            primary_dist = p.primary_dist_type(primary_params...)

            # Create double interval censored distribution with numerical methods
            double_dist = double_interval_censored(delay_dist, primary_dist;
                interval = p.interval,
                upper = p.upper,
                lower = p.lower,
                force_numeric = true)

            # Compute negative log-likelihood (with optional weights)
            if p.weights === nothing
                loglik = sum(logpdf(double_dist, datum) for datum in p.data)
            else
                loglik = sum(w * logpdf(double_dist, datum)
                for (w, datum) in zip(p.weights, p.data))
            end
            return -loglik

        catch e
            # Return large value for invalid parameters
            return 1e10
        end
    end

    # Set up and solve optimization problem
    optfun = OptimizationFunction(objective, Optimization.AutoForwardDiff())
    optprob = OptimizationProblem(optfun, x0, optimization_params)
    result = solve(optprob, optimizer)

    # Extract fitted parameters
    fitted_delay_params = _transform_to_constrained(
        delay_dist_type, result.u[1:n_delay_params])
    fitted_primary_params = _transform_to_constrained(
        primary_dist_type, result.u[(n_delay_params + 1):end])

    # Create fitted distributions
    fitted_delay = delay_dist_type(fitted_delay_params...)
    fitted_primary = primary_dist_type(fitted_primary_params...)

    # Return fitted double interval censored distribution with numerical methods
    return double_interval_censored(fitted_delay, fitted_primary;
        interval = interval,
        upper = upper,
        lower = lower,
        force_numeric = true)
end

# Helper function for double censored parameter initialization
function _get_default_init_params_double(::Type{LogNormal}, data, interval)
    # For LogNormal, work backwards from interval data
    # Convert interval boundaries to approximate continuous values
    continuous_approx = data .+ (interval / 2)
    # Remove non-positive values for LogNormal
    positive_data = continuous_approx[continuous_approx .> 0]
    if length(positive_data) == 0
        return [0.0, 1.0]  # Default LogNormal parameters
    end
    log_data = log.(positive_data)
    return [mean(log_data), std(log_data)]
end

function _get_default_init_params_double(::Type{Uniform}, data, interval)
    # For primary event Uniform, use conservative bounds based on data range
    continuous_approx = data .+ (interval / 2)
    data_range = maximum(continuous_approx) - minimum(continuous_approx)
    return [0.0, max(data_range * 0.3, 1.0)]  # Conservative primary event window, minimum 1.0
end

function _get_default_init_params(::Type{Uniform}, data, interval_spec)
    # For Uniform distribution, estimate bounds from data
    if isa(interval_spec, Real)
        continuous_approx = data .+ (interval_spec / 2)
    else
        # Handle arbitrary intervals
        continuous_approx = similar(data)
        for (i, left_boundary) in enumerate(data)
            idx = findfirst(x -> x == left_boundary, interval_spec[1:(end - 1)])
            if idx !== nothing
                continuous_approx[i] = (interval_spec[idx] + interval_spec[idx + 1]) / 2
            else
                continuous_approx[i] = left_boundary + 0.5
            end
        end
    end

    # Use data range with some padding
    min_val = minimum(continuous_approx)
    max_val = maximum(continuous_approx)
    padding = (max_val - min_val) * 0.1

    return [min_val - padding, max_val + padding]
end

function _get_default_init_params_double(dist_type, data, interval)
    throw(ArgumentError("Default initialization not implemented for double censored $(dist_type). Please provide initial parameters."))
end
