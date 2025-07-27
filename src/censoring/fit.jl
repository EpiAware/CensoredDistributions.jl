"""
Maximum Likelihood Estimation for CensoredDistributions.jl

This module provides MLE fitting functionality for censored distributions with
proper PDF implementations, following SciML Optimization.jl best practices and
the Distributions.jl interface.
"""

using Optimization
using OptimizationOptimJL
using Statistics

# Internal generic optimization function to reduce code duplication
"""
    _optimize_censored_distribution(data, param_specs, dist_constructor,
                                     weights, optimizer)

Internal function for optimizing parameters of censored distributions.

# Arguments
- `data`: Observed data values
- `param_specs`: Named tuple with parameter specifications
- `dist_constructor`: Function that constructs the distribution from parameters
- `weights`: Optional observation weights
- `optimizer`: SciML optimizer to use

# Returns
- Optimization result with fitted parameters
"""
function _optimize_censored_distribution(
        data::AbstractVector{<:Real},
        param_specs::NamedTuple,
        dist_constructor::Function,
        weights::Union{Nothing, AbstractVector},
        optimizer
)

    # Define objective function (negative log-likelihood)
    function objective(x, p)
        try
            # Construct distribution from parameters
            dist = p.dist_constructor(x, p)

            # Compute negative log-likelihood
            if p.weights === nothing
                loglik = sum(logpdf(dist, datum) for datum in p.data)
            else
                loglik = sum(
                    w * logpdf(dist, datum) for (w, datum) in zip(p.weights, p.data)
                )
            end
            return -loglik
        catch e
            # Return large value for invalid parameters
            return 1e10
        end
    end

    # Set up optimization parameters
    optimization_params = merge(
        param_specs, (data = data, weights = weights, dist_constructor = dist_constructor)
    )

    # Set up and solve optimization problem
    optfun = OptimizationFunction(objective, Optimization.AutoForwardDiff())
    optprob = OptimizationProblem(optfun, param_specs.x0, optimization_params)

    result = solve(optprob, optimizer)

    # Check if optimization succeeded
    if !SciMLBase.successful_retcode(result.retcode)
        @warn "Optimization did not converge successfully. " * "Retcode: $(result.retcode)"
    end

    return result
end

"""
    Distributions.fit_mle(::Type{IntervalCensored},
                         data::AbstractVector{<:Real};
                         dist_type=Normal, interval=1.0,
                         init_params=nothing, weights=nothing,
                         optimizer=OptimizationOptimJL.BFGS())

Fit an interval-censored distribution to data using maximum likelihood
estimation.

This extends the Distributions.jl `fit_mle` method to work with
`IntervalCensored` distributions.
The function estimates the parameters of the underlying continuous distribution
from interval-censored observations.

# Arguments
- `::Type{IntervalCensored}`: The distribution type to fit
- `data`: Vector of observed interval-censored values (interval left boundaries)
- `dist_type`: Type of underlying continuous distribution to fit
  (default: Normal)
- `interval`: Interval width for regular intervals (default: 1.0)
- `boundaries`: Custom interval boundaries (alternative to `interval`)
- `init_params`: Initial parameters for underlying distribution (optional)
- `weights`: Optional weights for observations (passed to underlying
  optimization)
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
function Distributions.fit_mle(
        ::Type{IntervalCensored},
        data::AbstractVector{<:Real};
        dist_type = Normal,
        interval = 1.0,
        boundaries = nothing,
        init_params = nothing,
        weights = nothing,
        optimizer = OptimizationOptimJL.BFGS()
)

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

    # Define distribution constructor
    function dist_constructor(x, p)
        # Transform parameters back to constrained space
        params = _transform_to_constrained(p.dist_type, x)
        # Create underlying distribution
        underlying_dist = p.dist_type(params...)
        # Create interval-censored distribution
        return interval_censored(underlying_dist, p.interval_spec)
    end

    # Set up parameter specifications
    param_specs = (x0 = x0, dist_type = dist_type, interval_spec = interval_spec)

    # Optimize using the generic function
    result = _optimize_censored_distribution(
        data, param_specs, dist_constructor, weights, optimizer
    )

    # Create fitted distribution
    fitted_params = _transform_to_constrained(dist_type, result.u)
    fitted_underlying = dist_type(fitted_params...)

    return interval_censored(fitted_underlying, interval_spec)
end

"""
    Distributions.fit(::Type{IntervalCensored},
                     data::AbstractVector{<:Real}; kwargs...)

Extend Distributions.jl fit method for IntervalCensored distributions.
This is a convenience wrapper around `fit_mle`.
"""
function Distributions.fit(
        ::Type{IntervalCensored}, data::AbstractVector{<:Real}; kwargs...
)
    return fit_mle(IntervalCensored, data; kwargs...)
end

# Internal utility functions

function _validate_data(data)
    length(data) > 0 || throw(ArgumentError("Data cannot be empty"))
    all(isfinite, data) || throw(ArgumentError("All data values must be finite"))
    # Note: Interval boundaries can be negative for distributions with
    # negative support
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

"""
Convert interval-censored data to approximate continuous values.
Used for parameter initialization.
"""
function _interval_to_continuous_approx(data, interval_spec)
    if isa(interval_spec, Real)
        # Regular intervals: data contains left boundaries, add half interval
        # width for midpoint
        return data .+ (interval_spec / 2)
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
        return continuous_approx
    end
end

function _get_default_init_params(::Type{<:Normal}, data, interval_spec)
    # Convert interval left boundaries to approximate continuous values
    continuous_approx = _interval_to_continuous_approx(data, interval_spec)

    μ_init = mean(continuous_approx)
    σ_init = std(continuous_approx)
    return [μ_init, σ_init]
end

function _get_default_init_params(::Type{<:Exponential}, data, interval_spec)
    # Convert interval left boundaries to approximate continuous values
    continuous_approx = _interval_to_continuous_approx(data, interval_spec)

    θ_init = mean(continuous_approx)  # Rate parameter
    return [θ_init]
end

function _get_default_init_params(dist_type, ::Any, ::Any)
    throw(
        ArgumentError(
        "Default initialization not implemented for $(dist_type). " *
        "Please provide init_params.",
    ),
    )
end

# Parameter transformation logic - generic approach

# Define parameter constraints for each distribution type
# Using a function to handle type hierarchies
function get_param_constraints(::Type{D}) where {D}
    if D <: Normal
        return [:real, :positive]      # μ (real), σ (positive)
    elseif D <: Exponential
        return [:positive]              # θ (positive)
    elseif D <: LogNormal
        return [:real, :positive]       # μ (real), σ (positive)
    elseif D <: Gamma
        return [:positive, :positive]   # α (positive), θ (positive)
    elseif D <: Uniform
        return [:lower_upper]           # special handling for a < b
    else
        return nothing
    end
end

"""
Transform constrained parameters to unconstrained space for optimization.
"""
function _transform_to_unconstrained(::Type{D}, params) where {D}
    constraints = get_param_constraints(D)
    constraints === nothing &&
        throw(ArgumentError("Parameter transformation not implemented for $(D)."))

    # Special case for Uniform distribution
    if D <: Uniform
        a, b = params
        return [a, log(b - a)]  # Transform to [a, log(width)]
    end

    # Generic transformation based on constraints
    unconstrained = similar(params)
    for (i, (param, constraint)) in enumerate(zip(params, constraints))
        unconstrained[i] = constraint == :positive ? log(param) : param
    end

    return unconstrained
end

"""
Transform unconstrained parameters back to constrained space.
"""
function _transform_to_constrained(::Type{D}, unconstrained) where {D}
    constraints = get_param_constraints(D)
    constraints === nothing &&
        throw(ArgumentError("Parameter transformation not implemented for $(D)."))

    # Special case for Uniform distribution
    if D <: Uniform
        a, log_width = unconstrained
        return [a, a + exp(log_width)]  # Transform back to [a, b]
    end

    # Generic transformation based on constraints
    constrained = similar(unconstrained)
    for (i, (param, constraint)) in enumerate(zip(unconstrained, constraints))
        constrained[i] = constraint == :positive ? exp(param) : param
    end

    return constrained
end

# Fitting support for Weighted distributions

"""
    Distributions.fit_mle(::Type{Weighted},
                         data::AbstractVector{<:Real};
                         underlying_dist_type=Normal,
                         weight_value=1.0, kwargs...)

Fit a weighted distribution to data using maximum likelihood estimation.

This extends the Distributions.jl `fit_mle` method to work with `Weighted`
distributions.
The weighted distribution is just a wrapper around an underlying distribution,
so this fits the underlying distribution and then wraps it with the specified
weight.

# Arguments
- `::Type{Weighted}`: The weighted distribution type to fit
- `data`: Vector of observed values
- `underlying_dist_type`: Type of underlying distribution to fit
  (default: Normal)
- `weight_value`: Weight value for the fitted distribution (default: 1.0)
- `kwargs...`: Additional arguments passed to the underlying distribution's
  fit method

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
function Distributions.fit_mle(
        ::Type{Weighted},
        data::AbstractVector{<:Real};
        underlying_dist_type = Normal,
        weight_value = 1.0,
        kwargs...
)

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

# Fitting support for Double Interval Censored distributions
# Only support force_numeric=true versions for numerical stability

"""
    Distributions.fit_mle(dist::IntervalCensored{<:PrimaryCensored},
                         data::AbstractVector{<:Real}; kwargs...)

Fit a double interval censored distribution to data using maximum likelihood
estimation.

This method handles distributions created by
`double_interval_censored(...; force_numeric=true)` that have numerical
solvers. The method extracts all necessary information from the distribution
object itself, providing a clean interface.

**Note**: Only distributions with `force_numeric=true` are supported for
fitting due to numerical stability requirements during optimization.

# Arguments
- `dist`: The distribution instance to fit (e.g., from
  `double_interval_censored(...; force_numeric=true)`)
- `data`: Vector of observed interval-censored values (interval left boundaries)
- `delay_init`: Initial parameters for delay distribution (optional)
- `primary_init`: Initial parameters for primary distribution (optional)
- `weights`: Optional observation weights
- `optimizer`: SciML optimizer (default: OptimizationOptimJL.BFGS())

# Returns
A fitted distribution with the same structure as the input type.

# Examples
```@example
using CensoredDistributions, Distributions

# Generate synthetic double interval censored data
true_delay = LogNormal(1.0, 0.8)
true_primary = Uniform(0, 2)
true_dist = double_interval_censored(true_delay, true_primary;
    interval=0.5, force_numeric=true)  # force_numeric required!
data = rand(true_dist, 1000)

# Fit using clean interface
fitted_dist = fit_mle(true_dist, data)
```
"""
function Distributions.fit_mle(
        dist::IntervalCensored{<:PrimaryCensored},
        data::AbstractVector{<:Real};
        delay_init = nothing,
        primary_init = nothing,
        weights = nothing,
        optimizer = OptimizationOptimJL.BFGS()
)

    # Runtime solver type checking
    if !(dist.dist.method isa CensoredDistributions.NumericSolver)
        throw(
            ArgumentError(
            "Fitting is only supported for distributions with " *
            "force_numeric=true. Please create your distribution with " *
            "double_interval_censored(...; force_numeric=true).",
        ),
        )
    end

    # Input validation
    _validate_data(data)
    _validate_weights(weights, data)

    # Extract types and parameters from the distribution object
    primary_censored = dist.dist
    D = typeof(primary_censored.dist)
    P = typeof(primary_censored.primary_event)
    interval_spec = dist.boundaries

    # Default parameter initialization
    if delay_init === nothing
        delay_init = _get_default_init_params_double(D, data, interval_spec)
    end
    if primary_init === nothing
        primary_init = _get_default_init_params_double(P, data, interval_spec)
    end

    # Transform to unconstrained space
    delay_unconstrained = _transform_to_unconstrained(D, delay_init)
    primary_unconstrained = _transform_to_unconstrained(P, primary_init)
    x0 = vcat(delay_unconstrained, primary_unconstrained)

    # Parameter organization
    n_delay_params = length(delay_init)
    n_primary_params = length(primary_init)

    # Define distribution constructor
    function dist_constructor(x, p)
        # Extract and transform parameters
        delay_params = _transform_to_constrained(p.delay_dist_type, x[1:p.n_delay_params])
        primary_params = _transform_to_constrained(
            p.primary_dist_type, x[(p.n_delay_params + 1):end]
        )

        # Create component distributions
        delay_dist = p.delay_dist_type(delay_params...)
        primary_dist = p.primary_dist_type(primary_params...)

        # Create double interval censored distribution with numerical methods
        return double_interval_censored(
            delay_dist, primary_dist; interval = p.interval_spec, force_numeric = true
        )
    end

    # Set up parameter specifications
    param_specs = (
        x0 = x0,
        delay_dist_type = D,
        primary_dist_type = P,
        interval_spec = interval_spec,
        n_delay_params = n_delay_params,
        n_primary_params = n_primary_params
    )

    # Optimize using the generic function
    result = _optimize_censored_distribution(
        data, param_specs, dist_constructor, weights, optimizer
    )

    # Extract fitted parameters
    fitted_delay_params = _transform_to_constrained(D, result.u[1:n_delay_params])
    fitted_primary_params = _transform_to_constrained(P, result.u[(n_delay_params + 1):end])

    # Create fitted distributions
    fitted_delay = D(fitted_delay_params...)
    fitted_primary = P(fitted_primary_params...)

    # Return fitted double interval censored distribution with numerical methods
    return double_interval_censored(
        fitted_delay, fitted_primary; interval = interval_spec, force_numeric = true
    )
end

"""
    Distributions.fit_mle(dist::IntervalCensored{<:Truncated{<:PrimaryCensored}},
                         data::AbstractVector{<:Real}; kwargs...)

Fit a truncated double interval censored distribution to data using maximum
likelihood estimation.

This method handles distributions created by
`double_interval_censored(...; force_numeric=true, upper=...)` with truncation
bounds. The method extracts all necessary information from the distribution
object.

**Note**: Only distributions with `force_numeric=true` are supported for
fitting.

# Arguments
- `dist`: The distribution instance to fit
- `data`: Vector of observed interval-censored values (interval left boundaries)
- `delay_init`: Initial parameters for delay distribution (optional)
- `primary_init`: Initial parameters for primary distribution (optional)
- `weights`: Optional observation weights
- `optimizer`: SciML optimizer (default: OptimizationOptimJL.BFGS())

# Returns
A fitted distribution with the same structure as the input.

# Examples
```@example
using CensoredDistributions, Distributions

# Generate synthetic truncated double interval censored data
true_delay = LogNormal(1.0, 0.8)
true_primary = Uniform(0, 2)
true_dist = double_interval_censored(true_delay, true_primary;
    interval=0.5, upper=10.0, force_numeric=true)
data = rand(true_dist, 1000)

# Fit using clean interface
fitted_dist = fit_mle(true_dist, data)
```
"""
function Distributions.fit_mle(
        dist::IntervalCensored{<:Truncated{<:PrimaryCensored}},
        data::AbstractVector{<:Real};
        delay_init = nothing,
        primary_init = nothing,
        weights = nothing,
        optimizer = OptimizationOptimJL.BFGS()
)

    # Runtime solver type checking
    if !(dist.dist.untruncated.method isa CensoredDistributions.NumericSolver)
        throw(
            ArgumentError(
            "Fitting is only supported for distributions with " *
            "force_numeric=true. Please create your distribution with " *
            "double_interval_censored(...; force_numeric=true).",
        ),
        )
    end

    # Input validation
    _validate_data(data)
    _validate_weights(weights, data)

    # Extract types and parameters from the distribution object
    truncated_dist = dist.dist
    primary_censored = truncated_dist.untruncated
    D = typeof(primary_censored.dist)
    P = typeof(primary_censored.primary_event)
    interval_spec = dist.boundaries
    lower_bound = truncated_dist.lower
    upper_bound = truncated_dist.upper

    # Default parameter initialization
    if delay_init === nothing
        delay_init = _get_default_init_params_double(D, data, interval_spec)
    end
    if primary_init === nothing
        primary_init = _get_default_init_params_double(P, data, interval_spec)
    end

    # Transform to unconstrained space
    delay_unconstrained = _transform_to_unconstrained(D, delay_init)
    primary_unconstrained = _transform_to_unconstrained(P, primary_init)
    x0 = vcat(delay_unconstrained, primary_unconstrained)

    # Parameter organization
    n_delay_params = length(delay_init)
    n_primary_params = length(primary_init)

    # Define distribution constructor
    function dist_constructor(x, p)
        # Extract and transform parameters
        delay_params = _transform_to_constrained(p.delay_dist_type, x[1:p.n_delay_params])
        primary_params = _transform_to_constrained(
            p.primary_dist_type, x[(p.n_delay_params + 1):end]
        )

        # Create component distributions
        delay_dist = p.delay_dist_type(delay_params...)
        primary_dist = p.primary_dist_type(primary_params...)

        # Create double interval censored distribution with numerical methods
        return double_interval_censored(
            delay_dist,
            primary_dist;
            interval = p.interval_spec,
            lower = p.lower_bound,
            upper = p.upper_bound,
            force_numeric = true
        )
    end

    # Set up parameter specifications
    param_specs = (
        x0 = x0,
        delay_dist_type = D,
        primary_dist_type = P,
        interval_spec = interval_spec,
        lower_bound = lower_bound,
        upper_bound = upper_bound,
        n_delay_params = n_delay_params,
        n_primary_params = n_primary_params
    )

    # Optimize using the generic function
    result = _optimize_censored_distribution(
        data, param_specs, dist_constructor, weights, optimizer
    )

    # Extract fitted parameters
    fitted_delay_params = _transform_to_constrained(D, result.u[1:n_delay_params])
    fitted_primary_params = _transform_to_constrained(P, result.u[(n_delay_params + 1):end])

    # Create fitted distributions
    fitted_delay = D(fitted_delay_params...)
    fitted_primary = P(fitted_primary_params...)

    # Return fitted double interval censored distribution with numerical methods
    return double_interval_censored(
        fitted_delay,
        fitted_primary;
        interval = interval_spec,
        lower = lower_bound,
        upper = upper_bound,
        force_numeric = true
    )
end

# Helper function for double censored parameter initialization
function _get_default_init_params_double(::Type{<:LogNormal}, data, interval)
    # For LogNormal, work backwards from interval data
    # Convert interval boundaries to approximate continuous values
    continuous_approx = _interval_to_continuous_approx(data, interval)
    # Remove non-positive values for LogNormal
    positive_data = continuous_approx[continuous_approx .> 0]
    if length(positive_data) == 0
        return [0.0, 1.0]  # Default LogNormal parameters
    end
    log_data = log.(positive_data)
    return [mean(log_data), std(log_data)]
end

function _get_default_init_params_double(::Type{<:Uniform}, data, interval)
    # For primary event Uniform, use conservative bounds based on data range
    continuous_approx = _interval_to_continuous_approx(data, interval)
    data_range = maximum(continuous_approx) - minimum(continuous_approx)
    # Conservative primary event window, minimum 1.0
    return [0.0, max(data_range * 0.3, 1.0)]
end

function _get_default_init_params(::Type{<:Gamma}, data, interval_spec)
    # Convert interval left boundaries to approximate continuous values
    continuous_approx = _interval_to_continuous_approx(data, interval_spec)

    # Method of moments for Gamma(α, θ)
    m = mean(continuous_approx)
    v = var(continuous_approx)

    # α = m²/v, θ = v/m
    α_init = m^2 / v
    θ_init = v / m

    return [α_init, θ_init]
end

function _get_default_init_params(::Type{<:Uniform}, data, interval_spec)
    # For Uniform distribution, estimate bounds from data
    continuous_approx = _interval_to_continuous_approx(data, interval_spec)

    # Use data range with some padding
    min_val = minimum(continuous_approx)
    max_val = maximum(continuous_approx)
    padding = (max_val - min_val) * 0.1

    return [min_val - padding, max_val + padding]
end

function _get_default_init_params_double(dist_type, ::Any, ::Any)
    throw(
        ArgumentError(
        "Default initialization not implemented for double censored " *
        "$(dist_type). Please provide initial parameters.",
    ),
    )
end

# Add fit() methods as convenience wrappers
function Distributions.fit(
        dist::IntervalCensored{<:PrimaryCensored}, data::AbstractVector{<:Real}; kwargs...
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
