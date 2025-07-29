"""
Generic MLE fitting infrastructure for CensoredDistributions.jl

This file provides the core optimization infrastructure that is shared across
all censored distribution types.
"""

"""
    _handle_fit_result(fitted_params, optimization_result, return_fit_object,
                      dist_constructor_func, additional_args...)

External function to handle fit result formatting across different fit functions.
This centralizes the logic for returning either just the fitted distribution or
a tuple with the optimization result.

# Arguments
- `fitted_params`: The optimized parameters
- `optimization_result`: The full optimization result object
- `return_fit_object`: Boolean flag for return format
- `dist_constructor_func`: Function to construct the fitted distribution
- `additional_args...`: Additional arguments passed to the distribution constructor

# Returns
- If `return_fit_object=false`: The fitted distribution
- If `return_fit_object=true`: Tuple of (fitted_distribution, optimization_result)
"""
function _handle_fit_result(
        fitted_params::AbstractVector{<:Real},
        optimization_result,
        return_fit_object::Bool,
        dist_constructor_func::Function,
        additional_args...
)
    fitted_dist = dist_constructor_func(fitted_params, additional_args...)
    return return_fit_object ? (fitted_dist, optimization_result) : fitted_dist
end

# Internal generic optimization function
"""
    _optimize_censored_distribution(data, initial_params, dist_constructor,
                                   bijector, weights, optimizer)

Internal function for optimizing parameters of censored distributions using
bijectors for parameter transformations.

# Arguments
- `data`: Observed data values
- `initial_params`: Initial parameter values in constrained space
- `dist_constructor`: Function that constructs the distribution from parameters
- `bijector`: Bijector for parameter transformation
- `weights`: Optional observation weights
- `optimizer`: SciML optimizer to use

# Keyword Arguments
- `autodiff`: Automatic differentiation method for optimization (default: Optimization.AutoForwardDiff())

# Returns
Vector of optimized parameters in constrained space and the optimization result as a tuple
"""
function _optimize_censored_distribution(
        data::AbstractVector{<:Real},
        initial_params::AbstractVector{<:Real},
        dist_constructor::Function,
        bijector,
        weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
        optimizer = OptimizationOptimJL.LBFGS();
        autodiff = Optimization.AutoForwardDiff()
)
    # Transform to unconstrained space
    initial_unconstrained = bijector(initial_params)

    # Define objective function (negative log-likelihood)
    function objective(x, p)
        try
            # Transform back to constrained space
            constrained_params = inverse(p.bijector)(x)

            # Construct distribution from parameters
            dist = p.dist_constructor(constrained_params)

            # Compute negative log-likelihood
            if p.weights === nothing
                loglik = sum(logpdf(dist, datum) for datum in p.data)
            else
                loglik = sum(
                    w * logpdf(dist, datum) for (w, datum) in zip(p.weights, p.data)
                )
            end

            # Add log absolute determinant of Jacobian for proper transformation
            log_abs_det_jac = logabsdetjac(p.bijector, constrained_params)

            return -(loglik + log_abs_det_jac)
        catch e
            # Return large value for invalid parameters
            return 1e10
        end
    end

    # Set up optimization parameters
    optimization_params = (
        data = data,
        weights = weights,
        dist_constructor = dist_constructor,
        bijector = bijector
    )

    # Set up and solve optimization problem
    # Create optimization function with specified autodiff method
    optfun = OptimizationFunction(objective, autodiff)
    optprob = OptimizationProblem(optfun, initial_unconstrained, optimization_params)

    result = solve(optprob, optimizer)

    # Check if optimization succeeded and produced meaningful results
    if !SciMLBase.successful_retcode(result.retcode)
        error("Optimization failed to converge. Retcode: $(result.retcode). " *
              "This indicates the fitting was unsuccessful. " *
              "Try different initial parameters, a different optimizer, " *
              "or verify that your data is compatible with the distribution.")
    end

    # Check for infinite objective values (indicates likelihood calculation issues)
    if !isfinite(result.objective)
        @warn "Optimization resulted in infinite objective value ($(result.objective)). " *
              "This typically indicates parameter transformation issues or data incompatibility. " *
              "Returning initial parameters. Try different initial parameters or check data compatibility."
        return (initial_params, result)
    end

    # Transform result back to constrained space
    fitted_params = inverse(bijector)(result.u)

    return (fitted_params, result)
end

# Input validation functions
function _validate_data(data)
    length(data) > 0 || throw(ArgumentError("Data cannot be empty"))
    all(isfinite, data) || throw(ArgumentError("All data values must be finite"))
end

function _validate_weights(weights, data)
    if weights !== nothing
        length(weights) == length(data) ||
            throw(DimensionMismatch("Weights must have same length as data"))
        all(w -> w >= 0, weights) ||
            throw(ArgumentError("All weights must be non-negative"))
        all(isfinite, weights) || throw(ArgumentError("All weights must be finite"))
        sum(weights) > 0 || throw(ArgumentError("At least one weight must be positive"))
    end
end
