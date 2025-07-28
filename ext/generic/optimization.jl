"""
Generic MLE fitting infrastructure for CensoredDistributions.jl

This file provides the core optimization infrastructure that is shared across
all censored distribution types.
"""

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

# Returns
- Optimization result with fitted parameters in constrained space
"""
function _optimize_censored_distribution(
        data::AbstractVector{<:Real},
        initial_params::AbstractVector{<:Real},
        dist_constructor::Function,
        bijector,
        weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
        optimizer = OptimizationOptimJL.BFGS()
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
    optfun = OptimizationFunction(objective, Optimization.AutoForwardDiff())
    optprob = OptimizationProblem(optfun, initial_unconstrained, optimization_params)

    result = solve(optprob, optimizer)

    # Check if optimization succeeded
    if !SciMLBase.successful_retcode(result.retcode)
        @warn "Optimization did not converge successfully. Retcode: $(result.retcode)"
    end

    # Transform result back to constrained space
    fitted_params = inverse(bijector)(result.u)

    return fitted_params
end

# Input validation functions
function _validate_data(data)
    length(data) > 0 || throw(ArgumentError("Data cannot be empty"))
    all(isfinite, data) || throw(ArgumentError("All data values must be finite"))
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
