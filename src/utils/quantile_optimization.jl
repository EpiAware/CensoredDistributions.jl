using Optimization: ReturnCode

@doc raw"
Internal utilities for quantile optimization shared across censored distributions.

Provides common optimization logic for computing quantiles when analytical
solutions are not available.
"

@doc raw"
    _quantile_optimization(d, p;
                          initial_guess_fn=nothing,
                          result_postprocess_fn=identity,
                          check_nan=false)

Internal function for quantile optimization using numerical methods.

Solves the equation `cdf(d, q) - p = 0` using the Nelder-Mead algorithm.
This is shared logic used by both `PrimaryCensored` and `IntervalCensored`
quantile functions.

# Arguments
- `d`: The distribution for which to compute the quantile
- `p`: The probability value in [0, 1]

# Keyword Arguments
- `initial_guess_fn`: Function that takes `(d, p)` and returns initial guess
  vector. If `nothing`, uses `quantile(get_dist(d), p)` as scalar initial guess.
- `result_postprocess_fn`: Function to post-process the optimization result.
  Defaults to identity (no post-processing).
- `check_nan`: If `true`, explicitly check for NaN input values.

# Returns
The quantile value after optimization and post-processing.

# Implementation Details
- Validates that p ∈ [0, 1] (with optional NaN checking)
- Handles boundary cases p=0 (minimum) and p=1 (maximum) analytically
- Creates objective function `(cdf(d, q) - p)^2` with support checking
- Uses heavy penalty for values outside distribution support
- Solves with Nelder-Mead algorithm with tight tolerances
- Checks convergence and applies post-processing to result

# Type Stability
This function is designed to maintain type stability when the initial guess
function and post-processing function are type-stable.
"
function _quantile_optimization(d, p::Real;
        initial_guess_fn = nothing,
        result_postprocess_fn = identity,
        check_nan::Bool = false)
    # Handle NaN input if requested
    if check_nan && isnan(p)
        throw(ArgumentError("p must be in [0, 1], got NaN"))
    end

    # Validate p is in [0, 1]
    if p < 0.0 || p > 1.0
        throw(ArgumentError("p must be in [0, 1]"))
    end

    # Handle boundary cases analytically
    if p == 0.0
        return minimum(d)
    elseif p == 1.0
        return maximum(d)
    end

    # Create objective function with proper support checking
    objective = function (q, _)
        q_val = q[1]
        # If outside support, apply large penalty to guide optimization
        if !insupport(d, q_val)
            return 1e10 + (q_val - minimum(d))^2
        end
        cdf_val = cdf(d, q_val)
        return (cdf_val - p)^2
    end

    # Compute initial guess
    if initial_guess_fn === nothing
        # Default: use quantile of underlying distribution
        underlying_quantile = float(quantile(get_dist(d), p))
        x0 = [underlying_quantile]
    else
        x0 = initial_guess_fn(d, p)
    end

    # Set up and solve optimization problem
    optfun = OptimizationFunction(objective)
    prob = OptimizationProblem(optfun, x0, nothing)

    sol = solve(prob, NelderMead();
        reltol = 1e-8, abstol = 1e-8, maxiters = 10000)

    # Check convergence
    if sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.Default
        result = sol.u[1]
        return result_postprocess_fn(result)
    else
        error("Quantile optimization failed to converge for p = $p")
    end
end
