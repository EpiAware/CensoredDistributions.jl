# Analytical CDF solutions for PrimaryCensored distributions

@doc raw"
Abstract type for solver methods used in CDF computation.

Subtypes determine whether analytical solutions are preferred or
numerical integration is forced.
"
abstract type AbstractSolverMethod end

@doc raw"
Solver that attempts analytical solutions when available, falling back to numerical integration.

Stores a numerical integration solver for use when no analytical solution exists
for a given distribution pair.
"
struct AnalyticalSolver{S} <: AbstractSolverMethod
    solver::S  # Fallback solver for when no analytical solution exists
end

@doc raw"
Solver that always uses numerical integration.

Forces numerical computation even when analytical solutions are available,
useful for testing and validation.
"
struct NumericSolver{S} <: AbstractSolverMethod
    solver::S
end

@doc raw"
Compute the CDF of a primary event censored distribution.

Dispatches to either analytical or numerical implementation based on the solver method.
Analytical solutions are available for specific distribution pairs with Uniform primary events.
Use `methods(primarycensored_cdf)` to see all available analytical implementations.

For distribution pairs without analytical solutions, numerical integration is used automatically
when called with an `AnalyticalSolver`. Use `NumericSolver` to force numerical integration
even when analytical solutions exist (useful for testing and validation).

# Arguments
- `dist`: The delay distribution from primary event to observation
- `primary_event`: The primary event time distribution
- `x`: Evaluation point for the CDF
- `method`: Solver method (`AnalyticalSolver` or `NumericSolver`)

# Returns
The cumulative probability P(X ≤ x) where X is the observed delay time.
"
function primarycensored_cdf(
        dist::D1, primary_event::D2,
        x::Real,
        method::AbstractSolverMethod
) where {D1 <: UnivariateDistribution, D2 <: UnivariateDistribution}
    error("primarycensored_cdf not implemented for method type $(typeof(method))")
end

@doc raw"
Generic fallback implementation for AnalyticalSolver.

When no specific analytical solution is available for a distribution pair,
this method falls back to numerical integration using the solver stored
in the AnalyticalSolver.

Specific analytical implementations exist for:
- Gamma delay with Uniform primary event
- LogNormal delay with Uniform primary event
- Weibull delay with Uniform primary event

For all other distribution pairs, numerical integration is used automatically.
"
function primarycensored_cdf(
        dist::D1, primary_event::D2,
        x::Real,
        method::AnalyticalSolver
) where {D1 <: UnivariateDistribution, D2 <: UnivariateDistribution}
    # Default behavior: use numerical integration
    primarycensored_cdf(dist, primary_event, x, NumericSolver(method.solver))
end

@doc raw"
Numerical CDF implementation for primary event censored distributions.

Computes the CDF using numerical integration when no analytical solution is available.
The integral computed is:
```math
F_{S+}(x) = \int_{max(x-u_{max}, s_{min})}^{x-u_{min}} F_S(u) f_U(x-u) du
```
where F_S is the delay distribution CDF, f_U is the primary event distribution PDF,
u_min and u_max are the primary event bounds, and s_min is the delay minimum.

Handles edge cases and uses the solver stored in the NumericSolver method.
"
function primarycensored_cdf(
        dist::D1, primary_event::D2,
        x::Real,
        method::NumericSolver
) where {D1 <: UnivariateDistribution, D2 <: UnivariateDistribution}
    # Edge cases
    if x <= minimum(dist)
        return 0.0
    elseif x == Inf
        return 1.0
    end

    # Define the integrand
    function integrand(u, x)
        return exp(logcdf(dist, u) + logpdf(primary_event, x - u))
    end

    # Compute integration bounds
    lower = max(x - maximum(primary_event), minimum(dist))
    upper = x - minimum(primary_event)

    # Check if bounds are valid
    if upper <= lower || upper - lower ≈ 0.0
        return 0.0
    end

    # Set up and solve the integral problem
    prob = IntegralProblem(integrand, (lower, upper), x)
    result = solve(prob, method.solver)[1]

    return result
end

# ============================================================================
# Analytical CDF implementations for specific distribution pairs
# ============================================================================

@doc raw"
Analytical CDF for Gamma delay with Uniform primary event distribution.

Implements the closed-form solution derived in Park et al. (2024) and
originally in Cori et al. (2013). Uses the partial expectation of the
Gamma distribution to avoid numerical integration.

The formula involves CDFs of Gamma distributions with shape parameters
k and k+1, computed in log-space for numerical stability.
"
function primarycensored_cdf(
        dist::Gamma, primary_event::Uniform, x::Real, ::AnalyticalSolver
)
    # Extract parameters
    k = shape(dist)  # shape parameter
    θ = scale(dist)  # scale parameter
    pwindow = maximum(primary_event) - minimum(primary_event)
    pmin = minimum(primary_event)

    # Adjust x for the primary event window offset
    t = x - pmin

    # Handle edge cases
    if t <= 0
        return 0.0
    end

    # Compute q = max(t - pwindow, 0)
    q = max(t - pwindow, 0.0)

    # Compute CDFs using Distributions.jl
    F_t = cdf(dist, t)

    # For the partial expectation, we need F(t; k+1, θ) and F(q; k+1, θ)
    dist_kplus1 = Gamma(k + 1, θ)
    F_t_kplus1 = cdf(dist_kplus1, t)

    if q > 0
        F_q = cdf(dist, q)
        F_q_kplus1 = cdf(dist_kplus1, q)

        # Compute differences
        ΔF_k = F_t - F_q
        ΔF_kplus1 = F_t_kplus1 - F_q_kplus1
    else
        ΔF_k = F_t
        ΔF_kplus1 = F_t_kplus1
    end

    # Compute the analytical CDF matching the Stan implementation
    # When q > 0: F_S+(d) = F_T(d) - exp(log_diff_exp(log(k*θ) + log(ΔF_{k+1}), log(d-w) + log(ΔF_k)) - log(w))
    # When q = 0: F_S+(d) = F_T(d) - exp(log_sum_exp(log(k*θ) + log(F_{k+1}), log(w-d) + log(F_k)) - log(w))

    if q > 0
        # Use log-space computation for numerical stability
        log_term1 = log(k * θ) + log(ΔF_kplus1)
        log_term2 = log(t - pwindow) + log(ΔF_k)
        log_diff = logsubexp(log_term1, log_term2) - log(pwindow)
        F_Splus = F_t - exp(log_diff)
    else
        # When q = 0, use log_sum_exp instead of log_diff_exp
        log_term1 = log(k * θ) + log(ΔF_kplus1)
        log_term2 = log(pwindow - t) + log(ΔF_k)
        log_sum = logaddexp(log_term1, log_term2) - log(pwindow)
        F_Splus = F_t - exp(log_sum)
    end

    return F_Splus
end

@doc raw"
Analytical CDF for LogNormal delay with Uniform primary event distribution.

Uses a parameter shift approach where the partial expectation of LogNormal(μ, σ)
can be expressed using the CDF of LogNormal(μ + σ², σ).
"
function primarycensored_cdf(
        dist::LogNormal, primary_event::Uniform, x::Real, ::AnalyticalSolver
)
    # Extract parameters
    μ = meanlogx(dist)
    σ = stdlogx(dist)
    pwindow = maximum(primary_event) - minimum(primary_event)
    pmin = minimum(primary_event)

    # Adjust x for the primary event window offset
    t = x - pmin

    # Handle edge cases
    if t <= 0
        return 0.0
    end

    # Compute q = max(t - pwindow, 0)
    q = max(t - pwindow, 0.0)

    # Compute CDFs using Distributions.jl
    F_t = cdf(dist, t)

    # For the partial expectation, we need F(t; μ+σ², σ)
    dist_shifted = LogNormal(μ + σ^2, σ)
    F_t_shifted = cdf(dist_shifted, t)

    if q > 0
        F_q = cdf(dist, q)
        F_q_shifted = cdf(dist_shifted, q)

        # Compute differences
        ΔF = F_t - F_q
        ΔF_shifted = F_t_shifted - F_q_shifted
    else
        ΔF = F_t
        ΔF_shifted = F_t_shifted
    end

    # Compute the analytical CDF matching the Stan implementation
    if q > 0
        # Use log-space computation for numerical stability
        log_term1 = (μ + 0.5 * σ^2) + log(ΔF_shifted)
        log_term2 = log(t - pwindow) + log(ΔF)
        log_diff = logsubexp(log_term1, log_term2) - log(pwindow)
        F_Splus = F_t - exp(log_diff)
    else
        # When q = 0, use log_sum_exp
        log_term1 = (μ + 0.5 * σ^2) + log(ΔF_shifted)
        log_term2 = log(pwindow - t) + log(ΔF)
        log_sum = logaddexp(log_term1, log_term2) - log(pwindow)
        F_Splus = F_t - exp(log_sum)
    end

    return F_Splus
end

@doc raw"
Analytical CDF for Weibull delay with Uniform primary event distribution.

Uses the lower incomplete gamma function to express the partial expectation
of the Weibull distribution analytically.
"
function primarycensored_cdf(
        dist::Weibull, primary_event::Uniform, x::Real, ::AnalyticalSolver
)
    # Extract parameters
    k = shape(dist)  # shape parameter
    λ = scale(dist)  # scale parameter
    pwindow = maximum(primary_event) - minimum(primary_event)
    pmin = minimum(primary_event)

    # Adjust x for the primary event window offset
    t = x - pmin

    # Handle edge cases
    if t <= 0
        return 0.0
    end

    # Compute q = max(t - pwindow, 0)
    q = max(t - pwindow, 0.0)

    # Compute CDFs using Distributions.jl
    F_t = cdf(dist, t)

    # Helper function for g(t; k, λ) = γ(1 + 1/k, (t/λ)^k)
    # where γ is the lower incomplete gamma function
    function weibull_g(t::Real, k::Real, λ::Real)
        if t <= 0
            return 0.0
        end
        x = (t / λ)^k
        a = 1 + 1/k
        # gamma_inc(a, x) returns (lower, upper) incomplete gamma functions
        lower_inc, _ = gamma_inc(a, x)
        return gamma(a) * lower_inc
    end

    # Compute g values
    g_t = weibull_g(t, k, λ)

    if q > 0
        F_q = cdf(dist, q)
        g_q = weibull_g(q, k, λ)

        # Compute differences
        ΔF = F_t - F_q
        Δg = g_t - g_q
    else
        ΔF = F_t
        Δg = g_t
    end

    # Compute the analytical CDF matching the Stan implementation
    if q > 0
        # Use log-space computation for numerical stability
        log_term1 = log(λ) + log(Δg)
        log_term2 = log(t - pwindow) + log(ΔF)
        log_diff = logsubexp(log_term1, log_term2) - log(pwindow)
        F_Splus = F_t - exp(log_diff)
    else
        # When q = 0, use log_sum_exp
        log_term1 = log(λ) + log(Δg)
        log_term2 = log(pwindow - t) + log(ΔF)
        log_sum = logaddexp(log_term1, log_term2) - log(pwindow)
        F_Splus = F_t - exp(log_sum)
    end

    return F_Splus
end

# ============================================================================
# Log-space implementations for numerical stability
# ============================================================================

@doc raw"
Compute the log CDF of a primary event censored distribution.

Computes log(CDF) with appropriate handling for edge cases where CDF = 0.
"
function primarycensored_logcdf(
        dist::D1, primary_event::D2,
        x::Real,
        method::AbstractSolverMethod
) where {D1 <: UnivariateDistribution, D2 <: UnivariateDistribution}
    if x <= minimum(dist)
        return -Inf
    elseif x == Inf
        return 0.0
    end

    # Compute CDF and take log
    cdf_val = primarycensored_cdf(dist, primary_event, x, method)
    return log(cdf_val)
end
