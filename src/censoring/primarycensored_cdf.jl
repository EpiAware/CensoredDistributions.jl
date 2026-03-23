@doc "

AD-compatible gamma CDF using HypergeometricFunctions.

Based on the identity: γ(a,z) = z^a/a * M(a, a+1, -z)
where γ is the lower incomplete gamma function and M is the confluent hypergeometric function.

Uses the same approach as the Weibull g function: P(a,z) = γ(a,z)/Γ(a) = z^a/a * M(a, a+1, -z) / Γ(a).
For integer a, Γ(a) = (a-1)!, but uses gamma(k) for generality.

# Arguments
- `k::Real`: Shape parameter
- `θ::Real`: Scale parameter
- `x::Real`: Evaluation point

# Returns
The gamma CDF value at x with shape k and scale θ.
"
function _gamma_cdf_ad_safe(k::Real, θ::Real, x::Real)
    if x <= 0
        return 0.0
    end
    z = x / θ
    return (z^k / k * M(k, k + 1, -z)) / gamma(k)
end

@doc "

Function factory for optimized Weibull g function.

Creates a specialized function with pre-computed constants for g(t; k, λ) = γ(1 + 1/k, (t/λ)^k)
where γ is the lower incomplete gamma function.

Pre-computes `inv_k = 1/k` and `a = 1 + inv_k` to avoid repeated computation
in the returned specialized function.

# Arguments
- `k::Real`: Weibull shape parameter
- `λ::Real`: Weibull scale parameter

# Returns
A specialized function `weibull_g_specialized(t::Real)` that efficiently computes
the Weibull g function using pre-computed constants.

# Examples
```julia
weibull_g_func = _make_weibull_g(2.0, 1.5)
g_val = weibull_g_func(3.0)
```
"
function _make_weibull_g(k::Real, λ::Real)
    inv_k = 1 / k
    a = 1 + inv_k

    function weibull_g_specialized(t::Real)
        if t <= 0
            return 0.0
        end
        x = (t / λ)^k
        # Use AD-compatible confluent hypergeometric function instead of gamma_inc
        # γ(a,z) = z^a/a * M(a, a+1, -z) where M is the confluent hypergeometric function
        # See: https://github.com/JuliaMath/HypergeometricFunctions.jl/issues/50#issuecomment-1397363491
        # This avoids gamma_inc which causes AD issues
        return x^a / a * M(a, a + 1, -x)
    end

    return weibull_g_specialized
end

@doc "
Abstract type for solver methods used in CDF computation.

Subtypes determine whether analytical solutions are preferred or
numerical integration is forced.
"
abstract type AbstractSolverMethod end

@doc "
Solver that attempts analytical solutions when available, falling back to numerical integration.

Stores a numerical integration solver for use when no analytical solution exists
for a given distribution pair.
"
struct AnalyticalSolver{S} <: AbstractSolverMethod
    solver::S  # Fallback solver for when no analytical solution exists
end

@doc "
Solver that always uses numerical integration.

Forces numerical computation even when analytical solutions are available,
useful for testing and validation.

The `solver` field contains the numerical integration solver to use.
"
struct NumericSolver{S} <: AbstractSolverMethod
    solver::S
end

@doc "
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

# Examples
```@example
using CensoredDistributions, Distributions, Integrals

# Analytical solution for Gamma with Uniform primary event
gamma_dist = Gamma(2.0, 1.5)
primary_uniform = Uniform(0, 2)
analytical_method = AnalyticalSolver(QuadGKJL())

cdf_val = primarycensored_cdf(gamma_dist, primary_uniform, 3.0, analytical_method)

# Force numerical integration
numeric_method = NumericSolver(QuadGKJL())
cdf_numeric = primarycensored_cdf(gamma_dist, primary_uniform, 3.0, numeric_method)
```

"
function primarycensored_cdf(
        dist::D1, primary_event::D2,
        x::Real,
        method::AbstractSolverMethod
) where {D1 <: UnivariateDistribution, D2 <: UnivariateDistribution}
    error("primarycensored_cdf not implemented for method type $(typeof(method))")
end

@doc "
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

@doc "
Numerical CDF implementation for primary event censored distributions.

Computes the CDF using numerical integration when no analytical solution is available.
The integral computed is:
```math
F_{S+}(x) = \\int_{\\max(x-u_{\\max}, s_{\\min})}^{x-u_{\\min}} F_S(u) f_U(x-u) du
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

@doc "

Analytical CDF for Gamma delay with Uniform primary event distribution.

Uses the partial expectation of the Gamma distribution to
avoid numerical integration. See the
[primarycensored R package](https://primarycensored.epinowcast.org/articles/analytic-solutions.html)
for the derivation.

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

    # Compute CDFs using AD-safe gamma CDF
    F_t = _gamma_cdf_ad_safe(k, θ, t)

    # For the partial expectation, we need F(t; k+1, θ) and F(q; k+1, θ)
    F_t_kplus1 = _gamma_cdf_ad_safe(k + 1, θ, t)

    if q > 0
        F_q = _gamma_cdf_ad_safe(k, θ, q)
        F_q_kplus1 = _gamma_cdf_ad_safe(k + 1, θ, q)

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

    # Handle numerical precision issues where differences might be slightly negative
    ΔF_k = max(ΔF_k, 0.0)
    ΔF_kplus1 = max(ΔF_kplus1, 0.0)

    if q > 0
        # Use log-space computation for numerical stability
        # Handle edge case where t - pwindow might be very close to 0
        t_minus_pwindow = max(t - pwindow, 0.0)
        log_term1 = log(k * θ) + log(ΔF_kplus1)
        log_term2 = log(t_minus_pwindow) + log(max(ΔF_k, 0.0))
        log_diff = logsubexp(log_term1, log_term2) - log(pwindow)
        F_Splus = F_t - exp(log_diff)
    else
        # When q = 0, use log_sum_exp instead of log_diff_exp
        # Handle edge case where pwindow - t might be very close to 0
        pwindow_minus_t = max(pwindow - t, 0.0)
        log_term1 = log(k * θ) + log(ΔF_kplus1)
        log_term2 = log(pwindow_minus_t) + log(max(ΔF_k, 0.0))
        log_sum = logaddexp(log_term1, log_term2) - log(pwindow)
        F_Splus = F_t - exp(log_sum)
    end

    return F_Splus
end

@doc "

Analytical CDF for LogNormal delay with Uniform primary event distribution.

Uses a parameter shift approach where the partial expectation of LogNormal(μ, σ)
can be expressed using the CDF of LogNormal(μ + σ², σ). See the
[primarycensored R package](https://primarycensored.epinowcast.org/articles/analytic-solutions.html)
for the derivation.
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
    # Handle numerical precision issues where differences might be slightly negative
    ΔF = max(ΔF, 0.0)
    ΔF_shifted = max(ΔF_shifted, 0.0)

    if q > 0
        # Use log-space computation for numerical stability
        # Handle edge case where t - pwindow might be very close to 0
        t_minus_pwindow = max(t - pwindow, 0.0)
        log_term1 = (μ + 0.5 * σ^2) + log(ΔF_shifted)
        log_term2 = log(t_minus_pwindow) + log(max(ΔF, 0.0))
        log_diff = logsubexp(log_term1, log_term2) - log(pwindow)
        F_Splus = F_t - exp(log_diff)
    else
        # When q = 0, use log_sum_exp
        # Handle edge case where pwindow - t might be very close to 0
        pwindow_minus_t = max(pwindow - t, 0.0)
        log_term1 = (μ + 0.5 * σ^2) + log(ΔF_shifted)
        log_term2 = log(pwindow_minus_t) + log(max(ΔF, 0.0))
        log_sum = logaddexp(log_term1, log_term2) - log(pwindow)
        F_Splus = F_t - exp(log_sum)
    end

    return F_Splus
end

@doc "

Analytical CDF for Weibull delay with Uniform primary event distribution.

Uses the lower incomplete gamma function to express the partial expectation
of the Weibull distribution analytically. See the
[primarycensored R package](https://primarycensored.epinowcast.org/articles/analytic-solutions.html)
for the derivation.
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

    # Create optimized g function with pre-computed constants
    weibull_g_func = _make_weibull_g(k, λ)

    # Compute g values using optimized function
    g_t = weibull_g_func(t)

    if q > 0
        F_q = cdf(dist, q)
        g_q = weibull_g_func(q)

        # Compute differences
        ΔF = F_t - F_q
        Δg = g_t - g_q
    else
        ΔF = F_t
        Δg = g_t
    end

    # Compute the analytical CDF matching the Stan implementation
    # Handle numerical precision issues where Δg or ΔF might be slightly negative
    Δg = max(Δg, 0.0)
    ΔF = max(ΔF, 0.0)

    if q > 0
        # Use log-space computation for numerical stability
        # Handle edge case where t - pwindow might be very close to 0
        t_minus_pwindow = max(t - pwindow, 0.0)
        log_term1 = log(λ) + log(Δg)
        log_term2 = log(t_minus_pwindow) + log(max(ΔF, 0.0))
        log_diff = logsubexp(log_term1, log_term2) - log(pwindow)
        F_Splus = F_t - exp(log_diff)
    else
        # When q = 0, use log_sum_exp
        # Handle edge case where pwindow - t might be very close to 0
        pwindow_minus_t = max(pwindow - t, 0.0)
        log_term1 = log(λ) + log(Δg)
        log_term2 = log(pwindow_minus_t) + log(max(ΔF, 0.0))
        log_sum = logaddexp(log_term1, log_term2) - log(pwindow)
        F_Splus = F_t - exp(log_sum)
    end

    return F_Splus
end

# ============================================================================
# Log-space implementations for numerical stability
# ============================================================================

@doc "

Compute the log CDF of a primary event censored distribution.

Dispatches to either analytical or numerical implementation based on the solver method.
Computes log(CDF) with appropriate handling for edge cases where CDF = 0.
Returns -Inf when the probability is zero or when numerical issues occur.

# Arguments
- `dist`: The delay distribution from primary event to observation
- `primary_event`: The primary event time distribution
- `x`: Evaluation point for the log CDF
- `method`: Solver method (`AnalyticalSolver` or `NumericSolver`)

# Examples
```@example
using CensoredDistributions, Distributions

# Create a Gamma delay with uniform primary event
dist = Gamma(2.0, 1.5)
primary_event = Uniform(0.0, 3.0)
method = AnalyticalSolver(nothing)

# Compute log CDF at x = 5.0
log_prob = primarycensored_logcdf(dist, primary_event, 5.0, method)
```

# See also
- [`primarycensored_cdf`](@ref): The underlying CDF computation
- [`cdf`](@ref): Interface method for PrimaryCensored distributions
- [`logcdf`](@ref): Interface method for PrimaryCensored distributions
"
function primarycensored_logcdf(
        dist::D1, primary_event::D2,
        x::Real,
        method::AbstractSolverMethod
) where {D1 <: UnivariateDistribution, D2 <: UnivariateDistribution}
    # Check support first for type stability
    if x <= minimum(dist)
        return -Inf
    elseif x == Inf
        return 0.0
    end

    # Compute CDF and take log directly for type stability
    try
        cdf_val = primarycensored_cdf(dist, primary_event, x, method)

        # Handle numerical precision issues where cdf_val might be slightly negative
        if cdf_val <= 0
            return -Inf
        end

        return log(cdf_val)
    catch e
        # If analytical solution fails (e.g., domain error), return -Inf log probability
        if isa(e, DomainError) || isa(e, BoundsError) || isa(e, ArgumentError)
            return -Inf
        else
            rethrow(e)
        end
    end
end
