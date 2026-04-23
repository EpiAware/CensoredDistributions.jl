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
    a = 1 + 1 / k
    inv_a = inv(a)

    function weibull_g_specialized(t::Real)
        if t <= 0
            return 0.0
        end
        # γ(a, x) = x^a / a · M(a, a+1, -x), using the AD-compatible
        # confluent hypergeometric M in place of `SpecialFunctions.gamma_inc`
        # (which is not differentiable w.r.t. the shape parameter).
        # See https://github.com/JuliaMath/HypergeometricFunctions.jl/issues/50#issuecomment-1397363491.
        # With x = (t/λ)^k, x^a = (t/λ)^(k·a) = (t/λ)^(k+1) = (t/λ) · x,
        # so we avoid a second real-exponent power call.
        u = t / λ
        x = u^k
        return u * x * inv_a * M(a, a + 1, -x)
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
    if isnan(x)
        return NaN
    elseif x <= minimum(dist)
        return 0.0
    elseif x == Inf
        return 1.0
    end

    # Define the integrand
    function integrand(u, x)
        return exp(logcdf(dist, u) +
                   logpdf(primary_event, x - u))
    end

    # Compute integration bounds
    lower = max(x - maximum(primary_event), minimum(dist))
    upper = x - minimum(primary_event)

    # Check if bounds are valid
    if upper <= lower || upper - lower ≈ 0.0
        return 0.0
    end

    # When the delay CDF at the lower bound is effectively
    # 1, integration is unnecessary and may produce NaN
    cdf_lower = cdf(dist, lower)
    if cdf_lower > 1 - eps(one(eltype(dist)))
        return one(cdf_lower)
    end

    # Set up and solve the integral problem
    prob = IntegralProblem(integrand, (lower, upper), x)
    result = solve(prob, method.solver)[1]

    # Clamp to valid CDF range for numerical stability
    return clamp(result, zero(result), one(result))
end

# ============================================================================
# Analytical CDF implementations for specific distribution pairs
# ============================================================================

@doc "
Direct CDF form of the primary-event-censored distribution with a Uniform
primary window, derived by integration by parts on the partial expectation:

```math
F_{S+}(d) = \\frac{d \\, F_T(d) - q \\, F_T(q) - (M_T(d) - M_T(q))}{w_P}
```

where ``d = x - u_{\\min}``, ``q = \\max(d - w_P, 0)``, ``w_P`` is the primary
window width, ``F_T`` is the delay CDF, and ``M_T(t) = \\int_0^t u \\, f_T(u) \\, du``
is the partial first moment of the delay.

This helper is the shared arithmetic shape of the three built-in analytical
pairs (Gamma / LogNormal / Weibull + Uniform) and is exposed as public (but
not exported) API so that downstream code can add its own
`primarycensored_cdf(::MyDelay, ::Uniform, x, ::AnalyticalSolver)` method
without re-deriving the formula. Supply ``M_T`` in whatever closed form the
delay distribution admits - for the built-ins:

- Gamma: ``M_T(t) = k\\theta \\cdot F_T(t; k+1, \\theta)``.
- LogNormal: ``M_T(t) = e^{\\mu + \\sigma^2/2} \\cdot F_T(t; \\mu + \\sigma^2, \\sigma)``.
- Weibull: ``M_T(t) = \\lambda \\cdot \\gamma(1 + 1/k, (t/\\lambda)^k)``.

When `q` is clamped to 0 (i.e. ``d \\le w_P``), pass `F_T_q = 0` and
`M_T_q = 0`; both terms drop out cleanly.

# Arguments

- `d::Real`: Primary-event-adjusted evaluation point (`x - u_{min}`).
- `q::Real`: Lower integration endpoint, `max(d - w_P, 0)`.
- `F_T_d::Real`: Delay CDF at `d`, `F_T(d)`.
- `F_T_q::Real`: Delay CDF at `q`, `F_T(q)` (pass `0` when `q = 0`).
- `M_T_d::Real`: Partial first moment at `d`, `M_T(d)`.
- `M_T_q::Real`: Partial first moment at `q`, `M_T(q)` (pass `0` when `q = 0`).
- `pwindow::Real`: Primary window width `w_P`.

# Returns

The primary-censored CDF `F_{S+}(x)`.
"
function primarycensored_cdf_formula(
        d, q, F_T_d, F_T_q, M_T_d, M_T_q, pwindow)
    return (d * F_T_d - q * F_T_q - (M_T_d - M_T_q)) / pwindow
end

@doc "

Analytical CDF for Gamma delay with Uniform primary event distribution.

Partial first moment is ``M_T(t) = k\\theta \\cdot F_T(t; k+1, \\theta)``. See the
[primarycensored R package](https://primarycensored.epinowcast.org/articles/analytic-solutions.html)
for the derivation.

``F_T(\\cdot; k+1, \\theta)`` is obtained from ``F_T(\\cdot; k, \\theta)`` via the
recursion ``P(k+1, y) = P(k, y) - y^k e^{-y} / \\Gamma(k+1)``, halving the
number of hypergeometric evaluations. `y^k` and `inv(Γ(k+1))` are shared
between ``F_T`` and ``M_T`` at each endpoint.
"
function primarycensored_cdf(
        dist::Gamma, primary_event::Uniform, x::Real, ::AnalyticalSolver
)
    k = shape(dist)
    θ = scale(dist)
    pwindow = maximum(primary_event) - minimum(primary_event)
    pmin = minimum(primary_event)

    d = x - pmin
    if d <= 0
        return 0.0
    end

    q = max(d - pwindow, 0.0)

    # Share `y^k / Γ(k+1)` between F_T(·; k), the F_T(·; k+1) recursion, and
    # the partial first moment M_T.
    # F_T(y; k)   = y^k · M(k, k+1, -y) / Γ(k+1)
    # F_T(y; k+1) = y^k · (M(k, k+1, -y) - exp(-y)) / Γ(k+1)       (recursion)
    # M_T(y)      = kθ · F_T(y; k+1)                               (partial first moment)
    # `M(k, k+1, ·)` is the AD-safe confluent hypergeometric (see _make_weibull_g).
    inv_gamma_kp1 = inv(gamma(k + 1))
    E_T = k * θ

    yd = d / θ
    coeff_d = yd^k * inv_gamma_kp1
    M_d = M(k, k + 1, -yd)
    F_T_d = coeff_d * M_d
    M_T_d = E_T * coeff_d * (M_d - exp(-yd))

    if q > 0
        yq = q / θ
        coeff_q = yq^k * inv_gamma_kp1
        M_q = M(k, k + 1, -yq)
        F_T_q = coeff_q * M_q
        M_T_q = E_T * coeff_q * (M_q - exp(-yq))
    else
        F_T_q = 0.0
        M_T_q = 0.0
    end

    return primarycensored_cdf_formula(d, q, F_T_d, F_T_q, M_T_d, M_T_q, pwindow)
end

@doc "

Analytical CDF for LogNormal delay with Uniform primary event distribution.

Partial first moment is
``M_T(t) = e^{\\mu + \\sigma^2/2} \\cdot F_T(t;\\, \\mu + \\sigma^2,\\, \\sigma)``,
i.e. the mean times the CDF of the parameter-shifted LogNormal. See the
[primarycensored R package](https://primarycensored.epinowcast.org/articles/analytic-solutions.html)
for the derivation.
"
function primarycensored_cdf(
        dist::LogNormal, primary_event::Uniform, x::Real, ::AnalyticalSolver
)
    μ = meanlogx(dist)
    σ = stdlogx(dist)
    pwindow = maximum(primary_event) - minimum(primary_event)
    pmin = minimum(primary_event)

    d = x - pmin
    if d <= 0
        return 0.0
    end

    q = max(d - pwindow, 0.0)

    dist_shifted = LogNormal(μ + σ^2, σ)
    E_T = exp(μ + σ^2 / 2)

    F_T_d = cdf(dist, d)
    M_T_d = E_T * cdf(dist_shifted, d)

    if q > 0
        F_T_q = cdf(dist, q)
        M_T_q = E_T * cdf(dist_shifted, q)
    else
        F_T_q = 0.0
        M_T_q = 0.0
    end

    return primarycensored_cdf_formula(d, q, F_T_d, F_T_q, M_T_d, M_T_q, pwindow)
end

@doc "

Analytical CDF for Weibull delay with Uniform primary event distribution.

Partial first moment is ``M_T(t) = \\lambda \\cdot g(t)`` where
``g(t) = \\gamma(1 + 1/k, (t/\\lambda)^k)`` is the lower incomplete gamma
helper. See the
[primarycensored R package](https://primarycensored.epinowcast.org/articles/analytic-solutions.html)
for the derivation.
"
function primarycensored_cdf(
        dist::Weibull, primary_event::Uniform, x::Real, ::AnalyticalSolver
)
    k = shape(dist)
    λ = scale(dist)
    pwindow = maximum(primary_event) - minimum(primary_event)
    pmin = minimum(primary_event)

    d = x - pmin
    if d <= 0
        return 0.0
    end

    q = max(d - pwindow, 0.0)

    weibull_g_func = _make_weibull_g(k, λ)
    F_T_d = cdf(dist, d)
    M_T_d = λ * weibull_g_func(d)

    if q > 0
        F_T_q = cdf(dist, q)
        M_T_q = λ * weibull_g_func(q)
    else
        F_T_q = 0.0
        M_T_q = 0.0
    end

    return primarycensored_cdf_formula(d, q, F_T_d, F_T_q, M_T_d, M_T_q, pwindow)
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
    if isnan(x)
        return NaN
    elseif x <= minimum(dist)
        return -Inf
    elseif x == Inf
        return 0.0
    end

    # Compute CDF and take log directly for type stability
    cdf_val = primarycensored_cdf(
        dist, primary_event, x, method)

    # Handle numerical precision issues where cdf_val might
    # be slightly negative
    if cdf_val <= 0
        return -Inf
    end

    return log(cdf_val)
end
