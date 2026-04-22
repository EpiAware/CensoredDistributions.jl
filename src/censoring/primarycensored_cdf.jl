@doc "

AD-compatible log gamma CDF using HypergeometricFunctions.

Computes ``\\log P(k, x/\\theta) = \\log \\gamma(k, x/\\theta) - \\log \\Gamma(k)``
via the identity ``\\gamma(k, z) = z^k / k \\cdot M(k, k+1, -z)``, where ``M``
is the confluent hypergeometric function. The hypergeometric form is
differentiable with respect to the shape parameter, unlike
`SpecialFunctions.gamma_inc`.

Returns `-Inf` for `x <= 0`.
"
function _log_gamma_cdf_ad_safe(k::Real, θ::Real, x::Real)
    T = float(promote_type(typeof(k), typeof(θ), typeof(x)))
    if x <= 0
        return T(-Inf)
    end
    z = x / θ
    m_val = M(k, k + 1, -z)
    # At very large z, M underflows or goes slightly negative from roundoff.
    # CDF saturates at 1, so return log(1) = 0.
    m_val <= 0 && return zero(T)
    logp = k * log(z) - log(k) + log(m_val) - loggamma(k)
    # Clamp to log(1) = 0 to defend against floating-point noise pushing the
    # result above zero.
    return min(logp, zero(logp))
end

@doc "

Function factory for the log of the Weibull g helper.

Returns a specialised function computing
``\\log g(t; k, \\lambda) = \\log \\gamma(1 + 1/k, (t/\\lambda)^k)`` with
``a = 1 + 1/k`` and ``\\log \\gamma(a, x) = a \\log x - \\log a + \\log M(a, a+1, -x)``
precomputed. The AD-safe confluent hypergeometric form is used so the log
is differentiable with respect to the Weibull shape.

Returns `-Inf` for `t <= 0`.
"
function _make_log_weibull_g(k::Real, λ::Real)
    inv_k = 1 / k
    a = 1 + inv_k
    log_gamma_a = loggamma(a)  # log(Γ(a)); saturation value of log g

    function log_weibull_g_specialized(t::Real)
        T = float(promote_type(typeof(k), typeof(λ), typeof(t)))
        t <= 0 && return T(-Inf)
        x = (t / λ)^k
        m_val = M(a, a + 1, -x)
        # γ(a, x) → Γ(a) for large x, so g → Γ(a); guard underflow / noise.
        m_val <= 0 && return log_gamma_a
        logg = a * log(x) - log(a) + log(m_val)
        return min(logg, log_gamma_a)
    end

    return log_weibull_g_specialized
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
# Analytical log-CDF implementations for specific distribution pairs
# ============================================================================
#
# For the analytical cases, the log CDF is the primitive: everything is
# computed in log space and `primarycensored_cdf` just calls `exp(...)`. This
# mirrors the Stan implementation in the primarycensored R package and keeps
# the arithmetic stable for autodiff - there is no catastrophic cancellation
# from subtracting close positive quantities, and no round-trip through
# `exp`/`log` when downstream consumers (e.g. `logpdf`) want a log value.

@doc "
Log-space form of the direct CDF identity

```math
F_{S+}(d) = \\frac{d\\,F_T(d) + E\\,\\tilde F_T(q) - q\\,F_T(q) - E\\,\\tilde F_T(d)}{w_P}.
```

Computed as ``\\log\\mathrm{diffexp}(\\log A, \\log B) - \\log w_P`` with
``\\log A = \\log\\mathrm{sumexp}(\\log d + \\log F_T(d),\\ \\log E + \\log \\tilde F_T(q))``
and ``\\log B = \\log\\mathrm{sumexp}(\\log q + \\log F_T(q),\\ \\log E + \\log \\tilde F_T(d))``.
The non-negativity of ``F_{S+}`` guarantees ``\\log A \\ge \\log B``. Terms with
``q = 0`` are passed in as ``-\\infty`` so the two ``\\log\\mathrm{sumexp}``
calls degenerate cleanly.
"
@inline function _analytical_logcdf_formula(
        log_d_F_T_d, log_E_tF_T_d, log_q_F_T_q, log_E_tF_T_q, log_pwindow
)
    log_A = logaddexp(log_d_F_T_d, log_E_tF_T_q)
    log_B = logaddexp(log_q_F_T_q, log_E_tF_T_d)
    # At saturation log_A ≈ log_B; clamp to log(1) = 0 to defend against
    # floating-point noise pushing the result above zero.
    return min(logsubexp(log_A, log_B) - log_pwindow, zero(log_pwindow))
end

@doc "

Analytical log CDF for Gamma delay with Uniform primary event distribution.

All arithmetic is in log space. ``F_T(\\cdot; k)`` is evaluated with the
AD-safe hypergeometric form; the shifted CDF ``F_T(\\cdot; k+1, \\theta)`` is
obtained from the recursion
``P(k+1, y) = P(k, y) - y^k e^{-y}/\\Gamma(k+1)``
via `log_diff_exp`, which halves the number of hypergeometric evaluations.
See the
[primarycensored R package](https://primarycensored.epinowcast.org/articles/analytic-solutions.html).
"
function primarycensored_logcdf(
        dist::Gamma, primary_event::Uniform, x::Real, ::AnalyticalSolver
)
    k = shape(dist)
    θ = scale(dist)
    pwindow = maximum(primary_event) - minimum(primary_event)
    pmin = minimum(primary_event)

    isnan(x) && return oftype(float(k / θ * x), NaN)
    x == Inf && return zero(float(k / θ * x))
    d = x - pmin
    d <= 0 && return oftype(float(k / θ * d), -Inf)

    q = max(d - pwindow, 0.0)
    log_pwindow = log(pwindow)
    log_E = log(k) + log(θ)  # E[T] = k θ

    log_F_T_d_k = _log_gamma_cdf_ad_safe(k, θ, d)
    # Recursion: log F_T(d; k+1) = log_diff_exp(log F_T(d; k),
    #                              k log(d/θ) - d/θ - logΓ(k+1))
    log_pdf_kp1_d = k * log(d / θ) - d / θ - loggamma(k + 1)
    log_F_T_d_kp1 = logsubexp(log_F_T_d_k, log_pdf_kp1_d)

    if q > 0
        log_F_T_q_k = _log_gamma_cdf_ad_safe(k, θ, q)
        log_pdf_kp1_q = k * log(q / θ) - q / θ - loggamma(k + 1)
        log_F_T_q_kp1 = logsubexp(log_F_T_q_k, log_pdf_kp1_q)

        log_d_F_T_d = log(d) + log_F_T_d_k
        log_q_F_T_q = log(q) + log_F_T_q_k
        log_E_tF_T_d = log_E + log_F_T_d_kp1
        log_E_tF_T_q = log_E + log_F_T_q_kp1
    else
        log_d_F_T_d = log(d) + log_F_T_d_k
        log_q_F_T_q = oftype(log_F_T_d_k, -Inf)
        log_E_tF_T_d = log_E + log_F_T_d_kp1
        log_E_tF_T_q = oftype(log_F_T_d_k, -Inf)
    end

    return _analytical_logcdf_formula(
        log_d_F_T_d, log_E_tF_T_d, log_q_F_T_q, log_E_tF_T_q, log_pwindow
    )
end

@doc "

Analytical log CDF for LogNormal delay with Uniform primary event distribution.

Computed in log space using the parameter-shift identity: the partial
expectation of ``\\mathrm{LogNormal}(\\mu, \\sigma)`` is proportional to the
CDF of ``\\mathrm{LogNormal}(\\mu + \\sigma^2, \\sigma)``, with
``E[T] = \\exp(\\mu + \\sigma^2/2)``. See the
[primarycensored R package](https://primarycensored.epinowcast.org/articles/analytic-solutions.html).
"
function primarycensored_logcdf(
        dist::LogNormal, primary_event::Uniform, x::Real, ::AnalyticalSolver
)
    μ = meanlogx(dist)
    σ = stdlogx(dist)
    pwindow = maximum(primary_event) - minimum(primary_event)
    pmin = minimum(primary_event)

    isnan(x) && return oftype(float(μ * σ * x), NaN)
    x == Inf && return zero(float(μ * σ * x))
    d = x - pmin
    d <= 0 && return oftype(float(μ * σ * d), -Inf)

    q = max(d - pwindow, 0.0)
    log_pwindow = log(pwindow)
    log_E = μ + σ^2 / 2

    dist_shifted = LogNormal(μ + σ^2, σ)
    log_F_T_d = logcdf(dist, d)
    log_F_T_d_shift = logcdf(dist_shifted, d)

    if q > 0
        log_F_T_q = logcdf(dist, q)
        log_F_T_q_shift = logcdf(dist_shifted, q)

        log_d_F_T_d = log(d) + log_F_T_d
        log_q_F_T_q = log(q) + log_F_T_q
        log_E_tF_T_d = log_E + log_F_T_d_shift
        log_E_tF_T_q = log_E + log_F_T_q_shift
    else
        log_d_F_T_d = log(d) + log_F_T_d
        log_q_F_T_q = oftype(log_F_T_d, -Inf)
        log_E_tF_T_d = log_E + log_F_T_d_shift
        log_E_tF_T_q = oftype(log_F_T_d, -Inf)
    end

    return _analytical_logcdf_formula(
        log_d_F_T_d, log_E_tF_T_d, log_q_F_T_q, log_E_tF_T_q, log_pwindow
    )
end

@doc "

Analytical log CDF for Weibull delay with Uniform primary event distribution.

Computed in log space. The role of ``E[T] \\cdot \\tilde F_T(t)`` is played
by ``\\lambda \\cdot g(t)`` with
``g(t) = \\gamma(1 + 1/k, (t/\\lambda)^k)``. `_make_log_weibull_g` returns
``\\log g`` via the AD-safe hypergeometric form. See the
[primarycensored R package](https://primarycensored.epinowcast.org/articles/analytic-solutions.html).
"
function primarycensored_logcdf(
        dist::Weibull, primary_event::Uniform, x::Real, ::AnalyticalSolver
)
    k = shape(dist)
    λ = scale(dist)
    pwindow = maximum(primary_event) - minimum(primary_event)
    pmin = minimum(primary_event)

    isnan(x) && return oftype(float(k / λ * x), NaN)
    x == Inf && return zero(float(k / λ * x))
    d = x - pmin
    d <= 0 && return oftype(float(k / λ * d), -Inf)

    q = max(d - pwindow, 0.0)
    log_pwindow = log(pwindow)
    log_scale = log(λ)

    log_weibull_g_func = _make_log_weibull_g(k, λ)
    log_F_T_d = logcdf(dist, d)
    log_g_d = log_weibull_g_func(d)

    if q > 0
        log_F_T_q = logcdf(dist, q)
        log_g_q = log_weibull_g_func(q)

        log_d_F_T_d = log(d) + log_F_T_d
        log_q_F_T_q = log(q) + log_F_T_q
        log_E_tF_T_d = log_scale + log_g_d
        log_E_tF_T_q = log_scale + log_g_q
    else
        log_d_F_T_d = log(d) + log_F_T_d
        log_q_F_T_q = oftype(log_F_T_d, -Inf)
        log_E_tF_T_d = log_scale + log_g_d
        log_E_tF_T_q = oftype(log_F_T_d, -Inf)
    end

    return _analytical_logcdf_formula(
        log_d_F_T_d, log_E_tF_T_d, log_q_F_T_q, log_E_tF_T_q, log_pwindow
    )
end

# CDF wrappers for the analytical pairs: cdf = exp(lcdf). The log form is the
# primitive - this exists only so the `cdf` interface still resolves to the
# analytical path.
function primarycensored_cdf(
        dist::Gamma, primary_event::Uniform, x::Real, method::AnalyticalSolver
)
    return exp(primarycensored_logcdf(dist, primary_event, x, method))
end

function primarycensored_cdf(
        dist::LogNormal, primary_event::Uniform, x::Real, method::AnalyticalSolver
)
    return exp(primarycensored_logcdf(dist, primary_event, x, method))
end

function primarycensored_cdf(
        dist::Weibull, primary_event::Uniform, x::Real, method::AnalyticalSolver
)
    return exp(primarycensored_logcdf(dist, primary_event, x, method))
end

# ============================================================================
# Generic log-CDF dispatch (fallback / numerical path)
# ============================================================================

@doc "

Compute the log CDF of a primary event censored distribution.

For distribution pairs with a specific analytical implementation (e.g.
Gamma, LogNormal, Weibull with Uniform primary), the log CDF is the
primitive: it is computed directly in log space and
`primarycensored_cdf` is defined as `exp(primarycensored_logcdf(...))`.

For all other pairs, this generic method falls back to taking `log` of
`primarycensored_cdf`, which itself either hits a specific analytical CDF
method or numerical integration. Edge cases (``x`` below the support or
``+\\infty``) short-circuit without calling the CDF path.

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
