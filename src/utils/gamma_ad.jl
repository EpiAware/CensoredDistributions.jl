@doc raw"""
AD-safe regularised lower incomplete gamma `P(a, z)` via the
absolutely-convergent series

```math
P(a, z) = z^a e^{-z} \sum_{n \geq 0} \frac{z^n}{\Gamma(a + n + 1)}
```

Term recurrence `term_{n+1} = term_n · z / (a + n + 1)`. Convergent for
all `z > 0`; converges fastest when `z \lesssim a`. Uses only `log`,
`exp`, `loggamma`, which all have ForwardDiff `DiffRules` entries, so
the series is dual-compatible without an explicit overload.
"""
function _gamma_p_series(a::Real, z::Real; rtol::Real = 1e-14, maxiter::Int = 10_000)
    z <= 0 && return zero(a) * zero(z)
    log_term0 = a * log(z) - z - loggamma(a + 1)
    term = exp(log_term0)
    s = term
    for n in 1:maxiter
        term *= z / (a + n)
        s += term
        abs(term) <= rtol * abs(s) && break
    end
    return s
end

@doc raw"""
Partial of the regularised lower incomplete gamma w.r.t. the shape
parameter via term-by-term differentiation of `_gamma_p_series`:

```math
\frac{\partial P(a, z)}{\partial a}
    = \log(z)\, P(a, z) -
      z^a e^{-z} \sum_{n \geq 0}
        \frac{\psi(a + n + 1)\, z^n}{\Gamma(a + n + 1)}
```

with `ψ(a + n + 1) = ψ(a + n) + 1 / (a + n)` propagated alongside the
same term recurrence. Used by the reverse-mode rules for [`_gamma_cdf`](@ref).
"""
function _grad_p_a_series(a::Real, z::Real; rtol::Real = 1e-14, maxiter::Int = 10_000)
    z <= 0 && return zero(a) * zero(z)
    log_term0 = a * log(z) - z - loggamma(a + 1)
    term = exp(log_term0)
    psi = digamma(a + 1)
    P = term
    S = term * psi
    for n in 1:maxiter
        term *= z / (a + n)
        psi += 1 / (a + n)
        P += term
        S += term * psi
        abs(term * psi) <= rtol * abs(S) &&
            abs(term) <= rtol * abs(P) && break
    end
    return log(z) * P - S
end

@doc raw"""
AD-safe Gamma CDF, `P(k, x/θ)`.

Computed via [`_gamma_p_series`](@ref) so the primal is dual-number
compatible (ForwardDiff). Reverse-mode AD picks up the analytic
gradient via the `ChainRulesCore.rrule` provided in the
`CensoredDistributionsChainRulesCoreExt` extension; Mooncake support is
wired in `CensoredDistributionsMooncakeExt`.

This replaces the earlier `HypergeometricFunctions.M`-based workaround,
removes a slow-path AD failure on Mooncake, and gives a single
implementation usable from both the analytical and numeric
primary-censored CDF paths.
"""
function _gamma_cdf(k::Real, θ::Real, x::Real)
    x <= 0 && return zero(k) * zero(θ) * zero(x)
    return _gamma_p_series(k, x / θ)
end

# Fast primal-only path for the AD-unwrapped, all-`Float64` case. Under AD
# the rrule on `_gamma_cdf(::Real, ::Real, ::Real)` intercepts the call
# before any method body runs, so the analytical α-gradient is preserved;
# this method is only hit by plain numeric calls (the hot path inside
# likelihood evaluation when no parameters are AD-tracked), where
# `SpecialFunctions.gamma_inc` is ~10x faster than the Julia series.
function _gamma_cdf(k::Float64, θ::Float64, x::Float64)
    x <= 0 && return 0.0
    return first(gamma_inc(k, x / θ))
end

@doc raw"""
AD-safe `logcdf(dist, u)` for use inside differentiable integrands.

Generic dispatch falls through to `Distributions.logcdf`. The
`Gamma` method routes through [`_gamma_cdf`](@ref) so its
`ChainRulesCore.rrule` is picked up by reverse-mode AD inside the
numeric `primarycensored_cdf` path; without this, the integrand calls
`gamma_inc` and breaks under every supported AD backend.
"""
_logcdf_ad_safe(dist::UnivariateDistribution, u::Real) = logcdf(dist, u)

function _logcdf_ad_safe(dist::Gamma, u::Real)
    u <= 0 && return oftype(float(u), -Inf)
    return log(_gamma_cdf(shape(dist), scale(dist), u))
end

@doc raw"""
AD-safe `cdf(dist, u)` companion to [`_logcdf_ad_safe`](@ref). Same
dispatch idea: route `Gamma` through [`_gamma_cdf`](@ref) so the
numeric-path optimisations that call `cdf(dist, lower)` for early
termination remain differentiable under reverse-mode AD.
"""
_cdf_ad_safe(dist::UnivariateDistribution, u::Real) = cdf(dist, u)

function _cdf_ad_safe(dist::Gamma, u::Real)
    return _gamma_cdf(shape(dist), scale(dist), u)
end
