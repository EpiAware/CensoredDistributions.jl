@doc raw"""
Partial of the regularised lower incomplete gamma w.r.t. the shape
parameter — the term `SpecialFunctions.gamma_inc` leaves as
`@not_implemented` in its `ChainRule`. Computed by term-by-term
differentiation of the Tricomi absolutely-convergent series for
`P(a, z) = z^a e^{-z} Σ_{n ≥ 0} z^n / Γ(a + n + 1)`:

```math
\frac{\partial P(a, z)}{\partial a}
    = \log(z)\, P(a, z) -
      z^a e^{-z} \sum_{n \geq 0}
        \frac{\psi(a + n + 1)\, z^n}{\Gamma(a + n + 1)}
```

with `ψ(a + n + 1) = ψ(a + n) + 1 / (a + n)` propagated alongside the
term recurrence `term_{n+1} = term_n · z / (a + n + 1)`. Used by the
reverse-mode rule in `CensoredDistributionsChainRulesCoreExt` and by
the forward-mode `Dual` methods in `CensoredDistributionsForwardDiffExt`.
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

Primal goes through `SpecialFunctions.gamma_inc` for every `Real`
subtype it supports (`Float64`, `Float32`, `BigFloat`) — same path the
non-AD hot path uses, full accuracy across all `z/a` regimes. AD
coverage is supplied by per-backend extensions:

- `CensoredDistributionsChainRulesCoreExt` defines the
  reverse-mode `rrule` (analytical partials, primal via `gamma_inc`).
- `CensoredDistributionsMooncakeExt` lifts the rrule into Mooncake.
- `CensoredDistributionsReverseDiffExt` lifts the rrule into ReverseDiff.
- `CensoredDistributionsForwardDiffExt` defines `Dual` methods on
  `_gamma_cdf` directly (forward-mode dispatches on argument types,
  not via ChainRules).

The α-partial that `gamma_inc`'s `ChainRule` leaves as
`@not_implemented` is supplied by [`_grad_p_a_series`](@ref).
"""
function _gamma_cdf(k::Real, θ::Real, x::Real)
    x <= 0 && return zero(k) * zero(θ) * zero(x)
    kp, zp = promote(k, x / θ)
    return first(gamma_inc(kp, zp))
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
