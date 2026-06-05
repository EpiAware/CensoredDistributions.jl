@doc """
Partial of the regularised lower incomplete gamma with respect to the
shape parameter — the term `SpecialFunctions.gamma_inc` leaves as
`@not_implemented` in its `ChainRule`. Computed by term-by-term
differentiation of the Tricomi absolutely-convergent series for
`P(a, z) = z^a e^{-z} Σ_{n ≥ 0} z^n / Γ(a + n + 1)`:

```math
\\frac{\\partial P(a, z)}{\\partial a}
    = \\log(z)\\, P(a, z) -
      z^a e^{-z} \\sum_{n \\geq 0}
        \\frac{\\psi(a + n + 1)\\, z^n}{\\Gamma(a + n + 1)}
```

with `ψ(a + n + 1) = ψ(a + n) + 1 / (a + n)` propagated alongside the
term recurrence `term_{n+1} = term_n · z / (a + n + 1)`. Used by the
reverse-mode rule in `CensoredDistributionsChainRulesCoreExt` and by
the forward-mode `Dual` methods in `CensoredDistributionsForwardDiffExt`.

# References

The series + digamma-recurrence form is Moore (1982), "Algorithm AS
187: Derivatives of the Incomplete Gamma Integral", *Applied
Statistics* 31:330-335. The same construction is used by Stan
(`stan/math/prim/fun/grad_reg_inc_gamma.hpp`) and JAX
(`jax._src.scipy.special.random_gamma_grad` / `igamma_grad_a`) for the
shape derivative of the regularised lower incomplete gamma.
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

@doc """
AD-safe Gamma CDF, `P(k, x/θ)`.

Primal goes through `SpecialFunctions.gamma_inc` for every `Real`
subtype it supports (`Float64`, `Float32`, `BigFloat`) — same path the
non-AD hot path uses, full accuracy across all `z/a` regimes. AD
coverage is supplied by per-backend extensions:

- `CensoredDistributionsChainRulesCoreExt` defines the reverse-mode
  `rrule` and forward-mode `frule` (analytical partials, primal via
  `gamma_inc`).
- `CensoredDistributionsMooncakeExt` lifts both the rrule and frule into
  Mooncake (reverse and forward mode).
- `CensoredDistributionsReverseDiffExt` lifts the rrule into ReverseDiff.
- `CensoredDistributionsForwardDiffExt` defines `Dual` methods on
  `_gamma_cdf` directly (forward-mode dispatches on argument types,
  not via ChainRules).

The α-partial that `gamma_inc`'s `ChainRule` leaves as
`@not_implemented` is supplied by [`_grad_p_a_series`](@ref), following
the series form Moore (1982) introduced as Algorithm AS 187 and that
Stan (`grad_reg_inc_gamma`) and JAX (`igamma_grad_a`) both use.
"""
function _gamma_cdf(k::Real, θ::Real, x::Real)
    x <= 0 && return zero(k) * zero(θ) * zero(x)
    kp, zp = promote(k, x / θ)
    return first(gamma_inc(kp, zp))
end

@doc """
Primal value and analytical partials `(Ω, dk, dθ, dx)` for
[`_gamma_cdf`](@ref). Shared by every per-backend AD extension so the
formulas live in one place:

- `dx = pdf(Gamma(k, θ), x)`
- `dθ = -(x/θ) · dx`
- `dk = _grad_p_a_series(k, x/θ)`

The non-positive-`x` branch returns zeros for the primal and all three
partials, matching `_gamma_cdf`'s early-return behaviour.
"""
function _gamma_cdf_value_and_partials(k::Real, θ::Real, x::Real)
    if x <= 0
        T = float(promote_type(typeof(k), typeof(θ), typeof(x)))
        z = zero(T)
        return (z, z, z, z)
    end
    z = x / θ
    Ω = first(gamma_inc(promote(k, z)...))
    f = pdf(Gamma(k, θ), x)
    dk = _grad_p_a_series(k, z)
    dθ = -(x / θ) * f
    dx = f
    return (Ω, dk, dθ, dx)
end

@doc """
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

@doc """
AD-safe `cdf(dist, u)` companion to [`_logcdf_ad_safe`](@ref). Same
dispatch idea: route `Gamma` through [`_gamma_cdf`](@ref) so the
numeric-path optimisations that call `cdf(dist, lower)` for early
termination remain differentiable under reverse-mode AD.
"""
_cdf_ad_safe(dist::UnivariateDistribution, u::Real) = cdf(dist, u)

function _cdf_ad_safe(dist::Gamma, u::Real)
    return _gamma_cdf(shape(dist), scale(dist), u)
end
