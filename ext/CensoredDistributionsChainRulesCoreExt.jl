module CensoredDistributionsChainRulesCoreExt

using CensoredDistributions: _gamma_cdf, _grad_p_a_series
using ChainRulesCore: ChainRulesCore, NoTangent
using Distributions: Gamma, pdf
using SpecialFunctions: gamma_inc

# Reverse-mode rule for `_gamma_cdf(k, θ, x) = P(k, x/θ)`.
#
# Partials:
#   ∂P/∂x = pdf(Gamma(k, θ), x)                       (standard)
#   ∂P/∂θ = -(x/θ) · pdf(Gamma(k, θ), x)              (chain rule on z = x/θ)
#   ∂P/∂k = via `_grad_p_a_series(k, x/θ)`            (Moore 1982 AS 187,
#                                                       used by Stan and JAX
#                                                       for the same gap)
#
# Primal and the x/θ partials route through `SpecialFunctions` /
# `Distributions` directly — fast and full accuracy. Only the
# k-partial is reimplemented here, because that's the term
# SpecialFunctions' own ChainRule leaves as `@not_implemented`.
# See `_grad_p_a_series` in `src/utils/gamma_ad.jl` for derivation
# and references.
function ChainRulesCore.rrule(::typeof(_gamma_cdf), k::Real, θ::Real, x::Real)
    if x <= 0
        T = float(promote_type(typeof(k), typeof(θ), typeof(x)))
        Ω = zero(T)
        _zero_pullback(_) = (NoTangent(), zero(T), zero(T), zero(T))
        return Ω, _zero_pullback
    end
    z = x / θ
    Ω = first(gamma_inc(k, z))
    f = pdf(Gamma(k, θ), x)
    dk = _grad_p_a_series(k, z)
    dθ = -(x / θ) * f
    dx = f
    function _gamma_cdf_pullback(ȳ)
        return (NoTangent(), dk * ȳ, dθ * ȳ, dx * ȳ)
    end
    return Ω, _gamma_cdf_pullback
end

end
