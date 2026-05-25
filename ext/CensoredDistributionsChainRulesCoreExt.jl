module CensoredDistributionsChainRulesCoreExt

using CensoredDistributions: _gamma_cdf, _grad_p_a_series
using ChainRulesCore: ChainRulesCore, NoTangent
using Distributions: Gamma, pdf
using SpecialFunctions: gamma_inc

# Reverse-mode rule for `_gamma_cdf(k, θ, x) = P(k, x/θ)`.
#
# Partials:
#   ∂P/∂x = pdf(Gamma(k, θ), x)
#   ∂P/∂θ = -(x/θ) · pdf(Gamma(k, θ), x)
#   ∂P/∂k = via `_grad_p_a_series(k, x/θ)` (series form; see
#           `src/utils/gamma_ad.jl` for derivation).
#
# Primal is computed via `SpecialFunctions.gamma_inc` rather than the
# package's own series because here we're given concrete `Real` values
# and the SpecialFunctions implementation is faster. The Julia series
# is still used for the primal when ForwardDiff `Dual`s flow through
# the un-ruled `_gamma_cdf` definition.
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
