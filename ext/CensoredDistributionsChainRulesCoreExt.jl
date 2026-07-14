module CensoredDistributionsChainRulesCoreExt

using CensoredDistributions: _gamma_cdf, _gamma_cdf_value_and_partials
using CensoredDistributions: _collect_unique_boundaries
using ChainRulesCore: ChainRulesCore, NoTangent

# `_collect_unique_boundaries(d, x)` returns the batched-pdf boundaries:
# functions of the (constant) lags and interval spec, not the AD parameters,
# so they carry no tangent. `@non_differentiable` covers reverse-mode AD
# (ReverseDiff) without tracing the `unique`/sort internals; the parameter
# gradient flows through the CDF evaluation in `_compute_boundary_cdfs`
# (#699, #701).
ChainRulesCore.@non_differentiable _collect_unique_boundaries(::Any, ::Any)

# Reverse- and forward-mode rules for `_gamma_cdf(k, θ, x) = P(k, x/θ)`.
# The analytical partials live in `_gamma_cdf_value_and_partials` (in
# `src/utils/gamma_ad.jl`) so the ForwardDiff Dual path and the direct
# Enzyme rule share the same formulas. Only the k-partial is non-trivial
# — that's the term SpecialFunctions' own ChainRule leaves as
# `@not_implemented`; see `_grad_p_a_series` for the series form (Moore
# 1982 AS 187, the same construction Stan and JAX use).
function ChainRulesCore.rrule(::typeof(_gamma_cdf), k::Real, θ::Real, x::Real)
    Ω, dk, dθ, dx = _gamma_cdf_value_and_partials(k, θ, x)
    function _gamma_cdf_pullback(ȳ)
        return (NoTangent(), dk * ȳ, dθ * ȳ, dx * ȳ)
    end
    return Ω, _gamma_cdf_pullback
end

# Forward-mode rule, used by Mooncake's forward mode via the
# `@from_chainrules` lift in `CensoredDistributionsMooncakeExt` (ForwardDiff
# dispatches on `Dual` types directly, so it never reaches here). Without
# this, Mooncake's forward lift calls `ChainRulesCore.frule`, which returns
# `nothing` for an undefined rule and trips `iterate(::Nothing)` (#270).
function ChainRulesCore.frule(
        (_, Δk, Δθ, Δx), ::typeof(_gamma_cdf), k::Real, θ::Real, x::Real)
    Ω, dk, dθ, dx = _gamma_cdf_value_and_partials(k, θ, x)
    return Ω, dk * Δk + dθ * Δθ + dx * Δx
end

end
