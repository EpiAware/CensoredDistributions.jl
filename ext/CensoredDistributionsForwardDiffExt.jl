module CensoredDistributionsForwardDiffExt

import CensoredDistributions: _gamma_cdf
using CensoredDistributions: _grad_p_a_series
using Distributions: Gamma, pdf
using ForwardDiff: ForwardDiff, Dual, value, partials
using SpecialFunctions: gamma_inc

# Forward-mode AD via explicit `Dual` methods on `_gamma_cdf`.
# ForwardDiff dispatches on `Dual` argument types (not via ChainRules),
# so the reverse-mode rule in `CensoredDistributionsChainRulesCoreExt`
# never fires for forward-mode. Each Dual/Real combination of the three
# arguments needs its own method declaration; the shared `_dual_impl`
# helper extracts values, calls `gamma_inc` on the value parts (full
# accuracy across all `z/a` regimes), and reconstructs partials from
# the same analytical formulas the rrule uses
# (`_grad_p_a_series` for ∂P/∂k, `pdf(Gamma(k, θ), x)` for ∂P/∂x,
# `-(x/θ) · pdf(Gamma(k, θ), x)` for ∂P/∂θ).

_val(x) = x isa Dual ? value(x) : x
_par(x, ::Val{N}) where {N} = x isa Dual ? partials(x) : ForwardDiff.Partials(ntuple(_ -> zero(_val(x)), N))

function _dual_impl(::Type{T}, k, θ, x, ::Val{N}) where {T, N}
    kv = _val(k); θv = _val(θ); xv = _val(x)
    if xv <= 0
        return Dual{T}(zero(kv * θv * xv), _par(k, Val(N)) * 0)
    end
    z = xv / θv
    Ω  = first(gamma_inc(promote(kv, z)...))
    f  = pdf(Gamma(kv, θv), xv)
    dk = _grad_p_a_series(kv, z)
    dθ = -(xv / θv) * f
    dx = f
    new_partials = dk * _par(k, Val(N)) + dθ * _par(θ, Val(N)) + dx * _par(x, Val(N))
    return Dual{T}(Ω, new_partials)
end

# All non-trivial Dual subsets of (k, θ, x). Triplet → single Dual paths
# cover every combination ForwardDiff dispatches on at use sites — both
# stand-alone (`gradient(f, AutoForwardDiff(), [k, θ, x])`) and through
# `primarycensored_cdf` (which propagates Duals into selected args only).
function _gamma_cdf(k::Dual{T, V, N}, θ::Dual{T, V, N}, x::Dual{T, V, N}) where {T, V, N}
    return _dual_impl(T, k, θ, x, Val(N))
end
function _gamma_cdf(k::Dual{T, V, N}, θ::Dual{T, V, N}, x::Real) where {T, V, N}
    return _dual_impl(T, k, θ, x, Val(N))
end
function _gamma_cdf(k::Dual{T, V, N}, θ::Real, x::Dual{T, V, N}) where {T, V, N}
    return _dual_impl(T, k, θ, x, Val(N))
end
function _gamma_cdf(k::Real, θ::Dual{T, V, N}, x::Dual{T, V, N}) where {T, V, N}
    return _dual_impl(T, k, θ, x, Val(N))
end
function _gamma_cdf(k::Dual{T, V, N}, θ::Real, x::Real) where {T, V, N}
    return _dual_impl(T, k, θ, x, Val(N))
end
function _gamma_cdf(k::Real, θ::Dual{T, V, N}, x::Real) where {T, V, N}
    return _dual_impl(T, k, θ, x, Val(N))
end
function _gamma_cdf(k::Real, θ::Real, x::Dual{T, V, N}) where {T, V, N}
    return _dual_impl(T, k, θ, x, Val(N))
end

end
