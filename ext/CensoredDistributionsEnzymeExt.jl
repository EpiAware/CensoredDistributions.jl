module CensoredDistributionsEnzymeExt

using CensoredDistributions: _gamma_cdf, _gamma_cdf_value_and_partials
using Enzyme: Enzyme
using Enzyme.EnzymeCore: EnzymeCore, Active, Annotation, BatchDuplicated,
                         BatchDuplicatedNoNeed, Const, Duplicated,
                         DuplicatedNoNeed
using Enzyme.EnzymeRules: EnzymeRules, AugmentedReturn, FwdConfigWidth,
                          RevConfigWidth, needs_primal, needs_shadow

# Direct reverse-mode Enzyme rule for `_gamma_cdf(k, θ, x)`. Previously
# the extension lifted the ChainRules rrule via `Enzyme.@import_rrule`,
# but that path produced a wrong ∂P/∂k partial (~8% off the value the
# other backends compute via the same formula). Writing the rule
# against `EnzymeRules` directly gives correct partials and routes
# `_gamma_cdf` away from Enzyme's attempt to differentiate
# `SpecialFunctions.gamma_inc` (whose recursive series + DomainError
# branches Enzyme cannot lower cleanly).
#
# The analytical partials live in `_gamma_cdf_value_and_partials` (in
# `src/utils/gamma_ad.jl`), shared with the ChainRules rrule and the
# ForwardDiff Dual path. Scope: width-1 reverse-mode rules — the only
# configuration the AD test suite and Turing sampling pipelines hit.
# Batched modes (`width > 1`) fall through to Enzyme's default
# handling.

function EnzymeRules.augmented_primal(
        config::RevConfigWidth{1},
        func::Const{typeof(_gamma_cdf)},
        ::Type{<:Active},
        k::Annotation{<:Real},
        θ::Annotation{<:Real},
        x::Annotation{<:Real})
    Ω, dk, dθ, dx = _gamma_cdf_value_and_partials(k.val, θ.val, x.val)
    primal = needs_primal(config) ? Ω : nothing
    return AugmentedReturn(primal, nothing, (dk, dθ, dx))
end

function EnzymeRules.reverse(
        ::RevConfigWidth{1},
        ::Const{typeof(_gamma_cdf)},
        dret::Active,
        tape::Tuple,
        k::Annotation{<:Real},
        θ::Annotation{<:Real},
        x::Annotation{<:Real})
    dk, dθ, dx = tape
    ȳ = dret.val
    return (
        k isa Const ? nothing : dk * ȳ,
        θ isa Const ? nothing : dθ * ȳ,
        x isa Const ? nothing : dx * ȳ
    )
end

# Forward-mode rule. The shadow in each batch lane is
# `dk · k.dval + dθ · θ.dval + dx · x.dval`, with Const-annotated args
# contributing zero. Supports arbitrary `width(config)` so it covers
# `DifferentiationInterface.gradient(..., AutoEnzyme(mode = Forward))`,
# which batches one dual lane per input parameter. Without this,
# `Enzyme.Forward` falls through to generic differentiation of
# `_gamma_cdf`'s body, which hits `SpecialFunctions.gamma_inc` and
# gives wrong partials (issue #259).
_dvals(::Const, Ω, ::Val{W}) where {W} = ntuple(_ -> zero(Ω), Val(W))
_dvals(d::Union{Duplicated, DuplicatedNoNeed}, _, ::Val{1}) = (d.dval,)
const _BatchDup = Union{BatchDuplicated, BatchDuplicatedNoNeed}
_dvals(d::_BatchDup, _, ::Val{W}) where {W} = d.dval

function EnzymeRules.forward(
        config::FwdConfigWidth{W},
        ::Const{typeof(_gamma_cdf)},
        ::Type{<:Annotation},
        k::Annotation{<:Real},
        θ::Annotation{<:Real},
        x::Annotation{<:Real}) where {W}
    Ω, dk, dθ, dx = _gamma_cdf_value_and_partials(k.val, θ.val, x.val)
    needs_shadow(config) || return needs_primal(config) ? Ω : nothing
    kd = _dvals(k, Ω, Val(W))
    θd = _dvals(θ, Ω, Val(W))
    xd = _dvals(x, Ω, Val(W))
    dΩ = ntuple(w -> dk * kd[w] + dθ * θd[w] + dx * xd[w], Val(W))
    if W == 1
        return needs_primal(config) ? Duplicated(Ω, dΩ[1]) : dΩ[1]
    else
        return needs_primal(config) ? BatchDuplicated(Ω, dΩ) : dΩ
    end
end

end
