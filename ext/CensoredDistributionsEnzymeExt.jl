module CensoredDistributionsEnzymeExt

using CensoredDistributions: _gamma_cdf, _gamma_cdf_value_and_partials,
                             _window_quantile, _subevent_slice
using Distributions: UnivariateDistribution
using Enzyme: Enzyme
using Enzyme.EnzymeRules: EnzymeRules
using SpecialFunctions: gamma, digamma

# `_subevent_slice(events, o_idx, ev_idx, n)` gathers a nested tree node's
# `[origin, leaf_events...]` sub-view from the CONSTANT event vector by pure
# index bookkeeping (`src/composers/censored_specialisations.jl`). Its output is
# a freshly-allocated `Vector{eltype(events)}` of event VALUES (data); the gradient
# flows only through the leaf distribution PARAMS at each `_tree_step`, never
# through these copied event times. On a multi-edge tree the event vector has a
# non-bits `Union{Missing, Float64}` element type, and Enzyme's reverse type
# analysis cannot statically prove the layout of that `Array` allocation inside the
# differentiated recursion (`EnzymeNoTypeError` at the `Array` ctor, #319). Marking
# the gather inactive runs it on the primal unchanged and treats the returned slice
# as `Const`, so Enzyme never type-analyses the union-array allocation while the
# leaf-param gradients still flow. This is the Enzyme analogue of the Mooncake
# `@zero_adjoint` shields on the event-name helpers. Correct because the slice
# depends only on inactive inputs (the Const event vector and `Int` indices).
EnzymeRules.inactive(::typeof(_subevent_slice), args...) = nothing

# `_window_quantile(comp, p)` returns a quadrature-window endpoint — the
# *location* at which to clamp an infinite integration limit
# (`_finite_window` in `src/distributions/Convolved.jl`). It is a
# non-differentiable hyperparameter of the quadrature (like the node
# count), so both Enzyme rules return a constant `Float64` with no
# tangent / no cotangent. Crucially this stops Enzyme tracing INTO the
# function at all, so it never reaches `quantile(::Gamma)` →
# `SpecialFunctions.gamma_inc_inv_qsmall`, which it cannot differentiate
# (issue #314: `IllegalTypeAnalysisException`). Other backends get the
# same treatment via the ChainRules `@non_differentiable _primal` mark
# and the ForwardDiff/ReverseDiff primal-stripping methods.
function EnzymeRules.forward(
        ::EnzymeRules.FwdConfig,
        ::Enzyme.Const{typeof(_window_quantile)},
        ::Type{RT}, comp::Enzyme.Annotation{<:UnivariateDistribution},
        p::Enzyme.Annotation) where {RT <: Enzyme.Annotation}
    primal = _window_quantile(comp.val, p.val)
    # The window endpoint carries no derivative, so every requested shadow is
    # zero. Handle the scalar (`Duplicated`) and batched (`BatchDuplicated`,
    # width > 1) return activities: a batched forward call (as
    # DifferentiationInterface issues) needs the shadow returned as an
    # `NTuple{W}` of zeros, else Enzyme rejects the rule's return type.
    if RT <: Enzyme.BatchDuplicatedNoNeed
        W = Enzyme.batch_size(RT)
        return ntuple(_ -> zero(primal), W)
    elseif RT <: Enzyme.BatchDuplicated
        W = Enzyme.batch_size(RT)
        return Enzyme.BatchDuplicated(primal, ntuple(_ -> zero(primal), W))
    elseif RT <: Enzyme.DuplicatedNoNeed
        return zero(primal)
    elseif RT <: Enzyme.Duplicated
        return Enzyme.Duplicated(primal, zero(primal))
    else
        return primal
    end
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        ::Enzyme.Const{typeof(_window_quantile)},
        ::Type{<:Enzyme.Annotation},
        comp::Enzyme.Annotation{<:UnivariateDistribution},
        p::Enzyme.Annotation)
    primal = EnzymeRules.needs_primal(config) ?
             _window_quantile(comp.val, p.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

function EnzymeRules.reverse(
        ::EnzymeRules.RevConfig,
        ::Enzyme.Const{typeof(_window_quantile)},
        ::Type{<:Enzyme.Annotation}, ::Nothing,
        comp::Enzyme.Annotation{<:UnivariateDistribution},
        p::Enzyme.Annotation)
    # No cotangent flows back: the window endpoint is non-differentiable.
    return (nothing, nothing)
end

# NOTE: when the window endpoint feeds directly into active arithmetic (a
# `Difference` whose differentiated subtrahend is unbounded above, so the upper
# bound is a quantile of that component), Enzyme REVERSE treats the call as an
# `Active` scalar return and passes the cotangent as an `Active` instance in the
# `dret` slot, which the rule above does not match. A bespoke `Active`-return
# reverse rule needs to emit a structured zero cotangent for the distribution
# argument, which Enzyme's return-type check rejects when written naively; this
# remains an unresolved Enzyme-reverse limitation for that scenario (registered
# broken in `test/ADFixtures`). Enzyme FORWARD and the other backends are
# correct. The forward rule above already returns zero (scalar and batched)
# shadows, so forward is unaffected.

# `EnzymeRules.@easy_rule` expands into both the reverse-mode
# (`augmented_primal` / `reverse`) and forward-mode (`forward`) rules
# for `_gamma_cdf`. The analytical (dk, dθ, dx) come from
# `_gamma_cdf_value_and_partials` in `src/utils/gamma_ad.jl`, the
# single source-of-truth helper shared with the ChainRules rrule and
# the ForwardDiff Dual path. Routing `_gamma_cdf` through this rule
# avoids Enzyme differentiating `SpecialFunctions.gamma_inc` directly
# (the previous `@import_rrule` lift returned a
# `k`-partial that was ~8% off).

EnzymeRules.@easy_rule(_gamma_cdf(k::Real, θ::Real, x::Real),
    @setup(_vp=_gamma_cdf_value_and_partials(k, θ, x),
        dk=_vp[2],
        dθ=_vp[3],
        dx=_vp[4],),
    (dk, dθ, dx))

# Rule for `SpecialFunctions.gamma`, derivative `d/dx Γ(x) = Γ(x) ψ(x)`
# (`Ω` binds to the primal `Γ(x)`; same formula as the ChainRules
# `gamma` frule/rrule that Mooncake/ReverseDiff pick up). Enzyme's own
# `EnzymeSpecialFunctionsExt` ships no `gamma` rule and instead
# mis-lowers `gamma(x)` to the `loggamma` known-op, returning `ψ(x)` —
# wrong by a factor of `Γ(x)` in both modes (upstream bug). The
# analytical Gamma and Weibull `primarycensored_cdf` paths call
# `gamma(k + 1)` / `gamma(1 + 1/k)` outside the `_gamma_cdf` rule, so
# without this the pipeline shape-partial is wrong.
EnzymeRules.@easy_rule(gamma(x::Real), (Ω * digamma(x),))

end
