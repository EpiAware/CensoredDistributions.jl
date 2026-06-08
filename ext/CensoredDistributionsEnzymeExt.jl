module CensoredDistributionsEnzymeExt

using CensoredDistributions: _gamma_cdf, _gamma_cdf_value_and_partials,
                             Sequential, Parallel, Competing, PrimaryCensored
using Distributions: Distribution, logpdf, params
using ForwardDiff: ForwardDiff
using Enzyme: Enzyme
using Enzyme.EnzymeRules: EnzymeRules
using Enzyme: Annotation, Const, Active, Duplicated, DuplicatedNoNeed,
              MixedDuplicated, BatchDuplicated, BatchDuplicatedNoNeed
using SpecialFunctions: gamma, digamma

# `EnzymeRules.@easy_rule` expands into both the reverse-mode
# (`augmented_primal` / `reverse`) and forward-mode (`forward`) rules
# for `_gamma_cdf`. The analytical (dk, dθ, dx) come from
# `_gamma_cdf_value_and_partials` in `src/utils/gamma_ad.jl`, the
# single source-of-truth helper shared with the ChainRules rrule and
# the ForwardDiff Dual path. Routing `_gamma_cdf` through this rule
# avoids Enzyme differentiating `SpecialFunctions.gamma_inc` directly
# (the cause of #259 — the previous `@import_rrule` lift returned a
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
# without this the pipeline shape-partial is wrong (#263).
EnzymeRules.@easy_rule(gamma(x::Real), (Ω * digamma(x),))

# ---------------------------------------------------------------------------
# SPIKE (#319): custom EnzymeRules for the nested-composer event-vector logpdf
# ---------------------------------------------------------------------------
#
# `logpdf(d::Union{Sequential,Parallel}, events)` recurses through a
# heterogeneous tuple of differing censored edge types (the nested tree /
# nested-Competing walk). Enzyme cannot type-analyse that recursion
# (`EnzymeNoTypeError` reverse, wrong gradient forward, #319). This pair of
# custom rules intercepts the call BEFORE Enzyme tries to build the inner
# shadow chain and delegates the parameter gradient to ForwardDiff (which
# already differentiates the recursion correctly), then maps the result onto
# Enzyme's structural shadow. The recursion and its type-stability are left
# untouched -- this is additive, design-preserving.
#
# Parameter <-> flat-vector mapping. The composer/distribution tree is walked
# in a fixed pre-order so `_flat_leaves`, `_rebuild` and `_grad_struct` agree
# on leaf order. Only `Real` parameter leaves are active; `Symbol` names,
# `Integer`s, the primary-event window and the solver method are inactive.
_flat_leaves(x::Real) = (x,)
_flat_leaves(::Symbol) = ()
_flat_leaves(::Integer) = ()
_flat_leaves(d::PrimaryCensored) = _flat_leaves(d.dist)
function _flat_leaves(d::Union{Sequential, Parallel})
    return mapreduce(_flat_leaves, (a, b) -> (a..., b...), d.components;
        init = ())
end
function _flat_leaves(c::Competing)
    dl = mapreduce(_flat_leaves, (a, b) -> (a..., b...), c.delays; init = ())
    bp = mapreduce(_flat_leaves, (a, b) -> (a..., b...), c.branch_probs;
        init = ())
    return (dl..., bp...)
end
function _flat_leaves(d::Distribution)
    return mapreduce(_flat_leaves, (a, b) -> (a..., b...), params(d); init = ())
end

# Consume the next scalar from `v` under a `Ref` cursor.
_take(v, i) = (i[] += 1; v[i[]])
# The parameterless distribution constructor (e.g. `LogNormal`, not
# `LogNormal{Float64}`) so AD `Dual`s / gradient leaves flow through.
_base_ctor(::Type{T}) where {T} = Base.typename(T).wrapper

# Rebuild a tree with new (AD-typed) scalar leaves consumed in pre-order.
_rebuild(::Real, v, i) = _take(v, i)
_rebuild(x::Symbol, v, i) = x
_rebuild(x::Integer, v, i) = x
function _rebuild(d::PrimaryCensored, v, i)
    return PrimaryCensored(_rebuild(d.dist, v, i), d.primary_event, d.method)
end
function _rebuild(d::Sequential, v, i)
    return Sequential(map(c -> _rebuild(c, v, i), d.components), d.names)
end
function _rebuild(d::Parallel, v, i)
    return Parallel(map(c -> _rebuild(c, v, i), d.components), d.names)
end
function _rebuild(c::Competing, v, i)
    return Competing(c.names, map(x -> _rebuild(x, v, i), c.delays),
        map(p -> _rebuild(p, v, i), c.branch_probs))
end
function _rebuild(d::D, v, i) where {D <: Distribution}
    ps = map(p -> _rebuild(p, v, i), params(d))
    return _base_ctor(D)(ps...)
end

# Scatter a flat gradient back into a fresh shadow struct of the SAME type as
# the primal. The param slots hold gradient values (which violate the
# distribution's domain, e.g. a negative scale), so leaf distributions are
# built with `check_args = false`; the inactive fields reuse the primal's.
_grad_struct(::Real, g, i) = _take(g, i)
_grad_struct(x::Symbol, g, i) = x
_grad_struct(x::Integer, g, i) = x
function _grad_struct(d::PrimaryCensored, g, i)
    return PrimaryCensored(
        _grad_struct(d.dist, g, i), d.primary_event, d.method)
end
function _grad_struct(d::Sequential, g, i)
    return Sequential(map(c -> _grad_struct(c, g, i), d.components), d.names)
end
function _grad_struct(d::Parallel, g, i)
    return Parallel(map(c -> _grad_struct(c, g, i), d.components), d.names)
end
function _grad_struct(c::Competing, g, i)
    return Competing(c.names, map(x -> _grad_struct(x, g, i), c.delays),
        map(p -> _grad_struct(p, g, i), c.branch_probs))
end
function _grad_struct(d::D, g, i) where {D <: Distribution}
    ps = map(p -> _grad_struct(p, g, i), params(d))
    return _base_ctor(D)(ps...; check_args = false)
end

const _CompOrPar = Union{Sequential, Parallel}

# Parameter gradient of `logpdf(d, events)` w.r.t. d's leaf params, via
# ForwardDiff over the flatten/rebuild closure.
function _logpdf_param_grad(d, events)
    leaves = collect(Float64, _flat_leaves(d))
    f(v) = logpdf(_rebuild(d, v, Ref(0)), events)
    return ForwardDiff.gradient(f, leaves)
end

# JVP for one tangent shadow `sh`: dot the param gradient with the shadow seed.
function _jvp_one(g, sh)::Float64
    dseed = collect(Float64, _flat_leaves(sh))
    return sum(g .* dseed)
end

function EnzymeRules.augmented_primal(
        config, func::Const{typeof(logpdf)}, ::Type{<:Annotation},
        d::Annotation{<:_CompOrPar}, events::Annotation)
    primal = EnzymeRules.needs_primal(config) ?
             func.val(d.val, events.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, d.val)
end

function EnzymeRules.reverse(
        config, func::Const{typeof(logpdf)}, dret, dval,
        d::Annotation{<:_CompOrPar}, events::Annotation)
    if d isa MixedDuplicated || d isa Duplicated
        dretv = dret isa Active ? dret.val : dret
        g = _logpdf_param_grad(dval, events.val) .* dretv
        gs = _grad_struct(dval, g, Ref(0))
        if d isa MixedDuplicated
            d.dval[] = gs
        else
            d.dval = gs
        end
    end
    return (nothing, nothing)
end

function EnzymeRules.forward(
        config, func::Const{typeof(logpdf)}, ::Type{RT},
        d::Annotation{<:_CompOrPar}, events::Annotation) where {RT}
    primal = func.val(d.val, events.val)
    if d isa Const
        if RT <: Const
            return primal
        elseif RT <: DuplicatedNoNeed
            return zero(primal)
        elseif RT <: BatchDuplicatedNoNeed
            return ntuple(_ -> zero(primal), EnzymeRules.width(config))
        elseif RT <: BatchDuplicated
            return BatchDuplicated(primal,
                ntuple(_ -> zero(primal), EnzymeRules.width(config)))
        else
            return Duplicated(primal, zero(primal))
        end
    end
    g = _logpdf_param_grad(d.val, events.val)
    if d isa BatchDuplicated || d isa BatchDuplicatedNoNeed
        dprimals = ntuple(k -> _jvp_one(g, d.dval[k]),
            EnzymeRules.width(config))
        return RT <: BatchDuplicatedNoNeed ? dprimals :
               BatchDuplicated(primal, dprimals)
    end
    sh = d isa MixedDuplicated ? d.dval[] : d.dval
    dprimal = _jvp_one(g, sh)
    if RT <: Const
        return primal
    elseif RT <: DuplicatedNoNeed
        return dprimal
    else
        return Duplicated(primal, dprimal)
    end
end

end
