using CensoredDistributions
using CensoredDistributions: Sequential, Parallel, Competing, PrimaryCensored
using Distributions
using ForwardDiff
using Enzyme
using Enzyme.EnzymeRules: EnzymeRules
import Enzyme.EnzymeRules: augmented_primal, reverse, forward
using Enzyme: Annotation, MixedDuplicated, Duplicated, BatchDuplicated,
              BatchDuplicatedNoNeed, Const, Active, DuplicatedNoNeed

# ---- generic flatten ----
_flat_leaves(x::Real) = (x,)
_flat_leaves(::Symbol) = ()
_flat_leaves(::Integer) = ()
_flat_leaves(d::PrimaryCensored) = _flat_leaves(d.dist)
function _flat_leaves(d::Union{Sequential, Parallel})
    mapreduce(_flat_leaves, (a, b) -> (a..., b...), d.components; init = ())
end
function _flat_leaves(c::Competing)
    dl = mapreduce(_flat_leaves, (a, b) -> (a..., b...), c.delays; init = ())
    bp = mapreduce(_flat_leaves, (a, b) -> (a..., b...), c.branch_probs; init = ())
    return (dl..., bp...)
end
function _flat_leaves(d::Distribution)
    mapreduce(_flat_leaves, (a, b) -> (a..., b...), params(d); init = ())
end
_flat_leaves(t::Tuple) = mapreduce(_flat_leaves, (a, b) -> (a..., b...), t; init = ())

_take(v, i) = (i[] += 1; v[i[]])
_base_ctor(::Type{T}) where {T} = Base.typename(T).wrapper
_rebuild(x::Real, v, i) = _take(v, i)
_rebuild(x::Symbol, v, i) = x
_rebuild(x::Integer, v, i) = x
function _rebuild(d::PrimaryCensored, v, i)
    PrimaryCensored(_rebuild(d.dist, v, i), d.primary_event, d.method)
end
function _rebuild(d::Sequential, v, i)
    Sequential(map(c -> _rebuild(c, v, i), d.components), d.names)
end
_rebuild(d::Parallel, v, i) = Parallel(map(c -> _rebuild(c, v, i), d.components), d.names)
function _rebuild(c::Competing, v, i)
    Competing(c.names, map(d -> _rebuild(d, v, i), c.delays),
        map(p -> _rebuild(p, v, i), c.branch_probs))
end
function _rebuild(d::D, v, i) where {D <: Distribution}
    ps = map(p -> _rebuild(p, v, i), params(d))
    return _base_ctor(D)(ps...)
end

# scatter gradient values into a fresh shadow struct of same type
_grad_struct(x::Real, g, i) = _take(g, i)
_grad_struct(x::Symbol, g, i) = x
_grad_struct(x::Integer, g, i) = x
function _grad_struct(d::PrimaryCensored, g, i)
    PrimaryCensored(_grad_struct(d.dist, g, i), d.primary_event, d.method)
end
function _grad_struct(d::Sequential, g, i)
    Sequential(map(c -> _grad_struct(c, g, i), d.components), d.names)
end
function _grad_struct(d::Parallel, g, i)
    Parallel(map(c -> _grad_struct(c, g, i), d.components), d.names)
end
function _grad_struct(c::Competing, g, i)
    Competing(c.names, map(d -> _grad_struct(d, g, i), c.delays),
        map(p -> _grad_struct(p, g, i), c.branch_probs))
end
function _grad_struct(d::D, g, i) where {D <: Distribution}
    ps = map(p -> _grad_struct(p, g, i), params(d))
    return _base_ctor(D)(ps...; check_args = false)
end

const _CompOrPar = Union{Sequential, Parallel}

function _logpdf_param_grad(d, events)
    leaves = collect(Float64, _flat_leaves(d))
    fwd(v) = (i = Ref(0); logpdf(_rebuild(d, v, i), events))
    return ForwardDiff.gradient(fwd, leaves)
end

# Directional derivative (JVP): grad . seed, where seed are the tangent leaves
# read out of the shadow struct.
function _logpdf_jvp(d, dseed, events)
    g = _logpdf_param_grad(d, events)
    return sum(g .* dseed)
end

# ===== REVERSE =====
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

# ===== FORWARD =====
# JVP for one tangent shadow `sh`: dot the param gradient with the shadow seed.
function _jvp_one(g, sh)::Float64
    dseed = collect(Float64, _flat_leaves(sh))
    return sum(g .* dseed)
end
function EnzymeRules.forward(
        config, func::Const{typeof(logpdf)}, ::Type{RT},
        d::Annotation{<:_CompOrPar}, events::Annotation) where {RT}
    primal = func.val(d.val, events.val)
    if d isa Const
        # no active input: tangent is zero
        if RT <: Const
            return primal
        elseif RT <: DuplicatedNoNeed
            return zero(primal)
        elseif RT <: BatchDuplicatedNoNeed
            n = EnzymeRules.width(config)
            return ntuple(_ -> zero(primal), n)
        elseif RT <: BatchDuplicated
            n = EnzymeRules.width(config)
            return BatchDuplicated(primal, ntuple(_ -> zero(primal), n))
        else
            return Duplicated(primal, zero(primal))
        end
    end
    g = _logpdf_param_grad(d.val, events.val)
    if d isa BatchDuplicated || d isa BatchDuplicatedNoNeed
        dprimals = ntuple(k -> _jvp_one(g, d.dval[k])::Float64,
            EnzymeRules.width(config))
        if RT <: BatchDuplicatedNoNeed
            return dprimals
        else
            return BatchDuplicated(primal, dprimals)
        end
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

# =========== tests ===========
ev_tree = Vector{Union{Missing, Float64}}([0.0, 4.0, 12.0, 11.0, 9.0])
function mk_tree(θ)
    Parallel(
        Sequential(
            primary_censored(LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0)),
            Parallel(
                primary_censored(Gamma(θ[3], θ[4]), Uniform(0.0, 1.0)),
                primary_censored(Gamma(θ[5], θ[6]), Uniform(0.0, 1.0)))),
        primary_censored(LogNormal(θ[7], θ[8]), Uniform(0.0, 1.0)))
end

ev_comp = Vector{Union{Missing, Float64}}([0.0, 4.0, 12.0, missing, 9.0])
function mk_comp(θ)
    Parallel(
        Sequential(
            primary_censored(LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0)),
            Competing(
                :death => (Gamma(θ[3], θ[4]), 0.3),
                :discharge => (Gamma(θ[5], θ[6]), 0.7))),
        primary_censored(LogNormal(θ[7], θ[8]), Uniform(0.0, 1.0)))
end

function check(name, mk, ev, θ)
    f(θ) = logpdf(mk(θ), ev)
    ref = ForwardDiff.gradient(f, θ)
    gr = Enzyme.gradient(set_runtime_activity(Reverse), f, θ)[1]
    gf = Enzyme.gradient(set_runtime_activity(Forward), f, θ)[1]
    println("=== ", name, " ===")
    println("  rev match = ", isapprox(gr, ref; rtol = 1e-5),
        "  fwd match = ", isapprox(collect(gf), ref; rtol = 1e-5))
    if !isapprox(gr, ref; rtol = 1e-5)
        ;
        println("  rev=", gr);
        println("  ref=", ref);
    end
    if !isapprox(collect(gf), ref; rtol = 1e-5)
        ;
        println("  fwd=", gf);
    end
end

θt = [1.4, 0.4, 2.0, 1.0, 2.0, 1.2, 1.9, 0.5]
θc = [1.4, 0.4, 2.0, 3.0, 2.0, 1.0, 1.9, 0.5]
check("nested tree", mk_tree, ev_tree, θt)
check("nested Competing", mk_comp, ev_comp, θc)
