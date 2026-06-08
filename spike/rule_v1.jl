# Candidate EnzymeRules rule on logpdf(d::Union{Sequential,Parallel}, events)
# that delegates the gradient to ForwardDiff via a generic flatten/reconstruct
# over the composer + Distributions structs, then scatters into Enzyme's
# structural shadow.
using CensoredDistributions
using CensoredDistributions: Sequential, Parallel, Competing, PrimaryCensored
using Distributions
using ForwardDiff
using Enzyme
using Enzyme.EnzymeRules: EnzymeRules
import Enzyme.EnzymeRules: augmented_primal, reverse
using Enzyme: Annotation, MixedDuplicated, Duplicated, Const, Active

# ---- generic flatten of the ACTIVE Real leaves of a composer/dist tree ----
# Walks the SAME field order used to reconstruct. Returns the leaves in order.
_flat_leaves(x::Real) = (x,)
_flat_leaves(::Symbol) = ()
_flat_leaves(::Integer) = ()         # not active
function _flat_leaves(d::PrimaryCensored)
    # only `dist` carries learnable params here; primary_event (Uniform) and
    # method are treated as constant (matches scenario: Uniform fixed bounds).
    return _flat_leaves(d.dist)
end
function _flat_leaves(d::Union{Sequential, Parallel})
    return mapreduce(_flat_leaves, (a, b) -> (a..., b...), d.components; init = ())
end
function _flat_leaves(c::Competing)
    # delays carry params; branch_probs are params too (Real tuple)
    dl = mapreduce(_flat_leaves, (a, b) -> (a..., b...), c.delays; init = ())
    bp = mapreduce(_flat_leaves, (a, b) -> (a..., b...), c.branch_probs; init = ())
    return (dl..., bp...)
end
# Generic distribution: flatten its params
function _flat_leaves(d::Distribution)
    return mapreduce(_flat_leaves, (a, b) -> (a..., b...), params(d); init = ())
end
_flat_leaves(t::Tuple) = mapreduce(_flat_leaves, (a, b) -> (a..., b...), t; init = ())

# ---- reconstruct a tree with new scalar leaves consumed from a Ref{Int} cursor
_take(v, i) = (i[] += 1; v[i[]])
_rebuild(x::Real, v, i) = _take(v, i)
_rebuild(x::Symbol, v, i) = x
_rebuild(x::Integer, v, i) = x
function _rebuild(d::PrimaryCensored, v, i)
    newdist = _rebuild(d.dist, v, i)
    return PrimaryCensored(newdist, d.primary_event, d.method)
end
function _rebuild(d::Sequential, v, i)
    return Sequential(map(c -> _rebuild(c, v, i), d.components), d.names)
end
function _rebuild(d::Parallel, v, i)
    return Parallel(map(c -> _rebuild(c, v, i), d.components), d.names)
end
function _rebuild(c::Competing, v, i)
    nd = map(d -> _rebuild(d, v, i), c.delays)
    np = map(p -> _rebuild(p, v, i), c.branch_probs)
    return Competing(c.names, nd, np)
end
# generic distribution: reconstruct via its BASE type and new params.
# Use the parameterless constructor (e.g. `LogNormal`, not `LogNormal{Float64}`)
# so AD `Dual`s flow through.
_base_ctor(::Type{T}) where {T} = Base.typename(T).wrapper
function _rebuild(d::D, v, i) where {D <: Distribution}
    ps = map(p -> _rebuild(p, v, i), params(d))
    return _base_ctor(D)(ps...)
end

# ---- scatter a flat gradient into Enzyme's structural shadow (lockstep) ----
# shadow leaves are mutable? No—immutable structs. So we BUILD a new shadow
# struct of the same type with grad values, then store into the Ref.
_grad_struct(x::Real, g, i) = _take(g, i)
_grad_struct(x::Symbol, g, i) = x
_grad_struct(x::Integer, g, i) = x
function _grad_struct(d::PrimaryCensored, g, i)
    return PrimaryCensored(_grad_struct(d.dist, g, i), d.primary_event, d.method)
end
function _grad_struct(d::Sequential, g, i)
    return Sequential(map(c -> _grad_struct(c, g, i), d.components), d.names)
end
function _grad_struct(d::Parallel, g, i)
    return Parallel(map(c -> _grad_struct(c, g, i), d.components), d.names)
end
function _grad_struct(c::Competing, g, i)
    nd = map(d -> _grad_struct(d, g, i), c.delays)
    np = map(p -> _grad_struct(p, g, i), c.branch_probs)
    return Competing(c.names, nd, np)
end
function _grad_struct(d::D, g, i) where {D <: Distribution}
    ps = map(p -> _grad_struct(p, g, i), params(d))
    # The shadow holds GRADIENT values in the param slots, which violate the
    # distribution's domain (e.g. a negative \u03c3); build unchecked.
    return _base_ctor(D)(ps...; check_args = false)
end

function _composer_param_grad(d, events, dret)
    leaves = collect(Float64, _flat_leaves(d))
    function fwd(v)
        i = Ref(0)
        d2 = _rebuild(d, v, i)
        return logpdf(d2, events)
    end
    g = ForwardDiff.gradient(fwd, leaves)
    g .*= dret
    return g
end

const _CompOrPar = Union{Sequential, Parallel}

function EnzymeRules.augmented_primal(
        config, func::Const{typeof(logpdf)}, ::Type{<:Annotation},
        d::Annotation{<:_CompOrPar}, events::Annotation)
    primal = EnzymeRules.needs_primal(config) ?
             func.val(d.val, events.val) : nothing
    # tape: keep the primal d.val for reverse
    return EnzymeRules.AugmentedReturn(primal, nothing, d.val)
end

function EnzymeRules.reverse(
        config, func::Const{typeof(logpdf)}, dret, dval,
        d::Annotation{<:_CompOrPar}, events::Annotation)
    if d isa MixedDuplicated || d isa Duplicated
        dretv = dret isa Active ? dret.val : dret
        g = _composer_param_grad(dval, events.val, dretv)
        i = Ref(0)
        gs = _grad_struct(dval, g, i)
        if d isa MixedDuplicated
            d.dval[] = gs
        else
            # Duplicated: accumulate not supported simply; overwrite
            error("unexpected Duplicated")
        end
    end
    return (nothing, nothing)
end

# ---------- test ----------
function mk(θ)
    Parallel(
        Sequential(
            primary_censored(LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0)),
            Parallel(
                primary_censored(Gamma(θ[3], θ[4]), Uniform(0.0, 1.0)),
                primary_censored(Gamma(θ[5], θ[6]), Uniform(0.0, 1.0)))),
        primary_censored(LogNormal(θ[7], θ[8]), Uniform(0.0, 1.0)))
end
ev = Vector{Union{Missing, Float64}}([0.0, 4.0, 12.0, 11.0, 9.0])
θ = [1.4, 0.4, 2.0, 1.0, 2.0, 1.2, 1.9, 0.5]
f(θ) = logpdf(mk(θ), ev)
ref = ForwardDiff.gradient(f, θ)
println("ref       = ", ref)
gE_rev = Enzyme.gradient(set_runtime_activity(Reverse), f, θ)[1]
println("Enzyme rev= ", gE_rev)
println("rev match = ", isapprox(gE_rev, ref; rtol = 1e-5))
