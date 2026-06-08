using CensoredDistributions
using CensoredDistributions: Sequential, Parallel
using Distributions
using Enzyme
using Enzyme.EnzymeRules: EnzymeRules
import Enzyme.EnzymeRules: forward, augmented_primal, reverse
using Enzyme: Const, Duplicated, Active, Annotation

# Rule on logpdf(d::Union{Sequential,Parallel}, events)
function EnzymeRules.augmented_primal(
        config, func::Const{typeof(logpdf)}, ::Type{<:Annotation},
        d::Annotation{<:Union{Sequential, Parallel}}, events::Annotation)
    println("AUG REACHED: d ann = ", nameof(typeof(d)))
    primal = func.val(d.val, events.val)
    println("AUG primal = ", primal)
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end
function EnzymeRules.reverse(
        config, func::Const{typeof(logpdf)}, dret, tape,
        d::Annotation{<:Union{Sequential, Parallel}}, events::Annotation)
    println("REV REACHED: dret = ", dret)
    return (nothing, nothing)
end
function EnzymeRules.forward(
        config, func::Const{typeof(logpdf)}, ::Type{RT},
        d::Annotation{<:Union{Sequential, Parallel}}, events::Annotation) where {RT}
    println("FWD REACHED: d ann = ", nameof(typeof(d)))
    return func.val(d.val, events.val)
end

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
println("primal = ", f(θ))
try
    g = Enzyme.gradient(set_runtime_activity(Reverse), f, θ)
    println("REV grad = ", g)
catch e
    println("REV failed: ", first(split(sprint(showerror, e), '\n')))
end

println("\n--- probe shadow type ---")
