# Probe what Enzyme passes to a custom reverse/forward rule for
# _nested_tree_logpdf, to learn the shadow structure of the composer.
using CensoredDistributions
using CensoredDistributions: _nested_tree_logpdf, Sequential, Parallel
using Distributions
using Enzyme
using Enzyme.EnzymeRules: EnzymeRules
import Enzyme.EnzymeRules: forward, augmented_primal, reverse
using Enzyme: Const, Duplicated, Active, Annotation

const NTL = typeof(_nested_tree_logpdf)

function EnzymeRules.augmented_primal(
        config, func::Const{NTL}, ::Type{<:Annotation},
        d::Annotation, events::Annotation, primary::Annotation, T::Annotation)
    println("AUG: d annotation = ", typeof(d))
    println("AUG: d.val type   = ", typeof(d.val))
    if d isa Duplicated
        println("AUG: d.dval type  = ", typeof(d.dval))
        @show d.dval
    end
    println("AUG: events ann   = ", typeof(events))
    println("AUG: primary ann  = ", typeof(primary))
    println("AUG: T ann        = ", typeof(T))
    primal = func.val(d.val, events.val, primary.val, T.val)
    println("AUG: primal = ", primal)
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

function EnzymeRules.reverse(
        config, func::Const{NTL}, dret, tape,
        d::Annotation, events::Annotation, primary::Annotation, T::Annotation)
    println("REV: dret = ", dret)
    return (nothing, nothing, nothing, nothing)
end

# Build the nested tree (same as the scenario)
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
println("primal logpdf = ", f(θ))

g = Enzyme.gradient(set_runtime_activity(Reverse), f, θ)
println("Enzyme grad = ", g)
