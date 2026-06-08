using CensoredDistributions
using CensoredDistributions: Sequential, Parallel
using Distributions
using Enzyme
using Enzyme.EnzymeRules: EnzymeRules
import Enzyme.EnzymeRules: augmented_primal, reverse
using Enzyme: Annotation, MixedDuplicated, Duplicated

function EnzymeRules.augmented_primal(
        config, func::Const{typeof(logpdf)}, ::Type{<:Annotation},
        d::Annotation{<:Union{Sequential, Parallel}}, events::Annotation)
    println("AUG: d isa MixedDuplicated = ", d isa MixedDuplicated,
        "  isa Duplicated = ", d isa Duplicated)
    if d isa MixedDuplicated
        println("  d.dval type = ", typeof(d.dval))
        println("  d.dval[] type = ", typeof(d.dval[]))
        sh = d.dval[]
        println("  shadow.components type = ", typeof(sh.components))
        # drill into first leaf
        c1 = sh.components[1]   # Sequential shadow
        println("  c1 type = ", typeof(c1))
    end
    primal = func.val(d.val, events.val)
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end
function EnzymeRules.reverse(config, func::Const{typeof(logpdf)}, dret, tape,
        d::Annotation{<:Union{Sequential, Parallel}}, events::Annotation)
    return (nothing, nothing)
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
Enzyme.gradient(set_runtime_activity(Reverse), f, θ)
