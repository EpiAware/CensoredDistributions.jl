using CensoredDistributions
using Distributions, ForwardDiff, Enzyme
using ADTypes, DifferentiationInterface
using DifferentiationInterface: Constant

ev = Vector{Union{Missing, Float64}}([0.0, 4.0, 12.0, 11.0, 9.0])
f = (θ,
    ev) -> logpdf(
    Parallel(
        Sequential(
            primary_censored(LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0)),
            Parallel(
                primary_censored(Gamma(θ[3], θ[4]), Uniform(0.0, 1.0)),
                primary_censored(Gamma(θ[5], θ[6]), Uniform(0.0, 1.0)))),
        primary_censored(LogNormal(θ[7], θ[8]), Uniform(0.0, 1.0))), ev)
θ = [1.4, 0.4, 2.0, 1.0, 2.0, 1.2, 1.9, 0.5]
ref = DifferentiationInterface.gradient(f, AutoForwardDiff(), θ, Constant(ev))
println("ref = ", ref)
for (nm,
    be) in (("rev", AutoEnzyme(mode = set_runtime_activity(Reverse))),
    ("fwd", AutoEnzyme(mode = set_runtime_activity(Forward))))
    print("DI ", nm, ": ")
    try
        g = DifferentiationInterface.gradient(f, be, θ, Constant(ev))
        println(isapprox(g, ref; rtol = 1e-5) ? "OK" : "WRONG $g")
    catch e
        ;
        println("ERR ", first(split(sprint(showerror, e), '\n')));
    end
end
