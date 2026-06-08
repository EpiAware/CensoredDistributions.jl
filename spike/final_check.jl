using CensoredDistributions
using CensoredDistributions: Sequential, Parallel, Competing
using Distributions, ForwardDiff, ReverseDiff, Mooncake, Enzyme
using ADTypes, DifferentiationInterface
const DI = DifferentiationInterface

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
            Competing(:death => (Gamma(θ[3], θ[4]), 0.3),
                :discharge => (Gamma(θ[5], θ[6]), 0.7))),
        primary_censored(LogNormal(θ[7], θ[8]), Uniform(0.0, 1.0)))
end

backends = [
    ("ForwardDiff", AutoForwardDiff()),
    ("ReverseDiff", AutoReverseDiff(compile = false)),
    ("Mooncake rev", AutoMooncake(config = nothing)),
    ("Mooncake fwd", AutoMooncakeForward()),
    ("Enzyme rev", AutoEnzyme(mode = Enzyme.set_runtime_activity(Enzyme.Reverse))),
    ("Enzyme fwd", AutoEnzyme(mode = Enzyme.set_runtime_activity(Enzyme.Forward)))
]
function run(name, mk, ev, θ)
    f(θ) = logpdf(mk(θ), ev)   # CAPTURED-CLOSURE shape
    ref = ForwardDiff.gradient(f, θ)
    println("=== ", name, " (captured-closure) ===")
    for (bn, be) in backends
        print("  ", rpad(bn, 13), ": ")
        try
            g = collect(DI.gradient(f, be, θ))
            println(isapprox(g, ref; rtol = 5e-2, atol = 1e-6) ? "OK" : "WRONG")
        catch e
            ;
            println("ERR ", first(split(sprint(showerror, e), '\n')));
        end
    end
end
run("nested tree", mk_tree, ev_tree, [1.4, 0.4, 2.0, 1.0, 2.0, 1.2, 1.9, 0.5])
run("nested Competing", mk_comp, ev_comp, [1.4, 0.4, 2.0, 3.0, 2.0, 1.0, 1.9, 0.5])
