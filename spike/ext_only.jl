using CensoredDistributions
using CensoredDistributions: Sequential, Parallel, Competing
using Distributions, ForwardDiff, Enzyme
# triggers ext load (Enzyme + ForwardDiff present)

ev = Vector{Union{Missing, Float64}}([0.0, 4.0, 12.0, 11.0, 9.0])
function mk(θ)
    Parallel(
        Sequential(
            primary_censored(LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0)),
            Parallel(
                primary_censored(Gamma(θ[3], θ[4]), Uniform(0.0, 1.0)),
                primary_censored(Gamma(θ[5], θ[6]), Uniform(0.0, 1.0)))),
        primary_censored(LogNormal(θ[7], θ[8]), Uniform(0.0, 1.0)))
end
θ = [1.4, 0.4, 2.0, 1.0, 2.0, 1.2, 1.9, 0.5]
f(θ) = logpdf(mk(θ), ev)
ref = ForwardDiff.gradient(f, θ)
println("ref = ", ref)
for (nm,
    m) in (("rev", set_runtime_activity(Reverse)),
    ("fwd", set_runtime_activity(Forward)))
    print("ext ", nm, ": ")
    try
        g = collect(Enzyme.gradient(m, f, θ)[1])
        println(isapprox(g, ref; rtol = 1e-5) ? "OK" : "WRONG $g")
    catch e
        ;
        println("ERR ", first(split(sprint(showerror, e), '\n')));
    end
end
