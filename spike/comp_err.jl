using CensoredDistributions
using CensoredDistributions: Sequential, Parallel, Competing
using Distributions, ForwardDiff, Enzyme
ev = Vector{Union{Missing, Float64}}([0.0, 4.0, 12.0, missing, 9.0])
function mk(θ)
    Parallel(
        Sequential(
            primary_censored(LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0)),
            Competing(:death => (Gamma(θ[3], θ[4]), 0.3),
                :discharge => (Gamma(θ[5], θ[6]), 0.7))),
        primary_censored(LogNormal(θ[7], θ[8]), Uniform(0.0, 1.0)))
end
θ = [1.4, 0.4, 2.0, 3.0, 2.0, 1.0, 1.9, 0.5]
f(θ) = logpdf(mk(θ), ev)
println("ref = ", ForwardDiff.gradient(f, θ))
try
    g = Enzyme.gradient(set_runtime_activity(Reverse), f, θ)[1]
    println("rev = ", collect(g))
catch e
    println("REV ERR:");
    showerror(stdout, e);
    println()
end
