using CensoredDistributions
using CensoredDistributions: Sequential, Parallel, Competing
using Distributions, ForwardDiff, Enzyme
using Enzyme: Const, Duplicated, MixedDuplicated, autodiff, Reverse, Forward,
              set_runtime_activity, make_zero

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
f(θ) = logpdf(mk(θ), ev)   # ev captured in closure (NOT a Const arg)
ref = ForwardDiff.gradient(f, θ)
println("ref = ", ref)
for (nm,
    m) in (("captured-closure rev", set_runtime_activity(Reverse)),
    ("captured-closure fwd", set_runtime_activity(Forward)))
    print(nm, ": ")
    try
        g = collect(Enzyme.gradient(m, f, θ)[1])
        println(isapprox(g, ref; rtol = 1e-5) ? "OK" : "WRONG")
    catch e
        ;
        println("ERR ", first(split(sprint(showerror, e), '\n')));
    end
end
