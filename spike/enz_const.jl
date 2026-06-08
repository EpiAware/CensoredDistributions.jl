using CensoredDistributions
using Distributions, ForwardDiff, Enzyme
using Enzyme: Const, Duplicated, autodiff, Reverse, Forward, set_runtime_activity

ev = Vector{Union{Missing, Float64}}([0.0, 4.0, 12.0, 11.0, 9.0])
function g(θ, ev)
    logpdf(
        Parallel(
            Sequential(
                primary_censored(LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0)),
                Parallel(
                    primary_censored(Gamma(θ[3], θ[4]), Uniform(0.0, 1.0)),
                    primary_censored(Gamma(θ[5], θ[6]), Uniform(0.0, 1.0)))),
            primary_censored(LogNormal(θ[7], θ[8]), Uniform(0.0, 1.0))), ev)
end
θ = [1.4, 0.4, 2.0, 1.0, 2.0, 1.2, 1.9, 0.5]
ref = ForwardDiff.gradient(t -> g(t, ev), θ)
println("ref = ", ref)

# Explicit Enzyme.autodiff reverse with Const(ev), Duplicated(θ)
dθ = zero(θ)
print("autodiff Reverse Const(ev): ")
try
    autodiff(set_runtime_activity(Reverse), Const(g), Duplicated(θ, dθ), Const(ev))
    println(isapprox(dθ, ref; rtol = 1e-5) ? "OK" : "WRONG $dθ")
catch e
    ;
    println("ERR ", first(split(sprint(showerror, e), '\n')));
end
