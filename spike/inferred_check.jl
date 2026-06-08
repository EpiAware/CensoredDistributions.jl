using CensoredDistributions
using CensoredDistributions: Sequential, Parallel, Competing
using Distributions, Test
ev = Vector{Union{Missing, Float64}}([0.0, 4.0, 12.0, 11.0, 9.0])
d = Parallel(
    Sequential(
        primary_censored(LogNormal(1.4, 0.4), Uniform(0.0, 1.0)),
        Parallel(
            primary_censored(Gamma(2.0, 1.0), Uniform(0.0, 1.0)),
            primary_censored(Gamma(2.0, 1.2), Uniform(0.0, 1.0)))),
    primary_censored(LogNormal(1.9, 0.5), Uniform(0.0, 1.0)))
try
    @inferred logpdf(d, ev)
    println("nested tree logpdf @inferred: OK   value=", logpdf(d, ev))
catch e
    println("nested tree logpdf @inferred: FAIL ", first(split(sprint(showerror, e), '\n')))
end
# timing sanity
logpdf(d, ev)
t = @elapsed for _ in 1:10000
    ;
    logpdf(d, ev);
end
println("10k nested-tree logpdf evals: ", round(t*1e3, digits = 2), " ms")
