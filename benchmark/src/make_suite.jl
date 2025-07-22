@doc raw"
Creates benchmark suites for CensoredDistributions models using DynamicPPL.TestUtils.AD.run_ad.

# Arguments
- `model`: A DynamicPPL model containing CensoredDistributions
- `adbackends`: Vector of AD backends to benchmark (default: [:forwarddiff, :reversediff])

# Returns
A BenchmarkGroup containing benchmarks for each specified AD backend.

# Examples
```julia
@model function censored_model(y)
    μ ~ Normal(1, 0.5)
    σ ~ truncated(Normal(0.5, 0.2), 0, Inf)
    dist = primarycensored(LogNormal(μ, σ), Uniform(0, 1))
    for i in eachindex(y)
        y[i] ~ dist
    end
end

model = censored_model([1.2, 2.1, 1.8])
suite = make_suite(model)
results = run(suite)
```
"
function make_suite(model; adbackends = [
    "ForwardDiff" => AutoForwardDiff(),
    "ReverseDiff" => AutoReverseDiff(; compile=false),
    "ReverseDiffCompiled" => AutoReverseDiff(; compile=true),
    "Mooncake" => AutoMooncake(; config=nothing)
])
    suite = BenchmarkTools.BenchmarkGroup()
    
    for (backend_name, adtype) in adbackends
        suite[backend_name] = BenchmarkTools.@benchmarkable begin
            DynamicPPL.TestUtils.AD.run_ad($model, $adtype)
        end
    end
    
    return suite
end
