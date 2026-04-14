md"""
# Automatic differentiation backends

CensoredDistributions.jl composes with Julia's automatic differentiation
(AD) ecosystem so that the censored `logpdf` can be used as the likelihood
in gradient-based inference, for example inside a
[Turing.jl](https://turinglang.org) model.
This tutorial shows how to compute gradients of a censored log-likelihood
via [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl)
and compares the wall-clock cost of each backend on a shared scenario set
defined with
[DifferentiationInterfaceTest.jl](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterfaceTest/stable/).
The same scenario list powers the gradient tests in `test/ad/runtests.jl`
and the benchmark suite in `benchmark/src/ad_gradients.jl`, so the numbers
reported here stay comparable with what CI tracks over time.
"""

md"""
## Packages used
"""

using CensoredDistributions
using Distributions
using ADTypes
using DifferentiationInterface
import DifferentiationInterfaceTest as DIT
using ForwardDiff
using ReverseDiff
using Enzyme
using Mooncake
using BenchmarkTools
using Printf
using Markdown

md"""
## A single scenario by hand

Before jumping to the shared scenario matrix, it is helpful to see the
basic pattern on one loss. A loss takes a parameter vector `θ` and
returns the scalar log-likelihood of the censored distribution on some
observed data:
"""

obs = [0.5, 1.2, 2.5, 3.8, 5.1]

function loss(θ)
    d = primary_censored(LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0))
    return sum(x -> logpdf(d, x), obs)
end

θ₀ = [1.0, 0.75]
loss(θ₀)

md"""
`DifferentiationInterface` provides a uniform API across AD packages.
Pick a backend from [ADTypes.jl](https://github.com/SciML/ADTypes.jl),
then call `gradient` the same way regardless of implementation:
"""

DifferentiationInterface.gradient(loss, AutoForwardDiff(), θ₀)

md"""
Swap the backend without touching the loss:
"""

DifferentiationInterface.gradient(loss, AutoReverseDiff(), θ₀)

md"""
## Shared scenarios

The test suite, the benchmark suite and this tutorial all pull their
scenarios from `test/ad/scenarios.jl`. Each scenario wraps a loss in a
`DIT.Scenario{:gradient, :out}` object that pairs the function with an
input vector. Including the file exposes `ad_backends` and
`ad_scenarios` at top level:
"""

include(joinpath(@__DIR__, "..", "..", "..", "..", "test", "ad", "scenarios.jl"))

scenarios = ad_scenarios()

md"""
## Timing gradients with a forward-pass baseline

For each scenario we record two numbers: the cost of the forward
`logpdf` evaluation itself (a lower bound any AD backend must beat) and
the cost of each backend's gradient call. Combinations whose gradient is
non-finite or throws are reported as `N/A`.
"""

function bench_one(f, backend, x)
    try
        g = DifferentiationInterface.gradient(f, backend, x)
        (g isa AbstractVector && all(isfinite, g)) || return missing
        return @belapsed DifferentiationInterface.gradient($f, $backend, $x) samples=5 evals=1
    catch
        return missing
    end
end

function bench_baseline(f, x)
    return @belapsed $f($x) samples=5 evals=1
end

results = map(scenarios) do scen
    row = Any[scen.name, bench_baseline(scen.f, scen.x)]
    for entry in ad_backends()
        push!(row, bench_one(scen.f, entry.backend, scen.x))
    end
    row
end

md"""
## Results
"""

io = IOBuffer()
header = ["Scenario", "logpdf (no AD)"]
for entry in ad_backends()
    push!(header, entry.name)
end
println(io, "| ", join(header, " | "), " |")
println(io, "|", repeat("---|", length(header)))
for row in results
    fields = Any[row[1]]
    for v in row[2:end]
        push!(fields, ismissing(v) ? "N/A" : @sprintf("%.1f μs", 1e6 * v))
    end
    println(io, "| ", join(fields, " | "), " |")
end

Markdown.parse(String(take!(io)))

md"""
Timings are reported in microseconds. The `logpdf (no AD)` column is the
forward-pass cost and acts as a lower bound any backend must beat. A
value of `N/A` means the backend either threw or returned a non-finite
gradient. Gamma scenarios still fail under several backends because of
the `_gamma_inc` `Dual` dispatch gap tracked in
[#217](https://github.com/EpiAware/CensoredDistributions.jl/issues/217).
"""

md"""
## When to use which backend

- **ForwardDiff** — default for small parameter dimensions and the most
  broadly compatible backend in this package.
- **ReverseDiff (tape) / (compiled)** — scales better as the parameter
  count grows; the compiled tape amortises its setup over many calls.
- **Enzyme forward / reverse** — high-performance AD via LLVM; forward
  mode suits very small dimensions, reverse mode the larger ones.
- **Mooncake** — newer reverse-mode AD; recommended where it works.

The full matrix of supported (scenario, backend) pairs is checked in
`test/ad/runtests.jl`.
"""

md"""
## See also

- `test/ad/runtests.jl` validates each backend's gradient against a
  central finite-difference reference using
  `DifferentiationInterfaceTest.test_differentiation`.
- `benchmark/src/ad_gradients.jl` runs the same scenarios under
  AirspeedVelocity. A benchmark history timeline is being tracked in a
  separate issue.
"""
