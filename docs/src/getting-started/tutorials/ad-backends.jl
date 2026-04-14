md"""
# Automatic differentiation backends

CensoredDistributions.jl is designed to compose with Julia's automatic
differentiation (AD) ecosystem so that the censored `logpdf` can be used as
the likelihood in gradient-based inference, for example inside a
[Turing.jl](https://turinglang.org) model.
This tutorial shows how to compute gradients of a censored log-likelihood
via [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl)
and compares the wall-clock cost of each backend on a small shared
benchmark.
The same loss functions power the gradient tests and the
AirspeedVelocity benchmark suite, so the numbers here correspond directly
to what CI tracks over time.
"""

md"""
## Packages used
"""

using CensoredDistributions
using Distributions
using ADTypes
using DifferentiationInterface
using ForwardDiff
using ReverseDiff
using Zygote
using BenchmarkTools
using Printf
using Markdown

md"""
## Defining a loss

The loss function matches the pattern used by the package's test and
benchmark suites: construct a censored distribution from a parameter
vector `θ` and sum `logpdf` over some observed data.
"""

obs = [0.5, 1.2, 2.5, 3.8, 5.1]

function loss(θ)
    d = primary_censored(LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0))
    return sum(x -> logpdf(d, x), obs)
end

θ₀ = [1.0, 0.75]
loss(θ₀)

md"""
## Computing gradients across backends

`DifferentiationInterface` provides a uniform API across AD packages.
Pick a backend from [ADTypes.jl](https://github.com/SciML/ADTypes.jl), then
call `gradient` the same way regardless of implementation.
"""

DifferentiationInterface.gradient(loss, AutoForwardDiff(), θ₀)

md"""
Swap the backend without touching the loss:
"""

DifferentiationInterface.gradient(loss, AutoReverseDiff(), θ₀)

md"""
## Benchmark across backends

The benchmark below exercises the same loss functions used by
`benchmark/src/ad_gradients.jl` and the test suite in
`test/ad/ad_gradients.jl`.
Cases that currently fail on a given backend are reported as `N/A`. The
known-broken combinations are
[#217](https://github.com/EpiAware/CensoredDistributions.jl/issues/217)
(`IntervalCensored` + `Gamma` with ForwardDiff/ReverseDiff) and
[#218](https://github.com/EpiAware/CensoredDistributions.jl/issues/218)
(Zygote fails on all censored `logpdf`s).
"""

obs_smooth = [0.5, 1.2, 2.5, 3.8, 5.1]
obs_int = [0.0, 1.0, 2.0, 3.0, 4.0]

function pc_ana_gamma(θ)
    sum(
        x -> logpdf(primary_censored(Gamma(θ[1], θ[2]), Uniform(0.0, 1.0)), x),
        obs_smooth)
end
function pc_ana_lognormal(θ)
    sum(
        x -> logpdf(primary_censored(LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0)), x),
        obs_smooth)
end
function pc_num_lognormal(θ)
    sum(
        x -> logpdf(
            primary_censored(
                LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0); force_numeric = true),
            x),
        obs_smooth)
end
function ic_lognormal(θ)
    sum(
        x -> logpdf(interval_censored(LogNormal(θ[1], θ[2]), 1.0), x),
        obs_int)
end
function dic_lognormal(θ)
    sum(
        x -> logpdf(
            double_interval_censored(
                LogNormal(θ[1], θ[2]);
                primary_event = Uniform(0.0, 1.0),
                upper = 10.0,
                interval = 1.0),
            x),
        [1.0, 2.0, 3.0, 4.0, 5.0])
end

cases = [
    ("PrimaryCensored Gamma+Uniform analytical", pc_ana_gamma, [2.0, 1.5]),
    ("PrimaryCensored LogNormal+Uniform analytical", pc_ana_lognormal, [1.0, 0.75]),
    ("PrimaryCensored LogNormal+Uniform numerical", pc_num_lognormal, [1.0, 0.75]),
    ("IntervalCensored LogNormal regular", ic_lognormal, [1.0, 0.75]),
    ("DoubleIntervalCensored LogNormal", dic_lognormal, [1.0, 0.75])
]

backends = [
    ("ForwardDiff", AutoForwardDiff()),
    ("ReverseDiff", AutoReverseDiff()),
    ("Zygote", AutoZygote())
]

md"""
For each (case, backend) pair, the benchmark first checks that the gradient
is finite and then records the minimum runtime from a small sample.
"""

function bench_one(f, backend, θ)
    try
        g = DifferentiationInterface.gradient(f, backend, θ)
        (g isa AbstractVector && all(isfinite, g)) || return missing
        return @belapsed DifferentiationInterface.gradient($f, $backend, $θ) samples=5 evals=1
    catch
        return missing
    end
end

results = map(cases) do (name, f, θ)
    row = Any[name]
    for (_, backend) in backends
        push!(row, bench_one(f, backend, θ))
    end
    row
end

md"""
## Results

Timings are reported in microseconds. A value of `N/A` means the backend
failed to produce a finite gradient for that case and is tracked as a
known issue.
"""

io = IOBuffer()
header = ["Case"; [b[1] for b in backends]]
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
## When to use which backend

- **ForwardDiff** is the most broadly compatible backend for this package
  and is what the test suite treats as required. It is a good default for
  problems with up to a few dozen parameters.
- **ReverseDiff** works for most of the censored distributions and scales
  better than ForwardDiff when the number of parameters grows, at the cost
  of a larger per-call overhead.
- **Zygote** is exercised best-effort. All censored `logpdf`s currently
  return non-finite gradients with Zygote; see
  [#218](https://github.com/EpiAware/CensoredDistributions.jl/issues/218)
  and the test suite (`test/ad/ad_gradients.jl`).

The gradient tests (`test/ad/ad_gradients.jl`) validate the numerical value
of the gradient returned by each backend against a central finite
difference, so any breakage caused by refactors will surface there.
"""
