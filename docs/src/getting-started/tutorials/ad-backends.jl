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
The same scenario list powers the gradient tests in
`test/ad/ad_gradients.jl` and the AirspeedVelocity benchmark suite in
`benchmark/src/ad_gradients.jl`, so the numbers here track what CI watches
over time.
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
using Zygote
using BenchmarkTools
using Printf
using Markdown

md"""
## Defining a loss

A loss takes a parameter vector `θ` and returns the scalar log-likelihood
of the censored distribution on some observed data:
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
call `gradient` the same way regardless of implementation:
"""

DifferentiationInterface.gradient(loss, AutoForwardDiff(), θ₀)

md"""
Swap the backend without touching the loss:
"""

DifferentiationInterface.gradient(loss, AutoReverseDiff(), θ₀)

md"""
## Scenarios driven by `DifferentiationInterfaceTest`

The test and benchmark suites wrap each loss in a
`DIT.Scenario{:gradient, :out}` object that pairs the function with an
input vector and an optional reference gradient. The same scenario list is
reused here to mirror what CI actually exercises.
"""

obs_smooth = [0.5, 1.2, 2.5, 3.8, 5.1]
obs_int = [0.0, 1.0, 2.0, 3.0, 4.0]

analytical_specs = (
    (name = "PrimaryCensored Gamma+Uniform analytical",
        ctor = Gamma, θ₀ = [2.0, 1.5]),
    (name = "PrimaryCensored LogNormal+Uniform analytical",
        ctor = LogNormal, θ₀ = [1.0, 0.75]),
    (name = "PrimaryCensored Weibull+Uniform analytical",
        ctor = Weibull, θ₀ = [2.0, 1.5])
)

numerical_specs = (
    (name = "PrimaryCensored LogNormal+Uniform numerical",
        ctor = LogNormal, θ₀ = [1.0, 0.75]),
    (name = "PrimaryCensored Weibull+Uniform numerical",
        ctor = Weibull, θ₀ = [2.0, 1.5])
)

scenarios = DIT.Scenario[]

for spec in analytical_specs
    f = let ctor = spec.ctor
        θ -> sum(
            x -> logpdf(primary_censored(ctor(θ[1], θ[2]), Uniform(0.0, 1.0)), x),
            obs_smooth)
    end
    push!(scenarios,
        DIT.Scenario{:gradient, :out}(f, spec.θ₀; name = spec.name))
end

for spec in numerical_specs
    f = let ctor = spec.ctor
        θ -> sum(
            x -> logpdf(
                primary_censored(
                    ctor(θ[1], θ[2]), Uniform(0.0, 1.0);
                    force_numeric = true),
                x),
            obs_smooth)
    end
    push!(scenarios,
        DIT.Scenario{:gradient, :out}(f, spec.θ₀; name = spec.name))
end

ic_loss(θ) = sum(
    x -> logpdf(interval_censored(LogNormal(θ[1], θ[2]), 1.0), x),
    obs_int)
push!(scenarios,
    DIT.Scenario{:gradient, :out}(
        ic_loss, [1.0, 0.75]; name = "IntervalCensored LogNormal regular"))

function dic_loss(θ)
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
push!(scenarios,
    DIT.Scenario{:gradient, :out}(
        dic_loss, [1.0, 0.75]; name = "DoubleIntervalCensored LogNormal"))

backends = [
    ("ForwardDiff", AutoForwardDiff()),
    ("ReverseDiff", AutoReverseDiff()),
    ("Zygote", AutoZygote())
]

md"""
For each (scenario, backend) pair, the benchmark first checks that the
gradient is finite and then records the minimum runtime from a small
sample. Combinations that currently fail — see
[#217](https://github.com/EpiAware/CensoredDistributions.jl/issues/217)
and
[#218](https://github.com/EpiAware/CensoredDistributions.jl/issues/218)
— are reported as `N/A`.
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

results = map(scenarios) do scen
    row = Any[scen.name]
    for (_, backend) in backends
        push!(row, bench_one(scen.f, backend, scen.x))
    end
    row
end

md"""
## Results

Timings are reported in microseconds. A value of `N/A` means the backend
failed to produce a finite gradient for that scenario.
"""

io = IOBuffer()
header = ["Scenario"; [b[1] for b in backends]]
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

`test/ad/ad_gradients.jl` validates the numerical gradient value returned
by each backend against a central finite-difference reference stored in
each `Scenario`'s `res1` field, using
`DifferentiationInterfaceTest.test_differentiation`.
"""
