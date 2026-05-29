md"""
# [Automatic differentiation backends](@id ad-backends)

CensoredDistributions.jl composes with Julia's automatic differentiation
(AD) ecosystem, so the censored `logpdf` can be used as the likelihood in
gradient-based inference, for example inside a
[Turing.jl](https://turinglang.org) model.
This page explains how to choose a backend, how to configure the ones that
need it, and reports the cost of each backend on the package's shared
scenario set.

## Forward vs reverse mode

Which backend is fastest depends on how many parameters you differentiate
with respect to.

- Forward mode (ForwardDiff, Enzyme forward, Mooncake forward) costs one
  pass per parameter, so it wins when the parameter count is small.
  Fitting a single censored delay distribution has two or three
  parameters, which sits squarely in this range.
- Reverse mode (ReverseDiff, Enzyme reverse, Mooncake reverse) costs one
  pass per output regardless of the parameter count, so it scales better
  once a censored distribution sits inside a larger Turing model with many
  latent parameters.

For the small-parameter case, ForwardDiff is the simplest fast default.
It works on every scenario and needs no configuration.
When you embed these distributions in a higher-dimensional model, switch
to a reverse-mode backend and pick the fastest one that works on your
model (see the benchmark below).

## Backend support

| ForwardDiff | ReverseDiff (tape) | Enzyme forward | Enzyme reverse | Mooncake reverse | Mooncake forward |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ![](https://img.shields.io/badge/ForwardDiff-full-brightgreen) | ![](https://img.shields.io/badge/ReverseDiff%20tape-full-brightgreen) | ![](https://img.shields.io/badge/Enzyme%20forward-partial-yellow) | ![](https://img.shields.io/badge/Enzyme%20reverse-full-brightgreen) | ![](https://img.shields.io/badge/Mooncake%20reverse-full-brightgreen) | ![](https://img.shields.io/badge/Mooncake%20forward-full-brightgreen) |

ForwardDiff, ReverseDiff (tape), Enzyme reverse, Mooncake reverse, and
Mooncake forward cover the whole scenario set.
Enzyme forward is partial: it differentiates the `ExponentiallyTilted` and
`LogNormal` interval scenarios but trips an upstream mixed-activity check
on the `Uniform`-primary delay constructors
([#225](https://github.com/EpiAware/CensoredDistributions.jl/issues/225)).

### Configuring Enzyme

Enzyme needs two caller-side settings to work through the numerical
(quadrature) paths.

```julia
using ADTypes, Enzyme
AutoEnzyme(
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation = Enzyme.Duplicated)
```

`function_annotation = Duplicated` stops Enzyme flagging the closure over
the observation data as non-readonly, a generic Enzyme requirement for
closures over arrays that also covers Turing likelihoods.
`set_runtime_activity` resolves per-value activity at runtime through the
quadrature and the distribution constructors.
The analytical paths work with just `Duplicated`; the numerical paths also
need `set_runtime_activity`, so passing both is the recommended default.
These are the settings the benchmark below uses.

## Debugging

ForwardDiff fails with ordinary Julia `MethodError`s that point at the
offending call, so it is the easiest backend to debug; start there when a
gradient misbehaves.
Enzyme and Mooncake report errors at the compiled-IR level, which are
harder to trace.
If a reverse-mode gradient looks wrong, compare it against the ForwardDiff
value on the same input — that is exactly what the gradient tests do.

The scenario set covers analytical and numerical paths for Gamma,
LogNormal, and Weibull delays with both `Uniform` and
`ExponentiallyTilted` primary events, plus `IntervalCensored` and
`DoubleIntervalCensored` constructions.
It is defined with
[DifferentiationInterfaceTest.jl](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterfaceTest/stable/)
in the `ADFixtures` path package at `test/ADFixtures`, and shared with the
gradient tests (`test/ad/runtests.jl`) and the benchmark suite
(`benchmark/src/ad_gradients.jl`).
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
using Chairmarks
using ADFixtures
using DataFramesMeta
using Statistics
using CairoMakie
using AlgebraOfGraphics

CairoMakie.activate!(type = "png", px_per_unit = 2)
set_theme!(theme_latexfonts(); fontsize = 14)

scenarios = ADFixtures.scenarios()
all_backends = [entry.backend for entry in ADFixtures.backends()]
backend_name = Dict(entry.backend => entry.name
for entry in ADFixtures.backends())

md"""
## Benchmark

`DifferentiationInterfaceTest.benchmark_differentiation` runs every
(backend, scenario) pair. We pass every backend and scenario so broken
combinations show up as gaps rather than being hidden.
"""

raw_bench = DIT.benchmark_differentiation(
    all_backends, scenarios;
    logging = false,
    benchmark_test = false
)

## `replace` order matters because `DoubleIntervalCensored` contains
## `IntervalCensored`; the longer key is matched first.
function _shorten(name)
    replace(name,
        "DoubleIntervalCensored" => "DIC",
        "IntervalCensored" => "IC",
        "PrimaryCensored" => "PC")
end

bench_long = @chain DataFrame(raw_bench) begin
    @rsubset :operator == ^(:gradient)
    @rtransform begin
        :backend = backend_name[:backend]
        :scenario = _shorten(:scenario.name)
        :time_us = :time * 1e6
        :bytes_kb = :bytes / 1024
    end
    @rsubset isfinite(:time_us) && isfinite(:bytes_kb)
    @select :backend :scenario :time_us :bytes_kb
end;

md"""
### Cost relative to ForwardDiff

Each backend's time and allocations are divided by the ForwardDiff value
on the same scenario, so ForwardDiff sits at 1.0 by construction.
Values below 1.0 are faster (or lighter); above 1.0 are slower (or
heavier).
"""

ref = @chain bench_long begin
    @rsubset :backend == "ForwardDiff"
    @select :scenario :ref_time=:time_us :ref_bytes=:bytes_kb
end

rel = @chain bench_long begin
    leftjoin(ref, on = :scenario)
    @rtransform begin
        :rel_time = :time_us / :ref_time
        :rel_bytes = :bytes_kb / :ref_bytes
    end
end;

md"""
### Summary

Geometric mean of the relative cost across the scenarios each backend can
handle. `Scenarios` reports coverage, since a partial backend averages
only over the scenarios it differentiates.
"""

## Geometric mean over positive values; guards against a zero-allocation
## scenario sending `log` to -Inf.
function geomean(x)
    pos = filter(>(0), x)
    isempty(pos) ? NaN : exp(mean(log.(pos)))
end

n_total = length(unique(bench_long.scenario))

summary = @chain rel begin
    @by :backend begin
        :rel_time = round(geomean(:rel_time); digits = 2)
        :rel_bytes = round(geomean(:rel_bytes); digits = 2)
        :scenarios = "$(length(:scenario))/$(n_total)"
    end
    @orderby :rel_time
    rename(
        :backend => "Backend",
        :rel_time => "Relative time",
        :rel_bytes => "Relative allocations",
        :scenarios => "Scenarios")
end

md"""
### Spread across scenarios

Each box summarises a backend's relative cost across the scenario set, on
a log scale so speed-ups and slow-downs are symmetric around the
ForwardDiff baseline at 1.0.
"""

plot_df = @chain rel begin
    stack([:rel_time, :rel_bytes],
        variable_name = :metric, value_name = :value)
    @rsubset :value > 0
    @rtransform :metric = :metric == "rel_time" ?
                          "Relative time" : "Relative allocations"
end

fig_relative = draw(
    data(plot_df) *
    mapping(
        :backend => "",
        :value => "Cost relative to ForwardDiff",
        col = :metric) *
    visual(BoxPlot);
    figure = (size = (900, 400),),
    axis = (yscale = log10, xticklabelrotation = pi / 4),
    facet = (; linkyaxes = :none)
)

fig_relative

md"""
The full long-format result is available as `raw_bench` if you want GC
fraction, compile fraction, the `value_and_gradient` rows, or absolute
timings.

## Reproducing this page

The numbers above are measured on the docs-build machine, so they reflect
that CPU.
To regenerate locally:

```
task docs
```

or, equivalently:

```
julia --project=docs docs/make.jl
```

## See also

- `test/ad/runtests.jl` validates each backend's gradient against a
  ForwardDiff reference using
  `DifferentiationInterfaceTest.test_differentiation`.
- `benchmark/src/ad_gradients.jl` runs the same scenarios under
  AirspeedVelocity. A benchmark history timeline is tracked in
  [#224](https://github.com/EpiAware/CensoredDistributions.jl/issues/224).
"""
