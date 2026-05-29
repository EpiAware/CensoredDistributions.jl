md"""
# [Automatic differentiation backends](@id ad-backends)

CensoredDistributions.jl composes with Julia's automatic differentiation
(AD) ecosystem, so the censored `logpdf` can be used as the likelihood in
gradient-based inference, for example inside a
[Turing.jl](https://turinglang.org) model.
This page reports which backends work, how to configure the ones that
need it, and what each costs on the package's shared scenario set.
Advice on choosing a backend and on debugging comes after the results.

## Backend support

Each backend has its own AD gradient CI run, so a transiently unstable
backend only reds its own badge.
The badges below show the latest run of each on `main`, tested on Julia 1
(the latest stable release).

```@raw html
<table>
<thead><tr>
<th>ForwardDiff</th><th>ReverseDiff (tape)</th><th>Enzyme forward</th>
<th>Enzyme reverse</th><th>Mooncake reverse</th><th>Mooncake forward</th>
</tr></thead>
<tbody><tr>
<td><a href="https://github.com/EpiAware/CensoredDistributions.jl/actions/workflows/ad-forwarddiff.yaml"><img src="https://github.com/EpiAware/CensoredDistributions.jl/actions/workflows/ad-forwarddiff.yaml/badge.svg?branch=main" alt="AD ForwardDiff"></a></td>
<td><a href="https://github.com/EpiAware/CensoredDistributions.jl/actions/workflows/ad-reversediff.yaml"><img src="https://github.com/EpiAware/CensoredDistributions.jl/actions/workflows/ad-reversediff.yaml/badge.svg?branch=main" alt="AD ReverseDiff"></a></td>
<td><a href="https://github.com/EpiAware/CensoredDistributions.jl/actions/workflows/ad-enzyme-forward.yaml"><img src="https://github.com/EpiAware/CensoredDistributions.jl/actions/workflows/ad-enzyme-forward.yaml/badge.svg?branch=main" alt="AD Enzyme forward"></a></td>
<td><a href="https://github.com/EpiAware/CensoredDistributions.jl/actions/workflows/ad-enzyme-reverse.yaml"><img src="https://github.com/EpiAware/CensoredDistributions.jl/actions/workflows/ad-enzyme-reverse.yaml/badge.svg?branch=main" alt="AD Enzyme reverse"></a></td>
<td><a href="https://github.com/EpiAware/CensoredDistributions.jl/actions/workflows/ad-mooncake-reverse.yaml"><img src="https://github.com/EpiAware/CensoredDistributions.jl/actions/workflows/ad-mooncake-reverse.yaml/badge.svg?branch=main" alt="AD Mooncake reverse"></a></td>
<td><a href="https://github.com/EpiAware/CensoredDistributions.jl/actions/workflows/ad-mooncake-forward.yaml"><img src="https://github.com/EpiAware/CensoredDistributions.jl/actions/workflows/ad-mooncake-forward.yaml/badge.svg?branch=main" alt="AD Mooncake forward"></a></td>
</tr></tbody>
</table>
```

A green badge means that backend differentiates the scenarios we test for
it; it does not by itself mean full coverage.
All six backends (ForwardDiff, ReverseDiff (tape), Enzyme forward, Enzyme
reverse, Mooncake reverse, Mooncake forward) cover the whole scenario set.

One caveat applies to Enzyme forward.
It trips an upstream "mixed activity for `jl_new_struct`" check when a
closure both captures a distribution constructor as a `Type` value and
calls a censoring constructor with a keyword argument
([#278](https://github.com/EpiAware/CensoredDistributions.jl/issues/278)).
Package scenarios avoid this by writing each constructor as a literal
rather than capturing it, so all of them differentiate.
If you hit the error, hoist the distribution construction out of the
captured closure (or pass it positionally) and Enzyme forward works.

### Configuring Enzyme

Enzyme needs two caller-side settings to work through the numerical
(quadrature) paths.

```julia
using ADTypes, Enzyme
AutoEnzyme(
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation = Enzyme.Duplicated)
```

`Duplicated` marks the closure over the observation data as active, and
`set_runtime_activity` defers per-value activity decisions to runtime; see
the [Enzyme FAQ](https://enzymead.github.io/Enzyme.jl/stable/faq/) for what
they do.
These are the settings the benchmark below uses.

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

md"""
```@raw html
<details><summary>Show setup code</summary>
```
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
```@raw html
</details>
```
"""

md"""
## Benchmark

`DifferentiationInterfaceTest.benchmark_differentiation` runs every
(backend, scenario) pair. We pass every backend and scenario so broken
combinations show up as gaps rather than being hidden.
Each backend's time and allocations are then divided by the ForwardDiff
value on the same scenario, so ForwardDiff sits at 1.0 by construction;
values below 1.0 are faster (or lighter), above 1.0 slower (or heavier).
"""

md"""
### Summary

Geometric mean of the relative cost across the scenarios each backend can
handle. `Scenarios` reports coverage, since a partial backend averages
only over the scenarios it differentiates.
"""

md"""
```@raw html
<details><summary>Show benchmark code</summary>
```
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
end;

md"""
```@raw html
</details>
```
"""

summary

md"""
### Spread across scenarios

Each box summarises a backend's relative cost across the scenario set, on
a log scale so speed-ups and slow-downs are symmetric around the
ForwardDiff baseline at 1.0.
"""

md"""
```@raw html
<details><summary>Show plotting code</summary>
```
"""

plot_df = @chain rel begin
    stack([:rel_time, :rel_bytes],
        variable_name = :metric, value_name = :value)
    @rsubset :value > 0
    @rtransform :metric = :metric == "rel_time" ?
                          "Relative time" : "Relative allocations"
end

## Order the facets time-then-allocations.
metric_order = sorter(["Relative time", "Relative allocations"])

fig_relative = draw(
    data(plot_df) *
    mapping(
        :backend => "",
        :value => "Cost relative to ForwardDiff",
        col = :metric => metric_order) *
    visual(BoxPlot);
    figure = (size = (900, 400),),
    axis = (yscale = log10, xticklabelrotation = pi / 4),
    facet = (; linkyaxes = :none)
);

md"""
```@raw html
</details>
```
"""

fig_relative

md"""
### Per scenario

The same data with one point per scenario, so individual outliers show
rather than being summarised.
Scenarios on the horizontal axis, relative cost on the vertical axis (log
scale), backends by colour, faceted by metric.
"""

md"""
```@raw html
<details><summary>Show plotting code</summary>
```
"""

fig_scenarios = draw(
    data(plot_df) *
    mapping(
        :scenario => "",
        :value => "Cost relative to ForwardDiff",
        color = :backend => "Backend",
        col = :metric => metric_order) *
    visual(Scatter, markersize = 9);
    figure = (size = (1000, 600),),
    axis = (yscale = log10, xticklabelrotation = pi / 4),
    facet = (; linkyaxes = :none)
);

md"""
```@raw html
</details>
```
"""

fig_scenarios

md"""
The full long-format result is available as `raw_bench` if you want GC
fraction, compile fraction, the `value_and_gradient` rows, or absolute
timings.

## Choosing a backend

The results above reflect a general rule: which backend is fastest depends
on how many parameters you differentiate with respect to.

- Forward mode (ForwardDiff, Enzyme forward, Mooncake forward) costs one
  pass per parameter, so it wins when the parameter count is small.
  Fitting a single censored delay distribution has two or three
  parameters, which is why ForwardDiff leads the table here.
- Reverse mode (ReverseDiff, Enzyme reverse, Mooncake reverse) costs one
  pass per output regardless of the parameter count, so it scales better
  once a censored distribution sits inside a larger Turing model with many
  latent parameters.

For the small-parameter case, ForwardDiff is the simplest fast default.
It works on every scenario and needs no configuration.
When you embed these distributions in a higher-dimensional model, switch
to a reverse-mode backend and pick the fastest one that works on your
model.

## Debugging

ForwardDiff fails with ordinary Julia `MethodError`s that point at the
offending call, so it is the easiest backend to debug; start there when a
gradient misbehaves.
Enzyme and Mooncake report errors at the compiled-IR level, which are
harder to trace.

[DifferentiationInterface](https://github.com/JuliaDiff/DifferentiationInterface.jl)
and
[DifferentiationInterfaceTest](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterfaceTest/stable/)
make this tractable.
DI gives one `gradient` call that swaps backends without touching the
model, so you can compare a suspect backend against the ForwardDiff value
on the same input (which is what the gradient tests do).
DIT runs a single function across several backends at once and flags the
ones that disagree with the reference.
Work bottom-up: differentiate one small piece first (a single `logpdf`,
then a `primary_censored` logpdf), confirm it, and build up to the full
model, so the construct a backend chokes on is easy to isolate.

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

- `test/ad/` holds the gradient tests as tagged `@testitem`s, validated
  against a ForwardDiff reference with
  `DifferentiationInterfaceTest.test_differentiation`. Pass a backend tag
  (e.g. `TAG=enzyme_reverse task test-ad-backend`) to run a single
  backend, as the per-backend CI does.
- `benchmark/src/ad_gradients.jl` runs the same scenarios under
  AirspeedVelocity. A benchmark history timeline is tracked in
  [#224](https://github.com/EpiAware/CensoredDistributions.jl/issues/224).
"""
