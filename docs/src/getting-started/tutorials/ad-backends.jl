#src MANAGED by EpiAwarePackageTools.scaffold — do not edit by hand.
#src Generalised from CensoredDistributions.jl's AD-backends page (the org
#src model page). The page body is re-applied on every scaffold_update so it
#src stays kit-current; everything package-specific it reports (scenarios,
#src backends, broken/skip declarations) is read at docs-build time from the
#src package-owned `test/ADFixtures` registry, so declare a broken scenario
#src there, never here. If this page cannot execute for this package, park it
#src via `FORCE_STUB_TUTORIALS` in `docs/docs_config.jl` instead of editing it.

md"""
# [Automatic differentiation backends](@id ad-backends)

CensoredDistributions.jl composes with Julia's automatic differentiation (AD)
ecosystem, so its differentiable quantities can be used in gradient-based
inference, for example inside a [Turing.jl](https://turinglang.org) model.
This page reports which backends work, how to configure the ones that
need it, and what each costs on the package's shared AD scenario set.
Advice on choosing a backend and on debugging comes after the results.

## Backend support

The AD gradient suite runs as one CI workflow with a job per backend, so a
transiently unstable backend only reds its own job.
The badge below is the latest run of that matrix on `main`, tested on
Julia 1 (the latest stable release).

[![AD](https://github.com/EpiAware/CensoredDistributions.jl/actions/workflows/ad.yaml/badge.svg?branch=main)](https://github.com/EpiAware/CensoredDistributions.jl/actions/workflows/ad.yaml)

The table below is each backend's code coverage from the gradient suite
(Codecov flag `ad-<backend>`), reporting which package lines that backend
exercises.

| ForwardDiff | ReverseDiff (tape) | Enzyme forward | Enzyme reverse | Mooncake reverse | Mooncake forward |
|:---:|:---:|:---:|:---:|:---:|:---:|
| [![cov ForwardDiff](https://codecov.io/gh/EpiAware/CensoredDistributions.jl/graph/badge.svg?flag=ad-forwarddiff)](https://app.codecov.io/gh/EpiAware/CensoredDistributions.jl?flags%5B0%5D=ad-forwarddiff) | [![cov ReverseDiff](https://codecov.io/gh/EpiAware/CensoredDistributions.jl/graph/badge.svg?flag=ad-reversediff)](https://app.codecov.io/gh/EpiAware/CensoredDistributions.jl?flags%5B0%5D=ad-reversediff) | [![cov Enzyme forward](https://codecov.io/gh/EpiAware/CensoredDistributions.jl/graph/badge.svg?flag=ad-enzyme-forward)](https://app.codecov.io/gh/EpiAware/CensoredDistributions.jl?flags%5B0%5D=ad-enzyme-forward) | [![cov Enzyme reverse](https://codecov.io/gh/EpiAware/CensoredDistributions.jl/graph/badge.svg?flag=ad-enzyme-reverse)](https://app.codecov.io/gh/EpiAware/CensoredDistributions.jl?flags%5B0%5D=ad-enzyme-reverse) | [![cov Mooncake reverse](https://codecov.io/gh/EpiAware/CensoredDistributions.jl/graph/badge.svg?flag=ad-mooncake-reverse)](https://app.codecov.io/gh/EpiAware/CensoredDistributions.jl?flags%5B0%5D=ad-mooncake-reverse) | [![cov Mooncake forward](https://codecov.io/gh/EpiAware/CensoredDistributions.jl/graph/badge.svg?flag=ad-mooncake-forward)](https://app.codecov.io/gh/EpiAware/CensoredDistributions.jl?flags%5B0%5D=ad-mooncake-forward) |

A green matrix means each backend differentiates the scenarios we test for
it, which does not by itself mean full coverage.
The next table reports that coverage per backend, rendered directly from
the package's AD-fixture registry (the `ADFixtures` path package at
`test/ADFixtures`), the same registry the gradient tests and the benchmark
below consume.
A scenario is declared broken or skipped on a backend through the
registry's optional `broken_scenario_names`, `backend_broken_scenarios`,
and `backend_skip_scenarios` accessors, so what this table shows cannot
drift from what the tests actually mark broken.
"""

md"""
```@raw html
<details><summary>Show table code</summary>
```
"""

using EpiAwarePackageTools
using ADFixtures
import Markdown

support_table = Markdown.parse(ad_backend_support_table(ADFixtures));

md"""
```@raw html
</details>
```
"""

support_table

md"""
### Configuring Enzyme

When the registry enables Enzyme, the standard configuration defers
per-value activity decisions to runtime:

```julia
using ADTypes, Enzyme
AutoEnzyme(mode = Enzyme.set_runtime_activity(Enzyme.Reverse))
```

See the [Enzyme FAQ](https://enzymead.github.io/Enzyme.jl/stable/faq/) for
what `set_runtime_activity` does.
Scenario data is passed as a `Constant` DifferentiationInterface context
rather than captured in a closure, which keeps the differentiated function
free of active fields.
Runtime activity is not free: on paths that do not need it, it can make
Enzyme several times slower, so where the registry applies one Enzyme
configuration to every scenario its benchmark rows are conservative.
Running through DifferentiationInterface, by contrast, adds no measurable
overhead.

The scenario set is package-owned.
It is defined with
[DifferentiationInterfaceTest.jl](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterfaceTest/stable/)
in the `ADFixtures` path package at `test/ADFixtures`, and shared with the
gradient tests (`test/ad/runtests.jl`), so this page, the tests, and the
per-backend CI all exercise the same set.
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
import DifferentiationInterfaceTest as DIT
using DataFramesMeta
using Statistics
using CairoMakie
using AlgebraOfGraphics

CairoMakie.activate!(type = "png", px_per_unit = 2)
set_theme!(theme_latexfonts(); fontsize = 14)

backend_entries = ADFixtures.backends()
scenario_list = ADFixtures.scenarios()

## The registry's optional bookkeeping accessors (see the ADRegistry
## contract): a missing accessor means no broken or skipped scenarios.
function _optional(name, default)
    isdefined(ADFixtures, name) ? getfield(ADFixtures, name)() : default
end
global_broken = Set(String.(_optional(:broken_scenario_names, String[])))
backend_broken = _optional(
    :backend_broken_scenarios, Dict{String, Set{String}}())
backend_skip = _optional(
    :backend_skip_scenarios, Dict{String, Set{String}}());

md"""
```@raw html
</details>
```
"""

md"""
## Benchmark

`DifferentiationInterfaceTest.benchmark_differentiation` runs every
(backend, scenario) pair the registry supports.
Combinations declared broken or skipped in the registry are excluded from
their backend's rows, so they show up as reduced scenario coverage here
and as named entries in the support table above, rather than as timings
of gradients that are wrong or crash.
The figures are the prepared per-call cost.
DifferentiationInterface prepares each backend once, recording a tape for
ReverseDiff and compiling a rule for Enzyme and Mooncake, and we time the
reused operator, so that one-off preparation is excluded.
This matches repeated use such as an MCMC run, where preparation is
amortised over many gradient calls.
Each backend's time and allocations are then divided by the ForwardDiff
value on the same scenario, so ForwardDiff sits at 1.0 by construction;
values below 1.0 are faster (or lighter), above 1.0 slower (or heavier).
Timings use short per-measurement budgets so the page stays cheap to
build; treat small differences as indicative rather than exact.
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

bench_parts = map(backend_entries) do entry
    excluded = union(global_broken,
        get(backend_broken, entry.name, Set{String}()),
        get(backend_skip, entry.name, Set{String}()))
    scens = filter(s -> !(s.name in excluded), scenario_list)
    part = DataFrame(DIT.benchmark_differentiation(
        [entry.backend], scens;
        logging = false,
        benchmark_test = false,
        benchmark_seconds = 0.5))
    ## Label rows with the registry's backend name, which distinguishes
    ## configurations (e.g. Enzyme forward vs reverse) that share a package.
    part[!, :backend_label] .= entry.name
    part
end
raw_bench = vcat(bench_parts...)

bench_long = @chain raw_bench begin
    @rsubset :operator == ^(:gradient)
    @rtransform begin
        :backend = :backend_label
        :scenario = :scenario.name
        :time_us = :time * 1e6
        :bytes_kb = :bytes / 1024
    end
    @rsubset isfinite(:time_us) && isfinite(:bytes_kb)
    @select :backend :scenario :time_us :bytes_kb
end;

## The baseline every cost is divided by: ForwardDiff when the registry has
## it (the org standard), otherwise the registry's first backend.
baseline = any(e -> e.name == "ForwardDiff", backend_entries) ?
           "ForwardDiff" : first(backend_entries).name

ref = @chain bench_long begin
    @rsubset :backend == baseline
    @select :scenario :ref_time=:time_us :ref_bytes=:bytes_kb
end

rel = @chain bench_long begin
    leftjoin(ref, on = :scenario)
    @rsubset !ismissing(:ref_time) && !ismissing(:ref_bytes)
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

n_total = length(scenario_list)

summary_table = @chain rel begin
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

summary_table

md"""
### Spread across scenarios

Each box summarises a backend's relative cost across the scenario set, on
a log scale so speed-ups and slow-downs are symmetric around the baseline
at 1.0.
"""

md"""
```@raw html
<details><summary>Show plotting code</summary>
```
"""

plot_df = @chain rel begin
    stack([:rel_time, :rel_bytes],
        variable_name = :metric, value_name = :value)
    @rsubset isfinite(:value) && :value > 0
    @rtransform begin
        :metric = :metric == "rel_time" ? "Relative time" :
                  "Relative allocations"
        :family = first(split(:backend))
        :mode = occursin("reverse", lowercase(:backend)) ? "reverse" :
                "forward"
    end
end

## Order the facets time-then-allocations.
metric_order = sorter(["Relative time", "Relative allocations"])

fig_relative = draw(
    data(plot_df) *
    mapping(
        :backend => "",
        :value => "Cost relative to $baseline",
        col = :metric => metric_order) *
    visual(BoxPlot);
    figure = (size = (1200, 500),),
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
        :value => "Cost relative to $baseline",
        color = :family => "Backend family",
        marker = :mode => "Mode",
        col = :metric => metric_order) *
    visual(Scatter, markersize = 11);
    figure = (size = (1600, 800),),
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
  Fitting a single distribution has a handful of parameters, which is why
  ForwardDiff usually leads the low-dimensional rows; among the forward
  backends it is typically the fastest on small smooth log densities.
- Reverse mode (ReverseDiff, Enzyme reverse, Mooncake reverse) costs one
  pass per output regardless of the parameter count, so it pays off once
  this package's quantities sit inside a larger model with many latent
  parameters.
  In high-dimensional scenarios Enzyme reverse and Mooncake reverse tend
  to run several times faster than ForwardDiff, while ReverseDiff's tape
  overhead can leave it slower even there.

Turing's
[AD guidance](https://turinglang.org/docs/usage/automatic-differentiation/)
puts the crossover around 20 parameters: forward mode below, reverse mode
above.
ForwardDiff is the simplest fast default for the small-parameter case and
needs no configuration.
For a higher-dimensional model, switch to a reverse-mode backend.
In a Turing model you set this through the sampler's `adtype`, for example
`sample(model, NUTS(; adtype = AutoMooncake()), 1000)`, and the surest
choice is to benchmark the backends on your own model.

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
then one of this package's own quantities), confirm it, and build up to
the full model, so the construct a backend chokes on is easy to isolate.
When a genuinely broken combination is confirmed, declare it in the
`ADFixtures` registry (`backend_broken_scenarios`, or
`backend_skip_scenarios` when it cannot run at all): the gradient tests
then record it as `@test_broken` and this page reports it in the support
table, instead of the suite going red.

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
- `test/ADFixtures` is the package-owned registry this page renders from;
  scenarios, backends, and broken/skip declarations all live there.
- The shared harness and the `ADRegistry` contract live in
  [EpiAwarePackageTools.jl](https://github.com/EpiAware/EpiAwarePackageTools.jl).
"""
