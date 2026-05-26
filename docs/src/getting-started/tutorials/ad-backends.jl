md"""
# Automatic differentiation backends

CensoredDistributions.jl composes with Julia's automatic differentiation
(AD) ecosystem so that the censored `logpdf` can be used as the likelihood
in gradient-based inference, for example inside a
[Turing.jl](https://turinglang.org) model.
This tutorial reports the wall-clock cost of each AD backend on the
package's shared scenario set, defined with
[DifferentiationInterfaceTest.jl](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterfaceTest/stable/)
and exposed via the `ADFixtures` path package at `test/ADFixtures`.
The same scenario list powers the gradient tests in `test/ad/runtests.jl`
and the benchmark suite in `benchmark/src/ad_gradients.jl`.

## Support matrix

| ForwardDiff | ReverseDiff (tape) | Enzyme forward | Enzyme reverse | Mooncake reverse | Mooncake forward |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ![](https://img.shields.io/badge/ForwardDiff-full-brightgreen) | ![](https://img.shields.io/badge/ReverseDiff%20tape-partial-yellow) | ![](https://img.shields.io/badge/Enzyme%20forward-broken-red) | ![](https://img.shields.io/badge/Enzyme%20reverse-broken-red) | ![](https://img.shields.io/badge/Mooncake%20reverse-partial-yellow) | ![](https://img.shields.io/badge/Mooncake%20forward-partial-yellow) |

- **ForwardDiff** works on every scenario.
- **ReverseDiff (tape)** works everywhere except the numerical-path
  LogNormal scenarios (with both `Uniform` and `ExponentiallyTilted`
  primary events), tracked in
  [#249](https://github.com/EpiAware/CensoredDistributions.jl/issues/249).
- **Mooncake** (reverse and forward) works via plain
  `DifferentiationInterface.gradient` on every scenario, but
  `DifferentiationInterfaceTest.test_differentiation` errors on the
  primary-censored Gamma/Weibull paths (both `Uniform` and
  `ExponentiallyTilted` priors) — a DIT-Mooncake interaction tracked
  in [#225](https://github.com/EpiAware/CensoredDistributions.jl/issues/225).
- **Enzyme** (forward and reverse) fails on every scenario; also
  [#225](https://github.com/EpiAware/CensoredDistributions.jl/issues/225).
- `IntervalCensored Gamma arbitrary` fails universally because it routes
  through `Distributions.cdf(Gamma)` → `gamma_inc`, which the ForwardDiff
  `Dual` extension does not cover. Tracked in
  [#217](https://github.com/EpiAware/CensoredDistributions.jl/issues/217).

The scenario set covers analytical and numerical paths for Gamma,
LogNormal, and Weibull delay distributions with both `Uniform` and
`ExponentiallyTilted` primary events, plus `IntervalCensored` (regular
and arbitrary boundaries) and `DoubleIntervalCensored` constructions.

## Reproducing this page

The numbers below are measured on the docs-build machine. To regenerate
locally:

```
task docs
```

or, equivalently:

```
julia --project=docs docs/make.jl
```

`docs/Project.toml` adds `ADFixtures` as a path dep so the same scenario
set is loaded; results will reflect the local CPU.
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

scenarios = ADFixtures.scenarios()
all_backends = [entry.backend for entry in ADFixtures.backends()]

md"""
## Benchmark table

`DifferentiationInterfaceTest.benchmark_differentiation` runs every
(backend, scenario) combination and returns a Tables.jl-compatible
result. We pass every backend and every scenario so failures are
visible in the output rather than silently hidden.
"""

raw_bench = DIT.benchmark_differentiation(
    all_backends, scenarios;
    logging = false,
    benchmark_test = false
)

md"""
The raw result carries timing, allocation, GC and compile-fraction
columns per (backend, scenario, operator). Below is a trimmed view —
the full table is available as `raw_bench` if you want to inspect
allocations or GC time.
"""

@chain DataFrame(raw_bench) begin
    @rsubset :operator == ^(:gradient)
    @rtransform begin
        :backend = string(:backend)
        :scenario = :scenario.name
        :time_us = round(:time * 1e6; digits = 2)
        :bytes_kb = round(:bytes / 1024; digits = 1)
    end
    @select :backend :scenario :time_us :bytes_kb
end

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
  ForwardDiff reference using
  `DifferentiationInterfaceTest.test_differentiation`.
- `benchmark/src/ad_gradients.jl` runs the same scenarios under
  AirspeedVelocity. A benchmark history timeline is tracked in
  [#224](https://github.com/EpiAware/CensoredDistributions.jl/issues/224).
"""
