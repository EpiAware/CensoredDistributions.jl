md"""
# [Automatic differentiation backends](@id ad-backends)

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

## Tested backends

| ForwardDiff | ReverseDiff (tape) | Enzyme forward | Enzyme reverse | Mooncake reverse | Mooncake forward |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ![](https://img.shields.io/badge/ForwardDiff-full-brightgreen) | ![](https://img.shields.io/badge/ReverseDiff%20tape-full-brightgreen) | ![](https://img.shields.io/badge/Enzyme%20forward-partial-yellow) | ![](https://img.shields.io/badge/Enzyme%20reverse-full-brightgreen) | ![](https://img.shields.io/badge/Mooncake%20reverse-full-brightgreen) | ![](https://img.shields.io/badge/Mooncake%20forward-full-brightgreen) |

- **ForwardDiff** works on every scenario via the Dual-number extension
  for `_gamma_cdf`.
- **Mooncake reverse** works on every scenario, picking up the
  `@from_chainrules` lift of our `_gamma_cdf` rrule. The package also
  passes `Mooncake.TestUtils.test_rule` (`test/ad/gamma_ad.jl`).
- **Mooncake forward** works on every scenario, picking up the
  `@from_chainrules` lift of our `_gamma_cdf` frule (default mode
  generates both `rrule!!` and `frule!!`). It also passes
  `Mooncake.TestUtils.test_rule` in forward mode. The earlier gap, where
  only an `rrule` was shipped so forward mode found no rule on the
  incomplete-gamma path, was closed in
  [#270](https://github.com/EpiAware/CensoredDistributions.jl/issues/270).
- **ReverseDiff (tape)** works on every scenario. The numerical-path
  LogNormal regression tracked in
  [#249](https://github.com/EpiAware/CensoredDistributions.jl/issues/249)
  was fixed by gating the CDF-saturation early-return so `cdf(dist, ...)`
  is never evaluated at the support boundary.
- **Enzyme** ships an extension (`ext/CensoredDistributionsEnzymeExt.jl`)
  that registers rules via `EnzymeRules.@easy_rule` (one declaration
  covers both modes) for `_gamma_cdf` and for `SpecialFunctions.gamma`.
  The `gamma` rule is needed because, with only `EnzymeSpecialFunctionsExt`
  loaded, Enzyme mis-lowers `gamma(x)` to the `loggamma` known-op and
  returns `ψ(x)` instead of `Γ(x) ψ(x)` — silently wrong by a factor of
  `Γ(x)`. The analytical Gamma and Weibull paths call `gamma(k + 1)` and
  `gamma(1 + 1/k)` outside `_gamma_cdf`, so this corrupted the shape
  partial of the whole pipeline (see
  [#263](https://github.com/EpiAware/CensoredDistributions.jl/issues/263)).
  - **Reverse mode** now matches the ForwardDiff reference on every
    scenario, but only when invoked with two settings that are the
    caller's responsibility (not something the extension can set):

    ```julia
    using ADTypes, Enzyme
    AutoEnzyme(
        mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
        function_annotation = Enzyme.Duplicated)
    ```

    `function_annotation = Duplicated` stops Enzyme flagging the closure
    over the observation data as non-readonly (a generic Enzyme
    requirement for closures over arrays, including Turing likelihoods),
    and `set_runtime_activity` resolves per-value activity at runtime
    through the `Integrals` quadrature and the distribution
    constructors. The analytical paths work with just `Duplicated`; the
    numerical (quadrature) paths additionally need `set_runtime_activity`,
    so passing both is the recommended default. These are the settings
    the `ADFixtures` `Enzyme reverse` backend uses.
  - **Forward mode** is partial: it works on the `ExponentiallyTilted`
    primaries and the LogNormal interval/double-interval scenarios but
    still trips an upstream Enzyme mixed-activity check on the
    `jl_new_struct` of the `Uniform`-primary delay constructors
    (Gamma/LogNormal/Weibull). Tracked in
    [#225](https://github.com/EpiAware/CensoredDistributions.jl/issues/225).
- `IntervalCensored Gamma arbitrary` fails universally, on every
  backend. Its CDF differences route through `Distributions.cdf(Gamma,
  x)` → `SpecialFunctions.gamma_inc`, which **bypasses** the
  `_gamma_cdf` helper our rules wrap. ForwardDiff hits a hard
  `MethodError` because `gamma_inc` has no `Dual` method; the reverse-
  mode backends (ReverseDiff, Mooncake) also fail because the rrule we
  ship targets `_gamma_cdf`, not `gamma_inc` — to be picked up here it
  would need to live on `gamma_inc` itself (i.e. upstream in
  SpecialFunctions). The package-side workaround is to re-route
  `IntervalCensored`'s CDF computation through `_gamma_cdf`, tracked in
  [#257](https://github.com/EpiAware/CensoredDistributions.jl/issues/257);
  the original `gamma_inc` Dual gap is
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
Below is two pivoted views — wall-clock time and bytes allocated —
with scenarios as rows and backends as columns. The raw long-format
result is available as `raw_bench` if you want GC fraction, compile
fraction, or the `value_and_gradient` rows.
"""

backend_name = Dict(entry.backend => entry.name
for entry in ADFixtures.backends())

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
        :time_us = round(:time * 1e6; digits = 2)
        :bytes_kb = round(:bytes / 1024; digits = 1)
    end
end;

md"""
Per-row winner — fastest backend in the time table, smallest
allocation footprint in the bytes table — is bolded. Cells where
DIT couldn't produce a number (broken combinations) appear blank.
"""

function bold_min_per_row(df; idcol = :scenario)
    cols = setdiff(propertynames(df), [idcol])
    out = DataFrame(idcol => df[!, idcol])
    for c in cols
        out[!, c] = Vector{Any}(undef, nrow(df))
    end
    for r in 1:nrow(df)
        vals = [(c, df[r, c]) for c in cols if !ismissing(df[r, c])]
        minv = isempty(vals) ? nothing : minimum(last.(vals))
        for c in cols
            v = df[r, c]
            out[r, c] = if ismissing(v)
                ""
            elseif v == minv
                "**$(v)**"
            else
                string(v)
            end
        end
    end
    out
end

md"""
### Time (μs)
"""

bold_min_per_row(unstack(bench_long, :scenario, :backend, :time_us))

md"""
### Allocations (KiB)
"""

bold_min_per_row(unstack(bench_long, :scenario, :backend, :bytes_kb))

md"""
## When to use which backend

- **ForwardDiff** — default for small parameter dimensions and the most
  broadly compatible backend in this package.
- **ReverseDiff (tape) / (compiled)** — scales better as the parameter
  count grows; the compiled tape amortises its setup over many calls.
- **Enzyme reverse** — high-performance AD via LLVM; works through the
  full pipeline when called with
  `AutoEnzyme(mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
  function_annotation = Enzyme.Duplicated)` (see the status notes
  above). Forward mode is only partial.
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
