md"""
# Automatic differentiation backends

CensoredDistributions.jl composes with Julia's automatic differentiation
(AD) ecosystem so that the censored `logpdf` can be used as the likelihood
in gradient-based inference, for example inside a
[Turing.jl](https://turinglang.org) model.
This tutorial reports the wall-clock cost of each AD backend on the
package's shared scenario set, defined with
[DifferentiationInterfaceTest.jl](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterfaceTest/stable/)
and exposed at
`docs/src/getting-started/tutorials/ad_scenarios.jl`.
The same scenario list powers the gradient tests in `test/ad/runtests.jl`
and the benchmark suite in `benchmark/src/ad_gradients.jl`.
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

scenarios = ADFixtures.scenarios()
all_backends = [entry.backend for entry in ADFixtures.backends()]

md"""
## Benchmark table

`DifferentiationInterfaceTest.benchmark_differentiation` runs every
(backend, scenario) combination and returns a Tables.jl-compatible
table of runtimes and allocations. We pass every backend and every
scenario so failures are visible in the output (rows with a `false`
success flag or missing timing) rather than silently hidden. Known
broken combinations are tracked in
[#217](https://github.com/EpiAware/CensoredDistributions.jl/issues/217)
and
[#225](https://github.com/EpiAware/CensoredDistributions.jl/issues/225).
"""

DIT.benchmark_differentiation(
    all_backends, scenarios;
    logging = false,
    benchmark_test = false
)

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
