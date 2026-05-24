## Unreleased

### AD gradient infrastructure

- New `test/ad/` sub-environment with `DifferentiationInterfaceTest`-driven
  gradient correctness tests for the `logpdf` of every censored
  distribution type, and a `task test-ad` target wired into `task test`
  and a dedicated CI job.
- Shared scenario file at
  `docs/src/getting-started/tutorials/ad_scenarios.jl` exposes the AD
  scenario set, working/broken backend lists, and per-backend broken
  scenarios — consumed by the test suite, the benchmark suite
  (`benchmark/src/ad_gradients.jl`), and the new docs tutorial.
- New "Automatic differentiation backends" tutorial under
  `docs/src/getting-started/tutorials/ad-backends.jl` renders
  per-backend gradient timings via `DIT.benchmark_differentiation`.
- README gains a per-backend support badge row reflecting the measured
  state (ForwardDiff full; ReverseDiff tape, Mooncake reverse, and
  Mooncake forward partial; Enzyme forward/reverse broken).
- Known follow-ups tracked in #217 (`_gamma_inc` Dual dispatch),
  #225 (Enzyme + Mooncake remediation), and #249 (ReverseDiff
  regression on `PrimaryCensored LogNormal+Uniform numerical` after
  #230).

## v0.1.0 - Initial Release

CensoredDistributions.jl extends Distributions.jl to support primary event
censoring and interval censoring, enabling modelling of scenarios where
observation times are subject to various forms of censoring and truncation.

### Core functionality

**Distribution Constructors**
- `primary_censored(dist, primary_event)`: Creates primary event censored
  distributions where an initiating event occurs within a time window, then
  experiences a delay
- `interval_censored(dist, interval)`: Creates interval censored distributions
  for continuous values observed only within discrete intervals (regular or
  arbitrary intervals)
- `double_interval_censored(dist; kwargs...)`: Combines primary event
  censoring, optional truncation, and optional secondary interval censoring
  in an appropriate mathematical sequence

**Distribution Types**
- `PrimaryCensored{D, P}`: Wraps delay distribution `D` with primary event
  distribution `P`
- `IntervalCensored{D, T}`: Wraps continuous distribution `D` with interval
  boundaries of type `T`
- `DoubleIntervalCensored`: Composition of multiple censoring mechanisms

**Extensible CDF Method**
- `primarycensored_cdf()`: User-extensible method for computing CDFs of
  primary censored distributions with analytical solutions for common
  distribution pairs

**Utility Functions**
- `weight(dist, weights)`: Creates weighted distributions for efficient
  likelihood computation

### Distributions.jl interface

Partial implementation of Distributions.jl interface including:
- `pdf`, `logpdf`: Probability density functions
- `cdf`, `logcdf`: Cumulative distribution functions
- `quantile`: Inverse CDF (where analytically tractable)
- `rand`: Random number generation
- `minimum`, `maximum`, `insupport`: Support queries
- `mean`, `var`, `std`: Moments (where analytically tractable)

### Migration from primarycensored R package

CensoredDistributions.jl provides a Julia-native implementation with enhanced
functionality compared to the primarycensored R package:

**Function Mapping**
- R's `dprimarycensored()` → Julia's `pdf(primary_censored(...))`
- R's `pprimarycensored()` → Julia's `cdf(primary_censored(...))`
- R's `qprimarycensored()` → Julia's `quantile(primary_censored(...))`
- R's `rprimarycensored()` → Julia's `rand(primary_censored(...))`

**Parameter Differences**
- Primary event window: Use `primary_event = Uniform(0, pwindow)` instead of
  R's `pwindow` parameter
- Solver selection: Configure via `solver` keyword instead of R's
  integration method parameters
- Truncation: Apply via `truncated()` from Distributions.jl or use
  `double_interval_censored()` convenience function


### Contributors

- Sam Abbott ([@seabbs](https://github.com/seabbs))
- Damon Bayer ([@damonbayer](https://github.com/damonbayer))
- Sam Brand ([@sambrand](https://github.com/sambrand))
- Michael DeWitt ([@dewittpe](https://github.com/dewittpe))
- Joseph Lemaitre
