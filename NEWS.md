## Unreleased

### Bug fixes

- `double_interval_censored` and `primary_censored` now select the
  analytic-vs-numeric solver by dispatching on the argument types rather
  than branching on a `Bool` value, so the return type stays concrete
  when the delay parameters are runtime values (e.g. inside a Turing
  model). Previously the value-level `force_numeric` branch inferred to a
  `Union` of the `AnalyticalSolver` and `NumericSolver` specialisations
  at non-constant-folded call sites, which propagated an abstract element
  type into downstream `pdf` loops and poisoned reverse-mode AD. Closes
  [#367](https://github.com/EpiAware/CensoredDistributions.jl/issues/367).

- Skip the CDF-saturation early-return in the numeric
  `primarycensored_cdf` path when the lower bound is at the distribution
  boundary (`lower == minimum(dist)`). Evaluating `cdf(dist, lower)`
  there is unnecessary (`cdf` is 0 by construction) and trips degenerate
  `0·(-Inf)` reverse-mode rules in `Distributions.jl`
  (e.g. `LogNormal` via `log(0)`), contaminating the ReverseDiff tape
  with NaN. Restores ReverseDiff and Mooncake gradient correctness on
  `PrimaryCensored LogNormal+Uniform numerical`. Closes
  [#249](https://github.com/EpiAware/CensoredDistributions.jl/issues/249).

### Added

- `latent(d)` wraps a censored delay (a `primary_censored` or
  `double_interval_censored` node) in its latent event-time form: a
  multivariate distribution over `[primary, observed]` whose joint
  `logpdf` scores the primary prior plus the conditional of the observed
  time given the primary, and whose `rand` draws a labelled
  `(primary, observed)` event record. The observed-delay interface (`cdf`,
  `pdf`, single-value `logpdf`, `logcdf`, `ccdf`, `logccdf`, `quantile`)
  is computed from the LATENT formulation, numerically integrating the
  augmented-data joint over the primary with the package's Gauss-Legendre
  quadrature, rather than delegating to the analytic marginal. This is a
  genuinely different computation from the analytic `primary_censored`
  marginal, and the two agree to quadrature tolerance, validating the
  formulation. The observed `logpdf` integrates in log space with the
  integration bounds clamped to the delay support, so it stays finite and
  finitely differentiable at extreme parameters and far-tail observations,
  which lets a Turing model on the latent form find valid initial
  parameters. `marginal` is the inverse, unwrapping a `Latent` back to the
  censored node it carries. `PrimaryConditional` and `get_primary_event`
  support this path. The composed latent form (over a composer tree) is a
  follow-up.

- `primary_censored` and `double_interval_censored` accept a `method`
  keyword taking a solver object, `AnalyticalSolver()` (the default) or
  `NumericSolver()`, each optionally given a quadrature solver
  (e.g. `NumericSolver(QuadGKJL())`). `AnalyticalSolver` and
  `NumericSolver` are now exported. Passing a concrete method object is
  the type-stable way to choose the CDF backend, and is preferred over
  the deprecated `force_numeric` flag.

### Deprecated

- The `force_numeric` keyword of `primary_censored` and
  `double_interval_censored` is deprecated. Pass `method = NumericSolver()`
  (or `method = AnalyticalSolver()`) instead. `force_numeric` still works
  but emits a deprecation warning and, being a runtime `Bool`, does not
  guarantee a concrete return type.

### Breaking

- `primary_censored(...; solver)` defaults to `GaussLegendre(; n = 64)`
  (was `QuadGKJL()`). The fixed-node solver traces cleanly through every
  AD backend, where adaptive quadrature does not. Pass
  `solver = QuadGKJL()` explicitly to keep adaptive accuracy.

### AD gradient infrastructure

- New `test/ad/` sub-environment with `DifferentiationInterfaceTest`-driven
  gradient correctness tests for the `logpdf` of every censored
  distribution type, plus unit-level tests for the gamma CDF rrule
  (`_grad_p_a_series` baseline, `Mooncake.TestUtils.test_rule`, and
  defensive guards). Wired into `task test-ad`, `task test`, and a
  dedicated CI job.
- New `test/ADFixtures` path package is the single source of truth for
  the scenario set and the working/broken backend matrix. The test
  suite, benchmark suite (`benchmark/src/ad_gradients.jl`), and docs
  tutorial all add it as a path dep so the three AD surfaces stay in
  lock-step.
- Gradient references are computed with ForwardDiff. Adaptive
  finite-difference baselines (e.g. `central_fdm(5, 1)`) disagree
  with every AD backend by ~10% on Weibull analytical scenarios
  because `PrimaryCensored.logpdf` internally finite-differences the
  CDF with a hardcoded `h = 1e-8`; ForwardDiff's Dual propagation
  through that same FD gives the exact derivative the package
  computes and matches the other backends to ~1e-6.
- New "Automatic differentiation backends" tutorial at
  `docs/src/getting-started/tutorials/ad-backends.jl` renders the
  full backend × scenario matrix via `DIT.benchmark_differentiation`
  with failures left visible.
- README gains a per-backend support badge row reflecting the measured
  state (ForwardDiff full; ReverseDiff tape, Mooncake reverse, and
  Mooncake forward partial; Enzyme forward/reverse broken).
- Known follow-ups tracked in
  [#217](https://github.com/EpiAware/CensoredDistributions.jl/issues/217)
  (`gamma_inc` `Dual` dispatch gap on the `Distributions.cdf(Gamma)`
  path used by `IntervalCensored Gamma arbitrary`),
  [#225](https://github.com/EpiAware/CensoredDistributions.jl/issues/225)
  (Enzyme + DIT-Mooncake interaction).

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
