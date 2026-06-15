## Unreleased

### Features

- Varying / partially-pooled per-stratum parameters for composed models. The
  batched record path previously assumed ONE shared composed distribution with
  global parameters (only `obs_time`/`weight` and the missingness pattern varied
  per record). `record_distributions(ds, rows; group)` now lets each record's
  edge use DIFFERENT (sampled) parameters: `ds` is a vector of composed
  distributions (one per stratum) and `group` is an integer stratum id per
  record. The grouping key is an integer from an AD-free data pass, so the
  sampled parameters (carried inside `ds`) never key a lookup, keeping the path
  AD-safe (ForwardDiff/ReverseDiff); records are bucketed by stratum and each
  stratum's segment construction is built once. A single stratum is bit-identical
  to the shared-`d` fast path. `batched_event_logpdf(ds, rows; group)` returns the
  grouped log density directly. The matching Turing entries are
  `composed_distribution_model(ds, table; group)` and
  `composed_parameters_model(template, strata_priors)` (a vector of per-stratum
  prior NamedTuples, each sampled under a `:stratumK` prefix), which lets the user
  encode no-pooling, full-pooling, or partial pooling (per-stratum parameters
  drawn off a shared hyperprior) freely.
- `linear_chain_stages`: lower an Exponential or Erlang (integer-shape Gamma)
  delay, or a `Sequential` chain of such leaves, to its linear-chain-trick
  `(rate, stages)` compartment structure (a `ChainStage` per step). This is the
  distributions -> compartments bridge an ODE/compartment model consumes:
  an Erlang(k, θ) delay is k Exponential sub-compartments leaving at rate 1/θ.
  Censoring wrappers are peeled to the free delay; non-Exp/Erlang families throw,
  since no exact finite linear chain represents them. A new tutorial composes
  the resulting delay compartments with an SIR-type ModelingToolkit system.
  Addresses #400.
- Labelled `NamedTuple` outputs for multivariate composed distributions. Any
  multivariate composed output is now self-labelling: `rand(d)` for a flat
  censored `Sequential`/`Parallel` returns a `NamedTuple` keyed by
  `event_names(d)` (the nested-tree `rand` already did), `rand(latent(d))`
  follows the same rule (a latent leaf draws `(primary, observed)`), and
  `mean(latent(d))`/`var`/`std` and the per-endpoint `mean(parallel)`/`var`/`std`
  return `NamedTuple`s keyed by the event / endpoint names. A univariate
  (collapsible) output stays a bare scalar. `logpdf` accepts the labelled
  `NamedTuple` draw (matched to the scored vector by name, field order
  irrelevant) so a draw round-trips straight back through `logpdf(d, rand(d))`;
  the internal vector-valued scored representation (and the
  `product_distribution` record path and AD) are unchanged. Closes #425.
- `marginal(d)` is the inverse of `latent`: it unwraps a `Latent` back to the
  marginal node it carries (`marginal(latent(d)) == d`) and is idempotent (a
  non-`Latent` node is returned unchanged).
- Batch latent model entry
  `composed_distribution_model(latent(d), rows)` /
  `composed_distribution_model(latent(d), table)`: a looping submodel that
  runs the per-record loop and `:recN` prefixing inside the package, so a
  latent fit collapses to one `~` (`obs ~ to_submodel(...)`) just like the
  marginal batch entry. Mirrors the marginal form's vector-of-rows and
  Tables.jl table signatures, so the two batch forms are symmetric. A
  per-record `obs_time` horizon stays rejected under `latent`. Closes #449.

### Bug fixes

- A `shared(:tag, ...)` censored leaf at a composed-tree origin no longer loses
  its censoring when scored. The censored-tree traversal strips every other
  wrapper (`Truncated`, `IntervalCensored`, `Weighted`) to recover a leaf's
  origin primary event and secondary interval, but did not descend through
  `Shared`, so a `shared(:inc, primary_censored(...))` first step silently
  scored as an uncensored origin and a shared `double_interval_censored` leaf
  dropped its interval discretisation, diverging from the untagged leaf. A
  shared censored leaf now scores identically to the same tree built from
  independent identical untagged leaves. Found while investigating the #395
  shared-dist compute-reuse follow-up; that cross-occurrence reuse yields no
  measurable speedup under the numerically-identical constraint (the
  primary-censored CDF quadrature is parameterised by the per-occurrence
  observed gap and the wrapper objects precompute nothing), so no reuse path was
  added; a benchmark records the like-for-like scoring cost for future work.
- `compose` no longer misclassifies a structural `NamedTuple` whose
  user-chosen branch keys are `:name`/`:dist` and carry distribution
  vectors (e.g. `(name = [d1, d2], dist = [d3, d4])`) as a `(name, dist)`
  column table. The column-table heuristic now also requires the `:dist`
  column to hold distributions and the `:name` column to hold
  non-distribution row labels, so such a NamedTuple builds the intended
  named `Parallel` of `Sequential` chains instead of a silently-wrong
  2-branch `Parallel`.
- `Convolved` `mean`/`var`/`std` docstrings no longer promise a PMF-weighting
  fallback for components without an analytic moment. That fallback was
  unreachable dead code (nothing opted in), so it has been removed; the
  docstrings now state that each component must provide an analytic
  `mean`/`var` or the call errors.
- Skip the CDF-saturation early-return in the numeric
  `primarycensored_cdf` path when the lower bound is at the distribution
  boundary (`lower == minimum(dist)`). Evaluating `cdf(dist, lower)`
  there is unnecessary (`cdf` is 0 by construction) and trips degenerate
  `0·(-Inf)` reverse-mode rules in `Distributions.jl`
  (e.g. `LogNormal` via `log(0)`), contaminating the ReverseDiff tape
  with NaN. Restores ReverseDiff and Mooncake gradient correctness on
  `PrimaryCensored LogNormal+Uniform numerical`. Closes
  [#249](https://github.com/EpiAware/CensoredDistributions.jl/issues/249).

### Breaking

- Multivariate composed `rand`/`mean`/`var`/`std` now return a `NamedTuple`
  instead of a bare `Vector` (see the Features note). Code that indexed the old
  Vector positionally (`r[1]`) still works via NamedTuple integer indexing, but
  code that relied on the `Vector` type (`r isa Vector`, broadcasting `sqrt.(r)`)
  must move to the NamedTuple (key access, `map(sqrt, r)`). A latent leaf's
  `rand`/`logpdf` use the labelled record `(primary, observed)`.
- `predict_events` is removed (all methods and the DynamicPPL extension method).
  Forward simulation is `rand(latent(d))` (a comprehension batches it); posterior
  event recovery is `DynamicPPL.predict(model, chain)` directly (the removed
  `predict_events(chain, model)` was a one-line pass-through to it).
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
