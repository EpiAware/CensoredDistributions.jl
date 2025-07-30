# CensoredDistributions.jl Release Notes

This file documents major releases of CensoredDistributions.jl.
For detailed patch and minor release notes, see our
[GitHub releases](https://github.com/EpiAware/CensoredDistributions.jl/releases).

## v0.1.0 - Initial Release

CensoredDistributions.jl extends Distributions.jl to support primary event
censoring and interval censoring, enabling modelling of scenarios where
observation times are subject to various forms of censoring and truncation.

### Core Functionality

**Distribution Constructors**
- `primary_censored(dist, primary_event)`: Creates primary event censored
  distributions where an initiating event occurs within a time window, then
  experiences a delay
- `interval_censored(dist, interval)`: Creates interval censored distributions
  for continuous values observed only within discrete intervals (regular or
  arbitrary intervals)
- `double_interval_censored(dist; kwargs...)`: Combines primary event
  censoring, optional truncation, and optional secondary interval censoring
  in appropriate mathematical sequence

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

### Distributions.jl Interface Compliance

Full implementation of Distributions.jl interface including:
- `pdf`, `logpdf`: Probability density functions
- `cdf`, `logcdf`: Cumulative distribution functions
- `quantile`: Inverse CDF (where analytically tractable)
- `rand`: Random number generation
- `minimum`, `maximum`, `insupport`: Support queries
- `mean`, `var`, `std`: Moments (where analytically tractable)

### Performance and Integration

**Analytical Solutions**
- Efficient analytical solutions for common distribution pairs (e.g., Gamma
  delay with uniform primary event)
- Numerical integration fallbacks using Integrals.jl for general cases
- Configurable solver selection with `solver` and `force_numeric` options

**Automatic Differentiation Compatibility**
- Full compatibility with ForwardDiff.jl, ReverseDiff.jl, Zygote.jl, and
  Enzyme.jl
- Seamless integration with Turing.jl for Bayesian inference
- Type-stable implementations for efficient precompilation

**Vectorisation Support**
- Leverages Distributions.jl broadcasting for efficient batch operations
- Supports vectorised PDF, CDF, and random number generation

### Applications and Use Cases

**Primary Event Censoring**
- Infection-to-symptom onset times with uncertain infection timing
- Exposure-to-outcome delays with uncertain exposure periods
- Any process where the initiating event time has uncertainty

**Interval Censoring**
- Daily reporting of continuous phenomena
- Discrete observation windows for continuous processes
- Custom interval boundaries for irregular observation schedules

**Double Interval Censoring**
- Complex observation processes combining multiple censoring mechanisms
- Epidemiological surveillance with reporting delays and discrete observation
- Clinical studies with multiple sources of observation uncertainty

### Migration from primarycensored R Package

CensoredDistributions.jl provides a Julia-native implementation with enhanced
functionality compared to the primarycensored R package:

**Function Mapping**
- R's `dprimarycensored()` → Julia's `pdf(primary_censored(...))`
- R's `pprimarycensored()` → Julia's `cdf(primary_censored(...))`
- R's `qprimarycensored()` → Julia's `quantile(primary_censored(...))`
- R's `rprimarycensored()` → Julia's `rand(primary_censored(...))`

**Enhanced Capabilities**
- Support for arbitrary primary event distributions beyond uniform
- Interval censoring for discrete observation windows
- Combined censoring mechanisms via `double_interval_censored()`
- Analytical solutions for additional distribution pairs
- Automatic differentiation support for gradient-based inference
- Integration with Julia's scientific computing ecosystem

**Parameter Differences**
- Primary event window: Use `primary_event = Uniform(0, pwindow)` instead of
  R's `pwindow` parameter
- Solver selection: Configure via `solver` keyword instead of R's
  integration method parameters
- Truncation: Apply via `truncated()` from Distributions.jl or use
  `double_interval_censored()` convenience function

### Dependencies and Compatibility

**Core Dependencies**
- Distributions.jl ≥ 0.25: Base distribution functionality
- Integrals.jl ≥ 4.5: Numerical integration for CDF computation
- LogExpFunctions.jl ≥ 0.3: Numerically stable log operations
- SpecialFunctions.jl ≥ 2.0: Mathematical special functions
- HypergeometricFunctions.jl ≥ 0.3: Hypergeometric function evaluation

**Julia Compatibility**
- Requires Julia ≥ 1.10
- Tested on Julia 1.10+ across multiple platforms

**Quality Assurance**
- Comprehensive test suite with TestItemRunner
- Aqua.jl quality checks for package health
- JET.jl static analysis for type stability
- SciML code style compliance
- Benchmark suite for performance monitoring

### Contributors

- Samuel Abbott ([@seabbs](https://github.com/seabbs))
- Damon Bayer ([@damonbayer](https://github.com/damonbayer))
- Michael DeWitt ([@dewittpe](https://github.com/dewittpe))
- Joseph Lemaitre ([@joelemaitre](https://github.com/joelemaitre))

### Ecosystem Integration

CensoredDistributions.jl is designed for seamless integration with:
- **Distributions.jl**: Full interface compliance enables drop-in replacement
- **Turing.jl**: Native support for Bayesian inference workflows
- **Optimization.jl**: Compatible with MLE and other optimization-based fitting
- **Plots.jl**: Automatic plotting support via Distributions.jl interface
- **StatsPlots.jl**: Statistical plotting integration

This initial release establishes CensoredDistributions.jl as a comprehensive
toolkit for censored distribution modelling in Julia's scientific computing
ecosystem.
