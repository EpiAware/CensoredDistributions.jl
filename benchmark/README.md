# CensoredDistributions.jl Benchmarks

Benchmarks for testing distribution operation performance, comparing analytical
vs numerical methods where applicable.

## Quick Start

Run benchmarks locally using the Task automation:

```bash
# Run full benchmark suite
task benchmark

# Quick benchmark (single sample)
task benchmark-quick

# Compare against main branch
task benchmark-compare
```

## Benchmark Structure

The benchmarks are organised by distribution type and method:

```
PrimaryCensored/
  Gamma+Uniform/
    analytical/  (cdf, pdf, logpdf, rand)
    numerical/   (cdf, pdf, logpdf, rand)
  LogNormal+Uniform/
    analytical/  (cdf, pdf, logpdf, rand)
    numerical/   (cdf, pdf, logpdf, rand)
  Weibull+Uniform/
    analytical/  (cdf, pdf, logpdf, rand)
    numerical/   (cdf, pdf, logpdf, rand)
  Exponential+Uniform/
    numerical/   (no analytical solution)
  Gamma+Exponential/
    numerical/   (non-Uniform primary)
  LogNormal+Exponential/
    numerical/   (non-Uniform primary)

IntervalCensored/
  Regular/     (cdf, pdf, logpdf, rand)
  Arbitrary/   (cdf, pdf, logpdf, rand)
  Exponential/ (cdf, pdf, logpdf, rand)

DoubleIntervalCensored/
  LogNormal+Uniform/  (cdf, pdf, logpdf, rand)
  Exponential+Uniform/ (cdf, pdf, logpdf, rand)
```

## Analytical vs Numerical

The `PrimaryCensored` distribution supports analytical solutions for certain
distribution pairs with Uniform primary events:

- Gamma delay with Uniform primary
- LogNormal delay with Uniform primary
- Weibull delay with Uniform primary

All other combinations use numerical integration. The benchmarks compare both
methods to verify analytical solutions provide expected speedups.

Use `force_numeric=true` to force numerical integration even when analytical
solutions are available, which is useful for debugging or AD compatibility.

## CI Integration

Benchmarks run automatically on pull requests that modify `src/` or `benchmark/`
using [AirspeedVelocity.jl](https://github.com/MilesCranmer/AirspeedVelocity.jl).
Results are posted as PR comments comparing against the main branch.

## Manual Benchmark Runs

For more control, use the AirspeedVelocity CLI directly:

```bash
# Install AirspeedVelocity
julia -e 'using Pkg; Pkg.add("AirspeedVelocity")'

# Run benchmarks
julia -e 'using AirspeedVelocity; AirspeedVelocity.benchmark(".")'

# Compare revisions
julia -e 'using AirspeedVelocity; AirspeedVelocity.benchmark(".", rev="main", rev_target="HEAD")'
```
