# CensoredDistributions.jl Benchmarks

Benchmarks for testing distribution operation performance, comparing analytical
vs numerical methods where applicable.

## Quick Start

### One-time Setup

Install the `benchpkg` CLI:

```bash
task benchmark-install
```

Ensure `~/.julia/bin` is on your PATH.

### Running Benchmarks

```bash
# Benchmark current state
task benchmark

# Compare main branch vs current state
task benchmark-compare

# Filter to specific benchmarks
task benchmark -- --filter=PrimaryCensored
task benchmark -- --filter=analytical
task benchmark-compare -- --filter=Gamma
```

## Benchmark Structure

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

All other combinations use numerical integration.
Use `force_numeric=true` to force numerical integration.

## CI Integration

Benchmarks run automatically on PRs using the
[AirspeedVelocity GitHub Action](https://github.com/MilesCranmer/AirspeedVelocity.jl).
Results are posted as PR comments.

## Direct CLI Usage

```bash
# View results
benchpkgtable CensoredDistributions

# Compare specific revisions
benchpkg --rev=v0.2.5,main,dirty

# Run with tuning (slower, more precise)
benchpkg --rev=dirty --tune
```
