# CensoredDistributions.jl Benchmarks

Benchmarks for testing distribution operation performance, comparing analytical
vs numerical methods where applicable.

## Quick Start

### One-time Setup

Install the `benchpkg` CLI to your global Julia environment:

```bash
task benchmark-install
# Or: julia -e 'using Pkg; Pkg.add("AirspeedVelocity")'
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
Pass `method = NumericSolver()` to force numerical integration.

## CI Integration

Benchmarks run automatically on PRs using the
[AirspeedVelocity GitHub Action](https://github.com/MilesCranmer/AirspeedVelocity.jl),
comparing the PR head against the base branch.

The action's own flat table (every benchmark, including one row per AD
scenario x backend pair) is written to the job summary rather than posted, as
it is unreadable once the AD-gradient suite is included.
`benchmark/comment/comment.jl` reads the same result JSON and posts a single
PR comment with a "most changed" summary, a compact AD scenario x backend
ratio matrix, and the full results folded behind a `<details>` block.

## Direct CLI Usage

```bash
# Run benchmarks
benchpkg --rev=dirty --script=benchmark/benchmarks.jl

# Compare specific revisions
benchpkg --rev=main,dirty --script=benchmark/benchmarks.jl

# Run with tuning (slower, more precise)
benchpkg --rev=dirty --script=benchmark/benchmarks.jl --tune

# View results
benchpkgtable CensoredDistributions
```
