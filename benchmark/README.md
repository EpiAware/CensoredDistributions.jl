# CensoredDistributions.jl Benchmarks

Benchmarks for testing distribution operation performance, comparing analytical
vs numerical methods where applicable.

## Quick Start

### One-time Setup

Install the [AirspeedVelocity.jl](https://github.com/MilesCranmer/AirspeedVelocity.jl)
CLI tools to `~/.julia/bin`:

```bash
task benchmark-install
# Or manually:
julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.add("AirspeedVelocity"); Pkg.build("AirspeedVelocity")'
```

Ensure `~/.julia/bin` is on your PATH:

```bash
export PATH="$HOME/.julia/bin:$PATH"
```

### Running Benchmarks

```bash
# Benchmark current state
task benchmark
# Or: benchpkg --rev=dirty

# Compare main branch vs current state
task benchmark-compare
# Or: benchpkg --rev=main,dirty

# Run with tuning enabled
task benchmark-tune
# Or: benchpkg --rev=dirty --tune
```

### Viewing Results

```bash
# Print results table
benchpkgtable CensoredDistributions

# Generate plots
benchpkgplot CensoredDistributions --format=png
```

### Running Specific Benchmarks

Use the `--filter` flag to run specific benchmark groups:

```bash
# Only PrimaryCensored benchmarks
benchpkg --rev=dirty --filter=PrimaryCensored

# Only analytical methods
benchpkg --rev=dirty --filter=analytical
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
using the [AirspeedVelocity GitHub Action](https://github.com/MilesCranmer/AirspeedVelocity.jl).
Results are posted as PR comments comparing against the main branch.

## Local Development Without CLI

For quick local checks without installing the CLI, you can run benchmarks
directly with BenchmarkTools:

```bash
# Set up benchmark environment
julia --project=benchmark -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'

# Run all benchmarks
julia --project=benchmark -e 'using BenchmarkTools; include("benchmark/benchmarks.jl"); run(SUITE, verbose=true)'

# Run specific benchmarks interactively
julia --project=benchmark
```

```julia
using BenchmarkTools
include("benchmark/benchmarks.jl")

# Run specific distribution benchmarks
run(SUITE["PrimaryCensored"]["Gamma+Uniform"], verbose=true)

# Compare analytical vs numerical
run(SUITE["PrimaryCensored"]["LogNormal+Uniform"]["analytical"]["cdf"])
run(SUITE["PrimaryCensored"]["LogNormal+Uniform"]["numerical"]["cdf"])
```
