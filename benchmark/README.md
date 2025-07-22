# CensoredDistributions.jl Benchmarks

Benchmarks for testing automatic differentiation (AD) performance with CensoredDistributions.jl models.

## Quick Start

First activate the benchmark environment:
```julia
using Pkg; Pkg.activate("benchmark")
```

Then load and run benchmarks:
```julia
using BenchCensoredDistributions

# Structure of the suite
BenchCensoredDistributions.suite

# Run specific distribution benchmarks
run(BenchCensoredDistributions.suite["primarycensored"]["ForwardDiff"])

# Run specific AD backend for a distribution
run(BenchCensoredDistributions.suite["discretised"]["ForwardDiff"])
```

## AD Backends Tested

- **ForwardDiff**: Forward-mode AD
- **ReverseDiff**: Reverse-mode AD (compiled and non-compiled)
- **Mooncake**: Modern reverse-mode AD
- **EnzymeForward/EnzymeReverse**: LLVM-based AD

## Files

- `make_suite.jl` - Creates benchmark suites for models
- `bench/` - Distribution-specific benchmark models
- `benchmarks.jl` - Main benchmark loader

## Running All Benchmarks

```bash
julia --project=benchmark benchmark/runbenchmarks.jl
```