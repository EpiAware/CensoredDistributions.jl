`CensoredDistributions.jl` benchmarks its core operations to track
performance over time: primary-censored, interval-censored and
double-interval-censored evaluation, and AD gradients across the
supported backends (`AD gradients/...` in the summary below).

Run the suite locally with `task benchmark` (see `benchmark/README.md`
for filtering and comparison options), or directly:

```bash
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark benchmark/run.jl
```

The performance history below is published by `benchmark-history.yaml` on
each push to `main` and each tagged release, benchmarking the last five
tags plus the pushed commit so a timeline accumulates.
