# Run the benchmark suite of a given checkout and save the results.
#
# Usage: julia --project=<checkout>/benchmark run.jl <benchmark_dir> <out.json>
#
# `benchmark_dir` is the `benchmark/` directory of the checkout to
# benchmark (the PR working tree or a `main` worktree). Including that
# directory's `benchmarks.jl` builds `SUITE` from the checkout's own
# sources and fixtures, so each revision is benchmarked against its own
# code. This is what lets the comparison cover the AD gradient suite even
# across an API change: `main` runs `main`'s fixtures, the PR runs its
# own. `run.jl` itself always comes from the PR, so a new runner does not
# need to exist on the baseline.
using BenchmarkTools

benchmark_dir = abspath(ARGS[1])
out_file = ARGS[2]

include(joinpath(benchmark_dir, "benchmarks.jl"))  # defines `SUITE`

# A short per-benchmark budget keeps the two-revision run affordable in
# CI; the minimum-time estimator used in the comparison is stable well
# below the default 5 s.
results = run(SUITE; verbose = true, seconds = 2)
BenchmarkTools.save(out_file, results)
println("Saved benchmark results to ", out_file)
