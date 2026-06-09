# Run this checkout's benchmark suite and save the results to JSON.
#
# Usage (from the repo root):
#   julia --project=benchmark benchmark/run.jl [out.json]
#
# The CI benchmark workflow inlines an equivalent runner rather than
# calling this file, so that it also works on the `main` baseline (which
# predates this script). Kept here as a local convenience and as the
# documented shape of what CI runs.
using BenchmarkTools

out_file = get(ARGS, 1, "results.json")

include(joinpath(@__DIR__, "benchmarks.jl"))  # defines `SUITE`

# A short per-benchmark budget keeps the run affordable; the
# minimum-time estimator used in the comparison is stable well below the
# default 5 s.
results = run(SUITE; verbose = true, seconds = 1)
BenchmarkTools.save(out_file, results)
println("Saved benchmark results to ", out_file)
