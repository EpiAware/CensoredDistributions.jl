#!/usr/bin/env julia
# MANAGED by EpiAwarePackageTools.scaffold — do not edit by hand.
#
# Build a legible PR benchmark comment from AirspeedVelocity result JSON,
# via the shared EpiAwarePackageTools benchmark harness
# (EpiAwarePackageTools.Benchmarks.asv_comment).
#
#   julia --project=benchmark/comment benchmark/comment/comment.jl \
#       <results-dir> <package> <base-rev> <head-rev> <out.md>
using EpiAwarePackageTools.Benchmarks: asv_comment
length(ARGS) == 5 || error(
    "usage: comment.jl <dir> <pkg> <base-rev> <head-rev> <out.md>")
dir, pkg, base_rev, head_rev, out = ARGS
write(out, asv_comment(dir, pkg, base_rev, head_rev))
println("Wrote benchmark comment to ", out)
