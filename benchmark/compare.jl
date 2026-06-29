#!/usr/bin/env julia
# MANAGED by EpiAwarePackageTools.scaffold — do not edit by hand.
#
# Compare two benchmark result files and write a Markdown PR comment, via the
# shared EpiAwarePackageTools benchmark harness. Per-(scenario x backend) AD rows
# are folded into a compact matrix using the `"AD gradients"` group convention.
#
#   julia --project=benchmark benchmark/compare.jl pr.json base.json out.md

using EpiAwarePackageTools.Benchmarks: compare_comment

const BACKEND_ORDER = ["ForwardDiff", "ReverseDiff (tape)", "Mooncake reverse",
    "Mooncake forward", "Enzyme reverse", "Enzyme forward"]

pr_file, base_file, out_file = ARGS[1], ARGS[2], ARGS[3]

comment = compare_comment(pr_file, base_file; backend_order = BACKEND_ORDER)
write(out_file, comment)
println("Wrote benchmark comparison to ", out_file)
