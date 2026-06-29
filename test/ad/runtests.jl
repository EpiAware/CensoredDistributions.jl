#!/usr/bin/env julia
# MANAGED by EpiAwarePackageTools.scaffold — do not edit by hand.
#
# AD gradient test entry, organised as `@testitem`s and run with
# TestItemRunner. The AD items live in their own environment (Enzyme, Mooncake,
# etc. are not main-test deps) and in dedicated per-backend CI.
#
#   julia --project=test/ad test/ad/runtests.jl              # all backends
#   julia --project=test/ad test/ad/runtests.jl enzyme_reverse  # one tag
#
# Per-backend tags let the per-backend CI run a single backend so a transiently
# unstable backend only reds its own badge. With no argument every AD item runs.

using TestItemRunner

if isempty(ARGS)
    TestItemRunner.run_tests(@__DIR__)
else
    selected = Symbol.(ARGS)
    TestItemRunner.run_tests(
        @__DIR__; filter = ti -> any(in(ti.tags), selected))
end
