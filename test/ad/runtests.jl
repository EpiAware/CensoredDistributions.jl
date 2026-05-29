#!/usr/bin/env julia
# AD gradient tests for CensoredDistributions, organised as `@testitem`s
# (as in the main test suite) and run with TestItemRunner.
#
#   julia --project=test/ad test/ad/runtests.jl              # all backends
#   julia --project=test/ad test/ad/runtests.jl enzyme_reverse  # one tag
#
# Per-backend tags (`forwarddiff`, `reversediff`, `enzyme_reverse`,
# `enzyme_forward`, `mooncake_reverse`, `mooncake_forward`) let the
# per-backend CI (#269) run a single backend, so a transiently unstable
# backend only reds its own badge. Group tags `enzyme` and `mooncake` are
# also available. With no argument every AD item runs.

using TestItemRunner

if isempty(ARGS)
    TestItemRunner.run_tests(@__DIR__)
else
    selected = Symbol.(ARGS)
    TestItemRunner.run_tests(
        @__DIR__; filter = ti -> any(in(ti.tags), selected))
end
