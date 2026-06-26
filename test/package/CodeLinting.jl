@testitem "Code linting" tags=[:quality] begin
    using EpiAwarePackageTools: test_linting
    # JET static analysis via the shared kit wrapper. JET runs in the
    # isolated `test/jet` environment (its `runtests.jl` is executed in a
    # subprocess and the test passes when it exits zero) to keep JET's
    # JuliaSyntax pin from clashing with the rest of the test environment.
    # The kit skips JET on experimental / pre-release Julia and when
    # `JULIA_CI_EXPERIMENTAL=true`, matching CD's previous guard.
    test_linting(CensoredDistributions;
        env = joinpath(@__DIR__, "..", "jet"))
end
