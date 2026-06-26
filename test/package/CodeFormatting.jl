@testitem "Code formatting" tags=[:quality] begin
    using EpiAwarePackageTools: test_formatting
    # JuliaFormatter check via the shared kit wrapper. The check runs in the
    # isolated `test/formatter` environment (its `runtests.jl` is executed in
    # a subprocess and the test passes when it exits zero) so JuliaFormatter's
    # JuliaSyntax pin stays out of the main test env, where it would clash
    # with JET. The formatter env owns the directory list and style config.
    test_formatting(CensoredDistributions;
        env = joinpath(@__DIR__, "..", "formatter"))
end
