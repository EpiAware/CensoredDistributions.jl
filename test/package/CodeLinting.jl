@testitem "Code linting" tags=[:quality] begin
    # Skip on experimental/pre-release Julia where JET may not be compatible
    if VERSION >= v"1.10" && get(ENV, "JULIA_CI_EXPERIMENTAL", "false") != "true"
        using JET
        using Distributions
        using CensoredDistributions
        JET.test_package(CensoredDistributions; target_defined_modules = true)
    end
end
