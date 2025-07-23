@testitem "Code linting" tags=[:jet] begin
    if VERSION >= v"1.10"
        using JET
        using Distributions
        using CensoredDistributions
        JET.test_package(CensoredDistributions; target_defined_modules = true)
    end
end
