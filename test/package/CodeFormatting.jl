@testitem "Code formatting" tags=[:quality] begin
    if VERSION >= v"1.10"
        using JuliaFormatter
        using CensoredDistributions
        @test JuliaFormatter.format(
            CensoredDistributions, verbose = true, overwrite = false)
    end
end
