@testitem "Code formatting" begin
    if VERSION >= v"1.10"
        using JuliaFormatter
        using CensoredDistributions
        @test JuliaFormatter.format(
            CensoredDistributions; verbose = false, overwrite = false
        )
    end
end
