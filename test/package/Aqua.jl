
@testitem "Aqua.jl" begin
    using Aqua
    Aqua.test_all(
        CensoredDistributions, ambiguities = false, persistent_tasks = false
    )
    Aqua.test_ambiguities(CensoredDistributions)
end
