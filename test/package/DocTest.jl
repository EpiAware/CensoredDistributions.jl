@testitem "Run docstring tests" tags=[:quality] begin
    using Documenter
    doctest(CensoredDistributions)
end
