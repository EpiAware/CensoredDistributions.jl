
@testitem "compose(origin; branches...) shares an origin across branches" begin
    using CensoredDistributions, Distributions

    incub = Gamma(2.5, 1.3)
    combined = compose(incub; cases = thin(Gamma(1.5, 1.2), 0.3),
        deaths = thin(Gamma(3.0, 4.0), 0.012))
    # A shared origin then a Parallel of the branch tails.
    @test combined isa CensoredDistributions.Sequential
    @test combined.components[2] isa CensoredDistributions.Parallel
    @test CensoredDistributions.component_names(combined.components[2]) ==
          (:cases, :deaths)

    # Equivalent to the explicit Sequential-ending-in-Parallel form.
    explicit = Sequential((incub,
        Parallel((thin(Gamma(1.5, 1.2), 0.3), thin(Gamma(3.0, 4.0), 0.012)),
            (:cases, :deaths))))
    @test combined == explicit

    # At least one branch is required.
    @test_throws ArgumentError compose(incub)
end
