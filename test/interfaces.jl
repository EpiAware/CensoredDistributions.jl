@testitem "public interface conformance over every composer shape" begin
    using CensoredDistributions
    using CensoredDistributions.TestUtils: test_interface, example_fixtures

    # Run the one interface checklist over the fixture set covering a bare
    # censored leaf, Sequential, Parallel, Competing, selecting, a nested mix, and
    # a latent-wrapped case. Each fixture's `@testset` records its own asserts.
    for fix in example_fixtures()
        test_interface(fix)
    end
end

@testitem "composers reject invalid construction" begin
    using CensoredDistributions
    using CensoredDistributions.TestUtils: test_rejects_invalid

    test_rejects_invalid()
end

@testitem "test_interface accepts a user distribution via keywords" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions.TestUtils: test_interface

    # The keyword-driven entry point, as a downstream author would call it.
    d = compose((onset_admit = Gamma(2.0, 1.0), admit_death = LogNormal(0.5, 0.4)))
    test_interface(d; name = "user chain", draw = rand(d),
        path = (:onset_admit,))
end
