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

    # The keyword-driven entry point, as a downstream author would call it. A
    # NamedTuple `compose` is a Parallel: the overall moment is a per-endpoint
    # NamedTuple and the full per-event NamedTuple is via `latent`.
    d = compose((onset_admit = Gamma(2.0, 1.0), admit_death = LogNormal(0.5, 0.4)))
    test_interface(d; name = "user chain", draw = rand(d),
        path = (:onset_admit,), overall = :vector, latent_moments = true,
        has_endpoint = false)
end

@testitem "test_node_interface over the built-in nodes and a leaf" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions.TestUtils: test_node_interface

    # The node contract holds for a plain leaf (one slot) and the built-in
    # composer nodes, which the harness walks as flat event vectors.
    test_node_interface(Gamma(2.0, 1.0); name = "leaf")
    test_node_interface(compose((a = Gamma(2.0, 1.0), b = LogNormal(0.5, 0.4)));
        name = "Parallel")
    test_node_interface(
        Sequential((Gamma(2.0, 1.0), Gamma(1.5, 2.0)), (:step_1, :step_2));
        name = "Sequential")
end

@testitem "test_node_interface accepts a user composer node" begin
    using CensoredDistributions, Distributions, Random
    import CensoredDistributions: child_nleaves, child_logpdf, child_rand!
    using CensoredDistributions.TestUtils: test_node_interface

    # A minimal user node combining two branches side by side, the worked example
    # from the extending docs. The public contract is reached by the qualified
    # name, the same way the leaf hooks are.
    struct Both{A, B}
        first::A
        second::B
    end
    child_nleaves(b::Both) = child_nleaves(b.first) + child_nleaves(b.second)
    function child_logpdf(b::Both, x, offset, ::Int)
        n1 = child_nleaves(b.first)
        n2 = child_nleaves(b.second)
        return child_logpdf(b.first, x, offset, n1) +
               child_logpdf(b.second, x, offset + n1, n2)
    end
    function child_rand!(out, offset, rng::AbstractRNG, b::Both)
        n1 = child_nleaves(b.first)
        child_rand!(out, offset, rng, b.first)
        child_rand!(out, offset + n1, rng, b.second)
        return nothing
    end

    node = Both(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    @test child_nleaves(node) == 2
    test_node_interface(node; name = "Both")

    # The underscored alias still resolves the same methods, for callers that
    # reached the contract before it was made public.
    @test CensoredDistributions._child_nleaves === child_nleaves
    @test CensoredDistributions._child_logpdf === child_logpdf
    @test CensoredDistributions._child_rand! === child_rand!
end
