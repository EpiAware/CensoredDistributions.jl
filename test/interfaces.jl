@testitem "public interface conformance over every composer shape" begin
    using CensoredDistributions
    using CensoredDistributions.TestUtils: test_interface, example_fixtures
    import ForwardDiff

    # Run the one interface checklist over the full fixture registry: a bare
    # censored leaf, Sequential, Parallel, Resolve, Compete, choose, a nested
    # mix, a latent-wrapped case, the distribution-modifier / derived leaves
    # (affine, modify all links, weight, thin, Convolved, Difference,
    # ExponentiallyTilted), a defective no-event Resolve, and the deep-nesting
    # matrix (#645/#653). ForwardDiff is injected as the AD backend so the
    # AD-safety contract (a finite logpdf gradient) runs on every fixture that
    # carries an `ad` probe. Each fixture's `@testset` records its own asserts.
    for fix in example_fixtures()
        test_interface(fix; ad_gradient = ForwardDiff.gradient)
    end
end

@testitem "fixture registry covers every public distribution type" begin
    using CensoredDistributions
    using CensoredDistributions.TestUtils: test_registry_coverage

    # A new public distribution / leaf / node type added without a
    # `test_interface` fixture fails here (the registry-completeness meta-test).
    test_registry_coverage()
end

@testitem "interface AD-safety under Mooncake (reverse-mode)" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions.TestUtils: test_ad_safety
    import Mooncake

    # AD-safety is a contract under reverse-mode too, not just ForwardDiff. A
    # Mooncake gradient closure over the fixture AD probes (a representative
    # subset: a censored leaf, a thinned defective leaf, a modified hazard, a
    # convolution) must be finite. Mooncake is a main-test-env dep; backends not
    # loaded in a given env simply skip this item.
    function mooncake_gradient(f, θ)
        rule = Mooncake.build_rrule(f, θ)
        _, grads = Mooncake.value_and_gradient!!(rule, f, θ)
        return grads[2]
    end

    probes = (
        ("dic leaf",
            θ -> logpdf(
                double_interval_censored(Gamma(θ[1], θ[2]);
                    primary_event = Uniform(0, 1), interval = 1.0),
                3.0),
            [2.0, 1.0]),
        ("thin (defective)",
            θ -> logpdf(thin(LogNormal(θ[1], θ[2]), 0.3), 2.0), [1.5, 0.5]),
        ("modify (log link)",
            θ -> logpdf(modify(LogNormal(θ[1], θ[2]), -log(2.0); link = log),
                2.0), [1.5, 0.5]),
        ("Convolved",
            θ -> logpdf(
                convolve_distributions(
                    Gamma(θ[1], θ[2]), LogNormal(0.5, 0.4)), 3.0), [2.0, 1.0])
    )
    for (nm, f, θ) in probes
        test_ad_safety(f, θ; ad_gradient = mooncake_gradient, name = nm)
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
