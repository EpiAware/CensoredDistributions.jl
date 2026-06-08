@testitem "Select scores the data-selected alternative" begin
    using CensoredDistributions, Distributions

    d = select(:index => Gamma(2.0, 1.0), :sourced => Gamma(5.0, 1.0))
    @test d isa CensoredDistributions.Select
    @test d.selector == :kind
    @test CensoredDistributions._n_alternatives(d) == 2

    # logpdf / pdf route to the selected alternative's own density.
    @test logpdf(d, 3.0; kind = :index) ≈ logpdf(Gamma(2.0, 1.0), 3.0)
    @test logpdf(d, 3.0; kind = :sourced) ≈ logpdf(Gamma(5.0, 1.0), 3.0)
    @test pdf(d, 3.0; kind = :index) ≈ pdf(Gamma(2.0, 1.0), 3.0)
    # The two alternatives genuinely differ, so the selection matters.
    @test logpdf(d, 3.0; kind = :index) != logpdf(d, 3.0; kind = :sourced)
end

@testitem "Select requires a selection for logpdf/rand" begin
    using CensoredDistributions, Distributions
    using Random: Xoshiro

    d = select(:a => Gamma(2.0, 1.0), :b => Gamma(5.0, 1.0))
    # No default selection: a Select has no single distribution to score/sample.
    @test_throws ArgumentError logpdf(d, 3.0)
    @test_throws ArgumentError rand(d)
    # An unknown name is rejected.
    @test_throws ArgumentError logpdf(d, 3.0; kind = :missing_name)
    @test_throws ArgumentError rand(d; kind = :missing_name)
    # A selection samples that alternative (seeded for determinism).
    x = rand(Xoshiro(1), d; kind = :a)
    @test x isa Real && x > 0
end

@testitem "Select rand samples the selected alternative's distribution" begin
    using CensoredDistributions, Distributions
    using Random: Xoshiro

    d = select(:short => Gamma(2.0, 0.5), :long => Gamma(20.0, 1.0))
    n = 4000
    short = [rand(Xoshiro(i), d; kind = :short) for i in 1:n]
    long = [rand(Xoshiro(i), d; kind = :long) for i in 1:n]
    # The selected branch governs the draw: the long branch has a far larger
    # mean, so the sample means separate cleanly.
    @test isapprox(sum(short) / n, mean(Gamma(2.0, 0.5)); rtol = 0.1)
    @test isapprox(sum(long) / n, mean(Gamma(20.0, 1.0)); rtol = 0.1)
    @test sum(long) / n > 5 * sum(short) / n
end

@testitem "Select hot path is type stable" begin
    using CensoredDistributions, Distributions
    using Test: @inferred

    # Heterogeneous alternatives (different concrete types) still infer, because
    # the selection barriers into the chosen alternative's concrete type.
    d = select(:index => Gamma(2.0, 1.0), :sourced => LogNormal(1.0, 0.5))
    score_idx(dd, x) = logpdf(dd, x; kind = :index)
    score_src(dd, x) = logpdf(dd, x; kind = :sourced)
    @test (@inferred score_idx(d, 3.0)) ≈ logpdf(Gamma(2.0, 1.0), 3.0)
    @test (@inferred score_src(d, 3.0)) ≈ logpdf(LogNormal(1.0, 0.5), 3.0)

    # The selection helper itself returns the concrete alternative type.
    pick_score(dd, x, k) = logpdf(CensoredDistributions._pick(dd, k), x)
    @inferred pick_score(d, 3.0, :index)
end

@testitem "Select validates construction" begin
    using CensoredDistributions, Distributions

    # At least two alternatives.
    @test_throws ArgumentError select(:only => Gamma(1.0, 1.0))
    # Unique names.
    @test_throws ArgumentError select(
        :a => Gamma(1.0, 1.0), :a => Gamma(2.0, 1.0))
    # A custom selector field name is honoured.
    d = select(:a => Gamma(1.0, 1.0), :b => Gamma(2.0, 1.0); selector = :case)
    @test d.selector == :case
end

@testitem "Select nests as a composer child and compares structurally" begin
    using CensoredDistributions, Distributions

    # A Select alternative may itself be a composer, and a Select nests inside a
    # Sequential/Parallel like any other node.
    inner = select(:a => Gamma(2.0, 1.0),
        :b => Sequential(Gamma(1.0, 1.0), LogNormal(0.5, 0.4)))
    @test inner isa CensoredDistributions.Select
    par = Parallel(Gamma(2.0, 1.0), inner)
    @test par isa CensoredDistributions.Parallel

    # Structural equality and hashing over names, alternatives, and selector.
    a = select(:x => Gamma(2.0, 1.0), :y => Gamma(5.0, 1.0))
    b = select(:x => Gamma(2.0, 1.0), :y => Gamma(5.0, 1.0))
    @test a == b
    @test hash(a) == hash(b)
    c = select(:x => Gamma(2.0, 1.0), :y => Gamma(5.0, 1.0); selector = :case)
    @test a != c
end
