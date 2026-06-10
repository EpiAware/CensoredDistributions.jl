@testitem "Select scores the data-selected alternative" begin
    using CensoredDistributions, Distributions

    d = select_branch(:index => Gamma(2.0, 1.0), :sourced => Gamma(5.0, 1.0))
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

    d = select_branch(:a => Gamma(2.0, 1.0), :b => Gamma(5.0, 1.0))
    # No default selection for scoring: a Select has no single distribution to
    # score without a kind.
    @test_throws ArgumentError logpdf(d, 3.0)
    # An unknown name is rejected.
    @test_throws ArgumentError logpdf(d, 3.0; kind = :missing_name)
    @test_throws ArgumentError rand(d; kind = :missing_name)
    # A selection samples that alternative (seeded for determinism).
    x = rand(Xoshiro(1), d; kind = :a)
    @test x isa Real && x > 0
    # Without a kind (forward simulation), an alternative is sampled.
    y = rand(Xoshiro(2), d)
    @test y isa Real && y > 0
end

@testitem "Select rand samples the selected alternative's distribution" begin
    using CensoredDistributions, Distributions
    using Random: Xoshiro

    d = select_branch(:short => Gamma(2.0, 0.5), :long => Gamma(20.0, 1.0))
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
    d = select_branch(:index => Gamma(2.0, 1.0),
        :sourced => LogNormal(1.0, 0.5))
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
    @test_throws ArgumentError select_branch(:only => Gamma(1.0, 1.0))
    # Unique names.
    @test_throws ArgumentError select_branch(
        :a => Gamma(1.0, 1.0), :a => Gamma(2.0, 1.0))
    # A custom selector field name is honoured.
    d = select_branch(:a => Gamma(1.0, 1.0), :b => Gamma(2.0, 1.0);
        selector = :case)
    @test d.selector == :case
end

@testitem "Select holds a composer alternative and compares structurally" begin
    using CensoredDistributions, Distributions

    # A Select alternative may itself be a composer (the supported nesting
    # direction: a composer INSIDE a Select).
    inner = select_branch(:a => Gamma(2.0, 1.0),
        :b => Sequential(Gamma(1.0, 1.0), LogNormal(0.5, 0.4)))
    @test inner isa CensoredDistributions.Select

    # Structural equality and hashing over names, alternatives, and selector.
    a = select_branch(:x => Gamma(2.0, 1.0), :y => Gamma(5.0, 1.0))
    b = select_branch(:x => Gamma(2.0, 1.0), :y => Gamma(5.0, 1.0))
    @test a == b
    @test hash(a) == hash(b)
    c = select_branch(:x => Gamma(2.0, 1.0), :y => Gamma(5.0, 1.0);
        selector = :case)
    @test a != c
end

@testitem "select_branch nests on select_branch and compose results" begin
    using CensoredDistributions, Distributions
    const CD = CensoredDistributions

    # compose-in-select: an alternative is a compose result.
    tree = compose((a = Gamma(2.0, 1.0), b = LogNormal(0.5, 0.4)))
    s1 = select_branch(:joint => tree, :leaf => Gamma(3.0, 1.0))
    @test s1 == CD.Select((:joint, :leaf), (tree, Gamma(3.0, 1.0)), :kind)

    # select-in-select: an alternative is itself a select.
    inner = select_branch(:short => Gamma(2.0, 1.0), :long => Gamma(5.0, 1.0))
    s2 = select_branch(:nested => inner, :flat => Gamma(1.0, 1.0))
    @test s2 == CD.Select((:nested, :flat), (inner, Gamma(1.0, 1.0)), :kind)

    # The supported direction (a composer/select INSIDE a Select) scores fine.
    @test logpdf(inner, 3.0; kind = :short) ≈ logpdf(Gamma(2.0, 1.0), 3.0)
end

@testitem "Select cannot be nested inside a Sequential/Parallel/compose" begin
    using CensoredDistributions, Distributions
    const CD = CensoredDistributions

    inner = select_branch(:a => Gamma(2.0, 1.0), :b => Gamma(5.0, 1.0))

    # A Select has no fixed contribution length, so it cannot be a composer child:
    # the constructors and `compose` reject it cleanly rather than constructing and
    # then MethodError-ing on length/logpdf/rand.
    @test_throws ArgumentError Parallel(Gamma(2.0, 1.0), inner)
    @test_throws ArgumentError Parallel(inner, Gamma(2.0, 1.0))
    @test_throws ArgumentError Sequential(Gamma(2.0, 1.0), inner)
    @test_throws ArgumentError compose((pick = inner, side = Normal(0.0, 1.0)))
    @test_throws ArgumentError compose([inner Gamma(2.0, 1.0)])
    # Every front-end (positional constructor, NamedTuple, matrix) gives the same
    # Select-specific guidance, not an opaque generic error.
    for thunk in (() -> Parallel(Gamma(2.0, 1.0), inner),
        () -> compose((pick = inner, side = Normal(0.0, 1.0))),
        () -> compose([inner Gamma(2.0, 1.0)]))
        err = try
            thunk()
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("Select", err.msg)
    end

    # The supported direction (a composer INSIDE a Select) still constructs AND is
    # operable: logpdf / rand run on the selected alternative.
    tree = compose((a = Gamma(2.0, 1.0), b = LogNormal(0.5, 0.4)))
    s = select_branch(:joint => tree, :leaf => Gamma(3.0, 1.0))
    @test s isa CD.Select
    @test logpdf(s, [1.0, 2.0]; kind = :joint) ≈ logpdf(tree, [1.0, 2.0])
    @test logpdf(s, 1.5; kind = :leaf) ≈ logpdf(Gamma(3.0, 1.0), 1.5)
    @test rand(s; kind = :leaf) isa Real
    @test rand(s; kind = :joint) isa AbstractVector
end

@testitem "nested select_branch scores and samples via model entry" begin
    using CensoredDistributions, Distributions, Random
    using DynamicPPL: @model, to_submodel, logjoint

    # A select alternative that is itself a select: the model entry reads the
    # selector, picks the alternative, and delegates to its own model.
    inner = select_branch(:short => Gamma(2.0, 1.0), :long => Gamma(5.0, 1.0);
        selector = :sub)
    d = select_branch(:nested => inner, :flat => Gamma(3.0, 1.0))

    @model gen(dist, row) = obs ~ to_submodel(
        composed_distribution_model(dist, row))

    # Routes through `:nested` then `:short`; equals the leaf marginal logpdf.
    row = (kind = :nested, sub = :short, value = 2.5)
    lp = only(logjoint(gen(d, row), (;)))
    @test isfinite(lp)

    Random.seed!(11)
    draw = gen(d, (kind = :flat, value = missing))()
    @test draw isa Real && isfinite(draw)
end
