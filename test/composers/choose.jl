@testitem "Choose scores the data-selected alternative" begin
    using CensoredDistributions, Distributions

    d = choose(:index => Gamma(2.0, 1.0), :sourced => Gamma(5.0, 1.0))
    @test d isa CensoredDistributions.Choose
    @test d.selector == :kind
    @test CensoredDistributions._n_alternatives(d) == 2

    # logpdf / pdf route to the selected alternative's own density.
    @test logpdf(d, 3.0; kind = :index) ≈ logpdf(Gamma(2.0, 1.0), 3.0)
    @test logpdf(d, 3.0; kind = :sourced) ≈ logpdf(Gamma(5.0, 1.0), 3.0)
    @test pdf(d, 3.0; kind = :index) ≈ pdf(Gamma(2.0, 1.0), 3.0)
    # The two alternatives genuinely differ, so the selection matters.
    @test logpdf(d, 3.0; kind = :index) != logpdf(d, 3.0; kind = :sourced)
end

@testitem "Choose requires a selection for logpdf/rand" begin
    using CensoredDistributions, Distributions
    using Random: Xoshiro

    d = choose(:a => Gamma(2.0, 1.0), :b => Gamma(5.0, 1.0))
    # No default selection for scoring: a Choose has no single distribution to
    # score without a kind.
    @test_throws ArgumentError logpdf(d, 3.0)
    # An unknown name is rejected.
    @test_throws ArgumentError logpdf(d, 3.0; kind = :missing_name)
    @test_throws ArgumentError rand(d; kind = :missing_name)
    # A selection samples that alternative directly (the committed draw is the
    # alternative's own raw value, no selector tag).
    x = rand(Xoshiro(1), d; kind = :a)
    @test x isa Real && x > 0
    # Without a kind (forward simulation) an alternative is sampled UNIFORMLY and
    # the draw is a self-describing record: the selector field names the drawn
    # alternative, and the value rides in `:value`. The record round-trips through
    # `logpdf` with no `kind`.
    y = rand(Xoshiro(2), d)
    @test y isa NamedTuple
    @test y.kind in (:a, :b)
    @test y.value isa Real && y.value > 0
    @test isfinite(logpdf(d, y))
end

@testitem "Choose rand samples the selected alternative's distribution" begin
    using CensoredDistributions, Distributions
    using Random: Xoshiro

    d = choose(:short => Gamma(2.0, 0.5), :long => Gamma(20.0, 1.0))
    n = 4000
    short = [rand(Xoshiro(i), d; kind = :short) for i in 1:n]
    long = [rand(Xoshiro(i), d; kind = :long) for i in 1:n]
    # The selected branch governs the draw: the long branch has a far larger
    # mean, so the sample means separate cleanly.
    @test isapprox(sum(short) / n, mean(Gamma(2.0, 0.5)); rtol = 0.1)
    @test isapprox(sum(long) / n, mean(Gamma(20.0, 1.0)); rtol = 0.1)
    @test sum(long) / n > 5 * sum(short) / n
end

@testitem "Choose hot path is type stable" begin
    using CensoredDistributions, Distributions
    using Test: @inferred

    # Heterogeneous alternatives (different concrete types) still infer, because
    # the selection barriers into the chosen alternative's concrete type.
    d = choose(:index => Gamma(2.0, 1.0),
        :sourced => LogNormal(1.0, 0.5))
    score_idx(dd, x) = logpdf(dd, x; kind = :index)
    score_src(dd, x) = logpdf(dd, x; kind = :sourced)
    @test (@inferred score_idx(d, 3.0)) ≈ logpdf(Gamma(2.0, 1.0), 3.0)
    @test (@inferred score_src(d, 3.0)) ≈ logpdf(LogNormal(1.0, 0.5), 3.0)

    # The selection helper itself returns the concrete alternative type.
    pick_score(dd, x, k) = logpdf(CensoredDistributions._pick(dd, k), x)
    @inferred pick_score(d, 3.0, :index)
end

@testitem "Choose validates construction" begin
    using CensoredDistributions, Distributions

    # At least two alternatives.
    @test_throws ArgumentError choose(:only => Gamma(1.0, 1.0))
    # Unique names.
    @test_throws ArgumentError choose(
        :a => Gamma(1.0, 1.0), :a => Gamma(2.0, 1.0))
    # A custom selector field name is honoured.
    d = choose(:a => Gamma(1.0, 1.0), :b => Gamma(2.0, 1.0);
        selector = :case)
    @test d.selector == :case
end

@testitem "Choose holds a composer alternative and compares structurally" begin
    using CensoredDistributions, Distributions

    # A Choose alternative may itself be a composer (the supported nesting
    # direction: a composer INSIDE a Choose).
    inner = choose(:a => Gamma(2.0, 1.0),
        :b => Sequential(Gamma(1.0, 1.0), LogNormal(0.5, 0.4)))
    @test inner isa CensoredDistributions.Choose

    # Structural equality and hashing over names, alternatives, and selector.
    a = choose(:x => Gamma(2.0, 1.0), :y => Gamma(5.0, 1.0))
    b = choose(:x => Gamma(2.0, 1.0), :y => Gamma(5.0, 1.0))
    @test a == b
    @test hash(a) == hash(b)
    c = choose(:x => Gamma(2.0, 1.0), :y => Gamma(5.0, 1.0);
        selector = :case)
    @test a != c
end

@testitem "choose nests on choose and compose results" begin
    using CensoredDistributions, Distributions
    const CD = CensoredDistributions

    # compose-in-choose: an alternative is a compose result.
    tree = compose((a = Gamma(2.0, 1.0), b = LogNormal(0.5, 0.4)))
    s1 = choose(:joint => tree, :leaf => Gamma(3.0, 1.0))
    @test s1 == CD.Choose((:joint, :leaf), (tree, Gamma(3.0, 1.0)), :kind)

    # choose-in-choose: an alternative is itself a choose.
    inner = choose(:short => Gamma(2.0, 1.0), :long => Gamma(5.0, 1.0))
    s2 = choose(:nested => inner, :flat => Gamma(1.0, 1.0))
    @test s2 == CD.Choose((:nested, :flat), (inner, Gamma(1.0, 1.0)), :kind)

    # The supported direction (a composer/select INSIDE a Choose) scores fine.
    @test logpdf(inner, 3.0; kind = :short) ≈ logpdf(Gamma(2.0, 1.0), 3.0)
end

@testitem "Choose nests inside a Sequential/Parallel/compose" begin
    using CensoredDistributions, Distributions
    const CD = CensoredDistributions

    # A Choose with equal-width alternatives occupies a fixed flat slot, so it is a
    # valid composer child: the constructors and `compose` accept it and the
    # flat path commits to the first alternative.
    inner = choose(:a => Gamma(2.0, 1.0), :b => Gamma(5.0, 1.0))
    @test Parallel(Gamma(2.0, 1.0), inner) isa CD.Parallel
    @test Parallel(inner, Gamma(2.0, 1.0)) isa CD.Parallel
    @test Sequential(Gamma(2.0, 1.0), inner) isa CD.Sequential
    @test compose((pick = inner, side = Normal(0.0, 1.0))) isa CD.Parallel
    # The flat (data-free) value path scores the first alternative.
    p = Parallel(Gamma(2.0, 1.0), inner)
    @test logpdf(p, [1.0, 2.0]) ≈
          logpdf(Gamma(2.0, 1.0), 1.0) + logpdf(Gamma(2.0, 1.0), 2.0)

    # Alternatives with differing leaf counts cannot share one flat slot: the
    # mismatch surfaces when the flat layout is needed (length / logpdf / rand).
    ragged = choose(:a => Gamma(2.0, 1.0),
        :b => compose((x = Gamma(2.0, 1.0), y = Gamma(2.0, 1.0))))
    rp = Parallel(Gamma(2.0, 1.0), ragged)
    @test_throws ArgumentError length(rp)

    # The supported direction (a composer INSIDE a Choose) still constructs AND is
    # operable: logpdf / rand run on the selected alternative.
    tree = compose((a = Gamma(2.0, 1.0), b = LogNormal(0.5, 0.4)))
    s = choose(:joint => tree, :leaf => Gamma(3.0, 1.0))
    @test s isa CD.Choose
    @test logpdf(s, [1.0, 2.0]; kind = :joint) ≈ logpdf(tree, [1.0, 2.0])
    @test logpdf(s, 1.5; kind = :leaf) ≈ logpdf(Gamma(3.0, 1.0), 1.5)
    @test rand(s; kind = :leaf) isa Real
    # The joint alternative is a multivariate composer, so its draw is a
    # labelled NamedTuple.
    @test rand(s; kind = :joint) isa NamedTuple
end

@testitem "nested choose scores and samples via model entry" begin
    using CensoredDistributions, Distributions, Random
    using DynamicPPL: @model, to_submodel, logjoint

    # A choose alternative that is itself a choose: the model entry reads the
    # selector, picks the alternative, and delegates to its own model.
    inner = choose(:short => Gamma(2.0, 1.0), :long => Gamma(5.0, 1.0);
        selector = :sub)
    d = choose(:nested => inner, :flat => Gamma(3.0, 1.0))

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

@testitem "standalone Choose: bare rand tags the drawn alternative" begin
    using CensoredDistributions, Distributions, Random

    d = choose(:short => Gamma(2.0, 1.0), :long => Gamma(5.0, 1.0))

    # A bare draw (no kind) is a self-describing record: the selector field names
    # the drawn alternative, the value rides in `:value`, and it round-trips
    # through `logpdf` with no `kind` argument (the selector recovers the
    # alternative).
    draw = rand(MersenneTwister(1), d)
    @test draw isa NamedTuple
    @test keys(draw) == (:kind, :value)
    @test draw.kind in (:short, :long)
    @test draw.value isa Real

    # Both alternatives are drawn over many samples, and every draw round-trips,
    # with the score equal to the named alternative's own logpdf at the value.
    function draw_and_check(rng, d, N)
        seen_short = false
        seen_long = false
        for _ in 1:N
            x = rand(rng, d)
            @test isfinite(logpdf(d, x))
            ref = logpdf(x.kind === :short ? Gamma(2.0, 1.0) : Gamma(5.0, 1.0),
                x.value)
            @test logpdf(d, x) ≈ ref
            x.kind === :short ? (seen_short = true) : (seen_long = true)
        end
        return seen_short, seen_long
    end
    ss, sl = draw_and_check(MersenneTwister(42), d, 300)
    @test ss && sl
end

@testitem "standalone Choose: custom selector names the tag field" begin
    using CensoredDistributions, Distributions, Random

    d = choose(:a => Gamma(2.0, 1.0), :b => LogNormal(0.5, 0.4);
        selector = :variant)
    draw = rand(MersenneTwister(2), d)
    @test keys(draw) == (:variant, :value)
    @test draw.variant in (:a, :b)
    @test isfinite(logpdf(d, draw))

    # The record's selector field is read by name: a record missing it errors
    # clearly (no silent default), and a non-Symbol selector is rejected.
    @test_throws ArgumentError logpdf(d, (value = 1.0,))
    @test_throws ArgumentError logpdf(d, (variant = "a", value = 1.0))
end

@testitem "standalone Choose: bare rand over composer alternatives" begin
    using CensoredDistributions, Distributions, Random

    # Plain composer alternatives: the tagged record carries the selector plus the
    # alternative's per-value fields, scored through the alternative's own logpdf.
    sfast = Sequential((Gamma(1.0, 1.0), Gamma(1.0, 1.0)), (:a, :b))
    sslow = Sequential((Gamma(2.0, 2.0), Gamma(2.0, 2.0)), (:c, :d))
    d = choose(:fast => sfast, :slow => sslow)

    rng = MersenneTwister(9)
    seen = Set{Symbol}()
    for _ in 1:100
        x = rand(rng, d)
        @test x isa NamedTuple
        @test x.kind in (:fast, :slow)
        @test isfinite(logpdf(d, x))
        push!(seen, x.kind)
    end
    @test seen == Set((:fast, :slow))

    # Censored composer alternatives round-trip too (the alternative's own event
    # record rides under the selector tag).
    ci = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    cseq = compose((onset_admit = ci,
        admit_death = primary_censored(Gamma(1.5, 1.0), Uniform(0, 1))))
    dc = choose(:x => cseq, :y => cseq)
    rc = rand(MersenneTwister(2), dc)
    @test rc isa NamedTuple
    @test rc.kind in (:x, :y)
    @test isfinite(logpdf(dc, rc))
end

@testitem "standalone Choose: batch rand(c, n) draws n tagged records" begin
    using CensoredDistributions, Distributions, Random

    d = choose(:short => Gamma(2.0, 1.0), :long => Gamma(5.0, 1.0))
    draws = rand(MersenneTwister(3), d, 50)
    @test draws isa AbstractVector
    @test length(draws) == 50
    @test all(x -> x isa NamedTuple && x.kind in (:short, :long), draws)
    # Every batched draw round-trips through logpdf.
    @test all(x -> isfinite(logpdf(d, x)), draws)
    # The no-rng count form works too.
    @test length(rand(d, 5)) == 5
end

@testitem "standalone Choose: explicit kind still returns the raw draw" begin
    using CensoredDistributions, Distributions, Random

    d = choose(:short => Gamma(2.0, 1.0), :long => Gamma(5.0, 1.0))
    # With an explicit kind the committed draw is the alternative's own raw value
    # (no selector tag): the caller already named the alternative.
    x = rand(MersenneTwister(1), d; kind = :short)
    @test x isa Real
    @test isfinite(logpdf(d, x; kind = :short))

    # A composer alternative under explicit kind is its own labelled record (no
    # selector tag added).
    seq = Sequential((Gamma(1.0, 1.0), Gamma(1.0, 1.0)), (:a, :b))
    dc = choose(:c => seq, :d => Gamma(3.0, 1.0))
    rc = rand(MersenneTwister(1), dc; kind = :c)
    @test rc isa NamedTuple
    @test !haskey(rc, :kind)
end
