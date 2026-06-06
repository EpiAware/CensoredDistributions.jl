@testitem "Sequential composes plain dists and scores step values" begin
    using Distributions

    s = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    @test s isa CensoredDistributions.Sequential
    @test length(s) == 2
    # logpdf is the sum of the per-step leaf log-densities.
    @test logpdf(s, [1.5, 2.0]) ≈
          logpdf(Gamma(2.0, 1.0), 1.5) + logpdf(LogNormal(0.5, 0.4), 2.0)
    @test pdf(s, [1.5, 2.0]) ≈ exp(logpdf(s, [1.5, 2.0]))
    @test length(rand(s)) == 2
    # vector and varargs constructors agree.
    @test Sequential([Gamma(2.0, 1.0), LogNormal(0.5, 0.4)]) == s

    @test_throws DimensionMismatch logpdf(s, [1.0])
    @test_throws ArgumentError Sequential(())
end

@testitem "Parallel composes plain dists and sums branch logpdfs" begin
    using Distributions

    p = Parallel(Gamma(2.0, 1.0), LogNormal(1.0, 0.5))
    @test p isa CensoredDistributions.Parallel
    @test length(p) == 2
    @test logpdf(p, [2.0, 3.0]) ≈
          logpdf(Gamma(2.0, 1.0), 2.0) + logpdf(LogNormal(1.0, 0.5), 3.0)
    @test pdf(p, [2.0, 3.0]) ≈ exp(logpdf(p, [2.0, 3.0]))
    @test length(rand(p)) == 2
    @test Parallel([Gamma(2.0, 1.0), LogNormal(1.0, 0.5)]) == p

    @test_throws DimensionMismatch logpdf(p, [1.0, 2.0, 3.0])
    @test_throws ArgumentError Parallel(())
end

@testitem "Competing lowers to a MixtureModel" begin
    using Distributions

    cfr = 0.3
    c = Competing(:death => (Gamma(1.5, 1.0), cfr),
        :disch => (Gamma(2.0, 1.5), 1 - cfr))
    @test c isa CensoredDistributions.Competing
    mix = as_mixture(c)
    @test mix isa MixtureModel
    # The univariate interface delegates to the mixture lowering.
    @test logpdf(c, 2.0) ≈ logpdf(mix, 2.0)
    @test pdf(c, 2.0) ≈ pdf(mix, 2.0)
    @test cdf(c, 2.0) ≈ cdf(mix, 2.0)
    @test mean(c) ≈ mean(mix)
    @test minimum(c) == minimum(mix)

    # Branch probabilities must be valid and sum to one; >= 2 outcomes.
    @test_throws ArgumentError Competing(:a => (Gamma(1.0, 1.0), 0.5))
    @test_throws ArgumentError Competing(
        :a => (Gamma(1.0, 1.0), 0.3), :b => (Gamma(1.0, 1.0), 0.3))
    @test_throws ArgumentError Competing(
        :a => (Gamma(1.0, 1.0), 1.5), :b => (Gamma(1.0, 1.0), -0.5))
end

@testitem "Composers nest recursively (the nesting is the tree)" begin
    using Distributions

    c = Competing(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))
    # Competing (univariate) inside Sequential inside Parallel.
    nested = Parallel(Sequential(Gamma(2.0, 1.0), c), LogNormal(1.0, 0.5))
    @test length(nested) == 3   # two leaves in the chain plus one branch

    r = rand(nested)
    @test length(r) == 3

    # logpdf consumes a contiguous slice per child and recurses.
    lp = logpdf(nested, [1.5, 2.0, 3.0])
    @test lp ≈
          logpdf(Gamma(2.0, 1.0), 1.5) + logpdf(c, 2.0) +
          logpdf(LogNormal(1.0, 0.5), 3.0)

    # A nested Parallel inside a Parallel flattens its leaf values.
    deep = Parallel(Parallel(Gamma(1.0, 1.0), Gamma(2.0, 1.0)), Normal(0.0, 1.0))
    @test length(deep) == 3
    @test logpdf(deep, [0.5, 1.5, 0.0]) ≈
          logpdf(Gamma(1.0, 1.0), 0.5) + logpdf(Gamma(2.0, 1.0), 1.5) +
          logpdf(Normal(0.0, 1.0), 0.0)
end

@testitem "compose: NamedTuple, table and matrix build the same stack" begin
    using Distributions

    # --- irregular tree: NamedTuple, column table, row table agree ---
    nt = (a = Gamma(2.0, 1.0), b = [LogNormal(0.5, 0.4), Gamma(1.0, 1.0)])
    column_table = (name = [:a, :b1, :b2],
        dist = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4), Gamma(1.0, 1.0)],
        chain = [0, 1, 1])
    row_table = [(name = :a, dist = Gamma(2.0, 1.0), chain = 0),
        (name = :b1, dist = LogNormal(0.5, 0.4), chain = 1),
        (name = :b2, dist = Gamma(1.0, 1.0), chain = 1)]
    target = Parallel(Gamma(2.0, 1.0),
        Sequential(LogNormal(0.5, 0.4), Gamma(1.0, 1.0)))

    @test compose(nt) == target
    @test compose(column_table) == target
    @test compose(row_table) == target
    @test compose(nt) == compose(column_table) == compose(row_table)

    # --- regular grid: NamedTuple, table and Matrix agree ---
    nt2 = (r1 = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4)],
        r2 = [Gamma(1.0, 1.0), Gamma(3.0, 1.0)])
    table2 = (name = [:a, :b, :c, :d],
        dist = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4),
            Gamma(1.0, 1.0), Gamma(3.0, 1.0)],
        chain = [1, 1, 2, 2])
    mat2 = [Gamma(2.0, 1.0) LogNormal(0.5, 0.4)
            Gamma(1.0, 1.0) Gamma(3.0, 1.0)]
    target2 = Parallel(Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4)),
        Sequential(Gamma(1.0, 1.0), Gamma(3.0, 1.0)))

    @test compose(nt2) == target2
    @test compose(table2) == target2
    @test compose(mat2) == target2
    @test compose(nt2) == compose(table2) == compose(mat2)
end

@testitem "compose: matrix orientation and flat equivalences" begin
    using Distributions

    # Rows are parallel branches, columns are sequential steps.
    flat_parallel = compose(reshape(
        [Gamma(2.0, 1.0), LogNormal(1.0, 0.5)], 2, 1))
    @test flat_parallel == compose((a = Gamma(2.0, 1.0), b = LogNormal(1.0, 0.5)))
    @test flat_parallel ==
          compose((name = [:a, :b], dist = [Gamma(2.0, 1.0), LogNormal(1.0, 0.5)]))

    # A one-row matrix is a single sequential branch.
    chain = compose([Gamma(2.0, 1.0) LogNormal(0.5, 0.4) Gamma(1.0, 1.0)])
    @test chain == Parallel(Sequential(
        Gamma(2.0, 1.0), LogNormal(0.5, 0.4), Gamma(1.0, 1.0)))

    # A composed stack scores and samples like any composer.
    d = compose((r1 = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4)],
        r2 = [Gamma(1.0, 1.0), Gamma(3.0, 1.0)]))
    @test length(rand(d)) == 4
    @test isfinite(logpdf(d, rand(d)))
end

@testitem "compose: input validation" begin
    using Distributions

    @test_throws ArgumentError compose((name = [:a], extra = [1]))
    @test_throws ArgumentError compose((name = [:a], dist = [1.0]))
    @test_throws ArgumentError compose(42)
    # A NamedTuple sequential child must hold distributions.
    @test_throws ArgumentError compose((a = [1.0, 2.0],))
    # Negative chain group ids would collide with auto-generated leaf ids.
    @test_throws ArgumentError compose((name = [:a, :b],
        dist = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4)], chain = [-1, -1]))
end

@testitem "Composers show readably" begin
    using Distributions

    s = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    p = Parallel(Gamma(2.0, 1.0), LogNormal(1.0, 0.5))
    c = Competing(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))

    @test occursin("Sequential", sprint(show, s))
    @test occursin("Sequential chain", sprint(show, MIME"text/plain"(), s))
    @test occursin("Parallel", sprint(show, p))
    @test occursin("Parallel composer", sprint(show, MIME"text/plain"(), p))
    @test occursin("Competing", sprint(show, c))
    @test occursin("death", sprint(show, MIME"text/plain"(), c))
end
