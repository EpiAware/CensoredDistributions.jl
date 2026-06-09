@testitem "Affine constructor and validation" begin
    using Distributions

    d = affine(LogNormal(1.5, 0.5); scale = 2.0, shift = 1.0)
    @test d isa CensoredDistributions.Affine
    @test d.scale == 2.0
    @test d.shift == 1.0

    # Defaults: identity transform.
    d0 = affine(Normal(0.0, 1.0))
    @test d0.scale == 1.0
    @test d0.shift == 0.0

    # Promotion of mixed scale/shift types.
    dm = affine(Gamma(2.0, 1.0); scale = 2, shift = 1.5)
    @test dm.scale isa Float64
    @test dm.shift isa Float64

    # scale must be positive.
    @test_throws ArgumentError affine(Normal(0.0, 1.0); scale = 0.0)
    @test_throws ArgumentError affine(Normal(0.0, 1.0); scale = -1.0)
end

@testitem "Affine change-of-variables logpdf/cdf" begin
    using Distributions

    inner = LogNormal(1.5, 0.5)
    d = affine(inner; scale = 2.0, shift = 1.0)
    for y in [2.0, 3.5, 5.0, 8.0]
        x = (y - 1.0) / 2.0
        @test logpdf(d, y) ≈ logpdf(inner, x) - log(2.0)
        @test pdf(d, y) ≈ pdf(inner, x) / 2.0
        @test cdf(d, y) ≈ cdf(inner, x)
        @test logcdf(d, y) ≈ logcdf(inner, x)
    end
end

@testitem "Affine matches Distributions affine specials" begin
    using Distributions

    # Shift-only of a Normal equals a relocated Normal.
    ds = affine(Normal(0.0, 1.0); shift = 3.0)
    ref_s = Normal(3.0, 1.0)
    for y in [2.5, 3.0, 3.5, 4.2]
        @test logpdf(ds, y) ≈ logpdf(ref_s, y)
        @test cdf(ds, y) ≈ cdf(ref_s, y)
    end

    # Scale-only of a Normal equals a rescaled Normal.
    dsc = affine(Normal(0.0, 1.0); scale = 2.5)
    ref_sc = Normal(0.0, 2.5)
    for y in [-1.0, 0.5, 1.7, 3.0]
        @test logpdf(dsc, y) ≈ logpdf(ref_sc, y)
        @test cdf(dsc, y) ≈ cdf(ref_sc, y)
    end
end

@testitem "Affine quantile, support, moments and params" begin
    using Distributions

    inner = Gamma(2.0, 1.5)
    d = affine(inner; scale = 3.0, shift = 2.0)

    for p in [0.1, 0.4, 0.9]
        @test quantile(d, p) ≈ 3.0 * quantile(inner, p) + 2.0
    end

    @test minimum(d) ≈ 3.0 * minimum(inner) + 2.0
    @test maximum(d) == Inf
    @test insupport(d, 2.5)
    @test !insupport(d, 1.0)

    @test mean(d) ≈ 3.0 * mean(inner) + 2.0
    @test var(d) ≈ 9.0 * var(inner)

    @test params(d) == (params(inner)..., 3.0, 2.0)
end

@testitem "Affine pdf integrates to one and matches cdf derivative" begin
    using Distributions

    d = affine(LogNormal(1.5, 0.5); scale = 2.0, shift = 1.0)

    ts = range(1.0, 80.0; length = 200_000)
    h = step(ts)
    integral = sum(pdf(d, t) for t in ts) * h
    @test isapprox(integral, 1.0; atol = 1e-3)

    y = 4.0
    ε = 1e-6
    fd = (cdf(d, y + ε) - cdf(d, y - ε)) / (2ε)
    @test isapprox(pdf(d, y), fd; rtol = 1e-4)
end

@testitem "Affine rand mean and quantile coverage" begin
    using Distributions, Random, Statistics

    rng = MersenneTwister(42)
    d = affine(Normal(1.0, 0.5); scale = 2.0, shift = 1.0)
    xs = rand(rng, d, 50_000)
    @test isapprox(mean(xs), mean(d); atol = 0.05)
    @test isapprox(std(xs), sqrt(var(d)); atol = 0.05)
end

@testitem "Affine nests as a leaf in compose" begin
    using Distributions

    tree = compose((
        a = affine(Gamma(2.0, 1.0); scale = 1.5, shift = 0.5),
        b = LogNormal(1.0, 0.4)))

    # Scores as a leaf branch in the composed tree.
    @test isfinite(logpdf(tree, [1.0, 2.0]))

    # params_table is transparent to the inner free delay: it shows the inner
    # Gamma's parameters, the fixed affine scale/shift hidden as structure.
    tbl = params_table(tree)
    @test tbl.edge == [:a, :a, :b, :b]
    @test tbl.param == [:shape, :scale, :mu, :sigma]
end

@testitem "Affine round-trips through update" begin
    using Distributions

    tree = compose((a = affine(Gamma(2.0, 1.0); scale = 1.5, shift = 0.5),))
    upd = update(tree, (a = (shape = 3.0, scale = 2.0),))

    leaf = get_event(upd, :a)
    @test leaf isa CensoredDistributions.Affine
    @test leaf.scale == 1.5
    @test leaf.shift == 0.5
    @test params(leaf.dist) == (3.0, 2.0)
end

# AD gradient flow (ForwardDiff/ReverseDiff/Enzyme/Mooncake) is covered by the
# "Affine LogNormal scale+shift logpdf" scenario in `test/ADFixtures`, run by
# the per-backend AD suite under `test/ad/`.
