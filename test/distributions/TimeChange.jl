@testitem "TimeChange constructor and validation" begin
    using Distributions

    d = CensoredDistributions.timechange(
        LogNormal(1.5, 0.5); scale = 2.0, rate = 0.1)
    @test d isa CensoredDistributions.TimeChange
    @test d.scale == 2.0
    @test d.rate == 0.1

    # Defaults: identity warp (operational time equals calendar time).
    d0 = CensoredDistributions.timechange(Normal(0.0, 1.0))
    @test d0.scale == 1.0
    @test d0.rate == 0.0

    # Promotion of mixed scale/rate types.
    dm = CensoredDistributions.timechange(
        Gamma(2.0, 1.0); scale = 2, rate = 1 // 10)
    @test dm.scale isa Float64
    @test dm.rate isa Float64

    # scale must be positive.
    @test_throws ArgumentError CensoredDistributions.timechange(
        Normal(0.0, 1.0); scale = 0.0)
    @test_throws ArgumentError CensoredDistributions.timechange(
        Normal(0.0, 1.0); scale = -1.0)
end

@testitem "TimeChange recovers affine scaling at rate zero" begin
    using Distributions

    # With rate = 0 the warp is linear, Λ(y) = scale * y, so the time-change
    # node reduces to the affine rescale `affine(X; scale = 1 / scale)`
    # (a clock running `scale` times faster, no shift).
    inner = LogNormal(1.5, 0.5)
    tc = CensoredDistributions.timechange(inner; scale = 2.0, rate = 0.0)
    aff = affine(inner; scale = 1 / 2.0)
    for y in [1.0, 2.5, 4.0, 7.0]
        @test logpdf(tc, y) ≈ logpdf(aff, y)
        @test cdf(tc, y) ≈ cdf(aff, y)
        @test pdf(tc, y) ≈ pdf(aff, y)
    end
end

@testitem "TimeChange change-of-variables logpdf/cdf" begin
    using Distributions

    inner = LogNormal(1.0, 0.4)
    scale, rate = 1.5, 0.2
    d = CensoredDistributions.timechange(
        inner; scale = scale, rate = rate)

    # Operational-time warp and its derivative (the calendar intensity λ(t)).
    Λ(y) = scale * (exp(rate * y) - 1) / rate
    λ(y) = scale * exp(rate * y)

    for y in [0.5, 1.5, 3.0, 5.0]
        @test cdf(d, y) ≈ cdf(inner, Λ(y))
        @test logcdf(d, y) ≈ logcdf(inner, Λ(y))
        @test logpdf(d, y) ≈ logpdf(inner, Λ(y)) + log(λ(y))
        @test pdf(d, y) ≈ pdf(inner, Λ(y)) * λ(y)
    end
end

@testitem "TimeChange pdf integrates to one and matches cdf derivative" begin
    using Distributions

    d = CensoredDistributions.timechange(
        LogNormal(1.0, 0.4); scale = 1.2, rate = 0.15)

    ts = range(0.0, 60.0; length = 200_000)
    h = step(ts)
    integral = sum(pdf(d, t) for t in ts) * h
    @test isapprox(integral, 1.0; atol = 1e-3)

    y = 3.0
    ε = 1e-6
    fd = (cdf(d, y + ε) - cdf(d, y - ε)) / (2ε)
    @test isapprox(pdf(d, y), fd; rtol = 1e-4)
end

@testitem "TimeChange quantile inverts the cdf" begin
    using Distributions

    inner = Gamma(2.0, 1.5)
    d = CensoredDistributions.timechange(inner; scale = 1.3, rate = 0.1)
    for p in [0.1, 0.4, 0.9]
        q = quantile(d, p)
        @test cdf(d, q) ≈ p
    end
end

@testitem "TimeChange rand matches the cdf" begin
    using Distributions, Random, Statistics

    rng = MersenneTwister(42)
    d = CensoredDistributions.timechange(
        LogNormal(0.8, 0.4); scale = 1.4, rate = 0.12)
    xs = rand(rng, d, 50_000)
    # Empirical cdf at a few points matches the analytic cdf.
    for q in [1.0, 2.0, 4.0]
        @test isapprox(mean(xs .<= q), cdf(d, q); atol = 0.01)
    end
end

@testitem "TimeChange support, params and intensity" begin
    using Distributions

    inner = Gamma(2.0, 1.5)
    d = CensoredDistributions.timechange(inner; scale = 1.3, rate = 0.2)

    @test minimum(d) ≈ 0.0
    @test maximum(d) == Inf
    @test insupport(d, 0.5)
    @test !insupport(d, -1.0)

    @test params(d) == (params(inner)..., 1.3, 0.2)
end

@testitem "TimeChange nests as a leaf in compose" begin
    using Distributions

    tree = compose((
        a = CensoredDistributions.timechange(
            Gamma(2.0, 1.0); scale = 1.5, rate = 0.1),
        b = LogNormal(1.0, 0.4)))

    @test isfinite(logpdf(tree, [1.0, 2.0]))

    # params_table is transparent to the inner free delay.
    tbl = params_table(tree)
    @test tbl.edge == [:a, :a, :b, :b]
    @test tbl.param == [:shape, :scale, :mu, :sigma]
end

@testitem "TimeChange round-trips through update" begin
    using Distributions

    tree = compose((a = CensoredDistributions.timechange(
        Gamma(2.0, 1.0); scale = 1.5, rate = 0.1),))
    upd = update(tree, (a = (shape = 3.0, scale = 2.0),))

    leaf = event(upd, :a)
    @test leaf isa CensoredDistributions.TimeChange
    @test leaf.scale == 1.5
    @test leaf.rate == 0.1
    @test params(leaf.dist) == (3.0, 2.0)
end

# AD gradient flow (ForwardDiff/ReverseDiff/Enzyme/Mooncake) is covered by the
# "TimeChange LogNormal logpdf" scenario in `test/ADFixtures`, run by the
# per-backend AD suite under `test/ad/`.
