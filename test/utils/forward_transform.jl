@testitem "thin / cumulative construction and validation" begin
    using Distributions

    d = thin(LogNormal(1.5, 0.5), 0.3)
    @test d isa CensoredDistributions.Scaled
    @test d.factor == 0.3
    @test CensoredDistributions.get_dist(d) === LogNormal(1.5, 0.5)

    # thin probability must be in [0, 1].
    @test_throws ArgumentError thin(Normal(0.0, 1.0), -0.1)
    @test_throws ArgumentError thin(Normal(0.0, 1.0), 1.5)
    @test thin(Normal(0.0, 1.0), 0.0).factor == 0.0
    @test thin(Normal(0.0, 1.0), 1.0).factor == 1.0

    # nothing threads through unchanged.
    base = Gamma(2.0, 1.0)
    @test thin(base, nothing) === base

    c = cumulative(Gamma(2.0, 1.0))
    @test c isa CensoredDistributions.Cumulative
    @test CensoredDistributions.get_dist(c) === Gamma(2.0, 1.0)
end

@testitem "forward transforms are transparent to logpdf/cdf" begin
    using Distributions

    inner = LogNormal(1.5, 0.5)
    for d in (thin(inner, 0.3), cumulative(inner))
        for x in [0.5, 1.0, 2.0, 4.0]
            @test logpdf(d, x) == logpdf(inner, x)
            @test pdf(d, x) == pdf(inner, x)
            @test cdf(d, x) == cdf(inner, x)
            @test logcdf(d, x) == logcdf(inner, x)
            @test quantile(d, 0.4) == quantile(inner, 0.4)
        end
        @test minimum(d) == minimum(inner)
        @test maximum(d) == maximum(inner)
        @test mean(d) == mean(inner)
    end
end

@testitem "forward transforms peel via free_leaf and rewrap" begin
    using Distributions

    base = Gamma(2.0, 1.0)
    d = thin(interval_censored(base, 1.0), 0.3)
    # free_leaf peels the forward op AND the censoring to the inner delay.
    @test CensoredDistributions.free_leaf(d) === base
    # rewrap restores the forward op + censoring around a new inner delay.
    new_inner = Gamma(3.0, 0.5)
    rebuilt = CensoredDistributions.rewrap_leaf(d, new_inner)
    @test rebuilt isa CensoredDistributions.Scaled
    @test rebuilt.factor == 0.3
    @test CensoredDistributions.free_leaf(rebuilt) === new_inner
end

@testitem "forward ops apply to a series via _apply_forward_ops" begin
    using Distributions

    series = [1.0, 2.0, 3.0, 4.0]

    # thin scales.
    _, ops = CensoredDistributions._peel_forward(thin(Gamma(2.0, 1.0), 0.5))
    @test CensoredDistributions._apply_forward_ops(series, ops) ≈ 0.5 .* series

    # cumulative accumulates.
    _, opsc = CensoredDistributions._peel_forward(cumulative(Gamma(2.0, 1.0)))
    @test CensoredDistributions._apply_forward_ops(series, opsc) == cumsum(series)

    # peeled delay is the underlying (censored) delay, forward op stripped.
    inner,
    _ = CensoredDistributions._peel_forward(
        thin(interval_censored(Gamma(2.0, 1.0), 1.0), 0.5))
    @test inner isa CensoredDistributions.IntervalCensored
end

@testitem "forward transforms nest in composers" begin
    using Distributions

    # thin/cumulative are UnivariateDistributions, so they compose as leaves.
    p = compose((cases = thin(Gamma(1.5, 1.0), 0.3),
        deaths = thin(Gamma(3.0, 2.0), 0.01)))
    @test p isa CensoredDistributions.Parallel
    s = Sequential((Gamma(2.0, 1.0), thin(Gamma(1.5, 1.0), 0.3)))
    @test s isa CensoredDistributions.Sequential
end
