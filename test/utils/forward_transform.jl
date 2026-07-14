@testitem "thin / cumulative construction and validation" begin
    using CensoredDistributions, Distributions

    d = thin(LogNormal(1.5, 0.5), 0.3)
    @test d isa CensoredDistributions.Transformed
    @test d.op isa CensoredDistributions.ThinOp
    @test d.op.factor == 0.3
    @test CensoredDistributions.get_dist(d) === LogNormal(1.5, 0.5)

    # thin probability must be in [0, 1].
    @test_throws ArgumentError thin(Normal(0.0, 1.0), -0.1)
    @test_throws ArgumentError thin(Normal(0.0, 1.0), 1.5)
    @test thin(Normal(0.0, 1.0), 0.0).op.factor == 0.0
    @test thin(Normal(0.0, 1.0), 1.0).op.factor == 1.0

    # nothing threads through unchanged.
    base = Gamma(2.0, 1.0)
    @test thin(base, nothing) === base

    c = cumulative(Gamma(2.0, 1.0))
    @test c isa CensoredDistributions.Transformed
    @test c.op isa CensoredDistributions.CumulativeOp
    @test CensoredDistributions.get_dist(c) === Gamma(2.0, 1.0)

    # thin and cumulative are specialisations of the generic transform.
    g = transform(Gamma(2.0, 1.0), s -> 2.0 .* s)
    @test g isa CensoredDistributions.Transformed
    @test logpdf(g, 2.0) == logpdf(Gamma(2.0, 1.0), 2.0)
end

@testitem "cumulative is transparent to logpdf/cdf" begin
    using CensoredDistributions, Distributions

    # `cumulative` (and the generic `transform`) stay logpdf-transparent: an
    # individual delay use ignores the forward op. `thin` is the exception
    # (tested separately as the resolve + NoEvent one-of).
    inner = LogNormal(1.5, 0.5)
    d = cumulative(inner)
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

@testitem "thin is the resolve + NoEvent one-of under logpdf / rand" begin
    using CensoredDistributions, Distributions, Random

    inner = LogNormal(1.5, 0.5)
    p = 0.3
    d = thin(inner, p)

    # thin is NOT logpdf-transparent: an event is reported w.p. p, so the
    # density is defective (mass p), matching the `:event` branch of the
    # resolve + NoEvent one-of (`log p + logpdf(inner, x)`).
    for x in [0.5, 1.0, 2.0, 4.0]
        @test logpdf(d, x) ≈ log(p) + logpdf(inner, x)
        @test pdf(d, x) ≈ p * pdf(inner, x)
        @test cdf(d, x) ≈ p * cdf(inner, x)
        @test logcdf(d, x) ≈ log(p) + logcdf(inner, x)
        @test ccdf(d, x) ≈ 1 - p * cdf(inner, x)
    end

    # Conditional-on-report time moments / quantile are the inner delay's.
    @test mean(d) == mean(inner)
    @test var(d) == var(inner)
    @test quantile(d, 0.4) == quantile(inner, 0.4)

    # rand reports w.p. p (a time from inner), else `missing` (no event).
    rng = MersenneTwister(42)
    draws = [rand(rng, d) for _ in 1:20000]
    @test count(ismissing, draws) / 20000≈1 - p atol=0.02
    @test all(x -> ismissing(x) || x >= 0, draws)

    # The standalone scalar logpdf equals the resolve + NoEvent `:event`-branch
    # conditioned log-density (the convolution-marginal equivalence's per-record
    # honest model).
    r = resolve(:event => (inner, p), :none => (NoEvent(), 1 - p))
    @test CensoredDistributions._has_no_event(r)
    @test logpdf(d, 2.0) ≈
          CensoredDistributions._one_of_condition_logpdf(
        (p, 1 - p), inner, 2.0, 1)
end

@testitem "thin convolution equals the resolve + NoEvent count scaling" begin
    using CensoredDistributions, Distributions

    # The convolution-marginal equivalence: under `convolved` the
    # new thin STILL scales the branch's count series by p (epinowcast's
    # aggregate-count ascertainment is unchanged), and equals the resolve +
    # NoEvent node's `:event`-branch series.
    inner = Gamma(2.0, 1.0)
    p = 0.3
    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]

    plain = convolved(inner, series)
    thinned = convolved(thin(inner, p), series)
    @test thinned ≈ p .* plain

    r = resolve(:event => (inner, p), :none => (NoEvent(), 1 - p))
    @test thinned ≈ convolved(r, series; events = :event)
end

@testitem "forward transforms peel via free_leaf and rewrap" begin
    using CensoredDistributions, Distributions

    base = Gamma(2.0, 1.0)
    d = thin(interval_censored(base, 1.0), 0.3)
    # free_leaf peels the forward op AND the censoring to the inner delay.
    @test CensoredDistributions.free_leaf(d) === base
    # rewrap restores the forward op + censoring around a new inner delay.
    new_inner = Gamma(3.0, 0.5)
    rebuilt = CensoredDistributions.rewrap_leaf(d, new_inner)
    @test rebuilt isa CensoredDistributions.Transformed
    @test rebuilt.op.factor == 0.3
    @test CensoredDistributions.free_leaf(rebuilt) === new_inner
end

@testitem "forward ops apply to a series via _apply_forward_ops" begin
    using CensoredDistributions, Distributions

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
    using CensoredDistributions, Distributions

    # thin/cumulative are UnivariateDistributions, so they compose as leaves.
    p = compose((cases = thin(Gamma(1.5, 1.0), 0.3),
        deaths = thin(Gamma(3.0, 2.0), 0.01)))
    @test p isa CensoredDistributions.Parallel
    s = Sequential((Gamma(2.0, 1.0), thin(Gamma(1.5, 1.0), 0.3)))
    @test s isa CensoredDistributions.Sequential
end
