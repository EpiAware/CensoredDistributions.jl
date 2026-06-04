@testitem "SequentialDistribution constructor and validation" begin
    using Distributions

    delays = [Gamma(2.0, 1.0), Gamma(1.0, 1.0), LogNormal(0.5, 0.4)]

    # Middle event unobserved -> two segments: a convolution run and a
    # bare-delay factor.
    d = sequential_distribution(delays, [0.0, missing, 3.0, 5.0])
    @test d isa CensoredDistributions.SequentialDistribution
    @test length(d) == 2
    @test d.segments[1] isa CensoredDistributions.Convolved
    @test d.segments[2] isa LogNormal

    # Value-pattern and boolean-design constructors give the same structure.
    db = sequential_distribution(delays, [true, false, true, true])
    @test length(db) == length(d)
    @test typeof(db.segments) == typeof(d.segments)

    # Errors: wrong length, fewer than two observed events.
    @test_throws ArgumentError sequential_distribution(delays, [0.0, 1.0, 3.0])
    @test_throws ArgumentError sequential_distribution(
        delays, [0.0, missing, missing, missing])
    @test_throws ArgumentError sequential_distribution(
        delays, [true, false, true])
end

@testitem "SequentialDistribution logpdf: observed factorises" begin
    using Distributions

    # Fully observed chain: every gap is its own delay, joint logpdf is the
    # sum of the per-delay logpdfs.
    delays = [Gamma(2.0, 1.0), Gamma(1.5, 1.0), LogNormal(0.5, 0.4)]
    d = sequential_distribution(delays, [0.0, 1.0, 3.0, 5.0])

    gaps = [1.2, 2.4, 1.8]
    @test logpdf(d, gaps) ≈ sum(logpdf(delays[i], gaps[i]) for i in 1:3)
    @test pdf(d, gaps) ≈ exp(logpdf(d, gaps))
end

@testitem "SequentialDistribution logpdf: unobserved run marginalises" begin
    using Distributions

    D1 = Gamma(2.0, 1.0)
    D2 = Gamma(1.5, 1.0)
    D3 = LogNormal(0.5, 0.4)

    # E1 unobserved -> first segment convolves D1,D2 (gap E0->E2), second
    # is D3 (gap E2->E3). The convolution marginal must match the direct
    # convolution and factorise from the observed-adjacent delay.
    d = sequential_distribution([D1, D2, D3], [0.0, missing, 3.0, 5.0])
    seg1 = generic_convolve(D1, D2)
    for (g1, g2) in [(3.0, 2.0), (4.5, 1.2), (2.0, 3.3)]
        @test logpdf(d, [g1, g2]) ≈ logpdf(seg1, g1) + logpdf(D3, g2) atol=1e-10
    end

    # All intermediates unobserved -> a single convolution of every delay.
    da = sequential_distribution([D1, D2, D3], [0.0, missing, missing, 6.0])
    seg = generic_convolve(D1, D2, D3)
    for g in [4.0, 6.0, 8.5]
        @test logpdf(da, [g]) ≈ logpdf(seg, g) atol=1e-10
    end
end

@testitem "SequentialDistribution logpdf: all-unobserved equals double-censor" begin
    using Distributions

    D1 = Gamma(2.0, 1.0)
    D2 = Gamma(1.5, 1.0)
    D3 = LogNormal(0.5, 0.4)

    # Equivalence required by the spec: an all-unobserved endpoint with full
    # censoring equals double_interval_censored of the continuous
    # convolution, censored at the endpoint.
    d = sequential_distribution(
        [D1, D2, D3], [true, false, false, true];
        primary_event = Uniform(0, 1), upper = 15.0, interval = 1.0)
    ref = double_interval_censored(
        generic_convolve(D1, D2, D3);
        primary_event = Uniform(0, 1), upper = 15.0, interval = 1.0)
    for g in [3.0, 5.0, 8.0]
        @test logpdf(d, [g]) ≈ logpdf(ref, g) atol=1e-8
    end
end

@testitem "SequentialDistribution endpoint censoring per segment" begin
    using Distributions

    D1 = Gamma(2.0, 1.0)
    D2 = Gamma(1.5, 1.0)
    primary = Uniform(0.0, 1.0)

    # Truncation applied to the marginalised run as a unit.
    dt = sequential_distribution([D1, D2], [0.0, missing, 4.0]; upper = 8.0)
    reft = truncated(generic_convolve(D1, D2); upper = 8.0)
    @test logpdf(dt, [5.0]) ≈ logpdf(reft, 5.0) atol=1e-8

    # Interval censoring applied to the marginalised run as a unit.
    di = sequential_distribution([D1, D2], [0.0, missing, 4.0]; interval = 1.0)
    refi = interval_censored(generic_convolve(D1, D2), 1.0)
    @test logpdf(di, [3.0]) ≈ logpdf(refi, 3.0) atol=1e-8

    # Primary censoring only at the origin end of the first segment.
    dp = sequential_distribution(
        [D1, D2], [0.0, missing, 4.0]; primary_event = primary)
    refp = double_interval_censored(
        generic_convolve(D1, D2); primary_event = primary)
    @test logpdf(dp, [4.0]) ≈ logpdf(refp, 4.0) atol=1e-8
end

@testitem "SequentialDistribution observed-adjacent gives separate factors" begin
    using Distributions

    D1 = Gamma(2.0, 1.0)
    D2 = Gamma(1.5, 1.0)
    primary = Uniform(0.0, 1.0)

    # Both intermediates observed -> two double-censored FACTORS, not a
    # convolution. Only the first factor carries primary (origin) censoring;
    # the second carries only truncation + interval censoring.
    d = sequential_distribution(
        [D1, D2], [0.0, 2.0, 5.0]; primary_event = primary,
        upper = 10.0, interval = 1.0)
    f1 = double_interval_censored(
        D1; primary_event = primary, upper = 10.0, interval = 1.0)
    f2 = interval_censored(truncated(D2; upper = 10.0), 1.0)
    @test logpdf(d, [2.0, 3.0]) ≈ logpdf(f1, 2.0) + logpdf(f2, 3.0) atol=1e-8
end

@testitem "SequentialDistribution marginalises continuous core of components" begin
    using Distributions

    D1 = Gamma(2.0, 1.0)
    D2 = Gamma(1.5, 1.0)

    # A censored component inside an unobserved run contributes only its
    # continuous delay to the convolution (never the discrete object), so
    # passing double_interval_censored components reduces to the continuous
    # convolution censored at the endpoint.
    c1 = double_interval_censored(D1; interval = 1.0)
    c2 = double_interval_censored(D2; interval = 1.0)
    d = sequential_distribution(
        [c1, c2], [true, false, true]; primary_event = Uniform(0, 1),
        interval = 1.0)
    ref = double_interval_censored(
        generic_convolve(D1, D2); primary_event = Uniform(0, 1),
        interval = 1.0)
    for g in [3.0, 5.0, 7.0]
        @test logpdf(d, [g]) ≈ logpdf(ref, g) atol=1e-8
    end

    # A truncated component in an unobserved run likewise contributes its
    # untruncated continuous core.
    dt = sequential_distribution(
        [truncated(D1, 0, 5), D2], [0.0, missing, 4.0])
    @test logpdf(dt, [3.0]) ≈ logpdf(generic_convolve(D1, D2), 3.0) atol=1e-8
end

@testitem "SequentialDistribution escape hatch sums pre-built segments" begin
    using Distributions

    # Pre-built segments are summed directly, retaining their censoring.
    run = generic_convolve(Gamma(2.0, 1.0), Gamma(1.0, 1.0))
    factor = double_interval_censored(LogNormal(0.5, 0.4); interval = 1.0)
    d = sequential_distribution([run, factor])
    @test logpdf(d, [3.0, 2.0]) ≈ logpdf(run, 3.0) + logpdf(factor, 2.0)

    # Tuple form is equivalent to the vector form.
    dt = sequential_distribution((run, factor))
    @test logpdf(dt, [3.0, 2.0]) == logpdf(d, [3.0, 2.0])
end

@testitem "SequentialDistribution nests recursively" begin
    using Distributions

    # A segment may itself be a SequentialDistribution collapsed to one gap
    # (a single-segment chain is a UnivariateDistribution-like factor via
    # its escape-hatch convolution). Demonstrate recursion: the inner chain's
    # convolution feeds the outer chain as a component, and logpdf recurses
    # through the component interface.
    inner_run = generic_convolve(Gamma(2.0, 1.0), Gamma(1.0, 1.0))
    outer = sequential_distribution([inner_run, LogNormal(0.5, 0.4)])
    @test logpdf(outer, [3.0, 2.0]) ≈
          logpdf(inner_run, 3.0) + logpdf(LogNormal(0.5, 0.4), 2.0)

    # Build the inner run through the constructor (unobserved sub-chain) and
    # reuse its single segment as an outer component, proving the segment
    # composition is itself a valid component.
    inner = sequential_distribution(
        [Gamma(2.0, 1.0), Gamma(1.0, 1.0)], [0.0, missing, 3.0])
    outer2 = sequential_distribution([inner.segments[1], LogNormal(0.5, 0.4)])
    @test logpdf(outer2, [3.0, 2.0]) ≈ logpdf(outer, [3.0, 2.0]) atol=1e-10
end

@testitem "SequentialDistribution interface helpers" begin
    using Distributions

    delays = [Gamma(2.0, 1.0), Gamma(1.5, 1.0), LogNormal(0.5, 0.4)]
    d = sequential_distribution(delays, [0.0, missing, 3.0, 5.0])

    @test insupport(d, [1.0, 1.0])
    @test !insupport(d, [-1.0, 1.0])
    @test !insupport(d, [1.0])
    @test_throws DimensionMismatch logpdf(d, [1.0, 2.0, 3.0])
    @test length(params(d)) == 2
end

@testitem "SequentialDistribution Monte-Carlo: unobserved-run pdf/cdf" begin
    using Distributions, Random, Statistics

    # Validate the marginalised (unobserved-intermediate) pdf/cdf against a
    # Monte-Carlo estimate from the shared continuous latent path. Large N
    # with a tolerance scaled to the MC standard error, matching the
    # Convolved tests' style.
    D1 = Gamma(2.0, 1.0)
    D2 = Gamma(1.5, 1.0)
    d = sequential_distribution([D1, D2], [0.0, missing, 4.0])

    rng = MersenneTwister(2024)
    N = 200_000
    # Shared continuous latent path: origin at 0, latent E1, endpoint E2.
    sums = [rand(rng, D1) + rand(rng, D2) for _ in 1:N]

    # CDF at several points via the empirical CDF of the endpoint gap.
    for g in [2.0, 4.0, 6.0]
        mc = count(<=(g), sums) / N
        @test cdf(d.segments[1], g) ≈ mc atol=0.01
    end
    # PDF via a histogram bin.
    g0, h = 4.0, 0.25
    mc_pdf = count(s -> g0 - h <= s < g0 + h, sums) / (N * 2h)
    @test pdf(d.segments[1], g0) ≈ mc_pdf atol=0.02
end

@testitem "SequentialDistribution Monte-Carlo: rand endpoint marginal" begin
    using Distributions, Random, Statistics

    # The endpoint-only marginal of rand must match the marginal-convolution
    # distribution (same shared continuous latents). Compare rand-sample
    # moments to the analytic convolution moments.
    D1 = Gamma(2.0, 1.0)   # mean 2, var 2
    D2 = Gamma(1.5, 1.0)   # mean 1.5, var 1.5
    d = sequential_distribution([D1, D2], [0.0, missing, 4.0])

    rng = MersenneTwister(11)
    N = 200_000
    draws = [rand(rng, d)[1] for _ in 1:N]
    @test mean(draws) ≈ mean(D1) + mean(D2) atol=0.02
    @test var(draws) ≈ var(D1) + var(D2) atol=0.05
end

@testitem "SequentialDistribution Monte-Carlo: observed-intermediate factors" begin
    using Distributions, Random, Statistics

    # Observed-intermediate (factorised) case: each segment's rand and pdf
    # match its generative delay, validated by MC moments and an empirical
    # CDF check.
    D1 = Gamma(2.0, 1.0)
    D2 = LogNormal(0.0, 0.5)
    d = sequential_distribution([D1, D2], [0.0, 1.0, 3.0])

    rng = MersenneTwister(3)
    N = 100_000
    draws = [rand(rng, d) for _ in 1:N]
    g1 = [v[1] for v in draws]
    g2 = [v[2] for v in draws]

    @test mean(g1) ≈ mean(D1) atol=0.02
    @test mean(g2) ≈ mean(D2) atol=0.02
    # Empirical CDF of the first factor matches the delay CDF.
    @test count(<=(2.0), g1) / N ≈ cdf(D1, 2.0) atol=0.01
end

# AD through `logpdf` is covered by the `SequentialDistribution Gamma+Gamma
# missing intermediate` scenario in `test/ADFixtures`, run across every
# backend by the dedicated `test/ad` environment (ForwardDiff and the
# other AD backends are not deps of the main test environment).
