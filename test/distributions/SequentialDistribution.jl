@testitem "SequentialDistribution constructor and validation" begin
    using Distributions

    delays = [Gamma(2.0, 1.0), Gamma(1.0, 1.0), LogNormal(0.5, 0.4)]

    # Data-free: the struct holds delays and censoring parameters only.
    d = sequential_distribution(delays)
    @test d isa CensoredDistributions.SequentialDistribution
    @test length(d) == 3
    @test d.delays == Tuple(delays)
    @test d.primary_event === nothing
    @test d.interval === nothing
    @test d.horizon === nothing

    # Tuple constructor is equivalent.
    dt = sequential_distribution(Tuple(delays))
    @test dt.delays == d.delays

    # Errors: wrong observation-vector length, fewer than two observed events.
    @test_throws DimensionMismatch logpdf(d, [0.0, 1.0, 3.0])
    @test_throws ArgumentError logpdf(d, [0.0, missing, missing, missing])
end

@testitem "SequentialDistribution logpdf dispatches on missingness" begin
    using Distributions

    D1 = Gamma(2.0, 1.0)
    D2 = Gamma(1.5, 1.0)
    D3 = LogNormal(0.5, 0.4)
    d = sequential_distribution([D1, D2, D3])

    # Observed intermediate events -> three independent factors.
    @test logpdf(d, [0.0, 1.0, 3.0, 5.0]) ≈
          logpdf(D1, 1.0) + logpdf(D2, 2.0) + logpdf(D3, 2.0)

    # Missing intermediate event E1 -> first gap marginalises D1+D2 by
    # convolution, second gap is the D3 factor.
    seg1 = generic_convolve(D1, D2)
    for (g1, g2) in [(3.0, 2.0), (4.5, 1.2)]
        obs = [0.0, missing, g1, g1 + g2]
        @test logpdf(d, obs) ≈ logpdf(seg1, g1) + logpdf(D3, g2) atol=1e-10
    end

    # pdf is exp(logpdf).
    @test pdf(d, [0.0, 1.0, 3.0, 5.0]) ≈ exp(logpdf(d, [0.0, 1.0, 3.0, 5.0]))
end

@testitem "SequentialDistribution all-unobserved equals double-censor" begin
    using Distributions

    D1 = Gamma(2.0, 1.0)
    D2 = Gamma(1.5, 1.0)
    D3 = LogNormal(0.5, 0.4)

    # Every intermediate unobserved with full censoring equals
    # double_interval_censored of the continuous convolution.
    d = sequential_distribution(
        [D1, D2, D3]; primary_event = Uniform(0, 1), horizon = 15.0,
        interval = 1.0)
    ref = double_interval_censored(
        generic_convolve(D1, D2, D3);
        primary_event = Uniform(0, 1), upper = 15.0, interval = 1.0)
    for g in [3.0, 5.0, 8.0]
        @test logpdf(d, [0.0, missing, missing, g]) ≈ logpdf(ref, g) atol=1e-8
    end
end

@testitem "SequentialDistribution endpoint censoring per segment" begin
    using Distributions

    D1 = Gamma(2.0, 1.0)
    D2 = Gamma(1.5, 1.0)
    primary = Uniform(0.0, 1.0)

    # Interval censoring applied to the marginalised run as a unit.
    di = sequential_distribution([D1, D2]; interval = 1.0)
    refi = interval_censored(generic_convolve(D1, D2), 1.0)
    @test logpdf(di, [0.0, missing, 3.0]) ≈ logpdf(refi, 3.0) atol=1e-8

    # Primary censoring at the origin only.
    dp = sequential_distribution([D1, D2]; primary_event = primary)
    refp = double_interval_censored(
        generic_convolve(D1, D2); primary_event = primary)
    @test logpdf(dp, [0.0, missing, 4.0]) ≈ logpdf(refp, 4.0) atol=1e-8

    # Observed intermediate -> two factors; only the first carries primary
    # (origin) censoring, the second only interval censoring.
    d = sequential_distribution(
        [D1, D2]; primary_event = primary, interval = 1.0)
    f1 = double_interval_censored(D1; primary_event = primary, interval = 1.0)
    f2 = interval_censored(D2, 1.0)
    @test logpdf(d, [0.0, 2.0, 5.0]) ≈ logpdf(f1, 2.0) + logpdf(f2, 3.0) atol=1e-8
end

@testitem "SequentialDistribution truncated(d, horizon) per-record denominator" begin
    using Distributions

    D1 = Gamma(2.0, 1.0)
    D2 = Gamma(1.5, 1.0)
    d = sequential_distribution([D1, D2])
    dt = truncated(d, 8.0)

    # truncated returns a SequentialDistribution carrying the horizon.
    @test dt isa CensoredDistributions.SequentialDistribution
    @test dt.horizon == 8.0

    # Missing intermediate -> convolved-chain denominator: the whole chain is
    # one right-truncated convolution.
    refc = truncated(generic_convolve(D1, D2); upper = 8.0)
    @test logpdf(dt, [0.0, missing, 4.0]) ≈ logpdf(refc, 4.0) atol=1e-8

    # Observed intermediate -> single-delay denominators: each segment is its
    # own right-truncated delay.
    refs = logpdf(truncated(D1; upper = 8.0), 2.0) +
           logpdf(truncated(D2; upper = 8.0), 2.0)
    @test logpdf(dt, [0.0, 2.0, 4.0]) ≈ refs atol=1e-8
end

@testitem "SequentialDistribution marginalises continuous core of components" begin
    using Distributions

    D1 = Gamma(2.0, 1.0)
    D2 = Gamma(1.5, 1.0)

    # A censored component inside an unobserved run contributes only its
    # continuous delay to the convolution (never the discrete object).
    c1 = double_interval_censored(D1; interval = 1.0)
    c2 = double_interval_censored(D2; interval = 1.0)
    d = sequential_distribution(
        [c1, c2]; primary_event = Uniform(0, 1), interval = 1.0)
    ref = double_interval_censored(
        generic_convolve(D1, D2); primary_event = Uniform(0, 1),
        interval = 1.0)
    for g in [3.0, 5.0, 7.0]
        @test logpdf(d, [0.0, missing, g]) ≈ logpdf(ref, g) atol=1e-8
    end
end

@testitem "SequentialDistribution rand full event-time path" begin
    using Distributions, Random, Statistics

    D1 = Gamma(2.0, 1.0)   # mean 2
    D2 = Gamma(1.5, 1.0)   # mean 1.5
    D3 = LogNormal(0.0, 0.5)
    d = sequential_distribution([D1, D2, D3])

    # rand returns the full event-time path E0..E3 (length k + 1).
    path = rand(MersenneTwister(1), d)
    @test length(path) == 4
    @test path[1] == 0.0
    @test issorted(path)            # cumulative non-negative delays

    # Event-time increments match the generative delay means.
    rng = MersenneTwister(7)
    paths = [rand(rng, d) for _ in 1:5000]
    @test mean(p[2] - p[1] for p in paths) ≈ mean(D1) atol=0.1
    @test mean(p[3] - p[2] for p in paths) ≈ mean(D2) atol=0.1
    @test mean(p[4] - p[3] for p in paths) ≈ mean(D3) atol=0.1
end

@testitem "SequentialDistribution interface helpers" begin
    using Distributions

    delays = [Gamma(2.0, 1.0), Gamma(1.5, 1.0), LogNormal(0.5, 0.4)]
    d = sequential_distribution(delays)

    @test length(d) == 3
    @test length(params(d)) == 3
    @test eltype(d) == Float64
end

@testitem "SequentialDistribution Monte-Carlo: unobserved-run pdf/cdf" begin
    using Distributions, Random, Statistics

    # Validate the marginalised (unobserved-intermediate) pdf/cdf against a
    # Monte-Carlo estimate from the shared continuous latent path.
    D1 = Gamma(2.0, 1.0)
    D2 = Gamma(1.5, 1.0)
    seg = generic_convolve(D1, D2)
    d = sequential_distribution([D1, D2])

    rng = MersenneTwister(2024)
    N = 200_000
    sums = [rand(rng, D1) + rand(rng, D2) for _ in 1:N]

    # The unobserved-intermediate gap density equals the convolution density;
    # check logpdf of the chain against the empirical pdf/cdf.
    for g in [2.0, 4.0, 6.0]
        mc = count(<=(g), sums) / N
        @test cdf(seg, g) ≈ mc atol=0.01
    end
    g0, h = 4.0, 0.25
    mc_pdf = count(s -> g0 - h <= s < g0 + h, sums) / (N * 2h)
    @test exp(logpdf(d, [0.0, missing, g0])) ≈ mc_pdf atol=0.02
end

@testitem "SequentialDistribution Monte-Carlo: rand endpoint marginal" begin
    using Distributions, Random, Statistics

    # The endpoint event time of the rand path equals the convolution of the
    # delays (shared continuous latents). Compare moments to the analytic
    # convolution moments.
    D1 = Gamma(2.0, 1.0)   # mean 2, var 2
    D2 = Gamma(1.5, 1.0)   # mean 1.5, var 1.5
    d = sequential_distribution([D1, D2])

    rng = MersenneTwister(11)
    N = 200_000
    endpoints = [rand(rng, d)[end] for _ in 1:N]
    @test mean(endpoints) ≈ mean(D1) + mean(D2) atol=0.02
    @test var(endpoints) ≈ var(D1) + var(D2) atol=0.05
end

@testitem "SequentialDistribution Monte-Carlo: observed-intermediate factors" begin
    using Distributions, Random, Statistics

    # Observed-intermediate (factorised) case: each event-time increment
    # matches its generative delay.
    D1 = Gamma(2.0, 1.0)
    D2 = LogNormal(0.0, 0.5)
    d = sequential_distribution([D1, D2])

    rng = MersenneTwister(3)
    N = 100_000
    paths = [rand(rng, d) for _ in 1:N]
    inc1 = [p[2] - p[1] for p in paths]
    inc2 = [p[3] - p[2] for p in paths]

    @test mean(inc1) ≈ mean(D1) atol=0.02
    @test mean(inc2) ≈ mean(D2) atol=0.02
    @test count(<=(2.0), inc1) / N ≈ cdf(D1, 2.0) atol=0.01
end

# AD through `logpdf` (with the `Union{Missing}` observation vector passed as
# a `Constant`) is covered by the `SequentialDistribution missing
# intermediate` scenario in `test/ADFixtures`, run across every backend by
# the dedicated `test/ad` environment.
