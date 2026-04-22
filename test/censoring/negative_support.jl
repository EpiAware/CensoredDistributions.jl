@testitem "Negative-support delays: PrimaryCensored" begin
    using Distributions
    using Random

    # Signed-support delay distributions
    delays = [Normal(0, 1), Logistic(0, 1), Cauchy(0, 1)]
    primary_events = [Uniform(0, 1), truncated(Normal(0, 2), -5, 5)]

    for dist in delays
        for primary in primary_events
            d = primary_censored(dist, primary)

            # Support
            @test minimum(d) == -Inf
            @test maximum(d) == Inf

            # Tail limits
            @test cdf(d, -100.0) < 0.01
            @test cdf(d, 100.0) > 0.99

            # Monotonicity
            xs = -10.0:1.0:10.0
            cdfs = [cdf(d, x) for x in xs]
            @test issorted(cdfs)

            # PMF as CDF difference (consistency check implicitly via logpdf structure)
            # logpdf is computed via difference in CDFs for PrimaryCensored
            for x in -5.0:2.0:5.0
                @test exp(logpdf(d, x)) >= 0.0
            end

            # Sampling
            samps = rand(MersenneTwister(42), d, 100)
            @test all(isfinite, samps)

            # Quantile round-trips
            # Need to avoid extreme quantiles for Cauchy
            ps = [0.1, 0.5, 0.9]
            for p in ps
                q = quantile(d, p)
                @test isfinite(q)
                @test cdf(d, q) ≈ p atol=1e-3
            end
        end
    end
end

@testitem "Negative-support delays: IntervalCensored" begin
    using Distributions

    delays = [Normal(0, 1), Logistic(0, 1)]

    for dist in delays
        d = interval_censored(dist, 1.0)

        # Support
        @test minimum(d) == -Inf
        @test maximum(d) == Inf

        # Tail limits
        @test cdf(d, -100.0) < 0.01
        @test cdf(d, 100.0) > 0.99

        # Monotonicity
        xs = -10.0:1.0:10.0
        cdfs = [cdf(d, x) for x in xs]
        @test issorted(cdfs)

        # Quantile round-trips
        ps = [0.1, 0.5, 0.9]
        for p in ps
            q = quantile(d, p)
            @test isfinite(q)
        end
    end
end

@testitem "Negative-support delays: DoubleIntervalCensored" begin
    using Distributions
    using Random

    delays = [Normal(0, 1), Logistic(0, 1)]

    for dist in delays
        # Un-truncated (lower=nothing)
        d = double_interval_censored(dist; upper = 10.0, interval = 1.0,
            primary_event = Uniform(0.0, 1.0), lower = nothing)

        # Sampling within [L, D) is not natively bounded below for lower=nothing
        # but the sampler should not fail.
        samps = rand(MersenneTwister(42), d, 100)
        @test all(isfinite, samps)
        # Should be strictly < D (D = 10.0)
        @test all(x -> x < 10.0, samps)
    end
end
