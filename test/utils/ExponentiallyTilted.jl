@testitem "ExponentiallyTilted distribution constructor and validation" begin
    using CensoredDistributions, Distributions, Test

    # Test valid constructors
    @test ExponentiallyTilted() isa ExponentiallyTilted{Float64}
    @test ExponentiallyTilted(0.0, 1.0, 0.0) isa ExponentiallyTilted{Float64}
    @test ExponentiallyTilted(0, 2, 1) isa ExponentiallyTilted{Int}
    @test ExponentiallyTilted(0.0, 2.0, 1.5) isa ExponentiallyTilted{Float64}

    # Test type promotion
    d = ExponentiallyTilted(0, 1.0, 2)
    @test d isa ExponentiallyTilted{Float64}
    @test params(d) == (0.0, 1.0, 2.0)

    # Test parameter validation
    @test_throws ArgumentError ExponentiallyTilted(1.0, 0.0, 0.0)  # min >= max
    @test_throws ArgumentError ExponentiallyTilted(1.0, 1.0, 0.0)  # min == max
    @test_throws ArgumentError ExponentiallyTilted(0.0, Inf, 0.0)  # infinite max
    @test_throws ArgumentError ExponentiallyTilted(-Inf, 1.0, 0.0) # infinite min
    @test_throws ArgumentError ExponentiallyTilted(0.0, 1.0, Inf)  # infinite r
    @test_throws ArgumentError ExponentiallyTilted(0.0, 1.0, NaN)  # NaN r
end

@testitem "ExponentiallyTilted distribution interface compliance" begin
    using CensoredDistributions, Distributions, Test

    # Test various parameter combinations
    test_dists = [
        ExponentiallyTilted(0.0, 1.0, 0.0),    # Uniform case
        ExponentiallyTilted(0.0, 1.0, 2.0),    # Growth case
        ExponentiallyTilted(0.0, 1.0, -1.5),   # Decay case
        ExponentiallyTilted(-1.0, 2.0, 0.5),   # Different bounds
        ExponentiallyTilted(1.0, 3.0, -0.8),   # Different bounds, decay
    ]

    for d in test_dists
        # Test parameters
        @test length(params(d)) == 3
        @test params(d) == (d.min, d.max, d.r)

        # Test support
        @test minimum(d) == d.min
        @test maximum(d) == d.max
        @test insupport(d, d.min) == true
        @test insupport(d, d.max) == true
        @test insupport(d, (d.min + d.max) / 2) == true
        @test insupport(d, d.min - 0.1) == false
        @test insupport(d, d.max + 0.1) == false

        # Test that methods exist and return reasonable values
        @test pdf(d, d.min) >= 0
        @test pdf(d, d.max) >= 0
        @test pdf(d, (d.min + d.max) / 2) >= 0
        @test cdf(d, d.min) >= 0
        @test cdf(d, d.max) <= 1
        @test isfinite(mean(d))
        @test var(d) >= 0
        @test std(d) >= 0
        @test isfinite(entropy(d))
        @test d.min <= mode(d) <= d.max
    end
end

@testitem "ExponentiallyTilted PDF properties" begin
    using CensoredDistributions, Distributions, Test

    # Test uniform case (r ≈ 0)
    d_uniform = ExponentiallyTilted(0.0, 2.0, 0.0)
    @test pdf(d_uniform, 0.5) ≈ 0.5  # 1/(2-0)
    @test pdf(d_uniform, 1.0) ≈ 0.5
    @test pdf(d_uniform, 1.5) ≈ 0.5

    # Test near-uniform case (small r)
    d_small_r = ExponentiallyTilted(0.0, 2.0, 1e-12)
    @test pdf(d_small_r, 1.0) ≈ 0.5 rtol=1e-6

    # Test growth case (r > 0)
    d_growth = ExponentiallyTilted(0.0, 1.0, 2.0)
    @test pdf(d_growth, 0.0) < pdf(d_growth, 1.0)  # Higher density at max

    # Test decay case (r < 0)
    d_decay = ExponentiallyTilted(0.0, 1.0, -2.0)
    @test pdf(d_decay, 0.0) > pdf(d_decay, 1.0)  # Higher density at min

    # Test outside support
    @test pdf(d_growth, -0.1) == 0.0
    @test pdf(d_growth, 1.1) == 0.0

    # Note: PDF integration to 1 is verified by the analytical implementation
    # and consistency with CDF boundary conditions tested elsewhere
end

@testitem "ExponentiallyTilted CDF properties" begin
    using CensoredDistributions, Distributions, Test

    test_dists = [
        ExponentiallyTilted(0.0, 1.0, 0.0),    # Uniform
        ExponentiallyTilted(0.0, 1.0, 2.0),    # Growth
        ExponentiallyTilted(0.0, 1.0, -1.5),   # Decay
    ]

    for d in test_dists
        # Test boundary conditions
        @test cdf(d, d.min - 1.0) == 0.0
        @test cdf(d, d.min) == 0.0
        @test cdf(d, d.max) == 1.0
        @test cdf(d, d.max + 1.0) == 1.0

        # Test monotonicity
        x1, x2, x3 = d.min + 0.1, (d.min + d.max) / 2, d.max - 0.1
        @test cdf(d, x1) <= cdf(d, x2) <= cdf(d, x3)

        # Test intermediate values are in [0, 1]
        @test 0 <= cdf(d, (d.min + d.max) / 2) <= 1
    end

    # Test uniform case specifically
    d_uniform = ExponentiallyTilted(0.0, 2.0, 0.0)
    @test cdf(d_uniform, 0.5) ≈ 0.25  # (0.5 - 0) / (2 - 0)
    @test cdf(d_uniform, 1.0) ≈ 0.5
    @test cdf(d_uniform, 1.5) ≈ 0.75
end

@testitem "ExponentiallyTilted quantile function properties" begin
    using CensoredDistributions, Distributions, Test

    test_dists = [
        ExponentiallyTilted(0.0, 1.0, 0.0),    # Uniform
        ExponentiallyTilted(0.0, 1.0, 2.0),    # Growth
        ExponentiallyTilted(0.0, 1.0, -1.5),   # Decay
        ExponentiallyTilted(-1.0, 3.0, 1.0),   # Different bounds
    ]

    for d in test_dists
        # Test boundary conditions
        @test quantile(d, 0.0) == d.min
        @test quantile(d, 1.0) == d.max

        # Test monotonicity
        @test quantile(d, 0.25) <= quantile(d, 0.5) <= quantile(d, 0.75)

        # Test quantile is inverse of CDF
        test_points = [0.1, 0.25, 0.5, 0.75, 0.9]
        for p in test_points
            x = quantile(d, p)
            @test cdf(d, x) ≈ p rtol=1e-12
        end

        # Test CDF is inverse of quantile
        test_x = [d.min + 0.1, (d.min + d.max) / 2, d.max - 0.1]
        for x in test_x
            p = cdf(d, x)
            @test quantile(d, p) ≈ x rtol=1e-12
        end

        # Test invalid probabilities
        @test_throws DomainError quantile(d, -0.1)
        @test_throws DomainError quantile(d, 1.1)
    end

    # Test uniform case specifically
    d_uniform = ExponentiallyTilted(0.0, 2.0, 0.0)
    @test quantile(d_uniform, 0.25) ≈ 0.5
    @test quantile(d_uniform, 0.5) ≈ 1.0
    @test quantile(d_uniform, 0.75) ≈ 1.5
end

@testitem "ExponentiallyTilted sampling and random generation" begin
    using CensoredDistributions, Distributions, Test, Random

    Random.seed!(123)

    test_dists = [
        ExponentiallyTilted(0.0, 1.0, 0.0),    # Uniform
        ExponentiallyTilted(0.0, 1.0, 1.0),    # Growth
        ExponentiallyTilted(0.0, 1.0, -1.0),   # Decay
    ]

    n_samples = 10000

    for d in test_dists
        # Generate samples
        samples = [rand(d) for _ in 1:n_samples]

        # Test all samples are in support
        @test all(d.min <= x <= d.max for x in samples)

        # Test empirical moments converge to theoretical
        empirical_mean = sum(samples) / n_samples
        theoretical_mean = mean(d)
        @test empirical_mean ≈ theoretical_mean rtol=0.05

        empirical_var = sum((x - empirical_mean)^2 for x in samples) / (n_samples - 1)
        theoretical_var = var(d)
        @test empirical_var ≈ theoretical_var rtol=0.1

        # Test with explicit RNG
        rng = MersenneTwister(456)
        sample1 = rand(rng, d)
        @test d.min <= sample1 <= d.max

        # Test reproducibility
        rng1 = MersenneTwister(789)
        rng2 = MersenneTwister(789)
        @test rand(rng1, d) == rand(rng2, d)
    end
end

@testitem "ExponentiallyTilted moments and statistics" begin
    using CensoredDistributions, Distributions, Test

    # Test uniform case
    d_uniform = ExponentiallyTilted(0.0, 2.0, 0.0)
    @test mean(d_uniform) ≈ 1.0  # (0 + 2) / 2
    @test var(d_uniform) ≈ 4.0 / 12  # (2 - 0)² / 12
    @test std(d_uniform) ≈ sqrt(4.0 / 12)
    @test mode(d_uniform) ≈ 1.0  # Midpoint for uniform
    @test entropy(d_uniform) ≈ log(2.0)  # log(max - min)

    # Test growth case properties
    d_growth = ExponentiallyTilted(0.0, 1.0, 2.0)
    @test mean(d_growth) > 0.5  # Should be > midpoint for growth
    @test mode(d_growth) == 1.0  # Mode at max for r > 0

    # Test decay case properties
    d_decay = ExponentiallyTilted(0.0, 1.0, -2.0)
    @test mean(d_decay) < 0.5  # Should be < midpoint for decay
    @test mode(d_decay) == 0.0  # Mode at min for r < 0

    # Test variance is always positive
    test_dists = [
        ExponentiallyTilted(0.0, 1.0, 0.0),
        ExponentiallyTilted(0.0, 1.0, 2.0),
        ExponentiallyTilted(0.0, 1.0, -1.5),
        ExponentiallyTilted(-2.0, 3.0, 0.8)
    ]

    for d in test_dists
        @test var(d) >= 0
        @test std(d) >= 0
        @test isfinite(mean(d))
        @test isfinite(var(d))
        @test isfinite(entropy(d))
    end
end

@testitem "ExponentiallyTilted log functions" begin
    using CensoredDistributions, Distributions, Test

    test_dists = [
        ExponentiallyTilted(0.0, 1.0, 0.0),    # Uniform
        ExponentiallyTilted(0.0, 1.0, 2.0),    # Growth
        ExponentiallyTilted(0.0, 1.0, -1.5)   # Decay
    ]

    for d in test_dists
        test_points = [d.min, d.min + 0.1, (d.min + d.max) / 2, d.max - 0.1, d.max]

        for x in test_points
            # Test logpdf consistency with pdf
            if pdf(d, x) > 0
                @test logpdf(d, x) ≈ log(pdf(d, x)) rtol=1e-12
            else
                @test logpdf(d, x) == -Inf
            end

            # Test logcdf consistency with cdf
            if cdf(d, x) > 0
                @test logcdf(d, x) ≈ log(cdf(d, x)) rtol=1e-12
            else
                @test logcdf(d, x) == -Inf
            end
        end

        # Test outside support
        @test logpdf(d, d.min - 0.1) == -Inf
        @test logpdf(d, d.max + 0.1) == -Inf
        @test logcdf(d, d.min - 0.1) == -Inf
        @test logcdf(d, d.max + 0.1) ≈ 0.0
    end
end

@testitem "ExponentiallyTilted limiting behavior near r=0" begin
    using CensoredDistributions, Distributions, Test

    # Test convergence to uniform as r → 0
    d_base = ExponentiallyTilted(0.0, 2.0, 0.0)  # True uniform

    small_r_values = [1e-11, 1e-12, 1e-13, 1e-14]
    test_points = [0.5, 1.0, 1.5]

    for r in small_r_values
        d_small = ExponentiallyTilted(0.0, 2.0, r)

        for x in test_points
            @test pdf(d_small, x) ≈ pdf(d_base, x) rtol=1e-10
            @test cdf(d_small, x) ≈ cdf(d_base, x) rtol=1e-10
            @test logpdf(d_small, x) ≈ logpdf(d_base, x) rtol=1e-10
            @test logcdf(d_small, x) ≈ logcdf(d_base, x) rtol=1e-10
        end

        @test mean(d_small) ≈ mean(d_base) rtol=1e-10
        @test var(d_small) ≈ var(d_base) rtol=1e-10
        @test entropy(d_small) ≈ entropy(d_base) rtol=1e-10

        # Test quantiles
        for p in [0.25, 0.5, 0.75]
            @test quantile(d_small, p) ≈ quantile(d_base, p) rtol=1e-10
        end
    end
end

@testitem "ExponentiallyTilted extreme parameter values" begin
    using CensoredDistributions, Distributions, Test

    # Test with large positive r
    d_large_pos = ExponentiallyTilted(0.0, 1.0, 10.0)
    @test isfinite(mean(d_large_pos))
    @test isfinite(var(d_large_pos))
    @test var(d_large_pos) >= 0
    @test mode(d_large_pos) == 1.0

    # Test with large negative r
    d_large_neg = ExponentiallyTilted(0.0, 1.0, -10.0)
    @test isfinite(mean(d_large_neg))
    @test isfinite(var(d_large_neg))
    @test var(d_large_neg) >= 0
    @test mode(d_large_neg) == 0.0

    # Test with very small interval
    d_small_interval = ExponentiallyTilted(0.0, 1e-6, 1.0)
    @test isfinite(mean(d_small_interval))
    @test isfinite(var(d_small_interval))
    @test var(d_small_interval) >= 0

    # Test with large interval
    d_large_interval = ExponentiallyTilted(0.0, 1000.0, 0.1)
    @test isfinite(mean(d_large_interval))
    @test isfinite(var(d_large_interval))
    @test var(d_large_interval) >= 0
end