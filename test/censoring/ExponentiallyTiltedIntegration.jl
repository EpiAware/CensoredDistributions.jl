@testitem "ExponentiallyTilted integration with PrimaryCensored" begin
    using CensoredDistributions, Distributions, Test
    
    # Test basic integration with common delay distributions
    delay_dists = [
        LogNormal(1.5, 0.75),
        Gamma(2.0, 1.5),
        Weibull(2.0, 3.0),
        Exponential(2.0)
    ]
    
    tilted_priors = [
        ExponentiallyTilted(0.0, 1.0, 0.0),    # Uniform (baseline)
        ExponentiallyTilted(0.0, 1.0, 1.0),    # Growth
        ExponentiallyTilted(0.0, 1.0, -0.5),   # Decay
        ExponentiallyTilted(0.0, 2.0, 0.8),    # Different window, growth
    ]
    
    for delay_dist in delay_dists
        for prior in tilted_priors
            # Test construction
            pc_dist = primary_censored(delay_dist, prior)
            @test pc_dist isa PrimaryCensored
            @test pc_dist.dist === delay_dist
            @test pc_dist.primary_event === prior
            
            # Test basic functionality
            @test isfinite(mean(pc_dist))
            @test var(pc_dist) >= 0
            @test std(pc_dist) >= 0
            
            # Test PDF/CDF evaluation at various points
            test_points = [0.1, 0.5, 1.0, 2.0, 5.0]
            for x in test_points
                @test pdf(pc_dist, x) >= 0
                @test 0 <= cdf(pc_dist, x) <= 1
                @test isfinite(logpdf(pc_dist, x))
                @test isfinite(logcdf(pc_dist, x))
            end
            
            # Test sampling
            samples = [rand(pc_dist) for _ in 1:100]
            @test all(s >= 0 for s in samples)  # All samples should be non-negative
            @test length(unique(samples)) > 1   # Should have variation
        end
    end
end

@testitem "ExponentiallyTilted vs Uniform primary event comparison" begin
    using CensoredDistributions, Distributions, Test
    
    # Compare ExponentiallyTilted with r≈0 to Uniform primary event
    delay_dist = LogNormal(1.5, 0.75)
    
    # Nearly uniform tilted distribution
    tilted_uniform = ExponentiallyTilted(0.0, 1.0, 1e-12)
    pc_tilted = primary_censored(delay_dist, tilted_uniform)
    
    # True uniform distribution
    uniform_prior = Uniform(0.0, 1.0)
    pc_uniform = primary_censored(delay_dist, uniform_prior)
    
    # Test that they give very similar results
    test_points = [0.1, 0.5, 1.0, 2.0, 5.0]
    for x in test_points
        @test pdf(pc_tilted, x) ≈ pdf(pc_uniform, x) rtol=1e-6
        @test cdf(pc_tilted, x) ≈ cdf(pc_uniform, x) rtol=1e-6
        @test logpdf(pc_tilted, x) ≈ logpdf(pc_uniform, x) rtol=1e-6
        @test logcdf(pc_tilted, x) ≈ logcdf(pc_uniform, x) rtol=1e-6
    end
    
    @test mean(pc_tilted) ≈ mean(pc_uniform) rtol=1e-6
    @test var(pc_tilted) ≈ var(pc_uniform) rtol=1e-6
end

@testitem "ExponentiallyTilted impact on delay distribution bias" begin
    using CensoredDistributions, Distributions, Test
    
    # Test how different growth rates affect the resulting delay distribution
    delay_dist = LogNormal(1.5, 0.75)  # Base delay distribution
    window_size = 1.0
    
    # Different epidemic scenarios
    scenarios = [
        ("decay", ExponentiallyTilted(0.0, window_size, -1.0)),
        ("uniform", ExponentiallyTilted(0.0, window_size, 0.0)),
        ("growth", ExponentiallyTilted(0.0, window_size, 1.0)),
    ]
    
    results = Dict()
    for (name, prior) in scenarios
        pc_dist = primary_censored(delay_dist, prior)
        results[name] = (
            mean = mean(pc_dist),
            var = var(pc_dist),
            dist = pc_dist
        )
    end
    
    # Test expected relationships
    # Growth should lead to shorter apparent delays (events happen later in window)
    @test results["growth"].mean < results["uniform"].mean
    @test results["uniform"].mean < results["decay"].mean
    
    # Test that all are reasonable values
    for (name, result) in results
        @test isfinite(result.mean)
        @test result.var >= 0
        @test isfinite(result.var)
    end
    
    # Test PDF comparison at a specific point
    x_test = 2.0
    @test pdf(results["decay"].dist, x_test) > pdf(results["uniform"].dist, x_test)
    @test pdf(results["uniform"].dist, x_test) > pdf(results["growth"].dist, x_test)
end

@testitem "ExponentiallyTilted with IntervalCensored" begin
    using CensoredDistributions, Distributions, Test
    
    # Test double censoring: primary event with exponential tilting + interval censoring
    delay_dist = LogNormal(1.5, 0.75)
    tilted_prior = ExponentiallyTilted(0.0, 1.0, 0.5)
    
    # First apply primary censoring with tilted distribution
    pc_dist = primary_censored(delay_dist, tilted_prior)
    
    # Then apply interval censoring
    interval_width = 0.5
    double_censored = interval_censored(pc_dist, interval_width)
    
    @test double_censored isa IntervalCensored
    @test double_censored.dist isa PrimaryCensored
    
    # Test basic functionality
    @test isfinite(mean(double_censored))
    @test var(double_censored) >= 0
    
    # Test evaluation
    test_points = [0.1, 0.5, 1.0, 2.0]
    for x in test_points
        @test pdf(double_censored, x) >= 0
        @test 0 <= cdf(double_censored, x) <= 1
    end
    
    # Test sampling
    samples = [rand(double_censored) for _ in 1:50]
    @test all(s >= 0 for s in samples)
    @test length(unique(samples)) > 1
end

@testitem "ExponentiallyTilted with DoubleIntervalCensored" begin
    using CensoredDistributions, Distributions, Test
    
    # Test integration with the convenience function double_interval_censored
    delay_dist = LogNormal(1.5, 0.75)
    
    # Test with different tilted priors
    tilted_priors = [
        ExponentiallyTilted(0.0, 1.0, 0.0),    # Uniform-like
        ExponentiallyTilted(0.0, 1.0, 1.0),    # Growth
        ExponentiallyTilted(0.0, 1.0, -0.8),   # Decay
    ]
    
    for prior in tilted_priors
        # Create double interval censored distribution
        double_censored = double_interval_censored(
            delay_dist;
            primary_event = prior,
            interval = 0.5
        )
        
        @test double_censored isa IntervalCensored
        
        # Test basic properties
        @test isfinite(mean(double_censored))
        @test var(double_censored) >= 0
        
        # Test evaluation
        @test pdf(double_censored, 1.0) >= 0
        @test 0 <= cdf(double_censored, 1.0) <= 1
        
        # Test sampling
        sample = rand(double_censored)
        @test sample >= 0
        @test isfinite(sample)
    end
end

@testitem "ExponentiallyTilted numerical vs analytical solutions" begin
    using CensoredDistributions, Distributions, Test
    
    # Test that ExponentiallyTilted works with existing analytical solutions
    # when they exist, and falls back to numerical integration otherwise
    
    # Test with distributions that have analytical solutions for Uniform primary events
    analytical_delay_dists = [
        Gamma(2.0, 1.5),
        LogNormal(1.5, 0.75),
        Weibull(2.0, 3.0)
    ]
    
    for delay_dist in analytical_delay_dists
        # Compare uniform vs slightly tilted
        uniform_prior = Uniform(0.0, 1.0)
        tilted_prior = ExponentiallyTilted(0.0, 1.0, 1e-8)  # Very slight tilt
        
        pc_uniform = primary_censored(delay_dist, uniform_prior)
        pc_tilted = primary_censored(delay_dist, tilted_prior)
        
        # They should be very close since the tilt is minimal
        test_points = [0.5, 1.0, 2.0]
        for x in test_points
            @test pdf(pc_tilted, x) ≈ pdf(pc_uniform, x) rtol=1e-4
            @test cdf(pc_tilted, x) ≈ cdf(pc_uniform, x) rtol=1e-4
        end
    end
    
    # Test with more significant tilting (should use numerical integration)
    delay_dist = LogNormal(1.5, 0.75)
    strong_tilt = ExponentiallyTilted(0.0, 1.0, 2.0)
    pc_strong = primary_censored(delay_dist, strong_tilt)
    
    # Should still work, just using numerical methods
    @test isfinite(mean(pc_strong))
    @test var(pc_strong) >= 0
    @test pdf(pc_strong, 1.0) >= 0
    @test 0 <= cdf(pc_strong, 1.0) <= 1
end

@testitem "ExponentiallyTilted parameter recovery simulation" begin
    using CensoredDistributions, Distributions, Test, Random
    
    Random.seed!(42)
    
    # Simulate data from a known exponentially tilted primary censored distribution
    true_delay = LogNormal(1.2, 0.6)
    true_growth_rate = 0.8
    true_window = 1.0
    
    true_prior = ExponentiallyTilted(0.0, true_window, true_growth_rate)
    true_dist = primary_censored(true_delay, true_prior)
    
    # Generate samples
    n_samples = 1000
    samples = [rand(true_dist) for _ in 1:n_samples]
    
    # Test that samples have expected properties
    empirical_mean = sum(samples) / n_samples
    theoretical_mean = mean(true_dist)
    
    @test empirical_mean ≈ theoretical_mean rtol=0.1  # Allow for sampling variation
    
    # Test that samples are all non-negative and finite
    @test all(s >= 0 for s in samples)
    @test all(isfinite(s) for s in samples)
    
    # Test that the distribution of samples makes sense
    # For growth case, we expect bias toward shorter delays
    uniform_prior = Uniform(0.0, true_window)
    uniform_dist = primary_censored(true_delay, uniform_prior)
    uniform_mean = mean(uniform_dist)
    
    @test empirical_mean < uniform_mean  # Growth should reduce apparent delay
end

@testitem "ExponentiallyTilted edge cases and robustness" begin
    using CensoredDistributions, Distributions, Test
    
    # Test with extreme parameter combinations
    delay_dist = LogNormal(1.0, 0.5)
    
    # Very narrow window
    narrow_prior = ExponentiallyTilted(0.0, 1e-6, 0.5)
    pc_narrow = primary_censored(delay_dist, narrow_prior)
    @test isfinite(mean(pc_narrow))
    @test var(pc_narrow) >= 0
    
    # Very wide window
    wide_prior = ExponentiallyTilted(0.0, 100.0, 0.1)
    pc_wide = primary_censored(delay_dist, wide_prior)
    @test isfinite(mean(pc_wide))
    @test var(pc_wide) >= 0
    
    # Large positive growth rate
    large_pos_prior = ExponentiallyTilted(0.0, 1.0, 5.0)
    pc_large_pos = primary_censored(delay_dist, large_pos_prior)
    @test isfinite(mean(pc_large_pos))
    @test var(pc_large_pos) >= 0
    
    # Large negative growth rate
    large_neg_prior = ExponentiallyTilted(0.0, 1.0, -5.0)
    pc_large_neg = primary_censored(delay_dist, large_neg_prior)
    @test isfinite(mean(pc_large_neg))
    @test var(pc_large_neg) >= 0
    
    # Test that extreme cases still maintain proper ordering
    @test mean(pc_large_neg) > mean(pc_large_pos)  # Decay > Growth
end

@testitem "ExponentiallyTilted with Weighted distributions" begin
    using CensoredDistributions, Distributions, Test
    
    # Test integration with Weighted utility
    delay_dist = LogNormal(1.5, 0.75)
    tilted_prior = ExponentiallyTilted(0.0, 1.0, 0.5)
    
    # Create weighted primary censored distribution
    pc_dist = primary_censored(delay_dist, tilted_prior)
    weighted_pc = weight(pc_dist, 2.5)
    
    @test weighted_pc isa Weighted
    @test weighted_pc.dist isa PrimaryCensored
    @test weighted_pc.weight == 2.5
    
    # Test basic functionality
    @test isfinite(mean(weighted_pc))
    @test var(weighted_pc) >= 0
    
    # Test evaluation
    @test pdf(weighted_pc, 1.0) >= 0
    @test 0 <= cdf(weighted_pc, 1.0) <= 1
    
    # Test sampling
    sample = rand(weighted_pc)
    @test sample >= 0
    @test isfinite(sample)
    
    # Test get_dist_recursive functionality
    @test get_dist_recursive(weighted_pc) === delay_dist
end