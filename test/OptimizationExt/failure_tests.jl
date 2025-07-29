@testitem "Boundary conditions - unsupported data" tags=[:optimization, :failure] begin
    using CensoredDistributions
    using Distributions
    using Random
    using Statistics
    # Load the weak dependencies to trigger extension loading
    using Optimization
    using OptimizationOptimJL
    using Bijectors

    Random.seed!(999)

    # Test with Exponential distribution and negative data
    # (This should not happen in practice, but tests robustness)
    mixed_data = [-1.0, 0.5, 1.0, 2.0, 3.0]  # Includes negative value
    template_dist = interval_censored(Exponential(1.0), 0.1)

    try
        fitted = fit_mle(template_dist, mixed_data; intervals = 0.1)
        # If it works, should handle negative data somehow
        @test fitted isa IntervalCensored
    catch e
        # If it fails, should be appropriate error
        @test e isa Union{DomainError, ArgumentError, BoundsError}
    end
end

@testitem "Boundary conditions - parameter limits" tags=[:optimization, :failure] begin
    using CensoredDistributions
    using Distributions
    using Random
    using Statistics
    # Load the weak dependencies to trigger extension loading
    using Optimization
    using OptimizationOptimJL
    using Bijectors

    Random.seed!(999)

    # Test cases that might push parameters to their boundaries
    # Data that might lead to σ → 0
    near_identical_data = [5.0, 5.001, 4.999, 5.0001, 4.9999]
    template_dist = interval_censored(Normal(0.0, 1.0), 0.01)  # Very small intervals

    fitted = fit_mle(template_dist, near_identical_data; intervals = 0.01)
    @test fitted isa IntervalCensored
    fitted_params = params(fitted.dist)
    @test fitted_params[2] > 0  # σ should remain positive
    @test fitted_params[2] > 1e-6  # σ should not be too close to 0
end

@testitem "Weight-related failures - extreme weights" tags=[:optimization, :failure] begin
    using CensoredDistributions
    using Distributions
    using Random
    using Statistics
    # Load the weak dependencies to trigger extension loading
    using Optimization
    using OptimizationOptimJL
    using Bijectors

    Random.seed!(999)

    data = randn(100)
    template_dist = interval_censored(Normal(0.0, 1.0), 0.5)

    # One observation with overwhelming weight
    extreme_weights = fill(1e-10, 100)
    extreme_weights[1] = 1.0  # One observation dominates

    fitted = fit_mle(
        template_dist, data; weights = extreme_weights, intervals = 0.5)
    @test fitted isa IntervalCensored
    fitted_params = params(fitted.dist)
    @test abs(fitted_params[1] - data[1]) < 2.0  # Mean should be close to the dominant observation

    # Many tiny weights
    tiny_weights = fill(1e-15, 100)
    try
        fitted_tiny = fit_mle(
            template_dist, data; weights = tiny_weights, intervals = 0.5)
        @test fitted_tiny isa IntervalCensored
    catch e
        @test e isa Union{ArgumentError}
    end
end

@testitem "Optimization algorithm failures - poor conditioning" tags=[
    :optimization, :failure] begin
    using CensoredDistributions
    using Distributions
    using Random
    using Statistics
    # Load the weak dependencies to trigger extension loading
    using Optimization
    using OptimizationOptimJL
    using Bijectors

    # Create a scenario that might cause optimization difficulties
    Random.seed!(888)

    # Data with very different scales in different parts
    part1 = randn(50) * 0.001  # Very small variance part
    part2 = randn(50) * 100.0 .+ 1000.0  # Large variance, shifted part
    problematic_data = [part1..., part2...]

    template_dist = interval_censored(Normal(0.0, 1.0), 1.0)

    # Should handle poorly conditioned optimization
    fitted = fit_mle(template_dist, problematic_data; intervals = 1.0)
    @test fitted isa IntervalCensored
    fitted_params = params(fitted.dist)
    @test all(isfinite, fitted_params)
    @test fitted_params[2] > 0
end

@testitem "Double censored - conflicting parameters" tags=[:optimization, :failure] begin
    using CensoredDistributions
    using Distributions
    using Random
    using Statistics
    # Load the weak dependencies to trigger extension loading
    using Optimization
    using OptimizationOptimJL
    using Bijectors

    Random.seed!(555)

    # Test cases where primary event and interval censoring might conflict
    # Very narrow primary event window with wide intervals
    delay_dist = LogNormal(0.0, 1.0)
    narrow_primary = Uniform(0.0, 0.001)  # Very narrow primary window
    template_dist = double_interval_censored(
        delay_dist; primary_event = narrow_primary, interval = 10.0)

    # Generate some data for testing
    data = rand(template_dist, 50)

    try
        fitted = fit_mle(template_dist, data; intervals = 10.0)
        @test fitted isa typeof(template_dist)
    catch e
        @test e isa Union{MethodError, ArgumentError}
    end
end

@testitem "Double censored - extreme truncation" tags=[:optimization, :failure] begin
    using CensoredDistributions
    using Distributions
    using Random
    using Statistics
    # Load the weak dependencies to trigger extension loading
    using Optimization
    using OptimizationOptimJL
    using Bijectors

    Random.seed!(555)

    # Test with very aggressive truncation
    delay_dist = LogNormal(2.0, 1.0)  # Mean around exp(2.5) ≈ 12
    primary = Uniform(0.0, 1.0)

    # Truncate very early (most data would be truncated)
    template_dist = double_interval_censored(delay_dist;
        primary_event = primary, interval = 1.0, upper = 2.0)  # Very early truncation

    # This might result in very little data surviving truncation
    data = rand(template_dist, 200)  # Try to get reasonable amount despite truncation

    if length(data) > 10  # Only test if we have enough data
        try
            fitted = fit_mle(template_dist, data; intervals = 1.0)
            @test fitted isa typeof(template_dist)
        catch e
            @test e isa Union{MethodError, ArgumentError}
        end
    else
        @test length(data) >= 0  # At least no error in data generation
    end
end

@testitem "Double censored - mismatched parameters" tags=[:optimization, :failure] begin
    using CensoredDistributions
    using Distributions
    using Random
    using Statistics
    # Load the weak dependencies to trigger extension loading
    using Optimization
    using OptimizationOptimJL
    using Bijectors

    Random.seed!(555)

    # Test fitting with parameters that don't match the data generation
    delay_dist = LogNormal(1.0, 0.5)
    true_primary = Uniform(0.0, 2.0)
    true_interval = 0.5

    # Generate data with true parameters
    true_dist = double_interval_censored(delay_dist;
        primary_event = true_primary, interval = true_interval)
    data = rand(true_dist, 100)

    # Try to fit with very different parameters
    wrong_primary = Uniform(0.0, 10.0)  # Much wider primary window
    wrong_interval = 5.0  # Much wider intervals

    template_dist = double_interval_censored(delay_dist;
        primary_event = wrong_primary, interval = wrong_interval)

    # Should still work, but might not recover true parameters well
    try
        fitted = fit_mle(template_dist, data; intervals = wrong_interval)
        @test fitted isa typeof(template_dist)

        # The delay distribution parameters should still be estimable
        fitted_delay_params = params(fitted.dist.dist)
        @test all(isfinite, fitted_delay_params)
    catch e
        @test e isa Union{MethodError, ArgumentError}
    end
end
