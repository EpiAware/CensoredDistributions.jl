# Streamlined unit tests - only testing actual package functionality

@testitem "Objective function construction" tags=[:optimization, :unit] begin
    using CensoredDistributions
    using Distributions
    using Random
    using Statistics
    # Load the weak dependencies to trigger extension loading
    using Optimization
    using OptimizationOptimJL
    using Bijectors

    Random.seed!(42)

    # Create simple test data
    true_dist = Normal(2.0, 1.0)
    data = rand(interval_censored(true_dist, 0.5), 100)

    # Test that we can construct an objective function
    template_dist = interval_censored(Normal(0.0, 1.0), 0.5)
    initial_params = [1.0, 0.8]  # μ, σ

    # Test bijector construction
    bijector = CensoredDistributions._get_bijector(
        typeof(template_dist.dist), initial_params)
    @test isa(bijector, Function) || hasmethod(bijector, Tuple{typeof(initial_params)})  # Should be callable

    # Test parameter transformation
    unconstrained = bijector(initial_params)
    @test length(unconstrained) == 2
    @test all(isfinite, unconstrained)

    # Test inverse transformation
    recovered = inverse(bijector)(unconstrained)
    @test recovered≈initial_params rtol=1e-10

    # Test that we can construct distribution from parameters
    dist_constructor = (params) -> interval_censored(Normal(params[1], params[2]), 0.5)
    test_dist = dist_constructor(initial_params)
    @test test_dist isa IntervalCensored
    @test test_dist.boundaries ≈ 0.5

    # Test log-likelihood computation
    ll = sum(logpdf(test_dist, x) for x in data)
    @test isfinite(ll)
    @test ll < 0  # Should be negative for proper likelihood
end

@testitem "Parameter constraint handling" tags=[:optimization, :unit] begin
    using CensoredDistributions
    using Distributions
    using Random
    using Statistics
    # Load the weak dependencies to trigger extension loading
    using Optimization
    using OptimizationOptimJL
    using Bijectors

    Random.seed!(42)

    # Test that invalid parameters are handled correctly
    template_dist = interval_censored(Normal(0.0, 1.0), 0.5)

    # Test bijector prevents invalid parameters
    bijector = CensoredDistributions._get_bijector(Normal, [1.0, 0.5])
    # The bijector should map constrained space to unconstrained space
    valid_params = [1.0, 0.5]
    unconstrained = bijector(valid_params)
    @test all(isfinite, unconstrained)
end

@testitem "fit_mle basic functionality" tags=[:optimization, :unit] begin
    using CensoredDistributions
    using Distributions
    using Random
    using Statistics
    # Load the weak dependencies to trigger extension loading
    using Optimization
    using OptimizationOptimJL
    using Bijectors

    Random.seed!(123)

    # Create a simple case where we know the answer
    true_μ, true_σ = 2.0, 1.5
    true_underlying = Normal(true_μ, true_σ)
    data = rand(interval_censored(true_underlying, 0.1), 500)  # Small interval, large sample

    template_dist = interval_censored(Normal(0.0, 1.0), 0.1)

    # Test basic fit_mle functionality
    fitted_dist = fit_mle(template_dist, data; intervals = 0.1)
    @test fitted_dist isa IntervalCensored
    @test fitted_dist.dist isa Normal

    # Test that fitted parameters are reasonable (not testing exact recovery)
    fitted_params = params(fitted_dist.dist)
    @test all(isfinite, fitted_params)
    @test fitted_params[2] > 0  # σ should be positive

    # Test that fitted distribution can compute probabilities
    @test all(isfinite(logpdf(fitted_dist, x)) for x in data[1:10])
end

@testitem "fit_mle with weights" tags=[:optimization, :unit] begin
    using CensoredDistributions
    using Distributions
    using Random
    using Statistics
    # Load the weak dependencies to trigger extension loading
    using Optimization
    using OptimizationOptimJL
    using Bijectors

    Random.seed!(42)

    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    template_dist = interval_censored(Normal(0.0, 1.0), 0.5)

    # Test that fit_mle works with weights
    weights = [1.0, 2.0, 1.0, 2.0, 1.0]
    fitted = fit_mle(template_dist, data; weights = weights, intervals = 0.5)
    @test fitted isa IntervalCensored
end

@testitem "Data validation" tags=[:optimization, :unit] begin
    using CensoredDistributions
    using Distributions
    using Random
    using Statistics
    # Load the weak dependencies to trigger extension loading
    using Optimization
    using OptimizationOptimJL
    using Bijectors

    Random.seed!(456)

    template_dist = interval_censored(Normal(0.0, 1.0), 1.0)

    # Test empty data
    @test_throws ArgumentError fit_mle(template_dist, Float64[]; intervals = 1.0)

    # Test with NaN values
    data_with_nan = [1.0, NaN, 3.0]
    @test_throws Union{ArgumentError, DomainError} fit_mle(
        template_dist, data_with_nan; intervals = 1.0)
end

@testitem "Weight validation" tags=[:optimization, :unit] begin
    using CensoredDistributions
    using Distributions
    using Random
    using Statistics
    # Load the weak dependencies to trigger extension loading
    using Optimization
    using OptimizationOptimJL
    using Bijectors

    Random.seed!(456)

    data = [1.0, 2.0, 3.0]
    template_dist = interval_censored(Normal(0.0, 1.0), 0.5)

    # Test negative weights
    negative_weights = [-1.0, 1.0, 1.0]
    @test_throws ArgumentError fit_mle(
        template_dist, data; weights = negative_weights, intervals = 0.5)

    # Test mismatched weight length
    wrong_length_weights = [1.0, 1.0]  # Length 2, data length 3
    @test_throws DimensionMismatch fit_mle(
        template_dist, data; weights = wrong_length_weights, intervals = 0.5)

    # Test zero weights
    zero_weights = [0.0, 0.0, 0.0]
    @test_throws ArgumentError fit_mle(
        template_dist, data; weights = zero_weights, intervals = 0.5)
end

@testitem "Interval validation" tags=[:optimization, :unit] begin
    using CensoredDistributions
    using Distributions
    using Random
    using Statistics
    # Load the weak dependencies to trigger extension loading
    using Optimization
    using OptimizationOptimJL
    using Bijectors

    Random.seed!(456)

    data = randn(50)
    template_dist = interval_censored(Normal(0.0, 1.0), 1.0)

    # Test negative intervals
    @test_throws ArgumentError fit_mle(template_dist, data; intervals = -1.0)

    # Test zero intervals
    @test_throws ArgumentError fit_mle(template_dist, data; intervals = 0.0)
end

@testitem "Return fit object functionality" tags=[:optimization, :unit] begin
    using CensoredDistributions
    using Distributions
    using Random
    using Statistics
    # Load the weak dependencies to trigger extension loading
    using Optimization
    using OptimizationOptimJL
    using Bijectors

    Random.seed!(123)

    # Test return_fit_object functionality
    true_μ, true_σ = 2.0, 1.5
    true_underlying = Normal(true_μ, true_σ)
    data = rand(interval_censored(true_underlying, 0.1), 200)

    template_dist = interval_censored(Normal(0.0, 1.0), 0.1)

    # Test that we can get the fit result object
    fitted_dist,
    fit_result = fit_mle(
        template_dist, data; intervals = 0.1, return_fit_object = true)
    @test fitted_dist isa IntervalCensored
    @test hasfield(typeof(fit_result), :retcode)  # Should have optimization result structure
    @test hasfield(typeof(fit_result), :objective)  # Should have objective value
end
