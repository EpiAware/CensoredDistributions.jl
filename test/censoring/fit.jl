@testitem "Fitting functionality" begin
    using CensoredDistributions
    using Distributions
    using Random
    using Statistics

    @testset "IntervalCensored fitting" begin
        Random.seed!(42)

        # Test basic Normal distribution fitting
        @testset "Normal distribution fitting" begin
            true_μ, true_σ = 2.5, 1.2
            true_underlying = Normal(true_μ, true_σ)
            interval_width = 0.5
            n_samples = 1000

            # Generate data
            true_censored = interval_censored(true_underlying, interval_width)
            data = rand(true_censored, n_samples)

            # Test fit_mle
            fitted_dist = fit_mle(IntervalCensored, data; interval = interval_width)
            fitted_params = params(fitted_dist.dist)

            # Parameter recovery should be within 5% for large samples
            μ_error = abs(fitted_params[1] - true_μ) / abs(true_μ)
            σ_error = abs(fitted_params[2] - true_σ) / abs(true_σ)

            @test μ_error < 0.05  # Less than 5% error
            @test σ_error < 0.1   # Less than 10% error for σ

            # Test that fit() and fit_mle() are equivalent
            fitted_dist2 = fit(IntervalCensored, data; interval = interval_width)
            @test params(fitted_dist.dist) == params(fitted_dist2.dist)

            # Test return type
            @test fitted_dist isa IntervalCensored
            @test fitted_dist.dist isa Normal
        end

        # Test Exponential distribution fitting
        @testset "Exponential distribution fitting" begin
            true_θ = 2.0
            true_underlying = Exponential(true_θ)
            interval_width = 0.3
            n_samples = 1000

            # Generate data
            true_censored = interval_censored(true_underlying, interval_width)
            data = rand(true_censored, n_samples)

            # Fit
            fitted_dist = fit_mle(IntervalCensored, data;
                dist_type = Exponential,
                interval = interval_width)
            fitted_params = params(fitted_dist.dist)

            # Parameter recovery
            θ_error = abs(fitted_params[1] - true_θ) / abs(true_θ)
            @test θ_error < 0.1  # Less than 10% error

            @test fitted_dist.dist isa Exponential
        end

        # Test custom boundaries
        @testset "Custom boundaries fitting" begin
            true_μ, true_σ = 3.0, 1.5
            true_underlying = Normal(true_μ, true_σ)
            boundaries = [0.0, 0.5, 1.2, 2.1, 3.5, 5.0, 7.0, 10.0]
            n_samples = 800

            # Generate data
            true_censored = interval_censored(true_underlying, boundaries)
            data = rand(true_censored, n_samples)

            # Fit
            fitted_dist = fit_mle(IntervalCensored, data; boundaries = boundaries)
            fitted_params = params(fitted_dist.dist)

            # Parameter recovery (looser bounds for irregular intervals)
            μ_error = abs(fitted_params[1] - true_μ) / abs(true_μ)
            σ_error = abs(fitted_params[2] - true_σ) / abs(true_σ)

            @test μ_error < 0.15  # Less than 15% error
            @test σ_error < 0.2   # Less than 20% error
        end

        # Test weighted fitting
        @testset "Weighted fitting" begin
            true_μ, true_σ = 1.8, 0.9
            true_underlying = Normal(true_μ, true_σ)
            interval_width = 0.4
            n_unique = 100

            # Generate unique data with weights
            unique_data = rand(interval_censored(true_underlying, interval_width), n_unique)
            weights = rand(1:10, n_unique)

            # Fit with and without weights
            unweighted_fit = fit_mle(
                IntervalCensored, unique_data; interval = interval_width)
            weighted_fit = fit_mle(IntervalCensored, unique_data;
                interval = interval_width, weights = weights)

            # Both should be valid fits, but potentially different
            @test unweighted_fit isa IntervalCensored
            @test weighted_fit isa IntervalCensored
            @test unweighted_fit.dist isa Normal
            @test weighted_fit.dist isa Normal

            # Weighted fit should generally be different from unweighted
            unweighted_params = params(unweighted_fit.dist)
            weighted_params = params(weighted_fit.dist)
            # Note: They might be the same by chance, so we don't test inequality
        end

        # Test initial parameters
        @testset "Custom initial parameters" begin
            true_μ, true_σ = 2.0, 1.0
            true_underlying = Normal(true_μ, true_σ)
            interval_width = 0.5
            n_samples = 500

            data = rand(interval_censored(true_underlying, interval_width), n_samples)

            # Test with custom initial parameters
            custom_init = [1.5, 0.8]  # Different from true values
            fitted_dist = fit_mle(IntervalCensored, data;
                interval = interval_width,
                init_params = custom_init)

            @test fitted_dist isa IntervalCensored
            fitted_params = params(fitted_dist.dist)

            # Should still recover parameters reasonably well
            μ_error = abs(fitted_params[1] - true_μ) / abs(true_μ)
            σ_error = abs(fitted_params[2] - true_σ) / abs(true_σ)

            @test μ_error < 0.1
            @test σ_error < 0.15
        end
    end

    @testset "Weighted distribution fitting" begin
        Random.seed!(123)

        @testset "Basic weighted fitting" begin
            true_μ, true_σ = 3.2, 1.4
            true_underlying = Normal(true_μ, true_σ)
            n_samples = 800

            data = rand(true_underlying, n_samples)
            weight_value = 5.0

            # Fit weighted distribution
            fitted_weighted = fit_mle(Weighted, data; weight_value = weight_value)

            @test fitted_weighted isa Weighted
            @test fitted_weighted.dist isa Normal
            @test fitted_weighted.weight == weight_value

            # Parameter recovery
            fitted_params = params(fitted_weighted.dist)
            μ_error = abs(fitted_params[1] - true_μ) / abs(true_μ)
            σ_error = abs(fitted_params[2] - true_σ) / abs(true_σ)

            @test μ_error < 0.05
            @test σ_error < 0.1

            # Test that fit() and fit_mle() are equivalent
            fitted_weighted2 = fit(Weighted, data; weight_value = weight_value)
            @test params(fitted_weighted.dist) == params(fitted_weighted2.dist)
            @test fitted_weighted.weight == fitted_weighted2.weight
        end

        @testset "Different distribution types for weighted fitting" begin
            # Test with Exponential
            true_θ = 2.5
            true_exp = Exponential(true_θ)
            data = rand(true_exp, 500)

            fitted_weighted = fit_mle(Weighted, data;
                underlying_dist_type = Exponential,
                weight_value = 3.0)

            @test fitted_weighted isa Weighted
            @test fitted_weighted.dist isa Exponential
            @test fitted_weighted.weight == 3.0

            fitted_params = params(fitted_weighted.dist)
            θ_error = abs(fitted_params[1] - true_θ) / abs(true_θ)
            @test θ_error < 0.1
        end

        @testset "Zero weight edge case" begin
            data = rand(Normal(0, 1), 100)

            fitted_weighted = fit_mle(Weighted, data; weight_value = 0.0)
            @test fitted_weighted.weight == 0.0
            @test logpdf(fitted_weighted, 0.0) == -Inf
        end
    end

    @testset "Double interval censored fitting" begin
        Random.seed!(456)

        @testset "Basic double censored fitting" begin
            # True parameters
            true_delay_μ, true_delay_σ = 1.2, 0.6
            true_primary_a, true_primary_b = 0.0, 1.5

            true_delay = LogNormal(true_delay_μ, true_delay_σ)
            true_primary = Uniform(true_primary_a, true_primary_b)
            interval_width = 0.3
            n_samples = 600

            # Generate data (with force_numeric=true for fitting)
            true_double = double_interval_censored(true_delay, true_primary;
                interval = interval_width, force_numeric = true)
            data = rand(true_double, n_samples)

            # Fit using clean interface
            fitted_double = fit_mle(true_double, data)

            # Extract fitted parameters
            fitted_delay_params = params(fitted_double.dist.dist)
            fitted_primary_params = params(fitted_double.dist.primary_event)

            # Parameter recovery - focus on delay parameters (primary event is marginalized nuisance parameter)
            delay_μ_error = abs(fitted_delay_params[1] - true_delay_μ) / abs(true_delay_μ)
            delay_σ_error = abs(fitted_delay_params[2] - true_delay_σ) / abs(true_delay_σ)

            @test delay_μ_error < 0.3   # 30% tolerance for complex double censoring
            @test delay_σ_error < 0.4   # 40% tolerance

            # Primary event parameters are nuisance parameters we marginalize over
            # We only test that they are reasonable (finite and positive for bounds)
            @test isfinite(fitted_primary_params[1])
            @test isfinite(fitted_primary_params[2])
            @test fitted_primary_params[2] > fitted_primary_params[1]  # b > a for Uniform
        end

        @testset "Double censored with weights" begin
            true_delay = LogNormal(1.0, 0.5)
            true_primary = Uniform(0, 1)
            interval_width = 0.2
            n_samples = 400

            true_double = double_interval_censored(true_delay, true_primary;
                interval = interval_width, force_numeric = true)
            data = rand(true_double, n_samples)
            weights = rand(1:5, n_samples)

            # Fit with weights using clean interface
            fitted_double = fit_mle(true_double, data; weights = weights)

            # Should return a valid distribution
            @test fitted_double isa IntervalCensored
            @test fitted_double.dist isa PrimaryCensored
        end

        @testset "Double censored with truncation" begin
            true_delay = LogNormal(0.8, 0.4)
            true_primary = Uniform(0, 0.8)
            interval_width = 0.25
            upper_bound = 5.0
            n_samples = 500

            true_double = double_interval_censored(true_delay, true_primary;
                interval = interval_width,
                upper = upper_bound, force_numeric = true)
            data = rand(true_double, n_samples)

            # Fit with truncation using clean interface
            fitted_double = fit_mle(true_double, data)

            @test fitted_double isa IntervalCensored
            # Data should be within bounds
            @test all(d <= upper_bound for d in data)
        end
    end

    @testset "Error handling and edge cases" begin
        @testset "Input validation" begin
            # Empty data
            @test_throws ArgumentError fit_mle(IntervalCensored, Float64[]; interval = 1.0)

            # Invalid weights
            data = [1.0, 2.0, 3.0]
            @test_throws ArgumentError fit_mle(IntervalCensored, data;
                interval = 1.0, weights = [1.0, 2.0])  # Wrong length
            @test_throws ArgumentError fit_mle(IntervalCensored, data;
                interval = 1.0, weights = [-1.0, 1.0, 2.0])  # Negative weights
            @test_throws ArgumentError fit_mle(IntervalCensored, data;
                interval = 1.0, weights = [0.0, 0.0, 0.0])  # All zero weights

            # Invalid weight value for Weighted
            @test_throws ArgumentError fit_mle(Weighted, data; weight_value = -1.0)

            # Non-finite data
            @test_throws ArgumentError fit_mle(
                IntervalCensored, [1.0, Inf, 3.0]; interval = 1.0)
            @test_throws ArgumentError fit_mle(
                IntervalCensored, [1.0, NaN, 3.0]; interval = 1.0)
        end

        @testset "Unsupported distribution types" begin
            data = [1.0, 2.0, 3.0, 4.0, 5.0]

            # Should throw for unsupported distribution without init params
            @test_throws ArgumentError fit_mle(IntervalCensored, data;
                dist_type = Gamma, interval = 1.0)
        end

        @testset "Small sample behavior" begin
            # Very small sample - should still work but with larger errors
            Random.seed!(789)
            true_dist = Normal(2.0, 1.0)
            small_data = rand(interval_censored(true_dist, 0.5), 10)

            fitted = fit_mle(IntervalCensored, small_data; interval = 0.5)
            @test fitted isa IntervalCensored
            @test fitted.dist isa Normal

            # Parameters might be off but should be finite
            fitted_params = params(fitted.dist)
            @test all(isfinite, fitted_params)
        end

        @testset "Boundary conditions" begin
            # Data near distribution boundaries
            Random.seed!(321)

            # Generate data from Exponential (support on [0, ∞))
            true_exp = Exponential(1.0)
            exp_data = rand(interval_censored(true_exp, 0.1), 200)

            fitted = fit_mle(IntervalCensored, exp_data;
                dist_type = Exponential, interval = 0.1)
            @test fitted isa IntervalCensored
            @test fitted.dist isa Exponential
        end
    end

    @testset "Parameter recovery across sample sizes" begin
        Random.seed!(999)

        true_μ, true_σ = 2.0, 1.0
        true_dist = Normal(true_μ, true_σ)
        interval_width = 0.5

        sample_sizes = [50, 100, 500, 1000]
        errors = Float64[]

        for n in sample_sizes
            data = rand(interval_censored(true_dist, interval_width), n)
            fitted = fit_mle(IntervalCensored, data; interval = interval_width)
            fitted_params = params(fitted.dist)

            μ_error = abs(fitted_params[1] - true_μ) / abs(true_μ)
            push!(errors, μ_error)
        end

        # Errors should generally decrease with sample size
        # (though this is stochastic, so we just check it's reasonable)
        @test all(e -> e < 0.2, errors)  # All errors less than 20%
        # Check that average of larger samples is better than average of smaller samples
        small_avg = mean(errors[1:2])  # First two sample sizes
        large_avg = mean(errors[3:4])  # Last two sample sizes
        @test large_avg < small_avg + 0.05  # Allow some tolerance for stochastic variation
    end
end
