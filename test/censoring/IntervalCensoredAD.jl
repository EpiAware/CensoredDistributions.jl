@testitem "IntervalCensored AD Compatibility" begin
    using Test
    using Distributions
    using CensoredDistributions
    using ForwardDiff

    # Helper function to construct distributions with AD-compatible constructors
    function reconstruct_distribution(base_dist, params)
        if typeof(base_dist) <: Normal
            return Normal(params[1], params[2])
        elseif typeof(base_dist) <: LogNormal
            return LogNormal(params[1], params[2])
        elseif typeof(base_dist) <: Gamma
            return Gamma(params[1], params[2])
        elseif typeof(base_dist) <: Exponential
            return Exponential(params[1])
        elseif isa(base_dist, Truncated{<:Normal})
            # For truncated Normal, reconstruct the underlying distribution
            return truncated(Normal(params[1], params[2]),
                base_dist.lower, base_dist.upper)
        else
            return typeof(base_dist)(params...)
        end
    end

    # Test distributions with various support types
    test_distributions = [
        Normal(2.0, 1.5),           # Unbounded support
        LogNormal(1.0, 0.5),        # Bounded support (positive)
        Gamma(3.0, 2.0),            # Bounded support (positive)
        Exponential(1.5),           # Bounded support (positive)
        truncated(Normal(0, 1), -2, 2)  # Truncated (bounded)
    ]

    @testset "ForwardDiff compatibility - regular intervals" begin
        for base_dist in test_distributions
            dist_name = typeof(base_dist)
            @testset "$dist_name" begin
                interval = 0.5
                ic = interval_censored(base_dist, interval)

                # Test parameter gradient for logpdf
                params = collect(Distributions.params(base_dist))
                if length(params) > 0
                    test_logpdf = p -> begin
                        new_dist = reconstruct_distribution(base_dist, p)
                        new_ic = interval_censored(new_dist, interval)
                        return logpdf(new_ic, 1.0)
                    end

                    grad = ForwardDiff.gradient(test_logpdf, params)
                    @test all(isfinite, grad)
                    @test !any(isnan, grad)
                    @test length(grad) == length(params)
                end

                # Test parameter gradient for cdf
                if length(params) > 0
                    test_cdf = p -> begin
                        new_dist = reconstruct_distribution(base_dist, p)
                        new_ic = interval_censored(new_dist, interval)
                        return cdf(new_ic, 1.5)
                    end

                    grad_cdf = ForwardDiff.gradient(test_cdf, params)
                    @test all(isfinite, grad_cdf)
                    @test !any(isnan, grad_cdf)
                    @test length(grad_cdf) == length(params)
                end

                # Test parameter gradient for logcdf
                if length(params) > 0
                    test_logcdf = p -> begin
                        new_dist = reconstruct_distribution(base_dist, p)
                        new_ic = interval_censored(new_dist, interval)
                        return logcdf(new_ic, 1.5)
                    end

                    grad_logcdf = ForwardDiff.gradient(test_logcdf, params)
                    @test all(g -> isfinite(g) || g == -Inf, grad_logcdf)
                    @test !any(isnan, grad_logcdf)
                    @test length(grad_logcdf) == length(params)
                end
            end
        end
    end

    @testset "ForwardDiff compatibility - arbitrary intervals" begin
        for base_dist in test_distributions
            dist_name = typeof(base_dist)
            @testset "$dist_name" begin
                # Use intervals that should contain some probability mass
                intervals = [0.0, 1.0, 2.5, 5.0]
                ic = interval_censored(base_dist, intervals)

                params = collect(Distributions.params(base_dist))
                if length(params) > 0
                    # Test logpdf gradient at interval boundary
                    test_logpdf = p -> begin
                        new_dist = reconstruct_distribution(base_dist, p)
                        new_ic = interval_censored(new_dist, intervals)
                        return logpdf(new_ic, 1.0)  # At interval boundary
                    end

                    grad = ForwardDiff.gradient(test_logpdf, params)
                    @test all(g -> isfinite(g) || g == -Inf, grad)
                    @test !any(isnan, grad)
                end
            end
        end
    end

    @testset "Boundary value AD tests" begin
        # Test AD behaviour at distribution boundaries
        d = LogNormal(1.0, 0.5)  # Support: (0, ∞)
        params = [1.0, 0.5]

        # Test near distribution minimum
        ic_reg = interval_censored(d, 0.1)
        test_near_min = p -> begin
            ic = interval_censored(reconstruct_distribution(d, p), 0.1)
            return logpdf(ic, 0.01)  # Very small positive value
        end

        grad_near_min = ForwardDiff.gradient(test_near_min, params)
        @test all(g -> isfinite(g) || g == -Inf, grad_near_min)
        @test !any(isnan, grad_near_min)

        # Test with arbitrary intervals including zero (outside support)
        intervals = [0.0, 0.5, 2.0, 5.0]
        ic_arb = interval_censored(d, intervals)
        test_at_boundary = p -> logpdf(
            interval_censored(reconstruct_distribution(d, p),
                intervals), 0.0)

        grad_boundary = ForwardDiff.gradient(test_at_boundary, params)
        @test all(g -> isfinite(g) || g == -Inf, grad_boundary)
        @test !any(isnan, grad_boundary)
    end

    @testset "Edge case AD tests" begin
        # Test with very small intervals
        d = Normal(0, 1)
        params = [0.0, 1.0]

        tiny_interval = 1e-10
        test_tiny = p -> begin
            ic = interval_censored(
                reconstruct_distribution(d, p), tiny_interval)
            return logpdf(ic, 0.0)
        end

        grad_tiny = ForwardDiff.gradient(test_tiny, params)
        @test all(g -> isfinite(g) || g == -Inf, grad_tiny)
        @test !any(isnan, grad_tiny)

        # Test with very large intervals
        huge_interval = 1e10
        test_huge = p -> begin
            ic = interval_censored(
                reconstruct_distribution(d, p), huge_interval)
            return logpdf(ic, 0.0)
        end

        grad_huge = ForwardDiff.gradient(test_huge, params)
        @test all(isfinite, grad_huge)
        @test !any(isnan, grad_huge)
    end

    @testset "CDF AD tests with step function behaviour" begin
        # Test CDF gradients where step function behaviour matters
        d = Normal(2.0, 1.0)
        params = [2.0, 1.0]
        intervals = [0.0, 1.0, 2.0, 3.0, 4.0]

        # Test cdf at various points
        test_points = [0.5, 1.5, 2.5, 3.5]

        for x in test_points
            test_cdf_at_x = p -> begin
                ic = interval_censored(
                    reconstruct_distribution(d, p), intervals)
                return cdf(ic, x)
            end

            grad_cdf = ForwardDiff.gradient(test_cdf_at_x, params)
            @test all(isfinite, grad_cdf)
            @test !any(isnan, grad_cdf)
        end
    end

    @testset "Multiple parameter AD tests" begin
        # Test distributions with multiple parameters
        gamma_params = [2.0, 1.5]  # shape, scale
        test_gamma_multi = p -> begin
            gamma_dist = Gamma(2.0, 1.5)  # Define base distribution
            ic = interval_censored(reconstruct_distribution(gamma_dist, p), [
                0.0, 1.0, 3.0, 6.0])
            return logpdf(ic, 1.0)
        end

        grad_gamma = ForwardDiff.gradient(test_gamma_multi, gamma_params)
        @test length(grad_gamma) == 2
        @test all(g -> isfinite(g) || g == -Inf, grad_gamma)
        @test !any(isnan, grad_gamma)

        # Test truncated distribution parameters
        if hasmethod(truncated, (Normal, Real, Real))
            truncated_dist = truncated(Normal(0, 1), -2, 2)
            trunc_params = [0.0, 1.0]  # mean, std of underlying Normal

            test_truncated = p -> begin
                base = reconstruct_distribution(truncated_dist, p)
                ic = interval_censored(base, 0.5)
                return logpdf(ic, 0.0)
            end

            grad_trunc = ForwardDiff.gradient(test_truncated, trunc_params)
            @test length(grad_trunc) == 2
            @test all(g -> isfinite(g) || g == -Inf, grad_trunc)
            @test !any(isnan, grad_trunc)
        end
    end

    @testset "Consistency tests - logpdf and pdf gradients" begin
        # Ensure logpdf and log(pdf) have consistent gradients where pdf > 0
        d = Normal(1.0, 0.8)
        params = [1.0, 0.8]

        test_logpdf = p -> begin
            ic = interval_censored(reconstruct_distribution(d, p), 0.5)
            return logpdf(ic, 1.0)
        end
        test_log_pdf = p -> begin
            ic = interval_censored(reconstruct_distribution(d, p), 0.5)
            return log(pdf(ic, 1.0))
        end

        grad_logpdf = ForwardDiff.gradient(test_logpdf, params)
        grad_log_pdf = ForwardDiff.gradient(test_log_pdf, params)

        @test all(isfinite, grad_logpdf)
        @test all(isfinite, grad_log_pdf)
        @test grad_logpdf ≈ grad_log_pdf rtol=1e-10
    end

    @testset "AD with extreme parameter values" begin
        # Test AD behaviour with extreme but valid parameter values
        extreme_cases = [
            (Normal(1e6, 1e-3), [1e6, 1e-3]),      # Very concentrated
            (Normal(-1e6, 1e3), [-1e6, 1e3]),      # Very spread, negative mean
            (Gamma(1e-3, 1e3), [1e-3, 1e3]),       # Extreme gamma parameters
            (Exponential(1e-6), [1e-6])             # Very small rate
        ]

        for (extreme_dist, extreme_params) in extreme_cases
            dist_type = typeof(extreme_dist)
            @testset "Extreme $dist_type" begin
                ic = interval_censored(extreme_dist, 1.0)

                test_extreme = p -> begin
                    new_dist = reconstruct_distribution(extreme_dist, p)
                    new_ic = interval_censored(new_dist, 1.0)
                    # Test at a reasonable value, not extreme
                    return logpdf(new_ic, 0.0)
                end

                grad_extreme = ForwardDiff.gradient(test_extreme,
                    extreme_params)
                @test all(g -> isfinite(g) || g == -Inf, grad_extreme)
                @test !any(isnan, grad_extreme)
            end
        end
    end
end
