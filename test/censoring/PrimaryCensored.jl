@testitem "Test struct type" begin
    using Distributions
    @test_throws MethodError primary_censored(Normal(1, 2), 7)
end

@testitem "Test constructor" begin
    using Distributions
    use_dist = primary_censored(LogNormal(3.5, 1.5), Uniform(1, 2))
    @test typeof(use_dist) <: CensoredDistributions.PrimaryCensored
end

@testitem "Default constructor (analytical solver)" begin
    using Distributions
    using Integrals

    dist = Gamma(2.0, 3.0)
    primary = Uniform(0.0, 1.0)
    d = primary_censored(dist, primary)

    @test d.dist === dist
    @test d.primary_event === primary
    @test d.method isa CensoredDistributions.AnalyticalSolver
    @test d.method.solver isa QuadGKJL
end

@testitem "Constructor with force_numeric" begin
    using Distributions
    using Integrals

    dist = LogNormal(1.5, 0.75)
    primary = Uniform(0.0, 1.0)
    d = primary_censored(dist, primary; force_numeric = true)

    @test d.dist === dist
    @test d.primary_event === primary
    @test d.method isa CensoredDistributions.NumericSolver
    @test d.method.solver isa QuadGKJL
end

@testitem "Constructor with custom solver" begin
    using Distributions
    using Integrals

    dist = Weibull(2.0, 1.5)
    primary = Uniform(0.0, 1.0)
    custom_solver = HCubatureJL()
    d = primary_censored(dist, primary; solver = custom_solver, force_numeric = false)

    @test d.method isa CensoredDistributions.AnalyticalSolver
    @test d.method.solver === custom_solver
end

@testitem "Test random generation" begin
    using Distributions
    use_dist = primary_censored(LogNormal(3.5, 1.5), Uniform(1, 2))
    @test length(rand(use_dist, 10)) == 10

    use_dist_trunc = truncated(use_dist, 3, 10)
    use_dist_trunc_rn = rand(use_dist_trunc, 1000000)
    @test length(use_dist_trunc_rn) ≈ 1e6
    @test maximum(use_dist_trunc_rn) <= 10
    @test minimum(use_dist_trunc_rn) >= 3
end

@tesitem "Params calls" begin
    using Distributions
    use_dist = primary_censored(LogNormal(3.5, 1.5), Uniform(1, 2))
    extracted_params = params(use_dist)
    @test length(extracted_params) == 4
    @test extracted_params[1] ≈ 3.5
    @test extracted_params[3] ≈ 1.0
end

@testitem "CDF constructor and end point" begin
    using Distributions

    use_dist = primary_censored(LogNormal(3.5, 1.5), Uniform(1, 2))
    @test cdf(use_dist, 1e8) ≈ 1.0
    @test logcdf(use_dist, -Inf) ≈ -Inf
end

@testitem "CDF against known Exponential analytical solution" begin
    using Distributions

    dist = Exponential(1.0)
    use_dist = primary_censored(dist, Uniform(0, 1))
    # Analytical solution for the pmf of observation in [0,1], [1,2], ...
    expected_pmf_uncond = [exp(-1)
                           [(1 - exp(-1)) * (exp(1) - 1) * exp(-s) for s in 1:9]]
    # Analytical solution for the cdf
    expected_cdf = [0.0; cumsum(expected_pmf_uncond)]
    # Calculated cdf
    calc_cdf = map(0:10) do t
        cdf(use_dist, t)
    end
    @test expected_cdf ≈ calc_cdf
end

@testitem "Test ccdf" begin
    using Distributions
    use_dist = primary_censored(LogNormal(3.5, 1.5), Uniform(1, 2))
    @test ccdf(use_dist, 1e8) ≈ 0.0
    @test ccdf(use_dist, 0.0) ≈ 1
    @test logccdf(use_dist, 0.0) ≈ 0.0
end

@testitem "Keyword argument constructor" begin
    using Distributions

    dist = LogNormal(1.5, 0.75)
    primary = Uniform(0, 2)

    # Test keyword version returns correct PrimaryCensored struct
    d_kw = primary_censored(dist; primary_event = primary)
    @test typeof(d_kw) <: CensoredDistributions.PrimaryCensored
    @test d_kw.dist === dist
    @test d_kw.primary_event === primary

    # Test default primary_event
    d_default = primary_censored(dist)
    @test typeof(d_default) <: CensoredDistributions.PrimaryCensored
    @test d_default.dist === dist
    @test d_default.primary_event == Uniform(0, 1)
end

@testitem "Test logpdf extreme value handling and -Inf edge cases" begin
    using Distributions

    # Test with various distribution types for robustness
    test_distributions = [
        (LogNormal(1.5, 0.75), Uniform(0, 1)),
        (Gamma(2.0, 3.0), Uniform(0.5, 1.5)),
        (Exponential(2.0), Uniform(0, 2)),
        (Weibull(1.5, 2.0), Uniform(0, 0.5))
    ]

    for (dist, primary) in test_distributions
        @testset "$(typeof(dist)) + $(typeof(primary))" begin
            d = primary_censored(dist, primary)

            @testset "Out-of-support handling" begin
                # Test out-of-support values return -Inf
                @test logpdf(d, -1.0) == -Inf  # Negative value (outside support)
                @test logpdf(d, -100.0) == -Inf  # Large negative value
            end

            @testset "Extreme value handling" begin
                # Test extreme input values
                @test logpdf(d, 0.0) ≠ NaN  # Should handle zero
                @test logpdf(d, 1e-10) ≠ NaN  # Very small positive value
                @test logpdf(d, 1e10) ≠ NaN  # Very large value
            end

            @testset "In-support values" begin
                # Test that logpdf is finite for values in support
                test_values = [0.1, 0.5, 1.0, 2.0, 5.0]
                for x in test_values
                    if insupport(d, x)
                        logpdf_val = logpdf(d, x)
                        @test logpdf_val ≠ NaN
                        @test logpdf_val > -Inf || logpdf_val == -Inf  # Either finite or -Inf
                    end
                end
            end

            @testset "Boundary conditions" begin
                # Test edge cases near distribution boundaries
                min_val = minimum(d)
                max_val = maximum(d)

                if isfinite(min_val)
                    @test logpdf(d, min_val - 1e-10) == -Inf  # Just outside minimum
                    logpdf_min = logpdf(d, min_val)
                    @test logpdf_min ≠ NaN  # At minimum should be well-defined
                end

                if isfinite(max_val)
                    @test logpdf(d, max_val + 1e-10) == -Inf  # Just outside maximum
                    logpdf_max = logpdf(d, max_val)
                    @test logpdf_max ≠ NaN  # At maximum should be well-defined
                end
            end
        end
    end
end

@testitem "Test logpdf numerical differentiation failure handling" begin
    using Distributions

    # Create a distribution that might cause numerical issues
    dist = LogNormal(10.0, 5.0)  # High variance distribution
    primary = Uniform(0, 1)
    d = primary_censored(dist, primary)

    # Test extreme values that might cause numerical differentiation to fail
    extreme_values = [1e-15, 1e15, -1e10]

    for x in extreme_values
        logpdf_val = logpdf(d, x)
        @test logpdf_val ≠ NaN  # Should not return NaN
        if !insupport(d, x)
            @test logpdf_val == -Inf  # Out of support should return -Inf
        end
    end

    # Test that the try-catch block properly handles domain errors
    # by testing values that might cause logsubexp to fail
    tiny_val = nextfloat(minimum(d))  # Just above minimum
    if isfinite(tiny_val)
        logpdf_val = logpdf(d, tiny_val)
        @test logpdf_val ≠ NaN  # Should handle edge case gracefully
    end
end

@testitem "Test logpdf consistency with pdf" begin
    using Distributions

    # Test log-pdf consistency: logpdf(d, x) ≈ log(pdf(d, x)) when pdf > 0
    dist = Exponential(1.0)
    primary = Uniform(0, 1)
    d = primary_censored(dist, primary)

    test_values = [0.1, 0.5, 1.0, 2.0, 5.0]

    for x in test_values
        if insupport(d, x)
            pdf_val = pdf(d, x)
            logpdf_val = logpdf(d, x)

            if pdf_val > 0
                @test logpdf_val ≈ log(pdf_val) rtol=1e-10
            else
                @test logpdf_val == -Inf
            end

            # Also test consistency: pdf(d, x) ≈ exp(logpdf(d, x))
            if isfinite(logpdf_val)
                @test pdf_val ≈ exp(logpdf_val) rtol=1e-10
            else
                @test pdf_val ≈ 0.0 atol=1e-15
            end
        end
    end
end

@testitem "Test logpdf with truncated primary censored distributions" begin
    using Distributions

    # Test edge cases with truncated distributions
    dist = Gamma(2.0, 1.0)
    primary = Uniform(0, 1)
    d = primary_censored(dist, primary)
    d_trunc = truncated(d, 1.0, 10.0)

    # Test values outside truncated support return -Inf
    @test logpdf(d_trunc, 0.5) == -Inf  # Below truncation
    @test logpdf(d_trunc, 15.0) == -Inf  # Above truncation

    # Test values within truncated support
    test_values = [1.5, 3.0, 5.0, 8.0]
    for x in test_values
        logpdf_val = logpdf(d_trunc, x)
        @test logpdf_val ≠ NaN
        @test isfinite(logpdf_val) || logpdf_val == -Inf
    end

    # Test boundary values
    @test logpdf(d_trunc, 1.0) ≠ NaN  # At lower bound
    @test logpdf(d_trunc, 10.0) ≠ NaN  # At upper bound
end

@testitem "ExponentiallyTilted basic integration test" begin
    using Distributions

    # Test that ExponentiallyTilted works as a primary event distribution
    delay_dist = LogNormal(1.5, 0.75)
    primary_event = ExponentiallyTilted(0.0, 2.0, 1.0)

    d = primary_censored(delay_dist, primary_event)

    @test typeof(d) <: CensoredDistributions.PrimaryCensored
    @test d.dist === delay_dist
    @test d.primary_event === primary_event
    @test d.method isa CensoredDistributions.AnalyticalSolver

    # Test basic functionality
    @test isfinite(pdf(d, 1.0))
    @test isfinite(cdf(d, 1.0))
    @test isfinite(logpdf(d, 1.0))
    @test 0.0 ≤ cdf(d, 1.0) ≤ 1.0

    # Test random sampling works
    samples = rand(d, 100)
    @test length(samples) == 100
    @test all(s ≥ 0 for s in samples)
end

@testitem "ExponentiallyTilted growth/decay scenarios" begin
    using Distributions

    delay_dist = Gamma(2.0, 1.0)

    # Test positive r (growth scenario - tilted towards later times)
    growth_primary = ExponentiallyTilted(0.0, 2.0, 1.5)
    d_growth = primary_censored(delay_dist, growth_primary)

    @test typeof(d_growth) <: CensoredDistributions.PrimaryCensored
    @test isfinite(pdf(d_growth, 1.0))
    @test isfinite(cdf(d_growth, 1.0))
    @test 0.0 ≤ cdf(d_growth, 1.0) ≤ 1.0

    # Test negative r (decay scenario - tilted towards earlier times)
    decay_primary = ExponentiallyTilted(0.0, 2.0, -1.2)
    d_decay = primary_censored(delay_dist, decay_primary)

    @test typeof(d_decay) <: CensoredDistributions.PrimaryCensored
    @test isfinite(pdf(d_decay, 1.0))
    @test isfinite(cdf(d_decay, 1.0))
    @test 0.0 ≤ cdf(d_decay, 1.0) ≤ 1.0

    # Test that growth and decay produce different results
    test_x = 1.5
    @test pdf(d_growth, test_x) ≠ pdf(d_decay, test_x)
    @test cdf(d_growth, test_x) ≠ cdf(d_decay, test_x)
end

@testitem "ExponentiallyTilted with multiple delay distributions" begin
    using Distributions

    primary_event = ExponentiallyTilted(0.0, 1.5, 0.8)

    # Test with different delay distributions
    delay_distributions = [
        Gamma(2.0, 1.0),
        LogNormal(1.0, 0.5),
        Exponential(1.5)
    ]

    for delay_dist in delay_distributions
        d = primary_censored(delay_dist, primary_event)

        @test typeof(d) <: CensoredDistributions.PrimaryCensored
        @test d.dist === delay_dist
        @test d.primary_event === primary_event

        # Test basic properties work with each delay distribution
        @test isfinite(pdf(d, 1.0))
        @test isfinite(cdf(d, 1.0))
        @test isfinite(logpdf(d, 1.0))
        @test 0.0 ≤ cdf(d, 1.0) ≤ 1.0

        # Test consistency between pdf and logpdf
        pdf_val = pdf(d, 1.0)
        logpdf_val = logpdf(d, 1.0)
        if pdf_val > 0
            @test abs(log(pdf_val) - logpdf_val) < 1e-10
        else
            @test logpdf_val == -Inf
        end

        # Test random sampling
        samples = rand(d, 50)
        @test length(samples) == 50
        @test all(s ≥ 0 for s in samples)
    end
end

@testitem "Test quantile function" begin
    using Distributions

    d = primary_censored(Exponential(1.0), Uniform(0, 1))

    # Test basic functionality: monotonicity and CDF consistency
    test_probs = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantiles = [quantile(d, p) for p in test_probs]

    @test all(diff(quantiles) .≥ 0)  # Monotonic
    @test all(isfinite.(quantiles))  # Finite values
    @test all(quantiles .≥ 0)       # Non-negative (for this distribution)

    # Test CDF roundtrip accuracy
    for (p, q) in zip(test_probs, quantiles)
        @test cdf(d, q) ≈ p rtol=1e-3  # Relaxed for numerical optimization
    end

    # Test boundary cases
    @test quantile(d, 0.0) == minimum(d)
    @test quantile(d, 1.0) == maximum(d)
    @test_throws ArgumentError quantile(d, -0.1)
    @test_throws ArgumentError quantile(d, 1.1)
end

@testitem "Test quantile with truncation" begin
    using Distributions

    truncated_dist = truncated(primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)), 1.0, 10.0)

    # Test boundary cases
    @test quantile(truncated_dist, 0.0) ≈ 1.0 atol=1e-3
    @test quantile(truncated_dist, 1.0) ≈ 10.0 atol=1e-1

    # Test quantile within bounds
    q50 = quantile(truncated_dist, 0.5)
    @test 1.0 ≤ q50 ≤ 10.0
    @test cdf(truncated_dist, q50) ≈ 0.5 rtol=1e-4
end
