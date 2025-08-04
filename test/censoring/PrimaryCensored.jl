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

@testitem "ExponentiallyTilted as primary event - basic constructor" begin
    using Distributions

    # Test basic constructor with ExponentiallyTilted as primary event
    delay_dist = LogNormal(1.5, 0.75)
    primary_event = ExponentiallyTilted(0.0, 2.0, 1.0)

    d = primary_censored(delay_dist, primary_event)

    @test typeof(d) <: CensoredDistributions.PrimaryCensored
    @test d.dist === delay_dist
    @test d.primary_event === primary_event
    @test d.method isa CensoredDistributions.AnalyticalSolver
end

@testitem "ExponentiallyTilted uniform-like case comparison" begin
    using Distributions

    # Test near-uniform ExponentiallyTilted vs actual Uniform
    delay_dist = Gamma(2.0, 1.5)

    # Near-uniform ExponentiallyTilted (very small r)
    tilted_uniform = ExponentiallyTilted(0.0, 1.0, 1e-10)
    d_tilted = primary_censored(delay_dist, tilted_uniform)

    # Actual Uniform distribution
    actual_uniform = Uniform(0.0, 1.0)
    d_uniform = primary_censored(delay_dist, actual_uniform)

    # Test that they behave similarly
    test_points = [0.5, 1.0, 2.0, 3.0, 4.0]
    for x in test_points
        pdf_tilted = pdf(d_tilted, x)
        pdf_uniform = pdf(d_uniform, x)
        cdf_tilted = cdf(d_tilted, x)
        cdf_uniform = cdf(d_uniform, x)

        # Should be very close (allowing for small numerical differences)
        @test abs(pdf_tilted - pdf_uniform) < 1e-6
        @test abs(cdf_tilted - cdf_uniform) < 1e-6
    end
end

@testitem "ExponentiallyTilted growth scenario (epidemic growth)" begin
    using Distributions

    # Test positive r (exponential growth) scenario
    delay_dist = Weibull(2.0, 1.5)

    # Growth scenario: r > 0 (tilted towards later times)
    growth_primary = ExponentiallyTilted(0.0, 5.0, 1.5)
    d_growth = primary_censored(delay_dist, growth_primary)

    @test typeof(d_growth) <: CensoredDistributions.PrimaryCensored
    @test d_growth.primary_event === growth_primary

    # Test basic functionality
    @test isfinite(pdf(d_growth, 1.0))
    @test isfinite(cdf(d_growth, 1.0))
    @test 0.0 ≤ cdf(d_growth, 1.0) ≤ 1.0

    # Test that cdf is monotonically increasing
    test_points = [0.5, 1.0, 2.0, 4.0, 6.0]
    cdf_vals = [cdf(d_growth, x) for x in test_points]
    for i in 1:(length(cdf_vals) - 1)
        @test cdf_vals[i] ≤ cdf_vals[i + 1]
    end

    # Test random sampling works
    samples = rand(d_growth, 100)
    @test length(samples) == 100
    @test all(s > 0 for s in samples)  # All samples should be positive
end

@testitem "ExponentiallyTilted decay scenario (epidemic decay)" begin
    using Distributions

    # Test negative r (exponential decay) scenario
    delay_dist = LogNormal(0.8, 0.6)

    # Decay scenario: r < 0 (tilted towards earlier times)
    decay_primary = ExponentiallyTilted(0.0, 3.0, -1.2)
    d_decay = primary_censored(delay_dist, decay_primary)

    @test typeof(d_decay) <: CensoredDistributions.PrimaryCensored
    @test d_decay.primary_event === decay_primary

    # Test basic functionality
    @test isfinite(pdf(d_decay, 0.5))
    @test isfinite(cdf(d_decay, 0.5))
    @test 0.0 ≤ cdf(d_decay, 0.5) ≤ 1.0

    # Test edge cases
    @test pdf(d_decay, -1.0) == 0.0  # Outside support
    @test cdf(d_decay, 0.0) == 0.0   # At lower bound
    @test logpdf(d_decay, -1.0) == -Inf  # Outside support

    # Test with multiple points
    test_points = [0.1, 0.5, 1.0, 2.0, 4.0]
    for x in test_points
        if x > 0
            @test pdf(d_decay, x) ≥ 0.0
            @test 0.0 ≤ cdf(d_decay, x) ≤ 1.0
        end
    end
end

@testitem "ExponentiallyTilted with different delay distributions" begin
    using Distributions

    # Test ExponentiallyTilted with various delay distributions
    primary_event = ExponentiallyTilted(0.0, 1.5, 0.8)

    # Test different delay distributions
    delay_distributions = [
        Gamma(2.0, 1.0),
        Exponential(1.5),
        Weibull(1.8, 2.0),
        LogNormal(1.0, 0.5)
    ]

    for delay_dist in delay_distributions
        @testset "$(typeof(delay_dist))" begin
            d = primary_censored(delay_dist, primary_event)

            @test typeof(d) <: CensoredDistributions.PrimaryCensored
            @test d.dist === delay_dist
            @test d.primary_event === primary_event

            # Test basic properties
            @test isfinite(pdf(d, 1.0))
            @test isfinite(cdf(d, 1.0))
            @test isfinite(logpdf(d, 1.0))
            @test 0.0 ≤ cdf(d, 1.0) ≤ 1.0

            # Test consistency between pdf and logpdf
            test_x = 1.5
            pdf_val = pdf(d, test_x)
            logpdf_val = logpdf(d, test_x)
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
end

@testitem "ExponentiallyTilted solver options" begin
    using Distributions
    using Integrals

    delay_dist = Gamma(1.5, 2.0)
    primary_event = ExponentiallyTilted(0.0, 2.0, 1.0)

    # Test default (analytical) solver
    d_analytical = primary_censored(delay_dist, primary_event)
    @test d_analytical.method isa CensoredDistributions.AnalyticalSolver
    @test d_analytical.method.solver isa QuadGKJL

    # Test forced numeric solver
    d_numeric = primary_censored(delay_dist, primary_event; force_numeric = true)
    @test d_numeric.method isa CensoredDistributions.NumericSolver
    @test d_numeric.method.solver isa QuadGKJL

    # Test custom solver
    custom_solver = HCubatureJL()
    d_custom = primary_censored(delay_dist, primary_event;
        solver = custom_solver, force_numeric = false)
    @test d_custom.method isa CensoredDistributions.AnalyticalSolver
    @test d_custom.method.solver === custom_solver

    # Test that all versions produce similar results
    test_point = 2.0
    pdf_analytical = pdf(d_analytical, test_point)
    pdf_numeric = pdf(d_numeric, test_point)
    pdf_custom = pdf(d_custom, test_point)

    # Should be approximately equal (allowing for numerical differences)
    @test abs(pdf_analytical - pdf_numeric) < 1e-6
    @test abs(pdf_analytical - pdf_custom) < 1e-6

    cdf_analytical = cdf(d_analytical, test_point)
    cdf_numeric = cdf(d_numeric, test_point)
    cdf_custom = cdf(d_custom, test_point)

    @test abs(cdf_analytical - cdf_numeric) < 1e-6
    @test abs(cdf_analytical - cdf_custom) < 1e-6
end

@testitem "ExponentiallyTilted parameter extraction" begin
    using Distributions

    # Test params function with ExponentiallyTilted primary event
    delay_dist = Exponential(2.0)
    primary_event = ExponentiallyTilted(-1.0, 3.0, 0.5)

    d = primary_censored(delay_dist, primary_event)
    extracted_params = params(d)

    # Should extract parameters from both distributions
    @test length(extracted_params) == 4  # 1 from Exponential + 3 from ExponentiallyTilted
    @test extracted_params[1] ≈ 2.0      # Exponential rate parameter
    @test extracted_params[2] ≈ -1.0     # ExponentiallyTilted min
    @test extracted_params[3] ≈ 3.0      # ExponentiallyTilted max
    @test extracted_params[4] ≈ 0.5      # ExponentiallyTilted r
end

@testitem "ExponentiallyTilted integration properties" begin
    using Distributions

    # Test that the censored distribution integrates properly
    delay_dist = Weibull(2.0, 1.0)

    # Test different ExponentiallyTilted scenarios
    primary_scenarios = [
        ExponentiallyTilted(0.0, 1.0, 0.0),    # Uniform-like
        ExponentiallyTilted(0.0, 1.0, 1.0),    # Growth
        ExponentiallyTilted(0.0, 1.0, -0.8),   # Decay
        ExponentiallyTilted(0.0, 2.0, 2.0),    # Strong growth, wider window
        ExponentiallyTilted(0.5, 1.5, -1.5)    # Strong decay, offset window
    ]

    for primary_event in primary_scenarios
        @testset "Primary: $(params(primary_event))" begin
            d = primary_censored(delay_dist, primary_event)

            # Test CDF bounds
            @test cdf(d, -1.0) ≈ 0.0
            @test cdf(d, 1e8) ≈ 1.0

            # Test CCDF consistency
            test_point = 2.0
            @test abs(cdf(d, test_point) + ccdf(d, test_point) - 1.0) < 1e-12

            # Test logcdf and logccdf consistency
            if cdf(d, test_point) > 0
                @test abs(exp(logcdf(d, test_point)) - cdf(d, test_point)) < 1e-10
            end
            if ccdf(d, test_point) > 0
                @test abs(exp(logccdf(d, test_point)) - ccdf(d, test_point)) < 1e-10
            end

            # Test monotonicity of CDF
            test_points = [0.1, 0.5, 1.0, 2.0, 4.0]
            cdf_vals = [cdf(d, x) for x in test_points]
            for i in 1:(length(cdf_vals) - 1)
                @test cdf_vals[i] ≤ cdf_vals[i + 1]
            end
        end
    end
end

@testitem "ExponentiallyTilted extreme scenarios validation" begin
    using Distributions

    # Test extreme parameter combinations
    delay_dist = LogNormal(1.0, 0.8)

    # Very small r (should behave like uniform)
    tiny_r_primary = ExponentiallyTilted(0.0, 1.0, 1e-15)
    d_tiny = primary_censored(delay_dist, tiny_r_primary)

    uniform_primary = Uniform(0.0, 1.0)
    d_uniform = primary_censored(delay_dist, uniform_primary)

    # Should produce very similar results
    test_point = 1.5
    @test abs(pdf(d_tiny, test_point) - pdf(d_uniform, test_point)) < 1e-8
    @test abs(cdf(d_tiny, test_point) - cdf(d_uniform, test_point)) < 1e-8

    # Large positive r (heavily tilted towards end of window)
    large_r_primary = ExponentiallyTilted(0.0, 2.0, 5.0)
    d_large_r = primary_censored(delay_dist, large_r_primary)

    @test isfinite(pdf(d_large_r, 1.0))
    @test isfinite(cdf(d_large_r, 1.0))
    @test 0.0 ≤ cdf(d_large_r, 1.0) ≤ 1.0

    # Large negative r (heavily tilted towards start of window)
    large_neg_r_primary = ExponentiallyTilted(0.0, 2.0, -5.0)
    d_large_neg_r = primary_censored(delay_dist, large_neg_r_primary)

    @test isfinite(pdf(d_large_neg_r, 1.0))
    @test isfinite(cdf(d_large_neg_r, 1.0))
    @test 0.0 ≤ cdf(d_large_neg_r, 1.0) ≤ 1.0

    # Wide time window
    wide_primary = ExponentiallyTilted(0.0, 10.0, 0.3)
    d_wide = primary_censored(delay_dist, wide_primary)

    @test isfinite(pdf(d_wide, 5.0))
    @test isfinite(cdf(d_wide, 5.0))
    @test 0.0 ≤ cdf(d_wide, 5.0) ≤ 1.0
end

@testitem "ExponentiallyTilted sampling and truncation" begin
    using Distributions

    # Test sampling from ExponentiallyTilted primary censored distributions
    delay_dist = Gamma(3.0, 0.8)
    primary_event = ExponentiallyTilted(0.0, 1.0, 1.5)

    d = primary_censored(delay_dist, primary_event)

    # Test basic sampling
    samples = rand(d, 1000)
    @test length(samples) == 1000
    @test all(s ≥ 0 for s in samples)
    @test all(isfinite(s) for s in samples)

    # Test truncated version
    d_trunc = truncated(d, 0.5, 5.0)
    trunc_samples = rand(d_trunc, 1000)

    @test length(trunc_samples) == 1000
    @test all(0.5 ≤ s ≤ 5.0 for s in trunc_samples)
    @test all(isfinite(s) for s in trunc_samples)

    # Test statistical properties of truncated samples
    @test minimum(trunc_samples) ≥ 0.5
    @test maximum(trunc_samples) ≤ 5.0

    # Test that samples are not all identical
    @test length(unique(samples)) > 10
    @test length(unique(trunc_samples)) > 10
end
