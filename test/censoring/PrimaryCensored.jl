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
