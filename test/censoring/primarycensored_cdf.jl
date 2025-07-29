@testitem "PrimaryCensored CDF Dispatch" begin
    using Distributions

    # Create a custom distribution type to test dispatch
    struct TestDistribution <: ContinuousUnivariateDistribution end
    Base.minimum(::TestDistribution) = 0.0
    Base.maximum(::TestDistribution) = Inf
    Distributions.cdf(::TestDistribution, x::Real) = 1 - exp(-x)  # Simple exponential CDF
    Distributions.logcdf(::TestDistribution, x::Real) = log(1 - exp(-x))
    Distributions.pdf(::TestDistribution, x::Real) = exp(-x)
    Distributions.logpdf(::TestDistribution, x::Real) = -x

    # Track whether analytical method was called
    analytical_called = Ref(false)
    numerical_called = Ref(false)

    # Define a dummy analytical implementation that sets a flag
    function CensoredDistributions.primarycensored_cdf(
            ::TestDistribution, ::Uniform, x::Real, ::CensoredDistributions.AnalyticalSolver
    )
        analytical_called[] = true
        return 0.12345  # Dummy value to verify it's being used
    end

    # Override numerical to track calls
    original_numerical = CensoredDistributions.primarycensored_cdf
    function CensoredDistributions.primarycensored_cdf(
            dist::TestDistribution, primary::Uniform, x::Real, method::CensoredDistributions.NumericSolver
    )
        numerical_called[] = true
        # Call the original numerical implementation
        return invoke(
            CensoredDistributions.primarycensored_cdf,
            Tuple{UnivariateDistribution, UnivariateDistribution,
                Real, CensoredDistributions.NumericSolver},
            dist, primary, x, method
        )
    end

    @testset "Dispatch to analytical method" begin
        # Reset flags
        analytical_called[] = false
        numerical_called[] = false

        # Create distribution with analytical solver (default)
        d = primary_censored(TestDistribution(), Uniform(0.0, 1.0))

        # Call CDF
        result = cdf(d, 2.0)

        # Verify analytical was called and numerical was not
        @test analytical_called[] == true
        @test numerical_called[] == false
        @test result == 0.12345  # Verify our dummy value was returned
    end

    @testset "Force numerical method" begin
        # Reset flags
        analytical_called[] = false
        numerical_called[] = false

        # Create distribution with forced numerical
        d = primary_censored(TestDistribution(), Uniform(0.0, 1.0); force_numeric = true)

        # Call CDF
        result = cdf(d, 2.0)

        # Verify numerical was called and analytical was not
        @test analytical_called[] == false
        @test numerical_called[] == true
        @test result != 0.12345  # Should not be our dummy value
    end

    @testset "Fallback for unsupported distributions" begin
        # For Exponential (no analytical implementation), it should use numerical
        d = primary_censored(Exponential(2.0), Uniform(0.0, 1.0))

        # This should work without error (falls back to numerical)
        result = cdf(d, 2.0)
        @test 0 < result < 1
    end

    @testset "Solver is actually used" begin
        # Create a broken solver that will error when used
        struct BrokenSolver end

        # For a distribution without analytical solution, the solver should be used
        d = primary_censored(Exponential(2.0), Uniform(0.0, 1.0); solver = BrokenSolver())

        # This should error because BrokenSolver doesn't implement the required interface
        @test_throws MethodError cdf(d, 2.0)

        # For a distribution with analytical solution but forced numeric
        d_forced = primary_censored(Gamma(2.0, 3.0), Uniform(0.0, 1.0);
            solver = BrokenSolver(), force_numeric = true)

        # This should also error because we're forcing numerical
        @test_throws MethodError cdf(d_forced, 2.0)

        # But with analytical solution and not forced, it should work
        d_analytical = primary_censored(Gamma(2.0, 3.0), Uniform(0.0, 1.0);
            solver = BrokenSolver())

        # This should work because analytical doesn't use the solver
        result = cdf(d_analytical, 2.0)
        @test 0 < result < 1
    end
end

@testitem "Gamma + Uniform analytical vs numerical agreement" begin
    using Distributions
    using Random

    Random.seed!(1234)

    # Test parameters
    rtol = 1e-6
    x_vals = range(0.1, 20, 50)

    # Create distributions
    dist = Gamma(2.0, 3.0)
    primary = Uniform(0.0, 1.0)

    # Analytical solution (default)
    d_analytical = primary_censored(dist, primary)

    # Numerical solution
    d_numerical = primary_censored(dist, primary; force_numeric = true)

    # Compare CDFs
    cdf_analytical = cdf.(Ref(d_analytical), x_vals)
    cdf_numerical = cdf.(Ref(d_numerical), x_vals)

    @test all(isapprox.(cdf_analytical, cdf_numerical, rtol = rtol))

    # Test log versions
    logcdf_analytical = logcdf.(Ref(d_analytical), x_vals)
    logcdf_numerical = logcdf.(Ref(d_numerical), x_vals)

    @test all(isapprox.(logcdf_analytical, logcdf_numerical, rtol = rtol))
end

@testitem "LogNormal + Uniform analytical vs numerical agreement" begin
    using Distributions
    using Random

    Random.seed!(1234)

    # Test parameters
    rtol = 1e-6
    x_vals = range(0.1, 20, 50)

    # Create distributions
    dist = LogNormal(1.5, 0.75)
    primary = Uniform(0.0, 1.0)

    # Analytical solution
    d_analytical = primary_censored(dist, primary)

    # Numerical solution
    d_numerical = primary_censored(dist, primary; force_numeric = true)

    # Compare CDFs
    cdf_analytical = cdf.(Ref(d_analytical), x_vals)
    cdf_numerical = cdf.(Ref(d_numerical), x_vals)

    @test all(isapprox.(cdf_analytical, cdf_numerical, rtol = rtol))
end

@testitem "Weibull + Uniform analytical vs numerical agreement" begin
    using Distributions
    using Random

    Random.seed!(1234)

    # Test parameters
    rtol = 1e-6
    x_vals = range(0.1, 20, 50)

    # Create distributions
    dist = Weibull(2.0, 1.5)
    primary = Uniform(0.0, 1.0)

    # Analytical solution
    d_analytical = primary_censored(dist, primary)

    # Numerical solution
    d_numerical = primary_censored(dist, primary; force_numeric = true)

    # Compare CDFs
    cdf_analytical = cdf.(Ref(d_analytical), x_vals)
    cdf_numerical = cdf.(Ref(d_numerical), x_vals)

    @test all(isapprox.(cdf_analytical, cdf_numerical, rtol = rtol))
end

@testitem "Non-uniform primary event window analytical vs numerical agreement" begin
    using Distributions
    using Random

    Random.seed!(1234)

    # Test parameters
    rtol = 1e-6

    # Test with non-zero minimum for primary event
    dist = Gamma(2.0, 3.0)
    primary = Uniform(2.0, 3.0)  # Window from 2 to 3

    d_analytical = primary_censored(dist, primary)
    d_numerical = primary_censored(dist, primary; force_numeric = true)

    # Adjust x_vals for the shifted window
    x_vals_shifted = range(2.1, 22, 50)

    cdf_analytical = cdf.(Ref(d_analytical), x_vals_shifted)
    cdf_numerical = cdf.(Ref(d_numerical), x_vals_shifted)

    @test all(isapprox.(cdf_analytical, cdf_numerical, rtol = rtol))
end

@testitem "PrimaryCensored Analytical Performance" begin
    using Distributions
    using BenchmarkTools

    # Test distributions
    test_cases = [
        (Gamma(2.0, 3.0), "Gamma"),
        (LogNormal(1.5, 0.75), "LogNormal"),
        (Weibull(2.0, 1.5), "Weibull")
    ]

    @testset "Performance improvement" begin
        for (dist, name) in test_cases
            @testset "$(name) speedup" begin
                primary = Uniform(0.0, 1.0)

                # Create analytical and numerical versions
                d_analytical = primary_censored(dist, primary)
                d_numerical = primary_censored(dist, primary; force_numeric = true)
                # Sample vectorised values to use for benchmarking
                x_vals = rand(dist, 100)

                # Create benchmarks with proper setup to avoid compilation overhead
                b_analytical = @benchmarkable begin
                    for x in $x_vals
                        cdf($d_analytical, x)
                    end
                end

                b_numerical = @benchmarkable begin
                    for x in $x_vals
                        cdf($d_numerical, x)
                    end
                end

                # Run benchmarks with proper parameters to account for setup time
                t_analytical = run(b_analytical, samples = 10)
                t_numerical = run(b_numerical, samples = 10)

                # Compare median times to avoid outliers
                analytical_time = median(t_analytical).time
                numerical_time = median(t_numerical).time

                # Analytical should be faster
                speedup = numerical_time / analytical_time
                @test speedup > 2  # More conservative threshold to account for variability
                @info "$(name) speedup: $(round(speedup, digits=1))x ($(analytical_time/1e6) ms vs $(numerical_time/1e6) ms)"
            end
        end
    end
end

@testitem "PrimaryCensored Edge Cases" begin
    using Distributions

    @testset "Edge case handling" begin
        dist = Gamma(2.0, 3.0)
        primary = Uniform(0.0, 1.0)

        d = primary_censored(dist, primary)

        # Test x = 0
        @test cdf(d, 0.0) == 0.0

        # Test x < 0
        @test cdf(d, -1.0) == 0.0

        # Test large x
        @test cdf(d, 1000.0)≈1.0 atol=1e-6

        # Test log versions
        @test logcdf(d, 0.0) == -Inf
        @test isfinite(logcdf(d, 5.0))

        # Test ccdf
        @test ccdf(d, 0.0) == 1.0
        @test ccdf(d, 1000.0)≈0.0 atol=1e-6
    end
end

@testitem "PrimaryCensored numerical integration bounds" begin
    using CensoredDistributions
    using Distributions
    using Test

    # Test 1: Standard case with distributions starting at 0
    @testset "Standard distributions" begin
        pc = CensoredDistributions.primary_censored(
            Exponential(1.0), Uniform(0.0, 1.0); force_numeric = true)

        # Should work for various x values
        @test cdf(pc, 0.5) ≥ 0.0
        @test cdf(pc, 1.0) ≥ cdf(pc, 0.5)
        @test cdf(pc, 5.0) ≤ 1.0
    end

    # Test 2: Verify non-zero minimum support is rejected
    @testset "Non-zero minimum support check" begin
        # Uniform delay on [2, 5] should be rejected
        delay_dist = Uniform(2.0, 5.0)
        primary_event = Uniform(0.0, 1.0)

        # Should throw an error for non-zero minimum
        @test_throws ArgumentError CensoredDistributions.primary_censored(
            delay_dist, primary_event)
    end

    # Test 3: Edge case with small x values
    @testset "Small x values" begin
        pc = CensoredDistributions.primary_censored(
            Gamma(2.0, 1.0), Uniform(0.0, 2.0); force_numeric = true)

        # Should handle small x gracefully
        @test cdf(pc, 0.1) ≥ 0.0
        @test cdf(pc, 0.5) ≥ cdf(pc, 0.1)

        # Test that the integration bounds are valid internally
        # This would have failed with the old bounds calculation
        @test !isnan(cdf(pc, 0.5))
        @test !isinf(cdf(pc, 0.5))
    end

    # Test 4: Compare with analytical solution where available
    @testset "Numerical vs Analytical consistency" begin
        # Use Gamma + Uniform which has analytical solution
        delay_dist = Gamma(2.0, 3.0)
        primary_event = Uniform(0.0, 1.0)

        pc_analytical = CensoredDistributions.primary_censored(delay_dist, primary_event)
        pc_numerical = CensoredDistributions.primary_censored(
            delay_dist, primary_event; force_numeric = true)

        x_values = [0.5, 1.0, 2.0, 5.0, 10.0]
        for x in x_values
            @test cdf(pc_analytical, x) ≈ cdf(pc_numerical, x) rtol=1e-6
        end
    end
end
