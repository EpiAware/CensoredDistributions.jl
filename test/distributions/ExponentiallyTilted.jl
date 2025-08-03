@testitem "ExponentiallyTilted constructor validation" begin
    using Distributions

    # Test valid construction
    d1 = ExponentiallyTilted(0.0, 1.0, 1.5)
    @test typeof(d1) <: ExponentiallyTilted
    @test d1.min == 0.0
    @test d1.max == 1.0
    @test d1.r == 1.5

    # Test different parameter types are promoted correctly
    d2 = ExponentiallyTilted(0, 1.0, 2)
    @test eltype(d2) == Float64
    @test d2.min == 0.0
    @test d2.max == 1.0
    @test d2.r == 2.0

    # Test with negative r (exponentially decreasing)
    d3 = ExponentiallyTilted(-1.0, 2.0, -0.5)
    @test d3.min == -1.0
    @test d3.max == 2.0
    @test d3.r == -0.5

    # Test with very small r (near-uniform case)
    d4 = ExponentiallyTilted(0.0, 1.0, 1e-12)
    @test d4.r == 1e-12

    # Test error cases
    @test_throws ArgumentError ExponentiallyTilted(1.0, 1.0, 1.0)  # max == min
    @test_throws ArgumentError ExponentiallyTilted(2.0, 1.0, 1.0)  # max < min
    @test_throws ArgumentError ExponentiallyTilted(Inf, 1.0, 1.0)  # infinite min
    @test_throws ArgumentError ExponentiallyTilted(0.0, Inf, 1.0)  # infinite max
    @test_throws ArgumentError ExponentiallyTilted(0.0, 1.0, Inf)  # infinite r
    @test_throws ArgumentError ExponentiallyTilted(0.0, 1.0, NaN)  # NaN r
end

@testitem "ExponentiallyTilted basic interface methods" begin
    using Distributions

    d = ExponentiallyTilted(-2.0, 3.0, 1.2)

    # Test params function
    p = params(d)
    @test p == (-2.0, 3.0, 1.2)

    # Test support bounds
    @test minimum(d) == -2.0
    @test maximum(d) == 3.0

    # Test insupport
    @test insupport(d, -2.0) == true   # Lower bound
    @test insupport(d, 3.0) == true    # Upper bound
    @test insupport(d, 0.0) == true    # Middle value
    @test insupport(d, -2.1) == false  # Below lower bound
    @test insupport(d, 3.1) == false   # Above upper bound

    # Test eltype
    @test eltype(d) == Float64
    @test eltype(ExponentiallyTilted(0.0, 1.0, 1.0)) == Float64
end

@testitem "ExponentiallyTilted PDF properties" begin
    using Distributions

    # Test exponentially increasing distribution
    d_inc = ExponentiallyTilted(0.0, 1.0, 2.0)

    # PDF should be positive within support
    @test pdf(d_inc, 0.0) > 0.0
    @test pdf(d_inc, 0.5) > 0.0
    @test pdf(d_inc, 1.0) > 0.0

    # PDF should be zero outside support
    @test pdf(d_inc, -0.1) == 0.0
    @test pdf(d_inc, 1.1) == 0.0

    # For increasing r, PDF should increase with x
    @test pdf(d_inc, 0.1) < pdf(d_inc, 0.5)
    @test pdf(d_inc, 0.5) < pdf(d_inc, 0.9)

    # Test exponentially decreasing distribution
    d_dec = ExponentiallyTilted(0.0, 1.0, -2.0)

    # For decreasing r, PDF should decrease with x
    @test pdf(d_dec, 0.1) > pdf(d_dec, 0.5)
    @test pdf(d_dec, 0.5) > pdf(d_dec, 0.9)

    # Test near-uniform case (small r)
    d_uniform = ExponentiallyTilted(0.0, 1.0, 1e-10)
    uniform_pdf = 1.0  # 1/(1-0)

    @test abs(pdf(d_uniform, 0.1) - uniform_pdf) < 1e-8
    @test abs(pdf(d_uniform, 0.5) - uniform_pdf) < 1e-8
    @test abs(pdf(d_uniform, 0.9) - uniform_pdf) < 1e-8
end

@testitem "ExponentiallyTilted logPDF properties" begin
    using Distributions

    d = ExponentiallyTilted(0.0, 2.0, 1.5)

    # LogPDF should be -Inf outside support
    @test logpdf(d, -0.1) == -Inf
    @test logpdf(d, 2.1) == -Inf

    # LogPDF should be finite within support
    @test isfinite(logpdf(d, 0.0))
    @test isfinite(logpdf(d, 1.0))
    @test isfinite(logpdf(d, 2.0))

    # Consistency with PDF
    x_vals = [0.1, 0.5, 1.0, 1.5, 1.9]
    for x in x_vals
        @test abs(exp(logpdf(d, x)) - pdf(d, x)) < 1e-12
    end

    # Test near-uniform case
    d_uniform = ExponentiallyTilted(0.0, 1.0, 1e-12)
    expected_logpdf = -log(1.0)  # -log(max - min)
    @test abs(logpdf(d_uniform, 0.5) - expected_logpdf) < 1e-10
end

@testitem "ExponentiallyTilted CDF properties" begin
    using Distributions

    d = ExponentiallyTilted(0.0, 1.0, 1.0)

    # CDF bounds
    @test cdf(d, -0.1) == 0.0  # Below support
    @test cdf(d, 1.1) == 1.0   # Above support
    @test cdf(d, 0.0) == 0.0   # Lower bound
    @test cdf(d, 1.0) == 1.0   # Upper bound

    # CDF should be monotonically increasing
    x_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
    cdf_vals = [cdf(d, x) for x in x_vals]
    @test all(cdf_vals[i] <= cdf_vals[i + 1] for i in 1:(length(cdf_vals) - 1))

    # CDF should be between 0 and 1
    for x in x_vals
        @test 0.0 <= cdf(d, x) <= 1.0
    end

    # Test near-uniform case
    d_uniform = ExponentiallyTilted(0.0, 1.0, 1e-12)
    @test abs(cdf(d_uniform, 0.5) - 0.5) < 1e-10  # Should be like uniform
    @test abs(cdf(d_uniform, 0.25) - 0.25) < 1e-10
end

@testitem "ExponentiallyTilted logCDF properties" begin
    using Distributions

    d = ExponentiallyTilted(0.0, 1.0, 1.5)

    # LogCDF should be -Inf at boundaries
    @test logcdf(d, -0.1) == -Inf  # Below support
    @test logcdf(d, 0.0) == -Inf   # Lower bound
    @test logcdf(d, 1.0) == 0.0    # Upper bound (log(1) = 0)

    # LogCDF should be finite within support (except at bounds)
    @test isfinite(logcdf(d, 0.5))
    @test isfinite(logcdf(d, 0.9))

    # Consistency with CDF
    x_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
    for x in x_vals
        cdf_val = cdf(d, x)
        if cdf_val > 0.0
            @test abs(exp(logcdf(d, x)) - cdf_val) < 1e-12
        end
    end
end

@testitem "ExponentiallyTilted quantile function" begin
    using Distributions

    d = ExponentiallyTilted(0.0, 1.0, 1.0)

    # Test boundary cases
    @test quantile(d, 0.0) == 0.0
    @test quantile(d, 1.0) == 1.0

    # Test quantile-CDF consistency
    p_vals = [0.1, 0.25, 0.5, 0.75, 0.9]
    for p in p_vals
        q = quantile(d, p)
        @test abs(cdf(d, q) - p) < 1e-10
    end

    # Test error for invalid probabilities
    @test_throws ArgumentError quantile(d, -0.1)
    @test_throws ArgumentError quantile(d, 1.1)

    # Test near-uniform case
    d_uniform = ExponentiallyTilted(0.0, 1.0, 1e-12)
    @test abs(quantile(d_uniform, 0.5) - 0.5) < 1e-10
    @test abs(quantile(d_uniform, 0.25) - 0.25) < 1e-10
end

@testitem "ExponentiallyTilted random number generation" begin
    using Distributions
    using Random

    Random.seed!(123)
    d = ExponentiallyTilted(0.0, 1.0, 1.0)

    # Test single random number
    r1 = rand(d)
    @test 0.0 <= r1 <= 1.0

    # Test multiple random numbers
    samples = [rand(d) for _ in 1:100]
    @test all(0.0 <= s <= 1.0 for s in samples)

    # Test that samples are not all the same
    @test length(unique(samples)) > 1

    # Test with custom RNG
    rng = MersenneTwister(456)
    r2 = rand(rng, d)
    @test 0.0 <= r2 <= 1.0
end

@testitem "ExponentiallyTilted PDF integration to 1" begin
    using Distributions

    # Test numerical integration of PDF equals 1
    # Using simple trapezoidal rule for verification
    function integrate_pdf(d, n_points = 1000)
        x_vals = range(minimum(d), maximum(d), length = n_points)
        dx = (maximum(d) - minimum(d)) / (n_points - 1)
        pdf_vals = [pdf(d, x) for x in x_vals]
        # Trapezoidal rule
        return dx * (0.5 * (pdf_vals[1] + pdf_vals[end]) + sum(pdf_vals[2:(end - 1)]))
    end

    # Test different parameter values
    test_cases = [
        (0.0, 1.0, 2.0),    # Increasing
        (0.0, 1.0, -1.5),   # Decreasing
        (-1.0, 2.0, 0.8),   # Different bounds
        (0.0, 1.0, 1e-10)  # Near uniform
    ]

    for (min_val, max_val, r_val) in test_cases
        d = ExponentiallyTilted(min_val, max_val, r_val)
        integral = integrate_pdf(d)
        @test abs(integral - 1.0) < 1e-4  # Should integrate to 1 (looser tolerance for numerical integration)
    end
end

@testitem "ExponentiallyTilted numerical stability with extreme parameters" begin
    using Distributions

    # Test with very large positive r
    d_large_r = ExponentiallyTilted(0.0, 1.0, 10.0)
    @test isfinite(pdf(d_large_r, 0.5))
    @test isfinite(cdf(d_large_r, 0.5))
    @test 0.0 <= cdf(d_large_r, 0.5) <= 1.0

    # Test with very large negative r
    d_neg_r = ExponentiallyTilted(0.0, 1.0, -10.0)
    @test isfinite(pdf(d_neg_r, 0.5))
    @test isfinite(cdf(d_neg_r, 0.5))
    @test 0.0 <= cdf(d_neg_r, 0.5) <= 1.0

    # Test with very small r (should behave like uniform)
    d_small_r = ExponentiallyTilted(0.0, 1.0, 1e-15)
    uniform_pdf = 1.0
    uniform_cdf_mid = 0.5

    @test abs(pdf(d_small_r, 0.5) - uniform_pdf) < 1e-10
    @test abs(cdf(d_small_r, 0.5) - uniform_cdf_mid) < 1e-10

    # Test boundary values don't cause numerical issues
    @test isfinite(pdf(d_large_r, 0.0))
    @test isfinite(pdf(d_large_r, 1.0))
    @test pdf(d_large_r, 0.0) >= 0.0
    @test pdf(d_large_r, 1.0) >= 0.0
end

@testitem "ExponentiallyTilted reduces to uniform when r ≈ 0" begin
    using Distributions

    # Test with various small r values
    small_r_values = [1e-8, 1e-10, 1e-12, 1e-14]

    for r in small_r_values
        d = ExponentiallyTilted(0.0, 2.0, r)
        uniform_d = Uniform(0.0, 2.0)

        # Test points within the support
        test_points = [0.1, 0.5, 1.0, 1.5, 1.9]

        for x in test_points
            # PDF should be approximately uniform
            @test abs(pdf(d, x) - pdf(uniform_d, x)) < 1e-8

            # CDF should be approximately uniform
            @test abs(cdf(d, x) - cdf(uniform_d, x)) < 1e-8
        end

        # Quantiles should be approximately uniform
        p_vals = [0.1, 0.25, 0.5, 0.75, 0.9]
        for p in p_vals
            @test abs(quantile(d, p) - quantile(uniform_d, p)) < 1e-6
        end
    end
end

@testitem "ExponentiallyTilted integration with primary_censored" begin
    using Distributions

    # Test that ExponentiallyTilted works as a primary event distribution
    delay_dist = LogNormal(1.0, 0.5)
    primary_dist = ExponentiallyTilted(0.0, 2.0, 1.0)

    # This should not throw an error
    pc_dist = primary_censored(delay_dist, primary_dist)
    @test typeof(pc_dist) <: CensoredDistributions.PrimaryCensored
    @test pc_dist.primary_event === primary_dist
    @test pc_dist.dist === delay_dist

    # Test basic functionality works
    @test isfinite(pdf(pc_dist, 1.0))
    @test isfinite(cdf(pc_dist, 1.0))
    @test 0.0 <= cdf(pc_dist, 1.0) <= 1.0

    # Test with near-uniform primary event distribution
    primary_uniform = ExponentiallyTilted(0.0, 1.0, 1e-10)
    pc_uniform = primary_censored(delay_dist, primary_uniform)

    @test isfinite(pdf(pc_uniform, 1.0))
    @test isfinite(cdf(pc_uniform, 1.0))
end

@testitem "ExponentiallyTilted mathematical consistency checks" begin
    using Distributions

    d = ExponentiallyTilted(0.0, 1.0, 1.5)

    # Test that CDF is the integral of PDF
    # Using numerical integration to verify
    function numerical_cdf(dist, x, n_points = 1000)
        if x <= minimum(dist)
            return 0.0
        elseif x >= maximum(dist)
            return 1.0
        end

        x_vals = range(minimum(dist), x, length = n_points)
        dx = (x - minimum(dist)) / (n_points - 1)
        pdf_vals = [pdf(dist, xi) for xi in x_vals]

        # Trapezoidal rule
        return dx * (0.5 * (pdf_vals[1] + pdf_vals[end]) + sum(pdf_vals[2:(end - 1)]))
    end

    test_points = [0.1, 0.3, 0.5, 0.7, 0.9]
    for x in test_points
        numerical_val = numerical_cdf(d, x)
        analytical_val = cdf(d, x)
        @test abs(numerical_val - analytical_val) < 1e-4
    end

    # Test derivative relationship: d/dx CDF(x) ≈ PDF(x)
    eps = 1e-6
    for x in [0.2, 0.5, 0.8]
        numerical_derivative = (cdf(d, x + eps) - cdf(d, x - eps)) / (2 * eps)
        analytical_pdf = pdf(d, x)
        @test abs(numerical_derivative - analytical_pdf) < 1e-4
    end
end
