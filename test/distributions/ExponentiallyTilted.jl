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
    # infinite min
    @test_throws ArgumentError ExponentiallyTilted(Inf, 1.0, 1.0)
    # infinite max
    @test_throws ArgumentError ExponentiallyTilted(0.0, Inf, 1.0)
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
    using Integrals

    # Test numerical integration of PDF equals 1 using Integrals.jl
    function integrate_pdf(d)
        prob = IntegralProblem(
            (x, p) -> pdf(d, x),
            minimum(d),
            maximum(d)
        )
        sol = solve(prob, QuadGKJL(); reltol = 1e-8)
        return sol.u
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
        # Should integrate to 1 with high precision
        @test abs(integral - 1.0) < 1e-10
    end
end

@testitem "ExponentiallyTilted numerical stability with extreme params" begin
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

@testitem "ExponentiallyTilted mean calculation" begin
    using Distributions

    # Test uniform case (r ≈ 0)
    d_uniform = ExponentiallyTilted(0.0, 1.0, 1e-12)
    @test abs(mean(d_uniform) - 0.5) < 1e-10

    # Test symmetric case
    d_symmetric = ExponentiallyTilted(-1.0, 1.0, 1e-12)
    @test abs(mean(d_symmetric) - 0.0) < 1e-10

    # Test with positive r (tilted towards max)
    d_positive = ExponentiallyTilted(0.0, 1.0, 1.0)
    @test mean(d_positive) > 0.5  # Should be greater than uniform mean

    # Test with negative r (tilted towards min)
    d_negative = ExponentiallyTilted(0.0, 1.0, -1.0)
    @test mean(d_negative) < 0.5  # Should be less than uniform mean

    # Test consistency with integration for a specific case
    d_test = ExponentiallyTilted(0.0, 2.0, 0.5)
    theoretical_mean = mean(d_test)

    # The mean should be finite and within the support bounds
    @test isfinite(theoretical_mean)
    @test 0.0 <= theoretical_mean <= 2.0

    # For positive r, mean should be greater than midpoint
    @test theoretical_mean > 1.0  # midpoint of [0, 2]
end

@testitem "ExponentiallyTilted std and variance calculation" begin
    using Distributions

    # Test uniform case (r ≈ 0)
    d_uniform = ExponentiallyTilted(0.0, 1.0, 1e-12)
    expected_var_uniform = (1.0)^2 / 12  # Uniform variance formula
    @test abs(var(d_uniform) - expected_var_uniform) < 1e-10
    @test abs(std(d_uniform) - sqrt(expected_var_uniform)) < 1e-10

    # Test symmetric case
    d_symmetric = ExponentiallyTilted(-1.0, 1.0, 1e-12)
    expected_var_symmetric = (2.0)^2 / 12  # Uniform(−1,1) variance
    @test abs(var(d_symmetric) - expected_var_symmetric) < 1e-10

    # Test with positive r (should have different variance than uniform)
    d_positive = ExponentiallyTilted(0.0, 1.0, 1.0)
    var_positive = var(d_positive)
    std_positive = std(d_positive)

    @test isfinite(var_positive) && var_positive > 0
    @test isfinite(std_positive) && std_positive > 0
    @test abs(std_positive - sqrt(var_positive)) < 1e-12  # Consistency check

    # Test with negative r
    d_negative = ExponentiallyTilted(0.0, 1.0, -1.0)
    var_negative = var(d_negative)

    @test isfinite(var_negative) && var_negative > 0
    @test abs(var_positive - var_negative) < 1e-12  # Same magnitude r gives same variance

    # Test different bounds
    d_wide = ExponentiallyTilted(0.0, 10.0, 0.5)
    @test var(d_wide) > var(d_positive)  # Wider interval should have larger variance
end

@testitem "ExponentiallyTilted median calculation" begin
    using Distributions

    # Test uniform case (r ≈ 0) - median should be at midpoint
    d_uniform = ExponentiallyTilted(0.0, 1.0, 1e-12)
    @test abs(median(d_uniform) - 0.5) < 1e-10

    # Test symmetric case
    d_symmetric = ExponentiallyTilted(-1.0, 1.0, 1e-12)
    @test abs(median(d_symmetric) - 0.0) < 1e-10

    # Test with positive r (median should be > midpoint)
    d_positive = ExponentiallyTilted(0.0, 1.0, 1.0)
    median_positive = median(d_positive)
    @test median_positive > 0.5  # Should be greater than uniform median
    @test 0.0 <= median_positive <= 1.0  # Within support

    # Test with negative r (median should be < midpoint)
    d_negative = ExponentiallyTilted(0.0, 1.0, -1.0)
    median_negative = median(d_negative)
    @test median_negative < 0.5  # Should be less than uniform median
    @test 0.0 <= median_negative <= 1.0  # Within support

    # Test consistency: median should be where CDF = 0.5
    for r in [-1.5, -0.5, 0.5, 1.5]
        d = ExponentiallyTilted(0.0, 2.0, r)
        med = median(d)
        @test abs(cdf(d, med) - 0.5) < 1e-10
    end

    # Test different bounds
    d_shifted = ExponentiallyTilted(5.0, 15.0, 0.8)
    median_shifted = median(d_shifted)
    @test 5.0 <= median_shifted <= 15.0
    @test median_shifted > 10.0  # Should be > midpoint for positive r
end

@testitem "ExponentiallyTilted mathematical consistency checks" begin
    using Distributions
    using Integrals

    d = ExponentiallyTilted(0.0, 1.0, 1.5)

    # Test that CDF is the integral of PDF using Integrals.jl
    function numerical_cdf(dist, x)
        if x <= minimum(dist)
            return 0.0
        elseif x >= maximum(dist)
            return 1.0
        end

        prob = IntegralProblem(
            (t, p) -> pdf(dist, t),
            minimum(dist),
            x
        )
        sol = solve(prob, QuadGKJL(); reltol = 1e-10)
        return sol.u
    end

    test_points = [0.1, 0.3, 0.5, 0.7, 0.9]
    for x in test_points
        numerical_val = numerical_cdf(d, x)
        analytical_val = cdf(d, x)
        @test abs(numerical_val - analytical_val) < 1e-8
    end

    # Test derivative relationship: d/dx CDF(x) ≈ PDF(x)
    eps = 1e-6
    for x in [0.2, 0.5, 0.8]
        numerical_derivative = (cdf(d, x + eps) - cdf(d, x - eps)) / (2 * eps)
        analytical_pdf = pdf(d, x)
        @test abs(numerical_derivative - analytical_pdf) < 1e-4
    end
end

@testitem "ExponentiallyTilted Monte Carlo validation" begin
    using Distributions
    using Random
    using Statistics
    using StatsBase
    using HypothesisTests

    # Set seed for reproducible tests
    Random.seed!(42)

    # Test parameters: (min, max, r)
    test_cases = [
        (0.0, 1.0, 0.0),    # Uniform case (r = 0)
        (0.0, 1.0, 1.0),    # Moderate positive tilting
        (0.0, 1.0, -0.5),   # Negative tilting
        (-1.0, 2.0, 0.8),   # Different bounds with positive tilting
        (0.0, 1.0, 2.5),    # Strong positive tilting
        (0.0, 1.0, -2.0)    # Strong negative tilting
    ]

    n_samples = 15000  # Large sample size for stability

    for (min_val, max_val, r_val) in test_cases
        d = ExponentiallyTilted(min_val, max_val, r_val)

        # Generate large sample
        samples = [rand(d) for _ in 1:n_samples]

        # Test 1: All samples within support
        @test all(min_val <= s <= max_val for s in samples)

        # Test 2: Kolmogorov-Smirnov test using HypothesisTests.jl
        ks_test = ExactOneSampleKSTest(samples, d)
        @test pvalue(ks_test) > 0.001  # Should not reject at α = 0.001

        # Test 3: Quantile consistency
        prob_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
        for p in prob_levels
            analytical_quantile = quantile(d, p)
            empirical_quantile = quantile(samples, p)

            # Allow larger tolerance for extreme quantiles
            tolerance = p in [0.1, 0.9] ? 0.05 : 0.03
            relative_error = abs(analytical_quantile - empirical_quantile) /
                             (max_val - min_val)
            @test relative_error < tolerance
        end

        # Test 4: Sample mean validation using analytical mean
        empirical_mean = mean(samples)
        expected_mean = mean(d)  # Use the implemented mean method

        # Test mean with appropriate tolerance
        mean_tolerance = 0.02 * (max_val - min_val)
        @test abs(empirical_mean - expected_mean) < mean_tolerance

        # Test 5: Sample standard deviation validation
        empirical_std = std(samples)
        expected_std = std(d)  # Use the implemented std method

        std_tolerance = 0.05 * expected_std  # 5% tolerance
        @test abs(empirical_std - expected_std) < std_tolerance

        # Test 6: Sample median validation
        empirical_median = median(samples)
        expected_median = median(d)  # Use the implemented median method

        median_tolerance = 0.03 * (max_val - min_val)
        @test abs(empirical_median - expected_median) < median_tolerance

        # Test 7: Variance consistency for uniform case
        if r_val == 0.0  # Uniform case - we know all moments
            empirical_var = var(samples)
            expected_var = var(d)  # Use implemented variance method
            @test abs(empirical_var - expected_var) < 0.05 * expected_var
        end
    end
end

@testitem "ExponentiallyTilted Monte Carlo edge cases" begin
    using Distributions
    using Random
    using Statistics

    Random.seed!(123)
    n_samples = 10000

    # Test edge case: very small positive r (should behave like uniform)
    d_small_r = ExponentiallyTilted(0.0, 1.0, 1e-12)
    samples_small_r = [rand(d_small_r) for _ in 1:n_samples]

    # Should behave like Uniform(0, 1) - use implemented methods for comparison
    @test abs(mean(samples_small_r) - mean(d_small_r)) < 0.02
    @test abs(std(samples_small_r) - std(d_small_r)) < 0.01
    @test abs(var(samples_small_r) - var(d_small_r)) < 0.01

    # Test edge case: large positive r (heavily tilted towards max)
    d_large_r = ExponentiallyTilted(0.0, 1.0, 5.0)
    samples_large_r = [rand(d_large_r) for _ in 1:n_samples]

    # Should be heavily skewed towards 1.0
    @test mean(samples_large_r) > 0.75
    @test quantile(samples_large_r, 0.9) > 0.9

    # Test edge case: large negative r (heavily tilted towards min)
    d_large_neg_r = ExponentiallyTilted(0.0, 1.0, -5.0)
    samples_large_neg_r = [rand(d_large_neg_r) for _ in 1:n_samples]

    # Should be heavily skewed towards 0.0
    @test mean(samples_large_neg_r) < 0.25
    @test quantile(samples_large_neg_r, 0.1) < 0.1

    # Test numerical stability with extreme parameters
    d_extreme = ExponentiallyTilted(-10.0, 10.0, 3.0)
    # Smaller sample for extreme case
    samples_extreme = [rand(d_extreme) for _ in 1:1000]

    @test all(-10.0 <= s <= 10.0 for s in samples_extreme)
    @test isfinite(mean(samples_extreme))
    @test isfinite(var(samples_extreme))
end
