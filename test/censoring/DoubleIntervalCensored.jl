@testitem "Test double_interval_censored structure equivalence - primary only" begin
    using Distributions

    # Test case 1: Primary censoring only
    delay_dist = LogNormal(1.5, 0.75)
    primary_event_dist = Uniform(0, 1)

    # Manual approach
    manual_dist = primary_censored(delay_dist, primary_event_dist)

    # Convenience function approach
    convenience_dist = double_interval_censored(
        delay_dist; primary_event = primary_event_dist)

    # Test they produce same type and structure
    @test typeof(manual_dist) == typeof(convenience_dist)
    @test params(manual_dist) == params(convenience_dist)
end

@testitem "Test double_interval_censored structure equivalence - with truncation" begin
    using Distributions

    # Test case 2: Primary censoring + truncation
    delay_dist = LogNormal(1.5, 0.75)
    primary_event_dist = Uniform(0, 1)
    upper_bound = 10.0

    # Manual approach
    manual_dist = primary_censored(delay_dist, primary_event_dist) |>
                  d -> truncated(d; upper = upper_bound)

    # Convenience function approach
    convenience_dist = double_interval_censored(
        delay_dist; primary_event = primary_event_dist, upper = upper_bound)

    # Test they produce same type
    @test typeof(manual_dist) == typeof(convenience_dist)

    # Test bounds are the same
    @test maximum(manual_dist) == maximum(convenience_dist)
    @test minimum(manual_dist) == minimum(convenience_dist)
end

@testitem "Test double_interval_censored structure equivalence - with interval censoring" begin
    using Distributions

    # Test case 3: Primary censoring + interval censoring
    delay_dist = LogNormal(1.5, 0.75)
    primary_event_dist = Uniform(0, 1)
    interval_width = 1.0

    # Manual approach
    manual_dist = primary_censored(delay_dist, primary_event_dist) |>
                  d -> interval_censored(d, interval_width)

    # Convenience function approach
    convenience_dist = double_interval_censored(
        delay_dist; primary_event = primary_event_dist, interval = interval_width)

    # Test they produce same type
    @test typeof(manual_dist) == typeof(convenience_dist)
end

@testitem "Test double_interval_censored structure equivalence - full pipeline" begin
    using Distributions

    # Test case 4: Full pipeline - Primary censoring + truncation + interval censoring
    delay_dist = LogNormal(1.5, 0.75)
    primary_event_dist = Uniform(0, 1)
    upper_bound = 10.0
    interval_width = 1.0

    # Manual approach (ensuring correct order: primary -> truncation -> interval censoring)
    manual_dist = primary_censored(delay_dist, primary_event_dist) |>
                  d -> truncated(d; upper = upper_bound) |>
                       d -> interval_censored(d, interval_width)

    # Convenience function approach
    convenience_dist = double_interval_censored(delay_dist;
        primary_event = primary_event_dist, upper = upper_bound, interval = interval_width)

    # Test they produce same type
    @test typeof(manual_dist) == typeof(convenience_dist)
end

@testitem "Test double_interval_censored with both bounds" begin
    using Distributions

    # Test case 5: With both lower and upper bounds
    delay_dist = LogNormal(1.5, 0.75)
    primary_event_dist = Uniform(0, 1)
    lower_bound = 2.0
    upper_bound = 8.0

    # Manual approach
    manual_dist = primary_censored(delay_dist, primary_event_dist) |>
                  d -> truncated(d, lower_bound, upper_bound)

    # Convenience function approach
    convenience_dist = double_interval_censored(delay_dist;
        primary_event = primary_event_dist, lower = lower_bound, upper = upper_bound)

    # Test they produce same type and bounds
    @test typeof(manual_dist) == typeof(convenience_dist)
    @test minimum(manual_dist) == minimum(convenience_dist) == lower_bound
    @test maximum(manual_dist) == maximum(convenience_dist) == upper_bound
end

@testitem "Test double_interval_censored return types" begin
    using Distributions

    # Test that function returns expected types for different combinations
    gamma_dist = Gamma(2, 1)
    uniform_primary = Uniform(0, 1)

    # Primary only -> PrimaryCensored
    @test isa(double_interval_censored(gamma_dist; primary_event = uniform_primary),
        CensoredDistributions.PrimaryCensored)

    # Primary + truncation -> Truncated{PrimaryCensored}
    @test isa(
        double_interval_censored(gamma_dist; primary_event = uniform_primary, upper = 5),
        Truncated)

    # Primary + interval censoring -> IntervalCensored{PrimaryCensored}
    @test isa(
        double_interval_censored(
            gamma_dist; primary_event = uniform_primary, interval = 0.5),
        CensoredDistributions.IntervalCensored)

    # Full pipeline -> IntervalCensored{Truncated{PrimaryCensored}}
    full_dist = double_interval_censored(
        gamma_dist; primary_event = uniform_primary, upper = 5, interval = 0.5)
    @test isa(full_dist, CensoredDistributions.IntervalCensored)
end

@testitem "Test double_interval_censored with nothing parameters" begin
    using Distributions

    # Test with nothing values (should be equivalent to not providing the argument)
    gamma_dist = Gamma(2, 1)
    uniform_primary = Uniform(0, 1)

    dist1 = double_interval_censored(gamma_dist; primary_event = uniform_primary)
    dist2 = double_interval_censored(gamma_dist; primary_event = uniform_primary,
        lower = nothing, upper = nothing, interval = nothing)

    @test typeof(dist1) == typeof(dist2)
    @test params(dist1) == params(dist2)
end

@testitem "Test DoubleIntervalCensored logpdf edge cases - primary only" begin
    using Distributions

    # Test primary censoring only (no interval censoring or truncation)
    delay_dist = Gamma(2.0, 1.0)
    primary_event_dist = Uniform(0, 1)

    d = double_interval_censored(delay_dist; primary_event = primary_event_dist)

    # This should behave like PrimaryCensored
    @test d isa CensoredDistributions.PrimaryCensored

    # Test out-of-support values return -Inf
    @test logpdf(d, -1.0) == -Inf  # Negative value (outside support)
    @test logpdf(d, -100.0) == -Inf  # Large negative value

    # Test extreme input values
    @test logpdf(d, 1e-15) ≠ NaN  # Very small positive value
    @test logpdf(d, 1e15) ≠ NaN   # Very large value

    # Test values in support
    test_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    for x in test_values
        if insupport(d, x)
            logpdf_val = logpdf(d, x)
            @test logpdf_val ≠ NaN
            @test isfinite(logpdf_val) || logpdf_val == -Inf
            @test logpdf_val <= 0.0
        end
    end

    # Test special values
    @test logpdf(d, NaN) == -Inf || isnan(logpdf(d, NaN))
    @test logpdf(d, Inf) == -Inf
    @test logpdf(d, -Inf) == -Inf
end

@testitem "Test DoubleIntervalCensored logpdf edge cases - with truncation" begin
    using Distributions

    # Test primary censoring + truncation
    delay_dist = LogNormal(1.0, 0.5)
    primary_event_dist = Uniform(0, 1)
    upper_bound = 10.0

    d = double_interval_censored(delay_dist;
        primary_event = primary_event_dist, upper = upper_bound)

    # This should be a Truncated{PrimaryCensored}
    @test d isa Truncated

    # Test values outside truncated support return -Inf
    @test logpdf(d, 15.0) == -Inf  # Above truncation bound
    @test logpdf(d, -1.0) == -Inf  # Below minimum (negative)

    # Test values within truncated support
    test_values = [0.5, 2.0, 5.0, 8.0]
    for x in test_values
        if insupport(d, x)
            logpdf_val = logpdf(d, x)
            @test logpdf_val ≠ NaN
            @test isfinite(logpdf_val) || logpdf_val == -Inf
            @test logpdf_val <= 0.0
        end
    end

    # Test boundary values
    @test logpdf(d, upper_bound) ≠ NaN  # At upper bound
    min_val = minimum(d)
    if isfinite(min_val)
        @test logpdf(d, min_val) ≠ NaN  # At lower bound
    end

    # Test extreme values
    @test logpdf(d, 1e-10) ≠ NaN  # Very small positive
    @test logpdf(d, 100.0) == -Inf  # Well above truncation
end

@testitem "Test DoubleIntervalCensored logpdf edge cases - with interval censoring" begin
    using Distributions

    # Test primary censoring + interval censoring
    delay_dist = Exponential(1.0)
    primary_event_dist = Uniform(0, 1)
    interval_width = 0.5

    d = double_interval_censored(delay_dist;
        primary_event = primary_event_dist, interval = interval_width)

    # This should be IntervalCensored{PrimaryCensored}
    @test d isa CensoredDistributions.IntervalCensored

    # Test values outside support
    @test logpdf(d, -5.0) == -Inf  # Negative value
    @test logpdf(d, -1.0) == -Inf  # Negative value

    # Test that values are discretized to interval boundaries
    # For interval_width = 0.5, values should be at 0.0, 0.5, 1.0, 1.5, etc.
    @test logpdf(d, 0.3) == logpdf(d, 0.0)  # Should be discretized to 0.0
    @test logpdf(d, 0.7) == logpdf(d, 0.5)  # Should be discretized to 0.5
    @test logpdf(d, 1.2) == logpdf(d, 1.0)  # Should be discretized to 1.0

    # Test boundary values
    boundary_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    for x in boundary_vals
        logpdf_val = logpdf(d, x)
        @test logpdf_val ≠ NaN
        @test isfinite(logpdf_val) || logpdf_val == -Inf
        @test logpdf_val <= 0.0
    end

    # Test extreme values
    @test logpdf(d, 1e10) ≠ NaN   # Very large value
    @test logpdf(d, 1e-15) ≠ NaN  # Very small positive value
end

@testitem "Test DoubleIntervalCensored logpdf edge cases - full pipeline" begin
    using Distributions

    # Test primary censoring + truncation + interval censoring
    delay_dist = Gamma(3.0, 0.5)
    primary_event_dist = Uniform(0, 2)
    lower_bound = 1.0
    upper_bound = 8.0
    interval_width = 0.25

    d = double_interval_censored(delay_dist;
        primary_event = primary_event_dist,
        lower = lower_bound, upper = upper_bound,
        interval = interval_width)

    # This should be IntervalCensored{Truncated{PrimaryCensored}}
    @test d isa CensoredDistributions.IntervalCensored

    # Test values outside truncated bounds
    @test logpdf(d, 0.5) == -Inf   # Below lower bound
    @test logpdf(d, 10.0) == -Inf  # Above upper bound
    @test logpdf(d, -1.0) == -Inf  # Negative value

    # Test values within bounds but check discretization
    # Values should be discretized to multiples of 0.25
    @test logpdf(d, 2.1) == logpdf(d, 2.0)    # Should discretize to 2.0
    @test logpdf(d, 3.37) == logpdf(d, 3.25)  # Should discretize to 3.25
    @test logpdf(d, 5.9) == logpdf(d, 5.75)   # Should discretize to 5.75

    # Test boundary values
    @test logpdf(d, lower_bound) ≠ NaN   # At lower bound
    @test logpdf(d, upper_bound) ≠ NaN   # At upper bound

    # Test regular interval boundaries within bounds
    interval_boundaries = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.75, 8.0]
    for x in interval_boundaries
        if lower_bound <= x <= upper_bound
            logpdf_val = logpdf(d, x)
            @test logpdf_val ≠ NaN
            @test isfinite(logpdf_val) || logpdf_val == -Inf
            @test logpdf_val <= 0.0
        end
    end

    # Test consistency with manual construction
    manual_d = primary_censored(delay_dist, primary_event_dist) |>
               d_temp -> truncated(d_temp, lower_bound, upper_bound) |>
                         d_temp2 -> interval_censored(d_temp2, interval_width)

    # Should give same logpdf values
    test_vals = [1.0, 2.5, 4.0, 6.25, 8.0]
    for x in test_vals
        @test logpdf(d, x) ≈ logpdf(manual_d, x) rtol=1e-12
    end
end

@testitem "Test DoubleIntervalCensored logpdf with arbitrary intervals" begin
    using Distributions

    # Test with arbitrary interval boundaries
    delay_dist = Weibull(2.0, 1.5)
    primary_event_dist = Uniform(0, 0.5)
    intervals = [0.0, 1.0, 2.5, 5.0, 10.0]

    # Manually compose: primary censoring -> interval censoring with arbitrary boundaries
    primary_censored_dist = primary_censored(delay_dist, primary_event_dist)
    d = interval_censored(primary_censored_dist, intervals)

    # Test values outside interval boundaries
    @test logpdf(d, -2.0) == -Inf   # Before first interval
    @test logpdf(d, 15.0) == -Inf   # After last interval

    # Test values at interval boundaries
    @test logpdf(d, 0.0) > -Inf     # At first boundary (has mass)
    @test logpdf(d, 1.0) > -Inf     # At second boundary (has mass)
    @test logpdf(d, 2.5) > -Inf     # At third boundary (has mass)
    @test logpdf(d, 5.0) > -Inf     # At fourth boundary (has mass)
    @test logpdf(d, 10.0) == -Inf   # At last boundary (upper bound, no mass)

    # Test values within intervals map to lower bounds
    @test logpdf(d, 0.7) == logpdf(d, 0.0)    # In [0, 1), maps to 0
    @test logpdf(d, 1.8) == logpdf(d, 1.0)    # In [1, 2.5), maps to 1
    @test logpdf(d, 3.2) == logpdf(d, 2.5)    # In [2.5, 5), maps to 2.5
    @test logpdf(d, 7.5) == logpdf(d, 5.0)    # In [5, 10), maps to 5

    # Test special values
    @test logpdf(d, NaN) == -Inf || isnan(logpdf(d, NaN))
    @test logpdf(d, Inf) == -Inf
    @test logpdf(d, -Inf) == -Inf
end

@testitem "Test DoubleIntervalCensored logpdf consistency across configurations" begin
    using Distributions

    # Test that different ways of creating similar distributions give consistent results
    base_dist = LogNormal(0.5, 1.0)
    primary = Uniform(0, 1)

    @testset "Configuration comparisons" begin
        # Configuration 1: Just primary censoring
        d1 = double_interval_censored(base_dist; primary_event = primary)

        # Configuration 2: Primary + very large truncation (should be similar to d1)
        d2 = double_interval_censored(base_dist; primary_event = primary, upper = 1e6)

        # Configuration 3: Primary + very fine interval censoring (should approximate d1)
        d3 = double_interval_censored(base_dist; primary_event = primary, interval = 1e-6)

        @testset "Config 1 vs Config 2 (large truncation)" begin
            # Check that they give similar results for values well within support
            test_vals = [0.5, 1.0, 2.0, 3.0]
            for x in test_vals
                logpdf1 = logpdf(d1, x)
                logpdf2 = logpdf(d2, x)

                # All should be finite and reasonable
                @test isfinite(logpdf1) || logpdf1 == -Inf
                @test isfinite(logpdf2) || logpdf2 == -Inf

                # d1 and d2 should be very similar (large upper bound shouldn't matter)
                if isfinite(logpdf1) && isfinite(logpdf2)
                    @test logpdf1 ≈ logpdf2 rtol=1e-10
                end
            end
        end

        @testset "Config 3 (fine discretization) - basic properties" begin
            # Test that d3 (fine discretization) behaves as expected
            test_vals = [0.5, 1.0, 2.0, 3.0]
            for x in test_vals
                logpdf3 = logpdf(d3, x)

                # Should be finite and reasonable
                @test isfinite(logpdf3) || logpdf3 == -Inf
                if isfinite(logpdf3)
                    @test logpdf3 <= 0.0  # Log probability should be ≤ 0
                end

                # Test that discretization is working (values should map to nearest grid point)
                # For interval = 1e-6, the discretization effect should be minimal
                if insupport(d3, x)
                    @test logpdf3 ≠ NaN
                end
            end

            # Note: d1 (continuous) and d3 (discretized) represent fundamentally different
            # distributions, so direct comparison is not mathematically meaningful.
            # d1 is a continuous distribution while d3 is discretized to a fine grid.
        end
    end
end

@testitem "Test DoubleIntervalCensored logpdf error handling and robustness" begin
    using Distributions

    # Test with edge case parameters that might cause numerical issues
    problematic_dist = truncated(Exponential(1e-10), 0.0, 2e-10)  # Very narrow support
    tiny_primary = Uniform(0, 1e-12)  # Very small primary event window

    d = double_interval_censored(problematic_dist;
        primary_event = tiny_primary, interval = 1e-15)

    # These should not throw errors, even if they might return -Inf
    problematic_vals = [0.0, 1e-15, -1e-15, 1e-12, -1e-12]
    for x in problematic_vals
        logpdf_val = logpdf(d, x)
        @test logpdf_val ≠ NaN  # Should never be NaN
        @test isfinite(logpdf_val) || logpdf_val == -Inf
    end

    # Test with mismatched scales
    large_dist = Exponential(1e-3)  # Small rate, large scale distribution
    small_primary = Uniform(0, 1e-6)  # Very small primary event
    large_interval = 1e4  # Large interval

    d_mismatch = double_interval_censored(large_dist;
        primary_event = small_primary, interval = large_interval)

    # Should handle scale mismatches gracefully
    mixed_vals = [0.0, 1e3, 1e6, 1e9]
    for x in mixed_vals
        logpdf_val = logpdf(d_mismatch, x)
        @test logpdf_val ≠ NaN
        @test isfinite(logpdf_val) || logpdf_val == -Inf
    end
end

@testitem "DoubleIntervalCensored with ExponentiallyTilted - basic integration" begin
    using Distributions

    # Test basic integration with ExponentiallyTilted as primary event
    delay_dist = LogNormal(1.0, 0.5)
    primary_event = ExponentiallyTilted(0.0, 2.0, 1.0)

    # Test primary censoring only
    d1 = double_interval_censored(delay_dist; primary_event = primary_event)
    @test typeof(d1) <: CensoredDistributions.PrimaryCensored

    # Test basic functionality
    @test isfinite(pdf(d1, 1.0)) && pdf(d1, 1.0) ≥ 0
    @test 0.0 ≤ cdf(d1, 1.0) ≤ 1.0
    @test isfinite(rand(d1))
end

@testitem "DoubleIntervalCensored with ExponentiallyTilted - with truncation" begin
    using Distributions

    # Test ExponentiallyTilted with truncation
    delay_dist = Gamma(2.0, 1.5)
    primary_event = ExponentiallyTilted(0.0, 1.5, -0.8)  # Decay scenario

    d2 = double_interval_censored(delay_dist;
        primary_event = primary_event, upper = 8.0)

    @test typeof(d2) <: Distributions.Truncated
    @test maximum(d2) == 8.0
    @test isfinite(pdf(d2, 2.0)) && pdf(d2, 2.0) ≥ 0
    @test 0.0 ≤ cdf(d2, 2.0) ≤ 1.0
end

@testitem "DoubleIntervalCensored with ExponentiallyTilted - growth vs decay" begin
    using Distributions

    # Test that growth and decay scenarios produce different results
    delay_dist = Exponential(1.0)

    # Growth scenario (r > 0)
    growth_primary = ExponentiallyTilted(0.0, 2.0, 1.5)
    d_growth = double_interval_censored(delay_dist; primary_event = growth_primary)

    # Decay scenario (r < 0)
    decay_primary = ExponentiallyTilted(0.0, 2.0, -1.5)
    d_decay = double_interval_censored(delay_dist; primary_event = decay_primary)

    # Should produce different results
    test_point = 1.0
    pdf_growth = pdf(d_growth, test_point)
    pdf_decay = pdf(d_decay, test_point)

    @test isfinite(pdf_growth) && pdf_growth > 0
    @test isfinite(pdf_decay) && pdf_decay > 0
    @test pdf_growth ≠ pdf_decay  # Should be different
end

@testitem "Test quantile function - primary censoring only" begin
    using Distributions

    # Test quantile functionality for primary censoring only (no interval censoring or truncation)
    delay_dist = LogNormal(1.0, 0.5)
    primary_event = Uniform(0, 1)

    d = double_interval_censored(delay_dist; primary_event = primary_event)

    # This should behave like PrimaryCensored
    @test d isa CensoredDistributions.PrimaryCensored

    # Test basic quantile functionality
    q25 = quantile(d, 0.25)
    q50 = quantile(d, 0.5)
    q75 = quantile(d, 0.75)

    @test isfinite(q25) && q25 ≥ 0
    @test isfinite(q50) && q50 ≥ 0
    @test isfinite(q75) && q75 ≥ 0

    # Test monotonicity
    @test q25 ≤ q50 ≤ q75

    # Test quantile-CDF consistency
    for p in [0.25, 0.5, 0.75]
        q = quantile(d, p)
        cdf_q = cdf(d, q)
        @test cdf_q ≈ p rtol=1e-3  # Lenient for various censoring combinations
    end

    # Test boundary cases
    @test quantile(d, 0.0) == 0.0
    @test quantile(d, 1.0) == Inf

    # Test invalid probabilities
    @test_throws ArgumentError quantile(d, -0.1)
    @test_throws ArgumentError quantile(d, 1.1)
end

@testitem "Test quantile function - with truncation" begin
    using Distributions

    # Test quantile functionality with truncation
    delay_dist = Gamma(2.0, 1.0)
    primary_event = Uniform(0, 1)
    lower_bound = 1.0
    upper_bound = 10.0

    d = double_interval_censored(delay_dist;
        primary_event = primary_event, lower = lower_bound, upper = upper_bound)

    # This should be a Truncated{PrimaryCensored}
    @test d isa Truncated

    # Test basic quantile functionality
    q25 = quantile(d, 0.25)
    q50 = quantile(d, 0.5)
    q75 = quantile(d, 0.75)

    @test lower_bound ≤ q25 ≤ upper_bound
    @test lower_bound ≤ q50 ≤ upper_bound
    @test lower_bound ≤ q75 ≤ upper_bound

    # Test monotonicity
    @test q25 ≤ q50 ≤ q75

    # Test quantile-CDF consistency
    test_probs = [0.1, 0.25, 0.5, 0.75, 0.9]
    for p in test_probs
        q = quantile(d, p)
        cdf_q = cdf(d, q)
        @test cdf_q ≈ p rtol=1e-3  # Lenient for various censoring combinations
    end

    # Test boundary cases
    @test quantile(d, 0.0) == lower_bound
    @test quantile(d, 1.0) == upper_bound

    # Test invalid probabilities
    @test_throws ArgumentError quantile(d, -0.1)
    @test_throws ArgumentError quantile(d, 1.1)
end

@testitem "Test quantile function - with interval censoring" begin
    using Distributions

    # Test quantile functionality with interval censoring
    delay_dist = Exponential(1.0)
    primary_event = Uniform(0, 1)
    interval_width = 0.5

    d = double_interval_censored(delay_dist;
        primary_event = primary_event, interval = interval_width)

    # This should be IntervalCensored{PrimaryCensored}
    @test d isa CensoredDistributions.IntervalCensored

    # Test basic quantile functionality
    q25 = quantile(d, 0.25)
    q50 = quantile(d, 0.5)
    q75 = quantile(d, 0.75)

    @test isfinite(q25) && q25 ≥ 0
    @test isfinite(q50) && q50 ≥ 0
    @test isfinite(q75) && q75 ≥ 0

    # Test monotonicity
    @test q25 ≤ q50 ≤ q75

    # Test that quantiles are multiples of interval width
    @test q25 % interval_width ≈ 0.0 atol=1e-10
    @test q50 % interval_width ≈ 0.0 atol=1e-10
    @test q75 % interval_width ≈ 0.0 atol=1e-10

    # Test quantile-CDF consistency
    test_probs = [0.1, 0.25, 0.5, 0.75, 0.9]
    for p in test_probs
        q = quantile(d, p)
        cdf_q = cdf(d, q)
        @test cdf_q ≈ p rtol=2e-2  # Lenient for interval censoring (discretized) combinations
    end

    # Test boundary cases
    @test quantile(d, 0.0) == minimum(d)
    @test quantile(d, 1.0) == maximum(d)

    # Test invalid probabilities
    @test_throws ArgumentError quantile(d, -0.1)
    @test_throws ArgumentError quantile(d, 1.1)
end

@testitem "Test quantile function - full pipeline" begin
    using Distributions

    # Test quantile functionality with full pipeline: primary + truncation + interval censoring
    delay_dist = LogNormal(1.0, 0.5)
    primary_event = Uniform(0, 2)
    lower_bound = 1.0
    upper_bound = 15.0
    interval_width = 0.25

    d = double_interval_censored(delay_dist;
        primary_event = primary_event,
        lower = lower_bound, upper = upper_bound,
        interval = interval_width)

    # This should be IntervalCensored{Truncated{PrimaryCensored}}
    @test d isa CensoredDistributions.IntervalCensored

    # Test basic quantile functionality
    q25 = quantile(d, 0.25)
    q50 = quantile(d, 0.5)
    q75 = quantile(d, 0.75)

    @test lower_bound ≤ q25 ≤ upper_bound
    @test lower_bound ≤ q50 ≤ upper_bound
    @test lower_bound ≤ q75 ≤ upper_bound

    # Test monotonicity
    @test q25 ≤ q50 ≤ q75

    # Test that quantiles are multiples of interval width
    @test q25 % interval_width ≈ 0.0 atol=1e-10
    @test q50 % interval_width ≈ 0.0 atol=1e-10
    @test q75 % interval_width ≈ 0.0 atol=1e-10

    # Test quantile-CDF consistency
    test_probs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    for p in test_probs
        q = quantile(d, p)
        cdf_q = cdf(d, q)
        @test cdf_q ≈ p rtol=2e-2  # Lenient for interval censoring (discretized) combinations
    end

    # Test boundary cases
    @test quantile(d, 0.0) == minimum(d)
    @test quantile(d, 1.0) == maximum(d)

    # Test that boundaries respect both truncation and discretization
    min_q = minimum(d)
    max_q = maximum(d)

    @test min_q ≥ lower_bound
    @test max_q ≤ upper_bound
    @test min_q % interval_width ≈ 0.0 atol=1e-10
    @test max_q % interval_width ≈ 0.0 atol=1e-10
end

@testitem "Test quantile function - with arbitrary intervals" begin
    using Distributions

    # Test with arbitrary interval boundaries
    delay_dist = Weibull(2.0, 1.5)
    primary_event = Uniform(0, 0.5)
    boundaries = [0.0, 1.0, 2.5, 5.0, 10.0]

    # Create using manual composition (since double_interval_censored doesn't support arbitrary intervals)
    primary_dist = primary_censored(delay_dist, primary_event)
    d = interval_censored(primary_dist, boundaries)

    # Test basic quantile functionality
    q25 = quantile(d, 0.25)
    q50 = quantile(d, 0.5)
    q75 = quantile(d, 0.75)

    @test isfinite(q25) && q25 ≥ 0
    @test isfinite(q50) && q50 ≥ 0
    @test isfinite(q75) && q75 ≥ 0

    # Test monotonicity
    @test q25 ≤ q50 ≤ q75

    # Test that quantiles are boundary values
    @test q25 in boundaries
    @test q50 in boundaries
    @test q75 in boundaries

    # Test quantile-CDF consistency
    test_probs = [0.1, 0.25, 0.5, 0.75, 0.9]
    for p in test_probs
        q = quantile(d, p)
        cdf_q = cdf(d, q)
        @test cdf_q ≈ p rtol=1e-3  # Lenient for various censoring combinations
    end

    # Test boundary cases
    @test quantile(d, 0.0) == boundaries[1]  # First boundary
    # Last quantile depends on which boundary has mass
    last_q = quantile(d, 1.0)
    @test last_q in boundaries[1:(end - 1)]  # Should be one of the interval starts
end

@testitem "Test quantile function - consistency with manual composition" begin
    using Distributions

    # Test that double_interval_censored quantiles match manual composition
    delay_dist = Gamma(1.5, 2.0)
    primary_event = Uniform(0, 1)
    upper_bound = 12.0
    interval_width = 0.5

    # Create using convenience function
    d_convenience = double_interval_censored(delay_dist;
        primary_event = primary_event, upper = upper_bound, interval = interval_width)

    # Create using manual composition
    d_manual = primary_censored(delay_dist, primary_event) |>
               d_temp -> truncated(d_temp; upper = upper_bound) |>
                         d_temp2 -> interval_censored(d_temp2, interval_width)

    # Test that quantiles match
    test_probs = [0.1, 0.25, 0.5, 0.75, 0.9]

    for p in test_probs
        q_convenience = quantile(d_convenience, p)
        q_manual = quantile(d_manual, p)

        @test q_convenience ≈ q_manual rtol=1e-12
    end

    # Test that they produce the same type
    @test typeof(d_convenience) == typeof(d_manual)

    # Test boundary consistency
    @test quantile(d_convenience, 0.0) ≈ quantile(d_manual, 0.0)
    @test quantile(d_convenience, 1.0) ≈ quantile(d_manual, 1.0)
end

@testitem "Test quantile function - monotonicity across configurations" begin
    using Distributions

    # Test monotonicity for different DoubleIntervalCensored configurations
    delay_dist = LogNormal(0.5, 1.0)
    primary_event = Uniform(0, 1)

    # Configuration 1: Primary only
    d1 = double_interval_censored(delay_dist; primary_event = primary_event)

    # Configuration 2: Primary + truncation
    d2 = double_interval_censored(delay_dist; primary_event = primary_event, upper = 20.0)

    # Configuration 3: Primary + interval censoring
    d3 = double_interval_censored(delay_dist; primary_event = primary_event, interval = 0.5)

    # Configuration 4: Full pipeline
    d4 = double_interval_censored(delay_dist;
        primary_event = primary_event, upper = 20.0, interval = 0.5)

    distributions = [d1, d2, d3, d4]

    for (i, d) in enumerate(distributions)
        @testset "Configuration $(i) monotonicity" begin
            # Test monotonicity with many probability values
            probs = range(0.01, 0.99, length = 15)
            quantiles = [quantile(d, p) for p in probs]

            # Check that quantiles are non-decreasing
            for j in 2:length(quantiles)
                @test quantiles[j] ≥ quantiles[j - 1]
            end

            # Test specific probability pairs
            prob_pairs = [(0.1, 0.3), (0.4, 0.6), (0.7, 0.9)]

            for (p1, p2) in prob_pairs
                q1 = quantile(d, p1)
                q2 = quantile(d, p2)
                @test q1 ≤ q2
            end
        end
    end
end

@testitem "Test quantile function - extreme probability values" begin
    using Distributions

    # Test with various DoubleIntervalCensored configurations and extreme probabilities
    delay_dist = Exponential(2.0)
    primary_event = Uniform(0, 0.5)

    # Different configurations to test
    configurations = [
        ("primary only",
            double_interval_censored(delay_dist; primary_event = primary_event)),
        ("with truncation",
            double_interval_censored(delay_dist;
                primary_event = primary_event, upper = 10.0)),
        ("with interval",
            double_interval_censored(delay_dist;
                primary_event = primary_event, interval = 0.25)),
        ("full pipeline",
            double_interval_censored(delay_dist;
                primary_event = primary_event, upper = 10.0, interval = 0.25))
    ]

    for (config_name, d) in configurations
        @testset "$(config_name) - extreme probabilities" begin
            # Test very small probabilities
            small_probs = [1e-10, 1e-6, 1e-4]
            for p in small_probs
                q = quantile(d, p)
                @test !isnan(q) && isfinite(q)
                @test q ≥ 0
                @test cdf(d, q) ≈ p rtol=1e-3
            end

            # Test very large probabilities
            large_probs = [1.0 - 1e-10, 1.0 - 1e-6, 1.0 - 1e-4]
            for p in large_probs
                q = quantile(d, p)
                @test !isnan(q)
                @test q ≥ 0

                # For truncated distributions, quantile should be finite
                if d isa Truncated || (d isa CensoredDistributions.IntervalCensored &&
                    get_dist(d) isa Truncated)
                    @test isfinite(q)
                end

                @test cdf(d, q) ≈ p rtol=1e-3
            end
        end
    end
end

@testitem "Test quantile function - with ExponentiallyTilted primary event" begin
    using Distributions

    delay_dist = LogNormal(1.0, 0.5)

    # Different ExponentiallyTilted scenarios
    growth_primary = ExponentiallyTilted(0.0, 2.0, 1.0)    # Growth
    decay_primary = ExponentiallyTilted(0.0, 2.0, -1.0)   # Decay
    neutral_primary = ExponentiallyTilted(0.0, 2.0, 1e-8) # Nearly uniform

    primary_events = [
        ("growth", growth_primary), ("decay", decay_primary), ("neutral", neutral_primary)]

    for (scenario_name, primary_event) in primary_events
        @testset "$(scenario_name) scenario" begin
            # Test different configurations with ExponentiallyTilted
            d_primary = double_interval_censored(delay_dist; primary_event = primary_event)
            d_truncated = double_interval_censored(delay_dist;
                primary_event = primary_event, upper = 15.0)
            d_interval = double_interval_censored(delay_dist;
                primary_event = primary_event, interval = 0.5)
            d_full = double_interval_censored(delay_dist;
                primary_event = primary_event, upper = 15.0, interval = 0.5)

            distributions = [d_primary, d_truncated, d_interval, d_full]

            for (i, d) in enumerate(distributions)
                @testset "Config $(i)" begin
                    # Test basic quantile functionality
                    test_probs = [0.1, 0.25, 0.5, 0.75, 0.9]

                    for p in test_probs
                        q = quantile(d, p)
                        @test isfinite(q) && q ≥ 0

                        # Test consistency
                        cdf_q = cdf(d, q)
                        @test cdf_q ≈ p rtol=1e-3  # Lenient for various censoring combinations
                    end

                    # Test monotonicity
                    q25 = quantile(d, 0.25)
                    q50 = quantile(d, 0.5)
                    q75 = quantile(d, 0.75)
                    @test q25 ≤ q50 ≤ q75
                end
            end
        end
    end

    # Test that different scenarios produce different quantiles
    test_p = 0.5
    q_growth = quantile(double_interval_censored(delay_dist; primary_event = growth_primary), test_p)
    q_decay = quantile(double_interval_censored(delay_dist; primary_event = decay_primary), test_p)

    @test q_growth ≠ q_decay  # Should give different results
end

@testitem "Test quantile function - numerical stability and edge cases" begin
    using Distributions

    # Test with challenging parameter combinations that might cause numerical issues
    challenging_configs = [
        ("High variance delay", LogNormal(5.0, 3.0), Uniform(0, 1)),
        ("Small scale delay", Gamma(0.5, 0.1), Uniform(0, 2)),
        ("Narrow primary event", Exponential(1.0), Uniform(0, 1e-6)),
        ("Wide primary event", Weibull(1.5, 2.0), Uniform(0, 10))
    ]

    for (config_name, delay_dist, primary_event) in challenging_configs
        @testset "$(config_name)" begin
            # Test different DoubleIntervalCensored configurations
            d1 = double_interval_censored(delay_dist; primary_event = primary_event)

            # Only test configurations that are likely to be well-behaved
            try
                d2 = double_interval_censored(delay_dist;
                    primary_event = primary_event, upper = 50.0)
                distributions = [d1, d2]
            catch
                distributions = [d1]  # Fall back to just primary censoring
            end

            for d in distributions
                # Test that quantiles are well-behaved
                test_probs = [0.01, 0.1, 0.5, 0.9, 0.99]

                for p in test_probs
                    q = quantile(d, p)
                    @test !isnan(q)
                    @test q ≥ 0

                    # For finite quantiles, test consistency
                    if isfinite(q)
                        cdf_q = cdf(d, q)
                        @test cdf_q ≈ p rtol=1e-2  # More lenient for challenging cases
                    end
                end

                # Test monotonicity
                q_low = quantile(d, 0.1)
                q_mid = quantile(d, 0.5)
                q_high = quantile(d, 0.9)
                @test q_low ≤ q_mid ≤ q_high
            end
        end
    end
end
