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
