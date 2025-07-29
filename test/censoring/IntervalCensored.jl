@testitem "Test IntervalCensored constructor" begin
    using Distributions

    # Test regular interval construction
    d = Normal(0, 1)
    ic_regular = interval_censored(d, 1.0)
    @test typeof(ic_regular) <: CensoredDistributions.IntervalCensored
    @test ic_regular.dist === d
    @test ic_regular.boundaries == 1.0
    @test CensoredDistributions.is_regular_intervals(ic_regular) == true

    # Test arbitrary interval construction
    intervals = [0.0, 2.0, 5.0, 10.0]
    ic_arbitrary = interval_censored(d, intervals)
    @test typeof(ic_arbitrary) <: CensoredDistributions.IntervalCensored
    @test ic_arbitrary.dist === d
    @test ic_arbitrary.boundaries == intervals
    @test CensoredDistributions.is_regular_intervals(ic_arbitrary) == false

    # Test constructor error handling
    @test_throws ArgumentError interval_censored(d, 0.0)  # Zero interval
    @test_throws ArgumentError interval_censored(d, -1.0)  # Negative interval
    @test_throws ArgumentError interval_censored(d, [1.0])  # Too few boundaries
    @test_throws ArgumentError interval_censored(d, [2.0, 1.0])  # Unsorted boundaries
    @test_throws ArgumentError interval_censored(d, [1.0, 1.0, 2.0])  # Non-increasing boundaries
end

@testitem "Test IntervalCensored distribution interface - regular intervals" begin
    using Distributions
    using Random

    d = LogNormal(1.0, 0.5)
    interval = 0.5
    ic = interval_censored(d, interval)

    # Test basic properties
    @test minimum(ic) == CensoredDistributions.floor_to_interval(minimum(d), interval)
    @test maximum(ic) == CensoredDistributions.floor_to_interval(maximum(d), interval)

    # Test params
    p = params(ic)
    @test p == (params(d)..., interval)

    # Test insupport - checks underlying distribution support
    @test insupport(ic, 0.0) == true  # In LogNormal support
    @test insupport(ic, 0.5) == true
    @test insupport(ic, 1.0) == true
    @test insupport(ic, 0.3) == true  # In LogNormal support
    @test insupport(ic, -1.0) == false  # Negative, not in LogNormal support
end

@testitem "Test IntervalCensored distribution interface - arbitrary intervals" begin
    using Distributions

    d = Normal(5, 2)
    intervals = [0.0, 2.0, 5.0, 10.0]
    ic = interval_censored(d, intervals)

    # Test basic properties
    @test minimum(ic) == 0.0  # First interval boundary
    @test maximum(ic) == 5.0  # Last interval boundary that could contain values

    # Test params
    p = params(ic)
    @test p == (params(d)..., intervals)

    # Test insupport - checks underlying distribution support
    @test insupport(ic, 0.0) == true  # In Normal support
    @test insupport(ic, 2.0) == true
    @test insupport(ic, 5.0) == true
    @test insupport(ic, 10.0) == true
    @test insupport(ic, 1.0) == true  # In Normal support
    @test insupport(ic, 3.0) == true  # In Normal support
end

@testitem "Test IntervalCensored pdf/cdf - regular intervals" begin
    using Distributions

    d = Normal(5, 2)
    interval = 1.0
    ic = interval_censored(d, interval)

    # Test pdf
    x = 5.0
    expected_pdf = cdf(d, x + interval) - cdf(d, x)
    @test pdf(ic, x) ≈ expected_pdf

    # Test logpdf
    @test logpdf(ic, x) ≈ log(expected_pdf)

    # Test cdf - should floor to interval
    @test cdf(ic, 5.3) ≈ cdf(d, 5.0)  # 5.3 floors to 5.0
    @test cdf(ic, 5.9) ≈ cdf(d, 5.0)  # 5.9 floors to 5.0
    @test cdf(ic, 6.0) ≈ cdf(d, 6.0)  # 6.0 floors to 6.0

    # Test ccdf and log versions
    @test ccdf(ic, x) ≈ 1 - cdf(ic, x)
    @test logcdf(ic, x) ≈ log(cdf(ic, x))
    @test logccdf(ic, x) ≈ log(ccdf(ic, x))
end

@testitem "Test IntervalCensored pdf/cdf - arbitrary intervals" begin
    using Distributions

    d = Normal(5, 2)
    intervals = [0.0, 2.0, 5.0, 10.0]
    ic = interval_censored(d, intervals)

    # Test pdf for values in intervals
    @test pdf(ic, 1.0) ≈ cdf(d, 2.0) - cdf(d, 0.0)  # In [0, 2]
    @test pdf(ic, 3.0) ≈ cdf(d, 5.0) - cdf(d, 2.0)  # In [2, 5]
    @test pdf(ic, 7.0) ≈ cdf(d, 10.0) - cdf(d, 5.0)  # In [5, 10]

    # Test pdf outside intervals
    @test pdf(ic, -1.0) ≈ 0.0  # Before first interval
    @test pdf(ic, 11.0) ≈ 0.0  # After last interval

    # Test cdf - should use lower bound of containing interval
    @test cdf(ic, 1.0) ≈ cdf(d, 0.0)  # In [0, 2], use 0
    @test cdf(ic, 3.0) ≈ cdf(d, 2.0)  # In [2, 5], use 2
    @test cdf(ic, 7.0) ≈ cdf(d, 5.0)  # In [5, 10], use 5
    @test cdf(ic, -1.0) ≈ 0.0  # Before intervals
    @test cdf(ic, 11.0) ≈ cdf(d, 10.0)  # After intervals
end

@testitem "Test IntervalCensored single interval compatibility" begin
    using Distributions

    # Test single interval case - should work same as multiple intervals
    d = Normal(5, 2)
    ic = interval_censored(d, [2.0, 4.0])

    # pdf at boundary points should work
    @test pdf(ic, 2.0) > 0
    @test pdf(ic, 4.0) ≈ 0  # At upper boundary, no mass

    # Should behave consistently with multi-interval case
    ic_multi = interval_censored(d, [2.0, 4.0, 6.0])
    @test pdf(ic, 2.0) ≈ pdf(ic_multi, 2.0)
end

@testitem "Test IntervalCensored sampling - regular intervals" begin
    using Distributions
    using Random
    using Statistics

    rng = MersenneTwister(123)
    d = Normal(5, 2)
    interval = 0.5
    ic = interval_censored(d, interval)

    # Generate samples
    n_samples = 10000
    samples = rand(rng, ic, n_samples)

    # All samples should be multiples of the interval
    @test all(s % interval ≈ 0.0 for s in samples)

    # Samples should be discretized versions of the underlying distribution
    # Test by checking that sample mean is close to discretized mean
    continuous_samples = rand(MersenneTwister(123), d, n_samples)
    discretized_samples = [CensoredDistributions.floor_to_interval(s, interval)
                           for s in continuous_samples]
    @test mean(samples) ≈ mean(discretized_samples) rtol=0.1
end

@testitem "Test IntervalCensored sampling - arbitrary intervals" begin
    using Distributions
    using Random

    rng = MersenneTwister(456)
    d = Normal(5, 2)

    # Test single interval case - should return interval boundaries based on where samples fall
    ic_single = interval_censored(d, [2.0, 4.0])
    samples_single = rand(rng, ic_single, 100)
    @test all(s in [2.0, 4.0] for s in samples_single)  # Samples should be interval boundaries

    # Test multiple intervals case
    intervals = [0.0, 2.0, 5.0, 10.0]
    ic_multi = interval_censored(d, intervals)
    samples_multi = rand(rng, ic_multi, 1000)

    # When sampling from intervals, we get the lower bound of the interval containing the sample
    unique_samples = unique(samples_multi)
    @test all([s in intervals for s in unique_samples])

    # Roughly check distribution of samples across intervals
    # Most samples should be in middle interval [2.0, 5.0] for Normal(5, 2)
    count_middle = sum([s == 2.0 for s in samples_multi])
    @test count_middle > length(samples_multi) * 0.3  # At least 30% in middle interval
end

@testitem "Test IntervalCensored with negative boundaries - comprehensive" begin
    using Distributions
    using Random

    rng = MersenneTwister(789)
    d = Normal(0, 2)  # Centered at 0, samples can be negative or positive

    # Test with boundaries including negative values
    boundaries = [-5.0, -2.0, 1.0, 4.0]  # Creates intervals [-5,-2), [-2,1), [1,4)
    ic = interval_censored(d, boundaries)

    # Test sampling
    samples = rand(rng, ic, 1000)
    unique_samples = unique(samples)
    @test all([s in boundaries for s in unique_samples])
    @test length(unique_samples) >= 2  # Should see multiple boundaries
    @test any(s < 0 for s in samples)  # Should have some negative samples

    # Test pdf at each boundary
    @test pdf(ic, -5.0) > 0  # Should have mass in first interval [-5,-2)
    @test pdf(ic, -2.0) > 0  # Should have mass in second interval [-2,1)
    @test pdf(ic, 1.0) > 0   # Should have mass in third interval [1,4)
    @test pdf(ic, 4.0) ≈ 0   # At upper boundary, no mass (outside intervals)

    # Test cdf behavior
    @test cdf(ic, -6.0) ≈ 0.0  # Before first boundary
    @test cdf(ic, -5.0) ≈ cdf(d, -5.0)  # At first boundary
    @test cdf(ic, -2.0) ≈ cdf(d, -2.0)  # At second boundary
    @test cdf(ic, 1.0) ≈ cdf(d, 1.0)    # At third boundary
    @test cdf(ic, 5.0) ≈ cdf(d, 4.0)    # Beyond last boundary

    # Test distribution interface
    @test minimum(ic) == -5.0  # Should be first boundary
    @test maximum(ic) == 1.0   # Should be last interval start (where sample can occur)
    @test insupport(ic, -5.0) == true
    @test insupport(ic, -2.0) == true
    @test insupport(ic, 1.0) == true
    @test insupport(ic, 4.0) == true  # Boundary is in support
    @test insupport(ic, 0.0) == true  # In Normal support

    # Test interval bounds computation
    lower, upper = CensoredDistributions.get_interval_bounds(ic, -3.0)  # In [-5,-2) interval
    @test lower == -5.0 && upper == -2.0

    lower, upper = CensoredDistributions.get_interval_bounds(ic, 0.0)   # In [-2,1) interval
    @test lower == -2.0 && upper == 1.0

    lower, upper = CensoredDistributions.get_interval_bounds(ic, 2.0)   # In [1,4) interval
    @test lower == 1.0 && upper == 4.0

    # Outside boundaries
    lower, upper = CensoredDistributions.get_interval_bounds(ic, -10.0)  # Before first
    @test isnan(lower) && isnan(upper)

    lower, upper = CensoredDistributions.get_interval_bounds(ic, 10.0)   # After last
    @test isnan(lower) && isnan(upper)
end

@testitem "Test IntervalCensored floor_to_interval helper" begin
    # Test the floor_to_interval function
    @test CensoredDistributions.floor_to_interval(5.7, 1.0) == 5.0
    @test CensoredDistributions.floor_to_interval(5.7, 0.5) == 5.5
    @test CensoredDistributions.floor_to_interval(5.7, 2.0) == 4.0
    @test CensoredDistributions.floor_to_interval(0.0, 1.0) == 0.0
    @test CensoredDistributions.floor_to_interval(-1.3, 0.5) == -1.5
end

@testitem "Test IntervalCensored find_interval_index helper" begin
    intervals = [0.0, 2.0, 5.0, 10.0]

    # Test values in intervals
    @test CensoredDistributions.find_interval_index(1.0, intervals) == 1  # In [0, 2]
    @test CensoredDistributions.find_interval_index(3.0, intervals) == 2  # In [2, 5]
    @test CensoredDistributions.find_interval_index(7.0, intervals) == 3  # In [5, 10]

    # Test boundary values
    @test CensoredDistributions.find_interval_index(0.0, intervals) == 1  # At 0
    @test CensoredDistributions.find_interval_index(2.0, intervals) == 2  # At 2
    @test CensoredDistributions.find_interval_index(5.0, intervals) == 3  # At 5

    # Test outside intervals
    @test CensoredDistributions.find_interval_index(-1.0, intervals) == 0  # Before
    @test CensoredDistributions.find_interval_index(10.0, intervals) == 4  # After (at last boundary)
    @test CensoredDistributions.find_interval_index(11.0, intervals) == 4  # After
end

@testitem "Test IntervalCensored logpdf extreme value handling and -Inf edge cases" begin
    using Distributions

    # Test with regular intervals
    d = Normal(5, 2)
    interval = 1.0
    ic = interval_censored(d, interval)

    # Test out-of-support values for underlying distribution
    # For Normal distribution, all real values are technically in support
    # but we can test extreme values
    @test logpdf(ic, -1e10) ≠ NaN  # Very large negative value
    @test logpdf(ic, 1e10) ≠ NaN   # Very large positive value
    @test isfinite(logpdf(ic, -1e10)) || logpdf(ic, -1e10) == -Inf
    @test isfinite(logpdf(ic, 1e10)) || logpdf(ic, 1e10) == -Inf

    # Test with bounded distribution
    bounded_dist = truncated(Exponential(1.0), 0.0, 10.0)
    ic_bounded = interval_censored(bounded_dist, 0.5)

    # Test values outside underlying distribution support
    @test logpdf(ic_bounded, -1.0) == -Inf  # Negative (outside Exponential support)
    @test logpdf(ic_bounded, -100.0) == -Inf  # Large negative
    @test logpdf(ic_bounded, 15.0) == -Inf   # Above truncation bound

    # Test with arbitrary intervals
    intervals = [0.0, 2.0, 5.0, 8.0]
    ic_arb = interval_censored(Normal(3, 1), intervals)

    # Test values outside interval boundaries
    @test logpdf(ic_arb, -5.0) == -Inf   # Before first interval
    @test logpdf(ic_arb, 12.0) == -Inf   # After last interval

    # Test edge cases at interval boundaries
    @test logpdf(ic_arb, 8.0) == -Inf   # At upper boundary (no mass)
    @test logpdf(ic_arb, 0.0) > -Inf    # At lower boundary (has mass)
    @test logpdf(ic_arb, 2.0) > -Inf    # At interval boundary (has mass)
    @test logpdf(ic_arb, 5.0) > -Inf    # At interval boundary (has mass)
end

@testitem "Test IntervalCensored logpdf numerical stability" begin
    using Distributions

    # Test with very small intervals that might cause numerical issues
    d = Normal(0, 1)
    tiny_interval = 1e-10
    ic_tiny = interval_censored(d, tiny_interval)

    # Test values that should have very small probability mass
    test_vals = [0.0, 1e-11, 5.0, -3.0]
    for x in test_vals
        logpdf_val = logpdf(ic_tiny, x)
        @test logpdf_val ≠ NaN
        @test isfinite(logpdf_val) || logpdf_val == -Inf
        @test logpdf_val <= 0.0  # Log probability should be ≤ 0
    end

    # Test with very wide intervals
    huge_interval = 1e10
    ic_huge = interval_censored(d, huge_interval)

    for x in test_vals
        logpdf_val = logpdf(ic_huge, x)
        @test logpdf_val ≠ NaN
        @test isfinite(logpdf_val) || logpdf_val == -Inf
        @test logpdf_val <= 0.0
    end

    # Test with extreme distribution parameters
    extreme_dist = Normal(1e6, 1e-6)  # Very concentrated distribution
    ic_extreme = interval_censored(extreme_dist, 1.0)

    # Test near the distribution centre
    centre = 1e6
    nearby_vals = [centre - 1.0, centre, centre + 1.0]
    for x in nearby_vals
        logpdf_val = logpdf(ic_extreme, x)
        @test logpdf_val ≠ NaN
        @test isfinite(logpdf_val) || logpdf_val == -Inf
    end
end

@testitem "Test IntervalCensored logpdf with degenerate cases" begin
    using Distributions

    # Test with very narrow arbitrary intervals
    d = Normal(5, 2)
    narrow_intervals = [4.99999, 5.00001]  # Very narrow interval around mean
    ic_narrow = interval_censored(d, narrow_intervals)

    # Most values should return -Inf except those in the narrow interval
    @test logpdf(ic_narrow, 4.0) == -Inf   # Outside interval
    @test logpdf(ic_narrow, 6.0) == -Inf   # Outside interval
    @test logpdf(ic_narrow, 4.99999) > -Inf  # At interval boundary

    # Test with very narrow intervals (nearly degenerate case)
    narrow_intervals = [2.0, 2.0001, 4.0, 4.0001]  # Very narrow intervals
    ic_narrow2 = interval_censored(d, narrow_intervals)

    # Should have very little mass in these narrow intervals
    @test logpdf(ic_narrow2, 2.0) > -Inf   # At narrow interval start
    @test logpdf(ic_narrow2, 4.0001) == -Inf  # At upper boundary (no mass there)
    @test logpdf(ic_narrow2, 1.0) == -Inf  # Outside all intervals

    # Test with minimum gap intervals
    min_gap_intervals = [0.0, 1.0, 1.0001, 2.0]  # Minimum allowed gap
    ic_min_gap = interval_censored(d, min_gap_intervals)

    # Should handle minimum gaps properly
    @test logpdf(ic_min_gap, 0.5) > -Inf   # In first interval
    @test logpdf(ic_min_gap, 1.5) > -Inf   # In second interval
    @test logpdf(ic_min_gap, 1.0) > -Inf   # At first interval boundary
end

@testitem "Test IntervalCensored logpdf consistency and edge cases" begin
    using Distributions

    # Test consistency between pdf and logpdf
    d = LogNormal(1, 0.5)
    ic = interval_censored(d, 0.5)

    test_values = [0.5, 1.0, 2.0, 3.5, 5.0]
    for x in test_values
        pdf_val = pdf(ic, x)
        logpdf_val = logpdf(ic, x)

        if pdf_val > 0
            @test logpdf_val ≈ log(pdf_val) rtol=1e-12
            @test pdf_val ≈ exp(logpdf_val) rtol=1e-12
        else
            @test logpdf_val == -Inf
            @test pdf_val == 0.0
        end
    end

    # Test that logpdf handles get_interval_bounds edge cases
    intervals = [-5.0, 0.0, 5.0]
    ic_mixed = interval_censored(Normal(0, 2), intervals)

    # Test values that return NaN from get_interval_bounds
    extreme_vals = [-100.0, 100.0]
    for x in extreme_vals
        lower, upper = CensoredDistributions.get_interval_bounds(ic_mixed, x)
        if isnan(lower) || isnan(upper)
            @test logpdf(ic_mixed, x) == -Inf
        end
    end

    # Test special floating point values
    @test logpdf(ic, NaN) == -Inf || isnan(logpdf(ic, NaN))
    @test logpdf(ic, Inf) == -Inf
    @test logpdf(ic, -Inf) == -Inf
end

@testitem "Test IntervalCensored logpdf with different underlying distributions" begin
    using Distributions

    # Test with various underlying distributions and interval types
    test_distributions = [
        (Normal(0, 1), 0.5),           # Regular interval
        (Gamma(2, 1), [0.0, 1.0, 3.0, 6.0]),  # Arbitrary intervals
        (Exponential(2), 1.0),         # Regular interval
        (Weibull(1.5, 2), [0.0, 0.5, 2.0]),   # Arbitrary intervals
        (truncated(Normal(0, 1), -2, 2), 0.25)  # Truncated underlying dist
    ]

    for (dist, intervals) in test_distributions
        interval_type = isa(intervals, AbstractVector) ? "arbitrary" : "regular"
        @testset "$(typeof(dist)) - $(interval_type) intervals" begin
            ic = interval_censored(dist, intervals)

            # Test a range of values
            if isa(intervals, AbstractVector)
                test_range = range(
                    minimum(intervals) - 1, maximum(intervals) + 1, length = 10)
            else
                test_range = range(-3, 10, length = 10)
            end

            for x in test_range
                logpdf_val = logpdf(ic, x)
                @test logpdf_val ≠ NaN
                @test isfinite(logpdf_val) || logpdf_val == -Inf
                @test logpdf_val <= 0.0

                # Test consistency with insupport
                if !insupport(dist, x)
                    # If underlying distribution doesn't support x, logpdf should be -Inf
                    @test logpdf_val == -Inf
                end
            end
        end
    end
end

@testitem "Test IntervalCensored logpdf boundary conditions" begin
    using Distributions

    # Test precise boundary handling
    d = Normal(0, 1)
    intervals = [-2.0, 0.0, 2.0]
    ic = interval_censored(d, intervals)

    # Test values exactly at boundaries
    @test logpdf(ic, -2.0) > -Inf  # Lower boundary of first interval
    @test logpdf(ic, 0.0) > -Inf   # Boundary between intervals
    @test logpdf(ic, 2.0) == -Inf  # Upper boundary of last interval (no mass)

    # Test values just inside/outside boundaries
    ε = 1e-12
    @test logpdf(ic, -2.0 + ε) > -Inf  # Just inside first interval
    @test logpdf(ic, -2.0 - ε) == -Inf # Just outside first interval
    @test logpdf(ic, 2.0 - ε) > -Inf   # Just inside last interval
    @test logpdf(ic, 2.0 + ε) == -Inf  # Just outside last interval

    # Test with regular intervals at boundaries
    ic_reg = interval_censored(d, 1.0)

    # Values at multiples of interval should work
    boundary_vals = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    for x in boundary_vals
        logpdf_val = logpdf(ic_reg, x)
        @test logpdf_val ≠ NaN
        @test isfinite(logpdf_val) || logpdf_val == -Inf
    end
end
