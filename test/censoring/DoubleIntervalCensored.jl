@testitem "Test double_interval_censored structure equivalence - primary only" begin
    using Distributions

    # Test case 1: Primary censoring only
    delay_dist = LogNormal(1.5, 0.75)
    primary_event_dist = Uniform(0, 1)

    # Manual approach
    manual_dist = primary_censored(delay_dist, primary_event_dist)

    # Convenience function approach
    convenience_dist = double_interval_censored(delay_dist, primary_event_dist)

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
        delay_dist, primary_event_dist; upper = upper_bound)

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
        delay_dist, primary_event_dist; interval = interval_width)

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
    convenience_dist = double_interval_censored(delay_dist, primary_event_dist;
        upper = upper_bound, interval = interval_width)

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
    convenience_dist = double_interval_censored(delay_dist, primary_event_dist;
        lower = lower_bound, upper = upper_bound)

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
    @test isa(double_interval_censored(gamma_dist, uniform_primary),
        CensoredDistributions.PrimaryCensored)

    # Primary + truncation -> Truncated{PrimaryCensored}
    @test isa(double_interval_censored(gamma_dist, uniform_primary; upper = 5),
        Truncated)

    # Primary + interval censoring -> IntervalCensored{PrimaryCensored}
    @test isa(double_interval_censored(gamma_dist, uniform_primary; interval = 0.5),
        CensoredDistributions.IntervalCensored)

    # Full pipeline -> IntervalCensored{Truncated{PrimaryCensored}}
    full_dist = double_interval_censored(
        gamma_dist, uniform_primary; upper = 5, interval = 0.5)
    @test isa(full_dist, CensoredDistributions.IntervalCensored)
end

@testitem "Test double_interval_censored with nothing parameters" begin
    using Distributions

    # Test with nothing values (should be equivalent to not providing the argument)
    gamma_dist = Gamma(2, 1)
    uniform_primary = Uniform(0, 1)

    dist1 = double_interval_censored(gamma_dist, uniform_primary)
    dist2 = double_interval_censored(gamma_dist, uniform_primary;
        lower = nothing, upper = nothing, interval = nothing)

    @test typeof(dist1) == typeof(dist2)
    @test params(dist1) == params(dist2)
end
