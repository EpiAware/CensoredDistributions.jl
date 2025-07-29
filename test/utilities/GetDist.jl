@testitem "Test get_dist fallback for base distributions" begin
    using Distributions

    # Test that unwrapped distributions return themselves
    d1 = Normal(0, 1)
    @test get_dist(d1) === d1

    d2 = Exponential(2.0)
    @test get_dist(d2) === d2

    d3 = Gamma(2.0, 3.0)
    @test get_dist(d3) === d3

    d4 = LogNormal(1.5, 0.75)
    @test get_dist(d4) === d4

    d5 = Uniform(0, 1)
    @test get_dist(d5) === d5

    # Test with discrete distributions
    d6 = Poisson(5)
    @test get_dist(d6) === d6

    d7 = Binomial(10, 0.3)
    @test get_dist(d7) === d7
end

@testitem "Test get_dist with PrimaryCensored distributions" begin
    using Distributions

    # Test extraction of delay distribution
    delay = LogNormal(1.5, 0.75)
    primary = Uniform(0, 1)
    pc = primary_censored(delay, primary)

    extracted = get_dist(pc)
    @test extracted === delay
    @test extracted isa LogNormal
    @test params(extracted) == params(delay)

    # Test with different distribution types
    delay2 = Gamma(2.0, 3.0)
    primary2 = Uniform(0, 2)
    pc2 = primary_censored(delay2, primary2)

    extracted2 = get_dist(pc2)
    @test extracted2 === delay2
    @test extracted2 isa Gamma

    # Test with Exponential delay
    delay3 = Exponential(1.5)
    pc3 = primary_censored(delay3, primary)

    extracted3 = get_dist(pc3)
    @test extracted3 === delay3
    @test extracted3 isa Exponential
end

@testitem "Test get_dist with IntervalCensored distributions" begin
    using Distributions

    # Test extraction of underlying continuous distribution
    continuous = Normal(5, 2)
    ic = interval_censored(continuous, 1.0)

    extracted = get_dist(ic)
    @test extracted === continuous
    @test extracted isa Normal
    @test params(extracted) == params(continuous)

    # Test with different underlying distribution
    continuous2 = LogNormal(2.0, 0.5)
    ic2 = interval_censored(continuous2, 0.5)

    extracted2 = get_dist(ic2)
    @test extracted2 === continuous2
    @test extracted2 isa LogNormal

    # Test with arbitrary boundaries
    continuous3 = Exponential(1.0)
    boundaries = [0.0, 1.0, 2.5, 5.0, 10.0]
    ic3 = interval_censored(continuous3, boundaries)

    extracted3 = get_dist(ic3)
    @test extracted3 === continuous3
    @test extracted3 isa Exponential
end

@testitem "Test get_dist with Weighted distributions" begin
    using Distributions

    # Test extraction of underlying distribution from weighted distribution
    base = Normal(0, 1)
    wd = weight(base, 2.5)

    extracted = get_dist(wd)
    @test extracted === base
    @test extracted isa Normal
    @test params(extracted) == params(base)

    # Test with different distribution types and weights
    base2 = LogNormal(1.5, 0.75)
    wd2 = weight(base2, 10.0)

    extracted2 = get_dist(wd2)
    @test extracted2 === base2
    @test extracted2 isa LogNormal

    # Test with zero weight
    wd_zero = weight(base, 0.0)
    extracted_zero = get_dist(wd_zero)
    @test extracted_zero === base

    # Test with integer weight
    wd_int = weight(base, 5)
    extracted_int = get_dist(wd_int)
    @test extracted_int === base
end

@testitem "Test get_dist with Product distributions" begin
    using Distributions

    # Test extraction of component distributions from Product
    d1 = Normal(0, 1)
    d2 = Exponential(1)
    pd = product_distribution([d1, d2])

    components = get_dist(pd)
    @test components isa Vector
    @test length(components) == 2
    @test components[1] === d1
    @test components[2] === d2

    # Test with identical components
    d3 = Gamma(2.0, 1.0)
    pd2 = product_distribution([d3, d3, d3])

    components2 = get_dist(pd2)
    @test length(components2) == 3
    @test all(comp === d3 for comp in components2)

    # Test with mixed distribution types
    d4 = Normal(1, 2)
    d5 = LogNormal(0.5, 1.0)
    d6 = Uniform(0, 5)
    pd3 = product_distribution([d4, d5, d6])

    components3 = get_dist(pd3)
    @test length(components3) == 3
    @test components3[1] === d4
    @test components3[2] === d5
    @test components3[3] === d6
    @test components3[1] isa Normal
    @test components3[2] isa LogNormal
    @test components3[3] isa Uniform

    # Test with weighted distributions in Product
    weights = [1.0, 2.0, 3.0]
    wd_array = weight(Normal(0, 1), weights)

    components_weighted = get_dist(wd_array)
    @test length(components_weighted) == 3
    @test all(
        comp isa CensoredDistributions.Weighted for comp in components_weighted
    )
end

@testitem "Test get_dist with nested distributions" begin
    using Distributions

    # Test IntervalCensored wrapping PrimaryCensored
    delay = LogNormal(1.5, 0.5)
    primary = Uniform(0, 1)
    pc = primary_censored(delay, primary)
    ic = interval_censored(pc, 1.0)

    # get_dist should extract the IntervalCensored's underlying distribution
    # (the PrimaryCensored)
    extracted = get_dist(ic)
    @test extracted === pc
    @test extracted isa CensoredDistributions.PrimaryCensored

    # Test Weighted wrapping PrimaryCensored
    wd = weight(pc, 5.0)
    extracted_weighted = get_dist(wd)
    @test extracted_weighted === pc
    @test extracted_weighted isa CensoredDistributions.PrimaryCensored

    # Test Weighted wrapping IntervalCensored
    base = Normal(2, 1)
    ic2 = interval_censored(base, 0.5)
    wd2 = weight(ic2, 2.0)

    extracted_weighted2 = get_dist(wd2)
    @test extracted_weighted2 === ic2
    @test extracted_weighted2 isa CensoredDistributions.IntervalCensored

    # Test nested extraction to get to base distribution
    # For this, we need to apply get_dist recursively
    nested_base = get_dist(get_dist(wd2))
    @test nested_base === base
    @test nested_base isa Normal
end

@testitem "Test get_dist with truncated distributions" begin
    using Distributions

    # Test that truncated distributions are handled by fallback
    trunc_normal = truncated(Normal(0, 1), -2, 2)
    @test get_dist(trunc_normal) === trunc_normal

    # Test weighted truncated distribution
    wd_trunc = weight(trunc_normal, 3.0)
    extracted = get_dist(wd_trunc)
    @test extracted === trunc_normal
    @test extracted isa Truncated

    # Test PrimaryCensored with truncated delay
    trunc_delay = truncated(LogNormal(1.0, 0.5), 0, 10)
    pc_trunc = primary_censored(trunc_delay, Uniform(0, 1))

    extracted_pc = get_dist(pc_trunc)
    @test extracted_pc === trunc_delay
    @test extracted_pc isa Truncated
end

@testitem "Test get_dist return types and type stability" begin
    using Distributions

    # Test type stability for base distributions
    d = Normal(0, 1)
    @inferred get_dist(d)
    @test typeof(get_dist(d)) === typeof(d)

    # Test type stability for PrimaryCensored
    delay = LogNormal(1.5, 0.75)
    primary = Uniform(0, 1)
    pc = primary_censored(delay, primary)
    @inferred get_dist(pc)
    @test typeof(get_dist(pc)) === typeof(delay)

    # Test type stability for IntervalCensored
    continuous = Gamma(2.0, 1.0)
    ic = interval_censored(continuous, 1.0)
    @inferred get_dist(ic)
    @test typeof(get_dist(ic)) === typeof(continuous)

    # Test type stability for Weighted
    base = Exponential(1.5)
    wd = weight(base, 2.0)
    @inferred get_dist(wd)
    @test typeof(get_dist(wd)) === typeof(base)

    # Test Product returns Vector type
    pd = product_distribution([Normal(0, 1), Exponential(1)])
    components = get_dist(pd)
    @test components isa Vector
end

@testitem "Test get_dist integration with distribution interface" begin
    using Distributions
    using Random

    Random.seed!(123)

    # Test that extracted distributions work with standard interface
    delay = LogNormal(1.5, 0.75)
    primary = Uniform(0, 1)
    pc = primary_censored(delay, primary)

    extracted = get_dist(pc)

    # Test that we can use standard distribution methods on extracted
    # distribution
    @test pdf(extracted, 2.0) == pdf(delay, 2.0)
    @test cdf(extracted, 2.0) == cdf(delay, 2.0)
    @test quantile(extracted, 0.5) == quantile(delay, 0.5)

    # Test sampling works
    samples = rand(extracted, 100)
    @test length(samples) == 100
    @test all(s -> s > 0, samples)  # LogNormal is positive

    # Test with IntervalCensored
    continuous = Normal(5, 2)
    ic = interval_censored(continuous, 1.0)
    extracted_ic = get_dist(ic)

    @test mean(extracted_ic) == mean(continuous)
    @test std(extracted_ic) == std(continuous)
    @test minimum(extracted_ic) == minimum(continuous)
    @test maximum(extracted_ic) == maximum(continuous)

    # Test with Weighted
    base = Gamma(2.0, 3.0)
    wd = weight(base, 5.0)
    extracted_wd = get_dist(wd)

    @test params(extracted_wd) == params(base)
    @test insupport(extracted_wd, 1.0) == insupport(base, 1.0)
    @test insupport(extracted_wd, -1.0) == insupport(base, -1.0)
end

@testitem "Test get_dist with edge cases" begin
    using Distributions

    # Test with very small/large parameter values
    tiny_normal = Normal(0, 1e-10)
    @test get_dist(tiny_normal) === tiny_normal

    large_normal = Normal(1e10, 1e5)
    @test get_dist(large_normal) === large_normal

    # Test with extreme weights
    base = Normal(0, 1)
    tiny_weight = weight(base, 1e-10)
    @test get_dist(tiny_weight) === base

    # Test with single-element product distribution
    # (creates DiagNormal, not Product)
    # This creates a DiagNormal which doesn't have a get_dist method,
    # so it returns unchanged
    single_pd = product_distribution([Normal(0, 1)])
    @test single_pd isa DiagNormal
    extracted_single = get_dist(single_pd)
    @test extracted_single === single_pd  # Fallback should return unchanged

    # Test IntervalCensored with very small intervals
    continuous = Normal(0, 1)
    tiny_ic = interval_censored(continuous, 1e-6)
    @test get_dist(tiny_ic) === continuous

    # Test PrimaryCensored with very narrow primary event window
    delay = Exponential(1.0)
    narrow_primary = Uniform(0, 1e-8)
    narrow_pc = primary_censored(delay, narrow_primary)
    @test get_dist(narrow_pc) === delay
end

@testitem "Test get_dist preserves distribution properties" begin
    using Distributions

    # Test that extracted distributions maintain their mathematical properties
    original = Gamma(2.5, 1.5)

    # Through PrimaryCensored
    pc = primary_censored(original, Uniform(0, 1))
    extracted_pc = get_dist(pc)
    @test mean(extracted_pc) == mean(original)
    @test var(extracted_pc) == var(original)

    # Through IntervalCensored
    ic = interval_censored(original, 0.5)
    extracted_ic = get_dist(ic)
    @test mean(extracted_ic) == mean(original)
    @test var(extracted_ic) == var(original)

    # Through Weighted
    wd = weight(original, 3.0)
    extracted_wd = get_dist(wd)
    @test mean(extracted_wd) == mean(original)
    @test var(extracted_wd) == var(original)

    # Test parameter preservation
    test_distributions = [
        Normal(2.5, 1.2),
        LogNormal(0.5, 0.8),
        Exponential(2.0),
        Gamma(3.0, 0.5),
        Uniform(1.0, 5.0)
    ]

    for dist in test_distributions
        # Test through various wrappers
        wrapped_distributions = Any[
            weight(dist, 2.0),
            interval_censored(dist, 1.0)
        ]

        # Only test PrimaryCensored for distributions suitable as delay
        # distributions
        # (non-negative support, appropriate for delay modeling)
        if (dist isa LogNormal || dist isa Exponential || dist isa Gamma) &&
           (minimum(dist) >= 0 || isinf(minimum(dist)))
            push!(wrapped_distributions, primary_censored(dist, Uniform(0, 1)))
        end

        for wrapped in wrapped_distributions
            extracted = get_dist(wrapped)
            @test extracted === dist
            @test params(extracted) == params(dist)
        end
    end
end
