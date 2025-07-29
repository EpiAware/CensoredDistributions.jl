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

@testitem "Test get_dist_recursive with single wrapper distributions" begin
    using Distributions

    # Test that single-wrapped distributions behave same as get_dist
    delay = LogNormal(1.5, 0.75)
    primary = Uniform(0, 1)
    pc = primary_censored(delay, primary)

    @test get_dist_recursive(pc) === get_dist(pc)
    @test get_dist_recursive(pc) === delay

    # Test with IntervalCensored
    continuous = Normal(5, 2)
    ic = interval_censored(continuous, 1.0)

    @test get_dist_recursive(ic) === get_dist(ic)
    @test get_dist_recursive(ic) === continuous

    # Test with Weighted
    base = Gamma(2.0, 3.0)
    wd = weight(base, 2.5)

    @test get_dist_recursive(wd) === get_dist(wd)
    @test get_dist_recursive(wd) === base

    # Test with base distribution (should return unchanged)
    d = Normal(0, 1)
    @test get_dist_recursive(d) === d
    @test get_dist_recursive(d) === get_dist(d)
end

@testitem "Test get_dist_recursive with nested wrapper distributions" begin
    using Distributions

    # Test deeply nested: Weighted{IntervalCensored{Normal}}
    base = Normal(5, 2)
    ic = interval_censored(base, 1.0)
    wd = weight(ic, 3.0)

    @test get_dist_recursive(wd) === base
    @test get_dist_recursive(wd) isa Normal
    @test params(get_dist_recursive(wd)) == params(base)

    # Test: Weighted{PrimaryCensored{LogNormal}}
    delay = LogNormal(1.0, 0.5)
    pc = primary_censored(delay, Uniform(0, 1))
    wd_pc = weight(pc, 1.5)

    @test get_dist_recursive(wd_pc) === delay
    @test get_dist_recursive(wd_pc) isa LogNormal

    # Test: IntervalCensored{Weighted{Gamma}}
    base_gamma = Gamma(2.0, 1.5)
    wd_gamma = weight(base_gamma, 2.0)
    ic_wd = interval_censored(wd_gamma, 0.5)

    @test get_dist_recursive(ic_wd) === base_gamma
    @test get_dist_recursive(ic_wd) isa Gamma

    # Test triple nesting: IntervalCensored{Weighted{PrimaryCensored{Exponential}}}
    base_exp = Exponential(2.0)
    pc_exp = primary_censored(base_exp, Uniform(0, 1))
    wd_pc_exp = weight(pc_exp, 1.8)
    ic_wd_pc = interval_censored(wd_pc_exp, 0.8)

    @test get_dist_recursive(ic_wd_pc) === base_exp
    @test get_dist_recursive(ic_wd_pc) isa Exponential
    @test params(get_dist_recursive(ic_wd_pc)) == params(base_exp)
end

@testitem "Test get_dist_recursive with Product distributions" begin
    using Distributions

    # Test Product with base distributions
    components = [Normal(0, 1), Exponential(1), Gamma(2, 1)]
    pd = product_distribution(components)

    recursive_result = get_dist_recursive(pd)
    @test recursive_result isa Vector
    @test length(recursive_result) == 3
    @test recursive_result[1] === components[1]
    @test recursive_result[2] === components[2]
    @test recursive_result[3] === components[3]

    # Test Product with wrapped distributions
    base1 = Normal(1, 2)
    base2 = Exponential(0.5)
    wd1 = weight(base1, 2.0)
    wd2 = weight(base2, 1.5)

    wrapped_components = [wd1, wd2]
    pd_wrapped = product_distribution(wrapped_components)

    recursive_wrapped = get_dist_recursive(pd_wrapped)
    @test recursive_wrapped isa Vector
    @test length(recursive_wrapped) == 2
    @test recursive_wrapped[1] === base1
    @test recursive_wrapped[2] === base2

    # Test Product with deeply nested wrappers
    deeply_nested = [
        weight(interval_censored(Normal(0, 1), 1.0), 2.0),
        interval_censored(weight(Exponential(1), 1.5), 0.5)
    ]
    pd_deep = product_distribution(deeply_nested)

    recursive_deep = get_dist_recursive(pd_deep)
    @test recursive_deep isa Vector
    @test length(recursive_deep) == 2
    @test recursive_deep[1] isa Normal
    @test recursive_deep[2] isa Exponential
end

@testitem "Test get_dist_recursive with truncated distributions" begin
    using Distributions

    # Test that truncated distributions stop recursion (fallback method)
    trunc_normal = truncated(Normal(0, 1), -2, 2)
    @test get_dist_recursive(trunc_normal) === trunc_normal

    # Test nested with truncated: Weighted{Truncated{Normal}}
    wd_trunc = weight(trunc_normal, 2.5)
    @test get_dist_recursive(wd_trunc) === trunc_normal
    @test get_dist_recursive(wd_trunc) isa Truncated

    # Test: IntervalCensored{Truncated{LogNormal}}
    trunc_lognormal = truncated(LogNormal(0, 1), 0.1, 10)
    ic_trunc = interval_censored(trunc_lognormal, 0.5)
    @test get_dist_recursive(ic_trunc) === trunc_lognormal
    @test get_dist_recursive(ic_trunc) isa Truncated

    # Test PrimaryCensored with truncated delay stops at truncated
    trunc_delay = truncated(Gamma(2, 1), 0, 5)
    pc_trunc = primary_censored(trunc_delay, Uniform(0, 1))
    @test get_dist_recursive(pc_trunc) === trunc_delay
    @test get_dist_recursive(pc_trunc) isa Truncated
end

@testitem "Test get_dist_recursive type consistency and performance" begin
    using Distributions

    # Test that recursive version returns same type as base for simple cases
    base = Normal(0, 1)
    wd = weight(base, 2.0)
    @test typeof(get_dist_recursive(wd)) === typeof(base)

    # Test with nested wrappers
    ic = interval_censored(base, 1.0)
    wd_ic = weight(ic, 1.5)
    @test typeof(get_dist_recursive(wd_ic)) === typeof(base)

    # Test with PrimaryCensored
    delay = LogNormal(1.5, 0.75)
    pc = primary_censored(delay, Uniform(0, 1))
    @test typeof(get_dist_recursive(pc)) === typeof(delay)

    # Test that repeated calls are consistent
    nested = weight(interval_censored(Normal(2, 3), 0.8), 2.2)
    result1 = get_dist_recursive(nested)
    result2 = get_dist_recursive(nested)
    @test result1 === result2

    # Test that results are identical to expected
    @test get_dist_recursive(wd) === base
    @test get_dist_recursive(wd_ic) === base
    @test get_dist_recursive(pc) === delay
end

@testitem "Test get_dist_recursive integration with distribution interface" begin
    using Distributions
    using Random

    Random.seed!(456)

    # Test that recursively extracted distributions work with standard methods
    base = Gamma(2.5, 1.5)
    nested = weight(interval_censored(base, 0.5), 3.0)
    extracted = get_dist_recursive(nested)

    @test extracted === base
    @test mean(extracted) == mean(base)
    @test var(extracted) == var(base)
    @test pdf(extracted, 2.0) == pdf(base, 2.0)
    @test cdf(extracted, 2.0) == cdf(base, 2.0)

    # Test sampling from recursively extracted distribution
    samples = rand(extracted, 50)
    @test length(samples) == 50
    @test all(s -> s > 0, samples)  # Gamma is positive

    # Test with Product distributions
    components = [Normal(1, 2), Exponential(1.5)]
    wrapped_components = [weight(components[1], 2.0), weight(components[2], 1.5)]
    pd = product_distribution(wrapped_components)

    recursive_components = get_dist_recursive(pd)
    @test length(recursive_components) == 2
    @test mean(recursive_components[1]) == mean(components[1])
    @test mean(recursive_components[2]) == mean(components[2])
end
