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

@testitem "Test get_dist with Truncated distributions" begin
    using Distributions

    # Test extraction of untruncated distribution from Truncated
    base = Normal(0, 1)
    trunc_dist = truncated(base, -2, 2)

    extracted = get_dist(trunc_dist)
    @test extracted === base
    @test extracted isa Normal
    @test params(extracted) == params(base)

    # Test with different distribution types
    base_gamma = Gamma(2.0, 1.5)
    trunc_gamma = truncated(base_gamma, 0.5, 10.0)

    extracted_gamma = get_dist(trunc_gamma)
    @test extracted_gamma === base_gamma
    @test extracted_gamma isa Gamma
    @test params(extracted_gamma) == params(base_gamma)

    # Test with LogNormal truncation
    base_lognormal = LogNormal(1.0, 0.5)
    trunc_lognormal = truncated(base_lognormal, 0.5, 5.0)

    extracted_lognormal = get_dist(trunc_lognormal)
    @test extracted_lognormal === base_lognormal
    @test extracted_lognormal isa LogNormal

    # Test with Exponential truncation
    base_exp = Exponential(2.0)
    trunc_exp = truncated(base_exp, 0.0, 10.0)

    extracted_exp = get_dist(trunc_exp)
    @test extracted_exp === base_exp
    @test extracted_exp isa Exponential

    # Test with Normal truncation (creates actual Truncated wrapper)
    base_normal2 = Normal(5, 2)
    trunc_normal2 = truncated(base_normal2, 2, 8)

    extracted_normal2 = get_dist(trunc_normal2)
    @test extracted_normal2 === base_normal2
    @test extracted_normal2 isa Normal

    # Test truncation with infinite bounds
    trunc_lower_only = truncated(Normal(0, 1), 0, Inf)
    extracted_lower = get_dist(trunc_lower_only)
    @test extracted_lower isa Normal
    @test extracted_lower.μ == 0
    @test extracted_lower.σ == 1

    trunc_upper_only = truncated(Normal(0, 1), -Inf, 0)
    extracted_upper = get_dist(trunc_upper_only)
    @test extracted_upper isa Normal
    @test extracted_upper.μ == 0
    @test extracted_upper.σ == 1
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

    # Test Weighted wrapping Truncated
    base_normal = Normal(0, 1)
    trunc_normal = truncated(base_normal, -2, 2)
    wd_trunc = weight(trunc_normal, 3.0)

    extracted_wd_trunc = get_dist(wd_trunc)
    @test extracted_wd_trunc === trunc_normal
    @test extracted_wd_trunc isa Truncated

    # Test IntervalCensored wrapping Truncated
    ic_trunc = interval_censored(trunc_normal, 0.5)
    extracted_ic_trunc = get_dist(ic_trunc)
    @test extracted_ic_trunc === trunc_normal
    @test extracted_ic_trunc isa Truncated

    # Test PrimaryCensored with Truncated delay
    trunc_delay = truncated(LogNormal(1.0, 0.5), 0, 10.0)
    pc_trunc = primary_censored(trunc_delay, Uniform(0, 1))

    extracted_pc_trunc = get_dist(pc_trunc)
    @test extracted_pc_trunc === trunc_delay
    @test extracted_pc_trunc isa Truncated
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

    # Test type stability for Truncated
    base_normal = Normal(2.0, 1.5)
    trunc_normal = truncated(base_normal, -1, 5)
    @inferred get_dist(trunc_normal)
    @test typeof(get_dist(trunc_normal)) === typeof(base_normal)

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

    # Test with Truncated
    base_trunc = Normal(0, 1)
    trunc_dist = truncated(base_trunc, -2, 2)
    extracted_trunc = get_dist(trunc_dist)

    @test params(extracted_trunc) == params(base_trunc)
    @test mean(extracted_trunc) == mean(base_trunc)
    @test std(extracted_trunc) == std(base_trunc)
    @test pdf(extracted_trunc, 0.0) == pdf(base_trunc, 0.0)
    @test cdf(extracted_trunc, 0.0) == cdf(base_trunc, 0.0)

    # Test sampling from extracted truncated distribution
    samples_trunc = rand(extracted_trunc, 50)
    @test length(samples_trunc) == 50
    # Note: samples from extracted (untruncated) can be outside [-2, 2]
    @test any(s -> abs(s) > 2, samples_trunc)  # Some should be outside bounds
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

    # Test Truncated with extreme bounds
    base_extreme = Normal(0, 1)
    extreme_trunc = truncated(base_extreme, -1e10, 1e10)
    @test get_dist(extreme_trunc) === base_extreme

    # Test Truncated with very narrow bounds
    narrow_trunc = truncated(Normal(0, 1), -1e-8, 1e-8)
    @test get_dist(narrow_trunc) isa Normal

    # Test Truncated with bounds at distribution bounds
    uniform_base = Uniform(0, 1)
    same_bounds_trunc = truncated(uniform_base, 0, 1)
    @test get_dist(same_bounds_trunc) === uniform_base
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

    # Through Truncated
    trunc = truncated(original, 0.5, 10.0)
    extracted_trunc = get_dist(trunc)
    @test mean(extracted_trunc) == mean(original)
    @test var(extracted_trunc) == var(original)

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

        # Only add truncated version if it creates a proper Truncated wrapper
        truncated_version = truncated(
            dist, quantile(dist, 0.1), quantile(dist, 0.9)
        )
        # Not the same type = proper Truncated wrapper
        if !(truncated_version isa typeof(dist))
            push!(wrapped_distributions, truncated_version)
        end

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

    # Test with Truncated
    base_trunc = Normal(0, 1)
    trunc_dist = truncated(base_trunc, -2, 2)

    @test get_dist_recursive(trunc_dist) === get_dist(trunc_dist)
    @test get_dist_recursive(trunc_dist) === base_trunc

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

    # Test triple nesting:
    # IntervalCensored{Weighted{PrimaryCensored{Exponential}}}
    base_exp = Exponential(2.0)
    pc_exp = primary_censored(base_exp, Uniform(0, 1))
    wd_pc_exp = weight(pc_exp, 1.8)
    ic_wd_pc = interval_censored(wd_pc_exp, 0.8)

    @test get_dist_recursive(ic_wd_pc) === base_exp
    @test get_dist_recursive(ic_wd_pc) isa Exponential
    @test params(get_dist_recursive(ic_wd_pc)) == params(base_exp)

    # Test nested with Truncated distributions
    base_normal = Normal(0, 1)
    trunc_normal = truncated(base_normal, -3, 3)
    wd_trunc = weight(trunc_normal, 2.5)

    @test get_dist_recursive(wd_trunc) === base_normal
    @test get_dist_recursive(wd_trunc) isa Normal

    # Test: IntervalCensored{Truncated{LogNormal}}
    base_lognormal = LogNormal(0, 1)
    trunc_lognormal = truncated(base_lognormal, 0.1, 10)
    ic_trunc = interval_censored(trunc_lognormal, 0.5)

    @test get_dist_recursive(ic_trunc) === base_lognormal
    @test get_dist_recursive(ic_trunc) isa LogNormal

    # Test: Weighted{IntervalCensored{Truncated{Gamma}}}
    base_gamma_deep = Gamma(2, 1)
    trunc_gamma = truncated(base_gamma_deep, 0.5, 8.0)
    ic_trunc_gamma = interval_censored(trunc_gamma, 0.25)
    wd_ic_trunc = weight(ic_trunc_gamma, 1.2)

    @test get_dist_recursive(wd_ic_trunc) === base_gamma_deep
    @test get_dist_recursive(wd_ic_trunc) isa Gamma

    # Test PrimaryCensored with Truncated delay, then wrapped
    trunc_delay = truncated(LogNormal(1.0, 0.5), 0, 5.0)
    pc_trunc = primary_censored(trunc_delay, Uniform(0, 1))
    wd_pc_trunc = weight(pc_trunc, 2.0)

    @test get_dist_recursive(wd_pc_trunc) === trunc_delay.untruncated
    @test get_dist_recursive(wd_pc_trunc) isa LogNormal
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

    # Test Product with Truncated distributions
    base_trunc = Normal(2, 1)
    trunc_dist = truncated(base_trunc, 0, 4)
    base_exp = Exponential(1.5)

    trunc_components = [trunc_dist, base_exp]
    pd_trunc = product_distribution(trunc_components)

    recursive_trunc = get_dist_recursive(pd_trunc)
    @test recursive_trunc isa Vector
    @test length(recursive_trunc) == 2
    @test recursive_trunc[1] === base_trunc  # Unwrapped from Truncated
    @test recursive_trunc[2] === base_exp    # Already base distribution

    # Test Product with nested Truncated distributions
    nested_trunc_components = [
        weight(truncated(Normal(0, 1), -2, 2), 1.5),
        interval_censored(truncated(Gamma(2, 1), 0.5, 10), 0.25)
    ]
    pd_nested_trunc = product_distribution(nested_trunc_components)

    recursive_nested_trunc = get_dist_recursive(pd_nested_trunc)
    @test recursive_nested_trunc isa Vector
    @test length(recursive_nested_trunc) == 2
    @test recursive_nested_trunc[1] isa Normal  # Fully unwrapped
    @test recursive_nested_trunc[2] isa Gamma   # Fully unwrapped
end

@testitem "Test get_dist_recursive with truncated distributions" begin
    using Distributions

    # Test that standalone truncated distributions unwrap completely
    base = Normal(0, 1)
    trunc_normal = truncated(base, -2, 2)
    @test get_dist_recursive(trunc_normal) === base

    # Test nested with truncated: Weighted{Truncated{Normal}}
    wd_trunc = weight(trunc_normal, 2.5)
    @test get_dist_recursive(wd_trunc) === base
    @test get_dist_recursive(wd_trunc) isa Normal

    # Test: IntervalCensored{Truncated{LogNormal}}
    base_lognormal = LogNormal(0, 1)
    trunc_lognormal = truncated(base_lognormal, 0.1, 10)
    ic_trunc = interval_censored(trunc_lognormal, 0.5)
    @test get_dist_recursive(ic_trunc) === base_lognormal
    @test get_dist_recursive(ic_trunc) isa LogNormal

    # Test PrimaryCensored with truncated delay unwraps fully
    base_gamma = Gamma(2, 1)
    trunc_delay = truncated(base_gamma, 0, 5)  # Use exact 0, not 0.0
    pc_trunc = primary_censored(trunc_delay, Uniform(0, 1))
    @test get_dist_recursive(pc_trunc) === base_gamma
    @test get_dist_recursive(pc_trunc) isa Gamma

    # Test deeply nested: Weighted{IntervalCensored{Truncated{Exponential}}}
    base_exp = Exponential(1.5)
    trunc_exp = truncated(base_exp, 0.1, 8.0)
    ic_trunc_exp = interval_censored(trunc_exp, 0.3)
    wd_ic_trunc_exp = weight(ic_trunc_exp, 1.8)

    @test get_dist_recursive(wd_ic_trunc_exp) === base_exp
    @test get_dist_recursive(wd_ic_trunc_exp) isa Exponential

    # Test multiple Truncated layers (shouldn't occur in practice, but test it)
    # Use Normal since truncated(Uniform) doesn't create Truncated wrappers
    base_normal_deep = Normal(5, 2)
    trunc1 = truncated(base_normal_deep, 1, 9)
    trunc2 = truncated(trunc1, 2, 8)  # Double truncation

    @test get_dist_recursive(trunc2) === base_normal_deep
    @test get_dist_recursive(trunc2) isa Normal
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

    # Test with Truncated
    trunc = truncated(base, -2, 2)
    @test typeof(get_dist_recursive(trunc)) === typeof(base)

    # Test with nested Truncated
    wd_trunc = weight(trunc, 2.0)
    @test typeof(get_dist_recursive(wd_trunc)) === typeof(base)

    # Test that repeated calls are consistent
    nested = weight(interval_censored(Normal(2, 3), 0.8), 2.2)
    result1 = get_dist_recursive(nested)
    result2 = get_dist_recursive(nested)
    @test result1 === result2

    # Test with Truncated nested
    nested_trunc = weight(truncated(Normal(1, 2), -1, 3), 1.5)
    result_trunc1 = get_dist_recursive(nested_trunc)
    result_trunc2 = get_dist_recursive(nested_trunc)
    @test result_trunc1 === result_trunc2

    # Test that results are identical to expected
    @test get_dist_recursive(wd) === base
    @test get_dist_recursive(wd_ic) === base
    @test get_dist_recursive(pc) === delay
    @test get_dist_recursive(trunc) === base
    @test get_dist_recursive(wd_trunc) === base
end

@testitem "Test get_dist_recursive with distribution interface" begin
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

    # Test with Truncated distributions
    base_normal = Normal(0, 1)
    trunc_dist = truncated(base_normal, -1, 1)
    nested_trunc = weight(interval_censored(trunc_dist, 0.1), 2.0)
    extracted_trunc = get_dist_recursive(nested_trunc)

    @test extracted_trunc === base_normal
    @test mean(extracted_trunc) == mean(base_normal)
    @test var(extracted_trunc) == var(base_normal)
    @test pdf(extracted_trunc, 0.0) == pdf(base_normal, 0.0)

    # Test sampling from recursively extracted truncated distribution
    samples_trunc = rand(extracted_trunc, 30)
    @test length(samples_trunc) == 30
    # Note: samples are from untruncated distribution, can exceed [-1, 1]
    @test any(s -> abs(s) > 1, samples_trunc)

    # Test with Product distributions containing Truncated
    base1 = Normal(1, 2)
    base2 = Exponential(1.5)
    trunc1 = truncated(base1, -1, 3)
    components = [weight(trunc1, 2.0), weight(base2, 1.5)]
    pd = product_distribution(components)

    recursive_components = get_dist_recursive(pd)
    @test length(recursive_components) == 2
    @test recursive_components[1] === base1  # Fully unwrapped
    @test recursive_components[2] === base2  # Fully unwrapped
    @test mean(recursive_components[1]) == mean(base1)
    @test mean(recursive_components[2]) == mean(base2)
end
