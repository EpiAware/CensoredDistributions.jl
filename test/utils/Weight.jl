@testitem "Test Weight constructor" begin
    using Distributions

    # Test valid construction
    d = Normal(0, 1)
    wd = weight(d, 10.0)
    @test typeof(wd) <: CensoredDistributions.Weighted
    @test wd.dist === d
    @test wd.weight == 10.0

    # Test with different distribution types
    wd_exp = weight(Exponential(2.0), 5.5)
    @test wd_exp.dist isa Exponential
    @test wd_exp.weight == 5.5

    # Test error on negative weight
    @test_throws ArgumentError weight(d, -1.0)

    # Test zero weight is allowed
    wd_zero = weight(d, 0.0)
    @test wd_zero.weight == 0.0
end

@testitem "Test Weight distribution interface" begin
    using Distributions
    using Random

    d = LogNormal(1.5, 0.5)
    w = 2.5
    wd = weight(d, w)

    # Test basic properties
    @test minimum(wd) == minimum(d)
    @test maximum(wd) == maximum(d)
    @test insupport(wd, 1.0) == insupport(d, 1.0)
    @test insupport(wd, -1.0) == insupport(d, -1.0)

    # Test params
    p = params(wd)
    @test p == (params(d)..., w)

    # Test eltype
    @test eltype(wd) == Float64
end

@testitem "Test Weight probability functions" begin
    using Distributions

    d = Normal(2.0, 1.0)
    w = 3.0
    wd = weight(d, w)
    x = 2.5

    # Test pdf - should be unweighted
    @test pdf(wd, x) == pdf(d, x)

    # Test logpdf - should be weighted
    @test logpdf(wd, x) == w * logpdf(d, x)

    # Test CDF methods - should be unweighted
    @test cdf(wd, x) == cdf(d, x)
    @test logcdf(wd, x) == logcdf(d, x)
    @test ccdf(wd, x) == ccdf(d, x)
    @test logccdf(wd, x) == logccdf(d, x)

    # Test quantile - should be unweighted
    @test quantile(wd, 0.5) == quantile(d, 0.5)
end

@testitem "Test Weight with zero weight" begin
    using Distributions

    d = Normal(0, 1)
    wd = weight(d, 0.0)

    # Test that logpdf returns -Inf for zero weight
    @test logpdf(wd, 0.0) == -Inf
    @test logpdf(wd, 1.0) == -Inf

    # Other methods should work normally
    @test pdf(wd, 0.0) == pdf(d, 0.0)
    @test cdf(wd, 0.0) == cdf(d, 0.0)
end

@testitem "Test Weight sampling" begin
    using Distributions
    using Random
    using Statistics

    Random.seed!(123)

    d = Normal(5.0, 2.0)
    w = 10.0
    wd = weight(d, w)

    # Sampling should be unaffected by weight
    samples_d = rand(MersenneTwister(123), d, 10000)
    samples_wd = rand(MersenneTwister(123), wd, 10000)

    # Check that samples follow the underlying distribution
    @test mean(samples_wd) ≈ mean(d) atol=0.1
    @test std(samples_wd) ≈ std(d) atol=0.1
    @test mean(samples_d) ≈ mean(samples_wd) atol=0.1
    @test std(samples_d) ≈ std(samples_wd) atol=0.1
end

@testitem "Test Weight with different numeric types" begin
    using Distributions

    d = Normal(0, 1)

    # Test with different weight types
    wd_int = weight(d, 5)
    @test wd_int.weight isa Int
    @test logpdf(wd_int, 0.0) == 5 * logpdf(d, 0.0)

    wd_float32 = weight(d, Float32(2.5))
    @test wd_float32.weight isa Float32
    @test logpdf(wd_float32, 0.0) ≈ 2.5 * logpdf(d, 0.0)
end

@testitem "Test Weight product distribution constructor" begin
    using Distributions

    d = Normal(1, 2)
    weights = [1.0, 2.0, 3.0]

    wd_array = weight(d, weights)

    # Should create a Product distribution
    @test wd_array isa Product
    @test length(wd_array) == 3

    # Each component should be a Weight distribution
    for (i, w) in enumerate(weights)
        component = wd_array.v[i]
        @test component isa CensoredDistributions.Weighted
        @test component.weight == w
        @test component.dist === d
    end

    # Test logpdf on array
    x = [0.5, 1.0, 1.5]
    expected_logpdf = sum(weights .* logpdf.(d, x))
    @test logpdf(wd_array, x) ≈ expected_logpdf
end

@testitem "Test Weight with truncated distributions" begin
    using Distributions

    d = truncated(Normal(0, 1), -2, 2)
    w = 5.0
    wd = weight(d, w)

    # Test that it works with truncated distributions
    @test minimum(wd) == -2
    @test maximum(wd) == 2
    @test insupport(wd, 0.0) == true
    @test insupport(wd, 3.0) == false

    # Test logpdf is weighted correctly
    x = 0.5
    @test logpdf(wd, x) == w * logpdf(d, x)
end

@testitem "Test Weight with PrimaryCensored distributions" begin
    using Distributions

    # Create a primary censored distribution
    pc_dist = primary_censored(LogNormal(2, 0.5), Uniform(0, 1))
    w = 7.0
    wd = weight(pc_dist, w)

    # Test that it works with PrimaryCensored distributions
    @test wd.dist === pc_dist
    @test wd.weight == w

    # Test CDF methods work
    x = 5.0
    @test cdf(wd, x) == cdf(pc_dist, x)
    @test logcdf(wd, x) == logcdf(pc_dist, x)
end

@testitem "Test Weighted with missing constructor weights" begin
    using Distributions

    # Test construction with missing weight
    d = Normal(0, 1)
    wd = CensoredDistributions.Weighted(d, missing)
    @test ismissing(wd.weight)
    @test wd.dist === d

    # Test logpdf with missing constructor weight should return -Inf
    @test logpdf(wd, 0.0) == -Inf
    @test logpdf(wd, 1.0) == -Inf

    # Test joint observation with missing constructor weight uses obs weight
    @test logpdf(wd, (value = 0.0, weight = 2.0)) == 2.0 * logpdf(d, 0.0)
end

@testitem "Test joint observation support" begin
    using Distributions

    d = Normal(2.0, 1.0)
    w = 3.0
    wd = weight(d, w)

    # Test scalar observation (existing functionality)
    x = 2.5
    @test logpdf(wd, x) == w * logpdf(d, x)

    # Test joint observation (value, weight)
    joint_obs = (value = x, weight = 2.0)
    expected_logpdf = (w * 2.0) * logpdf(d, x)  # weight stacking
    @test logpdf(wd, joint_obs) == expected_logpdf
end

@testitem "Test weight combination rules" begin
    using Distributions
    using CensoredDistributions: combine_weights

    # Test dispatch-based combination rules
    @test ismissing(combine_weights(missing, missing))
    @test combine_weights(3.0, missing) == 3.0
    @test combine_weights(missing, 2.0) == 2.0
    @test combine_weights(3.0, 2.0) == 6.0

    # Test zero weight handling
    @test combine_weights(0.0, 5.0) == 0.0
    @test combine_weights(5.0, 0.0) == 0.0
    @test combine_weights(0, 5.0) == 0

    # Test vector combinations
    constructor_weights = [2.0, 3.0, 1.0]
    obs_weights = [1.0, 2.0, 4.0]

    # Vector, Vector → element-wise combination
    result_vec = combine_weights(constructor_weights, obs_weights)
    @test result_vec == [2.0, 6.0, 4.0]

    # Vector, missing → keep constructor weights
    result_missing = combine_weights(constructor_weights, missing)
    @test result_missing == constructor_weights

    # Vector, scalar → broadcast scalar to all elements
    result_scalar = combine_weights(constructor_weights, 2.0)
    @test result_scalar == [4.0, 6.0, 2.0]

    # Test with zero scalar weight
    result_zero = combine_weights(constructor_weights, 0.0)
    @test result_zero == [0.0, 0.0, 0.0]

    # Test with mixed missing constructor weights and scalar observation weight
    mixed_weights = [2.0, missing, 3.0]
    result_mixed = combine_weights(mixed_weights, 5.0)
    @test result_mixed == [10.0, 5.0, 15.0]
end

@testitem "Test weight() constructor with missing weights" begin
    using Distributions

    dists = [Normal(0, 1), Normal(1, 1), Normal(2, 1)]
    weighted_dists = weight(dists)

    # Should create Product distribution
    @test weighted_dists isa Product
    @test length(weighted_dists) == 3

    # Each component should have missing weight
    for component in weighted_dists.v
        @test component isa CensoredDistributions.Weighted
        @test ismissing(component.weight)
    end
end

@testitem "Test Product{<:Any, <:Weighted} logpdf with joint observations" begin
    using Distributions

    # Create weighted distributions
    dists = [Normal(0, 1), Normal(1, 1), Normal(2, 1)]
    constructor_weights = [2.0, 3.0, 1.0]
    weighted_dists = weight(dists, constructor_weights)

    # Test values and observation weights
    values = [0.5, 1.5, 2.5]
    obs_weights = [1.0, 2.0, 3.0]
    joint_obs = (values = values, weights = obs_weights)

    # Expected: sum of (constructor_weight * obs_weight * logpdf(dist, value))
    expected = sum([constructor_weights[i] * obs_weights[i] * logpdf(dists[i], values[i])
                    for i in 1:3])

    @test logpdf(weighted_dists, joint_obs) ≈ expected
end

@testitem "Test Product{<:Any, <:Weighted} with missing constructor weights" begin
    using Distributions

    # Create weighted distributions with missing constructor weights
    dists = [Normal(0, 1), Normal(1, 1), Normal(2, 1)]
    weighted_dists = weight(dists)  # No constructor weights

    # Test values and observation weights
    values = [0.5, 1.5, 2.5]
    obs_weights = [2.0, 3.0, 1.0]
    joint_obs = (values = values, weights = obs_weights)

    # Expected: sum of (obs_weight * logpdf(dist, value)) since constructor weights are missing
    expected = sum([obs_weights[i] * logpdf(dists[i], values[i])
                    for i in 1:3])

    @test logpdf(weighted_dists, joint_obs) ≈ expected
end

@testitem "Test mixed missing weights in Product distribution" begin
    using Distributions

    # Create mixed scenario: some with weights, some without
    d1 = Normal(0, 1)
    d2 = Normal(1, 1)
    d3 = Normal(2, 1)

    wd1 = CensoredDistributions.Weighted(d1, 2.0)
    wd2 = CensoredDistributions.Weighted(d2, missing)
    wd3 = CensoredDistributions.Weighted(d3, 3.0)

    mixed_dist = product_distribution([wd1, wd2, wd3])

    values = [0.5, 1.5, 2.5]
    obs_weights = [1.0, 2.0, 1.0]
    joint_obs = (values = values, weights = obs_weights)

    # Expected weights: [2.0*1.0, missing*2.0, 3.0*1.0] = [2.0, 2.0, 3.0]
    expected = 2.0 * logpdf(d1, 0.5) + 2.0 * logpdf(d2, 1.5) + 3.0 * logpdf(d3, 2.5)

    @test logpdf(mixed_dist, joint_obs) ≈ expected
end

@testitem "Test weight stacking with zero weights" begin
    using Distributions

    d = Normal(0, 1)

    # Test zero constructor weight
    wd_zero = weight(d, 0.0)
    @test logpdf(wd_zero, (value = 1.0, weight = 5.0)) == -Inf  # 0 * 5 = 0, zero weight gives -Inf

    # Test zero observation weight
    wd_nonzero = weight(d, 3.0)
    @test logpdf(wd_nonzero, (value = 1.0, weight = 0.0)) == -Inf  # 3 * 0 = 0, zero weight gives -Inf
end

@testitem "Test backward compatibility" begin
    using Distributions

    # Existing API should continue to work unchanged
    d = LogNormal(1.5, 0.5)
    w = 2.5
    wd = weight(d, w)

    x = 2.0
    @test logpdf(wd, x) == w * logpdf(d, x)  # Original behaviour preserved
    @test pdf(wd, x) == pdf(d, x)            # pdf unaffected
    @test cdf(wd, x) == cdf(d, x)            # cdf unaffected
end

@testitem "Test type stability" begin
    using Distributions

    d = Normal(0.0, 1.0)
    wd_real = weight(d, 2.0)
    wd_missing = CensoredDistributions.Weighted(d, missing)

    # Type annotations should be correct
    @test wd_real isa CensoredDistributions.Weighted{<:Normal, Float64}
    @test wd_missing isa CensoredDistributions.Weighted{<:Normal, Missing}

    # logpdf should return Float64
    @test logpdf(wd_real, 0.0) isa Float64
    @test logpdf(wd_missing, 0.0) isa Float64  # -Inf is Float64
end

@testitem "Test Product distribution with zero weights in vector" begin
    using Distributions

    # Test the edge case where final_weights contains zeros after combination
    dists = [Normal(0, 1), Normal(1, 1), Normal(2, 1)]
    constructor_weights = [2.0, 3.0, 1.0]
    weighted_dists = weight(dists, constructor_weights)

    # Test case 1: observation weight of 0 makes final weight zero
    values = [0.5, 1.5, 2.5]
    obs_weights = [1.0, 0.0, 1.0]  # Middle weight is zero
    joint_obs = (values = values, weights = obs_weights)

    # Should return -Inf because one final weight is zero
    @test logpdf(weighted_dists, joint_obs) == -Inf

    # Test case 2: constructor weight of 0 makes final weight zero
    d1 = Normal(0, 1)
    d2 = Normal(1, 1)
    wd1 = weight(d1, 2.0)
    wd2 = weight(d2, 0.0)  # Zero constructor weight
    mixed_zero = product_distribution([wd1, wd2])

    values2 = [0.5, 1.5]
    @test logpdf(mixed_zero, values2) == -Inf
end

@testitem "Test loglikelihood with vectorized NamedTuple observations" begin
    using Distributions

    # Test loglikelihood for single Weighted distribution with vector observations
    d = Normal(0, 1)
    wd = weight(d, 2.0)

    values = [1.0, 0.5, -0.5]
    weights = [3, 2, 4]
    obs = (values = values, weights = weights)

    # Expected: sum of individual logpdf calls
    expected = sum([logpdf(wd, (value = v, weight = w)) for (v, w) in zip(values, weights)])
    result = loglikelihood(wd, obs)

    @test result ≈ expected
    @test result == sum([2.0 * w * logpdf(d, v) for (v, w) in zip(values, weights)])

    # Test with missing constructor weight
    wd_missing = CensoredDistributions.Weighted(d, missing)
    expected_missing = sum([logpdf(wd_missing, (value = v, weight = w))
                            for (v, w) in zip(values, weights)])
    result_missing = loglikelihood(wd_missing, obs)

    @test result_missing ≈ expected_missing
    @test result_missing == sum([w * logpdf(d, v) for (v, w) in zip(values, weights)])
end

@testitem "Test weight(dist) constructor with missing weight" begin
    using Distributions

    # Test the single-argument weight() constructor
    d = Normal(2.0, 1.0)
    wd = weight(d)

    # Should create Weighted with missing weight
    @test wd isa CensoredDistributions.Weighted
    @test wd.dist === d
    @test ismissing(wd.weight)

    # Test with different distribution types
    wd_exp = weight(Exponential(3.0))
    @test wd_exp.dist isa Exponential
    @test ismissing(wd_exp.weight)

    # Test logpdf with missing constructor weight returns -Inf
    @test logpdf(wd, 1.0) == -Inf

    # Test joint observation uses observation weight directly
    @test logpdf(wd, (value = 1.0, weight = 5.0)) == 5.0 * logpdf(d, 1.0)

    # Test with zero observation weight
    @test logpdf(wd, (value = 1.0, weight = 0.0)) == -Inf
end

@testitem "Test loglikelihood for single Weighted with scalar joint observations" begin
    using Distributions

    d = Normal(1.0, 0.5)
    wd = weight(d, 3.0)

    # Test loglikelihood with single joint observation
    joint_obs = (value = 2.0, weight = 4.0)
    expected = logpdf(wd, joint_obs)
    result = loglikelihood(wd, joint_obs)

    @test result == expected
    @test result == 3.0 * 4.0 * logpdf(d, 2.0)

    # Test with missing constructor weight
    wd_missing = weight(d)
    expected_missing = logpdf(wd_missing, joint_obs)
    result_missing = loglikelihood(wd_missing, joint_obs)

    @test result_missing == expected_missing
    @test result_missing == 4.0 * logpdf(d, 2.0)
end

@testitem "Test loglikelihood for Product distribution with joint observations" begin
    using Distributions

    # Test Product distribution loglikelihood
    dists = [Normal(0, 1), Normal(1, 1), Normal(2, 1)]
    constructor_weights = [2.0, 3.0, 1.5]
    weighted_dists = weight(dists, constructor_weights)

    values = [0.5, 1.2, 2.8]
    obs_weights = [1.0, 2.0, 0.5]
    joint_obs = (values = values, weights = obs_weights)

    # loglikelihood should delegate to logpdf for Product distributions
    expected = logpdf(weighted_dists, joint_obs)
    result = loglikelihood(weighted_dists, joint_obs)

    @test result == expected

    # Verify the actual computation
    expected_manual = sum([constructor_weights[i] * obs_weights[i] *
                           logpdf(dists[i], values[i])
                           for i in 1:3])
    @test result ≈ expected_manual

    # Test with missing constructor weights
    weighted_dists_missing = weight(dists)
    joint_obs_missing = (values = values, weights = obs_weights)

    expected_missing = logpdf(weighted_dists_missing, joint_obs_missing)
    result_missing = loglikelihood(weighted_dists_missing, joint_obs_missing)

    @test result_missing == expected_missing
    @test result_missing ≈ sum([obs_weights[i] * logpdf(dists[i], values[i]) for i in 1:3])
end

@testitem "Test comprehensive weight() method coverage" begin
    using Distributions

    d = LogNormal(1.0, 0.5)

    # Test all weight() constructor variants
    wd_with_weight = weight(d, 2.5)
    @test wd_with_weight.weight == 2.5

    wd_missing = weight(d)
    @test ismissing(wd_missing.weight)

    # Test vector variants
    dists = [d, Normal(0, 1)]
    weights_vec = [1.0, 2.0]

    # weight(dists, weights)
    wd_vec_weighted = weight(dists, weights_vec)
    @test wd_vec_weighted isa Product
    @test length(wd_vec_weighted) == 2
    @test wd_vec_weighted.v[1].weight == 1.0
    @test wd_vec_weighted.v[2].weight == 2.0

    # weight(dists) - missing weights
    wd_vec_missing = weight(dists)
    @test wd_vec_missing isa Product
    @test length(wd_vec_missing) == 2
    @test ismissing(wd_vec_missing.v[1].weight)
    @test ismissing(wd_vec_missing.v[2].weight)

    # Test vector of single distribution
    weights_single = [3.0, 4.0, 5.0]
    wd_single_vec = weight(d, weights_single)
    @test wd_single_vec isa Product
    @test length(wd_single_vec) == 3
    @test all([wd_single_vec.v[i].weight == weights_single[i] for i in 1:3])
end
