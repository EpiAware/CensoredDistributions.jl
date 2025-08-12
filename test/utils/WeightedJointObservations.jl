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
    @test logpdf(wd, (0.0, 2.0)) == 2.0 * logpdf(d, 0.0)
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
    joint_obs = (x, 2.0)
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
end

@testitem "Test extract_obs function" begin
    using Distributions
    using CensoredDistributions: extract_obs

    # Test scalar observation
    val, weight = extract_obs(2.5)
    @test val == 2.5
    @test ismissing(weight)

    # Test vector observation
    vals = [1.0, 2.0, 3.0]
    val2, weight2 = extract_obs(vals)
    @test val2 == vals
    @test ismissing(weight2)

    # Test joint observation tuple
    joint = ([1.0, 2.0], [3.0, 4.0])
    @test extract_obs(joint) == joint
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
    joint_obs = (values, obs_weights)

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
    joint_obs = (values, obs_weights)

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
    joint_obs = (values, obs_weights)

    # Expected weights: [2.0*1.0, missing*2.0, 3.0*1.0] = [2.0, 2.0, 3.0]
    expected = 2.0 * logpdf(d1, 0.5) + 2.0 * logpdf(d2, 1.5) + 3.0 * logpdf(d3, 2.5)

    @test logpdf(mixed_dist, joint_obs) ≈ expected
end

@testitem "Test weight stacking with zero weights" begin
    using Distributions

    d = Normal(0, 1)

    # Test zero constructor weight
    wd_zero = weight(d, 0.0)
    @test logpdf(wd_zero, (1.0, 5.0)) == -Inf  # 0 * 5 = 0, zero weight gives -Inf

    # Test zero observation weight
    wd_nonzero = weight(d, 3.0)
    @test logpdf(wd_nonzero, (1.0, 0.0)) == -Inf  # 3 * 0 = 0, zero weight gives -Inf
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
    joint_obs = (values, obs_weights)

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
