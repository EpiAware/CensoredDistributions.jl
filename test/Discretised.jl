@testitem "Test Discretised constructor" begin
    using Distributions

    # Test valid construction
    d = Normal(0, 1)
    dd = discretise(d, 1.0)
    @test typeof(dd) <: CensoredDistributions.Discretised
    @test dd.dist === d
    @test dd.interval == 1.0

    # Test with different interval types
    dd_int = discretise(d, 2)
    @test dd_int.interval isa Int
    @test dd_int.interval == 2

    # Test error on zero or negative interval
    @test_throws ArgumentError discretise(d, 0.0)
    @test_throws ArgumentError discretise(d, -1.0)
end

@testitem "Test Discretised aliases" begin
    using Distributions

    d = Normal(5, 2)

    # Test that discretize is an alias for discretise
    dd1 = discretise(d, 0.5)
    dd2 = discretize(d, 0.5)

    @test typeof(dd1) == typeof(dd2)
    @test dd1.dist === dd2.dist
    @test dd1.interval == dd2.interval
end

@testitem "Test Discretised distribution interface" begin
    using Distributions
    using Random

    d = LogNormal(1.0, 0.5)
    interval = 0.5
    dd = discretise(d, interval)

    # Test basic properties
    @test minimum(dd) == minimum(d)
    @test maximum(dd) == maximum(d)

    # Test params
    p = params(dd)
    @test p == (params(d)..., interval)

    # Test eltype
    @test eltype(dd) == Float64
end

@testitem "Test Discretised sampling" begin
    using Distributions
    using Random

    Random.seed!(123)

    d = Normal(10.0, 3.0)
    interval = 2.0
    dd = discretise(d, interval)

    # Test sampling
    samples = rand(dd, 1000)

    # All samples should be multiples of the interval (within floating point precision)
    @test all(abs(s % interval) < 1e-10 || abs((s % interval) - interval) < 1e-10
    for s in samples)

    # Samples should be in a reasonable range around the distribution mean
    @test minimum(samples) >= 0  # Should be non-negative for this test case
    @test maximum(samples) < 25  # Should be reasonable given mean=10, std=3
end

@testitem "Test Discretised with different distributions" begin
    using Distributions
    using Random

    Random.seed!(456)

    # Test with exponential distribution
    exp_dist = Exponential(2.0)
    dd_exp = discretise(exp_dist, 0.1)

    samples_exp = rand(dd_exp, 100)
    @test all(abs(s % 0.1) < 1e-10 || abs((s % 0.1) - 0.1) < 1e-10 for s in samples_exp)

    # Test with uniform distribution
    unif_dist = Uniform(5, 15)
    dd_unif = discretise(unif_dist, 1.0)

    samples_unif = rand(dd_unif, 100)
    @test all(abs(s % 1.0) < 1e-10 for s in samples_unif)
    @test all(5 <= s <= 15 for s in samples_unif)  # Should be within original bounds
end