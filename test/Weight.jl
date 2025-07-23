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