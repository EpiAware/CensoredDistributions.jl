@testitem "Test IntervalCensored constructor" begin
    using Distributions

    # Test regular interval construction
    d = Normal(0, 1)
    ic_regular = interval_censored(d, 1.0)
    @test typeof(ic_regular) <: CensoredDistributions.IntervalCensored
    @test ic_regular.dist === d
    @test ic_regular.intervals == 1.0
    @test CensoredDistributions.is_regular_intervals(ic_regular) == true

    # Test arbitrary interval construction
    intervals = [0.0, 2.0, 5.0, 10.0]
    ic_arbitrary = interval_censored(d, intervals)
    @test typeof(ic_arbitrary) <: CensoredDistributions.IntervalCensored
    @test ic_arbitrary.dist === d
    @test ic_arbitrary.intervals == intervals
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

    # Test insupport
    @test insupport(ic, 0.0) == true  # 0.0 is a multiple of 0.5
    @test insupport(ic, 0.5) == true
    @test insupport(ic, 1.0) == true
    @test insupport(ic, 0.3) == false  # Not a multiple of 0.5
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

    # Test insupport - only interval boundaries are in support
    @test insupport(ic, 0.0) == true
    @test insupport(ic, 2.0) == true
    @test insupport(ic, 5.0) == true
    @test insupport(ic, 10.0) == true
    @test insupport(ic, 1.0) == false  # Not a boundary
    @test insupport(ic, 3.0) == false  # Not a boundary
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

    # Test single interval case (WithinIntervalCensored compatibility)
    d = Normal(5, 2)
    ic = interval_censored(d, [2.0, 4.0])

    # pdf() without argument should work for single interval
    expected_pdf = cdf(d, 4.0) - cdf(d, 2.0)
    @test pdf(ic) ≈ expected_pdf
    @test logpdf(ic) ≈ log(expected_pdf)

    # Multiple intervals should throw error
    ic_multi = interval_censored(d, [2.0, 4.0, 6.0])
    @test_throws ArgumentError pdf(ic_multi)
    @test_throws ArgumentError logpdf(ic_multi)
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

    # Test single interval case - should return lower bound
    ic_single = interval_censored(d, [2.0, 4.0])
    samples_single = rand(rng, ic_single, 100)
    @test all(s == 2.0 for s in samples_single)  # All samples should be lower bound

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
