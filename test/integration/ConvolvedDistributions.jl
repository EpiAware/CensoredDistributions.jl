# Tests for CensoredDistributionsConvolvedDistributionsExt: the
# `convolve_series` bridge that discretises a censoring scheme onto the unit
# integer grid and forwards to ConvolvedDistributions' PMF-vector
# convolution. See EpiAware/ConvolvedDistributions.jl#31.

@testitem "convolve_series extension loads with ConvolvedDistributions" begin
    using CensoredDistributions
    using ConvolvedDistributions: ConvolvedDistributions

    ext = Base.get_extension(
        CensoredDistributions, :CensoredDistributionsConvolvedDistributionsExt)
    @test ext isa Module
end

@testitem "convolve_series PMF matches hand-computed masses" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    # double_interval_censored(...; interval = 1) is IntervalCensored around
    # a (truncated) PrimaryCensored inner distribution. The lag-k mass is the
    # interval mass on [k, k + 1), i.e. cdf(inner, k + 1) - cdf(inner, k) of
    # the inner distribution (a CD-tested primitive).
    dic = double_interval_censored(
        LogNormal(1.5, 0.75); upper = 10, interval = 1)
    inner = CensoredDistributions.get_dist(dic)
    n = 8

    ref_pmf = [cdf(inner, k + 1) - cdf(inner, k) for k in 0:(n - 1)]
    # The extension reads pdf(dic, k) as the lag-k mass; it must equal the
    # inner CDF difference.
    for k in 0:(n - 1)
        @test pdf(dic, k) ≈ ref_pmf[k + 1]
    end

    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0, 1.0]
    # convolve_series through the extension equals the PMF-vector method fed
    # the same hand-computed masses.
    @test convolve_series(dic, series) ≈ convolve_series(ref_pmf, series)
end

@testitem "convolve_series extension equals explicit causal convolution" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    dic = double_interval_censored(Gamma(2.0, 1.5); interval = 1)
    series = [2.0, 4.0, 7.0, 3.0, 1.0, 0.0, 5.0]
    n = length(series)
    pmf = [pdf(dic, k) for k in 0:(n - 1)]

    # Independent causal, window-truncated convolution of the same masses.
    expected = map(1:n) do i
        sum(pmf[k + 1] * series[i - k] for k in 0:(min(length(pmf), i) - 1))
    end
    @test convolve_series(dic, series) ≈ expected
end

@testitem "convolve_series extension: bare interval_censored unit grid" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    # A bare interval_censored(dist, 1) is also a supported unit grid.
    ic = interval_censored(Normal(5, 2), 1)
    series = [1.0, 2.0, 3.0, 4.0, 5.0]
    n = length(series)
    pmf = [pdf(ic, k) for k in 0:(n - 1)]
    @test convolve_series(ic, series) ≈ convolve_series(pmf, series)
end

@testitem "convolve_series extension rejects non-unit grids" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    series = [1.0, 2.0, 3.0]

    # Non-unit regular interval.
    ic_wide = interval_censored(Normal(5, 2), 2.0)
    @test_throws ArgumentError convolve_series(ic_wide, series)

    # double_interval_censored with a non-unit interval is the same reject.
    dic_wide = double_interval_censored(LogNormal(1.5, 0.75); interval = 2)
    @test_throws ArgumentError convolve_series(dic_wide, series)

    # Arbitrary interval boundaries.
    ic_arb = interval_censored(Normal(5, 2), [0.0, 1.0, 3.0, 6.0])
    @test_throws ArgumentError convolve_series(ic_arb, series)
end

@testitem "convolve_series extension rejects continuous primary censoring" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    pc = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
    series = [1.0, 2.0, 3.0]
    # Primary censoring is still continuous: no unit grid to convolve on.
    @test_throws ArgumentError convolve_series(pc, series)
end
