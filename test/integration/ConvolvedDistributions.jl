# Tests for the `convolve_series` bridge (src/censoring/convolve_series.jl):
# discretises a censoring scheme onto its own grid and forwards to
# ConvolvedDistributions' PMF-vector convolution. ConvolvedDistributions is a
# hard dependency (see the `sources` note in Project.toml), so these methods
# are always available -- no extension-load gate to test. See
# EpiAware/ConvolvedDistributions.jl#31.

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
    # convolve_series reads pdf(dic, k) as the lag-k mass; it must equal the
    # inner CDF difference.
    for k in 0:(n - 1)
        @test pdf(dic, k) ≈ ref_pmf[k + 1]
    end

    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0, 1.0]
    # convolve_series on the censored delay equals the PMF-vector method fed
    # the same hand-computed masses.
    @test convolve_series(dic, series) ≈ convolve_series(ref_pmf, series)
end

@testitem "convolve_series equals explicit causal convolution" begin
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

@testitem "convolve_series: bare interval_censored unit grid" begin
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

@testitem "convolve_series: weekly (w = 7) grid masses" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    # The grid width comes from the censored distribution's own interval, so
    # a weekly (interval = 7) delay discretises onto the 7-day grid. The
    # lag-k mass is the interval mass on [7k, 7(k + 1)), i.e.
    # cdf(inner, 7(k + 1)) - cdf(inner, 7k) of the inner (truncated,
    # primary-censored) distribution.
    w = 7
    dic = double_interval_censored(
        LogNormal(2.5, 0.75); upper = 70, interval = w)
    inner = CensoredDistributions.get_dist(dic)
    n = 8

    ref_pmf = [cdf(inner, w * (k + 1)) - cdf(inner, w * k) for k in 0:(n - 1)]
    # convolve_series reads pdf(dic, w * k) as the lag-k mass on [7k, 7(k+1));
    # it must equal the inner CDF difference on the weekly grid.
    for k in 0:(n - 1)
        @test pdf(dic, w * k) ≈ ref_pmf[k + 1]
    end

    # The series is read on the same weekly grid: convolve_series on the
    # censored delay equals the PMF-vector method fed the hand-computed
    # weekly masses.
    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0, 1.0]
    @test convolve_series(dic, series) ≈ convolve_series(ref_pmf, series)
end

@testitem "convolve_series: bare weekly interval_censored" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    # A bare interval_censored(dist, 7) discretises onto the same weekly grid
    # and is accepted; the width is read off the distribution, not assumed.
    w = 7
    ic = interval_censored(Normal(35, 8), w)
    series = [1.0, 2.0, 3.0, 4.0, 5.0]
    n = length(series)
    pmf = [pdf(ic, w * k) for k in 0:(n - 1)]
    @test convolve_series(ic, series) ≈ convolve_series(pmf, series)
end

@testitem "convolve_series rejects irregular boundaries" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    series = [1.0, 2.0, 3.0]

    # Arbitrary (irregular) interval boundaries have no single grid step for
    # the causal convolution to shift by, so they are rejected.
    ic_arb = interval_censored(Normal(5, 2), [0.0, 1.0, 3.0, 6.0])
    @test_throws ArgumentError convolve_series(ic_arb, series)
end

@testitem "convolve_series rejects continuous primary censoring" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    pc = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
    series = [1.0, 2.0, 3.0]
    # Primary censoring is still continuous: no unit grid to convolve on.
    @test_throws ArgumentError convolve_series(pc, series)
end
