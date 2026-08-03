@testitem "convolve_series PMF matches hand-computed masses" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    dic = double_interval_censored(
        LogNormal(1.5, 0.75); upper = 10, interval = 1)
    inner = CensoredDistributions.get_dist(dic)
    n = 8

    ref_pmf = [cdf(inner, k + 1) - cdf(inner, k) for k in 0:(n - 1)]
    for k in 0:(n - 1)
        @test pdf(dic, k) ≈ ref_pmf[k + 1]
    end

    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0, 1.0]
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

    w = 7
    dic = double_interval_censored(
        LogNormal(2.5, 0.75); upper = 70, interval = w)
    inner = CensoredDistributions.get_dist(dic)
    n = 8

    ref_pmf = [cdf(inner, w * (k + 1)) - cdf(inner, w * k) for k in 0:(n - 1)]
    for k in 0:(n - 1)
        @test pdf(dic, w * k) ≈ ref_pmf[k + 1]
    end

    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0, 1.0]
    @test convolve_series(dic, series) ≈ convolve_series(ref_pmf, series)
end

@testitem "convolve_series: bare weekly interval_censored" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

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

    ic_arb = interval_censored(Normal(5, 2), [0.0, 1.0, 3.0, 6.0])
    @test_throws ArgumentError convolve_series(ic_arb, series)
end

@testitem "convolve_series rejects continuous primary censoring" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    pc = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
    series = [1.0, 2.0, 3.0]
    @test_throws ArgumentError convolve_series(pc, series)
end

@testitem "convolve_series: continuous delay discretises via double interval" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    d = LogNormal(1.5, 0.75)
    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]
    ref = convolve_series(double_interval_censored(d; interval = 1), series)
    @test convolve_series(d, series) ≈ ref
end

@testitem "convolve_series: continuous delay honours the interval keyword" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    w = 7
    d = LogNormal(2.5, 0.75)
    series = [0.0, 1.0, 3.0, 6.0, 8.0]
    ref = convolve_series(double_interval_censored(d; interval = w), series)
    @test convolve_series(d, series; interval = w) ≈ ref
end

@testitem "convolve_series: continuous delay forwards keywords" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    d = LogNormal(1.5, 0.75)
    pe = Uniform(0, 2)
    series = [0.0, 1.0, 3.0, 6.0, 8.0]
    ref = convolve_series(
        double_interval_censored(d; interval = 1, primary_event = pe), series)
    @test convolve_series(d, series; primary_event = pe) ≈ ref
    # A different primary event gives a different result.
    @test !(convolve_series(d, series) ≈ ref)
end

@testitem "convolve_series: time-varying fast path reads the grid PMF" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    import ConvolvedDistributions: delay_masses
    using Distributions

    ic = double_interval_censored(LogNormal(1.5, 0.75); interval = 1)
    n = 6
    grid = [pdf(ic, k) for k in 0:(n - 1)]
    @test delay_masses(ic, n) ≈ grid
    # The generic hook recovers the same masses by convolving a unit impulse.
    impulse = [i == 1 ? 1.0 : 0.0 for i in 1:n]
    @test delay_masses(ic, n) ≈ convolve_series(ic, impulse)
end

@testitem "convolve_series: time-varying interval-censored delays" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    series = [0.0, 1.0, 3.0, 6.0, 8.0]
    n = length(series)
    delays = [double_interval_censored(LogNormal(m, 0.6); interval = 1)
              for m in range(1.0, 1.8; length = n)]

    # Reference: scatter each cohort forward through its own delay (:primary).
    masses = [[pdf(delays[j], k) for k in 0:(n - 1)] for j in 1:n]
    expected = zeros(n)
    for s in 1:n
        for lag in 0:(n - s)
            expected[s + lag] += masses[s][lag + 1] * series[s]
        end
    end
    @test convolve_series(delays, series) ≈ expected
end

@testitem "convolve_series: time-varying continuous delays discretise" begin
    using CensoredDistributions
    using ConvolvedDistributions: convolve_series
    using Distributions

    series = [0.0, 1.0, 3.0, 6.0, 8.0]
    n = length(series)
    raw = [LogNormal(m, 0.6) for m in range(1.0, 1.8; length = n)]
    dic = [double_interval_censored(d; interval = 1) for d in raw]
    @test convolve_series(raw, series) ≈ convolve_series(dic, series)
end
