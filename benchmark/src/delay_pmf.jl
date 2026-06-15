# Build-once delay PMF (#371): discretise the delay PMF ONCE and reuse it across
# a vector of reference-date series, vs rebuilding it on every call. The
# nowcasting shape: one delay, many reference-date count series.
#
# `discretise_pmf`/`DelayPMF` are not a redundant wrapper over
# `double_interval_censored` + `pdf`: the renewal `convolve_distributions(delay,
# series)` rediscretises the delay PMF on every call, so this benchmark measures
# lifting that (relatively expensive) discretisation out of the per-series loop.
# `DelayPMF` is the immutable, AD-safe, cache-free value object that holds the
# discretised masses once for reuse across the whole reference-date vector.

SUITE["DelayPMF"] = BenchmarkGroup()

let
    # A two-step total delay needing numeric discretisation (no analytic
    # convolution), so rediscretising per series is genuinely expensive.
    delay = convolve_distributions(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))

    # Many reference-date series sharing the one delay (fixed params).
    n_series = 50
    horizon = 30
    series_set = [collect(range(0.0, 10.0, length = horizon))
                  for _ in 1:n_series]
    maxlag = horizon - 1

    # Rebuild-every-time: rediscretise the delay inside each convolve call.
    SUITE["DelayPMF"]["rebuild_each"] = @benchmarkable begin
        for s in $series_set
            convolve_distributions($delay, s)
        end
    end

    # Build-once: discretise the PMF a single time, reuse across all series.
    SUITE["DelayPMF"]["build_once"] = @benchmarkable begin
        pmf = CensoredDistributions.discretise_pmf($delay, $maxlag)
        for s in $series_set
            convolve_distributions(pmf, s)
        end
    end
end
