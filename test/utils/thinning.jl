@testitem "log_completeness_probability equals the delay logcdf" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: log_completeness_probability

    d = LogNormal(1.5, 0.5)
    for w in (1.0, 3.5, 7.0, 21.0)
        @test log_completeness_probability(d, w) == logcdf(d, w)
        @test log_completeness_probability(d, w) ≈ log(cdf(d, w)) atol=1e-10
    end

    # Works on a Convolved chain, and stays finite where the linear-space
    # completeness underflows to zero.
    chain = convolved(Normal(0.0, 1.0), LogNormal(3.0, 0.3))
    for w in (5.0, 14.0, 30.0)
        @test log_completeness_probability(chain, w) == logcdf(chain, w)
    end
    # A short window against a long delay: completeness rounds to ~0 in linear
    # space, but the log stays finite and large-magnitude negative.
    short = log_completeness_probability(chain, 3.0)
    @test isfinite(short)
    @test short < -5
end

@testitem "log_thin_by_completeness thins a log rate in log space" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: log_thin_by_completeness

    d = LogNormal(1.5, 0.5)
    log_R = log(1.7)
    for w in (1.0, 7.0, 21.0)
        # log of the same thinned rate `R * cdf(d, w)` in linear space.
        @test log_thin_by_completeness(log_R, d, w) ≈
              log(exp(log_R) * cdf(d, w)) atol=1e-10
        @test exp(log_thin_by_completeness(log_R, d, w)) ≈
              exp(log_R) * cdf(d, w) atol=1e-10
    end

    # The log-space form keeps the thinned rate strictly positive even when the
    # completeness underflows, where the linear form collapses to exactly zero.
    chain = convolved(Normal(0.0, 1.0), LogNormal(3.5, 0.3))
    lr = log_thin_by_completeness(log(2.3), chain, 3.0)
    @test isfinite(lr)
    @test exp(lr) > 0
end

@testitem "andv real-time decomposition (index, sourced, R_eff)" begin
    using CensoredDistributions, Distributions

    # The andv real-time model carries three terms, as the andv tutorial
    # relies on. With the secondary incubation `inc` and the per-pair
    # transmission timing `delta`:
    #   1. index right-truncation  -logcdf(inc, window).
    #   2. sourced right-truncation  -log(cdf(inc ⊕ delta, window)), the
    #      convolved-chain completeness denominator (the splitting infection
    #      event is unobserved, so the denominator is the convolution's cdf).
    #   3. offspring completeness thinning  R_eff = R * cdf(chain, window).
    inc = LogNormal(3.06, 0.32)
    delta = Normal(0.17, 0.62)
    chain = convolved(delta, inc)

    R = 2.3
    for window in (37.0, 63.0, 102.0)
        index_norm = -logcdf(inc, window)
        sourced_norm = -logcdf(chain, window)
        @test sourced_norm ≈ -log(cdf(chain, window)) atol=1e-8

        # The two denominators genuinely differ: the convolved chain has more
        # mass beyond the window than the single incubation delay, so its
        # completeness is lower and its truncation penalty larger.
        @test cdf(chain, window) < cdf(inc, window)
        @test sourced_norm > index_norm

        # R_eff = R * cdf(chain, window), the offspring completeness thinning.
        p = cdf(chain, window)
        @test 0.0 < p <= 1.0
        @test R * p <= R
    end
end
