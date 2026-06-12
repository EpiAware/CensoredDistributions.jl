@testitem "completeness_probability equals the delay CDF (#349)" begin
    using Distributions

    d = LogNormal(1.5, 0.5)
    for w in (1.0, 3.5, 7.0, 21.0)
        @test completeness_probability(d, w) == cdf(d, w)
    end

    # Works on a Convolved chain too.
    chain = convolve_distributions(Gamma(2.0, 1.0), LogNormal(1.0, 0.4))
    for w in (5.0, 14.0, 30.0)
        @test completeness_probability(chain, w) == cdf(chain, w)
    end
end

@testitem "thin_by_completeness scales R by completeness (#349)" begin
    using Distributions

    d = LogNormal(1.5, 0.5)
    R = 1.7
    for w in (1.0, 7.0, 21.0)
        @test thin_by_completeness(R, d, w) == R * cdf(d, w)
    end

    # Convolved chain: R * cdf(convolve_distributions(...), window).
    chain = convolve_distributions(Gamma(2.0, 1.0), LogNormal(1.0, 0.4))
    @test thin_by_completeness(R, chain, 14.0) == R * cdf(chain, 14.0)

    # Thinning by a complete (large) horizon leaves R essentially unchanged.
    @test thin_by_completeness(R, d, 1e6) ≈ R
end

@testitem "andv real-time decomposition (index, sourced, R_eff) (#323)" begin
    using Distributions

    # The andv real-time model carries three terms that the exported helpers
    # must reproduce, as the andv tutorial relies on. With the secondary
    # incubation `inc` and the per-pair transmission timing `delta`:
    #   1. INDEX right-truncation  -logcdf(inc, window).
    #   2. SOURCED right-truncation  -log(cdf(inc ⊕ delta, window)), the
    #      convolved-chain completeness denominator (the splitting infection
    #      event is unobserved, so the denominator is the convolution's CDF).
    #   3. Offspring completeness thinning  R_eff = R * p, with
    #      p = cdf(inc ⊕ delta, window).
    inc = LogNormal(3.06, 0.32)
    delta = Normal(0.17, 0.62)
    chain = convolve_distributions(delta, inc)

    R = 2.3
    for window in (37.0, 63.0, 102.0)
        # Index single-delay denominator.
        index_norm = -log(completeness_probability(inc, window))
        @test completeness_probability(inc, window) == cdf(inc, window)
        @test index_norm ≈ -logcdf(inc, window) atol=1e-10

        # Sourced convolved-chain denominator: the tutorial scores
        # -log(completeness_probability(convolve_distributions(delta, inc), w)).
        sourced_norm = -log(completeness_probability(chain, window))
        @test completeness_probability(chain, window) == cdf(chain, window)
        @test sourced_norm ≈ -logcdf(chain, window) atol=1e-8

        # The two denominators genuinely differ: the convolved chain has more
        # mass beyond the window than the single incubation delay, so its
        # completeness is lower and its truncation penalty larger.
        @test completeness_probability(chain, window) <
              completeness_probability(inc, window)
        @test sourced_norm > index_norm

        # R_eff = R * p, the offspring completeness thinning.
        p = completeness_probability(chain, window)
        @test thin_by_completeness(R, chain, window) ≈ R * p atol=1e-12
        @test 0.0 < p <= 1.0
    end
end
