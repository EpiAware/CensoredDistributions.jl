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
