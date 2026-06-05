# Tests for the path-ascertainment / completeness-thinning layer (#300).

@testitem "completeness_probability is the chain-completion CDF" begin
    using Distributions

    incubation = LogNormal(1.5, 0.5)
    @test completeness_probability(incubation, 6.0) == cdf(incubation, 6.0)

    # A non-positive window returns zero (the horizon has not passed the
    # chain origin).
    @test completeness_probability(incubation, 0.0) == 0.0
    @test completeness_probability(incubation, -1.0) == 0.0

    # A convolved chain uses the total-delay CDF.
    chain = convolve_distributions(incubation, Gamma(2.0, 1.0))
    @test completeness_probability(chain, 6.0) == cdf(chain, 6.0)
end

@testitem "thin_by_completeness reproduces andv R_eff = R * p" begin
    using Distributions

    R = 2.5
    chain = convolve_distributions(LogNormal(1.5, 0.5), Gamma(2.0, 1.0))

    for horizon in (3.0, 6.0, 12.0)
        p = cdf(chain, horizon)
        @test thin_by_completeness(R, chain, horizon) ≈ R * p
    end

    # Past the horizon the offspring cannot have completed, so R_eff is zero.
    @test thin_by_completeness(R, chain, -1.0) == 0.0

    # A single-segment chain thins by the single-delay CDF.
    single = LogNormal(1.5, 0.5)
    @test thin_by_completeness(R, single, 6.0) ≈ R * cdf(single, 6.0)
end

@testitem "completeness thinning matches a Monte-Carlo completion rate" begin
    using Distributions, Random

    Random.seed!(42)
    R = 3.0
    chain = convolve_distributions(LogNormal(1.0, 0.4), Gamma(1.5, 1.0))
    horizon = 5.0

    # Simulate offspring whose chain delay either completes by the horizon or
    # not; the observed fraction is the completeness probability.
    n = 2_000_000
    completed = count(_ -> rand(chain) <= horizon, 1:n) / n
    @test completeness_probability(chain, horizon)≈completed rtol=0.02
    @test thin_by_completeness(R, chain, horizon)≈R * completed rtol=0.02
end
