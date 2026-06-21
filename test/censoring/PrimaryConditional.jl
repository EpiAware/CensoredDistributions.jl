@testitem "PrimaryConditional logpdf scores the delay at the implied gap" begin
    using Distributions

    delay = LogNormal(1.5, 0.75)
    d = primary_censored(delay, Uniform(0, 1))

    for (p, y) in [(0.3, 2.7), (0.0, 1.0), (0.5, 4.0)]
        pc = PrimaryConditional(d, p)
        @test pc isa UnivariateDistribution
        @test logpdf(pc, y) ≈ logpdf(delay, y - p)
        @test pdf(pc, y) ≈ pdf(delay, y - p)
        @test cdf(pc, y) ≈ cdf(delay, y - p)
        # The named function is the same as the distribution's logpdf.
        @test primary_conditional_logpdf(d, p, y) ≈ logpdf(pc, y)
    end
end

@testitem "PrimaryConditional support is y > p" begin
    using Distributions

    delay = LogNormal(1.5, 0.75)
    d = primary_censored(delay, Uniform(0, 1))
    p = 0.4
    pc = PrimaryConditional(d, p)

    @test minimum(pc) == p + minimum(delay)
    @test !insupport(pc, p - 0.1)        # below the primary
    @test insupport(pc, p + 0.5)
    @test logpdf(pc, p - 0.1) == -Inf
end

@testitem "PrimaryConditional rand produces p + delay (> p)" begin
    using Distributions, Random

    delay = LogNormal(1.5, 0.75)
    d = primary_censored(delay, Uniform(0, 1))
    p = 0.3
    pc = PrimaryConditional(d, p)

    rng = MersenneTwister(7)
    samples = rand(rng, pc, 1000)
    @test all(samples .> p)
    # rand is p plus an independent delay draw
    rng2 = MersenneTwister(11)
    @test rand(rng2, pc) ≈ p + rand(MersenneTwister(11), delay)
end

@testitem "PrimaryConditional works on a Latent wrapper" begin
    using Distributions

    delay = LogNormal(1.5, 0.75)
    d = primary_censored(delay, Uniform(0, 1))
    ld = latent(d)
    p, y = 0.3, 2.7

    # Delegates to the wrapped node, so the conditional is identical.
    @test logpdf(PrimaryConditional(ld, p), y) ≈ logpdf(PrimaryConditional(d, p), y)
    @test logpdf(PrimaryConditional(ld, p), y) ≈ logpdf(delay, y - p)
end

@testitem "Latent joint reuses PrimaryConditional (single source)" begin
    using Distributions, Random

    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0, 1)
    d = primary_censored(delay, pe)
    ld = latent(d)

    # The Latent joint is exactly the prior plus PrimaryConditional, with no
    # separate y - p logic.
    for (p, y) in [(0.3, 2.7), (0.1, 1.0), (0.9, 5.4)]
        @test logpdf(ld, [p, y]) ≈
              logpdf(pe, p) + logpdf(PrimaryConditional(d, p), y)
    end

    # rand draws the primary then the observed from PrimaryConditional.
    rng = MersenneTwister(3)
    x = rand(rng, ld)
    @test x[2] > x[1]
end
