@testitem "BoundedPrimary via primary_prior: construction and validation" begin
    using Distributions

    d = primary_censored(LogNormal(1.5, 0.75), Uniform(0.0, 1.0))
    bp = primary_prior(d, 0.7)
    @test bp isa CensoredDistributions.BoundedPrimary
    @test minimum(bp) == 0.0
    @test maximum(bp) == 0.7   # bounded above by secondary

    # secondary beyond the window keeps the full window
    bp2 = primary_prior(d, 5.0)
    @test maximum(bp2) == 1.0

    # Non-uniform primary event cannot use the coupled prior
    d_nonunif = primary_censored(
        LogNormal(1.5, 0.75), truncated(Normal(0.5, 0.2), 0, 1))
    @test_throws ArgumentError primary_prior(d_nonunif, 0.5)
end

@testitem "BoundedPrimary logpdf matches implicit uniform prior" begin
    using Distributions

    # Ported WWP property test: the Jacobian-corrected logpdf reproduces the
    # implicit independent-uniform-over-window prior of the marginalised model,
    # regardless of where the secondary time bounds the window.
    for (lower,
        width,
        secondary) in [
        (0.0, 1.0, 0.7), (0.0, 1.0, 0.4), (3.0, 1.0, 10.0),
        (0.0, 2.0, 1.5), (5.0, 0.5, 5.2)
    ]
        d = primary_censored(LogNormal(0.0, 1.0), Uniform(lower, lower + width))
        bp = primary_prior(d, secondary)
        implicit = Uniform(lower, lower + width)
        upper = min(lower + width, secondary)
        for t in range(lower + 1e-6, upper - 1e-6; length = 5)
            @test logpdf(bp, t) ≈ logpdf(implicit, t)
            @test pdf(bp, t) ≈ pdf(implicit, t)
        end
    end
end

@testitem "BoundedPrimary out of support, cdf and sampling" begin
    using Distributions, Random

    d = primary_censored(LogNormal(0.0, 1.0), Uniform(0.0, 1.0))
    bp = primary_prior(d, 0.6)

    @test logpdf(bp, -0.1) == -Inf
    @test logpdf(bp, 0.7) == -Inf
    @test pdf(bp, 0.7) == 0.0

    @test cdf(bp, -1.0) == 0.0
    @test cdf(bp, 0.3) ≈ 0.5
    @test cdf(bp, 0.6) == 1.0
    @test logcdf(bp, 0.3) ≈ log(0.5)

    rng = MersenneTwister(123)
    samples = rand(rng, bp, 10_000)
    @test all(0.0 .<= samples .<= 0.6)
    @test isapprox(mean(samples), 0.3; atol = 0.02)

    @test params(bp) == (0.0, 1.0, 0.6)
    @test eltype(typeof(bp)) == Float64
    @test sampler(bp) === bp
end

@testitem "BoundedPrimary integrates to the marginal observed delay" begin
    using Distributions, Random

    # Mirrors the bdbv PR #11 comparison: drawing the primary from the
    # bounded prior over the full window (Jacobian included) and adding an
    # independent delay sample reconstructs the marginalised primary-censored
    # CDF.
    rng = MersenneTwister(2024)
    delay = LogNormal(1.5, 0.75)
    width = 1.0
    d = primary_censored(delay, Uniform(0.0, width))
    pc = d  # marginal

    n = 2_000_000
    function mc_cdf(q)
        cnt = 0
        for _ in 1:n
            bp = primary_prior(d, width)  # full window
            t_pri = rand(rng, bp)
            obs = t_pri + rand(rng, delay)
            cnt += (obs <= q)
        end
        return cnt / n
    end

    for q in [1.0, 3.0, 6.0]
        @test isapprox(mc_cdf(q), cdf(pc, q); atol = 2e-3)
    end
end
