@testitem "WithinWindowPrimary constructor and validation" begin
    using Distributions

    d = within_window_primary(0.0, 1.0, 0.7)
    @test d isa CensoredDistributions.WithinWindowPrimary
    @test d.lower == 0.0
    @test d.width == 1.0
    @test d.secondary == 0.7

    # Type promotion
    dp = within_window_primary(0, 1, 0.5)
    @test eltype(typeof(dp)) == Float64

    @test_throws ArgumentError within_window_primary(0.0, 0.0, 0.5)
    @test_throws ArgumentError within_window_primary(0.0, -1.0, 0.5)
    # secondary below lower violates non-negative delay
    @test_throws ArgumentError within_window_primary(0.0, 1.0, -0.5)
end

@testitem "WithinWindowPrimary support bounded by secondary" begin
    using Distributions

    # secondary inside the window truncates the upper edge
    d = within_window_primary(0.0, 1.0, 0.6)
    @test minimum(d) == 0.0
    @test maximum(d) == 0.6
    @test insupport(d, 0.3)
    @test !insupport(d, 0.7)

    # secondary beyond the window keeps the full window
    d2 = within_window_primary(2.0, 1.0, 5.0)
    @test minimum(d2) == 2.0
    @test maximum(d2) == 3.0
end

@testitem "WithinWindowPrimary logpdf matches implicit uniform prior" begin
    using Distributions

    # Core acceptance property: the Jacobian-corrected logpdf reproduces the
    # implicit independent-uniform-over-window prior of the marginalised model,
    # regardless of where the secondary time bounds the window.
    for (lower,
        width,
        secondary) in [
        (0.0, 1.0, 0.7), (0.0, 1.0, 0.4), (3.0, 1.0, 10.0),
        (0.0, 2.0, 1.5), (5.0, 0.5, 5.2)
    ]
        d = within_window_primary(lower, width, secondary)
        implicit = Uniform(lower, lower + width)
        upper = min(lower + width, secondary)
        # Test points strictly inside the bounded support
        for t in range(lower + 1e-6, upper - 1e-6; length = 5)
            @test logpdf(d, t) ≈ logpdf(implicit, t)
            @test pdf(d, t) ≈ pdf(implicit, t)
        end
    end
end

@testitem "WithinWindowPrimary logpdf out of support" begin
    using Distributions

    d = within_window_primary(0.0, 1.0, 0.6)
    @test logpdf(d, -0.1) == -Inf
    @test logpdf(d, 0.7) == -Inf  # above truncated upper edge
    @test pdf(d, 0.7) == 0.0
end

@testitem "WithinWindowPrimary cdf over bounded support" begin
    using Distributions

    d = within_window_primary(0.0, 1.0, 0.5)
    @test cdf(d, -1.0) == 0.0
    @test cdf(d, 0.0) == 0.0
    @test cdf(d, 0.25) ≈ 0.5
    @test cdf(d, 0.5) == 1.0
    @test cdf(d, 1.0) == 1.0
    @test logcdf(d, 0.25) ≈ log(0.5)
end

@testitem "WithinWindowPrimary params, eltype and sampler" begin
    using Distributions

    d = within_window_primary(1.0, 2.0, 1.5)
    @test params(d) == (1.0, 2.0, 1.5)
    @test eltype(typeof(d)) == Float64

    # Float32 inputs preserve the element type
    d32 = within_window_primary(1.0f0, 2.0f0, 1.5f0)
    @test eltype(typeof(d32)) == Float32
    @test rand(d32) isa Float32

    # sampler returns the distribution itself (self-sampling)
    s = sampler(d)
    @test s === d
    r = rand(s)
    @test 1.0 <= r <= 1.5
end

@testitem "WithinWindowPrimary sampling stays in bounded window" begin
    using Distributions, Random

    rng = MersenneTwister(123)
    d = within_window_primary(2.0, 1.0, 2.6)
    samples = rand(rng, d, 10_000)
    @test all(2.0 .<= samples .<= 2.6)
    # Uniform over [2.0, 2.6], mean ~ 2.3
    @test isapprox(mean(samples), 2.3; atol = 0.02)
end

@testitem "WithinWindowPrimary reproduces primary-censored prior by integration" begin
    using Distributions, Random

    # Acceptance test mirroring the bdbv PR #11 comparison: building a delay in
    # the latent form (sample primary time within its window, score the delay as
    # the deterministic difference, contribute the Jacobian-corrected prior)
    # reproduces the marginalised primary-censored CDF.
    rng = MersenneTwister(2024)
    delay = LogNormal(1.5, 0.75)
    width = 1.0
    pc = primary_censored(delay, Uniform(0.0, width))

    # Monte Carlo over the latent representation. The implicit prior on the
    # primary time equals logpdf of WithinWindowPrimary (Jacobian included), so
    # drawing it uniformly over the window and adding an independent delay sample
    # reconstructs the marginal observed-delay distribution.
    n = 2_000_000
    function mc_cdf(q)
        cnt = 0
        for _ in 1:n
            d_lat = within_window_primary(0.0, width, width)  # full window
            t_pri = rand(rng, d_lat)
            obs = t_pri + rand(rng, delay)
            cnt += (obs <= q)
        end
        return cnt / n
    end

    for q in [1.0, 3.0, 6.0]
        @test isapprox(mc_cdf(q), cdf(pc, q); atol = 2e-3)
    end
end
