@testitem "latent constructs a multivariate wrapper over [primary, observed]" begin
    using Distributions

    d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
    ld = latent(d)

    @test ld isa CensoredDistributions.Latent
    @test ld isa Distribution{Multivariate, Continuous}
    @test length(ld) == 2
    # The plain node stays the univariate marginal default.
    @test d isa UnivariateDistribution

    # Accessors delegate to the wrapped node.
    @test get_dist(ld) === get_dist(d)
    @test get_primary_event(ld) === get_primary_event(d)
end

@testitem "latent rand produces the internal times [primary, observed]" begin
    using Distributions, Random

    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0, 1)
    ld = latent(primary_censored(delay, pe))

    rng = MersenneTwister(42)
    x = rand(rng, ld)
    @test x isa AbstractVector
    @test length(x) == 2
    p, y = x[1], x[2]
    @test insupport(pe, p)        # primary drawn from the primary prior
    @test y >= p                  # observed = primary + non-negative delay
end

@testitem "latent logpdf = primary prior + conditional" begin
    using Distributions

    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0, 1)
    d = primary_censored(delay, pe)
    ld = latent(d)

    for (p, y) in [(0.3, 2.7), (0.1, 1.0), (0.9, 5.4)]
        expected = logpdf(pe, p) + primary_conditional_logpdf(d, p, y)
        @test logpdf(ld, [p, y]) ≈ expected
    end
end

@testitem "primary_conditional_logpdf scores the delay at the implied gap" begin
    using Distributions

    delay = LogNormal(1.5, 0.75)
    d = primary_censored(delay, Uniform(0, 1))

    for (p, y) in [(0.3, 2.7), (0.0, 1.0), (0.5, 4.0)]
        @test primary_conditional_logpdf(d, p, y) ≈ logpdf(delay, y - p)
    end

    # Works on the Latent wrapper too (delegates to the wrapped node).
    @test primary_conditional_logpdf(latent(d), 0.3, 2.7) ≈
          logpdf(delay, 2.7 - 0.3)
end

@testitem "marginal logpdf equals the latent joint integrated over the primary" begin
    using Distributions

    # The marginal PrimaryCensored integrates the primary out inside logpdf; the
    # latent wrapper keeps it explicit. Integrating the latent joint over the
    # primary window must reproduce the marginal density.
    for (delay,
        pe) in [
        (LogNormal(1.5, 0.75), Uniform(0.0, 1.0)),
        (Gamma(2.0, 1.0), Uniform(0.0, 1.0)),
        (Weibull(2.0, 1.5), Uniform(0.0, 2.0))
    ]
        dm = primary_censored(delay, pe)
        ld = latent(dm)
        lo, hi = minimum(pe), maximum(pe)
        function integrate_primary(y; n = 200_000)
            ps = range(lo, hi; length = n)
            vals = map(p -> exp(logpdf(ld, [p, y])), ps)
            return sum((vals[1:(end - 1)] .+ vals[2:end]) ./ 2) * step(ps)
        end
        for y in [1.0, 2.5, 4.0]
            @test isapprox(integrate_primary(y), pdf(dm, y); rtol = 1e-3)
        end
    end
end
