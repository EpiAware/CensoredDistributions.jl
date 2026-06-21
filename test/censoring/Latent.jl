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

@testitem "latent rand produces the labelled record (primary, observed)" begin
    using Distributions, Random

    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0, 1)
    ld = latent(primary_censored(delay, pe))

    rng = MersenneTwister(42)
    x = rand(rng, ld)
    # The latent leaf draw is a labelled NamedTuple (the scored representation is
    # the vector `[primary, observed]`).
    @test x isa NamedTuple
    @test keys(x) == (:primary, :observed)
    @test insupport(pe, x.primary)   # primary drawn from the primary prior
    @test x.observed >= x.primary    # observed = primary + non-negative delay

    # The labelled record round-trips straight back through logpdf (matched by
    # name; the scored representation is the `[primary, observed]` vector).
    @test logpdf(ld, x) ≈ logpdf(ld, [x.primary, x.observed])
    # Field order does not matter; the names do.
    @test logpdf(ld, (observed = x.observed, primary = x.primary)) ≈
          logpdf(ld, x)
end

@testitem "marginal is the inverse of latent (idempotent)" begin
    using Distributions

    d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
    ld = latent(d)
    # marginal unwraps a Latent back to the marginal node it carries.
    @test marginal(ld) === d
    @test marginal(latent(d)) == d
    # Idempotent: a non-Latent node is returned unchanged.
    @test marginal(d) === d
    @test marginal(marginal(ld)) === marginal(ld)
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

@testitem "latent leaf reaches through interval/truncation wrappers" begin
    using Distributions
    using CensoredDistributions: get_primary_event, get_dist

    # A bare double_interval_censored leaf wraps its PrimaryCensored node in an
    # IntervalCensored (and a Truncated when bounds are given). latent over such a
    # leaf used to error in `get_primary_event`. It must now build, sample
    # and score, reaching the primary event and the bare continuous delay THROUGH
    # the wrappers.
    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0, 1)

    # Interval-censored leaf.
    dic = double_interval_censored(delay; primary_event = pe, interval = 1)
    @test dic isa CensoredDistributions.IntervalCensored
    lic = latent(dic)
    @test get_primary_event(lic) === pe
    # The latent conditional scores the BARE continuous delay (the
    # sampled-origin rule): no secondary interval reapplied. So it equals the
    # latent primary-censored leaf carrying the same primary and continuous
    # delay.
    lpc = latent(primary_censored(delay, pe))
    for (p, y) in [(0.3, 2.7), (0.1, 1.0), (0.9, 5.4)]
        @test logpdf(lic, [p, y]) ≈ logpdf(lpc, [p, y])
        @test logpdf(lic, [p, y]) ≈ logpdf(pe, p) + logpdf(delay, y - p)
    end
    # `rand` runs (no MethodError) and is in support.
    r = rand(lic)
    @test r.observed > r.primary
    @test get_dist(lic) === delay
    @test marginal(lic) == dic

    # Truncated + interval-censored leaf: get_primary_event reaches through both.
    dtic = double_interval_censored(
        delay; primary_event = pe, upper = 10, interval = 1)
    ltic = latent(dtic)
    @test get_primary_event(ltic) === pe
    @test get_dist(ltic) === delay
    @test logpdf(ltic, [0.3, 2.7]) ≈ logpdf(pe, 0.3) + logpdf(delay, 2.7 - 0.3)
end
