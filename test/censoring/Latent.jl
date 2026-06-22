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
    # The latent draw is a labelled NamedTuple (the scored representation is
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

@testitem "latent joint logpdf = primary prior + conditional" begin
    using Distributions

    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0, 1)
    d = primary_censored(delay, pe)
    ld = latent(d)

    for (p, y) in [(0.3, 2.7), (0.1, 1.0), (0.9, 5.4)]
        expected = logpdf(pe, p) + logpdf(PrimaryConditional(d, p), y)
        @test logpdf(ld, [p, y]) ≈ expected
    end
end

@testitem "PrimaryConditional scores the delay at the implied gap" begin
    using Distributions

    delay = LogNormal(1.5, 0.75)
    d = primary_censored(delay, Uniform(0, 1))

    for (p, y) in [(0.3, 2.7), (0.0, 1.0), (0.5, 4.0)]
        @test logpdf(PrimaryConditional(d, p), y) ≈ logpdf(delay, y - p)
    end

    # Works on the Latent wrapper too (delegates to the wrapped node).
    @test logpdf(PrimaryConditional(latent(d), 0.3), 2.7) ≈
          logpdf(delay, 2.7 - 0.3)
end

@testitem "latent observed marginal interface delegates to marginal(d)" begin
    using Distributions

    # The single-value observed marginal under the latent model equals the
    # wrapped marginal (primary-censored) node. Every Distributions method
    # delegates straight to `marginal(d)` analytically.
    d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
    ld = latent(d)

    for x in [0.5, 1.0, 2.5, 4.0, 7.0]
        @test logpdf(ld, x) ≈ logpdf(d, x)
        @test pdf(ld, x) ≈ pdf(d, x)
        @test cdf(ld, x) ≈ cdf(d, x)
        @test logcdf(ld, x) ≈ logcdf(d, x)
        @test ccdf(ld, x) ≈ ccdf(d, x)
        @test logccdf(ld, x) ≈ logccdf(d, x)
    end
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]
        @test quantile(ld, q) ≈ quantile(d, q)
    end

    # The single-value observed logpdf is distinct from the joint
    # [primary, observed] score.
    @test logpdf(ld, 2.5) != logpdf(ld, [0.3, 2.5])
end

@testitem "latent rand observed times round-trip to the observed marginal" begin
    using Distributions, Random

    # Drawing many latent records and keeping the observed component must
    # recover the observed-delay marginal cdf (the wrapped node's cdf).
    d = primary_censored(LogNormal(1.4, 0.5), Uniform(0, 1))
    ld = latent(d)

    rng = MersenneTwister(20240610)
    obs = [rand(rng, ld).observed for _ in 1:200_000]
    for x in [1.0, 3.0, 6.0]
        empirical = count(<=(x), obs) / length(obs)
        @test isapprox(empirical, cdf(ld, x); atol = 5e-3)
    end
end

@testitem "marginal pdf and cdf equal the latent joint integrated over the primary" begin
    using Distributions

    # CRITICAL density-correctness proof. The marginal node integrates the
    # primary out inside its own logpdf/cdf; the latent wrapper keeps it
    # explicit. Integrating the latent joint density over the primary window
    # must reproduce BOTH the marginal pdf and the marginal cdf. If they
    # disagree the latent density is wrong.
    for (delay,
        pe) in [
        (LogNormal(1.5, 0.75), Uniform(0.0, 1.0)),
        (Gamma(2.0, 1.0), Uniform(0.0, 1.0)),
        (Weibull(2.0, 1.5), Uniform(0.0, 2.0))
    ]
        dm = primary_censored(delay, pe)
        ld = latent(dm)
        lo, hi = minimum(pe), maximum(pe)

        # ∫ exp(logpdf(ld, [p, y])) dp over the primary window.
        function integrate_pdf(y; n = 200_000)
            ps = range(lo, hi; length = n)
            vals = map(p -> exp(logpdf(ld, [p, y])), ps)
            return sum((vals[1:(end - 1)] .+ vals[2:end]) ./ 2) * step(ps)
        end
        # ∫ exp(logpdf(prior, p)) * cdf(conditional, x) dp recovers the cdf:
        # the primary prior weighting the conditional cdf at each primary.
        function integrate_cdf(x; n = 200_000)
            ps = range(lo, hi; length = n)
            vals = map(ps) do p
                exp(logpdf(get_primary_event(ld), p)) *
                cdf(CensoredDistributions.PrimaryConditional(ld, p), x)
            end
            return sum((vals[1:(end - 1)] .+ vals[2:end]) ./ 2) * step(ps)
        end

        for y in [1.0, 2.5, 4.0]
            @test isapprox(integrate_pdf(y), pdf(dm, y); rtol = 1e-3)
            @test isapprox(integrate_pdf(y), pdf(ld, y); rtol = 1e-3)
        end
        for x in [1.0, 2.5, 4.0]
            @test isapprox(integrate_cdf(x), cdf(dm, x); rtol = 1e-3)
            @test isapprox(integrate_cdf(x), cdf(ld, x); rtol = 1e-3)
        end
    end
end

@testitem "double_interval_censored works in both marginal and latent forms" begin
    using Distributions

    # The censoring wrappers must work for both forms. The marginal
    # double_interval_censored node scores the interval-censored observed
    # marginal; wrapping it in latent and scoring a single observed value
    # gives the same density and cdf (latent observed marginal delegates to
    # marginal(d)).
    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0, 1)
    dm = double_interval_censored(delay; primary_event = pe, interval = 1)
    ld = latent(dm)

    @test marginal(ld) === dm
    for x in [0.0, 1.0, 2.0, 4.0, 6.0]
        @test logpdf(ld, x) ≈ logpdf(dm, x)
        @test cdf(ld, x) ≈ cdf(dm, x)
        @test ccdf(ld, x) ≈ ccdf(dm, x)
    end

    # Truncated + interval-censored double form behaves the same way.
    dtic = double_interval_censored(
        delay; primary_event = pe, upper = 10, interval = 1)
    ltic = latent(dtic)
    @test marginal(ltic) === dtic
    for x in [1.0, 3.0, 6.0, 9.0]
        @test logpdf(ltic, x) ≈ logpdf(dtic, x)
        @test cdf(ltic, x) ≈ cdf(dtic, x)
    end
end

@testitem "latent reaches through interval/truncation wrappers" begin
    using Distributions
    using CensoredDistributions: get_primary_event, get_dist

    # A bare double_interval_censored node wraps its PrimaryCensored node in an
    # IntervalCensored (and a Truncated when bounds are given). latent over such
    # a node must build, sample and score, reaching the primary event and the
    # bare continuous delay THROUGH the wrappers.
    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0, 1)

    # Interval-censored node.
    dic = double_interval_censored(delay; primary_event = pe, interval = 1)
    @test dic isa CensoredDistributions.IntervalCensored
    lic = latent(dic)
    @test get_primary_event(lic) === pe
    # The latent conditional scores the BARE continuous delay (the
    # sampled-origin rule): no secondary interval reapplied. So it equals the
    # latent primary-censored node carrying the same primary and continuous
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

    # Truncated + interval-censored node: get_primary_event reaches through both.
    dtic = double_interval_censored(
        delay; primary_event = pe, upper = 10, interval = 1)
    ltic = latent(dtic)
    @test get_primary_event(ltic) === pe
    @test get_dist(ltic) === delay
    @test logpdf(ltic, [0.3, 2.7]) ≈ logpdf(pe, 0.3) + logpdf(delay, 2.7 - 0.3)
end

@testitem "latent logpdf and cdf are ForwardDiff-safe in the parameters" begin
    using Distributions
    using ForwardDiff: gradient

    # The observed-marginal logpdf and cdf of the latent model must
    # differentiate cleanly through the delay parameters under ForwardDiff,
    # and match the marginal node they delegate to.
    pe = Uniform(0.0, 1.0)
    y = 2.5

    marginal_logpdf(θ) = logpdf(primary_censored(LogNormal(θ[1], θ[2]), pe), y)
    latent_logpdf(θ) = logpdf(latent(primary_censored(LogNormal(θ[1], θ[2]), pe)), y)
    marginal_cdf(θ) = cdf(primary_censored(LogNormal(θ[1], θ[2]), pe), y)
    latent_cdf(θ) = cdf(latent(primary_censored(LogNormal(θ[1], θ[2]), pe)), y)

    θ = [1.0, 0.5]
    glp_m = gradient(marginal_logpdf, θ)
    glp_l = gradient(latent_logpdf, θ)
    gcdf_m = gradient(marginal_cdf, θ)
    gcdf_l = gradient(latent_cdf, θ)
    @test all(isfinite, glp_l)
    @test all(isfinite, gcdf_l)
    # The latent observed-marginal gradients equal the marginal node's
    # (they are the same analytic function).
    @test isapprox(glp_l, glp_m; rtol = 1e-8)
    @test isapprox(gcdf_l, gcdf_m; rtol = 1e-8)
end

@testitem "latent joint logpdf is ForwardDiff-safe in the parameters" begin
    using Distributions
    using ForwardDiff: gradient

    # The marginal == latent equivalence must hold at the gradient level: the
    # parameter gradient of the marginal logpdf equals the gradient of the
    # latent joint integrated over the primary window.
    pe = Uniform(0.0, 1.0)
    y = 2.5

    marginal_logpdf(θ) = logpdf(primary_censored(LogNormal(θ[1], θ[2]), pe), y)
    function latent_int_logpdf(θ; n = 20_000)
        ld = latent(primary_censored(LogNormal(θ[1], θ[2]), pe))
        ps = range(0.0, 1.0; length = n)
        vals = map(p -> exp(logpdf(ld, [p, y])), ps)
        trap = sum((vals[1:(end - 1)] .+ vals[2:end]) ./ 2) * step(ps)
        return log(trap)
    end

    θ = [1.0, 0.5]
    gm = gradient(marginal_logpdf, θ)
    gl = gradient(latent_int_logpdf, θ)
    @test all(isfinite, gm)
    @test all(isfinite, gl)
    @test isapprox(gm, gl; rtol = 1e-3)
end
