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

@testitem "latent rand(d, n) batches n labelled records (no overflow)" begin
    using Distributions, Random

    # Regression for #675: the count form `rand(d, n)` StackOverflowed on a
    # Latent (multivariate but draws a NamedTuple, so the generic matrix
    # fallback recursed). It must batch into n independent labelled records.
    pe = Uniform(0, 1)
    ld = latent(primary_censored(LogNormal(1.4, 0.5), pe))

    rng = MersenneTwister(1)
    draws = rand(rng, ld, 5)
    @test draws isa AbstractVector
    @test length(draws) == 5
    # Each is a valid latent record with the right schema.
    @test all(x -> x isa NamedTuple, draws)
    @test all(x -> keys(x) == (:primary, :observed), draws)
    @test all(x -> insupport(pe, x.primary), draws)
    @test all(x -> x.observed >= x.primary, draws)

    # The no-rng count form batches too, and a seeded rng is reproducible.
    @test length(rand(ld, 4)) == 4
    @test rand(MersenneTwister(7), ld, 3) == rand(MersenneTwister(7), ld, 3)
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

@testitem "latent integration agrees with the analytic marginal target" begin
    using Distributions

    # The latent interface computes the observed density/cdf by integrating the
    # augmented-data joint over the primary (Gauss-Legendre quadrature), a
    # GENUINELY DIFFERENT computation from the analytic `primary_censored`
    # marginal (the target). They must agree to quadrature tolerance: a real
    # validation of the latent formulation, not a tautology.
    d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
    ld = latent(d)

    for x in [0.5, 1.0, 2.5, 4.0, 7.0]
        @test isapprox(logpdf(ld, x), logpdf(d, x); rtol = 1e-6)
        @test isapprox(pdf(ld, x), pdf(d, x); rtol = 1e-6)
        @test isapprox(cdf(ld, x), cdf(d, x); rtol = 1e-6)
        @test isapprox(logcdf(ld, x), logcdf(d, x); rtol = 1e-6)
        @test isapprox(ccdf(ld, x), ccdf(d, x); rtol = 1e-6)
        @test isapprox(logccdf(ld, x), logccdf(d, x); rtol = 1e-6)
    end
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]
        @test isapprox(quantile(ld, q), quantile(d, q); rtol = 1e-4)
    end

    # No method delegates straight to the analytic marginal: the latent cdf is
    # the numeric integral, so it differs from the analytic marginal at the
    # floating-point level even though they agree to quadrature tolerance.
    @test cdf(ld, 2.5) != cdf(d, 2.5)

    # The single-value observed logpdf is distinct from the joint
    # [primary, observed] score.
    @test logpdf(ld, 2.5) != logpdf(ld, [0.3, 2.5])
end

@testitem "latent observed logpdf is numerically robust for Turing init" begin
    using Distributions
    using ForwardDiff: gradient

    # The latent observed logpdf integrates the joint over the primary in log
    # space with the integration bounds clamped to the delay support, so it
    # stays finite and finitely-differentiable at extreme parameters where a
    # naive `log(linear integral)` underflows to `-Inf` and `NaN` gradients.
    # This is what lets the latent model find valid initial parameters under
    # Turing (see the fitting-with-turing tutorial).
    pe = Uniform(0.0, 2.0)
    x = 3.0

    # Below the support the density is genuinely zero: -Inf logpdf, 0 cdf.
    ld0 = latent(primary_censored(LogNormal(1.5, 0.75), pe))
    @test logpdf(ld0, 0.0) == -Inf
    @test cdf(ld0, 0.0) == 0.0

    # Extreme parameters: small-but-finite density that linear-space integration
    # rounds to zero. The log-space logpdf stays finite and ForwardDiff returns
    # a finite gradient.
    for θ in [[3.0, 0.1], [4.0, 0.05], [0.0, 3.0], [-2.0, 2.0]]
        lp(p) = logpdf(latent(primary_censored(LogNormal(p[1], p[2]), pe)), x)
        @test isfinite(lp(θ))
        @test all(isfinite, gradient(lp, θ))
    end
end

@testitem "latent rand observed times round-trip to the observed marginal" begin
    using Distributions, Random

    # Drawing many latent records and keeping the observed component must
    # recover the observed-delay marginal cdf (the latent integral cdf).
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

    # CRITICAL density-correctness proof. An INDEPENDENT high-resolution
    # trapezoidal integration of the latent joint over the primary window must
    # reproduce both the analytic marginal pdf/cdf AND the package's own latent
    # integral (`pdf(ld, ·)`/`cdf(ld, ·)`, the Gauss-Legendre augmented-data
    # integral). If they disagree the latent density is wrong.
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

    # The censoring wrappers must work for both forms. The latent form SAMPLES
    # the primary, so its conditional scores the bare continuous delay (the
    # sampled-origin rule); integrating the augmented joint over the primary
    # therefore reproduces the analytic CONTINUOUS primary-censored marginal,
    # not the interval-censored node. So the analytic target for a latent
    # double_interval_censored leaf is `primary_censored(delay, pe)`.
    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0, 1)
    target = primary_censored(delay, pe)

    dm = double_interval_censored(delay; primary_event = pe, interval = 1)
    ld = latent(dm)

    @test marginal(ld) === dm
    for x in [1.0, 2.0, 4.0, 6.0]
        @test isapprox(pdf(ld, x), pdf(target, x); rtol = 1e-6)
        @test isapprox(cdf(ld, x), cdf(target, x); rtol = 1e-6)
        @test isapprox(ccdf(ld, x), ccdf(target, x); rtol = 1e-6)
    end

    # Truncated + interval-censored double form integrates to the same
    # continuous primary-censored target.
    dtic = double_interval_censored(
        delay; primary_event = pe, upper = 10, interval = 1)
    ltic = latent(dtic)
    @test marginal(ltic) === dtic
    for x in [1.0, 3.0, 6.0, 9.0]
        @test isapprox(pdf(ltic, x), pdf(target, x); rtol = 1e-6)
        @test isapprox(cdf(ltic, x), cdf(target, x); rtol = 1e-6)
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

    # The observed-marginal logpdf and cdf of the latent model are the
    # augmented-data INTEGRAL over the primary; they must differentiate cleanly
    # through the delay parameters under ForwardDiff (the integral propagates
    # Duals through the Gauss-Legendre quadrature), and the integrated-latent
    # gradient must agree with the analytic marginal gradient to quadrature
    # tolerance.
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
    # The integrated-latent gradient agrees with the analytic marginal gradient
    # to quadrature tolerance.
    @test isapprox(glp_l, glp_m; rtol = 1e-5)
    @test isapprox(gcdf_l, gcdf_m; rtol = 1e-5)
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
