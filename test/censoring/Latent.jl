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

@testitem "latent defines no scalar observed density" begin
    using Distributions

    # The latent form is genuinely latent: it does not re-marginalise the primary
    # from a scalar observed value (that duplicates the marginal default and
    # defeats latent mode). The observed marginal is the marginal node's job,
    # recovered via `marginal(d)`. So a scalar density on `Latent` has no method.
    ld = latent(primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1)))

    @test_throws MethodError logpdf(ld, 2.5)
    @test_throws MethodError pdf(ld, 2.5)
    @test_throws MethodError cdf(ld, 2.5)

    # The marginal observed density is recovered from the unwrapped node.
    d = marginal(ld)
    @test logpdf(d, 2.5) isa Real
    @test cdf(d, 2.5) isa Real
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

@testitem "latent rand observed times round-trip to the marginal cdf" begin
    using Distributions, Random

    # Drawing many latent records and keeping the observed component must
    # recover the marginal observed-delay cdf (the marginal node's analytic cdf),
    # confirming the latent draw and the marginal density describe one process.
    d = primary_censored(LogNormal(1.4, 0.5), Uniform(0, 1))
    ld = latent(d)

    rng = MersenneTwister(20240610)
    obs = [rand(rng, ld).observed for _ in 1:200_000]
    for x in [1.0, 3.0, 6.0]
        empirical = count(<=(x), obs) / length(obs)
        @test isapprox(empirical, cdf(d, x); atol = 5e-3)
    end
end

@testitem "latent joint integrates over the primary to the marginal density" begin
    using Distributions

    # Density-correctness proof. A high-resolution trapezoidal integration of the
    # latent joint over the primary window must reproduce the analytic marginal
    # pdf/cdf. The latent joint and the marginal density therefore describe one
    # process; the joint adds the explicit primary, the marginal integrates it
    # out. If they disagree the latent density is wrong.
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
        # ∫ pdf(prior, p) * cdf(conditional, x) dp recovers the cdf: the primary
        # prior weighting the conditional cdf at each primary.
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
        end
        for x in [1.0, 2.5, 4.0]
            @test isapprox(integrate_cdf(x), cdf(dm, x); rtol = 1e-3)
        end
    end
end

@testitem "double_interval_censored works in both marginal and latent forms" begin
    using Distributions

    # The censoring wrappers must work for both forms. The latent form samples
    # the primary and keeps the secondary interval (and truncation) on the
    # conditional: the primary is realised, not marginalised, and the modifiers
    # stay on the total `p + delay` (#723). So the conditional is NOT the bare
    # delay, and integrating the joint over the primary reproduces the analytic
    # `double_interval_censored` marginal (the interval-censored node), not the
    # continuous primary-censored one.
    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0, 1)

    dm = double_interval_censored(delay; primary_event = pe, interval = 1)
    ld = latent(dm)

    @test marginal(ld) === dm
    for (p, y) in [(0.3, 2.7), (0.1, 1.0), (0.9, 5.4)]
        # The conditional keeps the secondary interval, so it differs from the
        # bare-delay shift (the #723 fix; stripping the interval was the bug).
        @test logpdf(ld, [p, y]) != logpdf(pe, p) + logpdf(delay, y - p)
        cond = CensoredDistributions.PrimaryConditional(dm, p)
        @test logpdf(ld, [p, y]) ≈ logpdf(pe, p) + logpdf(cond, y)
    end

    # Truncated + interval-censored double form keeps both modifiers.
    dtic = double_interval_censored(
        delay; primary_event = pe, upper = 10, interval = 1)
    ltic = latent(dtic)
    @test marginal(ltic) === dtic
    for (p, y) in [(0.2, 3.0), (0.5, 6.0)]
        cond = CensoredDistributions.PrimaryConditional(dtic, p)
        @test logpdf(ltic, [p, y]) ≈ logpdf(pe, p) + logpdf(cond, y)
    end
end

@testitem "latent pipeline joint integrates to the analytic marginal" begin
    using Distributions
    using CensoredDistributions: PrimaryConditional, get_primary_event

    # The latent `double_interval_censored` joint, integrated over the primary,
    # must reproduce the analytic interval-censored (and truncated) marginal: the
    # primary is sampled and the secondary modifiers kept, so the latent and
    # marginal describe one process. Covers interval-only, interval+upper, and
    # interval+lower+upper.
    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0, 1)

    function integrate_joint(marg, y; n = 200_000)
        ld = latent(marg)
        ps = range(0.0, 1.0; length = n)
        vals = map(ps) do p
            pdf(pe, p) * exp(logpdf(PrimaryConditional(ld, p), y))
        end
        return sum((vals[1:(end - 1)] .+ vals[2:end]) ./ 2) * step(ps)
    end

    margs = [
        double_interval_censored(delay; primary_event = pe, interval = 1),
        double_interval_censored(
            delay; primary_event = pe, upper = 10, interval = 1),
        double_interval_censored(
            delay; primary_event = pe, lower = 0.5, upper = 12, interval = 1)
    ]
    for marg in margs
        for y in [0.0, 1.0, 3.0, 6.0, 9.0]
            @test isapprox(integrate_joint(marg, y), pdf(marg, y); atol = 5e-4)
        end
    end
end

@testitem "latent reaches through interval/truncation wrappers" begin
    using Distributions
    using CensoredDistributions: get_primary_event, get_dist, PrimaryConditional

    # A bare double_interval_censored node wraps its PrimaryCensored node in an
    # IntervalCensored (and a Truncated when bounds are given). latent over such
    # a node must build, sample and score, reaching the primary event through the
    # wrappers while keeping the secondary interval on the conditional.
    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0, 1)

    # Interval-censored node.
    dic = double_interval_censored(delay; primary_event = pe, interval = 1)
    @test dic isa CensoredDistributions.IntervalCensored
    lic = latent(dic)
    @test get_primary_event(lic) === pe
    # The latent conditional keeps the secondary interval, so it differs from the
    # bare-delay shift (the separable `primary_censored` leaf still scores bare).
    lpc = latent(primary_censored(delay, pe))
    for (p, y) in [(0.3, 2.7), (0.1, 1.0), (0.9, 5.4)]
        @test logpdf(lic, [p, y]) != logpdf(lpc, [p, y])
        cond = PrimaryConditional(dic, p)
        @test logpdf(lic, [p, y]) ≈ logpdf(pe, p) + logpdf(cond, y)
    end
    # `rand` runs (no MethodError) and is in support.
    r = rand(lic)
    @test r.observed >= 0
    @test marginal(lic) == dic

    # Truncated + interval-censored node: get_primary_event reaches through both.
    dtic = double_interval_censored(
        delay; primary_event = pe, upper = 10, interval = 1)
    ltic = latent(dtic)
    @test get_primary_event(ltic) === pe
    cond = PrimaryConditional(dtic, 0.3)
    @test logpdf(ltic, [0.3, 2.7]) ≈ logpdf(pe, 0.3) + logpdf(cond, 2.7)
end

@testitem "latent joint logpdf is ForwardDiff-safe in the parameters" begin
    using Distributions
    using ForwardDiff: gradient

    # The marginal == latent equivalence holds at the gradient level: the
    # parameter gradient of the marginal logpdf equals the gradient of the latent
    # joint integrated over the primary window.
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
