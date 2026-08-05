@testitem "latent constructs a multivariate wrapper over [primary, observed]" begin
    using Distributions

    d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
    ld = latent(d)

    @test ld isa CensoredDistributions.Latent
    @test ld isa Distribution{Multivariate, Continuous}
    @test length(ld) == 2
    # The plain distribution stays the univariate marginal default.
    @test d isa UnivariateDistribution

    # Accessors delegate to the wrapped distribution.
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
    # The draw is a labelled NamedTuple (scored as `[primary, observed]`).
    @test x isa NamedTuple
    @test keys(x) == (:primary, :observed)
    @test insupport(pe, x.primary)   # primary drawn from the primary prior
    @test x.observed >= x.primary    # observed = primary + non-negative delay

    # The labelled record round-trips through logpdf, matched by field name.
    @test logpdf(ld, x) ≈ logpdf(ld, [x.primary, x.observed])
    # Field order does not matter; the names do.
    @test logpdf(ld, (observed = x.observed, primary = x.primary)) ≈
          logpdf(ld, x)
end

@testitem "latent rand(d, n) batches n labelled records (no overflow)" begin
    using Distributions, Random

    # Regression for #675: `rand(d, n)` StackOverflowed on a Latent; it must
    # batch into n independent labelled records.
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

    # A non-`Int` `Integer` count (e.g. `Int32`) dispatches to the generic
    # `::Integer` batch methods rather than the `::Int` disambiguator.
    @test length(rand(rng, ld, Int32(3))) == 3
    @test length(rand(ld, Int32(2))) == 2
end

@testitem "marginal is the inverse of latent (idempotent)" begin
    using Distributions

    d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
    ld = latent(d)
    # marginal unwraps a Latent back to the marginal distribution it carries.
    @test marginal(ld) === d
    @test marginal(latent(d)) == d
    # Idempotent: a non-Latent distribution is returned unchanged.
    @test marginal(d) === d
    @test marginal(marginal(ld)) === marginal(ld)
end

@testitem "latent joint logpdf = primary prior + conditional" begin
    using Distributions
    using CensoredDistributions: PrimaryConditional

    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0, 1)
    d = primary_censored(delay, pe)
    ld = latent(d)

    for (p, y) in [(0.3, 2.7), (0.1, 1.0), (0.9, 5.4)]
        expected = logpdf(pe, p) + logpdf(PrimaryConditional(d, p), y)
        @test logpdf(ld, [p, y]) ≈ expected
    end
end

@testitem "latent scalar density: conditional on a passed primary" begin
    using Distributions
    using CensoredDistributions: PrimaryConditional, marginal

    # `latent(d)` scalar density is the latent conditional form; with a primary
    # passed it is the deterministic `PrimaryConditional(d, p)` kernel.
    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0, 1)
    d = primary_censored(delay, pe)
    ld = latent(d)

    for (p, y) in [(0.3, 2.7), (0.1, 1.0), (0.9, 5.4)]
        cond = PrimaryConditional(d, p)
        # Deterministic: equals the conditional kernel, repeatable.
        @test logpdf(ld, y; primary = p) == logpdf(cond, y)
        @test logpdf(ld, y; primary = p) == logpdf(ld, y; primary = p)
        @test pdf(ld, y; primary = p) == pdf(cond, y)
        @test cdf(ld, y; primary = p) == cdf(cond, y)
        @test logcdf(ld, y; primary = p) == logcdf(cond, y)
        @test ccdf(ld, y; primary = p) == ccdf(cond, y)
        @test logccdf(ld, y; primary = p) == logccdf(cond, y)
        # The bare conditional is the delay shifted by `p`, no quadrature.
        @test logpdf(ld, y; primary = p) ≈ logpdf(delay, y - p)
    end

    # The conditional is not the marginal pointwise.
    @test logpdf(ld, 2.7; primary = 0.3) != logpdf(marginal(ld), 2.7)
end

@testitem "latent scalar density: MC-samples the primary when absent" begin
    using Distributions, Random
    using CensoredDistributions: get_primary_event

    # With no primary passed, a scalar density call samples one
    # `p ~ get_primary_event(d)` and conditions on it: a stochastic estimate.
    ld = latent(primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1)))

    # Reproducible under a seeded rng; different seeds give different draws.
    a = logpdf(ld, 2.7; rng = MersenneTwister(1))
    @test a == logpdf(ld, 2.7; rng = MersenneTwister(1))
    @test a != logpdf(ld, 2.7; rng = MersenneTwister(2))
    @test isfinite(a)
end

@testitem "latent scalar density: equivalence in expectation only" begin
    using Distributions, Random
    using CensoredDistributions: get_primary_event, marginal

    # The mean over sampled primaries of the conditional density equals the
    # marginal density, though no single conditional call equals it pointwise.
    delay = LogNormal(1.0, 0.5)
    pe = Uniform(0, 1)
    d = primary_censored(delay, pe)
    ld = latent(d)

    rng = MersenneTwister(20260630)
    for y in (1.5, 3.0, 5.0)
        # Monte-Carlo mean of the conditional density over sampled primaries.
        est = mean(pdf(ld, y; primary = rand(rng, pe)) for _ in 1:200_000)
        @test isapprox(est, pdf(marginal(ld), y); rtol = 2e-2)
        # No single conditional draw equals the marginal; only the mean does.
        @test pdf(ld, y; primary = 0.5) != pdf(marginal(ld), y)
    end
end

@testitem "latent scalar quantile and conditional draw" begin
    using Distributions, Random
    using CensoredDistributions: PrimaryConditional

    delay = LogNormal(1.2, 0.4)
    pe = Uniform(0, 1)
    d = primary_censored(delay, pe)
    ld = latent(d)

    # Quantile given a passed primary equals the kernel's, the delay quantile
    # shifted by `p`.
    p = 0.4
    @test quantile(ld, 0.5; primary = p) ==
          quantile(PrimaryConditional(d, p), 0.5)
    @test quantile(ld, 0.5; primary = p) ≈ p + quantile(delay, 0.5)

    # A conditional draw given `p` lands above `p`; the observed-only draw is
    # just `rand` on the conditional.
    r = rand(MersenneTwister(1), PrimaryConditional(d, p))
    @test r > p
    # Reproducible under a seeded rng.
    @test rand(MersenneTwister(3), PrimaryConditional(d, p)) ==
          rand(MersenneTwister(3), PrimaryConditional(d, p))
end

@testitem "latent moments: [primary, observed] vector, observed numeric" begin
    using Distributions, Random, Statistics

    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0, 1)

    # Every moment is the length-2 vector over the [primary, observed] marginals.
    # The primary entry is closed-form; the observed entry is numeric (median via
    # the marginal quantile, mean/var via a fixed-seed Monte Carlo estimate).
    bare = latent(primary_censored(delay, pe))
    mb, vb, sb, medb = mean(bare), var(bare), std(bare), median(bare)
    for x in (mb, vb, sb, medb)
        @test length(x) == 2
    end
    @test mb[1] ≈ mean(pe)
    @test vb[1] ≈ var(pe)
    @test sb ≈ sqrt.(vb)
    @test medb[1] ≈ median(pe)
    @test medb[2] ≈ median(marginal(bare))

    # Bare observed = primary + delay (independent), so the observed mean/var have
    # a closed form to check the Monte Carlo estimate against.
    @test isapprox(mb[2], mean(pe) + mean(delay); rtol = 3e-2)
    @test isapprox(vb[2], var(pe) + var(delay); rtol = 5e-2)

    # Fixed-seed: deterministic across repeated calls.
    @test mean(bare) == mean(bare)
    @test var(bare) == var(bare)

    # Interval-censored + truncated pipeline: the observed entry matches an
    # independent Monte Carlo estimate of the observed marginal.
    pipe = latent(double_interval_censored(delay; primary_event = pe,
        upper = 10, interval = 1))
    mp, vp = mean(pipe), var(pipe)
    @test mp[1] ≈ mean(pe)
    ref = rand(Xoshiro(11), marginal(pipe), 2_000_000)
    @test isapprox(mp[2], mean(ref); rtol = 3e-2)
    @test isapprox(vp[2], var(ref); rtol = 5e-2)
    @test median(pipe)[2] ≈ median(marginal(pipe))
end

@testitem "PrimaryConditional scores the delay at the implied gap" begin
    using Distributions
    using CensoredDistributions: PrimaryConditional

    delay = LogNormal(1.5, 0.75)
    d = primary_censored(delay, Uniform(0, 1))

    for (p, y) in [(0.3, 2.7), (0.0, 1.0), (0.5, 4.0)]
        @test logpdf(PrimaryConditional(d, p), y) ≈ logpdf(delay, y - p)
    end

    # Works on the Latent wrapper too (delegates to the wrapped distribution).
    @test logpdf(PrimaryConditional(latent(d), 0.3), 2.7) ≈
          logpdf(delay, 2.7 - 0.3)
end

@testitem "latent rand observed times round-trip to the marginal cdf" begin
    using Distributions, Random

    # Keeping the observed component of many latent draws recovers the marginal
    # observed-delay cdf.
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

    # A high-resolution trapezoidal integration of the latent joint over the
    # primary window reproduces the analytic marginal pdf/cdf.
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
        # ∫ pdf(prior, p) * cdf(conditional, x) dp recovers the cdf.
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

@testitem "latent conditional equals the marginal in expectation" begin
    using Distributions, Random, Statistics
    using CensoredDistributions: PrimaryConditional, get_primary_event

    # The latent form conditions on a sampled primary, so it matches the marginal
    # only in expectation over the primary prior,
    #   E_p[cdf(PrimaryConditional(d, p), x)] = cdf(marginal(d), x),
    # estimated here by Monte Carlo over `rand` draws of the primary.
    rng = MersenneTwister(20260629)
    n = 400_000

    for (delay,
        pe) in [
        (LogNormal(1.5, 0.75), Uniform(0.0, 1.0)),
        (Gamma(2.0, 1.0), Uniform(0.0, 2.0))
    ]
        dm = primary_censored(delay, pe)
        ld = latent(dm)
        ps = rand(rng, get_primary_event(ld), n)
        for x in [1.0, 2.5, 5.0]
            mc_cdf = mean(cdf(PrimaryConditional(ld, p), x) for p in ps)
            mc_pdf = mean(pdf(PrimaryConditional(ld, p), x) for p in ps)
            @test isapprox(mc_cdf, cdf(dm, x); atol = 5e-3)
            @test isapprox(mc_pdf, pdf(dm, x); atol = 5e-3)
        end
        # A single draw is genuinely conditional: one realised primary gives a
        # shifted conditional, not the marginal cdf.
        one_p = rand(rng, get_primary_event(ld))
        @test cdf(PrimaryConditional(ld, one_p), 2.5) != cdf(dm, 2.5)
    end
end

@testitem "double_interval_censored works in both marginal and latent forms" begin
    using Distributions

    # The censoring wrappers work for both forms. The latent form samples the
    # primary and keeps the secondary interval/truncation on the total
    # `p + delay` (#723), so the conditional is not the bare delay and the joint
    # integrates to the analytic double_interval_censored marginal.
    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0, 1)

    dm = double_interval_censored(delay; primary_event = pe, interval = 1)
    ld = latent(dm)

    @test marginal(ld) === dm
    for (p, y) in [(0.3, 2.7), (0.1, 1.0), (0.9, 5.4)]
        # The conditional keeps the secondary interval, so it differs from the
        # bare-delay shift (the #723 fix).
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
    # reproduces the analytic interval-censored (and truncated) marginal. Covers
    # interval-only, interval+upper, interval+lower+upper.
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

@testitem "latent truncated equivalence guards the marginal Z" begin
    using Distributions
    using CensoredDistributions: PrimaryConditional

    # E_p[latent] over the primary reproduces the analytic
    # double_interval_censored marginal to the numerical floor even for a
    # sensitive truncation window: the lower bound sits near the delay mode, so a
    # one-day primary shift swings the per-primary truncation mass materially.
    # This locks in the marginal (primary-integrated) truncation constant; a
    # per-primary Z with the plain prior would bias this by ~1e-3 and up to ~21%
    # for coarse primary windows (see PR #832), far above the floor here.
    function integrate_joint(marg, pe, y; n = 200_000)
        ld = latent(marg)
        lo, hi = minimum(pe), maximum(pe)
        ps = range(lo, hi; length = n)
        vals = map(
            p -> pdf(pe, p) * exp(logpdf(PrimaryConditional(ld, p), y)), ps)
        return sum((vals[1:(end - 1)] .+ vals[2:end]) ./ 2) * step(ps)
    end

    delay = LogNormal(1.0, 0.5)
    for (pe,
        lower,
        upper,
        ys) in [
        (Uniform(0, 1), 2.0, 6.0, 0.0:1.0:6.0),
        (Uniform(0, 3), 2.0, 8.0, 0.0:1.0:8.0)
    ]
        marg = double_interval_censored(delay; primary_event = pe,
            lower = lower, upper = upper, interval = 1)
        err = maximum(abs(integrate_joint(marg, pe, y) - pdf(marg, y))
        for y in ys)
        @test err < 1e-9
    end
end

@testitem "latent reaches through interval/truncation wrappers" begin
    using Distributions
    using CensoredDistributions: get_primary_event, get_dist, PrimaryConditional

    # A bare double_interval_censored wraps its PrimaryCensored in an
    # IntervalCensored (and Truncated when bounds are given); latent over it must
    # build, sample and score through the wrappers, keeping the secondary interval.
    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0, 1)

    # Interval-censored distribution.
    dic = double_interval_censored(delay; primary_event = pe, interval = 1)
    @test dic isa CensoredDistributions.IntervalCensored
    lic = latent(dic)
    @test get_primary_event(lic) === pe
    # The latent conditional keeps the secondary interval, so it differs from the
    # bare-delay shift.
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

    # Truncated + interval-censored dist: get_primary_event reaches through both.
    dtic = double_interval_censored(
        delay; primary_event = pe, upper = 10, interval = 1)
    ltic = latent(dtic)
    @test get_primary_event(ltic) === pe
    cond = PrimaryConditional(dtic, 0.3)
    @test logpdf(ltic, [0.3, 2.7]) ≈ logpdf(pe, 0.3) + logpdf(cond, 2.7)
end

@testitem "interval-of-latent conditional enforces secondary >= primary" begin
    using Distributions, Random, Statistics
    using CensoredDistributions: get_primary_event, marginal

    # The interval-censored conditional raises its lower bound to the primary
    # (secondary >= primary), so a primary inside a record's interval keeps
    # positive mass and a finite log density; a primary at or after the whole
    # interval is infeasible and stays `-Inf`.
    delay = LogNormal(1.5, 0.75)
    # A wide primary window (0, 3) lets a sampled primary fall inside or past the
    # unit interval containing y = 1 ([1, 2)).
    dic = double_interval_censored(delay; primary_event = Uniform(0, 3),
        upper = 12, interval = 1)
    ld = latent(dic)

    # Primary before the interval and primaries inside [1, 2): all finite, the
    # interior primaries kept in support by the clamp.
    for p in (0.5, 1.0, 1.5, 1.99)
        @test isfinite(logpdf(ld, 1.0; primary = p))
    end
    # Primary at or after the interval upper: genuinely infeasible, `-Inf`.
    @test logpdf(ld, 1.0; primary = 2.0) == -Inf
    @test logpdf(ld, 1.0; primary = 2.5) == -Inf

    # A record flooring to zero sits in `[0, 1)`, with the interval's lower edge
    # at the delay's support boundary; the below-support term is omitted so the
    # log density stays finite (gradient safety covered in the AD suite).
    for p in (0.0, 0.3, 0.7, 0.99)
        @test isfinite(logpdf(ld, 0.0; primary = p))
    end

    # The clamp is density-preserving: the conditional still matches the marginal
    # in expectation.
    pe = get_primary_event(ld)
    rng = MersenneTwister(20260702)
    ps = rand(rng, pe, 400_000)
    for y in (1.0, 3.0, 5.0)
        est = mean(pdf(ld, y; primary = p) for p in ps)
        @test isapprox(est, pdf(marginal(ld), y); atol = 5e-3)
    end
end
@testitem "batched latent logpdf sums the per-record conditional" begin
    using Distributions

    delay = LogNormal(1.5, 0.75)
    dist = latent(double_interval_censored(delay; primary_event = Uniform(0, 3),
        upper = 10, interval = 1))
    ys = [0.0, 1.0, 2.0, 3.0, 5.0, 0.0, 4.0]
    ps = [0.2, 0.4, 0.6, 0.8, 0.5, 0.9, 0.3]

    # The batched form equals the sum of the scalar conditional, for both the
    # pipeline and a bare primary-censored distribution.
    @test logpdf(dist, ys; primary = ps) ≈
          sum(logpdf(dist, ys[i]; primary = ps[i]) for i in eachindex(ys))
    bare = latent(primary_censored(delay, Uniform(0, 1)))
    yb = [2.5, 3.0, 4.0]
    pb = [0.2, 0.5, 0.3]
    @test logpdf(bare, yb; primary = pb) ≈
          sum(logpdf(bare, yb[i]; primary = pb[i]) for i in eachindex(yb))

    # A truncated-only pipeline dispatches to the `Truncated` methods; the batched
    # score still equals the per-record sum.
    tr = latent(truncated(primary_censored(delay, Uniform(0, 1)); upper = 10.0))
    yt = [2.5, 3.0, 4.0]
    pt = [0.2, 0.5, 0.3]
    @test logpdf(tr, yt; primary = pt) ≈
          sum(logpdf(tr, yt[i]; primary = pt[i]) for i in eachindex(yt))

    # Arbitrary interval boundaries route through the elementwise fallback; still
    # equal to the per-record sum.
    bnd = latent(interval_censored(
        truncated(primary_censored(delay, Uniform(0, 1)); upper = 10.0),
        [0.0, 1.0, 2.5, 5.0, 10.0]))
    yb2 = [1.5, 3.0, 0.5]
    pb2 = [0.2, 0.4, 0.1]
    @test logpdf(bnd, yb2; primary = pb2) ≈
          sum(logpdf(bnd, yb2[i]; primary = pb2[i]) for i in eachindex(yb2))

    # Unequal `ys`/`primary` lengths error rather than silently broadcasting.
    @test_throws DimensionMismatch logpdf(dist, [1.0, 2.0]; primary = [0.1])

    # An infeasible primary (at/after its whole interval) makes the batch `-Inf`.
    @test logpdf(dist, [0.0, 5.0]; primary = [2.5, 0.3]) == -Inf

    # Records flooring to zero score finitely (the sub-support term is omitted).
    @test isfinite(logpdf(dist, [0.0, 0.0, 0.0]; primary = [0.2, 0.5, 0.9]))

    # The joint `[primary, observed]` vector form is unaffected (no `primary`).
    @test isfinite(logpdf(dist, [0.4, 3.0]))
end

@testitem "secondary conditional rand throws when bounds exclude the delay" begin
    using Distributions, Random
    using CensoredDistributions: _SecondaryConditional

    # Truncation bounds that can never admit `p + delay` exhaust the bounded
    # resample and raise rather than looping forever.
    sc = _SecondaryConditional(Uniform(0.0, 1e-9), Uniform(0, 1),
        100.0, 101.0, 1.0, 0.0)
    @test_throws ErrorException rand(Random.default_rng(), sc)
end

@testitem "PrimaryEvent is the product of the per-record primary priors" begin
    using Distributions, Random
    using CensoredDistributions: get_primary_event

    # Heterogeneous per-record windows: the prior is the product of each
    # record's own primary event distribution.
    dists = [
        double_interval_censored(LogNormal(1.5, 0.75);
            primary_event = Uniform(0, 1), upper = 8, interval = 1),
        double_interval_censored(LogNormal(1.5, 0.75);
            primary_event = Uniform(0, 3), upper = 12, interval = 3)
    ]
    pe = PrimaryEvent(dists)
    @test length(pe) == 2
    ps = [0.4, 1.7]
    @test logpdf(pe, ps) ≈
          sum(logpdf(get_primary_event(dists[i]), ps[i]) for i in 1:2)
    # A draw sits inside each record's window.
    r = rand(Xoshiro(1), pe)
    @test length(r) == 2 && 0 <= r[1] <= 1 && 0 <= r[2] <= 3
    # A single-record vector still gives a valid product prior.
    @test length(PrimaryEvent(dists[1:1])) == 1
    # The scalar form extracts a single distribution's primary event.
    @test PrimaryEvent(dists[1]) == get_primary_event(dists[1])
    @test PrimaryEvent(dists[2]) == get_primary_event(dists[2])
end

@testitem "batched PrimaryConditional sums the per-record conditional" begin
    using Distributions, Random
    using CensoredDistributions: PrimaryConditional

    dists = [
        double_interval_censored(LogNormal(1.5, 0.75);
            primary_event = Uniform(0, 1), upper = 8, interval = 1),
        double_interval_censored(LogNormal(1.5, 0.75);
            primary_event = Uniform(0, 3), upper = 12, interval = 3),
        double_interval_censored(LogNormal(1.5, 0.75);
            primary_event = Uniform(0, 2), upper = 10, interval = 2)
    ]
    ys = [0.0, 6.0, 4.0]
    ps = [0.3, 1.2, 0.8]

    bc = PrimaryConditional(dists, ps)
    @test length(bc) == 3
    @test eltype(bc) == Float64
    # Batched multivariate logpdf equals the sum of the scalar conditionals.
    @test logpdf(bc, ys) ≈
          sum(logpdf(PrimaryConditional(dists[i], ps[i]), ys[i]) for i in 1:3)
    # `rand` draws one observed delay per record.
    r = rand(Xoshiro(3), bc)
    @test length(r) == 3 && all(isfinite, r)
    # The scalar kernel still scores a single record.
    @test isfinite(logpdf(PrimaryConditional(dists[1], 0.3), 3.0))
    # Unequal lengths error at construction and at scoring.
    @test_throws DimensionMismatch PrimaryConditional(dists, [0.1, 0.2])
    @test_throws DimensionMismatch logpdf(bc, [1.0, 2.0])
end

@testitem "primary-conditional fails loud on an unsupported inner dist" begin
    using Distributions
    using CensoredDistributions: PrimaryConditional

    # A wrapper that changes the delay density has no defined primary-conditional;
    # it raises an explanatory ArgumentError, not a MethodError.
    @test_throws ArgumentError logpdf(
        PrimaryConditional([LogNormal(1.5, 0.75)], [0.5]), [3.0])
end

@testitem "scalar PrimaryConditional exposes the Distributions interface" begin
    using Distributions, Random
    using CensoredDistributions: PrimaryConditional, get_dist

    delay = LogNormal(1.5, 0.75)
    # A bare primary-censored conditional is the delay shifted by the primary, so
    # it carries the full continuous interface (the pipeline form scores via
    # `logpdf`).
    sc = PrimaryConditional(primary_censored(delay, Uniform(0, 1)), 0.3)
    @test isfinite(logpdf(sc, 3.0))
    @test pdf(sc, 3.0) ≈ exp(logpdf(sc, 3.0))
    @test 0 <= cdf(sc, 3.0) <= 1
    @test logcdf(sc, 3.0) ≈ log(cdf(sc, 3.0)) atol = 1e-8
    @test ccdf(sc, 3.0) ≈ 1 - cdf(sc, 3.0)
    @test logccdf(sc, 3.0) ≈ log(ccdf(sc, 3.0)) atol = 1e-8
    @test quantile(sc, 0.5) > 0.3
    @test mean(sc) ≈ 0.3 + mean(delay)
    @test rand(Xoshiro(1), sc) > 0.3
    @test minimum(sc) ≈ 0.3
    @test maximum(sc) == Inf
    @test insupport(sc, 3.0)
    @test length(params(sc)) >= 1

    # The interval/truncation pipeline form scores via `logpdf`/`rand`; the
    # closed-form-only methods raise explicit errors, not bare MethodErrors.
    pc = PrimaryConditional(
        double_interval_censored(delay; primary_event = Uniform(0, 1),
            upper = 10, interval = 1), 0.4)
    @test isfinite(logpdf(pc, 3.0))
    @test rand(Xoshiro(1), pc) > 0.4
    @test_throws ArgumentError cdf(pc, 3.0)
    @test_throws ArgumentError quantile(pc, 0.5)
    @test_throws ArgumentError mean(pc)

    # `get_dist` reaches the bare continuous delay through the latent wrapper.
    @test get_dist(latent(primary_censored(delay, Uniform(0, 1)))) == delay
end

@testitem "batched PrimaryConditional is a full multivariate distribution" begin
    using Distributions, Random
    using CensoredDistributions: PrimaryConditional, PrimaryEvent

    dists = [
        double_interval_censored(LogNormal(1.5, 0.75);
            primary_event = Uniform(0, 1), upper = 8, interval = 1),
        double_interval_censored(LogNormal(1.5, 0.75);
            primary_event = Uniform(0, 3), upper = 12, interval = 3)]
    ps = rand(Xoshiro(1), PrimaryEvent(dists))
    pc = PrimaryConditional(dists, ps)

    # rand/logpdf/pdf/length/insupport all behave as a multivariate distribution.
    @test length(pc) == 2
    draw = rand(Xoshiro(2), pc)
    @test length(draw) == 2
    @test logpdf(pc, [2.0, 3.0]) isa Real
    @test pdf(pc, [2.0, 3.0]) ≈ exp(logpdf(pc, [2.0, 3.0]))
    @test insupport(pc, draw)
    # A wrong-length vector is out of support rather than an error.
    @test !insupport(pc, [2.0])
end

@testitem "latent scalar no-primary draw and rand cover the sampling paths" begin
    using Distributions, Random
    using CensoredDistributions: PrimaryConditional

    delay = LogNormal(1.5, 0.75)
    ld = latent(primary_censored(delay, Uniform(0, 1)))

    # With no `primary` each scalar method samples one internally, so it still
    # returns a finite/in-range value.
    @test isfinite(logpdf(ld, 3.0))
    @test isfinite(pdf(ld, 3.0))
    @test 0 <= cdf(ld, 3.0) <= 1
    @test isfinite(logcdf(ld, 3.0))
    @test 0 <= ccdf(ld, 3.0) <= 1
    @test isfinite(logccdf(ld, 3.0))
    @test isfinite(quantile(ld, 0.5))
    # The observed-only draw is just `rand` on the joint record or the conditional.
    @test rand(Xoshiro(1), ld).observed isa Real
    @test rand(PrimaryConditional(ld, 0.3)) isa Real

    # Joint and batched draws return labelled records.
    r = rand(Xoshiro(1), ld)
    @test r isa NamedTuple && Set(keys(r)) == Set((:primary, :observed))
    @test length(rand(Xoshiro(1), ld, 3)) == 3
    @test length(rand(ld, 2)) == 2

    # `marginal` inverts `latent`.
    @test marginal(ld) == primary_censored(delay, Uniform(0, 1))
    @test marginal(delay) == delay
end

@testitem "latent + conditional interface methods" begin
    using Distributions, Random
    using CensoredDistributions: PrimaryConditional, _pipeline_bounds

    delay = LogNormal(1.5, 0.75)
    dic = double_interval_censored(delay; primary_event = Uniform(0, 3),
        upper = 10, interval = 1)
    tr = truncated(primary_censored(delay, Uniform(0, 3)); upper = 10)
    bnd = interval_censored(
        truncated(primary_censored(delay, Uniform(0, 1)); upper = 10.0),
        [0.0, 1.0, 2.5, 5.0, 10.0])

    # `_SecondaryConditional` support: min/max and the interval `insupport`.
    pc = PrimaryConditional(dic, 0.4)
    @test minimum(pc) >= 0.4
    @test maximum(pc) <= 10
    @test insupport(pc, 3.0)
    @test !insupport(pc, 100.0)

    # The truncated-only (continuous) conditional: `insupport(::Nothing, ...)`
    # and the `_floor_observed(::Nothing, total)` draw.
    pct = PrimaryConditional(tr, 0.4)
    @test insupport(pct, 3.0)
    @test !insupport(pct, 100.0)
    @test rand(Xoshiro(1), pct) > 0.4

    # A scalar boundaries-interval logpdf hits `_interval_bounds(boundaries, y)`.
    @test isfinite(logpdf(latent(bnd), 3.0; primary = 0.2))

    # The generic `_pipeline_bounds` fallback for an unwrapped distribution.
    @test _pipeline_bounds(LogNormal(1.0, 1.0)) == (-Inf, Inf)

    # The batched `Truncated`-only method (continuous, no interval).
    @test logpdf(latent(tr), [2.0, 3.0]; primary = [0.3, 0.5]) isa Real

    # `Latent` distribution interface and the multi-draw `rand`.
    ld = latent(dic)
    @test length(ld) == 2
    @test eltype(typeof(ld)) <: Real
    @test length(params(ld)) >= 1
    @test length(rand(Xoshiro(2), ld, 3)) == 3
    @test length(rand(ld, 2)) == 2

    # `PrimaryConditional` element type and the no-primary Monte-Carlo draw
    # (`_latent_primary(rng, d, ::Nothing)`).
    @test eltype(typeof(pc)) <: Real
    @test isfinite(logpdf(ld, 3.0))
end

# The parameter-gradient marginal==latent equivalence needs `using ForwardDiff`,
# so it lives in the AD environment at `test/ad/latent_ad.jl`; the scalar and
# batched conditional gradients are also covered by the `Latent` scenarios in
# `test/ADFixtures`.
