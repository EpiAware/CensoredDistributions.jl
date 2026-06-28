@testitem "modify constructor and link options" begin
    using Distributions

    base = LogNormal(1.5, 0.5)

    # The bare functions and the symbols both resolve to the named links.
    @test modify(base, -0.3; link = log).link === LogLink
    @test modify(base, -0.3; link = :log).link === LogLink
    @test modify(base, 0.2; link = identity).link === IdentityLink
    @test modify(base, 0.2; link = :identity).link === IdentityLink
    @test modify(base, 0.2; link = :logit).link === LogitLink
    @test modify(base, 0.2; link = LogitLink).link === LogitLink

    # Default link is log (proportional hazards).
    @test modify(base, -0.3).link === LogLink

    # A Modified is a UnivariateDistribution that carries its base.
    d = modify(base, -0.3)
    @test d isa Modified
    @test get_dist(d) === base
    @test minimum(d) == minimum(base)
    @test maximum(d) == maximum(base)

    # Unknown symbol errors.
    @test_throws ArgumentError modify(base, 0.2; link = :probit)

    # params surface the base params then the scalar effect; a callable
    # effect carries no numeric param.
    @test params(modify(base, -0.3)) == (params(base)..., -0.3)
    @test params(modify(base, t -> 0.1 * t)) == params(base)
end

@testitem "modify continuous log link: analytic identities" begin
    using Distributions

    base = LogNormal(1.5, 0.5)
    β = -0.4
    θ = exp(β)
    d = modify(base, β; link = log)

    for x in [0.5, 2.0, 5.0, 10.0]
        # S* = S^exp(β).
        @test ccdf(d, x) ≈ ccdf(base, x)^θ
        @test logccdf(d, x) ≈ θ * logccdf(base, x)
        # logpdf* = β + logpdf + (exp(β) - 1) logccdf.
        @test logpdf(d, x) ≈ β + logpdf(base, x) +
                             (θ - 1) * logccdf(base, x)
        # cdf / ccdf are complementary.
        @test cdf(d, x) + ccdf(d, x) ≈ 1
    end

    # β = 0 is the identity modification.
    d0 = modify(base, 0.0; link = log)
    for x in [0.5, 2.0, 5.0]
        @test logpdf(d0, x) ≈ logpdf(base, x)
        @test ccdf(d0, x) ≈ ccdf(base, x)
    end
end

@testitem "modify continuous identity link: analytic identities" begin
    using Distributions

    base = LogNormal(1.2, 0.6)
    β = 0.15
    d = modify(base, β; link = identity)

    for x in [0.5, 2.0, 5.0]
        # S*(t) = S(t) exp(-β t).
        @test logccdf(d, x) ≈ logccdf(base, x) - β * x
        @test ccdf(d, x) ≈ ccdf(base, x) * exp(-β * x)
        @test cdf(d, x) + ccdf(d, x) ≈ 1
    end
end

@testitem "modify identity link stays a valid CDF under negative effects" begin
    using Distributions
    using ForwardDiff

    # An additive-hazard model is only valid where h(t) + β >= 0. For a
    # negative β on a base with h(0) = 0 (LogNormal, Gamma shape > 1, ...) the
    # raw hazard goes negative near the origin, so the modified hazard is
    # clamped to max(h + β, 0): logpdf, logccdf and cdf must all use that same
    # clamped hazard and stay mutually consistent (issue #670).
    bases = (LogNormal(1.5, 0.5), Gamma(2.0, 1.5), Weibull(2.0, 3.0))
    βs = (-0.1, -0.4, 0.0, 0.2, 0.5)
    grid = collect(0.0:0.25:12.0)

    for base in bases, β in βs

        d = modify(base, β; link = identity)

        cdfs = cdf.(Ref(d), grid)
        # CDF is in [0, 1] and monotone non-decreasing on [0, ∞).
        @test all(c -> -1e-10 <= c <= 1 + 1e-10, cdfs)
        @test all(diff(cdfs) .>= -1e-10)

        # cdf / ccdf are complementary and pdf is non-negative.
        for x in grid
            @test cdf(d, x) + ccdf(d, x) ≈ 1
            @test pdf(d, x) >= -1e-12
        end

        # logpdf is consistent with the cdf: where the density is positive the
        # log-derivative of the cdf matches, and where the hazard is clamped to
        # zero the density is exactly zero.
        for x in [0.5, 1.5, 3.0, 6.0]
            lp = logpdf(d, x)
            if isfinite(lp)
                fd = ForwardDiff.derivative(t -> cdf(d, t), x)
                # The negative-effect path integrates a kinked (clamped)
                # hazard, so the Gauss-Legendre cdf derivative matches the
                # density only to quadrature tolerance, not machine precision.
                @test exp(lp) ≈ max(fd, 0.0) atol = 1e-2
            else
                @test pdf(d, x) ≈ 0 atol = 1e-12
            end
        end

        # AD through the effect is finite (ForwardDiff over logpdf and cdf).
        g_lp = ForwardDiff.derivative(b -> logpdf(modify(base, b;
                    link = identity), 3.0), β)
        g_cdf = ForwardDiff.derivative(b -> cdf(modify(base, b;
                    link = identity), 3.0), β)
        @test isfinite(g_lp)
        @test isfinite(g_cdf)
    end

    # Pin the previously-broken case from issue #670: a non-monotone, negative
    # CDF. It must now be monotone and non-negative.
    md = modify(LogNormal(1.5, 0.5), -0.1; link = identity)
    @test cdf(md, 0.0) >= -1e-12
    @test cdf(md, 2.7) >= cdf(md, 0.0) - 1e-12
    @test cdf(md, 2.7) >= -1e-12
    @test cdf(md, 5.5) >= cdf(md, 2.7) - 1e-12
end

@testitem "modify analytic == numeric agreement" begin
    using Distributions

    base = LogNormal(1.5, 0.5)

    # A genuinely distinct closure for log/exp forces the numeric quadrature
    # path; it must match the closed form to quadrature tolerance.
    β = -0.4
    dlog = modify(base, β; link = log)
    dlog_num = modify(base, β; link = hazard_link(h -> log(h), η -> exp(η)))
    for x in [0.5, 2.0, 5.0, 10.0]
        @test logccdf(dlog, x) ≈ logccdf(dlog_num, x) atol = 1e-8
        @test logpdf(dlog, x) ≈ logpdf(dlog_num, x) atol = 1e-8
    end

    β2 = 0.15
    did = modify(base, β2; link = identity)
    did_num = modify(base, β2; link = hazard_link(h -> 1.0 * h, η -> 1.0 * η))
    for x in [0.5, 2.0, 5.0]
        @test logccdf(did, x) ≈ logccdf(did_num, x) atol = 1e-8
        @test logpdf(did, x) ≈ logpdf(did_num, x) atol = 1e-8
    end
end

@testitem "modify continuous is a proper density" begin
    using CensoredDistributions: gl_integrate, _GL
    using Distributions

    rule = _GL(400)
    for d in (
        modify(LogNormal(1.5, 0.5), -0.4; link = log),
        modify(LogNormal(1.2, 0.6), 0.15; link = identity),
        modify(Gamma(2.0, 1.5), 0.3; link = :logit)
    )
        total = gl_integrate(x -> pdf(d, x), 1e-6, 200.0, rule)
        @test total ≈ 1.0 atol = 1e-4
    end
end

@testitem "modify time-varying effect on numeric path" begin
    using Distributions

    base = LogNormal(1.5, 0.5)
    # A callable effect routes through the numeric solver and gives a proper,
    # finite density; a constant callable matches the scalar logit modify.
    dc = modify(base, t -> 0.3; link = :logit)
    ds = modify(base, 0.3; link = :logit)
    for x in [0.5, 2.0, 5.0]
        @test logpdf(dc, x) ≈ logpdf(ds, x) atol = 1e-10
    end
end

@testitem "modify discrete per-bin reconstruction" begin
    using CensoredDistributions: apply_hazard_effects
    using Distributions

    ic = interval_censored(LogNormal(1.5, 0.5), 1.0)
    n = 11
    base_pmf = map(g -> pdf(ic, Float64(g)), 0:(n - 1))
    effects = collect(range(-0.5, 0.5; length = n))

    # The logit-link discrete path is exactly `apply_hazard_effects` lifted
    # onto a distribution.
    m = modify(ic, effects; link = :logit)
    mpmf = [pdf(m, Float64(b)) for b in 0:(n - 1)]
    ref = apply_hazard_effects(base_pmf, effects)
    @test mpmf ≈ ref
    @test sum(mpmf) ≈ 1.0

    # The PMF reconstructs to a proper distribution for any link.
    for link in (:logit, log, identity, hazard_link(h -> h, η -> η))
        mm = modify(ic, effects; link = link)
        @test sum(pdf(mm, Float64(b)) for b in 0:(n - 1)) ≈ 1.0

        # cdf / ccdf are complementary and cdf accumulates the PMF.
        @test cdf(mm, 4.0) + ccdf(mm, 4.0) ≈ 1.0
        @test cdf(mm, 4.0) ≈ sum(pdf(mm, Float64(b)) for b in 0:4)
    end

    # Zero effect reconstructs the baseline hazard with the final-bin
    # maximum-delay constraint (the last bin absorbs the remaining mass so the
    # PMF sums to one); the first n-1 bins match the raw interval masses.
    m0 = modify(ic, zeros(n); link = :logit)
    m0pmf = [pdf(m0, Float64(b)) for b in 0:(n - 1)]
    @test m0pmf[1:(n - 1)] ≈ base_pmf[1:(n - 1)]
    @test sum(m0pmf) ≈ 1.0
    @test m0pmf ≈ apply_hazard_effects(base_pmf, zeros(n))

    # rand returns grid points and roughly tracks the PMF.
    using Random
    rng = MersenneTwister(1)
    samples = [rand(rng, m) for _ in 1:20000]
    @test all(s -> s in 0.0:1.0:Float64(n - 1), samples)
    @test count(==(0.0), samples) / 20000 ≈ mpmf[1] atol = 5e-3
end

@testitem "modify composes as a leaf" begin
    using Distributions

    m = modify(Gamma(2.0, 1.5), -0.3; link = log)
    m2 = modify(LogNormal(1.0, 0.5), 0.2; link = identity)

    # Nests in compose: the stack names its events and exposes the inner
    # free params of each Modified leaf through the introspection layer. A plain
    # (uncensored) stack keys its record on the edge names, so `event_names`
    # equals `keys(rand(stack))` (one per edge, no latent origin).
    stack = compose(m; tail = m2)
    @test event_names(stack) == keys(rand(stack))
    tbl = params_table(stack)
    @test tbl !== nothing

    # Nests in the censoring wrappers and truncation helpers.
    icm = interval_censored(m, 1.0)
    @test pdf(icm, 2.0) > 0

    pc = primary_censored(m, Uniform(0, 1))
    @test 0 < cdf(pc, 3.0) < 1

    tm = truncated(m; upper = 10.0)
    @test isfinite(logpdf(tm, 2.0))

    # Transparent to its inner free delay.
    @test CensoredDistributions.free_leaf(m) === Gamma(2.0, 1.5)
end

@testitem "modify continuous rand tracks the modified law" begin
    using Random
    using Distributions

    rng = MersenneTwister(42)

    # Proportional hazards: the empirical CDF matches the analytic CDF.
    d = modify(LogNormal(1.5, 0.5), -0.4; link = log)
    samples = [rand(rng, d) for _ in 1:40000]
    for x in [2.0, 5.0, 10.0]
        @test count(<=(x), samples) / 40000 ≈ cdf(d, x) atol = 1e-2
    end
end
