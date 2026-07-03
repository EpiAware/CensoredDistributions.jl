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

    # `latent(d)` scalar density is ALWAYS conditional on a primary, never the
    # integrating marginal. With a primary passed it is the deterministic
    # `PrimaryConditional(d, p)` kernel.
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
        # The bare-leaf conditional is the delay shifted by `p`, no quadrature.
        @test logpdf(ld, y; primary = p) ≈ logpdf(delay, y - p)
    end

    # The conditional is NOT the marginal pointwise (the project invariant: the
    # latent form never collapses to the integrating default).
    @test logpdf(ld, 2.7; primary = 0.3) != logpdf(marginal(ld), 2.7)
end

@testitem "latent scalar density: MC-samples the primary when absent" begin
    using Distributions, Random
    using CensoredDistributions: get_primary_event

    # With no primary passed, a single scalar density call MONTE-CARLO SAMPLES one
    # `p ~ get_primary_event(d)` and conditions on it: a stochastic single-draw
    # estimate, never the integrating marginal.
    ld = latent(primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1)))

    # Reproducible under a seeded rng; different seeds give different draws, the
    # signature of a genuinely sampled (stochastic) primary.
    a = logpdf(ld, 2.7; rng = MersenneTwister(1))
    @test a == logpdf(ld, 2.7; rng = MersenneTwister(1))
    @test a != logpdf(ld, 2.7; rng = MersenneTwister(2))
    @test isfinite(a)
end

@testitem "latent scalar density: equivalence in expectation only" begin
    using Distributions, Random
    using CensoredDistributions: get_primary_event, marginal

    # E over sampled primaries of the conditional density equals the MARGINAL
    # density (equivalence IN EXPECTATION), even though no single conditional call
    # equals the marginal pointwise and nothing integrates by quadrature.
    delay = LogNormal(1.0, 0.5)
    pe = Uniform(0, 1)
    d = primary_censored(delay, pe)
    ld = latent(d)

    rng = MersenneTwister(20260630)
    for y in (1.5, 3.0, 5.0)
        # Monte-Carlo mean of the conditional density over sampled primaries.
        est = mean(pdf(ld, y; primary = rand(rng, pe)) for _ in 1:200_000)
        @test isapprox(est, pdf(marginal(ld), y); rtol = 2e-2)
        # No single conditional draw equals the marginal (it is a point, not the
        # integral): the equivalence is only in expectation.
        @test pdf(ld, y; primary = 0.5) != pdf(marginal(ld), y)
    end
end

@testitem "latent scalar quantile and conditional draw" begin
    using Distributions, Random
    using CensoredDistributions: PrimaryConditional, rand_observed

    delay = LogNormal(1.2, 0.4)
    pe = Uniform(0, 1)
    d = primary_censored(delay, pe)
    ld = latent(d)

    # Quantile conditional on a passed primary equals the kernel's quantile, the
    # delay quantile shifted by `p` (closed form for the bare leaf).
    p = 0.4
    @test quantile(ld, 0.5; primary = p) ==
          quantile(PrimaryConditional(d, p), 0.5)
    @test quantile(ld, 0.5; primary = p) ≈ p + quantile(delay, 0.5)

    # A conditional observed draw given `p` lands above `p` (observed = p + delay).
    r = rand_observed(MersenneTwister(1), ld; primary = p)
    @test r > p
    # Reproducible under a seeded rng.
    @test rand_observed(MersenneTwister(3), ld; primary = p) ==
          rand_observed(MersenneTwister(3), ld; primary = p)
end

@testitem "PrimaryConditional scores the delay at the implied gap" begin
    using Distributions
    using CensoredDistributions: PrimaryConditional

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

@testitem "latent conditional equals the marginal in expectation" begin
    using Distributions, Random, Statistics
    using CensoredDistributions: PrimaryConditional, get_primary_event

    # The latent form conditions on a sampled primary (`PrimaryConditional` at
    # a `rand`-drawn `p`); it does not integrate the primary out and defines no
    # scalar observed density. So it matches the marginal only in expectation
    # over the primary prior,
    #   E_p[cdf(PrimaryConditional(d, p), x)] = cdf(marginal(d), x),
    # estimated here by Monte Carlo over genuine `rand` draws of the primary,
    # never by the latent itself integrating. This is the equivalence-in-
    # expectation that underwrites the latent and marginal fits recovering the
    # same parameters.
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
        # A single draw is genuinely conditional, not the marginal: one
        # realised primary gives a shifted conditional, not the marginal cdf.
        one_p = rand(rng, get_primary_event(ld))
        @test cdf(PrimaryConditional(ld, one_p), 2.5) != cdf(dm, 2.5)
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

@testitem "interval-of-latent conditional enforces secondary >= primary" begin
    using Distributions, Random, Statistics
    using CensoredDistributions: get_primary_event, marginal

    # The interval-censored secondary conditional raises its lower integration
    # bound to the primary (secondary >= primary), so a primary landing INSIDE a
    # record's observed interval keeps positive mass and a finite log density
    # rather than a stray `-Inf`; a primary at or after the whole interval is
    # genuinely infeasible (a negative delay) and stays `-Inf`. This is the
    # feasibility invariant the latent double_interval_censored fit relies on.
    delay = LogNormal(1.5, 0.75)
    # A wide primary window (0, 3) so a sampled primary can fall inside or past
    # a unit interval, exercising the clamp; the record y = 1 sits in [1, 2).
    dic = double_interval_censored(delay; primary_event = Uniform(0, 3),
        upper = 12, interval = 1)
    ld = latent(dic)

    # Primary before the interval (normal), and primaries INSIDE [1, 2): all
    # finite (feasible), the interior primaries kept in support by the clamp.
    for p in (0.5, 1.0, 1.5, 1.99)
        @test isfinite(logpdf(ld, 1.0; primary = p))
    end
    # Primary at or after the interval upper: genuinely infeasible, `-Inf`.
    @test logpdf(ld, 1.0; primary = 2.0) == -Inf
    @test logpdf(ld, 1.0; primary = 2.5) == -Inf

    # A record whose observed delay floors to zero sits in `[0, 1)`, putting the
    # interval's lower edge at the delay's support boundary. The below-support
    # term contributes no mass and is omitted rather than evaluated, so the log
    # density stays finite (the gradient safety of this is covered in the AD
    # suite).
    for p in (0.0, 0.3, 0.7, 0.99)
        @test isfinite(logpdf(ld, 0.0; primary = p))
    end

    # The clamp is density-preserving: the latent conditional still matches the
    # analytic marginal in expectation over the primary prior.
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
    leaf = latent(double_interval_censored(delay; primary_event = Uniform(0, 3),
        upper = 10, interval = 1))
    ys = [0.0, 1.0, 2.0, 3.0, 5.0, 0.0, 4.0]
    ps = [0.2, 0.4, 0.6, 0.8, 0.5, 0.9, 0.3]

    # The batched vector form equals the sum of the scalar conditional, for both
    # the interval/truncation pipeline leaf and a bare primary-censored leaf.
    @test logpdf(leaf, ys; primary = ps) ≈
          sum(logpdf(leaf, ys[i]; primary = ps[i]) for i in eachindex(ys))
    bare = latent(primary_censored(delay, Uniform(0, 1)))
    yb = [2.5, 3.0, 4.0]
    pb = [0.2, 0.5, 0.3]
    @test logpdf(bare, yb; primary = pb) ≈
          sum(logpdf(bare, yb[i]; primary = pb[i]) for i in eachindex(yb))

    # Unequal `ys`/`primary` lengths error rather than silently broadcasting.
    @test_throws DimensionMismatch logpdf(leaf, [1.0, 2.0]; primary = [0.1])

    # An infeasible primary (at/after its whole interval) makes the batch `-Inf`.
    @test logpdf(leaf, [0.0, 5.0]; primary = [2.5, 0.3]) == -Inf

    # Records whose observed delay floors to zero score finitely (the
    # sub-support term is omitted, see the guard above).
    @test isfinite(logpdf(leaf, [0.0, 0.0, 0.0]; primary = [0.2, 0.5, 0.9]))

    # The joint `[primary, observed]` vector form is unaffected (no `primary`).
    @test isfinite(logpdf(leaf, [0.4, 3.0]))
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

@testitem "primary-conditional fails loud on an unsupported inner node" begin
    using Distributions
    using CensoredDistributions: PrimaryConditional

    # A wrapper that changes the delay density (here a bare distribution) has no
    # defined primary-conditional; it must raise an explanatory ArgumentError,
    # not a MethodError and not a silently-wrong finite value.
    @test_throws ArgumentError logpdf(
        PrimaryConditional([LogNormal(1.5, 0.75)], [0.5]), [3.0])
end

@testitem "scalar PrimaryConditional exposes the Distributions interface" begin
    using Distributions, Random
    using CensoredDistributions: PrimaryConditional, get_dist

    delay = LogNormal(1.5, 0.75)
    # A bare primary-censored leaf's conditional is the delay shifted by the
    # primary, so it carries the full continuous cdf/quantile/rand interface (the
    # interval/truncation pipeline form instead scores via `logpdf`).
    sc = PrimaryConditional(primary_censored(delay, Uniform(0, 1)), 0.3)
    @test isfinite(logpdf(sc, 3.0))
    @test pdf(sc, 3.0) ≈ exp(logpdf(sc, 3.0))
    @test 0 <= cdf(sc, 3.0) <= 1
    @test logcdf(sc, 3.0) ≈ log(cdf(sc, 3.0)) atol = 1e-8
    @test ccdf(sc, 3.0) ≈ 1 - cdf(sc, 3.0)
    @test logccdf(sc, 3.0) ≈ log(ccdf(sc, 3.0)) atol = 1e-8
    @test quantile(sc, 0.5) > 0.3
    @test rand(Xoshiro(1), sc) > 0.3
    @test minimum(sc) ≈ 0.3
    @test maximum(sc) == Inf
    @test insupport(sc, 3.0)
    @test length(params(sc)) >= 1

    # The interval/truncation pipeline form scores via `logpdf`; `cdf` is an
    # explicit error rather than a bare MethodError.
    pc = PrimaryConditional(
        double_interval_censored(delay; primary_event = Uniform(0, 1),
            upper = 10, interval = 1), 0.4)
    @test isfinite(logpdf(pc, 3.0))
    @test_throws ArgumentError cdf(pc, 3.0)

    # `get_dist` reaches the bare continuous delay through the latent wrapper.
    @test get_dist(latent(primary_censored(delay, Uniform(0, 1)))) == delay
end

@testitem "latent scalar no-primary draw and rand cover the sampling paths" begin
    using Distributions, Random
    using CensoredDistributions: rand_observed

    delay = LogNormal(1.5, 0.75)
    ld = latent(primary_censored(delay, Uniform(0, 1)))

    # With no `primary` passed each scalar method samples one internally (a
    # stochastic single-draw estimate, not the marginal), so it still returns a
    # finite/in-range value.
    @test isfinite(logpdf(ld, 3.0))
    @test isfinite(pdf(ld, 3.0))
    @test 0 <= cdf(ld, 3.0) <= 1
    @test isfinite(logcdf(ld, 3.0))
    @test 0 <= ccdf(ld, 3.0) <= 1
    @test isfinite(logccdf(ld, 3.0))
    @test isfinite(quantile(ld, 0.5))
    @test rand_observed(Xoshiro(1), ld) isa Real
    @test rand_observed(ld; primary = 0.3) isa Real

    # Joint and batched draws return labelled records.
    r = rand(Xoshiro(1), ld)
    @test r isa NamedTuple && Set(keys(r)) == Set((:primary, :observed))
    @test length(rand(Xoshiro(1), ld, 3)) == 3
    @test length(rand(ld, 2)) == 2

    # `marginal` inverts `latent`.
    @test marginal(ld) == primary_censored(delay, Uniform(0, 1))
    @test marginal(delay) == delay
end

# The parameter-gradient marginal==latent equivalence (`using ForwardDiff`) is
# an AD test, so it lives in the AD environment at `test/ad/latent_ad.jl` (the
# main test env deliberately does not depend on ForwardDiff). The latent scalar
# and batched conditional gradients are additionally covered across the full
# backend matrix by the `Latent ...` scenarios in `test/ADFixtures`.
