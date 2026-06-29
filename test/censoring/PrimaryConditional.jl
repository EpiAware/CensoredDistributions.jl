@testitem "PrimaryConditional logpdf scores the delay at the implied gap" begin
    using Distributions

    delay = LogNormal(1.5, 0.75)
    d = primary_censored(delay, Uniform(0, 1))

    for (p, y) in [(0.3, 2.7), (0.0, 1.0), (0.5, 4.0)]
        pc = PrimaryConditional(d, p)
        @test pc isa UnivariateDistribution
        @test logpdf(pc, y) ≈ logpdf(delay, y - p)
        @test pdf(pc, y) ≈ pdf(delay, y - p)
        @test cdf(pc, y) ≈ cdf(delay, y - p)
    end
end

@testitem "PrimaryConditional support is y > p" begin
    using Distributions

    delay = LogNormal(1.5, 0.75)
    d = primary_censored(delay, Uniform(0, 1))
    p = 0.4
    pc = PrimaryConditional(d, p)

    @test minimum(pc) == p + minimum(delay)
    @test !insupport(pc, p - 0.1)        # below the primary
    @test insupport(pc, p + 0.5)
    @test logpdf(pc, p - 0.1) == -Inf
end

@testitem "PrimaryConditional rand produces p + delay (> p)" begin
    using Distributions, Random

    delay = LogNormal(1.5, 0.75)
    d = primary_censored(delay, Uniform(0, 1))
    p = 0.3
    pc = PrimaryConditional(d, p)

    rng = MersenneTwister(7)
    samples = rand(rng, pc, 1000)
    @test all(samples .> p)
    # rand is p plus an independent delay draw
    rng2 = MersenneTwister(11)
    @test rand(rng2, pc) ≈ p + rand(MersenneTwister(11), delay)
end

@testitem "PrimaryConditional keeps the secondary interval and truncation" begin
    using Distributions

    # For a `double_interval_censored` pipeline, the conditional given the primary
    # `p` keeps the secondary interval (and truncation) on the total `p + delay`
    # rather than stripping to the bare delay (#723). The interval mass over the
    # interval `[lo, hi)` containing `y` is `cdf(delay, hi - p) - cdf(delay,
    # lo - p)`, normalised by the truncation constant.
    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0, 1)
    p = 0.4

    dic = double_interval_censored(delay; primary_event = pe, interval = 1.0)
    pc = PrimaryConditional(dic, p)
    for y in [2.0, 4.0, 6.0]
        lo = floor(y)
        expected = log(cdf(delay, lo + 1 - p) - cdf(delay, lo - p))
        @test logpdf(pc, y) ≈ expected
        # Distinct from the bare-delay shift the separable leaf uses.
        @test logpdf(pc, y) != logpdf(delay, y - p)
    end

    # With an upper truncation the interval mass is divided by the global
    # truncation constant Z = cdf(primary_censored(delay, pe), upper).
    dtic = double_interval_censored(
        delay; primary_event = pe, upper = 10.0, interval = 1.0)
    pct = PrimaryConditional(dtic, p)
    Z = cdf(primary_censored(delay, pe), 10.0)
    for y in [2.0, 5.0, 9.0]
        lo = floor(y)
        hi = min(lo + 1, 10.0)
        expected = log(cdf(delay, hi - p) - cdf(delay, lo - p)) - log(Z)
        @test logpdf(pct, y) ≈ expected
    end

    # Truncated but NOT interval-censored: the conditional is the shifted delay
    # density inside the window, normalised by the same global Z.
    dtc = double_interval_censored(delay; primary_event = pe, upper = 10.0)
    pcc = PrimaryConditional(dtc, p)
    for y in [2.0, 5.0, 9.0]
        @test logpdf(pcc, y) ≈ logpdf(delay, y - p) - log(Z)
    end
    @test logpdf(pcc, 11.0) == -Inf   # above the truncation upper bound
end

@testitem "PrimaryConditional cdf: bare form only, pipeline scores via logpdf" begin
    using Distributions

    # The bare primary-censored conditional carries a real CDF (the shifted
    # delay). The interval/truncation pipeline form has no closed-form CDF and is
    # scored through `logpdf`, so `cdf` raises an explanatory error rather than a
    # bare `MethodError` (keeps the dispatch total; see #739 JET linting).
    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0, 1)
    p = 0.4

    bare = PrimaryConditional(primary_censored(delay, pe), p)
    @test cdf(bare, 2.7) ≈ cdf(delay, 2.7 - p)

    pipeline = PrimaryConditional(
        double_interval_censored(delay; primary_event = pe, interval = 1.0), p)
    @test_throws ArgumentError cdf(pipeline, 2.0)
end

@testitem "PrimaryConditional pipeline joint integrates to the marginal" begin
    using Distributions
    using CensoredDistributions: PrimaryConditional

    # Across the pipeline shapes, the joint (primary prior + conditional)
    # integrated over the primary must reproduce the analytic marginal density.
    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0, 1)

    function integrate_joint(marg, y; n = 150_000)
        ld = latent(marg)
        ps = range(0.0, 1.0; length = n)
        vals = map(p -> pdf(pe, p) * exp(logpdf(PrimaryConditional(ld, p), y)), ps)
        return sum((vals[1:(end - 1)] .+ vals[2:end]) ./ 2) * step(ps)
    end

    margs = [
        double_interval_censored(delay; primary_event = pe, interval = 1),
        double_interval_censored(
            delay; primary_event = pe, upper = 10, interval = 1),
        double_interval_censored(delay; primary_event = pe, upper = 10)
    ]
    for marg in margs
        for y in [1.0, 3.0, 6.0, 9.0]
            @test isapprox(integrate_joint(marg, y), pdf(marg, y); atol = 5e-4)
        end
    end
end

@testitem "PrimaryConditional works on a Latent wrapper" begin
    using Distributions

    delay = LogNormal(1.5, 0.75)
    d = primary_censored(delay, Uniform(0, 1))
    ld = latent(d)
    p, y = 0.3, 2.7

    # Delegates to the wrapped node, so the conditional is identical.
    @test logpdf(PrimaryConditional(ld, p), y) ≈ logpdf(PrimaryConditional(d, p), y)
    @test logpdf(PrimaryConditional(ld, p), y) ≈ logpdf(delay, y - p)
end

@testitem "Latent joint reuses PrimaryConditional (single source)" begin
    using Distributions, Random

    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0, 1)
    d = primary_censored(delay, pe)
    ld = latent(d)

    # The Latent joint is exactly the prior plus PrimaryConditional, with no
    # separate y - p logic.
    for (p, y) in [(0.3, 2.7), (0.1, 1.0), (0.9, 5.4)]
        @test logpdf(ld, [p, y]) ≈
              logpdf(pe, p) + logpdf(PrimaryConditional(d, p), y)
    end

    # rand draws the primary then the observed from PrimaryConditional.
    rng = MersenneTwister(3)
    x = rand(rng, ld)
    @test x[2] > x[1]
end
