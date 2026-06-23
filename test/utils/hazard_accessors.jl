# Tests for the tree-level hazard accessors (`hazard` / `loghazard` /
# `cumhazard` / `survival`): the hazard surface read off a composed delay
# through the verbs (north-star tenet 5). The reporting-hazard PMF layer lives
# in `test/utils/hazard.jl`; this file covers the per-node accessors and their
# consistency with `modify` and SurvivalDistributions.

@testitem "hazard accessors: leaf matches the survival identities" begin
    using CensoredDistributions, Distributions
    const CD = CensoredDistributions

    for base in (LogNormal(1.5, 0.5), Gamma(2.0, 1.3), Weibull(1.7, 2.0))
        for t in (0.5, 2.0, 5.0)
            S = ccdf(base, t)
            @test CD.survival(base, t) ≈ S
            @test CD.cumhazard(base, t) ≈ -log(S)
            @test CD.hazard(base, t) ≈ pdf(base, t) / S
            @test CD.loghazard(base, t) ≈ logpdf(base, t) - logccdf(base, t)
            # The four are mutually consistent.
            @test CD.hazard(base, t) ≈ exp(CD.loghazard(base, t))
            @test CD.cumhazard(base, t) ≈ -log(CD.survival(base, t))
        end
    end
end

@testitem "hazard accessors: Sequential = marginal convolution hazard" begin
    using CensoredDistributions, Distributions
    const CD = CensoredDistributions

    s1 = Gamma(2.0, 1.0)
    s2 = LogNormal(0.5, 0.4)
    seq = sequential(s1, s2)
    # The chain's total-time marginal is the convolution of the steps; its
    # hazard is the convolution's hazard, hand-computed from the same Convolved.
    conv = convolve_distributions(s1, s2)
    for t in (1.0, 3.0, 6.0)
        @test CD.hazard(seq, t) ≈ pdf(conv, t) / ccdf(conv, t) rtol=1e-6
        @test CD.cumhazard(seq, t) ≈ -logccdf(conv, t) rtol=1e-6
        @test CD.survival(seq, t) ≈ ccdf(conv, t) rtol=1e-6
        lh = logpdf(conv, t) - logccdf(conv, t)
        @test CD.loghazard(seq, t) ≈ lh rtol=1e-6
    end

    # A single-step chain is just that step's hazard.
    one_step = sequential(s1)
    for t in (1.0, 3.0)
        @test CD.hazard(one_step, t) ≈ pdf(s1, t) / ccdf(s1, t) rtol=1e-6
    end

    # A nested chain step contributes its own marginal time-to-event.
    nested = sequential(sequential(s1, Gamma(1.5, 1.0)), s2)
    cnest = convolve_distributions(
        convolve_distributions(s1, Gamma(1.5, 1.0)), s2)
    for t in (1.0, 3.0, 6.0)
        @test CD.hazard(nested, t) ≈ pdf(cnest, t) / ccdf(cnest, t) rtol=1e-5
    end
end

@testitem "hazard accessors: Compete is the racing (summed) hazard" begin
    using CensoredDistributions, Distributions
    const CD = CensoredDistributions

    # The minimum of independent exponentials has constant total hazard equal to
    # the sum of the component rates (the racing-hazard identity).
    c = compete(:a => Exponential(2.0), :b => Exponential(3.0))
    hand = 1 / 2.0 + 1 / 3.0
    for t in (0.5, 2.0, 4.0)
        @test CD.hazard(c, t) ≈ hand rtol=1e-6
        @test CD.cumhazard(c, t) ≈ hand * t rtol=1e-6
    end
end

@testitem "hazard accessors: Parallel hazard is ambiguous and errors" begin
    using CensoredDistributions, Distributions
    const CD = CensoredDistributions

    p = parallel(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    @test_throws ArgumentError CD.hazard(p, 2.0)
    @test_throws ArgumentError CD.cumhazard(p, 2.0)
    @test_throws ArgumentError CD.survival(p, 2.0)
    @test_throws ArgumentError CD.loghazard(p, 2.0)
end

@testitem "hazard accessors: consistent with modify (all links)" begin
    using CensoredDistributions, Distributions
    const CD = CensoredDistributions

    base = LogNormal(1.5, 0.5)
    h(t) = CD.hazard(base, t)

    # Proportional hazards (log link): h*(t) = θ h(t), θ = exp(β).
    β = log(2.0)
    mlog = modify(base, β; link = log)
    for t in (1.0, 3.0, 6.0)
        @test CD.hazard(mlog, t) ≈ exp(β) * h(t) rtol=1e-6
    end

    # Additive hazards (identity link), positive effect: h*(t) = h(t) + β.
    βa = 0.3
    mid = modify(base, βa; link = identity)
    for t in (1.0, 3.0, 6.0)
        @test CD.hazard(mid, t) ≈ h(t) + βa rtol=1e-5
    end

    # General-link reconstruction: hazard(modify(...)) = g⁻¹(g(h(t)) + effect)
    # for a user callable link (here cloglog), the modify construction itself.
    cloglog = CensoredDistributions.hazard_link(
        u -> log(-log1p(-u)), η -> -expm1(-exp(η)))
    βc = 0.4
    mcl = modify(base, βc; link = cloglog)
    for t in (1.0, 3.0)
        expected = cloglog.invlink(cloglog.g(h(t)) + βc)
        @test CD.hazard(mcl, t) ≈ expected rtol=1e-4
    end

    # A `Modified` STEP inside a chain keeps its hazard modification: the chain
    # hazard is the convolution of the modified step and the other step, NOT the
    # stripped base (which `_marginal_core` would give).
    seq = sequential(mlog, Gamma(1.5, 1.0))
    conv_mod = convolve_distributions(mlog, Gamma(1.5, 1.0))
    conv_base = convolve_distributions(base, Gamma(1.5, 1.0))
    for t in (1.0, 3.0, 6.0)
        @test CD.hazard(seq, t) ≈ pdf(conv_mod, t) / ccdf(conv_mod, t) rtol=1e-5
        @test !isapprox(CD.hazard(seq, t),
            pdf(conv_base, t) / ccdf(conv_base, t); rtol = 1e-3)
    end
end

@testitem "hazard accessors: log-space stable in the tail" begin
    using CensoredDistributions, Distributions
    const CD = CensoredDistributions

    # Deep in the tail S(t) → 0, so a naive `log(hazard)` / `-log(survival)`
    # would lose precision or overflow; the log-space accessors stay finite.
    base = LogNormal(0.5, 0.4)
    t = 50.0
    @test isfinite(CD.loghazard(base, t))
    @test isfinite(CD.cumhazard(base, t))
    @test CD.cumhazard(base, t) > 0
end
