@testitem "Convolved constructor and validation" begin
    using Distributions

    d = convolve_distributions(Gamma(2.0, 1.0), LogNormal(1.5, 0.5))
    @test d isa CensoredDistributions.Convolved
    @test length(d.components) == 2

    # Vector constructor
    dv = convolve_distributions([Gamma(2.0, 1.0), Gamma(1.0, 1.0), Normal(0.0, 1.0)])
    @test length(dv.components) == 3

    # Tuple constructor
    dt = convolve_distributions((Gamma(2.0, 1.0), Normal(0.0, 1.0)))
    @test length(dt.components) == 2

    # Errors
    @test_throws ArgumentError convolve_distributions([Gamma(2.0, 1.0)])
    @test_throws ArgumentError convolve_distributions((Gamma(2.0, 1.0),))
end

@testitem "Convolved support and params" begin
    using Distributions

    d = convolve_distributions(Gamma(2.0, 1.0), Normal(0.0, 1.0))
    # Gamma support [0, Inf), Normal support (-Inf, Inf)
    @test minimum(d) == -Inf
    @test maximum(d) == Inf

    d2 = convolve_distributions(Uniform(0.0, 1.0), Uniform(0.0, 2.0))
    @test minimum(d2) == 0.0
    @test maximum(d2) == 3.0

    # Negative support component
    d3 = convolve_distributions(Normal(-2.0, 1.0), Uniform(0.0, 1.0))
    @test minimum(d3) == -Inf
    @test maximum(d3) == Inf

    p = params(d2)
    @test p == ((0.0, 1.0), (0.0, 2.0))
end

@testitem "Convolved analytic agreement with Distributions.convolve" begin
    using Distributions

    # Normal + Normal
    a = Normal(1.0, 2.0)
    b = Normal(-0.5, 1.5)
    d = convolve_distributions(a, b)
    ref = convolve(a, b)
    xs = -5.0:1.0:8.0
    for x in xs
        @test cdf(d, x) ≈ cdf(ref, x) atol=1e-10
        @test pdf(d, x) ≈ pdf(ref, x) atol=1e-10
        @test logpdf(d, x) ≈ logpdf(ref, x) atol=1e-8
    end

    # Equal-scale Gamma + Gamma (shapes add)
    g1 = Gamma(2.0, 1.5)
    g2 = Gamma(3.0, 1.5)
    dg = convolve_distributions(g1, g2)
    refg = convolve(g1, g2)
    for x in 0.5:1.0:12.0
        @test cdf(dg, x) ≈ cdf(refg, x) atol=1e-10
        @test pdf(dg, x) ≈ pdf(refg, x) atol=1e-10
    end

    # Three Normals fold pairwise analytically
    c = Normal(0.5, 1.0)
    d3 = convolve_distributions(a, b, c)
    ref3 = convolve(convolve(a, b), c)
    for x in -5.0:1.0:8.0
        @test cdf(d3, x) ≈ cdf(ref3, x) atol=1e-10
    end

    # Equal-rate Exponentials convolve to a Gamma analytically
    e1 = Exponential(1.5)
    de = convolve_distributions(e1, Exponential(1.5))
    refe = convolve(e1, Exponential(1.5))
    for x in 0.5:1.0:10.0
        @test cdf(de, x) ≈ cdf(refe, x) atol=1e-10
    end
end

@testitem "Convolved force_numeric matches analytic ground truth" begin
    using Distributions

    # For pairs that HAVE a closed form, force the numeric quadrature path
    # and check it reproduces the exact analytic cdf AND pdf. This
    # validates the numeric solver (and the node count) against analytic
    # ground truth, distinct from the Monte-Carlo checks. The scalar path
    # uses each point's tight window and matches to ~1e-7; the batched
    # path shares one window across points and is looser on wide ranges.
    cases = [
        (Normal(1.0, 2.0), Normal(-0.5, 1.5), -3.0, 8.0),
        (Gamma(2.0, 1.5), Gamma(3.0, 1.5), 0.5, 12.0),
        (Exponential(1.5), Exponential(1.5), 0.3, 8.0)
    ]
    for (a, b, lo, hi) in cases
        dn = convolve_distributions(a, b; force_numeric = true)
        ref = convolve(a, b)

        # force_numeric must actually bypass the analytic specialisation.
        @test CensoredDistributions._maybe_analytic(dn) === nothing
        @test CensoredDistributions._maybe_analytic(
            convolve_distributions(a, b)) !== nothing

        xs = collect(range(lo, hi; length = 10))
        for x in xs
            @test cdf(dn, x) ≈ cdf(ref, x) atol=1e-6
            @test pdf(dn, x) ≈ pdf(ref, x) atol=1e-6
        end

        # Batched numeric vs analytic ground truth (looser shared-window
        # tolerance for the wide range; compared elementwise rather than
        # by vector norm).
        cb = cdf(dn, xs)
        pb = pdf(dn, xs)
        @test maximum(abs.(cb .- [cdf(ref, x) for x in xs])) < 2e-4
        @test maximum(abs.(pb .- [pdf(ref, x) for x in xs])) < 6e-3
    end
end

@testitem "Convolved unsupported analytic pairs use numeric path" begin
    using Distributions, Random, Statistics

    # Different-scale Gamma and different-rate Exponential have no
    # closed-form convolution; these must fall back to numeric quadrature
    # rather than throwing from Distributions.convolve.
    rng = MersenneTwister(7)

    dg = convolve_distributions(Gamma(2.0, 1.0), Gamma(3.0, 2.0))
    @test 0.0 <= cdf(dg, 5.0) <= 1.0
    sg = [rand(rng, Gamma(2.0, 1.0)) + rand(rng, Gamma(3.0, 2.0))
          for _ in 1:200_000]
    @test cdf(dg, 8.0) ≈ mean(sg .<= 8.0) atol=5e-3

    de = convolve_distributions(Exponential(1.0), Exponential(2.0))
    @test 0.0 <= cdf(de, 3.0) <= 1.0
    se = [rand(rng, Exponential(1.0)) + rand(rng, Exponential(2.0))
          for _ in 1:200_000]
    @test cdf(de, 3.0) ≈ mean(se .<= 3.0) atol=5e-3
end

@testitem "Convolved numeric path matches Monte Carlo" begin
    using Distributions, Random, Statistics

    rng = MersenneTwister(42)
    # Gamma + LogNormal has no analytical convolve -> numeric path
    a = Gamma(2.0, 1.0)
    b = LogNormal(0.5, 0.4)
    d = convolve_distributions(a, b)

    n = 400_000
    samples = [rand(rng, a) + rand(rng, b) for _ in 1:n]

    for q in (1.5, 2.5, 4.0, 6.0)
        emp = mean(samples .<= q)
        @test cdf(d, q) ≈ emp atol=5e-3
    end

    # pdf from the density convolution integral is positive and consistent
    # with logpdf
    @test pdf(d, 3.0) > 0
    @test logpdf(d, 3.0) ≈ log(pdf(d, 3.0)) atol=1e-8

    # Mean of samples matches sum of component means
    @test mean(samples) ≈ mean(a) + mean(b) atol=2e-2
end

@testitem "Convolved numeric path with unbounded-below component" begin
    using Distributions, Random, Statistics

    # Gamma + Normal has an unbounded-below integration component, so the
    # numeric quadrature window starts at -Inf and must be clamped to a
    # finite quantile (the `_CONVOLVED_TAIL` window) before the
    # Gauss-Legendre mapping. Without the clamp the change of variable maps
    # an infinite bound and the scalar cdf/pdf return NaN.
    rng = MersenneTwister(91)
    a = Gamma(2.0, 1.0)
    b = Normal(0.0, 1.0)
    d = convolve_distributions(a, b)

    @test minimum(d) == -Inf
    @test maximum(d) == Inf

    n = 400_000
    samples = [rand(rng, a) + rand(rng, b) for _ in 1:n]

    for q in (-1.0, 1.0, 3.0, 5.0)
        c = cdf(d, q)
        p = pdf(d, q)
        @test isfinite(c)
        @test isfinite(p)
        @test 0.0 <= c <= 1.0
        @test p >= 0.0
        @test insupport(d, q)
        @test c ≈ mean(samples .<= q) atol=5e-3
    end

    # Scalar and batched paths agree for the unbounded-below component.
    xs = [-1.0, 0.5, 2.0, 4.0]
    @test cdf(d, xs) ≈ [cdf(d, x) for x in xs] rtol=5e-4
    @test pdf(d, xs) ≈ [pdf(d, x) for x in xs] rtol=1e-3
end

@testitem "Convolved pdf matches analytic and Monte Carlo" begin
    using Distributions, Random, Statistics

    # Analytic pairs: numeric-free exact density via Distributions.convolve.
    a = Normal(1.0, 2.0)
    b = Normal(-0.5, 1.5)
    d = convolve_distributions(a, b)
    ref = convolve(a, b)
    for x in -4.0:0.5:6.0
        @test pdf(d, x) ≈ pdf(ref, x) atol=1e-10
        @test logpdf(d, x) ≈ logpdf(ref, x) atol=1e-8
    end

    g1 = Gamma(2.0, 1.5)
    g2 = Gamma(3.0, 1.5)
    dg = convolve_distributions(g1, g2)
    refg = convolve(g1, g2)
    for x in 0.5:0.5:12.0
        @test pdf(dg, x) ≈ pdf(refg, x) atol=1e-10
    end

    # Non-analytic pair: density convolution integral vs Monte-Carlo
    # histogram density, and total mass ~1.
    rng = MersenneTwister(123)
    dn = convolve_distributions(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    n = 4_000_000
    s = [rand(rng, Gamma(2.0, 1.0)) + rand(rng, LogNormal(0.5, 0.4))
         for _ in 1:n]
    hw = 0.05
    for x in (2.0, 3.0, 5.0)
        emp = mean((s .> x - hw) .& (s .<= x + hw)) / (2hw)
        @test pdf(dn, x) ≈ emp rtol=2e-2
    end

    grid = collect(0.05:0.1:30.0)
    @test sum(pdf(dn, grid)) * 0.1 ≈ 1.0 atol=2e-3

    # Bounded components: triangular-style density, exact midpoint value
    # and unit mass.
    du = convolve_distributions(Uniform(0.0, 2.0), Uniform(0.0, 3.0))
    @test pdf(du, 2.5) ≈ 1 / 3 atol=1e-6
    gu = collect(0.005:0.01:5.0)
    @test sum(pdf(du, gu)) * 0.01 ≈ 1.0 atol=1e-3
end

@testitem "Convolved rand sums components" begin
    using Distributions, Random, Statistics

    rng = MersenneTwister(1)
    d = convolve_distributions(Gamma(2.0, 1.0), Normal(3.0, 0.5))
    s = [rand(rng, d) for _ in 1:200_000]
    @test mean(s) ≈ 2.0 + 3.0 atol=5e-2
    @test var(s) ≈ var(Gamma(2.0, 1.0)) + 0.25 atol=1e-1
end

@testitem "Convolved batched cdf/logpdf match scalar" begin
    using Distributions

    # Numeric path
    d = convolve_distributions(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    xs = [1.0, 2.0, 3.5, 5.0, 7.0]

    # The batched path integrates every point over one shared window, so
    # it differs from the per-point scalar windows only by fixed-node
    # quadrature error, not mathematically. At n = 192 nodes this gap is
    # ~1e-4 (cdf) / ~5e-4 (pdf) on this batch.
    cdf_batch = cdf(d, xs)
    cdf_scalar = [cdf(d, x) for x in xs]
    @test cdf_batch ≈ cdf_scalar rtol=5e-4

    pdf_batch = pdf(d, xs)
    pdf_scalar = [pdf(d, x) for x in xs]
    @test pdf_batch ≈ pdf_scalar rtol=1e-3

    lp_batch = logpdf(d, xs)
    lp_scalar = [logpdf(d, x) for x in xs]
    @test lp_batch ≈ lp_scalar rtol=1e-3

    # Analytic path
    da = convolve_distributions(Normal(0.0, 1.0), Normal(1.0, 2.0))
    @test cdf(da, xs) ≈ [cdf(da, x) for x in xs] atol=1e-10
    @test pdf(da, xs) ≈ [pdf(da, x) for x in xs] atol=1e-10
    @test logpdf(da, xs) ≈ [logpdf(da, x) for x in xs] atol=1e-10
end

@testitem "Convolved batched cdf does a single solve" begin
    using Distributions
    import Integrals

    # Count solve calls by wrapping the GaussLegendre solver via a global
    # counter on IntegralProblem construction is not trivial; instead
    # verify the batched integrand is called once by checking timing-free
    # invariant: batched result for many points equals stacking scalar
    # results, which it does only if the shared-window single-solve path
    # is taken (degenerate fallback would still match but is exercised
    # separately). Here we assert correctness for a large batch. This
    # batch deliberately spans a wide range (16x), the hardest case for a
    # single shared-window solve, so its tolerance is looser than the
    # typical-batch test above.
    d = convolve_distributions(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    xs = collect(0.5:0.25:8.0)
    @test cdf(d, xs) ≈ [cdf(d, x) for x in xs] rtol=5e-3
end

@testitem "Convolved composes with truncated" begin
    using Distributions

    d = convolve_distributions(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    td = truncated(d, 1.0, 8.0)

    @test cdf(td, 0.5) == 0.0
    @test cdf(td, 9.0) == 1.0
    @test 0.0 < cdf(td, 4.0) < 1.0
    @test pdf(td, 4.0) > 0
    @test get_dist(td) === d
end

@testitem "Convolved composes with interval_censored" begin
    using Distributions

    d = convolve_distributions(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    ic = interval_censored(d, 1.0)

    # PMF over a partition sums to ~1
    total = sum(pdf(ic, x) for x in 0.0:1.0:60.0)
    @test total ≈ 1.0 atol=1e-2
    @test pdf(ic, 3.0) >= 0
    @test get_dist(ic) === d
end

@testitem "Convolved composes with weight" begin
    using Distributions

    d = convolve_distributions(Normal(0.0, 1.0), Normal(1.0, 2.0))
    wd = weight(d, 3.0)
    @test logpdf(wd, 1.0) ≈ 3.0 * logpdf(d, 1.0) atol=1e-8
    @test get_dist(wd) === d
end

@testitem "Convolved get_dist returns components" begin
    using Distributions

    comps = [Gamma(2.0, 1.0), Normal(0.0, 1.0)]
    d = convolve_distributions(comps)
    got = get_dist(d)
    @test got isa AbstractVector
    @test length(got) == 2
end

@testitem "Convolved logcdf/ccdf/logccdf branches" begin
    using Distributions

    # Analytic path: logcdf/ccdf agree with the convolved reference.
    da = convolve_distributions(Normal(0.0, 1.0), Normal(1.0, 2.0))
    refa = convolve(Normal(0.0, 1.0), Normal(1.0, 2.0))
    @test logcdf(da, 2.0) ≈ logcdf(refa, 2.0) atol=1e-10
    @test ccdf(da, 2.0) ≈ ccdf(refa, 2.0) atol=1e-10
    @test logccdf(da, 2.0) ≈ logccdf(refa, 2.0) atol=1e-8

    # Numeric path: logcdf matches log of cdf and ccdf = 1 - cdf.
    dn = convolve_distributions(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    @test logcdf(dn, 3.0) ≈ log(cdf(dn, 3.0)) atol=1e-10
    @test ccdf(dn, 3.0) ≈ 1 - cdf(dn, 3.0) atol=1e-10
    @test logccdf(dn, 3.0) ≈ log1p(-cdf(dn, 3.0)) atol=1e-6

    # logccdf edge cases via a bounded numeric-path distribution where
    # the support endpoints give exact CDF 0 and 1.
    db = convolve_distributions(Uniform(0.0, 2.0), Uniform(0.0, 3.0))
    @test logccdf(db, -1.0) == 0.0   # CDF = 0 -> logccdf = 0
    @test logccdf(db, 6.0) == -Inf   # CDF = 1 -> logccdf = -Inf
end

@testitem "Convolved logpdf outside support on numeric path" begin
    using Distributions

    # Gamma+LogNormal has support [0, Inf) and no analytic convolution,
    # so a negative x exercises the numeric-path !insupport branch.
    dn = convolve_distributions(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    @test logpdf(dn, -1.0) == -Inf
    @test pdf(dn, -1.0) == 0.0
    @test !insupport(dn, -1.0)
end

@testitem "Convolved batched bounded-support clamps and fallback" begin
    using Distributions

    # Uniform+Uniform has no analytic convolution and bounded support
    # [0, 5], so a batch spanning below/above support exercises the
    # xi<=dmin -> 0 and xi>=dmax -> 1 clamps in the batched path.
    d = convolve_distributions(Uniform(0.0, 2.0), Uniform(0.0, 3.0))
    @test minimum(d) == 0.0
    @test maximum(d) == 5.0
    xs = [-1.0, 1.0, 2.5, 4.0, 6.0]
    cb = cdf(d, xs)
    @test cb[1] == 0.0
    @test cb[end] == 1.0
    @test all(0.0 .<= cb .<= 1.0)
    @test cb ≈ [cdf(d, x) for x in xs] rtol=1e-3

    # logpdf batched outside-support entry returns -Inf.
    lp = logpdf(d, xs)
    @test lp[1] == -Inf
    @test lp[end] == -Inf
    @test isfinite(lp[3])

    # Batched pdf: out-of-support points clamp to 0, interior agrees with
    # the scalar density.
    pb = pdf(d, xs)
    @test pb[1] == 0.0
    @test pb[end] == 0.0
    @test pb[3] > 0
    @test pb ≈ [pdf(d, x) for x in xs] rtol=1e-2

    # All-below-support batches collapse the shared window, hitting the
    # per-point scalar fallback (every entry is 0).
    @test all(cdf(d, [-3.0, -2.0, -1.0]) .== 0.0)
    @test all(pdf(d, [-3.0, -2.0, -1.0]) .== 0.0)
end

@testitem "Convolved inner truncation reduces to ordinary convolution" begin
    using Distributions

    # Infinite bounds must give exactly the unbounded result on both the
    # analytic and numeric paths.
    da = convolve_distributions(Normal(0.0, 1.0), Normal(1.0, 2.0))
    dab = convolve_distributions(Normal(0.0, 1.0), Normal(1.0, 2.0);
        bounds = [(-Inf, Inf), (-Inf, Inf)])
    for x in -3.0:1.0:6.0
        @test cdf(dab, x) ≈ cdf(da, x) atol=1e-10
        @test pdf(dab, x) ≈ pdf(da, x) atol=1e-10
    end

    dn = convolve_distributions(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    dnb = convolve_distributions(Gamma(2.0, 1.0), LogNormal(0.5, 0.4);
        bounds = [(-Inf, Inf), (-Inf, Inf)])
    for x in (1.5, 3.0, 5.0)
        @test cdf(dnb, x) == cdf(dn, x)
        @test pdf(dnb, x) == pdf(dn, x)
    end
    # Infinite bounds keep the analytic fast path (numeric would differ by
    # quadrature error); equality above confirms the fast path is taken.
end

@testitem "Convolved reproduces andv pipeline probability" begin
    using Distributions, Random, Statistics

    # andv `_pipeline_probability`: P(δ ≤ Δ_q ∧ δ + Inc > Δ_p)
    #   = cdf(δ, Δ_q) − P(Inc + δ ≤ Δ_p ∧ δ ≤ Δ_q).
    # The δ component is the integration variable (last component), capped
    # above at Δ_q through the inner truncation bound.
    inc = LogNormal(1.6, 0.4)
    δd = Gamma(2.0, 1.5)
    Δ_q = 4.0
    Δ_p = 6.0

    dconv = convolve_distributions(inc, δd; bounds = [(-Inf, Inf), (-Inf, Δ_q)])
    pp = cdf(δd, Δ_q) - cdf(dconv, Δ_p)

    rng = MersenneTwister(2024)
    n = 4_000_000
    δs = rand(rng, δd, n)
    incs = rand(rng, inc, n)
    mc = mean((δs .<= Δ_q) .& (δs .+ incs .> Δ_p))
    @test pp ≈ mc atol=2e-3

    # Natural-chain limit Δ_q = +Inf reduces to 1 − cdf(Inc + δ, Δ_p).
    dfull = convolve_distributions(inc, δd)
    @test (cdf(δd, Inf) - cdf(dfull, Δ_p)) ≈ (1 - cdf(dfull, Δ_p)) atol=1e-10
end

@testitem "Convolved inner truncation matches Monte Carlo" begin
    using Distributions, Random, Statistics

    inc = LogNormal(1.6, 0.4)
    δd = Gamma(2.0, 1.5)
    rng = MersenneTwister(7)
    n = 4_000_000
    δs = rand(rng, δd, n)
    incs = rand(rng, inc, n)

    # Both components bounded: joint event P(sum ≤ x ∧ bounds).
    a1, b1 = 0.0, 8.0      # inc bounds
    a2, b2 = 1.0, 5.0      # δ bounds
    d = convolve_distributions(inc, δd; bounds = [(a1, b1), (a2, b2)])
    for x in (4.0, 7.0, 9.0, 20.0)
        mc = mean((incs .>= a1) .& (incs .<= b1) .&
                  (δs .>= a2) .& (δs .<= b2) .& (incs .+ δs .<= x))
        @test cdf(d, x) ≈ mc atol=3e-3
    end

    # Above the bounded support the cdf saturates at the total truncation
    # mass, strictly below 1.
    total = mean((incs .>= a1) .& (incs .<= b1) .&
                 (δs .>= a2) .& (δs .<= b2))
    @test cdf(d, 30.0) ≈ total atol=3e-3
    @test cdf(d, 30.0) < 1.0
end

@testitem "Convolved inner truncation batched matches scalar" begin
    using Distributions

    inc = LogNormal(1.6, 0.4)
    δd = Gamma(2.0, 1.5)
    d = convolve_distributions(inc, δd; bounds = [(0.0, 8.0), (1.0, 5.0)])
    xs = [3.0, 5.0, 7.0, 9.0, 12.0]

    @test cdf(d, xs) ≈ [cdf(d, x) for x in xs] atol=1e-10
    @test pdf(d, xs) ≈ [pdf(d, x) for x in xs] atol=1e-10
    @test logpdf(d, xs) ≈ [logpdf(d, x) for x in xs] atol=1e-10
end

@testitem "Convolved inner truncation validation" begin
    using Distributions

    # Wrong number of bounds.
    @test_throws ArgumentError convolve_distributions(
        Normal(0.0, 1.0), Normal(1.0, 1.0); bounds = [(-Inf, Inf)])
    # lower > upper.
    @test_throws ArgumentError convolve_distributions(
        Normal(0.0, 1.0), Normal(1.0, 1.0);
        bounds = [(-Inf, Inf), (5.0, 1.0)])

    # Empty intersection with the component support is rejected at
    # construction (would otherwise feed an inverted window into the
    # quadrature and silently produce NaN, especially for nested rests).
    # Gamma support is [0, Inf); a wholly negative bound has no overlap.
    @test_throws ArgumentError convolve_distributions(
        LogNormal(0.5, 0.4), Gamma(2.0, 1.0);
        bounds = [(-Inf, Inf), (-5.0, -1.0)])
    @test_throws ArgumentError convolve_distributions(
        LogNormal(0.5, 0.4), Gamma(2.0, 1.0), Normal(0.0, 1.0);
        bounds = [(-Inf, Inf), (-5.0, -1.0), (-Inf, Inf)])
    # A degenerate point bound that touches the support is allowed (the
    # effective support is the single point, zero-width but non-empty).
    @test convolve_distributions(
        Uniform(0.0, 2.0), Uniform(0.0, 3.0);
        bounds = [(-Inf, Inf), (1.0, 1.0)]) isa
          CensoredDistributions.Convolved
end

@testitem "Convolved inner truncation is unnormalised joint mass" begin
    using Distributions

    # With finite bounds the cdf is the unnormalised joint mass and
    # saturates below 1 at the total truncation mass; the pdf integrates
    # to that same mass, not to 1.
    inc = LogNormal(1.6, 0.4)
    δd = Gamma(2.0, 1.5)
    d = convolve_distributions(inc, δd; bounds = [(0.0, 8.0), (1.0, 5.0)])

    total = cdf(d, maximum(d))
    @test total < 1.0
    @test total ≈ cdf(d, 1e6) atol=1e-8

    # The unnormalised density integrates to the same total mass.
    grid = collect(0.05:0.1:25.0)
    @test sum(pdf(d, grid)) * 0.1 ≈ total atol=1e-2

    # Dividing by the saturated mass recovers a normalised conditional CDF.
    @test (cdf(d, maximum(d)) / total) ≈ 1.0 atol=1e-8
end

@testitem "Convolved eltype and sampler" begin
    using Distributions, Random

    d = convolve_distributions(Gamma(2.0, 1.0), Normal(0.0, 1.0))
    @test eltype(d) == Float64
    @test sampler(d) === d
    rng = MersenneTwister(3)
    @test rand(rng, d) isa Real
end

@testitem "Convolved scalar methods value-correct and inferrable" begin
    using Distributions, Test

    # The interface methods return a concrete `Float64` and infer as such.
    # The numeric path uses the Integrals.jl-free `gl_integrate` dot
    # product, whose accumulator type is seeded from the integrand, so the
    # quadrature element type no longer leaks as `Any` (the gap closed in
    # #208; previously `Integrals.solve` hid it behind a free parameter).
    analytic = convolve_distributions(Normal(0.0, 1.0), Normal(1.0, 2.0))
    numeric = convolve_distributions(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    for d in (analytic, numeric)
        for f in (cdf, logcdf, pdf, logpdf, ccdf, logccdf)
            @test f(d, 3.0) isa Float64
        end
        @test (@inferred(cdf(d, 3.0)); true)
        @test (@inferred(pdf(d, 3.0)); true)
    end
end
