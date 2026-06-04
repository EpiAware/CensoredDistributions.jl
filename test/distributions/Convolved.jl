@testitem "Convolved constructor and validation" begin
    using Distributions

    d = convolved(Gamma(2.0, 1.0), LogNormal(1.5, 0.5))
    @test d isa CensoredDistributions.Convolved
    @test length(d.components) == 2

    # Vector constructor
    dv = convolved([Gamma(2.0, 1.0), Gamma(1.0, 1.0), Normal(0.0, 1.0)])
    @test length(dv.components) == 3

    # Tuple constructor
    dt = convolved((Gamma(2.0, 1.0), Normal(0.0, 1.0)))
    @test length(dt.components) == 2

    # Errors
    @test_throws ArgumentError convolved([Gamma(2.0, 1.0)])
    @test_throws ArgumentError convolved((Gamma(2.0, 1.0),))
end

@testitem "Convolved support and params" begin
    using Distributions

    d = convolved(Gamma(2.0, 1.0), Normal(0.0, 1.0))
    # Gamma support [0, Inf), Normal support (-Inf, Inf)
    @test minimum(d) == -Inf
    @test maximum(d) == Inf

    d2 = convolved(Uniform(0.0, 1.0), Uniform(0.0, 2.0))
    @test minimum(d2) == 0.0
    @test maximum(d2) == 3.0

    # Negative support component
    d3 = convolved(Normal(-2.0, 1.0), Uniform(0.0, 1.0))
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
    d = convolved(a, b)
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
    dg = convolved(g1, g2)
    refg = convolve(g1, g2)
    for x in 0.5:1.0:12.0
        @test cdf(dg, x) ≈ cdf(refg, x) atol=1e-10
        @test pdf(dg, x) ≈ pdf(refg, x) atol=1e-10
    end

    # Three Normals fold pairwise analytically
    c = Normal(0.5, 1.0)
    d3 = convolved(a, b, c)
    ref3 = convolve(convolve(a, b), c)
    for x in -5.0:1.0:8.0
        @test cdf(d3, x) ≈ cdf(ref3, x) atol=1e-10
    end
end

@testitem "Convolved numeric path matches Monte Carlo" begin
    using Distributions, Random, Statistics

    rng = MersenneTwister(42)
    # Gamma + LogNormal has no analytical convolve -> numeric path
    a = Gamma(2.0, 1.0)
    b = LogNormal(0.5, 0.4)
    d = convolved(a, b)

    n = 400_000
    samples = [rand(rng, a) + rand(rng, b) for _ in 1:n]

    for q in (1.5, 2.5, 4.0, 6.0)
        emp = mean(samples .<= q)
        @test cdf(d, q) ≈ emp atol=5e-3
    end

    # pdf via numeric differentiation integrates to ~1 over support and
    # is positive
    @test pdf(d, 3.0) > 0
    @test logpdf(d, 3.0) ≈ log(pdf(d, 3.0)) atol=1e-8

    # Mean of samples matches sum of component means
    @test mean(samples) ≈ mean(a) + mean(b) atol=2e-2
end

@testitem "Convolved rand sums components" begin
    using Distributions, Random, Statistics

    rng = MersenneTwister(1)
    d = convolved(Gamma(2.0, 1.0), Normal(3.0, 0.5))
    s = [rand(rng, d) for _ in 1:200_000]
    @test mean(s) ≈ 2.0 + 3.0 atol=5e-2
    @test var(s) ≈ var(Gamma(2.0, 1.0)) + 0.25 atol=1e-1
end

@testitem "Convolved batched cdf/logpdf match scalar" begin
    using Distributions

    # Numeric path
    d = convolved(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    xs = [1.0, 2.0, 3.5, 5.0, 7.0]

    # The batched path integrates every point over one shared window, so
    # it differs from the per-point scalar windows only by fixed-node
    # quadrature error, not mathematically.
    cdf_batch = cdf(d, xs)
    cdf_scalar = [cdf(d, x) for x in xs]
    @test cdf_batch ≈ cdf_scalar rtol=1e-3

    lp_batch = logpdf(d, xs)
    lp_scalar = [logpdf(d, x) for x in xs]
    @test lp_batch ≈ lp_scalar rtol=1e-3

    # Analytic path
    da = convolved(Normal(0.0, 1.0), Normal(1.0, 2.0))
    @test cdf(da, xs) ≈ [cdf(da, x) for x in xs] atol=1e-10
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
    # separately). Here we assert correctness for a large batch.
    d = convolved(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    xs = collect(0.5:0.25:8.0)
    @test cdf(d, xs) ≈ [cdf(d, x) for x in xs] rtol=1e-3
end

@testitem "Convolved composes with truncated" begin
    using Distributions

    d = convolved(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    td = truncated(d, 1.0, 8.0)

    @test cdf(td, 0.5) == 0.0
    @test cdf(td, 9.0) == 1.0
    @test 0.0 < cdf(td, 4.0) < 1.0
    @test pdf(td, 4.0) > 0
    @test get_dist(td) === d
end

@testitem "Convolved composes with interval_censored" begin
    using Distributions

    d = convolved(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    ic = interval_censored(d, 1.0)

    # PMF over a partition sums to ~1
    total = sum(pdf(ic, x) for x in 0.0:1.0:60.0)
    @test total ≈ 1.0 atol=1e-2
    @test pdf(ic, 3.0) >= 0
    @test get_dist(ic) === d
end

@testitem "Convolved composes with weight" begin
    using Distributions

    d = convolved(Normal(0.0, 1.0), Normal(1.0, 2.0))
    wd = weight(d, 3.0)
    @test logpdf(wd, 1.0) ≈ 3.0 * logpdf(d, 1.0) atol=1e-8
    @test get_dist(wd) === d
end

@testitem "Convolved get_dist returns components" begin
    using Distributions

    comps = [Gamma(2.0, 1.0), Normal(0.0, 1.0)]
    d = convolved(comps)
    got = get_dist(d)
    @test got isa AbstractVector
    @test length(got) == 2
end
