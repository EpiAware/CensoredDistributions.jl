@testitem "ParallelDistribution constructor and validation" begin
    using Distributions

    d = parallel_distribution(
        Uniform(0.0, 1.0), Gamma(2.0, 1.0), LogNormal(1.0, 0.5))
    @test d isa CensoredDistributions.ParallelDistribution
    @test length(d) == 2
    @test d.primary_event == Uniform(0.0, 1.0)
    @test all(b -> b == (-Inf, Inf), d.child_bounds)
    @test d.log_norm == 0.0

    # Single child is allowed.
    d1 = parallel_distribution(Uniform(0.0, 1.0), Gamma(2.0, 1.0))
    @test length(d1) == 1

    # child_bounds must match the number of delays.
    @test_throws ArgumentError parallel_distribution(
        Uniform(0.0, 1.0), Gamma(2.0, 1.0), LogNormal(1.0, 0.5);
        child_bounds = [(0.0, 5.0)])

    # Each bound must be ordered.
    @test_throws ArgumentError parallel_distribution(
        Uniform(0.0, 1.0), Gamma(2.0, 1.0);
        child_bounds = [(5.0, 1.0)])
end

@testitem "ParallelDistribution primary_prior and params" begin
    using Distributions

    pe = Uniform(0.0, 2.0)
    d = parallel_distribution(pe, Gamma(2.0, 1.0), LogNormal(1.0, 0.5))
    @test primary_prior(d) === pe
    p = params(d)
    @test p[1] == params(pe)
    @test p[2] == params(Gamma(2.0, 1.0))
    @test p[3] == params(LogNormal(1.0, 0.5))
end

@testitem "ParallelDistribution n=1 reduces to primary_censored" begin
    using Distributions

    # With one child and infinite bounds the marginal of the single child
    # must equal primary_censored(delay, primary_event) numerics for both
    # the analytical (Gamma/Uniform) and numeric primary-censored paths.
    pe = Uniform(0.0, 1.0)
    for delay in (Gamma(2.0, 1.0), LogNormal(1.0, 0.5), Weibull(2.0, 1.5))
        pd = parallel_distribution(pe, delay)
        pc = primary_censored(delay, pe)
        for x in [0.5, 1.5, 2.5, 4.0, 6.0]
            # cdf is the same integral in both constructions.
            @test cdf(pd, [x]) ≈ cdf(pc, x) atol=1e-9
            # logcdf agrees.
            @test logcdf(pd, [x]) ≈ logcdf(pc, x) atol=1e-8
        end
    end
end

@testitem "ParallelDistribution n=1 logpdf integrates to 1" begin
    using Distributions

    # The single-child marginal density must integrate to 1.
    pe = Uniform(0.0, 1.0)
    pd = parallel_distribution(pe, Gamma(2.0, 1.0))
    xs = range(0.001, 20.0; length = 4000)
    dx = step(xs)
    total = sum(x -> exp(logpdf(pd, [x])), xs) * dx
    @test total ≈ 1.0 atol=1e-3
end

@testitem "ParallelDistribution cdf matches Monte Carlo (n=2)" begin
    using Distributions, Random, Statistics

    # Validate the joint cdf (the 1-D shared-origin integral) against a
    # Monte-Carlo estimate of P(Y1 ≤ x1 ∧ Y2 ≤ x2) where Y_i = O + D_i and
    # O is a single shared draw. N = 4_000_000 gives a Monte-Carlo standard
    # error well under the 3e-3 tolerance.
    rng = MersenneTwister(20240601)
    pe = Uniform(0.0, 1.0)
    d1 = Gamma(2.0, 1.0)
    d2 = LogNormal(1.0, 0.5)
    pd = parallel_distribution(pe, d1, d2)

    N = 4_000_000
    o = rand(rng, pe, N)
    y1 = o .+ rand(rng, d1, N)
    y2 = o .+ rand(rng, d2, N)

    for x in ([2.0, 3.0], [1.0, 2.0], [4.0, 5.0], [3.0, 1.5])
        mc = mean((y1 .<= x[1]) .& (y2 .<= x[2]))
        @test cdf(pd, x) ≈ mc atol=3e-3
    end
end

@testitem "ParallelDistribution logpdf matches Monte Carlo (n=2)" begin
    using Distributions, Random, Statistics

    # Validate the joint density against a Monte-Carlo estimate of the joint
    # density on a small grid box: P(Y ∈ box) / area ≈ f(centre). Box mass
    # is estimated from N = 6_000_000 shared-origin draws.
    rng = MersenneTwister(11)
    pe = Uniform(0.0, 1.0)
    d1 = Gamma(2.0, 1.0)
    d2 = LogNormal(1.0, 0.5)
    pd = parallel_distribution(pe, d1, d2)

    N = 6_000_000
    o = rand(rng, pe, N)
    y1 = o .+ rand(rng, d1, N)
    y2 = o .+ rand(rng, d2, N)

    h = 0.1
    for c in ([2.0, 3.0], [1.5, 2.0])
        in_box = (abs.(y1 .- c[1]) .<= h / 2) .& (abs.(y2 .- c[2]) .<= h / 2)
        mc_density = mean(in_box) / h^2
        @test exp(logpdf(pd, c)) ≈ mc_density rtol=5e-2
    end
end

@testitem "ParallelDistribution joint density integrates to 1 (n=3)" begin
    using Distributions, Random, Statistics

    # n=3 branches: validate normalisation by 3-D Monte-Carlo integration
    # (the joint density integrates to 1 over the children), and that the
    # joint cdf at a point matches the shared-origin generative process.
    rng = MersenneTwister(7)
    pe = Uniform(0.0, 1.0)
    delays = (Gamma(2.0, 1.0), LogNormal(1.0, 0.5), Gamma(1.5, 1.5))
    pd = parallel_distribution(pe, delays...)

    # cdf vs Monte Carlo for a 3-vector.
    N = 4_000_000
    o = rand(rng, pe, N)
    ys = [o .+ rand(rng, dl, N) for dl in delays]
    x = [3.0, 3.0, 4.0]
    mc = mean((ys[1] .<= x[1]) .& (ys[2] .<= x[2]) .& (ys[3] .<= x[3]))
    @test cdf(pd, x) ≈ mc atol=3e-3

    # Monte-Carlo importance estimate of ∫ f = 1: sample children from an
    # independent proposal and average f / q.
    Nimp = 2_000_000
    q1 = Gamma(3.0, 1.5)
    q2 = LogNormal(1.2, 0.7)
    q3 = Gamma(2.5, 2.0)
    s1 = rand(rng, q1, Nimp)
    s2 = rand(rng, q2, Nimp)
    s3 = rand(rng, q3, Nimp)
    mean_w = mean(1:Nimp) do i
        logq = logpdf(q1, s1[i]) + logpdf(q2, s2[i]) + logpdf(q3, s3[i])
        exp(logpdf(pd, [s1[i], s2[i], s3[i]]) - logq)
    end
    @test mean_w ≈ 1.0 atol=2e-2
end

@testitem "ParallelDistribution bounded children normalisation" begin
    using Distributions, Random, Statistics

    # With finite child bounds the log-normalisation constant must equal
    # the closed product Π_i (F_{D_i}(b_i) - F_{D_i}(a_i)) (origin
    # independent), and the normalised density must integrate to 1 over the
    # children.
    pe = Uniform(0.0, 1.0)
    d1 = Gamma(2.0, 1.0)
    d2 = LogNormal(1.0, 0.5)
    bounds = [(0.0, 4.0), (0.0, 6.0)]
    pd = parallel_distribution(pe, d1, d2; child_bounds = bounds)

    expected = log(cdf(d1, 4.0) - cdf(d1, 0.0)) +
               log(cdf(d2, 6.0) - cdf(d2, 0.0))
    @test pd.log_norm ≈ expected atol=1e-10

    # Children y_i = o + D_i with o ∈ [0,1], D1 ∈ [0,4] ⇒ y1 ∈ [0,5];
    # D2 ∈ [0,6] ⇒ y2 ∈ [0,7]. Integrate the normalised density finely.
    as = range(0.0, 5.2; length = 400)
    bs = range(0.0, 7.2; length = 400)
    da = step(as)
    db = step(bs)
    total = sum(exp(logpdf(pd, [a, b])) * da * db for a in as, b in bs)
    @test total ≈ 1.0 atol=1e-2
end

@testitem "ParallelDistribution rand matches generative process" begin
    using Distributions, Random, Statistics

    # rand draws one shared origin and adds each (bound-truncated) child
    # delay. Validate against the generative moments: E[Y_i] = E[O] +
    # E[D_i] and the shared origin induces positive covariance Cov(Y_i,
    # Y_j) = Var(O).
    rng = MersenneTwister(99)
    pe = Uniform(0.0, 1.0)
    d1 = Gamma(2.0, 1.0)
    d2 = LogNormal(1.0, 0.5)
    pd = parallel_distribution(pe, d1, d2)

    N = 2_000_000
    samples = [rand(rng, pd) for _ in 1:N]
    s1 = getindex.(samples, 1)
    s2 = getindex.(samples, 2)

    @test mean(s1) ≈ mean(pe) + mean(d1) atol=5e-3
    @test mean(s2) ≈ mean(pe) + mean(d2) atol=1e-2
    # Shared origin couples the children: covariance equals Var(O).
    @test cov(s1, s2) ≈ var(pe) atol=5e-3

    # Bounded child draws stay within their delay window plus the origin.
    pdb = parallel_distribution(pe, d1, d2; child_bounds = [(0.0, 3.0),
        (-Inf, Inf)])
    sb = [rand(rng, pdb) for _ in 1:100_000]
    # y1 = o + truncated(D1, 0, 3) ⇒ y1 ∈ [0, 1+3].
    @test all(0.0 .<= getindex.(sb, 1) .<= 4.0)
end

@testitem "ParallelDistribution truncated and primary-censored branches" begin
    using Distributions, Random, Statistics

    # Branch components are generic UnivariateDistributions: a plain delay,
    # a truncated delay, and a primary-censored delay all expose a
    # continuous pdf/cdf the origin integrand consumes. Validate each
    # composed-branch joint cdf against the shared-origin generative
    # process.
    rng = MersenneTwister(2024)
    pe = Uniform(0.0, 1.0)

    trunc_branch = truncated(Gamma(2.0, 1.0), 0.0, 8.0)
    pc_branch = primary_censored(LogNormal(1.0, 0.5), Uniform(0.0, 1.0))

    pd = parallel_distribution(pe, trunc_branch, pc_branch)
    @test length(pd) == 2

    N = 3_000_000
    o = rand(rng, pe, N)
    y1 = o .+ rand(rng, trunc_branch, N)
    y2 = o .+ rand(rng, pc_branch, N)
    for x in ([3.0, 3.0], [2.0, 4.0])
        mc = mean((y1 .<= x[1]) .& (y2 .<= x[2]))
        @test cdf(pd, x) ≈ mc atol=4e-3
    end
end

@testitem "ParallelDistribution accepts a composed (Convolved) branch" begin
    using Distributions, Random, Statistics

    # The branch interface is generic: a Convolved (sum-of-delays) branch
    # nests cleanly because it is a UnivariateDistribution independent of
    # the shared origin. Validate the joint cdf against the generative
    # process and that the joint density integrates to 1. Cross-nesting two
    # ParallelDistributions that share the *same* origin (a coupled
    # conditional) is a documented convergence follow-up, not covered here.
    rng = MersenneTwister(303)
    pe = Uniform(0.0, 1.0)
    conv = generic_convolve(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    leaf = Gamma(1.5, 1.0)
    pd = parallel_distribution(pe, conv, leaf)

    N = 3_000_000
    o = rand(rng, pe, N)
    y1 = o .+ rand(rng, conv, N)
    y2 = o .+ rand(rng, leaf, N)
    x = [6.0, 2.0]
    mc = mean((y1 .<= x[1]) .& (y2 .<= x[2]))
    @test cdf(pd, x) ≈ mc atol=4e-3

    as = range(0.01, 25.0; length = 160)
    bs = range(0.01, 14.0; length = 160)
    da = step(as)
    db = step(bs)
    total = sum(exp(logpdf(pd, [a, b])) * da * db for a in as, b in bs)
    @test total ≈ 1.0 atol=2e-2
end

@testitem "ParallelDistribution edge cases" begin
    using Distributions

    pe = Uniform(0.0, 1.0)
    pd = parallel_distribution(pe, Gamma(2.0, 1.0), LogNormal(1.0, 0.5))

    # NaN propagates.
    @test isnan(cdf(pd, [NaN, 1.0]))
    @test isnan(logpdf(pd, [1.0, NaN]))

    # An observation below the origin support gives zero density / cdf.
    @test logpdf(pd, [-1.0, 2.0]) == -Inf
    @test cdf(pd, [-1.0, 2.0]) == 0.0

    # logcdf is the log of cdf.
    x = [2.0, 3.0]
    @test logcdf(pd, x) ≈ log(cdf(pd, x)) atol=1e-10
end
