@testitem "ParallelPrimaryCensored constructor and validation" begin
    using Distributions

    d = primary_censored(
        [Gamma(2.0, 1.0), LogNormal(1.0, 0.5)], Uniform(0.0, 1.0))
    @test d isa CensoredDistributions.ParallelPrimaryCensored
    # length is 1 (shared primary) + number of branches.
    @test length(d) == 3
    @test d.primary_event == Uniform(0.0, 1.0)
    @test all(b -> b == (-Inf, Inf), d.child_bounds)
    @test d.log_norm == 0.0

    # Single branch is allowed.
    d1 = primary_censored([Gamma(2.0, 1.0)], Uniform(0.0, 1.0))
    @test length(d1) == 2

    # child_bounds must match the number of delays.
    @test_throws ArgumentError primary_censored(
        [Gamma(2.0, 1.0), LogNormal(1.0, 0.5)], Uniform(0.0, 1.0);
        child_bounds = [(0.0, 5.0)])

    # Each bound must be ordered.
    @test_throws ArgumentError primary_censored(
        [Gamma(2.0, 1.0)], Uniform(0.0, 1.0);
        child_bounds = [(5.0, 1.0)])

    # An empty delay vector is rejected.
    @test_throws ArgumentError primary_censored(
        UnivariateDistribution[], Uniform(0.0, 1.0))
end

@testitem "ParallelPrimaryCensored primary_prior and params" begin
    using Distributions

    pe = Uniform(0.0, 2.0)
    d = primary_censored(
        [Gamma(2.0, 1.0), LogNormal(1.0, 0.5)], pe)
    @test primary_prior(d) === pe
    p = params(d)
    @test p[1] == params(pe)
    @test p[2] == params(Gamma(2.0, 1.0))
    @test p[3] == params(LogNormal(1.0, 0.5))
end

@testitem "ParallelPrimaryCensored n=1 reduces to primary_censored" begin
    using Distributions

    # With one branch and infinite bounds the marginal of the single branch
    # must match primary_censored(delay, primary_event) numerics for the
    # analytical (Gamma/Uniform) and numeric primary-censored paths.
    pe = Uniform(0.0, 1.0)
    for delay in (Gamma(2.0, 1.0), LogNormal(1.0, 0.5), Weibull(2.0, 1.5))
        pd = primary_censored([delay], pe)
        pc = primary_censored(delay, pe)
        for x in [0.5, 1.5, 2.5, 4.0, 6.0]
            # The joint cdf over the one branch is the same integral.
            @test cdf(pd, [missing, x]) ≈ cdf(pc, x) atol=1e-9
            @test logcdf(pd, [missing, x]) ≈ logcdf(pc, x) atol=1e-8
            # The marginalised (missing primary) logpdf matches the scalar
            # marginal logpdf. The parallel path integrates the origin
            # density directly while the scalar `primary_censored` path
            # finite-differences its cdf with a fixed step h=1e-8, so the two
            # agree only to the finite-difference truncation error (~1e-4);
            # the exact-quadrature `cdf` reduction above matches to 1e-9.
            @test logpdf(pd, [missing, x]) ≈ logpdf(pc, x) rtol=1e-3
        end
    end
end

@testitem "ParallelPrimaryCensored n=1 marginal logpdf integrates to 1" begin
    using Distributions

    pe = Uniform(0.0, 1.0)
    pd = primary_censored([Gamma(2.0, 1.0)], pe)
    xs = range(0.001, 20.0; length = 4000)
    dx = step(xs)
    total = sum(x -> exp(logpdf(pd, [missing, x])), xs) * dx
    @test total ≈ 1.0 atol=1e-3
end

@testitem "ParallelPrimaryCensored cdf matches Monte Carlo (n=2)" begin
    using Distributions, Random, Statistics

    # Validate the joint cdf (the 1-D shared-origin integral) against a
    # Monte-Carlo estimate of P(Y1 ≤ x1 ∧ Y2 ≤ x2) where Y_i = O + D_i and
    # O is a single shared draw.
    rng = MersenneTwister(20240601)
    pe = Uniform(0.0, 1.0)
    d1 = Gamma(2.0, 1.0)
    d2 = LogNormal(1.0, 0.5)
    pd = primary_censored([d1, d2], pe)

    N = 4_000_000
    o = rand(rng, pe, N)
    y1 = o .+ rand(rng, d1, N)
    y2 = o .+ rand(rng, d2, N)

    for x in ([2.0, 3.0], [1.0, 2.0], [4.0, 5.0], [3.0, 1.5])
        mc = mean((y1 .<= x[1]) .& (y2 .<= x[2]))
        @test cdf(pd, [missing, x[1], x[2]]) ≈ mc atol=3e-3
    end
end

@testitem "ParallelPrimaryCensored logpdf matches Monte Carlo (n=2)" begin
    using Distributions, Random, Statistics

    # Validate the joint density (marginalised primary) against a Monte-Carlo
    # estimate of the joint density on a small grid box.
    rng = MersenneTwister(11)
    pe = Uniform(0.0, 1.0)
    d1 = Gamma(2.0, 1.0)
    d2 = LogNormal(1.0, 0.5)
    pd = primary_censored([d1, d2], pe)

    N = 6_000_000
    o = rand(rng, pe, N)
    y1 = o .+ rand(rng, d1, N)
    y2 = o .+ rand(rng, d2, N)

    h = 0.1
    for c in ([2.0, 3.0], [1.5, 2.0])
        in_box = (abs.(y1 .- c[1]) .<= h / 2) .& (abs.(y2 .- c[2]) .<= h / 2)
        mc_density = mean(in_box) / h^2
        @test exp(logpdf(pd, [missing, c[1], c[2]])) ≈ mc_density rtol=5e-2
    end
end

@testitem "ParallelPrimaryCensored joint density integrates to 1 (n=3)" begin
    using Distributions, Random, Statistics

    rng = MersenneTwister(7)
    pe = Uniform(0.0, 1.0)
    delays = [Gamma(2.0, 1.0), LogNormal(1.0, 0.5), Gamma(1.5, 1.5)]
    pd = primary_censored(delays, pe)

    # cdf vs Monte Carlo for a 3-branch vector.
    N = 4_000_000
    o = rand(rng, pe, N)
    ys = [o .+ rand(rng, dl, N) for dl in delays]
    x = [3.0, 3.0, 4.0]
    mc = mean((ys[1] .<= x[1]) .& (ys[2] .<= x[2]) .& (ys[3] .<= x[3]))
    @test cdf(pd, [missing, x...]) ≈ mc atol=3e-3

    # Importance estimate of ∫ f = 1: sample branches from an independent
    # proposal and average f / q.
    Nimp = 2_000_000
    q1 = Gamma(3.0, 1.5)
    q2 = LogNormal(1.2, 0.7)
    q3 = Gamma(2.5, 2.0)
    s1 = rand(rng, q1, Nimp)
    s2 = rand(rng, q2, Nimp)
    s3 = rand(rng, q3, Nimp)
    mean_w = mean(1:Nimp) do i
        logq = logpdf(q1, s1[i]) + logpdf(q2, s2[i]) + logpdf(q3, s3[i])
        exp(logpdf(pd, [missing, s1[i], s2[i], s3[i]]) - logq)
    end
    @test mean_w ≈ 1.0 atol=2e-2
end

@testitem "ParallelPrimaryCensored bounded children normalisation" begin
    using Distributions

    # With finite child bounds the log-normalisation constant must equal the
    # closed product Π_i (F_{D_i}(b_i) - F_{D_i}(a_i)) (origin independent),
    # and the normalised density must integrate to 1 over the children.
    pe = Uniform(0.0, 1.0)
    d1 = Gamma(2.0, 1.0)
    d2 = LogNormal(1.0, 0.5)
    bounds = [(0.0, 4.0), (0.0, 6.0)]
    pd = primary_censored([d1, d2], pe; child_bounds = bounds)

    expected = log(cdf(d1, 4.0) - cdf(d1, 0.0)) +
               log(cdf(d2, 6.0) - cdf(d2, 0.0))
    @test pd.log_norm ≈ expected atol=1e-10

    as = range(0.0, 5.2; length = 400)
    bs = range(0.0, 7.2; length = 400)
    da = step(as)
    db = step(bs)
    total = sum(
        exp(logpdf(pd, [missing, a, b])) * da * db for a in as, b in bs)
    @test total ≈ 1.0 atol=1e-2
end

@testitem "ParallelPrimaryCensored missingness dispatch" begin
    using Distributions

    pe = Uniform(0.0, 1.0)
    d1 = Gamma(2.0, 1.0)
    d2 = LogNormal(1.0, 0.5)
    pd = primary_censored([d1, d2], pe)

    # A concrete primary conditions: log f_O(p) + Σ log f_{D_i}(y_i - p).
    p = 0.3
    y1 = 2.0
    y2 = 3.0
    expected_cond = logpdf(pe, p) + logpdf(d1, y1 - p) + logpdf(d2, y2 - p)
    @test logpdf(pd, [p, y1, y2]) ≈ expected_cond atol=1e-10

    # A missing branch is marginalised: dropping branch 2 from the concrete
    # case removes its delay-density factor.
    expected_drop = logpdf(pe, p) + logpdf(d1, y1 - p)
    @test logpdf(pd, [p, y1, missing]) ≈ expected_drop atol=1e-10

    # A single present branch with a missing primary reduces to that branch's
    # single-delay primary-censored marginal.
    pc1 = primary_censored(d1, pe)
    @test logpdf(pd, [missing, y1, missing]) ≈ logpdf(pc1, y1) atol=1e-5

    # An all-missing observation marginalises to 1 (log 0).
    @test logpdf(pd, [missing, missing, missing]) == 0.0
end

@testitem "ParallelPrimaryCensored rand matches generative process" begin
    using Distributions, Random, Statistics

    # rand draws one shared origin and adds each (bound-truncated) branch
    # delay; the vector is [primary, observed_1, observed_2]. Validate against
    # the generative moments and the shared-origin covariance.
    rng = MersenneTwister(99)
    pe = Uniform(0.0, 1.0)
    d1 = Gamma(2.0, 1.0)
    d2 = LogNormal(1.0, 0.5)
    pd = primary_censored([d1, d2], pe)

    N = 2_000_000
    samples = [rand(rng, pd) for _ in 1:N]
    origin = getindex.(samples, 1)
    s1 = getindex.(samples, 2)
    s2 = getindex.(samples, 3)

    @test mean(origin) ≈ mean(pe) atol=5e-3
    @test mean(s1) ≈ mean(pe) + mean(d1) atol=5e-3
    @test mean(s2) ≈ mean(pe) + mean(d2) atol=1e-2
    # Shared origin couples the branches: covariance equals Var(O).
    @test cov(s1, s2) ≈ var(pe) atol=5e-3

    # Bounded branch draws stay within their delay window plus the origin.
    pdb = primary_censored([d1, d2], pe;
        child_bounds = [(0.0, 3.0), (-Inf, Inf)])
    sb = [rand(rng, pdb) for _ in 1:100_000]
    # observed_1 = o + truncated(D1, 0, 3) ⇒ observed_1 ∈ [0, 1+3].
    @test all(0.0 .<= getindex.(sb, 2) .<= 4.0)
end

@testitem "ParallelPrimaryCensored composed branches" begin
    using Distributions, Random, Statistics

    # Branch components are generic UnivariateDistributions: a plain delay, a
    # truncated delay, and a primary-censored delay all expose a continuous
    # pdf/cdf the origin integrand consumes. Validate the joint cdf against the
    # shared-origin generative process.
    rng = MersenneTwister(2024)
    pe = Uniform(0.0, 1.0)

    trunc_branch = truncated(Gamma(2.0, 1.0), 0.0, 8.0)
    pc_branch = primary_censored(LogNormal(1.0, 0.5), Uniform(0.0, 1.0))

    pd = primary_censored([trunc_branch, pc_branch], pe)
    @test length(pd) == 3

    N = 3_000_000
    o = rand(rng, pe, N)
    y1 = o .+ rand(rng, trunc_branch, N)
    y2 = o .+ rand(rng, pc_branch, N)
    for x in ([3.0, 3.0], [2.0, 4.0])
        mc = mean((y1 .<= x[1]) .& (y2 .<= x[2]))
        @test cdf(pd, [missing, x[1], x[2]]) ≈ mc atol=4e-3
    end
end

@testitem "ParallelPrimaryCensored real-time truncated()" begin
    using Distributions, Random, Statistics

    # truncated(d, horizon) conditions the branches seen by `horizon`; a
    # not-yet-seen branch passed as `missing` marginalises in the numerator
    # while the joint-over-origin denominator P(all branches <= horizon) is
    # the shared-origin cdf at the horizon on every branch.
    pe = Uniform(0.0, 1.0)
    d1 = Gamma(2.0, 1.0)
    d2 = LogNormal(1.0, 0.5)
    pd = primary_censored([d1, d2], pe)
    horizon = 5.0
    dt = truncated(pd, horizon)

    @test length(dt) == 3
    @test primary_prior(dt) === pe

    den = log(cdf(pd, [missing, horizon, horizon]))

    # Both branches seen and conditioned.
    obs = [missing, 2.0, 3.0]
    @test logpdf(dt, obs) ≈ logpdf(pd, obs) - den atol=1e-9

    # One branch not yet seen (missing) -> marginalised in the numerator.
    obs2 = [missing, 2.0, missing]
    @test logpdf(dt, obs2) ≈ logpdf(pd, obs2) - den atol=1e-9

    # Truncated density (both branches present) integrates to 1 over the
    # observation region at or below the horizon.
    as = range(0.0, horizon; length = 300)
    bs = range(0.0, horizon; length = 300)
    da = step(as)
    db = step(bs)
    total = sum(
        exp(logpdf(dt, [missing, a, b])) * da * db for a in as, b in bs)
    @test total ≈ 1.0 atol=1e-2

    # Rejection sampling stays within the horizon on every branch.
    rng = MersenneTwister(5)
    sb = [rand(rng, dt) for _ in 1:50_000]
    @test all(s -> s[2] <= horizon && s[3] <= horizon, sb)
end

@testitem "ParallelPrimaryCensored edge cases" begin
    using Distributions

    pe = Uniform(0.0, 1.0)
    pd = primary_censored(
        [Gamma(2.0, 1.0), LogNormal(1.0, 0.5)], pe)

    # NaN propagates.
    @test isnan(cdf(pd, [missing, NaN, 1.0]))
    @test isnan(logpdf(pd, [missing, 1.0, NaN]))

    # An observation below the origin support gives zero density / cdf.
    @test logpdf(pd, [missing, -1.0, 2.0]) == -Inf
    @test cdf(pd, [missing, -1.0, 2.0]) == 0.0

    # logcdf is the log of cdf.
    x = [missing, 2.0, 3.0]
    @test logcdf(pd, x) ≈ log(cdf(pd, x)) atol=1e-10

    # Wrong-length vectors error.
    @test_throws DimensionMismatch logpdf(pd, [missing, 1.0])
    @test_throws DimensionMismatch cdf(pd, [missing, 1.0])
end
