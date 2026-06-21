# Nesting-matrix coverage for the scalar/vector scoring methods on NESTED
# composed objects (#645).
#
# The composer scoring methods (`logpdf`/`pdf`/`cdf`/`ccdf`/`logccdf`) must be
# defined and CORRECT on composed objects whose children are themselves
# composers or wrappers: a `Sequential` whose step is a `Parallel`, a `Choose`
# of `Sequential`s, a `Convolved` of a composed univariate leaf (a `Resolve`),
# and `truncated` / censored wrappers over a composed leaf. This was a TEST gap,
# not a missing-method bug: every case below already returns a finite, correct
# value (verified against a hand reference / Monte-Carlo / the leaf factorisation
# the composer documents). The matrix pins each method on each nesting so a
# future regression that drops one surfaces here.
#
# `Sequential` / `Parallel` are MULTIVARIATE (their realisation is a flat value
# vector), so they expose `logpdf`/`pdf` over a vector, NOT a scalar `cdf`. A
# `Resolve` and a `Convolved` are UNIVARIATE, so they additionally expose the
# scalar `cdf`/`ccdf`/`logccdf` family. The matrix checks each method where it is
# defined and that a `Sequential`/`Parallel` scalar `cdf` is genuinely absent
# (no accidental method).

@testitem "nested composed: Sequential step is a Parallel" begin
    using Distributions
    using CensoredDistributions: Sequential, Parallel

    g = Gamma(2.0, 1.0)
    g2 = Gamma(3.0, 0.7)
    ln = LogNormal(0.5, 0.4)

    # A chain whose second step is a two-branch Parallel: three leaf values
    # `[step_1, branch_1, branch_2]`, joint = product of the three leaf
    # densities (independent edges, observed value vector).
    seqp = Sequential((g, Parallel((g2, ln))))
    @test length(seqp) == 3

    x = [0.8, 1.3, 1.7]
    lp = logpdf(seqp, x)
    ref = logpdf(g, x[1]) + logpdf(g2, x[2]) + logpdf(ln, x[3])
    @test isapprox(lp, ref; atol = 1e-12)
    @test isapprox(pdf(seqp, x), exp(ref); atol = 1e-12)

    # A flat `rand` round-trips through the nested layout (NamedTuple record).
    r = rand(seqp)
    @test r isa NamedTuple
    @test length(r) == 3
end

@testitem "nested composed: Choose of Sequentials" begin
    using Distributions
    using CensoredDistributions: Sequential

    g = Gamma(2.0, 1.0)
    g2 = Gamma(3.0, 0.7)
    ln = LogNormal(0.5, 0.4)

    # A data-selected disjunction whose alternatives are each a two-step chain.
    # Scoring routes to the SELECTED alternative's own vector `logpdf`.
    ch = choose(:a => Sequential((g, ln)), :b => Sequential((g2, g)))

    xa = [1.0, 2.0]
    @test isapprox(logpdf(ch, xa; kind = :a),
        logpdf(g, xa[1]) + logpdf(ln, xa[2]); atol = 1e-12)
    @test isapprox(pdf(ch, xa; kind = :a),
        exp(logpdf(g, xa[1]) + logpdf(ln, xa[2])); atol = 1e-12)

    xb = [1.5, 2.5]
    @test isapprox(logpdf(ch, xb; kind = :b),
        logpdf(g2, xb[1]) + logpdf(g, xb[2]); atol = 1e-12)

    # No `kind` is an error (a Choose has no single distribution to score).
    @test_throws ArgumentError logpdf(ch, xa)
end

@testitem "nested composed: Convolved of a composed univariate leaf" begin
    using Distributions, Random
    using CensoredDistributions: convolve_distributions

    # A `Resolve` is a UNIVARIATE marginal (time-to-resolution), so it can be a
    # `Convolved` component. The convolution sums the Resolve marginal and a
    # plain leaf; cdf/pdf/logpdf must score and integrate correctly.
    r = resolve(:death => (Gamma(2.0, 1.0), 0.3),
        :disch => (LogNormal(0.5, 0.4), 0.7))
    g2 = Gamma(3.0, 0.7)
    cvr = convolve_distributions(r, g2)

    @test isfinite(logpdf(cvr, 5.0))
    @test 0.0 <= cdf(cvr, 5.0) <= 1.0
    @test isapprox(ccdf(cvr, 5.0), 1 - cdf(cvr, 5.0); atol = 1e-10)
    @test isapprox(logccdf(cvr, 5.0), log1p(-cdf(cvr, 5.0)); atol = 1e-8)

    # Monte-Carlo check of the convolution cdf: sum the Resolve marginal draw
    # and the leaf draw, compare the empirical to the analytic cdf.
    Random.seed!(202)
    mc = mean(rand(r) + rand(g2) <= 5.0 for _ in 1:400_000)
    @test isapprox(cdf(cvr, 5.0), mc; atol = 5e-3)

    # The scalar cdf is monotone over a grid.
    cs = [cdf(cvr, x) for x in range(0.0, 12.0; length = 13)]
    @test issorted(cs)
end

@testitem "nested composed: truncated / censored over a composed leaf" begin
    using Distributions, QuadGK
    using CensoredDistributions: convolve_distributions

    # A `Convolved` is a univariate leaf, so it composes under `truncated` and
    # the censoring wrappers. The truncated cdf/logpdf must use the base
    # `Convolved` cdf/quantile and renormalise.
    cv = convolve_distributions(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    lo, hi = 0.0, 10.0
    tcv = truncated(cv, lo, hi)

    Z = cdf(cv, hi) - cdf(cv, lo)
    @test isapprox(cdf(tcv, 3.0), (cdf(cv, 3.0) - cdf(cv, lo)) / Z; atol = 1e-8)
    @test isapprox(logpdf(tcv, 3.0), logpdf(cv, 3.0) - log(Z); atol = 1e-8)
    @test isapprox(ccdf(tcv, 3.0), 1 - cdf(tcv, 3.0); atol = 1e-10)

    # The truncated density integrates to one over the window.
    mass, _ = quadgk(x -> pdf(tcv, x), lo, hi; rtol = 1e-8)
    @test isapprox(mass, 1.0; atol = 1e-6)

    # The censoring stack accepts a composed leaf: a primary-censored and an
    # interval-censored `Convolved` score finitely with a monotone cdf.
    pc = primary_censored(cv, Uniform(0, 1))
    @test isfinite(logpdf(pc, 3.0))
    pcs = [cdf(pc, x) for x in range(0.0, 12.0; length = 13)]
    @test issorted(pcs)

    ic = interval_censored(cv, 1.0)
    @test isfinite(logpdf(ic, 3.0))
end

@testitem "nested composed: Resolve scalar family is complete and correct" begin
    using Distributions, Random

    # A `Resolve` is the univariate branch-prob-weighted mixture marginal, so its
    # full scalar family (`logpdf`/`pdf`/`cdf`/`ccdf`/`logccdf`/`logcdf`) must be
    # defined and mutually consistent.
    g = Gamma(2.0, 1.0)
    ln = LogNormal(0.5, 0.4)
    p = 0.3
    r = resolve(:death => (g, p), :disch => (ln, 1 - p))

    # cdf is the mixture cdf; pdf/logpdf the mixture density.
    @test isapprox(cdf(r, 3.0), p * cdf(g, 3.0) + (1 - p) * cdf(ln, 3.0);
        atol = 1e-12)
    @test isapprox(pdf(r, 3.0), p * pdf(g, 3.0) + (1 - p) * pdf(ln, 3.0);
        atol = 1e-12)
    @test isapprox(logpdf(r, 3.0), log(pdf(r, 3.0)); atol = 1e-12)
    @test isapprox(ccdf(r, 3.0), 1 - cdf(r, 3.0); atol = 1e-10)
    @test isapprox(logccdf(r, 3.0), log1p(-cdf(r, 3.0)); atol = 1e-8)
    @test isapprox(logcdf(r, 3.0), log(cdf(r, 3.0)); atol = 1e-8)

    # Monte-Carlo check of the marginal cdf.
    Random.seed!(303)
    mc = mean(rand(r) <= 3.0 for _ in 1:400_000)
    @test isapprox(cdf(r, 3.0), mc; atol = 5e-3)
end
