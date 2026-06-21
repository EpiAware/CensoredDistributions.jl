@testitem "Difference constructor and fields" begin
    using Distributions

    d = difference(Normal(5.0, 1.0), Normal(2.0, 1.0))
    @test d isa CensoredDistributions.Difference
    @test d.x == Normal(5.0, 1.0)
    @test d.y == Normal(2.0, 1.0)
    @test d.method isa AnalyticalSolver

    dn = difference(Gamma(2.0, 1.0), LogNormal(0.5, 0.4); method = NumericSolver())
    @test dn.method isa NumericSolver
end

@testitem "Difference support is two-sided and can be negative" begin
    using Distributions

    # Gamma - Gamma: support is (min(X) - max(Y)) .. (max(X) - min(Y)).
    d = difference(Gamma(3.0, 1.0), Gamma(2.0, 1.0))
    @test minimum(d) == -Inf      # 0 - Inf
    @test maximum(d) == Inf       # Inf - 0

    # Bounded components give a finite, possibly negative, two-sided support.
    du = difference(Uniform(0.0, 1.0), Uniform(0.0, 2.0))
    @test minimum(du) == -2.0     # 0 - 2
    @test maximum(du) == 1.0      # 1 - 0
    @test minimum(du) < 0.0

    # Normal - Normal is unbounded both ways.
    dnn = difference(Normal(0.0, 1.0), Normal(0.0, 1.0))
    @test minimum(dnn) == -Inf
    @test maximum(dnn) == Inf
end

@testitem "Difference params and eltype" begin
    using Distributions

    d = difference(Uniform(0.0, 1.0), Uniform(0.0, 2.0))
    @test params(d) == ((0.0, 1.0), (0.0, 2.0))

    d2 = difference(Gamma(2.0, 1.0), Normal(0.0, 1.0))
    @test eltype(d2) == Float64
end

@testitem "Difference Normal-Normal matches the closed form" begin
    using Distributions

    # Z = X - Y for independent normals is Normal(μx - μy, sqrt(σx² + σy²)).
    x = Normal(5.0, 1.5)
    y = Normal(2.0, 2.0)
    d = difference(x, y)
    ref = Normal(5.0 - 2.0, sqrt(1.5^2 + 2.0^2))

    @test mean(d) ≈ mean(ref)
    @test var(d) ≈ var(ref)
    @test std(d) ≈ std(ref)

    for z in -6.0:1.0:10.0
        @test cdf(d, z) ≈ cdf(ref, z) atol=1e-10
        @test pdf(d, z) ≈ pdf(ref, z) atol=1e-10
        @test logpdf(d, z) ≈ logpdf(ref, z) atol=1e-8
        @test logcdf(d, z) ≈ logcdf(ref, z) atol=1e-8
        @test ccdf(d, z) ≈ ccdf(ref, z) atol=1e-10
    end
end

@testitem "Difference NumericSolver matches Normal-Normal closed form" begin
    using Distributions

    # Force the numeric cross-correlation for a pair that has a closed form
    # and check it reproduces the analytic Normal difference.
    x = Normal(5.0, 1.5)
    y = Normal(2.0, 2.0)
    dn = difference(x, y; method = NumericSolver())
    ref = Normal(3.0, sqrt(1.5^2 + 2.0^2))

    @test CensoredDistributions._maybe_analytic(dn) === nothing
    @test CensoredDistributions._maybe_analytic(difference(x, y)) !== nothing

    for z in range(-4.0, 10.0; length = 12)
        @test cdf(dn, z) ≈ cdf(ref, z) atol=1e-6
        @test pdf(dn, z) ≈ pdf(ref, z) atol=1e-6
    end
end

@testitem "Difference numeric path matches Monte Carlo" begin
    using Distributions, Random, Statistics

    rng = MersenneTwister(42)
    # Gamma - LogNormal has no closed form -> numeric cross-correlation.
    x = Gamma(3.0, 1.0)
    y = LogNormal(0.5, 0.4)
    d = difference(x, y)

    n = 400_000
    samples = [rand(rng, x) - rand(rng, y) for _ in 1:n]

    for z in (-1.0, 0.5, 2.0, 4.0)
        @test cdf(d, z) ≈ mean(samples .<= z) atol=5e-3
    end

    @test pdf(d, 2.0) > 0
    @test logpdf(d, 2.0) ≈ log(pdf(d, 2.0)) atol=1e-8

    # Mean / var match the analytic difference / sum.
    @test mean(samples) ≈ mean(x) - mean(y) atol=2e-2
    @test var(samples) ≈ var(x) + var(y) atol=1e-1
end

@testitem "Difference pdf integrates to one" begin
    using Distributions

    # Bounded components: density integrates to 1 over the two-sided support.
    du = difference(Uniform(0.0, 2.0), Uniform(0.0, 3.0))
    grid = collect(-3.0:0.01:2.0)
    @test sum(pdf(du, z) for z in grid) * 0.01 ≈ 1.0 atol=2e-3

    # Unbounded numeric pair: density integrates to ~1 over a wide grid.
    dn = difference(Gamma(3.0, 1.0), LogNormal(0.5, 0.4))
    g = collect(-8.0:0.02:12.0)
    @test sum(pdf(dn, z) for z in g) * 0.02 ≈ 1.0 atol=3e-3
end

@testitem "Difference cdf is monotone and in [0, 1]" begin
    using Distributions

    dn = difference(Gamma(3.0, 1.0), LogNormal(0.5, 0.4))
    zs = collect(-6.0:0.5:10.0)
    cs = [cdf(dn, z) for z in zs]
    @test all(0.0 .<= cs .<= 1.0)
    @test all(diff(cs) .>= -1e-10)        # non-decreasing

    # Far below support is exactly 0; far above approaches 1 to within the
    # quadrature window's tail truncation (`_CONVOLVED_TAIL`), since the
    # support is unbounded above and there is no exact short-circuit.
    @test cdf(dn, -1e3) == 0.0
    @test cdf(dn, 1e3) ≈ 1.0 atol=1e-6
end

@testitem "Difference symmetry: X-Y is the reflection of Y-X" begin
    using Distributions

    # If Z = X - Y and W = Y - X then W = -Z, so
    # F_Z(z) = P(Z <= z) = P(W >= -z) = ccdf(W, -z) and f_Z(z) = f_W(-z).
    x = Gamma(3.0, 1.0)
    y = LogNormal(0.5, 0.4)
    dxy = difference(x, y)
    dyx = difference(y, x)

    # X - Y and Y - X integrate over different (unbounded-above) component
    # windows, so the two numeric quadratures agree only to quadrature
    # accuracy rather than exactly.
    for z in (-2.0, -0.5, 1.0, 3.0)
        @test cdf(dxy, z) ≈ ccdf(dyx, -z) atol=1e-5
        @test pdf(dxy, z) ≈ pdf(dyx, -z) atol=1e-5
    end

    # Support reflects too.
    @test minimum(dxy) == -maximum(dyx)
    @test maximum(dxy) == -minimum(dyx)
end

@testitem "Difference rand mean/var match the analytic moments" begin
    using Distributions, Random, Statistics

    rng = MersenneTwister(1)
    d = difference(Gamma(3.0, 1.0), Normal(2.0, 0.5))
    s = [rand(rng, d) for _ in 1:200_000]
    # mean = mean(X) - mean(Y); var = var(X) + var(Y).
    @test mean(s) ≈ (3.0 - 2.0) atol=5e-2
    @test var(s) ≈ (var(Gamma(3.0, 1.0)) + 0.25) atol=1e-1
    @test mean(s) ≈ mean(d) atol=5e-2
    @test var(s) ≈ var(d) atol=1e-1
end

@testitem "Difference logcdf/ccdf/logccdf branches" begin
    using Distributions

    # Analytic path agrees with the reference Normal difference.
    da = difference(Normal(1.0, 2.0), Normal(0.0, 1.0))
    refa = Normal(1.0, sqrt(4.0 + 1.0))
    @test logcdf(da, 2.0) ≈ logcdf(refa, 2.0) atol=1e-10
    @test ccdf(da, 2.0) ≈ ccdf(refa, 2.0) atol=1e-10
    @test logccdf(da, 2.0) ≈ logccdf(refa, 2.0) atol=1e-8

    # Numeric path: logcdf matches log(cdf) and ccdf = 1 - cdf.
    dn = difference(Gamma(3.0, 1.0), LogNormal(0.5, 0.4))
    @test logcdf(dn, 1.0) ≈ log(cdf(dn, 1.0)) atol=1e-10
    @test ccdf(dn, 1.0) ≈ 1 - cdf(dn, 1.0) atol=1e-10
    @test logccdf(dn, 1.0) ≈ log1p(-cdf(dn, 1.0)) atol=1e-6

    # logccdf edge cases via a bounded numeric-path difference.
    db = difference(Uniform(0.0, 2.0), Uniform(0.0, 3.0))
    @test logccdf(db, -3.0) == 0.0   # CDF = 0 -> logccdf = 0
    @test logccdf(db, 2.0) == -Inf   # CDF = 1 -> logccdf = -Inf
end

@testitem "Difference logpdf outside support is -Inf" begin
    using Distributions

    # Bounded support [-2, 1]; points outside are zero density.
    d = difference(Uniform(0.0, 1.0), Uniform(0.0, 2.0))
    @test logpdf(d, -3.0) == -Inf
    @test pdf(d, -3.0) == 0.0
    @test logpdf(d, 2.0) == -Inf
    @test pdf(d, 2.0) == 0.0
    @test !insupport(d, -3.0)
    @test insupport(d, -0.5)
end

@testitem "Difference scalar methods value-correct and inferrable" begin
    using Distributions, Test

    analytic = difference(Normal(1.0, 2.0), Normal(0.0, 1.0))
    numeric = difference(Gamma(3.0, 1.0), LogNormal(0.5, 0.4))
    for d in (analytic, numeric)
        for f in (cdf, logcdf, pdf, logpdf, ccdf, logccdf)
            @test f(d, 1.0) isa Float64
        end
        @test (@inferred(cdf(d, 1.0)); true)
        @test (@inferred(pdf(d, 1.0)); true)
    end
end

@testitem "Difference eltype and sampler" begin
    using Distributions, Random

    d = difference(Gamma(2.0, 1.0), Normal(0.0, 1.0))
    @test eltype(d) == Float64
    @test sampler(d) === d
    rng = MersenneTwister(3)
    @test rand(rng, d) isa Real
end

@testitem "Difference quantile inverts cdf and composes with truncated" begin
    using Distributions, Random, Statistics

    # Numeric path: quantile is the cdf inverse (optimiser accuracy ~1e-3).
    d = difference(Gamma(3.0, 1.0), LogNormal(0.5, 0.4))
    for p in (0.25, 0.5, 0.75)
        q = quantile(d, p)
        @test cdf(d, q) ≈ p atol=1e-3
    end

    # truncated composes over a Difference (negative bounds allowed).
    td = truncated(d, -2.0, 5.0)
    @test cdf(td, -3.0) == 0.0
    @test cdf(td, 6.0) == 1.0
    @test 0.0 < cdf(td, 1.0) < 1.0
    @test pdf(td, 1.0) > 0
    @test get_dist(td) === d

    rng = MersenneTwister(8)
    samples = rand(rng, td, 100_000)
    @test all(-2.0 .<= samples .<= 5.0)
end

@testitem "Difference composes with weight" begin
    using Distributions

    d = difference(Normal(1.0, 1.0), Normal(0.0, 2.0))
    wd = weight(d, 3.0)
    @test logpdf(wd, 1.0) ≈ 3.0 * logpdf(d, 1.0) atol=1e-8
    @test get_dist(wd) === d
end

# The AD-safety of Difference (gradients flowing through both the minuend X and
# the subtrahend Y parameters, on the numeric cross-correlation path) is covered
# by the multi-backend AD suite in `test/ADFixtures` ("Difference ..."), which
# has the AD backends as dependencies; the main test env does not.
