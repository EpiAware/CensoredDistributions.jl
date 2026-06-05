@testitem "Default is Auto (univariate scalar + missingness dispatch)" begin
    using Distributions

    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0.0, 1.0)

    d = primary_censored(delay, pe)
    # Auto keeps the classic univariate scalar interface (so the common path is
    # unchanged) and adds vector logpdf for the missingness dispatch.
    @test d isa UnivariateDistribution
    @test d.method isa CensoredDistributions.Auto

    # Scalar observed time takes the marginal path (the classic reference
    # values, unchanged): cdf, logpdf, quantile, rand all work as today.
    @test cdf(d, 3.0) ≈ 0.2183282452603626
    @test logpdf(d, 3.0) ≈ -1.8626929385055817
    @test quantile(d, 0.5) isa Real
    @test length(rand(d, 5)) == 5
end

@testitem "Auto missingness dispatch: missing marginalises, value conditions" begin
    using Distributions

    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0.0, 1.0)
    d = primary_censored(delay, pe)
    dm = primary_censored(delay, pe; method = Marginal())

    for y in [1.0, 2.5, 5.0]
        # [missing, y] marginalises -> equals the forced Marginal logpdf(y)
        @test logpdf(d, [missing, y]) ≈ logpdf(dm, y)
        # scalar y is the same marginal path
        @test logpdf(d, y) ≈ logpdf(dm, y)
        @test pdf(d, [missing, y]) ≈ pdf(dm, y)
    end

    # [p, y] conditions on the concrete primary
    for (p, y) in [(0.3, 2.7), (0.1, 1.0)]
        @test logpdf(d, [p, y]) ≈ logpdf(pe, p) + logpdf(delay, y - p)
    end
end

@testitem "Force overrides: Marginal univariate, Latent condition-only" begin
    using Distributions

    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0.0, 1.0)

    # method = Marginal() forces the univariate scalar formulation
    dm = primary_censored(delay, pe; method = Marginal())
    @test dm isa UnivariateDistribution
    @test dm.method isa Marginal
    @test cdf(dm, 3.0) ≈ 0.2183282452603626
    @test logpdf(dm, 3.0) ≈ -1.8626929385055817

    # method = Latent() forces conditioning; a missing primary errors
    dl = primary_censored(delay, pe; method = Latent())
    @test dl isa Distribution{Multivariate, Continuous}
    @test logpdf(dl, [0.3, 2.7]) ≈ logpdf(pe, 0.3) + logpdf(delay, 2.7 - 0.3)
    @test_throws ArgumentError logpdf(dl, [missing, 2.7])
end

@testitem "Latent is multivariate over [primary, observed]" begin
    using Distributions, Random

    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0.0, 1.0)

    d = primary_censored(delay, pe; latent = true)
    @test d isa Distribution{Multivariate, Continuous}
    @test d.method isa Latent
    @test length(d) == 2

    # method = Latent() is equivalent to latent = true
    d2 = primary_censored(delay, pe; method = Latent())
    @test d2 isa Distribution{Multivariate, Continuous}

    rng = MersenneTwister(42)
    x = rand(rng, d)
    @test x isa AbstractVector
    @test length(x) == 2
    p, y = x[1], x[2]
    # observed = primary + delay, so observed >= primary
    @test y >= p
    @test insupport(pe, p)
end

@testitem "Latent logpdf is the combined joint" begin
    using Distributions

    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0.0, 1.0)
    d = primary_censored(delay, pe; latent = true)

    for (p, y) in [(0.3, 2.7), (0.1, 1.0), (0.9, 5.4)]
        expected = logpdf(pe, p) + logpdf(delay, y - p)
        @test logpdf(d, [p, y]) ≈ expected
        @test pdf(d, [p, y]) ≈ exp(expected)
    end

    # Outside support: primary out of window or negative implied delay
    @test !insupport(d, [1.5, 2.0])     # primary above window
    @test !insupport(d, [0.5, 0.2])     # implied delay negative
end

@testitem "Integrating Latent over primary_prior recovers Marginal" begin
    using Distributions

    # The defining equivalence: integrating the conditional delay density over
    # the primary-event prior reproduces the marginal pdf.
    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0.0, 1.0)
    dm = primary_censored(delay, pe)
    prior = primary_prior(dm)

    function integrate_latent(x; n = 40_000)
        ps = range(minimum(prior), maximum(prior); length = n)
        vals = map(p -> pdf(delay, x - p) * pdf(prior, p), ps)
        return sum((vals[1:(end - 1)] .+ vals[2:end]) ./ 2) * step(ps)
    end

    for x in [1.0, 2.5, 5.0]
        @test isapprox(integrate_latent(x), pdf(dm, x); rtol = 1e-3)
    end
end

@testitem "primary_prior accessor (uncoupled and coupled)" begin
    using Distributions

    delay = LogNormal(1.5, 0.75)
    d = primary_censored(delay, Uniform(0.0, 1.0))

    # Uncoupled prior is the primary_event itself
    @test primary_prior(d) === d.primary_event

    # Coupled prior is a BoundedPrimary truncated by the secondary time
    bp = primary_prior(d, 0.6)
    @test bp isa CensoredDistributions.BoundedPrimary
    @test minimum(bp) == 0.0
    @test maximum(bp) == 0.6

    # secondary beyond the window keeps the full window
    @test maximum(primary_prior(d, 5.0)) == 1.0
end

@testitem "BoundedPrimary Jacobian property and non-Uniform error" begin
    using Distributions

    # Jacobian-corrected logpdf reproduces the implicit uniform-over-window
    # prior, independent of where the secondary time bounds the window.
    for (lower,
        width,
        secondary) in [
        (0.0, 1.0, 0.7), (0.0, 1.0, 0.4), (3.0, 1.0, 10.0),
        (0.0, 2.0, 1.5), (5.0, 0.5, 5.2)
    ]
        d = primary_censored(LogNormal(0.0, 1.0), Uniform(lower, lower + width))
        bp = primary_prior(d, secondary)
        implicit = Uniform(lower, lower + width)
        upper = min(lower + width, secondary)
        for t in range(lower + 1e-6, upper - 1e-6; length = 5)
            @test logpdf(bp, t) ≈ logpdf(implicit, t)
        end
    end

    # Non-Uniform primary event must error (the Jacobian is uniform-only).
    d_nonunif = primary_censored(
        LogNormal(0.0, 1.0), truncated(Normal(0.5, 0.2), 0, 1))
    @test_throws ArgumentError primary_prior(d_nonunif, 0.5)
end

@testitem "BoundedPrimary MC integration reproduces marginal CDF" begin
    using Distributions, Random

    rng = MersenneTwister(2024)
    delay = LogNormal(1.5, 0.75)
    width = 1.0
    d = primary_censored(delay, Uniform(0.0, width))

    n = 2_000_000
    function mc_cdf(q)
        cnt = 0
        for _ in 1:n
            bp = primary_prior(d, width)  # full window
            t_pri = rand(rng, bp)
            obs = t_pri + rand(rng, delay)
            cnt += (obs <= q)
        end
        return cnt / n
    end

    for q in [1.0, 3.0, 6.0]
        @test isapprox(mc_cdf(q), cdf(d, q); atol = 2e-3)
    end
end

@testitem "IntervalCensored over Latent censors the observed coordinate" begin
    using Distributions, Random

    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0.0, 1.0)
    dl = primary_censored(delay, pe; latent = true)
    ic = interval_censored(dl, 1.0)

    @test ic isa Distribution{Multivariate, Continuous}
    @test length(ic) == 2

    # logpdf = primary prior + log P(delay in interval containing observed - p)
    p, y = 0.3, 2.7
    lower = floor(y)          # interval width 1.0 -> [2, 3)
    upper = lower + 1.0
    expected = logpdf(pe, p) +
               log(cdf(delay, upper - p) - cdf(delay, lower - p))
    @test logpdf(ic, [p, y]) ≈ expected

    # rand censors the observed coordinate to its interval, primary continuous
    rng = MersenneTwister(7)
    s = rand(rng, ic)
    @test s[2] == floor(s[2])           # observed snapped to integer interval
    @test s[1] != floor(s[1]) || insupport(pe, s[1])  # primary continuous

    # interval_censored on a forced Marginal stays univariate (unchanged path)
    dm = primary_censored(delay, pe; method = Marginal())
    @test interval_censored(dm, 1.0) isa UnivariateDistribution

    # interval_censored on the default Auto also uses the univariate marginal
    # path (Auto keeps the classic scalar interface)
    da = primary_censored(delay, pe)
    @test interval_censored(da, 1.0) isa UnivariateDistribution
end
