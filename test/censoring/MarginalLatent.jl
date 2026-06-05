@testitem "Default formulation marginalises (classic univariate)" begin
    using Distributions

    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0.0, 1.0)

    d = primary_censored(delay, pe)
    # The default marginalises the primary: a univariate distribution with the
    # full classic scalar interface, unchanged.
    @test d isa UnivariateDistribution

    @test cdf(d, 3.0) ≈ 0.2183282452603626
    @test logpdf(d, 3.0) ≈ -1.8626929385055817
    @test quantile(d, 0.5) isa Real
    @test length(rand(d, 5)) == 5
end

@testitem "Latent is the single opt-in (sampler-owned, logpdf owns prior)" begin
    using Distributions, Random

    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0.0, 1.0)

    # method = Latent() keeps the primary as a sampler-owned latent variable.
    dl = primary_censored(delay, pe; method = Latent())
    @test dl isa Distribution{Multivariate, Continuous}
    @test dl.method isa Latent

    # rand produces the internal event times; logpdf owns the primary prior, so
    # logpdf([p, y]) is the full self-contained joint.
    rng = MersenneTwister(1)
    x = rand(rng, dl)
    @test length(x) == 2
    @test logpdf(dl, x) ≈ logpdf(pe, x[1]) + logpdf(delay, x[2] - x[1])
    @test logpdf(dl, [0.3, 2.7]) ≈ logpdf(pe, 0.3) + logpdf(delay, 2.7 - 0.3)

    # Latent does not marginalise: a missing primary errors.
    @test_throws ArgumentError logpdf(dl, [missing, 2.7])

    # The marginal default equals the latent joint integrated over the primary
    # (the two formulations agree).
    dm = primary_censored(delay, pe)
    integ = let n = 40_000
        ps = range(0.0, 1.0; length = n)
        vals = map(p -> exp(logpdf(dl, [p, 2.7])), ps)
        sum((vals[1:(end - 1)] .+ vals[2:end]) ./ 2) * step(ps)
    end
    @test isapprox(integ, pdf(dm, 2.7); rtol = 1e-3)
end

@testitem "Latent is multivariate over [primary, observed]" begin
    using Distributions, Random

    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0.0, 1.0)

    d = primary_censored(delay, pe; method = Latent())
    @test d isa Distribution{Multivariate, Continuous}
    @test d.method isa Latent
    @test length(d) == 2

    # Latent forces the multivariate formulation
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
    d = primary_censored(delay, pe; method = Latent())

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

@testitem "get_primary_event accessor (uncoupled and coupled)" begin
    using Distributions

    delay = LogNormal(1.5, 0.75)
    d = primary_censored(delay, Uniform(0.0, 1.0))

    # Uncoupled accessor returns the primary_event itself
    @test get_primary_event(d) === d.primary_event

    # Coupled prior is a BoundedPrimary truncated by the secondary time
    bp = get_primary_event(d, 0.6)
    @test bp isa CensoredDistributions.BoundedPrimary
    @test minimum(bp) == 0.0
    @test maximum(bp) == 0.6

    # secondary beyond the window keeps the full window
    @test maximum(get_primary_event(d, 5.0)) == 1.0

    # primary_prior is a backward-compatible alias of get_primary_event
    @test primary_prior(d) === get_primary_event(d)
    @test maximum(primary_prior(d, 0.6)) == maximum(get_primary_event(d, 0.6))
end

@testitem "Latent rand produces the internal times; logpdf scores the joint" begin
    using Distributions, Random

    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0.0, 1.0)
    dl = primary_censored(delay, pe; method = Latent())

    # rand produces the internal censored event times [primary, observed]; the
    # user does not pass the primary in.
    rng = MersenneTwister(3)
    x = rand(rng, dl)
    @test length(x) == 2
    p, y = x[1], x[2]
    @test insupport(pe, p)
    @test y >= p                      # observed = primary + non-negative delay

    # logpdf scores the joint of the sampler-owned [primary, observed].
    @test logpdf(dl, x) ≈ logpdf(pe, p) + logpdf(delay, y - p)
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

@testitem "BoundedPrimary with multiple secondaries (min bound)" begin
    using Distributions

    # bdbv case: admission's primary window is bounded above by the EARLIEST of
    # several downstream secondaries (e.g. min(day + 1, T_death, T_disch)).
    d = primary_censored(LogNormal(0.0, 1.0), Uniform(0.0, 1.0))

    # Two secondaries inside the window: the earliest (0.5) binds.
    bp = primary_prior(d, (0.8, 0.5))
    @test maximum(bp) == 0.5
    # Vector form gives the same bound.
    @test maximum(primary_prior(d, [0.8, 0.5])) == 0.5
    # Order does not matter.
    @test maximum(primary_prior(d, (0.5, 0.8))) == 0.5

    # A single secondary still works and matches the scalar call.
    @test maximum(primary_prior(d, (0.6,))) == maximum(primary_prior(d, 0.6))

    # When every secondary is beyond the window the full window is kept.
    @test maximum(primary_prior(d, (5.0, 3.0))) == 1.0

    # Jacobian-equals-uniform property holds with the combined min bound.
    bp2 = primary_prior(d, (0.8, 0.5, 0.9))   # binds at 0.5
    implicit = Uniform(0.0, 1.0)
    for t in range(1e-6, 0.5 - 1e-6; length = 5)
        @test logpdf(bp2, t) ≈ logpdf(implicit, t)
    end

    # Empty secondaries is an error; non-Uniform primary still errors.
    @test_throws ArgumentError primary_prior(d, Float64[])
    d_nonunif = primary_censored(
        LogNormal(0.0, 1.0), truncated(Normal(0.5, 0.2), 0, 1))
    @test_throws ArgumentError primary_prior(d_nonunif, (0.5, 0.6))
end

@testitem "Multi-secondary reproduces bdbv admit-with-two-secondaries" begin
    using Distributions

    # Reproduce the bdbv admission step: a daily primary window [day, day+1]
    # bounded above by both T_death and T_disch. The bounded prior must place
    # the primary in [day, min(day+1, T_death, T_disch)] and, with the Jacobian,
    # score as the implicit uniform-over-day-window prior.
    day = 3.0
    delay = LogNormal(0.0, 1.0)
    d = primary_censored(delay, Uniform(day, day + 1.0))

    T_death = day + 0.7
    T_disch = day + 0.4   # discharge is the earlier of the two -> binds
    bp = primary_prior(d, (T_death, T_disch))

    @test minimum(bp) == day
    @test maximum(bp) ≈ min(day + 1.0, T_death, T_disch)
    @test maximum(bp) ≈ day + 0.4

    # Jacobian-corrected logpdf equals the implicit uniform-over-day prior.
    implicit = Uniform(day, day + 1.0)
    for t in range(day + 1e-6, (day + 0.4) - 1e-6; length = 5)
        @test logpdf(bp, t) ≈ logpdf(implicit, t)
    end

    # Sampling stays within the combined bound.
    using Random
    rng = MersenneTwister(11)
    samples = rand(rng, bp, 5_000)
    @test all(day .<= samples .<= day + 0.4)
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
    dl = primary_censored(delay, pe; method = Latent())
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
    draws = [rand(rng, ic) for _ in 1:100]
    @test all(s -> s[2] == floor(s[2]), draws)   # observed snapped to interval
    @test all(s -> insupport(pe, s[1]), draws)   # primary in window
    @test any(s -> s[1] != floor(s[1]), draws)   # primary genuinely continuous

    # interval_censored on the default (marginal) distribution stays univariate
    dm = primary_censored(delay, pe)
    @test interval_censored(dm, 1.0) isa UnivariateDistribution
end
