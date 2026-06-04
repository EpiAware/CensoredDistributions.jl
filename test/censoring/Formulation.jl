@testitem "Formulation: default is Marginal, behaviour unchanged" begin
    using Distributions

    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0.0, 1.0)

    d_default = primary_censored(delay, pe)
    d_marginal = primary_censored(delay, pe; formulation = Marginal())

    @test d_default.formulation isa Marginal
    # Default and explicit Marginal must give identical results
    for x in [1.0, 2.5, 5.0]
        @test cdf(d_default, x) == cdf(d_marginal, x)
        @test logpdf(d_default, x) == logpdf(d_marginal, x)
    end
end

@testitem "Formulation: Latent conditional cdf equals shifted delay cdf" begin
    using Distributions

    delay = LogNormal(1.5, 0.75)
    p = 0.4
    d = primary_censored(delay, Uniform(0.0, 1.0); formulation = Latent(p))

    @test d.formulation isa Latent
    @test d.formulation.p == p

    for x in [1.0, 2.5, 5.0]
        @test cdf(d, x) ≈ cdf(delay, x - p)
        @test logcdf(d, x) ≈ logcdf(delay, x - p)
        # pdf/logpdf cascade exactly (no quadrature, no finite diff)
        @test logpdf(d, x) ≈ logpdf(delay, x - p)
        @test pdf(d, x) ≈ pdf(delay, x - p)
    end
end

@testitem "Formulation: Latent works across delay families" begin
    using Distributions

    p = 0.3
    for delay in [Gamma(2.0, 1.5), LogNormal(1.0, 0.5), Weibull(2.0, 1.5),
        Exponential(1.5)]
        d = primary_censored(delay, Uniform(0.0, 1.0); formulation = Latent(p))
        for x in [1.0, 2.0, 4.0]
            @test logpdf(d, x) ≈ logpdf(delay, x - p)
            @test cdf(d, x) ≈ cdf(delay, x - p)
        end
    end
end

@testitem "Formulation: integrating Latent over prior recovers Marginal" begin
    using Distributions

    # Acceptance test: ∫ pdf_delay(x - p) * pdf_prior(p) dp == marginal pdf(x).
    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0.0, 1.0)
    d_marg = primary_censored(delay, pe)
    prior = primary_prior(d_marg)

    function integrate_latent(x; n = 40_000)
        ps = range(minimum(prior), maximum(prior); length = n)
        vals = map(ps) do p
            dl = primary_censored(delay, pe; formulation = Latent(p))
            pdf(dl, x) * pdf(prior, p)
        end
        # trapezoidal rule
        return sum((vals[1:(end - 1)] .+ vals[2:end]) ./ 2) * step(ps)
    end

    for x in [1.0, 2.5, 5.0]
        @test isapprox(integrate_latent(x), pdf(d_marg, x); rtol = 1e-3)
    end
end

@testitem "Formulation: rand and support shared with Marginal" begin
    using Distributions, Random

    delay = LogNormal(1.0, 0.5)
    pe = Uniform(0.0, 1.0)
    d_marg = primary_censored(delay, pe)
    d_lat = primary_censored(delay, pe; formulation = Latent(0.5))

    # Support, params and rng draw machinery are shared (formulation only
    # specialises the cdf/logpdf path).
    @test minimum(d_lat) == minimum(d_marg)
    @test maximum(d_lat) == maximum(d_marg)
    @test params(d_lat) == params(d_marg)
    @test insupport(d_lat, 2.0) == insupport(d_marg, 2.0)

    rng = MersenneTwister(1)
    @test rand(copy(rng), d_lat) == rand(copy(rng), d_marg)
end

@testitem "Formulation: double_interval_censored forwards formulation" begin
    using Distributions

    delay = LogNormal(1.5, 0.75)
    p = 0.3

    dic_marg = double_interval_censored(delay; upper = 10, interval = 1)
    dic_lat = double_interval_censored(
        delay; upper = 10, interval = 1, formulation = Latent(p))

    # The two render different likelihoods (Latent conditions on p)
    @test logpdf(dic_lat, 3.0) != logpdf(dic_marg, 3.0)
    @test isfinite(logpdf(dic_lat, 3.0))
end
