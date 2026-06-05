# Tests for the DynamicPPL extension submodels (issue #88):
# `primary_censored_model`, `interval_censored_model`, and
# `double_interval_censored_model`. These exercise the extension that loads when
# `DynamicPPL`/`Turing` is present (the core package stays Turing-free), checking
# that the submodels compile, that the marginal submodel's log-density matches
# the direct `logpdf`, that the easy marginal-versus-latent switch works by
# flipping only the `mode` on `d`, and that a short NUTS run samples.

@testitem "primary_censored_model marginal matches logpdf" tags=[:turing] begin
    using CensoredDistributions
    using CensoredDistributions: MarginalPrimaryCensored
    using Distributions
    using DynamicPPL
    using DynamicPPL: to_submodel

    d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
    @test d isa MarginalPrimaryCensored

    @model function fit(d, y)
        inner ~ to_submodel(primary_censored_model(d, y), false)
    end

    # The marginal submodel exposes no latent, so the model log-joint at the
    # (empty) parameter set equals the pure marginal logpdf.
    for y in (1.0, 2.0, 4.0)
        m = fit(d, y)
        @test logjoint(m, (;)) ≈ logpdf(d, y)
    end
end

@testitem "primary_censored_model marginal weighting via weight" tags=[
    :turing] begin
    using CensoredDistributions
    using Distributions
    using DynamicPPL
    using DynamicPPL: to_submodel

    d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))

    @model function fit(d, y, w)
        inner ~ to_submodel(primary_censored_model(d, y; weight = w), false)
    end

    # Weighting goes through the `weight` distribution wrapper inside the `~`
    # (not an `@addlogprob!` hack), so the contribution is `w * logpdf`.
    for (y, w) in ((2.0, 7.0), (3.0, 3.0))
        m = fit(d, y, w)
        @test logjoint(m, (;)) ≈ w * logpdf(d, y)
    end
end

@testitem "primary_censored_model latent declares internal primary" tags=[
    :turing] begin
    using CensoredDistributions
    using CensoredDistributions: LatentPrimaryCensored
    using Distributions
    using DynamicPPL
    using DynamicPPL: to_submodel

    # Flip ONLY the mode to switch to latent; everything else is identical.
    d = primary_censored(
        LogNormal(1.5, 0.75), Uniform(0, 1); mode = Latent())
    @test d isa LatentPrimaryCensored

    @model function fit(d, y)
        inner ~ to_submodel(primary_censored_model(d, y), false)
    end

    m = fit(d, 2.0)
    vi = VarInfo(m)
    # The latent primary `p` lives inside the submodel (prefixed under `inner`)
    # and is a free variable the sampler owns; the user never declared it.
    varnames = string.(collect(keys(vi)))
    @test any(contains("p"), varnames)
end

@testitem "Easy marginal<->latent switch via mode only" tags=[:turing] begin
    using CensoredDistributions
    using CensoredDistributions: MarginalPrimaryCensored, LatentPrimaryCensored
    using Distributions
    using DynamicPPL
    using DynamicPPL: to_submodel

    # The SAME model definition, used with two distributions that differ only in
    # their `mode`, selects the marginal or latent submodel automatically. No
    # code in `fit` changes between the two.
    @model function fit(d, y)
        inner ~ to_submodel(primary_censored_model(d, y), false)
    end

    delay = LogNormal(1.5, 0.75)
    pe = Uniform(0, 1)

    d_marg = primary_censored(delay, pe)                  # Marginal default
    d_lat = primary_censored(delay, pe; mode = Latent())  # flip only the mode

    @test d_marg isa MarginalPrimaryCensored
    @test d_lat isa LatentPrimaryCensored

    # Marginal: no latent variables.
    @test isempty(keys(VarInfo(fit(d_marg, 2.0))))
    # Latent: a sampler-owned latent primary appears.
    @test !isempty(keys(VarInfo(fit(d_lat, 2.0))))
end

@testitem "interval_censored_model matches logpdf" tags=[:turing] begin
    using CensoredDistributions
    using Distributions
    using DynamicPPL
    using DynamicPPL: to_submodel

    d = interval_censored(Normal(5, 2), 1.0)

    @model function fit(d, y)
        inner ~ to_submodel(interval_censored_model(d, y), false)
    end

    for y in (3.0, 5.0, 7.0)
        @test logjoint(fit(d, y), (;)) ≈ logpdf(d, y)
    end

    # Weighting via the `weight` wrapper.
    @model function fitw(d, y, w)
        inner ~ to_submodel(interval_censored_model(d, y; weight = w), false)
    end
    @test logjoint(fitw(d, 5.0, 4.0), (;)) ≈ 4.0 * logpdf(d, 5.0)
end

@testitem "double_interval_censored_model matches logpdf" tags=[:turing] begin
    using CensoredDistributions
    using Distributions
    using DynamicPPL
    using DynamicPPL: to_submodel

    d = double_interval_censored(
        LogNormal(1.5, 0.75); upper = 10, interval = 1)

    @model function fit(d, y)
        inner ~ to_submodel(double_interval_censored_model(d, y), false)
    end

    for y in (1.0, 3.0, 6.0)
        @test logjoint(fit(d, y), (;)) ≈ logpdf(d, y)
    end

    # Weighting via the `weight` wrapper.
    @model function fitw(d, y, w)
        inner ~ to_submodel(
            double_interval_censored_model(d, y; weight = w), false)
    end
    @test logjoint(fitw(d, 3.0, 5.0), (;)) ≈ 5.0 * logpdf(d, 3.0)
end

@testitem "primary_censored_model NUTS smoke test (marginal)" tags=[
    :turing] begin
    using CensoredDistributions
    using Distributions
    using Turing
    using DynamicPPL: to_submodel, prefix
    using Random

    Random.seed!(42)
    d_gen = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
    ys = rand(d_gen, 60)

    @model function marg_fit(ys)
        mu ~ Normal(1.0, 1.0)
        sigma ~ truncated(Normal(0.5, 0.5); lower = 0.01)
        d = primary_censored(LogNormal(mu, sigma), Uniform(0, 1))
        for i in eachindex(ys)
            obs ~ to_submodel(
                prefix(primary_censored_model(d, ys[i]), Symbol("obs", i)),
                false)
        end
    end

    chain = sample(marg_fit(ys), NUTS(), 50; progress = false)
    @test all(isfinite, chain[:mu])
    @test all(chain[:sigma] .> 0)
    # Loose recovery check: the posterior mean should be near the truth.
    @test abs(mean(chain[:mu]) - 1.5) < 0.5
end

@testitem "primary_censored_model NUTS smoke test (latent)" tags=[:turing] begin
    using CensoredDistributions
    using Distributions
    using Turing
    using DynamicPPL: to_submodel, prefix
    using Random

    Random.seed!(42)
    d_gen = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
    ys = rand(d_gen, 40)

    # Same model body as the marginal smoke test, but the delay is built with
    # mode = Latent(): each observation gets its own sampler-owned primary.
    @model function lat_fit(ys)
        mu ~ Normal(1.0, 1.0)
        sigma ~ truncated(Normal(0.5, 0.5); lower = 0.01)
        d = primary_censored(
            LogNormal(mu, sigma), Uniform(0, 1); mode = Latent())
        for i in eachindex(ys)
            obs ~ to_submodel(
                prefix(primary_censored_model(d, ys[i]), Symbol("obs", i)),
                false)
        end
    end

    chain = sample(lat_fit(ys), NUTS(), 50; progress = false)
    @test all(isfinite, chain[:mu])
    @test all(chain[:sigma] .> 0)
end
