# Tests for the DynamicPPL extension submodels: `primary_censored_model`,
# `interval_censored_model`, `double_interval_censored_model`. They check that
# the submodels compile, that the marginal submodel log-density equals the direct
# `logpdf`, that the `weight` and `origin` keywords behave, and that a short NUTS
# run samples. DynamicPPL/Turing are test dependencies; the core stays
# Turing-free.

@testitem "primary_censored_model: marginal == logpdf" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))

    @model demo(d, y) = obs ~ to_submodel(primary_censored_model(d, y))

    for y in (0.5, 2.0, 5.0)
        @test only(logjoint(demo(d, y), (;))) ≈ logpdf(d, y)
    end
end

@testitem "primary_censored_model: weight scales the contribution" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))

    @model function demo(d, y, w)
        obs ~ to_submodel(primary_censored_model(d, y; weight = w))
    end

    y = 2.0
    @test only(logjoint(demo(d, y, 3), (;))) ≈ 3 * logpdf(d, y)
    # `nothing` weight leaves the contribution unweighted.
    @model demo_nothing(d, y) = obs ~ to_submodel(
        primary_censored_model(d, y; weight = nothing))
    @test only(logjoint(demo_nothing(d, y), (;))) ≈ logpdf(d, y)
end

@testitem "primary_censored_model: latent samples p inside the model" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint, condition, VarInfo,
                      @varname

    d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
    ld = latent(d)

    # The latent submodel declares the primary `p` INSIDE the model; the user
    # never passes it. It appears in the model's variables (prefixed by the
    # submodel LHS `obs`).
    @model demo(ld, y) = obs ~ to_submodel(primary_censored_model(ld, y))

    vi = VarInfo(demo(ld, 3.0))
    pkeys = collect(keys(vi))
    @test any(vn -> occursin("p", string(vn)), pkeys)
    # `p` is sampled inside, not passed: the only model variable is the latent.
    @test length(pkeys) == 1

    # Conditioning on a value of the latent `p`, the log-density equals the
    # primary prior plus the conditional of the observed given that `p`.
    p, y = 0.4, 3.0
    conditioned = condition(demo(ld, y), (@varname(obs.p) => p))
    @test logjoint(conditioned, (;)) ≈
          logpdf(get_primary_event(d), p) +
          primary_conditional_logpdf(d, p, y)
end

@testitem "interval_censored_model: marginal == logpdf" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    d = interval_censored(LogNormal(1.5, 0.75), 1.0)

    @model demo(d, y) = obs ~ to_submodel(interval_censored_model(d, y))

    for y in (1.0, 2.0, 4.0)
        @test only(logjoint(demo(d, y), (;))) ≈ logpdf(d, y)
    end

    @model function demo_w(d, y, w)
        obs ~ to_submodel(interval_censored_model(d, y; weight = w))
    end
    @test only(logjoint(demo_w(d, 2.0, 5), (;))) ≈ 5 * logpdf(d, 2.0)
end

@testitem "double_interval_censored_model: marginal == logpdf" begin
    using CensoredDistributions, Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    d = double_interval_censored(
        LogNormal(1.5, 0.75); upper = 10, interval = 1)

    @model demo(d, y) = obs ~ to_submodel(double_interval_censored_model(d, y))

    for y in (1.0, 3.0, 6.0)
        @test only(logjoint(demo(d, y), (;))) ≈ logpdf(d, y)
    end

    @model function demo_w(d, y, w)
        obs ~ to_submodel(double_interval_censored_model(d, y; weight = w))
    end
    @test only(logjoint(demo_w(d, 3.0, 4), (;))) ≈ 4 * logpdf(d, 3.0)
end

@testitem "primary_censored_model: short NUTS run samples" begin
    using CensoredDistributions, Distributions, Random
    using Turing: Turing, NUTS, sample, @model, to_submodel, @addlogprob!

    rng = Random.MersenneTwister(1)
    truth = LogNormal(1.5, 0.5)
    n = 50
    samples = rand(rng, truth, n) .+ rand(rng, Uniform(0, 1), n)

    # A single distribution scores every record. The submodel exposes no latent
    # in the marginal case, so the records can be accumulated directly.
    @model function fit(samples)
        mu ~ Normal(1.5, 1.0)
        sigma ~ truncated(Normal(0.5, 0.5); lower = 0.05)
        d = primary_censored(LogNormal(mu, sigma), Uniform(0, 1))
        for s in samples
            @addlogprob! logpdf(d, s)
        end
    end

    chain = sample(rng, fit(samples), NUTS(), 100; progress = false)
    @test size(chain, 1) == 100
    @test all(isfinite, chain[:mu])
end

@testitem "primary_censored_model: submodel composes in a NUTS run" begin
    using CensoredDistributions, Distributions, Random
    using Turing: Turing, NUTS, sample, @model, to_submodel

    rng = Random.MersenneTwister(2)
    truth = LogNormal(1.5, 0.5)
    sample_val = rand(rng, truth) + rand(rng, Uniform(0, 1))

    # A single-record model exercising the submodel inside a sampled `@model`.
    @model function fit(y)
        mu ~ Normal(1.5, 1.0)
        sigma ~ truncated(Normal(0.5, 0.5); lower = 0.05)
        d = primary_censored(LogNormal(mu, sigma), Uniform(0, 1))
        obs ~ to_submodel(primary_censored_model(d, y))
    end

    chain = sample(rng, fit(sample_val), NUTS(), 100; progress = false)
    @test size(chain, 1) == 100
    @test all(isfinite, chain[:mu])
end

@testitem "primary_censored_model: latent flow samples in a NUTS run" begin
    using CensoredDistributions, Distributions, Random
    using Turing: Turing, NUTS, sample, @model, to_submodel
    using DynamicPPL: prefix

    rng = Random.MersenneTwister(3)
    truth = LogNormal(1.5, 0.5)
    n = 20
    samples = rand(rng, truth, n) .+ rand(rng, Uniform(0, 1), n)

    # Latent flow: each record's primary `p` is sampled inside its submodel, so
    # the sampler explores `mu`, `sigma`, and one latent `p` per record. Each
    # submodel is prefixed so the per-record latents get distinct names.
    @model function fit(samples)
        mu ~ Normal(1.5, 1.0)
        sigma ~ truncated(Normal(0.5, 0.5); lower = 0.05)
        d = latent(primary_censored(LogNormal(mu, sigma), Uniform(0, 1)))
        for i in eachindex(samples)
            x ~ to_submodel(
                prefix(primary_censored_model(d, samples[i]),
                    Symbol("rec", i)), false)
        end
    end

    chain = sample(rng, fit(samples), NUTS(), 100; progress = false)
    @test size(chain, 1) == 100
    @test all(isfinite, chain[:mu])
end
