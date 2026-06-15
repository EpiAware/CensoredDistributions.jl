# Event-based simulation and posterior event recovery now use the standard
# entry points (the `predict_events` convenience was dropped, #425):
#   - forward-simulating full event paths: `rand(latent(d))` (a single labelled
#     event record) and a comprehension `[rand(latent(d)) for _ in 1:n]` for a
#     batch;
#   - recovering observed records' integrated-out latent event times from a
#     marginal-fit posterior: `DynamicPPL.predict(model, chain)` directly.
# Turing/DynamicPPL/FlexiChains are test dependencies; the simulation path needs
# none of them.

# ===========================================================================
# Forward simulation via `rand(latent(d))` (Turing-free, core)
# ===========================================================================

@testitem "rand(latent(d)): single draw is a labelled event record" begin
    using CensoredDistributions, Distributions, Random

    ld = latent(primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1)))
    path = rand(MersenneTwister(1), ld)

    # The latent leaf draws the labelled record `(primary, observed)`: the
    # primary lies in the unit window and the observed time exceeds it.
    @test path isa NamedTuple
    @test keys(path) == (:primary, :observed)
    @test 0 <= path.primary <= 1
    @test path.observed > path.primary
end

@testitem "rand(latent(d)): a batch of draws via a comprehension" begin
    using CensoredDistributions, Distributions, Random

    ld = latent(primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)))
    rng = MersenneTwister(2)
    paths = [rand(rng, ld) for _ in 1:200]

    @test length(paths) == 200
    @test all(p -> 0 <= p.primary <= 1 && p.observed > p.primary, paths)
end

@testitem "rand(latent(d)): reproducible under a seeded rng" begin
    using CensoredDistributions, Distributions, Random

    ld = latent(primary_censored(LogNormal(1.0, 0.4), Uniform(0, 1)))
    a = [rand(MersenneTwister(99), ld) for _ in 1:1]
    b = [rand(MersenneTwister(99), ld) for _ in 1:1]
    @test a == b
end

@testitem "rand(latent(d)): rebuild per parameter set, simulate each" begin
    using CensoredDistributions, Distributions, Random

    build(p) = latent(
        primary_censored(LogNormal(p.mu, p.sigma), Uniform(0, 1)))
    draws = [(mu = 1.4, sigma = 0.5), (mu = 1.6, sigma = 0.4),
        (mu = 1.2, sigma = 0.6)]
    rng = MersenneTwister(3)
    paths = [rand(rng, build(p)) for p in draws]

    @test length(paths) == length(draws)
    @test all(p -> 0 <= p.primary <= 1 && p.observed > p.primary, paths)
end

# ===========================================================================
# Posterior event recovery via `DynamicPPL.predict` (extension territory)
# ===========================================================================

@testitem "DynamicPPL.predict recovers observed records' latent primaries" begin
    using CensoredDistributions, Distributions, Random
    using Turing: Turing, NUTS, sample, @model, to_submodel
    using DynamicPPL: DynamicPPL, predict, prefix, @varname
    using FlexiChains: Parameter

    rng = Random.MersenneTwister(42)
    truth = LogNormal(1.4, 0.5)
    n = 15
    primaries = rand(rng, Uniform(0, 1), n)
    ys = primaries .+ rand(rng, truth, n)

    # Fit the efficient MARGINAL form (no latent dims).
    @model function fit_marginal(y)
        mu ~ Normal(1.4, 0.5)
        sigma ~ truncated(Normal(0.5, 0.3); lower = 0.05)
        d = primary_censored(LogNormal(mu, sigma), Uniform(0, 1))
        for i in eachindex(y)
            obs ~ to_submodel(
                prefix(primary_censored_model(d, y[i]), Symbol(:rec, i)),
                false)
        end
    end
    chain = sample(rng, fit_marginal(ys), NUTS(), 40; progress = false)

    # The LATENT form: same parameters, latent-wrapped node, observed `y`.
    @model function latent_recovery(y)
        mu ~ Normal(1.4, 0.5)
        sigma ~ truncated(Normal(0.5, 0.3); lower = 0.05)
        d = latent(primary_censored(LogNormal(mu, sigma), Uniform(0, 1)))
        for i in eachindex(y)
            x ~ to_submodel(
                prefix(primary_censored_model(d, y[i]), Symbol(:rec, i)),
                false)
        end
    end

    # Posterior event recovery is `DynamicPPL.predict(model, chain)` directly.
    events = predict(latent_recovery(ys), chain)

    # The recovered variables are the per-record latent primaries `rec_i.p`
    # (only `p` is sampled; the observed `y` is conditioned), one per record.
    # Consistency with the marginal-equals-latent equivalence (#301): every
    # recovered primary lies in the unit window and reproduces the observed
    # delay gap as a positive value (`y_i - p_i > 0`).
    for i in 1:n
        p_i = vec(events[Parameter(@varname($(Symbol("rec", i)).p))])
        @test all(0 .<= p_i .<= 1)
        @test all(ys[i] .- p_i .> 0)
    end
end

@testitem "DynamicPPL.predict samples new-record event paths" begin
    using CensoredDistributions, Distributions, Random
    using Turing: Turing, NUTS, sample, @model, to_submodel
    using DynamicPPL: DynamicPPL, predict, prefix
    using FlexiChains: Parameter

    rng = Random.MersenneTwister(7)
    truth = LogNormal(1.4, 0.5)
    n = 12
    ys = rand(rng, Uniform(0, 1), n) .+ rand(rng, truth, n)

    @model function fit_marginal(y)
        mu ~ Normal(1.4, 0.5)
        d = primary_censored(LogNormal(mu, 0.5), Uniform(0, 1))
        for i in eachindex(y)
            obs ~ to_submodel(
                prefix(primary_censored_model(d, y[i]), Symbol(:rec, i)),
                false)
        end
    end
    chain = sample(rng, fit_marginal(ys), NUTS(), 40; progress = false)

    # New-record predictive: a latent model with `missing` events samples the
    # full event path per posterior draw, recovered with `DynamicPPL.predict`.
    @model function new_records(nrec)
        mu ~ Normal(1.4, 0.5)
        d = latent(primary_censored(LogNormal(mu, 0.5), Uniform(0, 1)))
        for i in 1:nrec
            x ~ to_submodel(
                prefix(primary_censored_model(d, missing), Symbol(:new, i)),
                false)
        end
    end

    events = predict(new_records(4), chain)
    arr = Array(events)
    @test all(isfinite, arr)
    @test size(arr, 1) == size(Array(chain), 1)
end
