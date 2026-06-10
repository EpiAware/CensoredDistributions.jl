# Tests for `predict_events` (#350), the event-based draw convenience with two
# dispatch paths:
#   - raw distribution (Turing-free, core): forward-simulate full event paths
#     from a latent/composed distribution via `rand`;
#   - fitted Turing model (extension): recover the observed records'
#     integrated-out latent event times from a marginal-fit posterior.
# Turing/DynamicPPL/FlexiChains are test dependencies; the core path needs none
# of them.

# ===========================================================================
# Raw-distribution path (Turing-free, core)
# ===========================================================================

@testitem "predict_events: raw single draw is a full event path" begin
    using CensoredDistributions, Distributions, Random

    ld = latent(primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1)))
    path = predict_events(ld; rng = MersenneTwister(1))

    # The latent leaf draws `[primary, observed]`: the primary lies in the unit
    # window and the observed time exceeds it (a positive delay).
    @test length(path) == 2
    @test 0 <= path[1] <= 1
    @test path[2] > path[1]
end

@testitem "predict_events: raw n draws give n event paths" begin
    using CensoredDistributions, Distributions, Random

    ld = latent(primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)))
    paths = predict_events(ld, 200; rng = MersenneTwister(2))

    @test length(paths) == 200
    @test all(p -> 0 <= p[1] <= 1 && p[2] > p[1], paths)
end

@testitem "predict_events: raw is reproducible under a seeded rng" begin
    using CensoredDistributions, Distributions, Random

    ld = latent(primary_censored(LogNormal(1.0, 0.4), Uniform(0, 1)))
    a = predict_events(ld, 50; rng = MersenneTwister(99))
    b = predict_events(ld, 50; rng = MersenneTwister(99))
    @test a == b
end

@testitem "predict_events: raw over parameter draws rebuilds per draw" begin
    using CensoredDistributions, Distributions, Random

    build(p) = latent(
        primary_censored(LogNormal(p.mu, p.sigma), Uniform(0, 1)))
    draws = [(mu = 1.4, sigma = 0.5), (mu = 1.6, sigma = 0.4),
        (mu = 1.2, sigma = 0.6)]
    paths = predict_events(build, draws; rng = MersenneTwister(3))

    @test length(paths) == length(draws)
    @test all(p -> 0 <= p[1] <= 1 && p[2] > p[1], paths)
end

@testitem "predict_events: composer n draws give a labelled column table" begin
    using CensoredDistributions, Distributions, Random
    import Tables

    # A censored chain `[onset, admit, death]`; the batch draw labels each event
    # by `tree_event_names` so the result drops straight into the batch scoring
    # entry as a Tables.jl source.
    seq = Sequential(
        (primary_censored(Gamma(2.0, 1.5), Uniform(0, 1)),
            primary_censored(Gamma(1.5, 2.0), Uniform(0, 1))),
        (:onset_admit, :admit_death))

    n = 25
    sim = predict_events(seq, n; rng = MersenneTwister(20260610))

    # The result is a Tables.jl COLUMN table: a NamedTuple of equal-length
    # vectors keyed by the event names.
    @test Tables.istable(sim)
    @test !(sim isa AbstractVector)
    @test Tables.columnnames(sim) ==
          CensoredDistributions.tree_event_names(seq)
    @test sim isa NamedTuple
    cols = Tables.columns(sim)
    for name in Tables.columnnames(sim)
        @test length(Tables.getcolumn(cols, name)) == n
    end

    # Each record is a valid event path: onset >= 0, admit >= onset,
    # death >= admit.
    onset = Tables.getcolumn(cols, :onset)
    admit = Tables.getcolumn(cols, :admit)
    death = Tables.getcolumn(cols, :death)
    @test all(onset .>= 0)
    @test all(admit .>= onset)
    @test all(death .>= admit)
end

@testitem "predict_events: composer table is reproducible under a seeded rng" begin
    using CensoredDistributions, Distributions, Random
    import Tables

    seq = Sequential(
        (primary_censored(Gamma(2.0, 1.5), Uniform(0, 1)),
            primary_censored(Gamma(1.5, 2.0), Uniform(0, 1))),
        (:onset_admit, :admit_death))

    a = predict_events(seq, 30; rng = MersenneTwister(7))
    b = predict_events(seq, 30; rng = MersenneTwister(7))
    @test Tables.columntable(a) == Tables.columntable(b)
end

# ===========================================================================
# Fitted-model path (extension): observed-record latent recovery
# ===========================================================================

@testitem "predict_events: recovers observed records' latent primaries" begin
    using CensoredDistributions, Distributions, Random
    using Turing: Turing, NUTS, sample, @model, to_submodel
    using DynamicPPL: prefix, @varname
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

    events = predict_events(chain, latent_recovery(ys))

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

@testitem "predict_events: new-record posterior predictive via the model" begin
    using CensoredDistributions, Distributions, Random
    using Turing: Turing, NUTS, sample, @model, to_submodel
    using DynamicPPL: prefix
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
    # full event path per posterior draw.
    @model function new_records(nrec)
        mu ~ Normal(1.4, 0.5)
        d = latent(primary_censored(LogNormal(mu, 0.5), Uniform(0, 1)))
        for i in 1:nrec
            x ~ to_submodel(
                prefix(primary_censored_model(d, missing), Symbol(:new, i)),
                false)
        end
    end

    events = predict_events(chain, new_records(4))
    arr = Array(events)
    @test all(isfinite, arr)
    @test size(arr, 1) == size(Array(chain), 1)
end
