# A genuinely-sampled latent leaf fit, end to end through the composer toolkit.
#
# The marginal double-censored leaf integrates the within-window primary out
# inside `logpdf`; its latent twin (`latent(leaf)`) samples it
# (`p ~ get_primary_event(d)`) and scores the secondary as
# `logpdf(PrimaryConditional(d, p), y)`. The two forms describe one model and
# agree in expectation, so the sampled-primary fit must recover the parameters
# its own simulation was drawn from. This guards the leaf latent path through
# `composed_parameters_model` + `latent` + `param_draws`, distinct from the
# density-equivalence checks in `latent_marginal_equivalence.jl`.

@testitem "latent leaf fit recovers the sampled-primary delay" tags=[:turing] begin
    using CensoredDistributions, Distributions, Turing, Random
    using DynamicPPL: to_submodel, InitFromParams, InitFromPrior
    using FlexiChains: VNChain, parameters
    using Statistics: mean
    using ADTypes: AutoForwardDiff

    meanlog = 1.5
    sdlog = 0.75
    leaf = double_interval_censored(LogNormal(meanlog, sdlog);
        primary_event = Uniform(0, 1), interval = 1.0)
    priors = build_priors(params_table(leaf))

    # Untruncated double-censored reports; keep reports of at least one day so
    # every record's conditional is in support for any primary in (0, 1).
    rng = MersenneTwister(123)
    template = sequential(:onset_report => leaf)
    paths = rand(rng, template, fill((onset = missing, report = missing), 2000))
    reports = filter(r -> r >= 1, [p.report for p in paths])
    n = 50
    data = [(report = r,) for r in reports[1:n]]

    # `latent(delays)` samples each record's within-window primary and scores
    # the secondary conditional on it: the genuinely-sampled latent form.
    @model function latent_fit(template, priors, data)
        delays ~ to_submodel(composed_parameters_model(template, priors))
        obs ~ to_submodel(composed_distribution_model(latent(delays), data))
    end

    # Pin the two delay parameters at a neutral start; the per-record primaries
    # initialise from their Uniform(0, 1) prior, so the joint starts finite.
    init = InitFromParams((delays = (mu = 1.0, sigma = 1.0),), InitFromPrior())

    chain = sample(Xoshiro(1),
        latent_fit(leaf, priors, data),
        NUTS(0.8; adtype = AutoForwardDiff()), 250;
        chain_type = VNChain, progress = false, initial_params = init)

    # One sampled primary per record (`obs.recN.p`) sits alongside the two delay
    # parameters, so the primary is genuinely sampled, not integrated out.
    @test length(collect(parameters(chain))) == n + 2

    draws = param_draws(leaf, chain; prefix = :delays)
    mu_draws = [d.mu for d in draws]
    sigma_draws = [d.sigma for d in draws]
    @test all(isfinite, mu_draws)
    @test all(>(0), sigma_draws)
    # Recovery: posterior means sit near the simulated truth, so the latent form
    # agrees with the marginal in expectation.
    @test isapprox(mean(mu_draws), meanlog; atol = 0.3)
    @test isapprox(mean(sigma_draws), sdlog; atol = 0.3)

    # The composed interface reads the fitted delay back off the same leaf.
    recovered = params_table(update(leaf, chain; prefix = :delays))
    @test length(recovered.value) == 2
    @test all(isfinite, recovered.value)
end
