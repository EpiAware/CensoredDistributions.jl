@testitem "Turing.jl prior sampling" tags=[:turing] begin
    using Turing
    using DynamicPPL
    using Distributions
    using Random
    using FlexiChains
    using FlexiChains: Prefixed, niters  # conflict with MCMCChains re-exports

    # Define the latent delay distribution submodel
    @model function latent_delay_dist()
        mu ~ Normal(1.0, 2.0)
        sigma ~ truncated(Normal(0.5, 1); lower = 0.0)
        return LogNormal(mu, sigma)
    end

    Random.seed!(123)

    # Sample from the latent delay distribution prior
    samples = sample(
        latent_delay_dist(), Prior(), 100;
        chain_type = VNChain, progress = false
    )

    # Test that sampling succeeds
    @test niters(samples) == 100

    # Test that samples are reasonable
    mu_samples = samples[@varname(mu)]
    sigma_samples = samples[@varname(sigma)]

    @test all(isfinite.(mu_samples))
    @test all(sigma_samples .> 0)
end

@testitem "Turing.jl interval-only model sampling" tags=[:turing] begin
    using Turing
    using DynamicPPL
    using Distributions
    using Random
    using FlexiChains
    using FlexiChains: Prefixed, niters  # conflict with MCMCChains re-exports

    @model function latent_delay_dist()
        mu ~ Normal(1.0, 2.0)
        sigma ~ truncated(Normal(0.5, 1); lower = 0.0)
        return LogNormal(mu, sigma)
    end

    @model function interval_only_model(swindow_bounds, obs_time_bounds)
        swindows ~ product_distribution([Uniform(sw[1], sw[2]) for sw in swindow_bounds])
        obs_times ~ product_distribution([Uniform(ot[1], ot[2]) for ot in obs_time_bounds])

        dist ~ to_submodel(latent_delay_dist())

        icens_dists = map(obs_times, swindows) do D, sw
            truncated(interval_censored(dist, sw), upper = D)
        end
        obs ~ weight(icens_dists)
        return obs
    end

    Random.seed!(111)

    # Create test data
    n = 5
    swindow_bounds = fill((1, 3), n)
    obs_time_bounds = fill((8, 12), n)
    test_obs = [2, 3, 4, 3, 5]
    test_swindows = [2, 1, 2, 3, 1]
    test_obs_times = [10, 9, 11, 10, 12]

    # Create and condition the model
    model = interval_only_model(swindow_bounds, obs_time_bounds)
    conditioned_model = fix(
        model,
        (
            @varname(swindows) => test_swindows,
            @varname(obs_times) => test_obs_times
        )
    )
    conditioned_model = condition(
        conditioned_model,
        obs = (values = test_obs, weights = ones(n))
    )

    # Test MCMC sampling (short chain for speed)
    chain = sample(
        conditioned_model, NUTS(), 100;
        chain_type = VNChain, progress = false
    )

    @test niters(chain) == 100

    # Test that submodel-prefixed parameters are accessible via Prefixed
    mu_samples = chain[Prefixed(@varname(mu))]
    sigma_samples = chain[Prefixed(@varname(sigma))]

    @test all(isfinite.(mu_samples))
    @test all(sigma_samples .> 0)
end

@testitem "Turing.jl double censored model recovers the delay" tags=[:turing] begin
    using Turing
    using DynamicPPL
    using Distributions
    using Random
    using Statistics: mean
    using FlexiChains
    using FlexiChains: Prefixed, niters  # conflict with MCMCChains re-exports

    # A genuine recovery assertion over the full double-censored Turing path
    # (DiscreteUniform primary/secondary windows + per-record observation-time
    # truncation + a `weight`ed likelihood). The marginal-vs-logpdf and weight
    # scaling are covered exactly in `CensoredModels.jl`; this guards that the
    # end-to-end NUTS fit through the windowed, truncated, weighted likelihood
    # still returns the parameters its own simulation was drawn from.
    @model function latent_delay_dist()
        mu ~ Normal(1.0, 2.0)
        sigma ~ truncated(Normal(0.5, 1); lower = 0.0)
        return LogNormal(mu, sigma)
    end

    @model function double_censored_model(pwindow_bounds, swindow_bounds,
            obs_time_bounds)
        pwindows ~ product_distribution([DiscreteUniform(pw[1], pw[2])
                              for pw in pwindow_bounds])
        swindows ~ product_distribution([DiscreteUniform(sw[1], sw[2])
                              for sw in swindow_bounds])
        obs_times ~ product_distribution([DiscreteUniform(ot[1], ot[2])
                              for ot in obs_time_bounds])

        dist ~ to_submodel(latent_delay_dist())

        pcens_dists = map(pwindows, obs_times, swindows) do pw, D, sw
            pe = Uniform(0, pw)
            double_interval_censored(
                dist; primary_event = pe, upper = D, interval = sw)
        end

        obs ~ weight(pcens_dists)
    end

    Random.seed!(333)

    # Simulate from known truth, then fit and recover it.
    n = 200
    true_mu = 1.5
    true_sigma = 0.5
    pwindow_bounds = fill((1, 3), n)
    swindow_bounds = fill((1, 3), n)
    obs_time_bounds = fill((8, 15), n)

    base_model = double_censored_model(
        pwindow_bounds, swindow_bounds, obs_time_bounds)
    sim = fix(base_model,
        (@varname(dist.mu) => true_mu, @varname(dist.sigma) => true_sigma))
    simulated = rand(sim)

    conditioned_model = fix(
        base_model,
        (
            @varname(pwindows) => simulated[@varname(pwindows)],
            @varname(swindows) => simulated[@varname(swindows)],
            @varname(obs_times) => simulated[@varname(obs_times)]
        )
    )
    conditioned_model = condition(
        conditioned_model,
        obs = (values = simulated[@varname(obs)], weights = ones(n))
    )

    chain = sample(
        conditioned_model, NUTS(), 500;
        chain_type = VNChain, progress = false
    )

    @test niters(chain) == 500

    mu_samples = chain[Prefixed(@varname(mu))]
    sigma_samples = chain[Prefixed(@varname(sigma))]

    @test all(isfinite.(mu_samples))
    @test all(sigma_samples .> 0)
    # Recovery: posterior means sit near the simulated truth.
    @test isapprox(mean(mu_samples), true_mu; atol = 0.3)
    @test isapprox(mean(sigma_samples), true_sigma; atol = 0.3)
end

@testitem "Turing.jl simulation from double censored model" tags=[:turing] begin
    using Turing
    using DynamicPPL
    using Distributions
    using Random

    @model function latent_delay_dist()
        mu ~ Normal(1.0, 2.0)
        sigma ~ truncated(Normal(0.5, 1); lower = 0.0)
        return LogNormal(mu, sigma)
    end

    @model function double_censored_model(pwindow_bounds, swindow_bounds, obs_time_bounds)
        pwindows ~ product_distribution([DiscreteUniform(pw[1], pw[2])
                              for pw in pwindow_bounds])
        swindows ~ product_distribution([DiscreteUniform(sw[1], sw[2])
                              for sw in swindow_bounds])
        obs_times ~ product_distribution([DiscreteUniform(ot[1], ot[2])
                              for ot in obs_time_bounds])

        dist ~ to_submodel(latent_delay_dist())

        pcens_dists = map(pwindows, obs_times, swindows) do pw, D, sw
            pe = Uniform(0, pw)
            double_interval_censored(dist; primary_event = pe, upper = D, interval = sw)
        end

        obs ~ weight(pcens_dists)
    end

    Random.seed!(444)

    n = 50
    meanlog = 1.5
    sdlog = 0.75

    # Define bounds for each observation
    pwindow_bounds = fill((1, 3), n)
    swindow_bounds = fill((1, 3), n)
    obs_time_bounds = fill((8, 12), n)

    # Create the model and fix parameters to true values
    base_model = double_censored_model(pwindow_bounds, swindow_bounds, obs_time_bounds)
    simulation_model = fix(
        base_model,
        (
            @varname(dist.mu) => meanlog,
            @varname(dist.sigma) => sdlog
        )
    )

    # Sample from the model
    simulated_data = rand(simulation_model)

    # Test that data was generated
    @test length(simulated_data[@varname(obs)]) == n
    @test length(simulated_data[@varname(pwindows)]) == n
    @test length(simulated_data[@varname(swindows)]) == n
    @test length(simulated_data[@varname(obs_times)]) == n

    # Test that observations are reasonable
    @test all(simulated_data[@varname(obs)] .>= 0)
    @test all(simulated_data[@varname(pwindows)] .>= 1)
    @test all(simulated_data[@varname(pwindows)] .<= 3)
    @test all(simulated_data[@varname(swindows)] .>= 1)
    @test all(simulated_data[@varname(swindows)] .<= 3)
    @test all(simulated_data[@varname(obs_times)] .>= 8)
    @test all(simulated_data[@varname(obs_times)] .<= 12)
end
