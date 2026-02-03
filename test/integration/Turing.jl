@testitem "Turing.jl prior sampling" tags=[:turing] begin
    using Turing
    using DynamicPPL
    using Distributions
    using Random

    # Define the latent delay distribution submodel
    @model function latent_delay_dist()
        mu ~ Normal(1.0, 2.0)
        sigma ~ truncated(Normal(0.5, 1); lower = 0.0)
        return LogNormal(mu, sigma)
    end

    Random.seed!(123)

    # Sample from the latent delay distribution prior
    samples = sample(latent_delay_dist(), Prior(), 100; progress = false)

    # Test that sampling succeeds
    @test size(samples, 1) == 100

    # Test that samples are reasonable
    mu_samples = samples[:mu]
    sigma_samples = samples[:sigma]

    @test all(isfinite.(mu_samples))
    @test all(sigma_samples .> 0)
end

@testitem "Turing.jl naive model sampling" tags=[:turing] begin
    using Turing
    using DynamicPPL
    using Distributions
    using Random

    @model function latent_delay_dist()
        mu ~ Normal(1.0, 2.0)
        sigma ~ truncated(Normal(0.5, 1); lower = 0.0)
        return LogNormal(mu, sigma)
    end

    @model function naive_model()
        dist ~ to_submodel(latent_delay_dist())
        obs ~ weight(dist)
    end

    Random.seed!(789)

    # Generate simple test data
    test_obs = [2.0, 3.0, 4.0, 5.0, 3.5, 2.5]
    conditioned_model = condition(
        naive_model(),
        obs = (values = test_obs .+ 1e-6, weights = ones(length(test_obs)))
    )

    # Test MCMC sampling (short chain for speed)
    chain = sample(conditioned_model, NUTS(), 100; progress = false)

    @test size(chain, 1) == 100

    # Check that the parameter names exist in the chain
    param_names = names(chain)
    @test Symbol("dist.mu") in param_names
    @test Symbol("dist.sigma") in param_names

    # Test that samples are reasonable
    mu_samples = Array(chain[Symbol("dist.mu")])
    sigma_samples = Array(chain[Symbol("dist.sigma")])

    @test all(isfinite.(mu_samples))
    @test all(sigma_samples .> 0)
end

@testitem "Turing.jl interval-only model sampling" tags=[:turing] begin
    using Turing
    using DynamicPPL
    using Distributions
    using Random

    @model function latent_delay_dist()
        mu ~ Normal(1.0, 2.0)
        sigma ~ truncated(Normal(0.5, 1); lower = 0.0)
        return LogNormal(mu, sigma)
    end

    @model function interval_only_model(swindow_bounds, obs_time_bounds)
        swindows ~ arraydist([Uniform(sw[1], sw[2]) for sw in swindow_bounds])
        obs_times ~ arraydist([Uniform(ot[1], ot[2]) for ot in obs_time_bounds])

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
    chain = sample(conditioned_model, NUTS(), 100; progress = false)

    @test size(chain, 1) == 100

    # Check that the parameter names exist in the chain
    param_names = names(chain)
    @test Symbol("dist.mu") in param_names
    @test Symbol("dist.sigma") in param_names
end

@testitem "Turing.jl double censored model sampling" tags=[:turing] begin
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
        pwindows ~ arraydist([DiscreteUniform(pw[1], pw[2]) for pw in pwindow_bounds])
        swindows ~ arraydist([DiscreteUniform(sw[1], sw[2]) for sw in swindow_bounds])
        obs_times ~ arraydist([DiscreteUniform(ot[1], ot[2]) for ot in obs_time_bounds])

        dist ~ to_submodel(latent_delay_dist())

        pcens_dists = map(pwindows, obs_times, swindows) do pw, D, sw
            pe = Uniform(0, pw)
            double_interval_censored(dist; primary_event = pe, upper = D, interval = sw)
        end

        obs ~ weight(pcens_dists)
    end

    Random.seed!(333)

    # Create test data
    n = 10
    pwindow_bounds = fill((1, 3), n)
    swindow_bounds = fill((1, 3), n)
    obs_time_bounds = fill((8, 12), n)
    test_obs = [2, 3, 4, 3, 5, 2, 4, 3, 5, 6]
    test_pwindows = [2, 1, 3, 2, 1, 2, 1, 3, 2, 1]
    test_swindows = [2, 1, 2, 3, 1, 2, 1, 2, 3, 1]
    test_obs_times = [10, 9, 11, 10, 12, 10, 9, 11, 10, 12]

    # Create and condition the model
    model = double_censored_model(pwindow_bounds, swindow_bounds, obs_time_bounds)
    conditioned_model = fix(
        model,
        (
            @varname(pwindows) => test_pwindows,
            @varname(swindows) => test_swindows,
            @varname(obs_times) => test_obs_times
        )
    )
    conditioned_model = condition(
        conditioned_model,
        obs = (values = test_obs, weights = ones(n))
    )

    # Test MCMC sampling (short chain for speed)
    chain = sample(conditioned_model, NUTS(), 100; progress = false)

    @test size(chain, 1) == 100

    # Check that the parameter names exist in the chain
    param_names = names(chain)
    @test Symbol("dist.mu") in param_names
    @test Symbol("dist.sigma") in param_names

    # Test that samples are reasonable
    mu_samples = Array(chain[Symbol("dist.mu")])
    sigma_samples = Array(chain[Symbol("dist.sigma")])

    @test all(isfinite.(mu_samples))
    @test all(sigma_samples .> 0)
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
        pwindows ~ arraydist([DiscreteUniform(pw[1], pw[2]) for pw in pwindow_bounds])
        swindows ~ arraydist([DiscreteUniform(sw[1], sw[2]) for sw in swindow_bounds])
        obs_times ~ arraydist([DiscreteUniform(ot[1], ot[2]) for ot in obs_time_bounds])

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
    @test length(simulated_data.obs) == n
    @test length(simulated_data.pwindows) == n
    @test length(simulated_data.swindows) == n
    @test length(simulated_data.obs_times) == n

    # Test that observations are reasonable
    @test all(simulated_data.obs .>= 0)
    @test all(simulated_data.pwindows .>= 1)
    @test all(simulated_data.pwindows .<= 3)
    @test all(simulated_data.swindows .>= 1)
    @test all(simulated_data.swindows .<= 3)
    @test all(simulated_data.obs_times .>= 8)
    @test all(simulated_data.obs_times .<= 12)
end

@testitem "Turing.jl double_interval_censored logpdf is valid" tags=[:turing] begin
    using Distributions
    using Random

    Random.seed!(555)

    # Test that double_interval_censored produces valid logpdf values
    # This verifies the distribution works correctly when used in Turing models
    delay_dist = LogNormal(1.5, 0.75)
    pwindow = 2
    swindow = 1
    obs_time = 10

    # Construct distribution as used in Turing models
    primary_event = Uniform(0, pwindow)
    dist = double_interval_censored(
        delay_dist; primary_event = primary_event, upper = obs_time, interval = swindow
    )

    # Test at a specific observation value
    test_obs = 3.0
    lp = logpdf(dist, test_obs)

    @test isfinite(lp)
    @test lp <= 0.0
end

@testitem "Turing.jl weighted observations" tags=[:turing] begin
    using Turing
    using DynamicPPL
    using Distributions
    using Random

    @model function latent_delay_dist()
        mu ~ Normal(1.0, 2.0)
        sigma ~ truncated(Normal(0.5, 1); lower = 0.0)
        return LogNormal(mu, sigma)
    end

    @model function naive_model()
        dist ~ to_submodel(latent_delay_dist())
        obs ~ weight(dist)
    end

    Random.seed!(666)

    # Test that weighted observations work correctly
    test_obs = [2.0, 3.0, 4.0]
    weights = [2.0, 1.0, 1.0]  # First observation weighted twice

    # Condition naive model with weights
    conditioned_model = condition(
        naive_model(),
        obs = (values = test_obs .+ 1e-6, weights = weights)
    )

    # Test MCMC sampling (verifies weighted likelihood works)
    chain = sample(conditioned_model, NUTS(), 100; progress = false)

    @test size(chain, 1) == 100

    # Check that the parameter names exist in the chain
    param_names = names(chain)
    @test Symbol("dist.mu") in param_names
    @test Symbol("dist.sigma") in param_names

    # Test that samples are reasonable
    mu_samples = Array(chain[Symbol("dist.mu")])
    sigma_samples = Array(chain[Symbol("dist.sigma")])

    @test all(isfinite.(mu_samples))
    @test all(sigma_samples .> 0)
end
