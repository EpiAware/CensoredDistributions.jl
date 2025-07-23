let
    # Model specifically testing primarycensored distribution  
    @model function primarycensored_model(y)
        # Simple priors for LogNormal parameters
        μ ~ Normal(1.0, 0.5)
        σ ~ truncated(Normal(0.5, 0.2), 0, Inf)

        # Create primarycensored distribution
        delay_dist = LogNormal(μ, σ)
        primary_dist = Uniform(0, 1)
        censored_dist = primarycensored(delay_dist, primary_dist)

        # Likelihood
        for i in eachindex(y)
            y[i] ~ censored_dist
        end
    end

    # Test data for primarycensored benchmarks
    test_data = [1.2, 2.1, 1.8, 2.5, 1.9]

    # Create model instance and benchmark suite
    model = primarycensored_model(test_data)
    suite["primarycensored"] = make_suite(model)
end
