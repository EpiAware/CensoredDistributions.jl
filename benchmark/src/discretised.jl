let
    # Model specifically testing discretised distribution
    @model function discretised_model(y)
        # Simple priors for underlying distribution
        μ ~ Normal(5.0, 1.0)
        σ ~ truncated(Normal(1.0, 0.3), 0, Inf)

        # Create discretised distribution
        underlying_dist = Normal(μ, σ)
        interval = 1.0  # Fixed interval for now
        disc_dist = discretise(underlying_dist, interval)

        # Likelihood
        for i in eachindex(y)
            y[i] ~ disc_dist
        end
    end

    # Test data for discretised benchmarks (should be integers/multiples)
    test_data = [4.0, 5.0, 6.0, 5.0, 7.0]

    # Create model instance and benchmark suite
    model = discretised_model(test_data)
    suite["discretised"] = make_suite(model)
end
