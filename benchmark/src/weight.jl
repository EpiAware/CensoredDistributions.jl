let
    # Model specifically testing weighted distribution
    @model function weight_model(y, weights)
        # Simple priors
        μ ~ Normal(2.0, 1.0)
        σ ~ truncated(Normal(1.0, 0.3), 0, Inf)

        # Create base distribution
        base_dist = Normal(μ, σ)

        # Create weighted distributions
        weighted_dists = weight(base_dist, weights)

        # Likelihood (assuming y and weights have same length)
        for i in eachindex(y)
            y[i] ~ weighted_dists.v[i]  # Access individual weighted components
        end
    end

    # Test data for weighted benchmarks
    test_data = [1.8, 2.3, 2.1, 2.7, 2.0]
    test_weights = [0.5, 1.0, 1.5, 2.0, 0.8]

    # Create model instance and benchmark suite
    model = weight_model(test_data, test_weights)
    suite["weight"] = make_suite(model)
end
