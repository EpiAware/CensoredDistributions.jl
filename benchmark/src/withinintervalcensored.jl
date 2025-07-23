let
    # Model specifically testing within_interval_censored distribution
    @model function withininterval_model(y)
        # Simple priors
        μ ~ Normal(3.0, 1.0)
        σ ~ truncated(Normal(1.0, 0.3), 0, Inf)
        
        # Create base distribution
        base_dist = LogNormal(μ, σ)
        
        # Create within interval censored distribution
        lower_bound = 1.0
        upper_bound = 10.0
        censored_dist = within_interval_censored(base_dist, lower_bound, upper_bound)
        
        # Likelihood
        for i in eachindex(y)
            y[i] ~ censored_dist
        end
    end

    # Test data for within interval censored benchmarks
    test_data = [2.3, 4.1, 3.8, 5.2, 4.9]
    
    # Create model instance and benchmark suite
    model = withininterval_model(test_data)
    suite["withinintervalcensored"] = make_suite(model)
end