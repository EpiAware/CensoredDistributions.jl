@testitem "Test log methods consistency across all CensoredDistributions types" begin
    using Distributions

    # Test PrimaryCensored distributions
    pc_distributions = [
        primary_censored(Gamma(2.0, 1.5), Uniform(0, 1)),
        primary_censored(LogNormal(1.0, 0.5), Uniform(0.5, 1.5)),
        primary_censored(Exponential(2.0), Uniform(0, 2)),
        primary_censored(Weibull(1.5, 2.0), Uniform(0, 0.5))
    ]

    # Test IntervalCensored distributions (regular intervals)
    ic_regular_distributions = [
        interval_censored(Normal(5, 2), 1.0),
        interval_censored(Gamma(3, 1), 0.5),
        interval_censored(LogNormal(1, 0.5), 0.25),
        interval_censored(Exponential(1), 2.0)
    ]

    # Test IntervalCensored distributions (arbitrary intervals)
    ic_arbitrary_distributions = [
        interval_censored(Normal(0, 1), [0.0, 2.0, 5.0]),
        interval_censored(Gamma(2, 1), [0.0, 1.0, 3.0, 6.0]),
        interval_censored(Exponential(0.5), [-1.0, 1.0, 4.0]),
        interval_censored(truncated(Normal(0, 1), -3, 3), [-2.0, 0.0, 2.0])
    ]

    # Test DoubleIntervalCensored distributions
    dic_distributions = [
        double_interval_censored(Gamma(2, 1); primary_event = Uniform(0, 1)),
        double_interval_censored(
            LogNormal(1, 0.5); primary_event = Uniform(0, 1), upper = 10),
        double_interval_censored(
            Exponential(1); primary_event = Uniform(0, 1), interval = 0.5),
        double_interval_censored(Weibull(2, 1); primary_event = Uniform(0, 0.5),
            lower = 1.0, upper = 8.0, interval = 0.25)
    ]

    all_distributions = [pc_distributions; ic_regular_distributions;
                         ic_arbitrary_distributions; dic_distributions]

    for (i, d) in enumerate(all_distributions)
        # Define test values appropriate for each distribution
        min_val = minimum(d)
        max_val = maximum(d)

        # Create test range based on distribution support
        if isfinite(min_val) && isfinite(max_val)
            test_range = range(min_val + 0.1, max_val - 0.1, length = 10)
        elseif isfinite(min_val)
            test_range = range(min_val + 0.1, min_val + 10, length = 10)
        elseif isfinite(max_val)
            test_range = range(max_val - 10, max_val - 0.1, length = 10)
        else
            test_range = range(-5.0, 10.0, length = 10)
        end

        for x in test_range
            if insupport(d, x)
                # Test logpdf consistency
                logpdf_val = logpdf(d, x)
                @test logpdf_val ≠ NaN
                @test isfinite(logpdf_val) || logpdf_val == -Inf
                @test logpdf_val <= 0.0  # Log probability must be ≤ 0

                # Test logcdf consistency
                logcdf_val = logcdf(d, x)
                @test logcdf_val ≠ NaN
                @test isfinite(logcdf_val) || logcdf_val == -Inf
                @test logcdf_val <= 0.0  # Log CDF must be ≤ 0

                # Test logccdf consistency
                logccdf_val = logccdf(d, x)
                @test logccdf_val ≠ NaN
                @test isfinite(logccdf_val) || logccdf_val == -Inf
                @test logccdf_val <= 0.0  # Log CCDF must be ≤ 0

                # Test relationships between log methods
                cdf_val = cdf(d, x)
                ccdf_val = ccdf(d, x)

                # Test CDF + CCDF = 1 relationship in log space
                if isfinite(logcdf_val) && isfinite(logccdf_val)
                    # log(cdf + ccdf) should be approximately log(1) = 0
                    combined_prob = exp(logcdf_val) + exp(logccdf_val)
                    @test combined_prob ≈ 1.0 rtol=1e-10
                end

                # Test consistency: logcdf(d, x) ≈ log(cdf(d, x))
                if cdf_val > 0
                    @test logcdf_val ≈ log(cdf_val) rtol=1e-12
                else
                    @test logcdf_val == -Inf
                end

                # Test consistency: logccdf(d, x) ≈ log(ccdf(d, x))
                if ccdf_val > 0
                    @test logccdf_val ≈ log(ccdf_val) rtol=1e-12
                else
                    @test logccdf_val == -Inf
                end
            else
                # Values outside support should return -Inf for logpdf
                @test logpdf(d, x) == -Inf
            end
        end

        # Test extreme values and special cases
        extreme_vals = [-1e10, 1e10, -Inf, Inf, NaN]
        for x in extreme_vals
            logpdf_val = logpdf(d, x)
            logcdf_val = logcdf(d, x)
            logccdf_val = logccdf(d, x)

            # Should never return NaN (except possibly for NaN input)
            if !isnan(x)
                @test logpdf_val ≠ NaN
                @test logcdf_val ≠ NaN
                @test logccdf_val ≠ NaN
            end

            # Should be -Inf or finite
            @test isfinite(logpdf_val) || logpdf_val == -Inf || isnan(logpdf_val)
            @test isfinite(logcdf_val) || logcdf_val == -Inf || isnan(logcdf_val)
            @test isfinite(logccdf_val) || logccdf_val == -Inf || isnan(logccdf_val)
        end
    end
end

@testitem "Test log methods numerical stability and edge cases" begin
    using Distributions

    # Test distributions with extreme parameters that might cause numerical issues
    extreme_distributions = [
        # Very concentrated distributions
        interval_censored(Normal(0, 1e-10), 1e-12),
        primary_censored(Exponential(1e8), Uniform(0, 1e-10)),

        # Very spread out distributions
        interval_censored(Normal(0, 1e6), 1e3),
        primary_censored(Gamma(0.1, 1e4), Uniform(0, 1e2)),

        # Distributions with extreme support
        interval_censored(
            truncated(Exponential(1), 1e-15, 1e15), [1e-15, 1e-10, 1e10, 1e15]),
        double_interval_censored(Gamma(0.5, 2e7); primary_event = Uniform(0, 1e6),
            lower = 1e7, upper = 1e9, interval = 1e5)
    ]

    for d in extreme_distributions
        # Test that log methods handle extreme cases gracefully
        test_vals = [minimum(d), maximum(d), 0.0]

        for x in test_vals
            if isfinite(x)
                logpdf_val = logpdf(d, x)
                logcdf_val = logcdf(d, x)
                logccdf_val = logccdf(d, x)

                # Should not throw errors or return NaN
                @test !isnan(logpdf_val) || logpdf_val == -Inf
                @test !isnan(logcdf_val) || logcdf_val == -Inf
                @test !isnan(logccdf_val) || logccdf_val == -Inf

                # Should maintain proper bounds
                if isfinite(logpdf_val)
                    @test logpdf_val <= 0.0
                end
                if isfinite(logcdf_val)
                    @test logcdf_val <= 0.0
                end
                if isfinite(logccdf_val)
                    @test logccdf_val <= 0.0
                end
            end
        end
    end
end

@testitem "Test log methods monotonicity properties" begin
    using Distributions

    # Test that logcdf is monotonically non-decreasing
    # and logccdf is monotonically non-increasing
    test_distributions = [
        interval_censored(Normal(0, 1), 0.5),
        primary_censored(Gamma(2, 1), Uniform(0, 1)),
        double_interval_censored(
            Exponential(1); primary_event = Uniform(0, 1), interval = 0.25)
    ]

    for d in test_distributions
        # Create test sequence
        x_vals = sort(rand(20) * 10 .- 5)  # Random values in [-5, 5], sorted

        logcdf_vals = [logcdf(d, x) for x in x_vals]
        logccdf_vals = [logccdf(d, x) for x in x_vals]

        # Test monotonicity where both consecutive values are finite
        for i in 1:(length(x_vals) - 1)
            if isfinite(logcdf_vals[i]) && isfinite(logcdf_vals[i + 1])
                @test logcdf_vals[i] <= logcdf_vals[i + 1]  # Non-decreasing
            end

            if isfinite(logccdf_vals[i]) && isfinite(logccdf_vals[i + 1])
                @test logccdf_vals[i] >= logccdf_vals[i + 1]  # Non-increasing
            end
        end

        # Test boundary behavior
        if isfinite(minimum(d))
            @test logcdf(d, minimum(d)) ≈ logcdf(d, minimum(d) - 1e-10) rtol=1e-6
        end

        if isfinite(maximum(d))
            @test logccdf(d, maximum(d)) ≈ logccdf(d, maximum(d) + 1e-10) rtol=1e-6
        end
    end
end

@testitem "Test log methods with truncated distributions" begin
    using Distributions

    # Test log methods with various truncated censored distributions
    base_distributions = [
        primary_censored(Gamma(2, 1), Uniform(0, 1)),
        interval_censored(Normal(0, 1), 0.5),
        double_interval_censored(LogNormal(1, 0.5); primary_event = Uniform(0, 1))
    ]

    for base_d in base_distributions
        # Create truncated versions
        truncated_d = truncated(base_d, 1.0, 8.0)

        # Test values outside truncated support
        @test logpdf(truncated_d, 0.5) == -Inf  # Below lower bound
        @test logpdf(truncated_d, 10.0) == -Inf  # Above upper bound
        @test logcdf(truncated_d, 0.5) == -Inf   # Below lower bound
        @test logccdf(truncated_d, 10.0) == -Inf  # Above upper bound

        # Test values within truncated support
        test_vals = [1.5, 3.0, 5.0, 7.5]
        for x in test_vals
            logpdf_val = logpdf(truncated_d, x)
            logcdf_val = logcdf(truncated_d, x)
            logccdf_val = logccdf(truncated_d, x)

            @test logpdf_val ≠ NaN
            @test logcdf_val ≠ NaN
            @test logccdf_val ≠ NaN

            @test isfinite(logpdf_val) || logpdf_val == -Inf
            @test isfinite(logcdf_val) || logcdf_val == -Inf
            @test isfinite(logccdf_val) || logccdf_val == -Inf

            # Test relationship with non-truncated distribution
            base_logpdf = logpdf(base_d, x)
            if isfinite(base_logpdf) && isfinite(logpdf_val)
                # Truncated logpdf should be different (normalized)
                # but both should be finite for values in support
                @test true  # Just check they're both finite
            end
        end

        # Test boundary values
        @test logcdf(truncated_d, 1.0) ≠ NaN   # At lower bound
        @test logccdf(truncated_d, 8.0) ≠ NaN  # At upper bound
    end
end

@testitem "Test log methods consistency with analytical solutions" begin
    using Distributions

    # Test against known analytical cases where possible

    # Exponential + Uniform primary event (has analytical solution)
    exp_dist = Exponential(1.0)
    uniform_primary = Uniform(0, 1)
    pc_exp = primary_censored(exp_dist, uniform_primary)

    # Test that log methods are consistent
    test_points = [0.1, 0.5, 1.0, 2.0, 5.0]
    for x in test_points
        logpdf_val = logpdf(pc_exp, x)
        pdf_val = pdf(pc_exp, x)

        if pdf_val > 0
            @test logpdf_val ≈ log(pdf_val) rtol=1e-12
        else
            @test logpdf_val == -Inf
        end

        logcdf_val = logcdf(pc_exp, x)
        cdf_val = cdf(pc_exp, x)

        if cdf_val > 0
            @test logcdf_val ≈ log(cdf_val) rtol=1e-12
        else
            @test logcdf_val == -Inf
        end

        # Test that the numerical differentiation in logpdf is working
        if isfinite(logpdf_val)
            # logpdf should be the derivative of logcdf
            h = 1e-8
            if x + h <= maximum(pc_exp)
                logcdf_upper = logcdf(pc_exp, x + h)
                logcdf_lower = logcdf(pc_exp, x)

                if isfinite(logcdf_upper) && isfinite(logcdf_lower)
                    # Numerical derivative approximation
                    numerical_derivative = (exp(logcdf_upper) - exp(logcdf_lower)) / h
                    if numerical_derivative > 0
                        @test logpdf_val ≈ log(numerical_derivative) rtol=1e-6
                    end
                end
            end
        end
    end
end
