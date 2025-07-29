# Simplified parameter recovery tests for distributions with analytical solutions
# Testing with and without forced numerical integration

@testmodule RecoveryTestSetup begin
    using CensoredDistributions
    using Distributions
    using Random
    using Statistics
    using Optimization
    using OptimizationOptimJL
    using Bijectors
    using SpecialFunctions
    using Test

    # Common test functions for reuse
    function setup_recovery_test(true_params, DistType, interval_width,
            n_samples; force_numeric = false, seed = 12345)
        Random.seed!(seed)

        # Create true distribution and generate data
        true_underlying = DistType(true_params...)
        true_primary = Uniform(0.0, 1.0)
        true_dist = primary_censored(
            true_underlying, true_primary; force_numeric = force_numeric)
        data = rand(true_dist, n_samples)

        # Create template with reasonable initial guess close to truth
        if DistType == Distributions.Gamma
            # Use method of moments estimate as initial guess
            sample_mean = mean(data)
            sample_var = var(data)
            init_shape = sample_mean^2 / sample_var
            init_scale = sample_var / sample_mean
            template_dist = interval_censored(Gamma(init_shape, init_scale), interval_width)
        elseif DistType == Distributions.LogNormal
            # Use log-transformed moments
            log_data = log.(max.(data, 1e-10))  # Avoid log(0)
            init_μ = mean(log_data)
            init_σ = std(log_data)
            template_dist = interval_censored(LogNormal(init_μ, init_σ), interval_width)
        elseif DistType == Distributions.Weibull
            # Simple initial guess
            sample_mean = mean(data)
            init_shape = 2.0  # Common shape parameter
            init_scale = sample_mean / SpecialFunctions.gamma(1 + 1 / init_shape)
            template_dist = interval_censored(
                Weibull(init_shape, init_scale), interval_width)
        else
            error("Unsupported distribution type: $DistType")
        end

        return data, template_dist, true_params
    end

    function test_parameter_recovery(
            fitted_params, true_params, param_names, tolerance)
        errors = Float64[]
        for i in 1:length(true_params)
            error = abs(fitted_params[i] - true_params[i]) / abs(true_params[i])
            push!(errors, error)
            @test error < tolerance
        end

        # Log results
        error_strs = ["$(param_names[i])_error=$(round(errors[i]*100, digits=1))%"
                      for i in 1:length(errors)]
        @info "Parameter recovery: " * join(error_strs, ", ")

        return errors
    end
end # @testmodule RecoveryTestSetup

# ===== GAMMA + UNIFORM =====

@testitem "Gamma+Uniform recovery - analytical" tags=[:optimization, :recovery] setup=[RecoveryTestSetup] begin
    using CensoredDistributions
    using Distributions
    using Random
    using Statistics
    using Optimization
    using OptimizationOptimJL
    using Bijectors

    # Test parameters
    true_params=(2.5, 1.8)  # shape, scale
    interval_width=0.15
    n_samples=1500

    # Generate data and setup
    (data,
        template_dist,
        true_params)=RecoveryTestSetup.setup_recovery_test(
        true_params, Gamma, interval_width, n_samples)

    # Fit using default settings
    fitted_dist=fit_mle(template_dist, data; intervals = interval_width)
    fitted_params=params(fitted_dist.dist)

    # Test recovery
    RecoveryTestSetup.test_parameter_recovery(
        fitted_params, true_params, ["shape", "scale"], 0.20)
end

@testitem "Gamma+Uniform recovery - forced numerical" tags=[:optimization, :recovery] setup=[RecoveryTestSetup] begin
    using CensoredDistributions
    using Distributions
    using Random
    using Statistics
    using Optimization
    using OptimizationOptimJL
    using Bijectors

    # Test parameters
    true_params=(2.0, 1.5)  # shape, scale
    interval_width=0.15
    n_samples=1200

    # Generate data with forced numerical integration
    (data,
        template_dist,
        true_params)=RecoveryTestSetup.setup_recovery_test(
        true_params, Gamma, interval_width, n_samples; force_numeric = true, seed = 12346)

    # Fit using default settings
    fitted_dist=fit_mle(template_dist, data; intervals = interval_width)
    fitted_params=params(fitted_dist.dist)

    # Test recovery (more tolerant for numerical)
    RecoveryTestSetup.test_parameter_recovery(
        fitted_params, true_params, ["shape", "scale"], 0.3)
end

# ===== LOGNORMAL + UNIFORM =====

@testitem "LogNormal+Uniform recovery - analytical" tags=[:optimization, :recovery] setup=[RecoveryTestSetup] begin
    using CensoredDistributions
    using Distributions
    using Random
    using Statistics
    using Optimization
    using OptimizationOptimJL
    using Bijectors

    # Test parameters
    true_params=(1.5, 0.8)  # μ, σ
    interval_width=0.12
    n_samples=1500

    # Generate data and setup
    (data,
        template_dist,
        true_params)=RecoveryTestSetup.setup_recovery_test(
        true_params, LogNormal, interval_width, n_samples; seed = 22345)

    # Fit using default settings
    fitted_dist=fit_mle(template_dist, data; intervals = interval_width)
    fitted_params=params(fitted_dist.dist)

    # Test recovery
    RecoveryTestSetup.test_parameter_recovery(fitted_params, true_params, ["μ", "σ"], 0.20)
end

@testitem "LogNormal+Uniform recovery - forced numerical" tags=[:optimization, :recovery] setup=[RecoveryTestSetup] begin
    using CensoredDistributions
    using Distributions
    using Random
    using Statistics
    using Optimization
    using OptimizationOptimJL
    using Bijectors

    # Test parameters
    true_params=(1.0, 0.6)  # μ, σ
    interval_width=0.15
    n_samples=1200

    # Generate data with forced numerical integration
    (data,
        template_dist,
        true_params)=RecoveryTestSetup.setup_recovery_test(
        true_params, LogNormal, interval_width,
        n_samples; force_numeric = true, seed = 22346)

    # Fit using default settings
    fitted_dist=fit_mle(template_dist, data; intervals = interval_width)
    fitted_params=params(fitted_dist.dist)

    # Test recovery (more tolerant for numerical)
    RecoveryTestSetup.test_parameter_recovery(fitted_params, true_params, ["μ", "σ"], 0.25)
end

# ===== WEIBULL + UNIFORM =====

@testitem "Weibull+Uniform recovery - analytical" tags=[:optimization, :recovery] setup=[RecoveryTestSetup] begin
    using CensoredDistributions
    using Distributions
    using Random
    using Statistics
    using Optimization
    using OptimizationOptimJL
    using Bijectors

    # Test parameters
    true_params=(2.0, 1.5)  # shape, scale
    interval_width=0.12
    n_samples=1500

    # Generate data and setup
    (data,
        template_dist,
        true_params)=RecoveryTestSetup.setup_recovery_test(
        true_params, Weibull, interval_width, n_samples; seed = 32345)

    # Fit using default settings
    fitted_dist=fit_mle(template_dist, data; intervals = interval_width)
    fitted_params=params(fitted_dist.dist)

    # Test recovery
    RecoveryTestSetup.test_parameter_recovery(
        fitted_params, true_params, ["shape", "scale"], 0.20)
end

@testitem "Weibull+Uniform recovery - forced numerical" tags=[:optimization, :recovery] setup=[RecoveryTestSetup] begin
    using CensoredDistributions
    using Distributions
    using Random
    using Statistics
    using Optimization
    using OptimizationOptimJL
    using Bijectors

    # Test parameters
    true_params=(1.8, 2.0)  # shape, scale
    interval_width=0.15
    n_samples=1200

    # Generate data with forced numerical integration
    (data,
        template_dist,
        true_params)=RecoveryTestSetup.setup_recovery_test(
        true_params, Weibull, interval_width, n_samples; force_numeric = true, seed = 32346)

    # Fit using default settings
    fitted_dist=fit_mle(template_dist, data; intervals = interval_width)
    fitted_params=params(fitted_dist.dist)

    # Test recovery (more tolerant for numerical)
    RecoveryTestSetup.test_parameter_recovery(
        fitted_params, true_params, ["shape", "scale"], 0.25)
end

# ===== SAMPLE SIZE EFFECTS =====

@testitem "Sample size scaling - Gamma+Uniform" tags=[:optimization, :recovery] setup=[RecoveryTestSetup] begin
    using CensoredDistributions
    using Distributions
    using Random
    using Statistics
    using Optimization
    using OptimizationOptimJL
    using Bijectors

    # Test parameters
    original_true_params=(2.0, 1.5)  # shape, scale
    interval_width=0.15
    sample_sizes=[500, 1000, 2000]

    shape_errors=Float64[]
    scale_errors=Float64[]

    for (i, n_samples) in enumerate(sample_sizes)
        # Generate data and setup (different seed for each sample size)
        (data,
            template_dist,
            _)=RecoveryTestSetup.setup_recovery_test(
            original_true_params, Gamma, interval_width, n_samples; seed = 52345+i)

        # Fit using default settings
        fitted_dist=fit_mle(template_dist, data; intervals = interval_width)
        fitted_params=params(fitted_dist.dist)

        # Calculate errors
        shape_error=abs(fitted_params[1]-original_true_params[1])/
        abs(original_true_params[1])
        scale_error=abs(fitted_params[2]-original_true_params[2])/
        abs(original_true_params[2])

        push!(shape_errors, shape_error)
        push!(scale_errors, scale_error)

        @info "Sample size $n_samples: shape_error=$(round(shape_error*100, digits=1))%, scale_error=$(round(scale_error*100, digits=1))%"
    end

    # Errors should generally decrease with sample size (allow some variability)
    @test shape_errors[end] <= shape_errors[1] * 1.5
    @test scale_errors[end] <= scale_errors[1] * 1.5
end

# ===== WEIGHTED FITTING =====

@testitem "Weighted fitting - LogNormal+Uniform" tags=[:optimization, :recovery] setup=[RecoveryTestSetup] begin
    using CensoredDistributions
    using Distributions
    using Random
    using Statistics
    using Optimization
    using OptimizationOptimJL
    using Bijectors

    # Test parameters
    true_params=(0.8, 0.6)  # μ, σ
    interval_width=0.12
    n_samples=1200

    # Generate data and setup
    (data,
        template_dist,
        true_params)=RecoveryTestSetup.setup_recovery_test(
        true_params, LogNormal, interval_width, n_samples; seed = 62345)

    # Test unweighted
    fitted_unweighted=fit_mle(template_dist, data; intervals = interval_width)
    params_unweighted=params(fitted_unweighted.dist)

    # Test with uniform weights (should be similar)
    uniform_weights=ones(length(data))
    fitted_uniform_weighted=fit_mle(
        template_dist, data; weights = uniform_weights, intervals = interval_width)
    params_uniform_weighted=params(fitted_uniform_weighted.dist)

    # Test with non-uniform weights
    data_mean=mean(data)
    weights=exp.(-0.1*abs.(data .- data_mean))
    weights./=sum(weights)/length(weights)  # Normalize

    fitted_weighted=fit_mle(
        template_dist, data; weights = weights, intervals = interval_width)
    params_weighted=params(fitted_weighted.dist)

    # Unweighted and uniform weighted should be very similar
    @test abs(params_unweighted[1] - params_uniform_weighted[1]) < 0.05
    @test abs(params_unweighted[2] - params_uniform_weighted[2]) < 0.05

    # Test parameter recovery for both
    @info "Unweighted fitting:"
    RecoveryTestSetup.test_parameter_recovery(
        params_unweighted, true_params, ["μ", "σ"], 0.25)

    @info "Weighted fitting:"
    RecoveryTestSetup.test_parameter_recovery(
        params_weighted, true_params, ["μ", "σ"], 0.30)
end

# ===== DIFFERENT OPTIMIZERS =====

@testitem "Multiple optimizers - Gamma+Uniform" tags=[:optimization, :recovery] setup=[RecoveryTestSetup] begin
    using CensoredDistributions
    using Distributions
    using Random
    using Statistics
    using Optimization
    using OptimizationOptimJL
    using Bijectors

    # Test parameters
    true_params=(2.2, 1.6)  # shape, scale
    interval_width=0.15
    n_samples=1200

    # Generate data and setup
    (data,
        template_dist,
        true_params)=RecoveryTestSetup.setup_recovery_test(
        true_params, Gamma, interval_width, n_samples; seed = 42345)

    # Test different optimizers
    optimizers=[
        (OptimizationOptimJL.LBFGS(), "LBFGS"),
        (OptimizationOptimJL.BFGS(), "BFGS")
    ]

    for (optimizer, name) in optimizers
        fitted_dist=fit_mle(
            template_dist, data; intervals = interval_width, optimizer = optimizer)
        fitted_params=params(fitted_dist.dist)

        @info "Optimizer: $name"
        RecoveryTestSetup.test_parameter_recovery(
            fitted_params, true_params, ["shape", "scale"], 0.25)
    end
end
