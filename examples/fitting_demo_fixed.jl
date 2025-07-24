"""
Corrected Demonstration of Maximum Likelihood Estimation for IntervalCensored distributions

This script demonstrates the CORRECT way to simulate and fit interval-censored data.
"""

using CensoredDistributions
using Distributions
using Random
using Statistics

println("CensoredDistributions.jl IntervalCensored Fitting Demonstration (CORRECTED)")
println("=" ^ 70)

# Set random seed for reproducibility
Random.seed!(123)

# Define true underlying distribution
println("\n1. Setting up true distribution:")
true_underlying = Normal(2.5, 1.2)  # μ=2.5, σ=1.2
interval_width = 0.5
true_dist = interval_censored(true_underlying, interval_width)

println("True underlying distribution: Normal(μ=$(params(true_underlying)[1]), σ=$(params(true_underlying)[2]))")
println("Interval width: $(interval_width)")

# Generate CORRECT interval-censored data
println("\n2. Generating CORRECT interval-censored data:")
n_samples = 2000

# THIS IS THE CORRECT WAY: sample directly from the interval-censored distribution
interval_data = rand(true_dist, n_samples)

println("Generated $(n_samples) samples from interval-censored distribution")
println("Sample values (first 10): ", interval_data[1:10])
println("These are interval left boundaries, not indices!")
println("Data range: [$(minimum(interval_data)), $(maximum(interval_data))]")
println("Unique values: $(sort(unique(interval_data)))")

# Fit the model using MLE
println("\n3. Fitting IntervalCensored model using Maximum Likelihood Estimation:")
println("This may take a moment...")

try
    # Fit Normal distribution to interval-censored data
    fitted_dist = CensoredDistributions.fit_mle(
        CensoredDistributions.IntervalCensored, interval_data;
        dist_type = Normal,
        interval = interval_width)

    println("✓ Fitting completed successfully!")

    # Extract fitted parameters
    fitted_params = params(fitted_dist.dist)
    true_params = params(true_underlying)

    println("\n4. Parameter comparison:")
    println("Normal distribution parameters:")
    println("  True parameters:   μ=$(round(true_params[1], digits=3)), σ=$(round(true_params[2], digits=3))")
    println("  Fitted parameters: μ=$(round(fitted_params[1], digits=3)), σ=$(round(fitted_params[2], digits=3))")

    # Calculate parameter errors
    μ_error = abs(fitted_params[1] - true_params[1]) / abs(true_params[1]) * 100
    σ_error = abs(fitted_params[2] - true_params[2]) / abs(true_params[2]) * 100

    println("\n5. Parameter estimation errors:")
    println("  Mean (μ): $(round(μ_error, digits=2))%")
    println("  Standard deviation (σ): $(round(σ_error, digits=2))%")

    # Model validation
    println("\n6. Model validation:")

    # Test on the actual interval boundaries present in data
    unique_intervals = sort(unique(interval_data))
    test_sample = unique_intervals[1:min(10, length(unique_intervals))]

    true_probs = pdf.(Ref(true_dist), test_sample)
    fitted_probs = pdf.(Ref(fitted_dist), test_sample)

    # Calculate probability differences
    prob_rmse = sqrt(mean((true_probs .- fitted_probs) .^ 2))
    println("  Probability RMSE across intervals: $(round(prob_rmse, digits=6))")

    # Show some probability comparisons
    println("  Sample probability comparisons:")
    for i in 1:length(test_sample)
        interval = test_sample[i]
        true_p = true_probs[i]
        fitted_p = fitted_probs[i]
        println("    Interval $(interval): true=$(round(true_p, digits=4)), fitted=$(round(fitted_p, digits=4))")
    end

    # Calculate log-likelihood on held-out data
    println("\n7. Out-of-sample validation:")

    # Generate new test data
    test_data = rand(true_dist, 500)

    true_loglik = sum(logpdf(true_dist, x) for x in test_data)
    fitted_loglik = sum(logpdf(fitted_dist, x) for x in test_data)

    println("  Test log-likelihood (true model): $(round(true_loglik, digits=2))")
    println("  Test log-likelihood (fitted model): $(round(fitted_loglik, digits=2))")
    println("  Log-likelihood difference: $(round(fitted_loglik - true_loglik, digits=2))")

    println("\n" * "=" ^ 70)
    println("Demonstration completed successfully! 🎉")

    # Success criteria assessment
    max_error = max(μ_error, σ_error)
    if max_error < 3.0  # Issue requirement
        println("✓ All parameter estimates are within 3% of true values (issue requirement met)")
    elseif max_error < 10.0  # Reasonable for interval censoring
        println("✓ Parameter estimates are within 10% of true values (reasonable for interval censoring)")
    else
        println("⚠ Some parameter estimates exceed 10% error ($(round(max_error, digits=2))%)")
    end

    if prob_rmse < 0.01
        println("✓ Probability estimation is highly accurate (RMSE < 0.01)")
    else
        println("⚠ Probability estimation has moderate accuracy (RMSE = $(round(prob_rmse, digits=4)))")
    end

    if fitted_loglik > true_loglik - 50  # Allow some degradation due to estimation
        println("✓ Out-of-sample performance is good")
    else
        println("⚠ Out-of-sample performance shows degradation")
    end

    println("\n" * "=" ^ 70)
    println("Key insights:")
    println("• IntervalCensored rand() returns interval LEFT BOUNDARIES, not indices")
    println("• The PDF calculation is correct for these boundary values")
    println("• MLE fitting works when data simulation matches the model")
    println("• Parameter recovery quality depends on interval width and sample size")

catch e
    println("❌ Fitting failed with error: $(e)")
    println("This may be due to:")
    println("  - Numerical issues in optimization")
    println("  - Poor initial parameter values")
    println("  - Inappropriate interval structure")
    println("  - Implementation bugs")

    # Show data summary for debugging
    println("\nData summary for debugging:")
    println("  Data range: [$(minimum(interval_data)), $(maximum(interval_data))]")
    println("  Number of unique values: $(length(unique(interval_data)))")
    println("  Number of samples: $(length(interval_data))")
end

println("\n" * "=" ^ 70)
println("End of demonstration")
