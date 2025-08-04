# # Exponentially Tilted Primary Event Distributions

# This tutorial demonstrates how to use exponentially tilted primary event distributions
# to model realistic epidemic scenarios where infection timing within censoring windows
# is biased by epidemic growth or decay patterns.

using CensoredDistributions
using Distributions
using Plots
using Random

# ## Background

# During epidemic growth or decay, the assumption of uniformly distributed primary event 
# times within censoring windows becomes invalid. The timing of infections within reporting 
# windows is biased by the epidemic's growth rate:
#
# - **Exponential growth** (r > 0): Infections are more likely to occur near the end of 
#   the censoring window due to increasing incidence
# - **Exponential decay** (r < 0): Infections are more likely to occur near the start of 
#   the censoring window due to decreasing incidence
#
# This bias significantly affects delay distribution inference and downstream epidemic 
# parameter estimation.

# ## Creating Exponentially Tilted Distributions

# The `ExponentiallyTilted` distribution provides a principled approach to model primary 
# event timing when events occur at exponentially changing rates.

# ### Basic Construction

# Uniform-like distribution (no growth bias)
d_uniform = ExponentiallyTilted(0.0, 1.0, 0.0)
println("Uniform-like: r = 0.0")
println("  Mean: $(round(mean(d_uniform), digits=3))")
println("  Std:  $(round(std(d_uniform), digits=3))")
println("  Mode: $(round(mode(d_uniform), digits=3))")
println()

# Exponential growth scenario (r > 0)
d_growth = ExponentiallyTilted(0.0, 1.0, 2.0)
println("Growth scenario: r = 2.0")
println("  Mean: $(round(mean(d_growth), digits=3))")
println("  Std:  $(round(std(d_growth), digits=3))")
println("  Mode: $(round(mode(d_growth), digits=3))")
println()

# Exponential decay scenario (r < 0)
d_decay = ExponentiallyTilted(0.0, 1.0, -1.5)
println("Decay scenario: r = -1.5")
println("  Mean: $(round(mean(d_decay), digits=3))")
println("  Std:  $(round(std(d_decay), digits=3))")
println("  Mode: $(round(mode(d_decay), digits=3))")
println()

# ## Visualizing the Effect of Growth Rate

# Let's visualize how different growth rates affect the probability density

x_range = 0:0.01:1
growth_rates = [-2.0, -0.5, 0.0, 0.5, 2.0]
colors = [:red, :orange, :black, :blue, :purple]

p1 = plot(title="Exponentially Tilted PDFs", xlabel="Time", ylabel="Density", 
          legend=:topright)

for (i, r) in enumerate(growth_rates)
    d = ExponentiallyTilted(0.0, 1.0, r)
    y_vals = [pdf(d, x) for x in x_range]
    plot!(p1, x_range, y_vals, label="r = $r", color=colors[i], linewidth=2)
end

display(p1)

# ## Integration with Primary Censored Distributions

# The real power of exponentially tilted distributions comes when used as primary event
# distributions in epidemiological delay modeling.

# ### Example: Incubation Period Estimation During Epidemic Growth

# Define a delay distribution (incubation period)
incubation_dist = LogNormal(1.5, 0.75)  # Mean ≈ 5.8 days, std ≈ 5.2 days

# Define different epidemic scenarios
scenarios = [
    ("Strong Decay", ExponentiallyTilted(0.0, 1.0, -2.0)),
    ("Mild Decay", ExponentiallyTilted(0.0, 1.0, -0.5)),
    ("Uniform", ExponentiallyTilted(0.0, 1.0, 0.0)),
    ("Mild Growth", ExponentiallyTilted(0.0, 1.0, 0.5)),
    ("Strong Growth", ExponentiallyTilted(0.0, 1.0, 2.0))
]

# Create primary censored distributions for each scenario
censored_dists = Dict()
for (name, prior) in scenarios
    censored_dists[name] = primary_censored(incubation_dist, prior)
end

# Compare the resulting delay distributions
println("Impact of epidemic dynamics on observed delay distributions:")
println("Scenario          | Mean    | Std     | Mode")
println("------------------|---------|---------|--------")

for (name, _) in scenarios
    d = censored_dists[name]
    println("$(rpad(name, 17)) | $(rpad(round(mean(d), digits=2), 7)) | $(rpad(round(std(d), digits=2), 7)) | $(round(mode(d), digits=2))")
end

# ## Visualizing the Bias Effect

# Plot the CDFs to show how epidemic dynamics bias delay distributions

x_eval = 0:0.1:15
p2 = plot(title="Impact of Epidemic Dynamics on Delay Distributions", 
          xlabel="Observed Delay (days)", ylabel="Cumulative Probability",
          legend=:bottomright)

# Add the original (uncensored) distribution for reference
y_original = [cdf(incubation_dist, x) for x in x_eval]
plot!(p2, x_eval, y_original, label="Original (uncensored)", 
      color=:gray, linestyle=:dash, linewidth=2)

# Add each scenario
scenario_colors = [:red, :orange, :black, :blue, :purple]
for (i, (name, _)) in enumerate(scenarios)
    d = censored_dists[name]
    y_vals = [cdf(d, x) for x in x_eval]
    plot!(p2, x_eval, y_vals, label=name, color=scenario_colors[i], linewidth=2)
end

display(p2)

# ## Sampling and Parameter Recovery

# Demonstrate how to sample from exponentially tilted primary censored distributions
# and the importance of accounting for epidemic dynamics

Random.seed!(123)

# Simulate observed data under different epidemic scenarios
n_samples = 1000

println("\nSimulated data characteristics:")
println("Scenario          | Sample Mean | Sample Std | Bias vs Original")
println("------------------|-------------|------------|------------------")

original_mean = mean(incubation_dist)

for (name, _) in scenarios
    d = censored_dists[name]
    samples = rand(d, n_samples)
    sample_mean = mean(samples)
    sample_std = std(samples)
    bias = sample_mean - original_mean
    
    println("$(rpad(name, 17)) | $(rpad(round(sample_mean, digits=2), 11)) | $(rpad(round(sample_std, digits=2), 10)) | $(round(bias, digits=2))")
end

# ## Double Interval Censoring

# ExponentiallyTilted distributions also work with double interval censoring,
# combining primary event bias with secondary observation censoring

# Create a double interval censored distribution
double_censored = double_interval_censored(
    incubation_dist;
    primary_event = ExponentiallyTilted(0.0, 1.0, 1.0),  # Growth scenario
    interval = 0.5  # Secondary censoring interval
)

println("\nDouble interval censored distribution:")
println("  Mean: $(round(mean(double_censored), digits=2))")
println("  Std:  $(round(std(double_censored), digits=2))")

# Sample from the double censored distribution
double_samples = rand(double_censored, 500)
println("  Sample mean: $(round(mean(double_samples), digits=2))")
println("  Sample std:  $(round(std(double_samples), digits=2))")

# ## Parameter Selection Guidelines

# ### Choosing Growth Rates

# The growth rate parameter `r` should be chosen based on the epidemic context:
#
# - **r = 0**: Use when epidemic is in steady state or growth rate is unknown
# - **r > 0**: Use during epidemic growth phase
#   - Small values (0.1-0.5) for slow growth
#   - Medium values (0.5-1.5) for moderate growth  
#   - Large values (1.5+) for rapid growth
# - **r < 0**: Use during epidemic decay phase
#   - Values should be negative of corresponding growth rates
#
# ### Window Size Considerations

# The window size (max - min) should reflect the actual censoring window:
# - Daily reporting: window = 1.0
# - Weekly reporting: window = 7.0  
# - Custom intervals: set appropriately

# ### Practical Example: Real-world Parameter Selection

# Suppose we have an epidemic with doubling time of 7 days
doubling_time = 7.0
r_epidemic = log(2) / doubling_time  # ≈ 0.099
window_size = 1.0  # Daily reporting

practical_prior = ExponentiallyTilted(0.0, window_size, r_epidemic)
practical_censored = primary_censored(incubation_dist, practical_prior)

println("\nPractical example (7-day doubling time):")
println("  Epidemic growth rate: $(round(r_epidemic, digits=3))")
println("  Censored mean: $(round(mean(practical_censored), digits=2))")
println("  Bias vs uniform: $(round(mean(practical_censored) - mean(censored_dists["Uniform"]), digits=2))")

# ## Summary

# The `ExponentiallyTilted` distribution provides a principled way to account for
# epidemic dynamics when modeling primary event censoring. Key takeaways:
#
# 1. **Growth bias**: Positive r values lead to shorter apparent delays
# 2. **Decay bias**: Negative r values lead to longer apparent delays  
# 3. **Magnitude matters**: Larger |r| values create stronger bias
# 4. **Integration**: Works seamlessly with existing censoring infrastructure
# 5. **Numerical stability**: Automatically handles edge cases near r = 0
#
# This approach aligns with current best practices in epidemiological delay 
# distribution estimation and helps produce more accurate parameter estimates
# in realistic epidemic scenarios.