# Example: Exponentially Tilted Primary Event Distributions
# This script demonstrates the basic usage of ExponentiallyTilted distributions
# for modeling epidemic scenarios with dynamical bias.

using CensoredDistributions
using Distributions

println("=== Exponentially Tilted Primary Event Distributions ===\n")

# 1. Create different tilted distributions
println("1. Creating exponentially tilted distributions:")

# Uniform-like (no bias)
d_uniform = ExponentiallyTilted(0.0, 1.0, 0.0)
println("   Uniform (r=0.0): mean=$(round(mean(d_uniform), digits=3)), std=$(round(std(d_uniform), digits=3))")

# Growth scenario (events more likely at end of window)
d_growth = ExponentiallyTilted(0.0, 1.0, 1.0)
println("   Growth (r=1.0):  mean=$(round(mean(d_growth), digits=3)), std=$(round(std(d_growth), digits=3))")

# Decay scenario (events more likely at start of window)
d_decay = ExponentiallyTilted(0.0, 1.0, -1.0)
println("   Decay (r=-1.0):  mean=$(round(mean(d_decay), digits=3)), std=$(round(std(d_decay), digits=3))")

println()

# 2. Test basic functionality
println("2. Testing distribution functions:")
x_test = 0.7

for (name, d) in [("Uniform", d_uniform), ("Growth", d_growth), ("Decay", d_decay)]
    pdf_val = pdf(d, x_test)
    cdf_val = cdf(d, x_test)
    println("   $name: pdf($x_test)=$(round(pdf_val, digits=3)), cdf($x_test)=$(round(cdf_val, digits=3))")
end

println()

# 3. Integration with primary censoring
println("3. Primary censoring with different epidemic scenarios:")

# Define a delay distribution (e.g., incubation period)
delay_dist = LogNormal(1.5, 0.75)
println("   Base delay distribution: LogNormal(1.5, 0.75)")
println("   Uncensored mean: $(round(mean(delay_dist), digits=2)) days")

# Create primary censored distributions for different scenarios
scenarios = [
    ("Decay", ExponentiallyTilted(0.0, 1.0, -0.5)),
    ("Uniform", ExponentiallyTilted(0.0, 1.0, 0.0)),
    ("Growth", ExponentiallyTilted(0.0, 1.0, 0.5))
]

println("\n   Primary censored results:")
for (name, prior) in scenarios
    pc_dist = primary_censored(delay_dist, prior)
    println("   $name scenario: mean=$(round(mean(pc_dist), digits=2)) days, std=$(round(std(pc_dist), digits=2)) days")
end

println()

# 4. Sampling demonstration
println("4. Sampling from exponentially tilted distributions:")

# Generate samples from each scenario
n_samples = 1000
println("   Generating $n_samples samples from each scenario...")

for (name, prior) in scenarios
    pc_dist = primary_censored(delay_dist, prior)
    samples = rand(pc_dist, n_samples)
    
    empirical_mean = sum(samples) / n_samples
    empirical_std = sqrt(sum((x - empirical_mean)^2 for x in samples) / (n_samples - 1))
    
    println("   $name: empirical_mean=$(round(empirical_mean, digits=2)), empirical_std=$(round(empirical_std, digits=2))")
end

println()

# 5. Double interval censoring example
println("5. Double interval censoring with exponential tilting:")

# Combine primary event tilting with secondary interval censoring
double_censored = double_interval_censored(
    delay_dist;
    primary_event = ExponentiallyTilted(0.0, 1.0, 0.8),  # Growth scenario
    interval = 0.5  # Secondary censoring interval
)

println("   Double censored (growth + interval): mean=$(round(mean(double_censored), digits=2)), std=$(round(std(double_censored), digits=2))")

# Sample from double censored distribution
double_samples = rand(double_censored, 500)
double_empirical_mean = sum(double_samples) / length(double_samples)
println("   Sample mean from double censored: $(round(double_empirical_mean, digits=2))")

println()

# 6. Parameter validation
println("6. Parameter validation examples:")

try
    # This should work
    valid_dist = ExponentiallyTilted(0.0, 2.0, 1.5)
    println("   Valid parameters (0.0, 2.0, 1.5): ✓")
catch e
    println("   Valid parameters failed: $e")
end

try
    # This should fail (min >= max)
    invalid_dist = ExponentiallyTilted(2.0, 1.0, 0.0)
    println("   Invalid parameters (2.0, 1.0, 0.0): This shouldn't print")
catch e
    println("   Invalid parameters (2.0, 1.0, 0.0): ✗ (Expected error: $(typeof(e)))")
end

println("\n=== Example completed successfully! ===")