### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# Static tutorial - no interactivity needed for documentation rendering

# ╔═╡ a1b2c3d4-e5f6-7890-abcd-ef1234567890
begin
    let
        docs_dir = (dirname ∘ dirname ∘ dirname)(@__DIR__)
        using Pkg: Pkg
        Pkg.activate(docs_dir)
        Pkg.instantiate()
    end
end

# ╔═╡ 5af3c6a1-5518-49d7-bacc-d59d429d8367
begin
    using CensoredDistributions
    using Distributions
    using Plots
    using DataFrames
    using DataFramesMeta
    using Statistics
    using Random

    Random.seed!(123)
end

# ╔═╡ 0ff7feee-5685-45e0-8145-99bdcd834757
md"""
# How Epidemic Phase Affects Observed Delay Distributions

## Introduction

### What are we going to do in this exercise

This tutorial demonstrates how epidemic growth creates bias in observed delay distributions through the **primary event window mechanism** - distinct from truncation bias. We'll cover:

1. **ExponentiallyTilted vs Uniform primary events** - How growth rates shape infection timing
2. **Double interval censoring** - Combining primary events with secondary observation windows
3. **Impact on delay distributions** - How epidemic phase biases what we observe
4. **Parameter sensitivity** - Effects of window length and growth rate
5. **Practical implications** - When and why this bias matters

### What might I need to know before starting

This tutorial builds on [Getting Started with CensoredDistributions.jl](../index.md) and focuses specifically on **primary event censoring bias** - the systematic error introduced when infections occur non-uniformly within the observation window.

### Packages used

We use Plots.jl for visualisation, DataFramesMeta.jl for data manipulation, and core CensoredDistributions.jl functionality.

## The Primary Event Bias Problem

**Key insight**: This is separate from truncation bias. During epidemic growth/decline, infections don't occur uniformly within our observation window. This non-uniform timing biases which delays we observe:

- **Growth phase**: Recent infections over-represented → shorter observed delays
- **Decline phase**: Older infections more represented → longer observed delays
- **Steady state**: Uniform timing → minimal bias

**Primary factors**: Window length and growth rate both affect bias magnitude.
"""

# ╔═╡ 2b436d16-51ec-47a7-8bd1-83dfee693702
begin
    # True delay distribution (incubation period: mean = 6 days)
    true_delay = Gamma(4.0, 1.5)

    md"""
    ## Setup

    We'll examine epidemic phase bias using a realistic scenario:
    - **True incubation period**: Gamma(4.0, 1.5) with mean $(round(mean(true_delay), digits=1)) days
    - **Primary event windows**: 7-day observation periods
    - **Growth scenarios**: r ∈ {-1, 0, +1} representing decline, steady state, and growth
    - **Window sensitivity**: Testing 3, 7, and 14-day windows
    """
end

# ╔═╡ 778badaf-dcdd-4e87-a6b9-ee2d789e3675
md"""
## Part 1: ExponentiallyTilted vs Uniform Primary Events

First, we compare how ExponentiallyTilted distributions with different growth rates shape infection timing compared to uniform (steady state) patterns.
"""

# ╔═╡ 947ef70e-cb74-4ee7-aa1c-80903305d6bf
begin
    # Define static scenarios for comparison
    window_length = 7.0
    growth_rates = [-1.0, 0.0, 1.0]
    scenario_names = ["Decline", "Steady", "Growth"]
    colors = [:green, :blue, :red]

    # Create scenario data using DataFrames
    scenarios_df = DataFrame(
        name = scenario_names,
        r = growth_rates,
        color = colors,
        window = fill(window_length, 3)
    )

    # Create primary event distributions
    scenarios_df = @transform(scenarios_df,
        :primary_event=ExponentiallyTilted.(0.0, :window, :r),
        :uniform_ref=Uniform.(0.0, :window))

    md"Created $(nrow(scenarios_df)) epidemic scenarios for comparison"
end

# ╔═╡ 334379d2-f7b5-476a-bf04-96dfa2c34734
begin
    # Compare ExponentiallyTilted vs Uniform primary events
    x_primary = range(0, window_length, length = 100)

    p1 = plot(title = "Primary Event Timing: ExponentiallyTilted vs Uniform",
        xlabel = "Days before observation end", ylabel = "Probability density",
        size = (700, 400), legend = :topright)

    # Plot ExponentiallyTilted distributions
    for row in eachrow(scenarios_df)
        y_exponential = pdf.(row.primary_event, x_primary)
        plot!(p1, x_primary, y_exponential,
            label = "$(row.name) (r=$(row.r))",
            color = row.color, linewidth = 3)
    end

    # Add uniform reference (steady state comparison)
    uniform_ref = Uniform(0.0, window_length)
    y_uniform = pdf.(uniform_ref, x_primary)
    plot!(p1, x_primary, y_uniform,
        label = "Uniform (reference)", color = :black,
        linestyle = :dash, linewidth = 2)

    p1
end

# ╔═╡ 9c3b331b-a832-4f94-98b3-f9fb1bf90261
md"""
## Part 2: Double Interval Censoring - Primary + Secondary Windows

Now we demonstrate the **double interval censoring** concept: infections occur within primary windows (epidemic-driven), then we observe delays within secondary windows (surveillance window minus primary event time).
"""

# ╔═╡ 0321b0af-23ca-4d1a-a657-197b0a4a5a1f
begin
    # Demonstrate double interval censoring using sampling
    n_samples = 10000
    secondary_window = 10.0  # Total surveillance window

    # Sample from each scenario
    double_censoring_df = DataFrame()

    for row in eachrow(scenarios_df)
        # Sample primary event times (when infections occurred)
        primary_times = rand(row.primary_event, n_samples)

        # Sample delay times from true distribution
        delay_times = rand(true_delay, n_samples)

        # Calculate effective secondary window (secondary - primary)
        effective_secondary = secondary_window .- primary_times

        # Only keep delays that fit within the effective secondary window
        # This is the double censoring effect
        valid_delays = delay_times .< effective_secondary

        scenario_data = DataFrame(
            scenario = fill(row.name, sum(valid_delays)),
            r_value = fill(row.r, sum(valid_delays)),
            color = fill(row.color, sum(valid_delays)),
            primary_time = primary_times[valid_delays],
            delay = delay_times[valid_delays],
            effective_window = effective_secondary[valid_delays],
            observed_delay = delay_times[valid_delays]  # What we actually observe
        )

        double_censoring_df = vcat(double_censoring_df, scenario_data)
    end

    md"Simulated $(nrow(double_censoring_df)) observations with double interval censoring"
end

# ╔═╡ 3abce2f9-30b0-4603-98b9-f018dc7db1d7
begin
    # Visualise double interval censoring effect
    p2 = plot(title = "Double Interval Censoring: Secondary - Primary Windows",
        xlabel = "Primary event time (days)", ylabel = "Effective secondary window (days)",
        size = (700, 400))

    # Show how effective secondary window varies with primary timing
    x_primary_demo = 0:0.1:secondary_window
    effective_secondary_demo = secondary_window .- x_primary_demo

    plot!(p2, x_primary_demo, effective_secondary_demo,
        label = "Effective secondary window", color = :black, linewidth = 3)

    # Add scatter points for sampled data (subset for clarity)
    sample_subset = @subset(double_censoring_df, rand(nrow(double_censoring_df)) .< 0.01)

    for scenario_name in unique(sample_subset.scenario)
        scenario_data = @subset(sample_subset, :scenario .== scenario_name)
        scatter!(p2, scenario_data.primary_time, scenario_data.effective_window,
            label = scenario_name, alpha = 0.6, markersize = 2,
            color = scenario_data.color[1])
    end

    p2
end

# ╔═╡ d1025566-b83d-4e7a-80eb-c3ebc8c5df9e
md"""
## Part 3: Impact on Observed Delay Distributions

Now we examine how epidemic phase bias affects the delays we actually observe, comparing against the true distribution.
"""

# ╔═╡ 1d5196ac-f677-40a8-9385-d3fe7cca44f2
begin
    # Create censored distributions and analyze bias
    scenarios_df = @transform(scenarios_df,
        :censored_dist = primary_censored.(Ref(true_delay), :primary_event))

    # Calculate theoretical bias statistics
    bias_stats_df = @transform(scenarios_df,
        :obs_mean=mean.(:censored_dist),
        :obs_std=std.(:censored_dist),
        :obs_median=median.(:censored_dist))

    # Add bias calculations
    true_mean = mean(true_delay)
    true_std = std(true_delay)
    true_median = median(true_delay)

    bias_stats_df = @transform(bias_stats_df,
        :mean_bias=:obs_mean .- true_mean,
        :std_bias=:obs_std .- true_std,
        :median_bias=:obs_median .- true_median,
        :mean_rel_bias=(:mean_bias ./ true_mean) .* 100)

    # Display results
    select(bias_stats_df, :name, :r, :obs_mean, :obs_median,
        :mean_bias, :median_bias, :mean_rel_bias)
end

# ╔═╡ c4922ef2-7ffd-4abc-b163-e9271acfa72d
begin
    # Visualise observed vs true delay distributions
    x_delay = range(0, 15, length = 100)

    p3 = plot(title = "Observed vs True Delay Distributions",
        xlabel = "Delay (days)", ylabel = "Probability density",
        size = (700, 400))

    # True distribution (reference)
    y_true = pdf.(true_delay, x_delay)
    plot!(p3, x_delay, y_true,
        label = "True distribution", color = :black,
        linestyle = :dash, linewidth = 3)

    # Observed distributions for each epidemic phase
    for row in eachrow(scenarios_df)
        y_obs = pdf.(row.censored_dist, x_delay)
        plot!(p3, x_delay, y_obs,
            label = "$(row.name) (r=$(row.r))",
            color = row.color, linewidth = 2)
    end

    p3
end

# ╔═╡ 0bc8affe-57ee-4c6d-9960-1d2b8b614632
md"""
## Part 4: Parameter Sensitivity Analysis

Finally, we examine how the two key factors - **window length** and **growth rate** - affect bias magnitude.
"""

# ╔═╡ a1b2c3d4-e5f6-4789-b0c1-234567890abc
begin
    # Parameter sensitivity analysis
    window_lengths = [3.0, 7.0, 14.0]
    growth_rates = [-2.0, -1.0, 0.0, 1.0, 2.0]

    # Create comprehensive parameter grid
    sensitivity_df = DataFrame()

    for window in window_lengths
        for r in growth_rates
            primary_event = ExponentiallyTilted(0.0, window, r)
            censored_dist = primary_censored(true_delay, primary_event)

            row_data = DataFrame(
                window_length = window,
                growth_rate = r,
                mean_bias = mean(censored_dist) - true_mean,
                median_bias = median(censored_dist) - true_median,
                mean_rel_bias = ((mean(censored_dist) - true_mean) / true_mean) * 100
            )

            sensitivity_df = vcat(sensitivity_df, row_data)
        end
    end

    # Visualise sensitivity analysis
    p4 = plot(title = "Bias Sensitivity: Window Length & Growth Rate",
        xlabel = "Growth rate (r)", ylabel = "Mean bias (days)",
        size = (700, 400))

    for window in window_lengths
        window_data = @subset(sensitivity_df, :window_length .== window)
        plot!(p4, window_data.growth_rate, window_data.mean_bias,
            label = "$(Int(window))-day window", marker = :circle, linewidth = 2)
    end

    hline!(p4, [0], color = :black, linestyle = :dash, alpha = 0.5, label = "No bias")

    p4
end

# ╔═╡ b1c2d3e4-f5g6-4h7i-j8k9-l0m1n2o3p4q5
md"""
## Key Insights

1. **PRIMARY EVENT BIAS IS DISTINCT FROM TRUNCATION BIAS:**
   - Occurs when infections don't happen uniformly within observation windows
   - Growth phase: Recent infections over-represented → shorter observed delays
   - Decline phase: Older infections more represented → longer observed delays
   - Steady state: Uniform timing → minimal bias

2. **TWO KEY FACTORS CONTROL BIAS MAGNITUDE:**
   - **Growth rate magnitude**: Stronger growth (larger |r|) creates more bias
   - **Window length**: Longer windows reduce bias for same growth rate
   - Bias can be several days for realistic epidemic parameters

3. **DOUBLE INTERVAL CENSORING REVEALS THE MECHANISM:**
   - Primary events occur within epidemic-driven windows
   - Secondary observation window depends on primary timing
   - Effective observation time = secondary window - primary event time

4. **PRACTICAL IMPLICATIONS:**
   - Ignoring epidemic phase leads to systematic parameter estimation errors
   - ExponentiallyTilted distributions can correct for this bias
   - Critical for accurate outbreak analysis and forecasting

5. **WHEN TO USE ExponentiallyTilted:**
   - During clear epidemic growth or decline phases
   - When unbiased delay estimates are essential
   - For robust epidemiological parameter inference

## References

- Park et al. (2024): "Estimating epidemiological delay distributions for infectious diseases"
- Charniga et al. (2024): "Primary event censoring in infectious disease surveillance"
- [SISMID Tutorial](https://nfidd.github.io/sismid/sessions/biases-in-delay-distributions.html): Interactive bias demonstrations
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CensoredDistributions = "878ddba8-8f10-4809-8c0c-20e5f95b8fe1"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
CensoredDistributions = "~0.1"
Distributions = "~0.25"
Plots = "~1.40"
PlutoUI = "~0.7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.5"
manifest_format = "2.0"
project_hash = "da39a3ee5e6b4b0d3255bfef95601890afd80709" # pragma: allowlist secret

[deps]
"""
