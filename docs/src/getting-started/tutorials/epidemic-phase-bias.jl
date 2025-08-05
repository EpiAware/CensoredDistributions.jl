### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ a1cfb960-d5f2-44f7-9aa2-eb421bbc771f
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

We'll demonstrate how epidemic growth creates bias in observed delay distributions through primary event censoring - distinct from truncation bias. We'll cover:

1. Exponentially tilted vs Uniform primary events
2. Double interval censoring effects
3. Impact on delay distributions
4. Parameter sensitivity analysis

### What might I need to know before starting

This tutorial builds on [Getting Started with CensoredDistributions.jl](@ref getting-started) and focusses on **epidemic phase bias for primary events** - the systematic error when primary events occur non-uniformly within observation windows (due here to exponential dynamics).

During epidemic growth/decline, primary events don't occur uniformly within our observation window:
- **Growth phase**: Recent primary events over-represented → shorter observed delays
- **Decline phase**: Older primary events more represented → longer observed delays
- **Steady state**: Uniform timing → minimal additional bias
"""

# ╔═╡ 2b436d16-51ec-47a7-8bd1-83dfee693702
md"""
## Setup

We'll examine epidemic phase bias using a realistic scenario:
- **True incubation period**: Gamma(4.0, 1.5) with mean 6.0 days
- **Primary event windows**: 7-day observation periods
- **Growth rate scenarios**: r ∈ {-10%, -5%, 0%, +5%, +10%}
- **Window sensitivity**: Testing 3, 7, and 14-day windows
"""

# ╔═╡ d09daadf-93a0-4065-b2b9-0a060f75f46a
begin
    # True delay distribution (incubation period: mean = 6 days)
    true_delay = Gamma(4.0, 1.5)
end

# ╔═╡ 778badaf-dcdd-4e87-a6b9-ee2d789e3675
md"""
## Part 1: Exponentially Tilted vs Uniform Primary Events

First, we compare how Exponentially tilted distributions with different growth rates shape primary event timing compared to uniform (steady state) patterns.
"""

# ╔═╡ 947ef70e-cb74-4ee7-aa1c-80903305d6bf
begin
    # Create scenario data directly in DataFrame
    window_length = 7
    scenarios_df = @chain DataFrame(
        name = ["Decline 10%", "Decline 5%", "Steady", "Growth 5%", "Growth 10%"],
        r = [-0.10, -0.05, 0.0, 0.05, 0.10],
        color = [:darkgreen, :lightgreen, :blue, :orange, :red]
    ) begin
        @transform :window = window_length
    end

    # Create primary event distributions
    @chain scenarios_df begin
        @transform! :primary_event = ExponentiallyTilted.(0.0, :window, :r)
        @transform! :uniform_ref = Uniform.(0.0, :window)
    end
end

# ╔═╡ 468e7035-a64d-4dd8-87d1-46c26a157db4
md"Created $(nrow(scenarios_df)) epidemic scenarios for comparison"

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

Now we demonstrate the **double interval censoring** concept: primary events occur within surveillance-defined primary windows, but during epidemics their timing distribution becomes non-uniform (ExponentiallyTilted), then we observe delays within secondary windows (surveillance window minus primary event time). For the secondary even the distribution doesn't impact what we observe (see references for why).
"""

# ╔═╡ 0321b0af-23ca-4d1a-a657-197b0a4a5a1f
begin
    # Demonstrate effective censoring windows from combining primary + secondary
    n_samples = 10000
    secondary_window = 10.0  # Total surveillance window

    # Create expanded dataset with samples for each scenario
    expanded_scenarios = DataFrame()
    for row in eachrow(scenarios_df)
        scenario_samples = DataFrame(
            sample_id = 1:n_samples,
            scenario = row.name,
            r_value = row.r,
            scenario_color = row.color,
            primary_dist = [row.primary_event for _ in 1:n_samples]
        )
        append!(expanded_scenarios, scenario_samples)
    end

    # Sample primary event times and calculate effective windows (no delay filtering)
    censoring_windows_df = @chain expanded_scenarios begin
        @transform :primary_time = [rand(pd) for pd in :primary_dist]
        @transform :effective_window = secondary_window .- :primary_time
        @select :scenario, :r_value, :scenario_color, :primary_time, :effective_window
    end
end

# ╔═╡ 61da92eb-8f60-41dc-bf45-031ec6dc52b8
md"Generated $(nrow(censoring_windows_df)) effective censoring windows across scenarios"

# ╔═╡ 3abce2f9-30b0-4603-98b9-f018dc7db1d7
begin
    # Visualise effective censoring windows
    p2 = plot(title = "Double Interval Censoring: Secondary - Primary Windows",
        xlabel = "Primary event time (days)", ylabel = "Effective secondary window (days)",
        size = (700, 400))

    # Show how effective secondary window varies with primary timing
    x_primary_demo = 0:0.1:secondary_window
    effective_secondary_demo = secondary_window .- x_primary_demo

    plot!(p2, x_primary_demo, effective_secondary_demo,
        label = "Effective secondary window", color = :black, linewidth = 3)

    # Add scatter points for sampled data (subset for clarity)
    sample_subset = @subset(censoring_windows_df, rand(nrow(censoring_windows_df)) .< 0.01)

    for scenario_name in unique(sample_subset.scenario)
        scenario_data = @subset(sample_subset, :scenario .== scenario_name)
        scatter!(p2, scenario_data.primary_time, scenario_data.effective_window,
            label = scenario_name, alpha = 0.6, markersize = 2,
            color = scenario_data.scenario_color[1])
    end

    p2
end

# ╔═╡ 22d4ada7-ff16-4be6-b99a-e600a5a1ed75
begin
    # Create double interval censored distributions for each scenario
    double_censored_df = @chain scenarios_df begin
        @transform :double_censored_dist = double_interval_censored.(
            Ref(true_delay), :primary_event;
            secondary_dist = Uniform(0.0, secondary_window)
        )
        @transform :double_mean = mean.(:double_censored_dist)
        @transform :double_std = std.(:double_censored_dist)
        @transform :double_median = median.(:double_censored_dist)
    end

    # Compare with single censoring (primary only)
    @chain double_censored_df begin
        @transform :single_censored_dist = primary_censored.(Ref(true_delay), :primary_event)
        @transform :single_mean = mean.(:single_censored_dist)
        @transform :single_std = std.(:single_censored_dist)
    end
end

# ╔═╡ f5173b0c-5240-497e-951a-6148f1ac2381
md"Created double interval censored distributions for comparison with single censoring"

# ╔═╡ d1025566-b83d-4e7a-80eb-c3ebc8c5df9e
md"""
## Part 3: Impact on Observed Delay Distributions

Now we examine how epidemic phase bias affects the delays we actually observe, comparing against the true distribution.
"""

# ╔═╡ 1d5196ac-f677-40a8-9385-d3fe7cca44f2
begin
    # Create censored distributions and analyze bias
    @chain scenarios_df begin
        @transform! :censored_dist = primary_censored.(Ref(true_delay), :primary_event)
    end

    # Calculate theoretical bias statistics
    true_mean = mean(true_delay)
    true_std = std(true_delay)
    true_median = median(true_delay)

    bias_stats_df = @chain scenarios_df begin
        @transform :obs_mean = mean.(:censored_dist)
        @transform :obs_std = std.(:censored_dist)
        @transform :obs_median = median.(:censored_dist)
        @transform :mean_bias = :obs_mean .- true_mean
        @transform :std_bias = :obs_std .- true_std
        @transform :median_bias = :obs_median .- true_median
        @transform :mean_rel_bias = (:mean_bias ./ true_mean) .* 100
    end

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

# ╔═╡ f4a019a4-c825-4399-9c17-102155fad57c
begin
    # Parameter sensitivity analysis using same scenarios with different windows
    window_lengths = [3.0, 7.0, 14.0]

    # Create comprehensive parameter grid using existing scenarios
    sensitivity_df = @chain DataFrame(
        window_length = repeat(window_lengths, inner = length(scenarios_df.r)),
        r = repeat(scenarios_df.r, outer = length(window_lengths)),
        name = repeat(scenarios_df.name, outer = length(window_lengths))
    ) begin
        @transform :primary_event = ExponentiallyTilted.(0.0, :window_length, :r)
        @transform :censored_dist = primary_censored.(Ref(true_delay), :primary_event)
        @transform :mean_bias = mean.(:censored_dist) .- true_mean
        @transform :median_bias = median.(:censored_dist) .- true_median
        @transform :mean_rel_bias = ((:mean_bias ./ true_mean) .* 100)
    end

    # Visualise sensitivity analysis
    p4 = plot(title = "Bias Sensitivity: Window Length & Growth Rate",
        xlabel = "Growth rate (r)", ylabel = "Mean bias (days)",
        size = (700, 400))

    for window in window_lengths
        window_data = @subset(sensitivity_df, :window_length .== window)
        plot!(p4, window_data.r, window_data.mean_bias,
            label = "$(Int(window))-day window", marker = :circle, linewidth = 2)
    end

    hline!(p4, [0], color = :black, linestyle = :dash, alpha = 0.5, label = "No bias")

    p4
end

# ╔═╡ 902e2702-fd10-467c-9ea9-8eb35da46474
md"""
## Key Insights

1. **Primary event bias is distinct from truncation bias:**
   - Occurs when primary events don't happen uniformly within observation windows
   - Growth phase: Recent primary events over-represented → shorter observed delays
   - Decline phase: Older primary events more represented → longer observed delays
   - Steady state: Uniform timing → minimal bias

2. **Two key factors control bias magnitude:**
   - **Growth rate magnitude**: Stronger growth (larger |r|) creates more bias
   - **Window length**: Longer windows reduce bias for same growth rate
   - Bias can be several days for realistic epidemic parameters

3. **Double interval censoring reveals the mechanism:**
   - Primary events occur within surveillance windows, but their distribution matters
   - During epidemic processes, primary events are likely non-uniform (e.g., ExponentiallyTilted)
   - Secondary observation window depends on primary timing
   - Effective observation time = secondary window - primary event time

4. **Practical implications:**
   - Ignoring epidemic phase leads to systematic parameter estimation errors
   - ExponentiallyTilted distributions can correct for this bias
   - Critical for accurate outbreak analysis and forecasting

5. **When to use ExponentiallyTilted:**
   - Most important when primary censoring intervals are wide (multi-day windows)
   - For daily primary censoring and moderate epidemic growth, Uniform distributions are often a reasonable approximation
   - Key benefit of Uniform: Analytical solutions available that are much faster computationally
   - Use ExponentiallyTilted when precision is critical and computational cost is acceptable

## References

- Park et al. (2024): "Estimating epidemiological delay distributions for infectious diseases"
- Charniga et al. (2024): "Primary event censoring in infectious disease surveillance"
- [SISMID Tutorial](https://nfidd.github.io/sismid/sessions/biases-in-delay-distributions.html): Interactive bias demonstrations
"""

# ╔═╡ Cell order:
# ╠═a1cfb960-d5f2-44f7-9aa2-eb421bbc771f
# ╠═5af3c6a1-5518-49d7-bacc-d59d429d8367
# ╠═0ff7feee-5685-45e0-8145-99bdcd834757
# ╟─2b436d16-51ec-47a7-8bd1-83dfee693702
# ╠═d09daadf-93a0-4065-b2b9-0a060f75f46a
# ╟─778badaf-dcdd-4e87-a6b9-ee2d789e3675
# ╠═947ef70e-cb74-4ee7-aa1c-80903305d6bf
# ╟─468e7035-a64d-4dd8-87d1-46c26a157db4
# ╠═334379d2-f7b5-476a-bf04-96dfa2c34734
# ╟─9c3b331b-a832-4f94-98b3-f9fb1bf90261
# ╠═0321b0af-23ca-4d1a-a657-197b0a4a5a1f
# ╟─61da92eb-8f60-41dc-bf45-031ec6dc52b8
# ╠═3abce2f9-30b0-4603-98b9-f018dc7db1d7
# ╠═22d4ada7-ff16-4be6-b99a-e600a5a1ed75
# ╟─f5173b0c-5240-497e-951a-6148f1ac2381
# ╟─d1025566-b83d-4e7a-80eb-c3ebc8c5df9e
# ╠═1d5196ac-f677-40a8-9385-d3fe7cca44f2
# ╠═c4922ef2-7ffd-4abc-b163-e9271acfa72d
# ╟─0bc8affe-57ee-4c6d-9960-1d2b8b614632
# ╠═f4a019a4-c825-4399-9c17-102155fad57c
# ╟─902e2702-fd10-467c-9ea9-8eb35da46474
