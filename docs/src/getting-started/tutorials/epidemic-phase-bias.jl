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

# ╔═╡ abae7fb4-26f5-4ea5-b8ad-1b40ec1cd075
begin
    using CensoredDistributions
    using Distributions
    using Plots
    using StatsPlots
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

We'll demonstrate how epidemic growth creates additional bias in observed delay distributions through primary event censoring - distinct from truncation bias. We'll cover:

1. Exponentially tilted vs Uniform primary events
2. Double interval censoring effects
3. Impact on delay distributions

### What might I need to know before starting

This tutorial builds on [Getting Started with CensoredDistributions.jl](@ref getting-started) and focusses on **epidemic phase bias for primary events** - when primary events occur non-uniformly within observation windows (due here to exponential dynamics).

During epidemic growth/decline, primary events don't occur uniformly within our observation window:
- **Growth phase**: Recent primary events over-represented → shorter observed delays
- **Decline phase**: Older primary events more represented → longer observed delays
- **Steady state**: Uniform timing → minimal additional bias
"""

# ╔═╡ 2b436d16-51ec-47a7-8bd1-83dfee693702
md"""
## Setup

### Packages used
"""

# ╔═╡ d3452fdb-8fb6-4e3f-aa20-da60ec45fa7a
md"""
### Simulated scenarios

We'll examine epidemic phase bias using a realistic scenario:
- **True incubation period**: Gamma(4.0, 1.5) with mean 6.0 days
- **Primary event windows**: 7-day observation periods
- **Growth rate scenarios**: r ∈ {-10%, -5%, 0%, +5%, +10%}
"""

# ╔═╡ d09daadf-93a0-4065-b2b9-0a060f75f46a
true_delay = Gamma(4.0, 1.5);

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
md"Created $(nrow(scenarios_df)) epidemic scenarios for comparison. Now lets plot the primary event timing distributions."

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
    n_samples = 1000000
    secondary_window = 7.0  # Secondary window length

    # DataFramesMeta lacks nest/unnest so we have to get clunky
    censoring_windows_df = @chain scenarios_df begin
        vcat([DataFrame(
                  name = fill(row.name, n_samples),
                  r = fill(row.r, n_samples),
                  color = fill(row.color, n_samples),
                  sample_id = 1:n_samples,
                  primary_time = rand(row.primary_event, n_samples)
              ) for row in eachrow(_)]...)
        @transform :effective_window = secondary_window .- :primary_time
    end
end

# ╔═╡ 61da92eb-8f60-41dc-bf45-031ec6dc52b8
md"Now we can plot the distribution of effective censoring windows by scenario."

# ╔═╡ 3abce2f9-30b0-4603-98b9-f018dc7db1d7
begin
    # Plot distribution of effective censoring windows by scenario
    p2 = plot(
        title = "Distribution of Effective Censoring Windows by Epidemic Phase",
        xlabel = "Effective censoring window (days)",
        ylabel = "Density",
        size = (700, 400),
        legend = :topright
    )

    # Plot density for each scenario
    for scenario_name in unique(censoring_windows_df.name)
        scenario_data = @subset(censoring_windows_df, :name .== scenario_name)
        scenario_color = scenario_data.color[1]

        # Use StatsPlots density plot
        density!(p2, scenario_data.effective_window,
            label = scenario_name,
            color = scenario_color,
            linewidth = 3,
            alpha = 0.7)
    end

    p2
end

# ╔═╡ d1025566-b83d-4e7a-80eb-c3ebc8c5df9e
md"""
## Part 3: Impact on Censored Delay Distributions

Now we examine how epidemic phase bias affects the delays we might observe. We start with the primary censored delay distributions. Note that most of the time, we won't observe these continuous distributions as the secondary event will also be censored.
"""

# ╔═╡ c4922ef2-7ffd-4abc-b163-e9271acfa72d
begin
    # Setup the primary censored distributions
    @chain scenarios_df begin
        @transform! :censored_dist = primary_censored.(Ref(true_delay), :primary_event)
    end

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

# ╔═╡ 926ad257-5778-4bd0-8c5c-316eece80e7a
md"We can also plot the impact on the double censored distribution (which is what we would often observe). Here we show the CDF."

# ╔═╡ 3735b6d9-049a-4cad-b964-735a4ab055a8
begin
    # Setup the double interval censored distributions
    @chain scenarios_df begin
        @rtransform! :double_censored_dist = double_interval_censored(
            true_delay; primary_event = :primary_event,
            interval = secondary_window
        )
    end

    # Visualise double vs single censored delay distributions
    x_delay_d = range(0, 40, length = 100)

    # True distribution (reference)
    y_true_d = cdf.(true_delay, x_delay_d)
    p4 = plot(title = "Double vs Single Interval Censored Distributions",
        xlabel = "Delay (days)", ylabel = "Probability density",
        size = (700, 400))

    plot!(p4, x_delay_d, y_true_d,
        label = "True distribution", color = :black,
        linestyle = :dash, linewidth = 3)

    # Double censored distributions for each epidemic phase
    for row in eachrow(scenarios_df)
        y_double = cdf.(row.double_censored_dist, x_delay_d)
        plot!(p4, x_delay_d, y_double,
            label = "$(row.name) Double (r=$(row.r))",
            color = row.color, linewidth = 2, linestyle = :solid)
    end

    p4
end

# ╔═╡ 902e2702-fd10-467c-9ea9-8eb35da46474
md"""
## Key Insights

1. **Primary event bias is distinct from truncation bias:**
   - Occurs regardless of the primary event distribution but is more complex when the primary event distribution is non-uniform.
   - Growth phase: Recent primary events over-represented → shorter observed delays
   - Decline phase: Older primary events more represented → longer observed delays

2. **Two key factors control bias magnitude:**
   - **Growth rate magnitude**: Stronger growth (larger |r|) creates more divergence from the uniform case
   - **Window length**: Longer windows increases divergence for same growth rate

3. **When to use `ExponentiallyTilted`:**
   - Most important when primary censoring intervals are wide (multi-day windows)
   - For daily primary censoring and moderate epidemic growth, Uniform distributions are often a reasonable approximation
   - Key benefit of Uniform: Analytical solutions available that are much faster computationally
   - Use `ExponentiallyTilted` when precision is critical, the computational cost is acceptable, and the growth rate is relatively well known.

## References

- Park et al. (2024): "Estimating epidemiological delay distributions for infectious diseases"
- Charniga et al. (2024): "Primary event censoring in infectious disease surveillance"
- [SISMID Tutorial](https://nfidd.github.io/sismid/sessions/biases-in-delay-distributions.html): Interactive bias demonstrations
"""

# ╔═╡ Cell order:
# ╠═a1cfb960-d5f2-44f7-9aa2-eb421bbc771f
# ╠═abae7fb4-26f5-4ea5-b8ad-1b40ec1cd075
# ╟─0ff7feee-5685-45e0-8145-99bdcd834757
# ╟─2b436d16-51ec-47a7-8bd1-83dfee693702
# ╟─d3452fdb-8fb6-4e3f-aa20-da60ec45fa7a
# ╠═d09daadf-93a0-4065-b2b9-0a060f75f46a
# ╟─778badaf-dcdd-4e87-a6b9-ee2d789e3675
# ╠═947ef70e-cb74-4ee7-aa1c-80903305d6bf
# ╟─468e7035-a64d-4dd8-87d1-46c26a157db4
# ╠═334379d2-f7b5-476a-bf04-96dfa2c34734
# ╟─9c3b331b-a832-4f94-98b3-f9fb1bf90261
# ╠═0321b0af-23ca-4d1a-a657-197b0a4a5a1f
# ╟─61da92eb-8f60-41dc-bf45-031ec6dc52b8
# ╠═3abce2f9-30b0-4603-98b9-f018dc7db1d7
# ╟─d1025566-b83d-4e7a-80eb-c3ebc8c5df9e
# ╠═c4922ef2-7ffd-4abc-b163-e9271acfa72d
# ╟─926ad257-5778-4bd0-8c5c-316eece80e7a
# ╠═3735b6d9-049a-4cad-b964-735a4ab055a8
# ╟─902e2702-fd10-467c-9ea9-8eb35da46474
