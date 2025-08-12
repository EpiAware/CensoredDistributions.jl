### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ bb9c75db-6638-48fe-afcb-e78c4bcc057d
begin
    let
        docs_dir = (dirname ∘ dirname ∘ dirname)(@__DIR__)
        using Pkg: Pkg
        Pkg.activate(docs_dir)
        Pkg.instantiate()
    end
end

# ╔═╡ 3690c122-d630-4fd0-aaf2-aea9226df086
begin
    using DataFramesMeta
    using Turing
    using DynamicPPL
    using Distributions
    using Random
    using CairoMakie, PairPlots
    using StatsBase
    using CensoredDistributions
end

# ╔═╡ 30511a27-984e-40b7-9b1e-34bc87cb8d56
md"""
# Fitting CensoredDistributions.jl modified distributions with Turing.jl

## Introduction

### What are we going to do in this exercise

We'll demonstrate how to use `CensoredDistributions.jl` in conjunction with
Turing.jl for Bayesian inference of epidemiological delay distributions.
We'll cover the following key points:

1. Defining a simple delay distribution model without observation processes.
2. Exploring the prior distribution of this model.
3. Defining a Bayesian model that incorporates double censoring and right
   truncation
4. Generating synthetic data from the model using fixed parameters
5. Fitting a naive model that ignores censoring
6. Fitting a model that accounts for secondary event censoring and truncation but not primary event censoring.
7. Fitting the full model that accounts for double censoring and right
   truncation.
8. Using improved weight conditioning with joint observations and fix() patterns
9. Demonstrating StatsBase.AbstractWeights integration patterns

## What might I need to know before starting

This tutorial builds on the concepts introduced in
[Getting Started with CensoredDistributions.jl](@ getting-started).

## Packages used
We use CairoMakie for plotting, Turing for probabilistic programming,
Chain.jl for data pipeline workflows, DataFramesMeta, Random, and StatsBase.
"""

# ╔═╡ c5ec0d58-ce3d-4b0b-a261-dbd37b119f71
md"""
## Generate synthetic data using Turing model simulation

We'll generate synthetic data by simulating from our Turing model with known
true parameters. This approach ensures consistency between the data generation
process and the model we'll use for inference, demonstrating how Turing models
can be used for both simulation and fitting.

The proper Turing simulation approach:
1. Define a Turing model that incorporates double censoring and right truncation
2. Create a model instance with missing observations for simulation
3. Use DynamicPPL's `fix` function to set parameters to their true values
4. Sample from the prior predictive distribution by calling the model as a function
"""

# ╔═╡ b4409687-7bee-4028-824d-03b209aee68d
Random.seed!(123) # Set seed for reproducibility

# ╔═╡ 30e99e77-aad1-43e8-9284-ab0bf8ae741f
md"### Define the true parameters for generating synthetic data"

# ╔═╡ 2fff24bf-74d3-47b8-be3f-f9d866d85903
md"We start by defining the number of samples and the true parameters of the
lognormal."

# ╔═╡ 28bcd612-19f6-4e25-b6df-cb43df4f2a73
n = 2000;

# ╔═╡ 04e414ab-c790-4d31-b216-18776534a287
meanlog = 1.5;

# ╔═╡ 54700ad7-6b2a-440f-903a-c126b4c60c0e
sdlog = 0.75;

# ╔═╡ d3a7196a-185d-445b-afb7-99b546b1f72a
md"Now we can define a lognormal distribution using Distributions.jl."

# ╔═╡ 7f82b991-00ca-4eed-994a-981c6d66454c
true_dist = LogNormal(meanlog, sdlog);

# ╔═╡ 767a58ed-9d7b-41db-a488-10f98a777474
md"For each individual we now sample a primary and secondary event window as
well as a relative observation time (relative to their censored primary event)."

# ╔═╡ 767a58ff-9d7b-41db-a488-10f98a777475
md"### Define a reusable submodel for the latent delay distribution

To avoid code duplication across our models, we define a submodel that
encapsulates the latent delay distribution parameters. This pattern allows
us to reuse the same prior structure across all our models:"

# ╔═╡ 767a5900-9d7b-41db-a488-10f98a777476
@model function latent_delay_dist()
    mu ~ Normal(1.0, 2.0);
    sigma ~ truncated(Normal(0.5, 1); lower = 0.0)
    return LogNormal(mu, sigma)
end

# ╔═╡ 767a5900-9d7b-41db-a488-10f98a777477
md"and define a helper function to standardize our pairplot visualizations
across all model fits:"

# ╔═╡ 767a5900-9d7b-41db-a488-10f98a777478
function plot_fit_with_truth(chain, truth_dict)
    f = pairplot(
        chain,
        PairPlots.Truth(
            truth_dict,
            label = "True Values"
        )
    )
    return f
end

# ╔═╡ d75a13b5-1c21-49c1-b10b-d26fcafc2736
md"### Prior predictive checks using pairplot

First, let's visualise the prior predictive distribution by sampling from
the instantiated model with uninformative priors and
comparing against our true parameters. This shows what the model believes
before seeing any data."

# ╔═╡ 767a5900-9d7b-41db-a488-10f98a777479
begin
    # Sample from the latent delay distribution prior
    latent_prior_samples = sample(latent_delay_dist(), Prior(), 1000)

    # Visualize the prior distribution
    plot_fit_with_truth(latent_prior_samples, (; mu = meanlog, sigma = sdlog))
end

# ╔═╡ 767a5900-9d7b-41db-a488-10f98a777480
md"### Define the double censored model for simulation and fitting

Now we define our full model that incorporates double censoring and right
truncation. This model uses the `latent_delay_dist()` submodel via
`to_submodel()` to include the delay distribution parameters. It also uses our `double_interval_censored()` function to define each double censored and right truncated delay:"

# ╔═╡ a8f036dd-fd59-4834-8422-f1ea7da616e0
@model function CensoredDistributions_model(pwindow_bounds, swindow_bounds, obs_time_bounds)
    # Window parameters as uniform distributions with bounds
    pwindows ~ arraydist([Uniform(pw[1], pw[2]) for pw in pwindow_bounds])
    swindows ~ arraydist([Uniform(sw[1], sw[2]) for sw in swindow_bounds])
    obs_times ~ arraydist([Uniform(ot[1], ot[2]) for ot in obs_time_bounds])

    dist ~ to_submodel(latent_delay_dist())

    pcens_dists = map(pwindows, obs_times, swindows) do pw, D, sw
        pe = Uniform(0, pw)
        double_interval_censored(
            dist; primary_event = pe, upper = D, interval = sw)
    end

    obs ~ weight(pcens_dists)
    return obs
end

# ╔═╡ 72f12b29-0779-4be8-aa35-9b22aa20c3b3
md"""
We also need to define our simulated observation windows for each observed delay and the amount of censored time in which events have been observed (required to adjust for truncation). We do this in a data frame.
"""

# ╔═╡ 35472e04-e096-4948-a218-3de53923f271
# Define scenario bounds first (these represent our knowledge about the scenarios)
pwindow_bounds = [(0.0, 1.0), (0.0, 2.0)]  # Two primary window scenarios
swindow_bounds = [(0.0, 1.0), (0.0, 2.0)]  # Two secondary window scenarios
obs_time_bounds = [(7.0, 9.0), (9.0, 11.0), (11.0, 13.0)]  # Three observation time scenarios

# Create scenario assignments for each observation
n_scenarios_p = length(pwindow_bounds)
n_scenarios_s = length(swindow_bounds)
n_scenarios_o = length(obs_time_bounds)

simulated_scenario = DataFrame(
    pwindow_idx = rand(1:n_scenarios_p, n),
    swindow_idx = rand(1:n_scenarios_s, n),
    obs_time_idx = rand(1:n_scenarios_o, n)
)

# ╔═╡ b5598cc7-ddd1-4d90-af9b-110a518416ac
md"### Simulate from the double censored distribution for each individual"

# ╔═╡ 2e04be98-625f-45f4-bf5e-a0074ea1ea01
md"Using the double censored model, we simulate data by sampling from the
model using known true parameters. We use Turing's proper simulation approach
with DynamicPPL's `fix` function to set parameters to their true values and
sample from the prior predictive distribution. We first define a set of primary event distributions:"

# ╔═╡ c548931f-f5e3-4de9-9183-eb64575b6bdb
# Create primary event distributions from pwindows
@chain simulated_scenario begin
    @transform! :primary_dist = Uniform.(0.0, :pwindow)
end;

# ╔═╡ f3568b69-875d-494c-82cf-5a3db767cdaa
md"Then we can define the model using our observation windows."

# ╔═╡ 8cbb8a46-c090-420f-bbb9-32b971a963f0
model_for_simulation = CensoredDistributions_model(pwindow_bounds, swindow_bounds, obs_time_bounds)

# ╔═╡ 5516cadb-f2f5-4852-8215-1493b001ab4d
md"We can then fix our priors based on the known values."

# ╔═╡ cf588dc1-3ac7-46a2-9fab-38d90aa391c5

# Use Turing to simulate window parameters from bounds, then fix to specific values
# First, sample from the window parameter priors
window_samples = sample(
    CensoredDistributions_model(pwindow_bounds, swindow_bounds, obs_time_bounds),
    Prior(),
    1
)

# Extract the simulated window values
true_pwindows = window_samples[:pwindows][1]
true_swindows = window_samples[:swindows][1]
true_obs_times = window_samples[:obs_times][1]

# Now fix the model with these simulated window values
fixed_model = fix(
    model_for_simulation,
    (
        @varname(dist.mu) => meanlog,
        @varname(dist.sigma) => sdlog,
        @varname(pwindows) => true_pwindows,
        @varname(swindows) => true_swindows,
        @varname(obs_times) => true_obs_times
    )
)

# ╔═╡ fcc1d4ba-13ca-41be-8451-7d035c8ff4a2
DynamicPPL.fixed(fixed_model)

# ╔═╡ 2a0c4692-3ac5-4c46-9ac0-a057256a0b37
md"To simulate from this model all we need to do is call it:"

# ╔═╡ a52a18c5-7625-4d9d-a7f7-bce5cb6ccb3f
begin
    # Simulate from the fixed model by calling it directly
    simulated_obs = fixed_model()

    # Extract the observed delays for use in data frame
    observed_delays = simulated_obs
end

# ╔═╡ 8e3ff244-c4d5-4562-99a5-7e63c6860a1e
md"Now lets create a simulated data frame using our scenarios data frame and simulated data."

# ╔═╡ f4ed78e2-cdbb-4534-890a-fb346dd65f36
simulated_data = @chain simulated_scenario begin
    @transform :observed_delay = observed_delays
    @transform :observed_delay_upper = :observed_delay .+ :swindow
end;

# ╔═╡ 50757759-9ec3-42d0-a765-df212642885a
md"""
### Visualise the simulated data

To make handling the data easier and later to speed up our models we first create a dataframe with the data we just generated, aggregated to unique
combinations and count occurrences.
"""

# ╔═╡ 6fd01b5c-e374-4f5c-9f1c-ea75d06132af
simulated_counts = @chain simulated_data begin
    @groupby All()
    @combine :n = length(:pwindow)
end

# ╔═╡ 993f1f74-4a55-47a7-9e3e-c725cba13c0a
md"""
Now let's compare the samples with and without double interval censoring to the true
distribution. First let's calculate the empirical CDF:
"""

# ╔═╡ ccd8dd8e-c361-43ba-b4f1-2444ec6008fc
empirical_cdf_obs = @with(simulated_counts, ecdf(:observed_delay, weights = :n));

# ╔═╡ 2b773594-5187-45bc-96f4-22a3d726b7d2
# Create a sequence of x values for the theoretical CDF
x_seq = @with simulated_counts begin
    range(minimum(:observed_delay), stop = maximum(:observed_delay) + 2, length = 100);
end

# ╔═╡ a5b04acc-acc5-4d4d-8871-09d54caab185
begin
    # Calculate theoretical CDF using true log-normal distribution
    theoretical_cdf = @chain x_seq begin
        cdf.(true_dist, _)
    end;

    # Generate uncensored samples from the true distribution for comparison
    uncensored_samples = rand(true_dist, n);
    empirical_cdf_uncensored = ecdf(uncensored_samples);
end

# ╔═╡ fb6dc898-21a9-4f8d-aa14-5b45974c2242
let
    f = Figure()
    ax = Axis(f[1, 1],
        title = "Comparison of Censored vs Uncensored vs Theoretical CDF",
        ylabel = "Cumulative Probability",
        xlabel = "Delay"
    )
    scatter!(
        ax,
        x_seq,
        empirical_cdf_obs.(x_seq),
        label = "Empirical CDF (Censored)",
        color = :blue
    )
    scatter!(
        ax,
        x_seq,
        empirical_cdf_uncensored.(x_seq),
        label = "Empirical CDF (Uncensored)",
        color = :red,
        marker = :cross
    )
    lines!(ax, x_seq, theoretical_cdf, label = "Theoretical CDF",
        color = :black, linewidth = 2)
    vlines!(ax, [mean(simulated_data.observed_delay)], color = :blue, linestyle = :dash,
        label = "Censored mean", linewidth = 2)
    vlines!(ax, [mean(uncensored_samples)], color = :red, linestyle = :dash,
        label = "Uncensored mean", linewidth = 2)
    vlines!(ax, [mean(true_dist)], linestyle = :dash,
        label = "Theoretical mean", color = :black, linewidth = 2)
    axislegend(position = :rb)

    f
end

# ╔═╡ 91279812-9848-48bc-9258-b6f86c9fe923
md"
## Fitting a naive model using Turing

We'll now fit a naive model that ignores the censoring process. This model
treats the observed delay data as if it came directly from the uncensored
delay distribution, providing a baseline for comparison.
"

# ╔═╡ a257ce07-efbe-45e1-a8b0-ada40c29de8d
@model function naive_model()
    dist ~ to_submodel(latent_delay_dist())
    obs ~ weight(dist)
    return obs
end

# ╔═╡ 49846128-379c-4c3b-9ec1-567ffa92e079
md"
Now let's instantiate and condition this model using weighted observations. We use a
small constant to avoid issues at zero for this simple model and create weighted
observations with FrequencyWeights from StatsBase.
"

# ╔═╡ 4cf596f1-0042-4990-8d0a-caa8ba1db0c7
begin
    # Create weighted observations using FrequencyWeights
    obs_values_adj = @chain simulated_counts @transform(:obs_adj = :observed_delay .+ 1e-6) :obs_adj
    obs_counts = @chain simulated_counts :n

    # Condition on tuple of (values, counts) directly
    naive_mdl = naive_model() | (obs = (obs_values_adj, obs_counts),)
end

# ╔═╡ 71900c43-9f52-474d-adc7-becdc74045da
md"
Now let's fit the conditioned model. Note how we use weighted observations
`(values, FrequencyWeights)` with missing constructor weights - this demonstrates
the new weight conditioning pattern where weights are provided at observation time.
"

# ╔═╡ cd26da77-02fb-4b65-bd7b-88060d0c97e8
naive_fit = sample(naive_mdl, NUTS(), MCMCThreads(), 500, 4);

# ╔═╡ 10278d0c-8c72-4c5f-b857-d3bc6ff2c242
summarize(naive_fit)

# ╔═╡ 2c0b4f97-5953-497d-bca9-d1aa46c5150b
plot_fit_with_truth(naive_fit, Dict("dist.mu" => meanlog, "dist.sigma" => sdlog))

# ╔═╡ 7122bd53-81f6-4ea5-a024-86fdd7a7207a
md"
We see that the model has converged and the diagnostics look good. However,
just from the model posterior summary we see that we might not be very happy
with the fit. `mu` is smaller than the target $(meanlog) and `sigma` is
larger than the target $(sdlog).
"

# ╔═╡ 4cb137e3-93e9-43bf-a39f-b063dd6daac6
md"
## Fitting a truncation-adjusted interval model

Now let's fit an intermediate model that accounts for interval censoring and
right truncation but ignores the primary censoring process. This provides
a comparison point between the naive model and the full model.
"

# ╔═╡ c3afeed1-20ec-44c8-933c-ca0e75cda788
@model function interval_only_model(swindow_bounds, obs_time_bounds)
    # Window parameters as uniform distributions with bounds
    swindows ~ arraydist([Uniform(sw[1], sw[2]) for sw in swindow_bounds])
    obs_times ~ arraydist([Uniform(ot[1], ot[2]) for ot in obs_time_bounds])

    dist ~ to_submodel(latent_delay_dist())

    icens_dists = map(obs_times, swindows) do D, sw
        truncated(interval_censored(dist, sw), upper = D)
    end
    obs ~ weight(icens_dists)
    return obs
end

# ╔═╡ 4fc543fa-dca5-40c3-810b-979c536dfe0d
md"Create the interval-only model with bounds, fix the window parameters, and condition on observations"

# ╔═╡ 6a274882-df7d-4972-80a6-ea62d932a906
begin
    # Create the interval-only model with bounds
    interval_only_unfixed = interval_only_model(swindow_bounds, obs_time_bounds)

    # Fix window parameters to their true values
    interval_only_fixed = fix(
        interval_only_unfixed,
        (
            @varname(swindows) => true_swindows,
            @varname(obs_times) => true_obs_times
        )
    )

    # Create weighted observations using FrequencyWeights
    interval_obs_values = @chain simulated_counts :observed_delay
    interval_obs_counts = @chain simulated_counts :n
    interval_weighted_obs = fweights(interval_obs_counts)

    # Condition the model on weighted observations
    interval_only_mdl = interval_only_fixed |
                        (obs = (interval_obs_values, interval_weighted_obs),)
end

# ╔═╡ 38790b6c-4fef-4b28-9442-6bfaab9d3c5a
md"
Fit the interval-only model (*Note: `Turing.jl` supports a wide range of fitting
methods but here we use the No-U-turn sampler*):
"

# ╔═╡ 8e1764ec-345a-453a-830c-748c2a077eb7
interval_only_fit = sample(interval_only_mdl, NUTS(), MCMCThreads(), 500, 4);

# ╔═╡ e0912175-6a02-480f-b8df-abd7c06f67e9
summarize(interval_only_fit)

# ╔═╡ 08f819a5-db15-4016-9fd6-b1c4b56cb985
md"Lets plot the posterior compared to the true values again. *Note: An annoying feature to `to_submodel()` is that it automatically prefixes the LHS name to all variables names in the model meaning we need to customise our postprocessing or turn this feature off."

# ╔═╡ 01a37638-6494-4ce5-a02d-9d7f76f39ab7
plot_fit_with_truth(interval_only_fit, Dict("dist.mu" => meanlog, "dist.sigma" => sdlog))

# ╔═╡ 080c1bca-afcd-46c0-80b8-1708e8d05ae6
md"## Fitting the full CensoredDistributions model

Now we'll fit the full model that accounts for the censoring process.
Since the CensoredDistributions_model was defined earlier and used for
simulation, we'll reuse it for fitting - demonstrating the consistency
of our approach."

# ╔═╡ a59e371a-b671-4648-984d-7bcaac367d32
begin
    # Create the full CensoredDistributions model with bounds
    full_model_unfixed = CensoredDistributions_model(pwindow_bounds, swindow_bounds, obs_time_bounds)

    # Fix window parameters to their true values
    full_model_fixed = fix(
        full_model_unfixed,
        (
            @varname(pwindows) => true_pwindows,
            @varname(swindows) => true_swindows,
            @varname(obs_times) => true_obs_times
        )
    )

    # Create weighted observations using FrequencyWeights
    full_obs_values = @chain simulated_counts :observed_delay
    full_obs_counts = @chain simulated_counts :n
    full_weighted_obs = fweights(full_obs_counts)

    # Condition the model on weighted observations
    CensoredDistributions_mdl = full_model_fixed |
                                (obs = (full_obs_values, full_weighted_obs),)
end

# ╔═╡ 691e3d54-1a31-4686-a70d-711c2fc45dc1
md"
Now we fit the model to recover the true
parameters from the synthetic data we generated earlier. This demonstrates
the package's ability to perform accurate parameter recovery when the
censoring process is properly modelled.
"

# ╔═╡ b5cd8b13-e3db-4ed1-80ce-e3ac1c57932c
CensoredDistributions_fit = sample(
    CensoredDistributions_mdl, NUTS(), MCMCThreads(), 1000, 4);

# ╔═╡ a53a78b3-dcbe-4b62-a336-a26e647dc8c8
summarize(CensoredDistributions_fit)

# ╔═╡ f0c02e4a-c0cc-41de-b1bf-f5fad7e7dfdb
plot_fit_with_truth(
    CensoredDistributions_fit, Dict("dist.mu" => meanlog, "dist.sigma" => sdlog))

# ╔═╡ c045caa6-a44d-4a54-b122-1e50b1e0fe75
md"
We see that the model has converged and the diagnostics look good.
We also see that the posterior means are near the true parameters and the
90% credible intervals include the true parameters.
"

# ╔═╡ 1ba50a36-d7c8-45e9-b2a1-73f89dc4e8b3
md"
## Improved models using weight conditioning with joint observations

The weight conditioning functionality implemented in issue #124 enables more
flexible and efficient model specification patterns. We'll demonstrate three
key improvements:

1. **Using `fix()` for known window bounds** instead of treating them as parameters
2. **Joint observation conditioning** with `(value, weight)` tuples
3. **Missing constructor weights** pattern with `weight(censored_dists)`

### Enhanced data preparation for improved models

First, let's prepare our data in formats that demonstrate the new functionality.
We'll create window bounds that can be fixed, and prepare joint observations.
"

# ╔═╡ 2ba50a36-d7c8-45e9-b2a1-73f89dc4e8b4
md"Define the window bounds that we'll fix in our models rather than treating
as parameters:"

# ╔═╡ 3ba50a36-d7c8-45e9-b2a1-73f89dc4e8b5
begin
    # The bounds were already created earlier - let's display them for reference
    n_scenarios = length(pwindow_bounds)

    println("Number of scenarios: $n_scenarios")
    println("Primary window bounds: $pwindow_bounds")
    println("Secondary window bounds: $swindow_bounds")
    println("Observation time bounds: $obs_time_bounds")
end

# ╔═╡ 4ba50a36-d7c8-45e9-b2a1-73f89dc4e8b6
md"Create aggregated data using DataFramesMeta for consistent data handling:"

# ╔═╡ 5ba50a36-d7c8-45e9-b2a1-73f89dc4e8b7
begin
    # Extract values and weights using DataFramesMeta
    improved_obs_values = @chain simulated_counts :observed_delay
    improved_obs_counts = @chain simulated_counts :n

    # Create FrequencyWeights for improved models
    improved_weighted_obs = fweights(improved_obs_counts)
end;

# ╔═╡ 6ba50a36-d7c8-45e9-b2a1-73f89dc4e8b8
md"### Improved model with fixed bounds and weighted observations

This model demonstrates the new patterns:
- Window parameters defined as uniform distributions with specified bounds
- Using `fix()` to set window parameters rather than treating them as unknowns
- Weighted observations using StatsBase.FrequencyWeights
- Missing constructor weights pattern with `weight(censored_dists)`"

# ╔═╡ 7ba50a36-d7c8-45e9-b2a1-73f89dc4e8b9
@model function improved_double_censored_model(
        pwindow_bounds, swindow_bounds, obs_time_bounds)
    # Window parameters as uniform distributions with bounds
    pwindows ~ arraydist([Uniform(pw[1], pw[2]) for pw in pwindow_bounds])
    swindows ~ arraydist([Uniform(sw[1], sw[2]) for sw in swindow_bounds])
    obs_times ~ arraydist([Uniform(ot[1], ot[2]) for ot in obs_time_bounds])

    # Latent delay distribution
    dist ~ to_submodel(latent_delay_dist())

    # Create censored distributions using obs_times
    censored_dists = map(pwindows, swindows, obs_times) do pw, sw, ot
        pe = Uniform(0, pw)
        double_interval_censored(dist; primary_event = pe, interval = sw, obs_time = ot)
    end

    # Use missing constructor weights - weights provided via joint observations
    obs ~ weight(censored_dists)

    return obs
end

# ╔═╡ 8ba50a36-d7c8-45e9-b2a1-73f89dc4e8b8
md"### Create scenario mappings for improved model

We need to map our aggregated data back to the window scenarios:"

# ╔═╡ 9ba50a36-d7c8-45e9-b2a1-73f89dc4e8b9
scenario_mapping = @chain simulated_counts begin
    @transform :scenario_idx = map(
        row -> begin
            findfirst(
                i -> bounds_data.pwindow[i] == row.pwindow &&
                     bounds_data.swindow[i] == row.swindow &&
                     bounds_data.obs_time[i] == row.obs_time,
                1:n_scenarios)
        end,
        eachrow(_))
end;

# ╔═╡ aba50a36-d7c8-45e9-b2a1-73f89dc4e8ba
md"Create expanded weighted observations for each scenario:"

# ╔═╡ bba50a36-d7c8-45e9-b2a1-73f89dc4e8bb
scenario_weighted_obs = map(1:n_scenarios) do i
    scenario_data = @chain scenario_mapping begin
        @subset :scenario_idx .== i
    end

    if nrow(scenario_data) > 0
        obs_vals = @chain scenario_data :observed_delay
        obs_counts = @chain scenario_data :n
        return (obs_vals, fweights(obs_counts))
    else
        return (Float64[], fweights(Int[]))  # Empty for unused scenarios
    end
end;

# ╔═╡ cba50a36-d7c8-45e9-b2a1-73f89dc4e8bc
md"### Instantiate and fit the improved model with fixed bounds

Create the model with bounds that will be fixed, then use `fix()` to set
the window parameters to their true values:"

# ╔═╡ dba50a36-d7c8-45e9-b2a1-73f89dc4e8bd
begin
    # Create the model with missing observations for simulation/fitting
    improved_model_unfixed = improved_double_censored_model(
        pwindow_bounds, swindow_bounds, obs_time_bounds)

    # Fix the window parameters to their true values from bounds
    true_pwindows = [pw[2] for pw in pwindow_bounds]  # Upper bounds
    true_swindows = [sw[2] for sw in swindow_bounds]  # Upper bounds
    true_obs_times = [(ot[1] + ot[2]) / 2 for ot in obs_time_bounds]  # Midpoints

    improved_model_fixed = fix(
        improved_model_unfixed,
        (
            @varname(pwindows) => true_pwindows,
            @varname(swindows) => true_swindows,
            @varname(obs_times) => true_obs_times
        )
    )
end;

# ╔═╡ eba50a36-d7c8-45e9-b2a1-73f89dc4e8be
md"Check what parameters are fixed:"

# ╔═╡ fba50a36-d7c8-45e9-b2a1-73f89dc4e8bf
DynamicPPL.fixed(improved_model_fixed)

# ╔═╡ 0ca50a36-d7c8-45e9-b2a1-73f89dc4e8c0
md"Now condition the model on our weighted observations and fit. This demonstrates
how weighted observations `(values, FrequencyWeights)` work with missing constructor weights:"

# ╔═╡ 1ca50a36-d7c8-45e9-b2a1-73f89dc4e8c1
improved_conditioned_model = improved_model_fixed | (obs = scenario_weighted_obs,);

# ╔═╡ 2ca50a36-d7c8-45e9-b2a1-73f89dc4e8c2
md"Fit the improved model:"

# ╔═╡ 3ca50a36-d7c8-45e9-b2a1-73f89dc4e8c3
improved_fit = sample(improved_conditioned_model, NUTS(), MCMCThreads(), 500, 4);

# ╔═╡ 4ca50a36-d7c8-45e9-b2a1-73f89dc4e8c4
summarize(improved_fit)

# ╔═╡ 5ca50a36-d7c8-45e9-b2a1-73f89dc4e8c5
plot_fit_with_truth(improved_fit, Dict("dist.mu" => meanlog, "dist.sigma" => sdlog))

# ╔═╡ 6ca50a36-d7c8-45e9-b2a1-73f89dc4e8c6
md"## Alternative patterns with StatsBase.AbstractWeights

The weight system also integrates with StatsBase.jl's AbstractWeights types.
This section demonstrates alternative approaches using FrequencyWeights."

# ╔═╡ 7ca50a36-d7c8-45e9-b2a1-73f89dc4e8c7
md"### Prepare data using StatsBase FrequencyWeights"

# ╔═╡ 8ca50a36-d7c8-45e9-b2a1-73f89dc4e8c8
begin
    # Extract values and weights for FrequencyWeights
    obs_values = simulated_counts.observed_delay
    obs_weights = simulated_counts.n

    # Create FrequencyWeights object
    weighted_obs = fweights(obs_weights)

    println("FrequencyWeights type: $(typeof(weighted_obs))")
    println("Values: $(obs_values[1:5])...")
    println("Weights: $(values(weighted_obs)[1:5])...")
end

# ╔═╡ 9ca50a36-d7c8-45e9-b2a1-73f89dc4e8c9
md"### Simple model demonstrating weight patterns

This demonstrates a simpler model using the weight vector approach.
Note: while FrequencyWeights are used in tuples above, direct AbstractWeights
integration would require additional `extract_obs` method definitions in the
Weighted.jl utilities:"

# ╔═╡ aca50a36-d7c8-45e9-b2a1-73f89dc4e8ca
@model function weights_pattern_model(y_values, weights_vec)
    dist ~ to_submodel(latent_delay_dist())

    # Create weighted observations - this pattern could be extended for AbstractWeights
    y_values ~ weight(dist, weights_vec)
end

# ╔═╡ bca50a36-d7c8-45e9-b2a1-73f89dc4e8cb
md"Fit the weights pattern model:"

# ╔═╡ cca50a36-d7c8-45e9-b2a1-73f89dc4e8cc
weights_model = weights_pattern_model(obs_values .+ 1e-6, obs_weights);

# ╔═╡ dca50a36-d7c8-45e9-b2a1-73f89dc4e8cd
weights_fit = sample(weights_model, NUTS(), MCMCThreads(), 500, 4);

# ╔═╡ eca50a36-d7c8-45e9-b2a1-73f89dc4e8ce
summarize(weights_fit)

# ╔═╡ fca50a36-d7c8-45e9-b2a1-73f89dc4e8cf
plot_fit_with_truth(weights_fit, Dict("dist.mu" => meanlog, "dist.sigma" => sdlog))

# ╔═╡ 0da50a36-d7c8-45e9-b2a1-73f89dc4e8d0
md"
## Summary: Weight conditioning improvements

This tutorial has demonstrated several key improvements to weight conditioning:

### 1. Fixed parameter patterns with `fix()`
- Window bounds can be fixed rather than estimated when they are known
- Use `fix(model, (@varname(param) => value, ...))` to set parameters
- This is more efficient than conditioning on parameter values

### 2. Weighted observation conditioning
- Use `(values, FrequencyWeights)` tuples to provide observations with counts
- Missing constructor weights `weight(distributions)` work with weighted observations
- Constructor weights and observation weights are combined via multiplication

### 3. Flexible weight specification
- Constructor weights: `weight(dist, constructor_weight)`
- Missing constructor weights: `weight(distributions)` for weighted observations
- Vector weights: `weight(dist, weight_vector)` for multiple observations
- Distribution vectors: `weight(dist_vector, weight_vector)`

### 4. Integration patterns with StatsBase
- StatsBase.AbstractWeights types provide structured weight handling
- FrequencyWeights, ProbabilityWeights, etc. offer semantic weight types
- Current implementation demonstrates tuple patterns `(values, FrequencyWeights)`
- Full AbstractWeights integration could be extended for additional flexibility

### 5. Consistent data handling with DataFramesMeta
- Use `@chain` and `@transform` for consistent data processing
- Extract values and weights using DataFramesMeta patterns
- Maintains code readability and consistency across model types

These improvements make model specification more flexible and efficient whilst
maintaining the same statistical semantics as the original implementation.
"

# ╔═╡ Cell order:
# ╟─30511a27-984e-40b7-9b1e-34bc87cb8d56
# ╟─bb9c75db-6638-48fe-afcb-e78c4bcc057d
# ╠═3690c122-d630-4fd0-aaf2-aea9226df086
# ╟─c5ec0d58-ce3d-4b0b-a261-dbd37b119f71
# ╠═b4409687-7bee-4028-824d-03b209aee68d
# ╟─30e99e77-aad1-43e8-9284-ab0bf8ae741f
# ╟─2fff24bf-74d3-47b8-be3f-f9d866d85903
# ╠═28bcd612-19f6-4e25-b6df-cb43df4f2a73
# ╠═04e414ab-c790-4d31-b216-18776534a287
# ╠═54700ad7-6b2a-440f-903a-c126b4c60c0e
# ╟─d3a7196a-185d-445b-afb7-99b546b1f72a
# ╠═7f82b991-00ca-4eed-994a-981c6d66454c
# ╟─767a58ed-9d7b-41db-a488-10f98a777474
# ╟─767a58ff-9d7b-41db-a488-10f98a777475
# ╠═767a5900-9d7b-41db-a488-10f98a777476
# ╟─767a5900-9d7b-41db-a488-10f98a777477
# ╠═767a5900-9d7b-41db-a488-10f98a777478
# ╟─d75a13b5-1c21-49c1-b10b-d26fcafc2736
# ╠═767a5900-9d7b-41db-a488-10f98a777479
# ╟─767a5900-9d7b-41db-a488-10f98a777480
# ╠═a8f036dd-fd59-4834-8422-f1ea7da616e0
# ╟─72f12b29-0779-4be8-aa35-9b22aa20c3b3
# ╠═35472e04-e096-4948-a218-3de53923f271
# ╟─b5598cc7-ddd1-4d90-af9b-110a518416ac
# ╟─2e04be98-625f-45f4-bf5e-a0074ea1ea01
# ╠═c548931f-f5e3-4de9-9183-eb64575b6bdb
# ╟─f3568b69-875d-494c-82cf-5a3db767cdaa
# ╠═8cbb8a46-c090-420f-bbb9-32b971a963f0
# ╟─5516cadb-f2f5-4852-8215-1493b001ab4d
# ╠═cf588dc1-3ac7-46a2-9fab-38d90aa391c5
# ╠═fcc1d4ba-13ca-41be-8451-7d035c8ff4a2
# ╟─2a0c4692-3ac5-4c46-9ac0-a057256a0b37
# ╠═a52a18c5-7625-4d9d-a7f7-bce5cb6ccb3f
# ╟─8e3ff244-c4d5-4562-99a5-7e63c6860a1e
# ╠═f4ed78e2-cdbb-4534-890a-fb346dd65f36
# ╟─50757759-9ec3-42d0-a765-df212642885a
# ╠═6fd01b5c-e374-4f5c-9f1c-ea75d06132af
# ╟─993f1f74-4a55-47a7-9e3e-c725cba13c0a
# ╠═ccd8dd8e-c361-43ba-b4f1-2444ec6008fc
# ╠═2b773594-5187-45bc-96f4-22a3d726b7d2
# ╠═a5b04acc-acc5-4d4d-8871-09d54caab185
# ╠═fb6dc898-21a9-4f8d-aa14-5b45974c2242
# ╟─91279812-9848-48bc-9258-b6f86c9fe923
# ╠═a257ce07-efbe-45e1-a8b0-ada40c29de8d
# ╟─49846128-379c-4c3b-9ec1-567ffa92e079
# ╠═4cf596f1-0042-4990-8d0a-caa8ba1db0c7
# ╟─71900c43-9f52-474d-adc7-becdc74045da
# ╠═cd26da77-02fb-4b65-bd7b-88060d0c97e8
# ╠═10278d0c-8c72-4c5f-b857-d3bc6ff2c242
# ╠═2c0b4f97-5953-497d-bca9-d1aa46c5150b
# ╟─7122bd53-81f6-4ea5-a024-86fdd7a7207a
# ╟─4cb137e3-93e9-43bf-a39f-b063dd6daac6
# ╠═c3afeed1-20ec-44c8-933c-ca0e75cda788
# ╟─4fc543fa-dca5-40c3-810b-979c536dfe0d
# ╠═6a274882-df7d-4972-80a6-ea62d932a906
# ╟─38790b6c-4fef-4b28-9442-6bfaab9d3c5a
# ╠═8e1764ec-345a-453a-830c-748c2a077eb7
# ╠═e0912175-6a02-480f-b8df-abd7c06f67e9
# ╟─08f819a5-db15-4016-9fd6-b1c4b56cb985
# ╠═01a37638-6494-4ce5-a02d-9d7f76f39ab7
# ╟─080c1bca-afcd-46c0-80b8-1708e8d05ae6
# ╠═a59e371a-b671-4648-984d-7bcaac367d32
# ╟─691e3d54-1a31-4686-a70d-711c2fc45dc1
# ╠═b5cd8b13-e3db-4ed1-80ce-e3ac1c57932c
# ╠═a53a78b3-dcbe-4b62-a336-a26e647dc8c8
# ╠═f0c02e4a-c0cc-41de-b1bf-f5fad7e7dfdb
# ╟─c045caa6-a44d-4a54-b122-1e50b1e0fe75
# ╟─1ba50a36-d7c8-45e9-b2a1-73f89dc4e8b3
# ╟─2ba50a36-d7c8-45e9-b2a1-73f89dc4e8b4
# ╠═3ba50a36-d7c8-45e9-b2a1-73f89dc4e8b5
# ╟─4ba50a36-d7c8-45e9-b2a1-73f89dc4e8b6
# ╠═5ba50a36-d7c8-45e9-b2a1-73f89dc4e8b7
# ╟─6ba50a36-d7c8-45e9-b2a1-73f89dc4e8b8
# ╠═7ba50a36-d7c8-45e9-b2a1-73f89dc4e8b9
# ╟─8ba50a36-d7c8-45e9-b2a1-73f89dc4e8b8
# ╠═9ba50a36-d7c8-45e9-b2a1-73f89dc4e8b9
# ╟─aba50a36-d7c8-45e9-b2a1-73f89dc4e8ba
# ╠═bba50a36-d7c8-45e9-b2a1-73f89dc4e8bb
# ╟─cba50a36-d7c8-45e9-b2a1-73f89dc4e8bc
# ╠═dba50a36-d7c8-45e9-b2a1-73f89dc4e8bd
# ╟─eba50a36-d7c8-45e9-b2a1-73f89dc4e8be
# ╠═fba50a36-d7c8-45e9-b2a1-73f89dc4e8bf
# ╟─0ca50a36-d7c8-45e9-b2a1-73f89dc4e8c0
# ╠═1ca50a36-d7c8-45e9-b2a1-73f89dc4e8c1
# ╟─2ca50a36-d7c8-45e9-b2a1-73f89dc4e8c2
# ╠═3ca50a36-d7c8-45e9-b2a1-73f89dc4e8c3
# ╠═4ca50a36-d7c8-45e9-b2a1-73f89dc4e8c4
# ╠═5ca50a36-d7c8-45e9-b2a1-73f89dc4e8c5
# ╟─6ca50a36-d7c8-45e9-b2a1-73f89dc4e8c6
# ╟─7ca50a36-d7c8-45e9-b2a1-73f89dc4e8c7
# ╠═8ca50a36-d7c8-45e9-b2a1-73f89dc4e8c8
# ╟─9ca50a36-d7c8-45e9-b2a1-73f89dc4e8c9
# ╠═aca50a36-d7c8-45e9-b2a1-73f89dc4e8ca
# ╟─bca50a36-d7c8-45e9-b2a1-73f89dc4e8cb
# ╠═cca50a36-d7c8-45e9-b2a1-73f89dc4e8cc
# ╠═dca50a36-d7c8-45e9-b2a1-73f89dc4e8cd
# ╠═eca50a36-d7c8-45e9-b2a1-73f89dc4e8ce
# ╠═fca50a36-d7c8-45e9-b2a1-73f89dc4e8cf
# ╟─0da50a36-d7c8-45e9-b2a1-73f89dc4e8d0
