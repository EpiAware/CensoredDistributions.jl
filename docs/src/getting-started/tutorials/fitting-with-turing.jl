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
    using Chain
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

1. Defining a Bayesian model that incorporates double censoring and right
   truncation
2. Generating synthetic data from the model using fixed parameters
3. Fitting a naive model that ignores censoring
4. Evaluating the naive model's performance
5. Fitting the full model that accounts for double censoring and right
   truncation.
6. Comparing the models' performance

## What might I need to know before starting

This tutorial builds on the concepts introduced in
[Getting Started with CensoredDistributions.jl](@ getting-started).

## Packages used
We use CairoMakie for plotting, Turing for probabilistic programming,
Chain.jl for data pipeline workflows, DataFrames, Random, and StatsBase.
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
4. Sample from the prior predictive distribution using `sample(model, Prior(), n)`
5. Extract the simulated observations from the resulting chain
6. Validate the simulated data with prior predictive checks
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

# ╔═╡ 767a58ff-9d7b-41db-a488-10f98a777475
md"### Define a reusable submodel for the latent delay distribution

To avoid code duplication across our models, we define a submodel that
encapsulates the latent delay distribution parameters. This pattern allows
us to reuse the same prior structure across all our models:"

# ╔═╡ 767a5900-9d7b-41db-a488-10f98a777476
@model function latent_delay_dist()
    mu ~ Normal(1.0, 2.0)
    sigma ~ truncated(Normal(0.5, 0.5); lower = 0.0)
    dist = LogNormal(mu, sigma)
    return dist
end

# ╔═╡ 767a5900-9d7b-41db-a488-10f98a777478
md"We can sample from this submodel directly to understand our prior beliefs
about the underlying delay distribution:"

# ╔═╡ 767a5900-9d7b-41db-a488-10f98a777479
begin
    # Sample from the latent delay distribution prior
    latent_prior_samples = sample(latent_delay_dist(), Prior(), 1000)

    # Visualize the prior distribution
    let
        f = pairplot(latent_prior_samples)
        CairoMakie.vlines!(f[1, 1], [meanlog], linewidth = 3, color = :red,
            label = "True μ")
        CairoMakie.vlines!(f[2, 2], [sdlog], linewidth = 3, color = :red,
            label = "True σ")
        f
    end
end

# ╔═╡ 767a5900-9d7b-41db-a488-10f98a777480
md"### Define the double censored model for simulation and fitting

Now we define our full model that incorporates double censoring and right
truncation. This model uses the `latent_delay_dist()` submodel via
`to_submodel()` to include the delay distribution parameters:"

# ╔═╡ 767a5901-9d7b-41db-a488-10f98a777477
@model function CensoredDistributions_model(y, n, pws, sws, Ds)
    dist = to_submodel(latent_delay_dist())

    pcens_dists = map(pws, Ds, sws) do pw, D, sw
        double_interval_censored(
            dist; primary_event = Uniform(0.0, pw), upper = D, interval = sw)
    end

    y ~ weight(pcens_dists, n)
    return y
end

# ╔═╡ 767a58ed-9d7b-41db-a488-10f98a777474
md"For each individual we now sample a primary and secondary event window as
well as a relative observation time (relative to their censored primary event)."

# ╔═╡ 35472e04-e096-4948-a218-3de53923f271
pwindows = rand(1:2, n);

# ╔═╡ 2d0ca6e6-0333-4aec-93d4-43eb9985dc14
swindows = rand(1:2, n);

# ╔═╡ 6465e51b-8d71-4c85-ba40-e6d230aa53b1
obs_times = rand(8:12, n);

# ╔═╡ b5598cc7-ddd1-4d90-af9b-110a518416ac
md"### Simulate from the double censored distribution for each individual"

# ╔═╡ 2e04be98-625f-45f4-bf5e-a0074ea1ea01
md"Using the double censored model, we simulate data by sampling from the
model using known true parameters. We use Turing's proper simulation approach
with DynamicPPL's `fix` function to set parameters to their true values and
sample from the prior predictive distribution."

# ╔═╡ f4ed7900-cdbb-4534-890a-fb346dd65f34
md"We use Turing's proper simulation approach with DynamicPPL's `fix` function.
This is the recommended way to simulate from Turing models as it leverages
the model's structure and handles the censoring process correctly:"

# ╔═╡ f4ed7901-cdbb-4534-890a-fb346dd65f35
# ╔═╡ f4ed7902-cdbb-4534-890a-fb346dd65f36
md"Now we simulate data using the true parameter values with Turing's
simulation approach. This ensures consistency between data generation
and model fitting, and demonstrates the proper way to simulate from
Turing models rather than manual approaches:"

# ╔═╡ f4ed78df-cdbb-4534-890a-fb346dd65f33
# Create model with missing y values for simulation
model_for_simulation = CensoredDistributions_model(
    missing, n, pwindows, swindows, obs_times)

# ╔═╡ f4ed78e0-cdbb-4534-890a-fb346dd65f34
# Fix parameters to their true values
fixed_model = fix(model_for_simulation, (mu = meanlog, sigma = sdlog))

# ╔═╡ f4ed78e1-cdbb-4534-890a-fb346dd65f35
# Sample from the prior predictive distribution
simulation_chain = sample(fixed_model, Prior(), 1)

# ╔═╡ f4ed78e2-cdbb-4534-890a-fb346dd65f36
# Extract the simulated y values and create a DataFrame with all specifications
simulated_data = @chain simulation_chain begin
    _[:y][1]
    DataFrame(
        observed_delay = _,
        pwindow = pwindows,
        swindow = swindows,
        obs_time = obs_times
    )
end;

# ╔═╡ f4ed7903-cdbb-4534-890a-fb346dd65f37
md"### Prior predictive checks using pairplot

First, let's visualise the prior predictive distribution by sampling from
the instantiated model (model_for_simulation) with uninformative priors and
compare against our true parameters. This shows what the model believes
before seeing any data. Note that this will be the same as the submodel
since we're using the same priors:"

# ╔═╡ f4ed7904-cdbb-4534-890a-fb346dd65f38
# Sample from prior predictive distribution using the instantiated model
prior_chain = sample(model_for_simulation, Prior(), 500)

# ╔═╡ f4ed7905-cdbb-4534-890a-fb346dd65f40
let
    f = pairplot(prior_chain)
    CairoMakie.vlines!(f[1, 1], [meanlog], linewidth = 3, color = :red,
        label = "True μ")
    CairoMakie.vlines!(f[2, 2], [sdlog], linewidth = 3, color = :red,
        label = "True σ")
    f
end

# ╔═╡ f4ed7906-cdbb-4534-890a-fb346dd65f41
md"### Data validation checks

We validate that our simulated data is consistent with the expected
distribution characteristics."

# ╔═╡ 50757759-9ec3-42d0-a765-df212642885a
md"Create a dataframe with the data we just generated aggregated to unique
combinations and count occurrences.
"

# ╔═╡ 5aed77d3-5798-4538-b3eb-3f4ce43d0423
delay_counts = @chain simulated_data begin
    @transform(:observed_delay_upper = :observed_delay .+ :swindow)
    @groupby(:pwindow, :swindow, :obs_time, :observed_delay,
        :observed_delay_upper)
    @combine(:n = length(:pwindow))
end

# ╔═╡ 993f1f74-4a55-47a7-9e3e-c725cba13c0a
md"""
Compare the samples with and without secondary censoring to the true
distribution. First let's calculate the empirical CDF:
"""

# ╔═╡ 5a6d605d-bff6-4b7d-97f0-ca35750411d3

# ╔═╡ ccd8dd8e-c361-43ba-b4f1-2444ec6008fc
empirical_cdf_obs = ecdf(delay_counts.observed_delay, weights = delay_counts.n);

# ╔═╡ 2b773594-5187-45bc-96f4-22a3d726b7d2
# Create a sequence of x values for the theoretical CDF
x_seq = range(minimum(simulated_data.observed_delay),
    stop = maximum(simulated_data.observed_delay), length = 100);

# ╔═╡ a5b04acc-acc5-4d4d-8871-09d54caab185
# Calculate theoretical CDF using true log-normal distribution
theoretical_cdf = @chain x_seq begin
    cdf.(true_dist, _)
end;

# Generate uncensored samples from the true distribution for comparison
uncensored_samples = rand(true_dist, n);
empirical_cdf_uncensored = ecdf(uncensored_samples);

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

# ╔═╡ 9c8aebbe-8606-41e7-8e86-23129b1cbc8d
md"""
We've aggregated the data to unique combinations of `pwindow`, `swindow`,
and `obs_time` and counted the number of occurrences of each `observed_delay`
for each combination. This is the data we will use to fit our model.
"""

# ╔═╡ 91279812-9848-48bc-9258-b6f86c9fe923
md"
## Fitting a naive model using Turing

We'll now fit a naive model that ignores the censoring process. This model
treats the observed delay data as if it came directly from the uncensored
delay distribution, providing a baseline for comparison.
"

# ╔═╡ a257ce07-efbe-45e1-a8b0-ada40c29de8d
@model function naive_model(y, n)
    dist = to_submodel(latent_delay_dist())
    y ~ weight(dist, n)
end

# ╔═╡ 49846128-379c-4c3b-9ec1-567ffa92e079
md"
Now lets instantiate this model with data
"

# ╔═╡ 4cf596f1-0042-4990-8d0a-caa8ba1db0c7
naive_mdl = naive_model(
    delay_counts.observed_delay .+ 1e-6, # Add a small constant to avoid log(0)
    delay_counts.n)

# ╔═╡ 82e9a2d9-1f00-4b52-b8c3-c824c8ab10c5

# ╔═╡ 71900c43-9f52-474d-adc7-becdc74045da
md"
and now let's fit the compiled model.
"

# ╔═╡ cd26da77-02fb-4b65-bd7b-88060d0c97e8
naive_fit = sample(naive_mdl, NUTS(), MCMCThreads(), 500, 4);

# ╔═╡ 10278d0c-8c72-4c5f-b857-d3bc6ff2c242
summarize(naive_fit)

# ╔═╡ 2c0b4f97-5953-497d-bca9-d1aa46c5150b
let
    f = pairplot(naive_fit)
    vlines!(f[1, 1], [meanlog], linewidth = 4)
    vlines!(f[2, 2], [sdlog], linewidth = 4)
    f
end

# ╔═╡ 7122bd53-81f6-4ea5-a024-86fdd7a7207a
md"
We see that the model has converged and the diagnostics look good. However,
just from the model posterior summary we see that we might not be very happy
with the fit. `mu` is smaller than the target $(meanlog) and `sigma` is
larger than the target $(sdlog).
"

# ╔═╡ 7c8e3f12-a1b4-4c5d-8e9f-1234567890ab
md"
## Fitting a truncation-adjusted interval model

Now let's fit an intermediate model that accounts for interval censoring and
right truncation but ignores the primary censoring process. This provides
a comparison point between the naive model and the full model.
"

# ╔═╡ 8d9f4023-b2c5-4d6e-9f0g-2345678901bc
@model function interval_only_model(y, n, sws, Ds)
    dist = to_submodel(latent_delay_dist())

    icens_dists = map(sws, Ds) do sw, D
        interval_censored(
            truncated(dist; upper = D), sw
        )
    end

    y ~ weight(icens_dists, n)
    return y
end

# ╔═╡ 9e0f5134-c3d6-4e7f-a10h-3456789012cd
md"
Instantiate the interval-only model with our observed data:
"

# ╔═╡ af105245-d4e7-4f80-b11i-4567890123de
interval_only_mdl = interval_only_model(
    delay_counts.observed_delay,
    delay_counts.n,
    delay_counts.swindow,
    delay_counts.obs_time
)

# ╔═╡ b0215356-e5f8-5091-c22j-5678901234ef
md"
Fit the interval-only model:
"

# ╔═╡ c1326467-f6f9-51a2-d33k-6789012345f0
interval_only_fit = sample(interval_only_mdl, NUTS(), MCMCThreads(), 500, 4);

# ╔═╡ d2437578-07fa-62b3-e44l-7890123456g1
summarize(interval_only_fit)

# ╔═╡ e3548689-18fb-73c4-f55m-8901234567h2
let
    f = pairplot(interval_only_fit)
    vlines!(f[1, 1], [meanlog], linewidth = 4, color = :green)
    vlines!(f[2, 2], [sdlog], linewidth = 4, color = :green)
    f
end

# ╔═╡ 080c1bca-afcd-46c0-80b8-1708e8d05ae6
md"## Fitting the full CensoredDistributions model

Now we'll fit the full model that accounts for the censoring process.
Since the CensoredDistributions_model was defined earlier and used for
simulation, we'll reuse it for fitting - demonstrating the consistency
of our approach."

# ╔═╡ 825227da-5788-4bbd-8546-2d8a30996aaa
md"The model uses the same `@submodel` pattern as the other models,
ensuring consistent parameter priors across all approaches."

# ╔═╡ e24c231a-0bf3-4a03-a307-2ab43cdbecf4
md"
Then we instantiate this model with our observed data.
"

# ╔═╡ a59e371a-b671-4648-984d-7bcaac367d32
CensoredDistributions_mdl = CensoredDistributions_model(
    delay_counts.observed_delay,
    delay_counts.n,
    delay_counts.pwindow,
    delay_counts.swindow,
    delay_counts.obs_time
)

# ╔═╡ 691e3d54-1a31-4686-a70d-711c2fc45dc1
md"
Now we fit the model (*Note: `Turing.jl` supports a wide range of fitting
methods but here we use the No-U-turn sampler*) to recover the true
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
let
    f = pairplot(CensoredDistributions_fit)
    CairoMakie.vlines!(f[1, 1], [meanlog], linewidth = 3)
    CairoMakie.vlines!(f[2, 2], [sdlog], linewidth = 3)
    f
end

# ╔═╡ c045caa6-a44d-4a54-b122-1e50b1e0fe75
md"
We see that the model has converged and the diagnostics look good.
We also see that the posterior means are near the true parameters and the
90% credible intervals include the true parameters.
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
# ╟─767a5900-9d7b-41db-a488-10f98a777478
# ╠═767a5900-9d7b-41db-a488-10f98a777479
# ╟─767a5900-9d7b-41db-a488-10f98a777480
# ╠═767a5901-9d7b-41db-a488-10f98a777477
# ╠═35472e04-e096-4948-a218-3de53923f271
# ╠═2d0ca6e6-0333-4aec-93d4-43eb9985dc14
# ╠═6465e51b-8d71-4c85-ba40-e6d230aa53b1
# ╟─b5598cc7-ddd1-4d90-af9b-110a518416ac
# ╟─2e04be98-625f-45f4-bf5e-a0074ea1ea01
# ╠═f4ed78df-cdbb-4534-890a-fb346dd65f33
# ╠═f4ed78e0-cdbb-4534-890a-fb346dd65f34
# ╠═f4ed78e1-cdbb-4534-890a-fb346dd65f35
# ╠═f4ed78e2-cdbb-4534-890a-fb346dd65f36
# ╟─f4ed7903-cdbb-4534-890a-fb346dd65f37
# ╠═f4ed7904-cdbb-4534-890a-fb346dd65f38
# ╠═f4ed7905-cdbb-4534-890a-fb346dd65f40
# ╟─50757759-9ec3-42d0-a765-df212642885a
# ╠═5aed77d3-5798-4538-b3eb-3f4ce43d0423
# ╟─993f1f74-4a55-47a7-9e3e-c725cba13c0a
# ╠═ccd8dd8e-c361-43ba-b4f1-2444ec6008fc
# ╠═2b773594-5187-45bc-96f4-22a3d726b7d2
# ╠═a5b04acc-acc5-4d4d-8871-09d54caab185
# ╠═fb6dc898-21a9-4f8d-aa14-5b45974c2242
# ╟─9c8aebbe-8606-41e7-8e86-23129b1cbc8d
# ╟─91279812-9848-48bc-9258-b6f86c9fe923
# ╠═a257ce07-efbe-45e1-a8b0-ada40c29de8d
# ╟─49846128-379c-4c3b-9ec1-567ffa92e079
# ╠═4cf596f1-0042-4990-8d0a-caa8ba1db0c7
# ╠═82e9a2d9-1f00-4b52-b8c3-c824c8ab10c5
# ╟─71900c43-9f52-474d-adc7-becdc74045da
# ╠═cd26da77-02fb-4b65-bd7b-88060d0c97e8
# ╠═10278d0c-8c72-4c5f-b857-d3bc6ff2c242
# ╠═2c0b4f97-5953-497d-bca9-d1aa46c5150b
# ╟─7122bd53-81f6-4ea5-a024-86fdd7a7207a
# ╟─7c8e3f12-a1b4-4c5d-8e9f-1234567890ab
# ╠═8d9f4023-b2c5-4d6e-9f0g-2345678901bc
# ╟─9e0f5134-c3d6-4e7f-a10h-3456789012cd
# ╠═af105245-d4e7-4f80-b11i-4567890123de
# ╟─b0215356-e5f8-5091-c22j-5678901234ef
# ╠═c1326467-f6f9-51a2-d33k-6789012345f0
# ╠═d2437578-07fa-62b3-e44l-7890123456g1
# ╠═e3548689-18fb-73c4-f55m-8901234567h2
# ╟─080c1bca-afcd-46c0-80b8-1708e8d05ae6
# ╠═825227da-5788-4bbd-8546-2d8a30996aaa
# ╟─e24c231a-0bf3-4a03-a307-2ab43cdbecf4
# ╠═a59e371a-b671-4648-984d-7bcaac367d32
# ╟─691e3d54-1a31-4686-a70d-711c2fc45dc1
# ╠═b5cd8b13-e3db-4ed1-80ce-e3ac1c57932c
# ╠═a53a78b3-dcbe-4b62-a336-a26e647dc8c8
# ╠═f0c02e4a-c0cc-41de-b1bf-f5fad7e7dfdb
# ╟─c045caa6-a44d-4a54-b122-1e50b1e0fe75
