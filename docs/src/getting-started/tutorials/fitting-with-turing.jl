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
6. Fitting a model that accounts for secondary event censoring and
   truncation but not primary event censoring.
7. Fitting the full model that accounts for double censoring and right
   truncation.
8. Using improved weight conditioning with joint observations and
   fix() patterns
9. Demonstrating StatsBase.AbstractWeights integration patterns

## What might I need to know before starting

This tutorial builds on the concepts introduced in
[Getting Started with CensoredDistributions.jl](@ref getting-started).

## Packages used
We use CairoMakie for plotting, Turing for probabilistic programming,
FlexiChains for working with MCMC output, Chain.jl for data pipeline
workflows, DataFramesMeta, Random, and StatsBase.
"""

using DataFramesMeta
using Turing
using DynamicPPL
using Distributions
using Random
using CairoMakie, PairPlots
using StatsBase
using FlexiChains: FlexiChains, VNChain, Prefixed, @varname, summarystats
using CensoredDistributions

md"""
## Generate synthetic data using Turing model simulation

We'll generate synthetic data by simulating from our Turing model with
known true parameters. This approach ensures consistency between the
data generation process and the model we'll use for inference,
demonstrating how Turing models can be used for both simulation and
fitting.

The proper Turing simulation approach:
1. Define a Turing model that incorporates double censoring and right
   truncation
2. Create a model instance with missing observations for simulation
3. Use DynamicPPL's `fix` function to set parameters to their true
   values
4. Sample from the prior predictive distribution by calling the model
   as a function
"""

md"""
### Define the true parameters for generating synthetic data
"""

md"""
We start by defining the number of samples and the true parameters
of the lognormal.
"""

n = 2000;

meanlog = 1.5;

sdlog = 0.75;

md"""
Now we can define a lognormal distribution using Distributions.jl.
"""

true_dist = LogNormal(meanlog, sdlog);

md"""
For each individual we now sample a primary and secondary event
window as well as a relative observation time (relative to their
censored primary event).
"""

md"""
### Define a reusable submodel for the latent delay distribution

To avoid code duplication across our models, we define a submodel
that encapsulates the latent delay distribution parameters. This
pattern allows us to reuse the same prior structure across all our
models:
"""

@model function latent_delay_dist()
    mu ~ Normal(1.0, 2.0);
    sigma ~ truncated(Normal(0.5, 1); lower = 0.0)
    return LogNormal(mu, sigma)
end

md"""
and define a helper function to standardise our pairplot
visualisations across all model fits. We sample with
`chain_type = VNChain` to get a FlexiChains `VNChain`, and use
`Prefixed` so that the same helper works for both the plain
`latent_delay_dist()` chain and chains where `latent_delay_dist()`
has been used as a submodel (and the parameters are automatically
prefixed, e.g. `dist.mu`):
"""

function plot_fit_with_truth(chain, truth_nt)
    samples_nt = (;
        mu = vec(chain[Prefixed(@varname(mu))]),
        sigma = vec(chain[Prefixed(@varname(sigma))])
    )
    f = pairplot(
        samples_nt,
        PairPlots.Truth(
            truth_nt,
            label = "True Values"
        )
    )
    return f
end

md"""
### Prior predictive checks using pairplot

First, let's visualise the prior predictive distribution by sampling
from the instantiated model with uninformative priors and comparing
against our true parameters. This shows what the model believes
before seeing any data.
"""

Random.seed!(123);

## Sample from the latent delay distribution prior
latent_prior_samples = sample(
    latent_delay_dist(), Prior(), 1000; chain_type = VNChain
)

## Visualise the prior distribution
plot_fit_with_truth(
    latent_prior_samples,
    (; mu = meanlog, sigma = sdlog)
)

md"""
### Define the double censored model for simulation and fitting

Now we define our full model that incorporates double censoring and
right truncation. This model uses the `latent_delay_dist()` submodel
via `to_submodel()` to include the delay distribution parameters.
It also uses our `double_interval_censored()` function to define
each double censored and right truncated delay:
"""

@model function CensoredDistributions_model(
        pwindow_bounds, swindow_bounds, obs_time_bounds
)
    pwindows ~ product_distribution(
        [DiscreteUniform(pw[1], pw[2]) for pw in pwindow_bounds]
    )
    swindows ~ product_distribution(
        [DiscreteUniform(sw[1], sw[2]) for sw in swindow_bounds]
    )
    obs_times ~ product_distribution(
        [DiscreteUniform(ot[1], ot[2]) for ot in obs_time_bounds]
    )

    dist ~ to_submodel(latent_delay_dist())

    pcens_dists = map(
        pwindows, obs_times, swindows
    ) do pw, D, sw
        pe = Uniform(0, pw)
        double_interval_censored(
            dist;
            primary_event = pe,
            upper = D,
            interval = sw
        )
    end

    obs ~ weight(pcens_dists)
end

md"""
We also need to define our simulated observation window bounds for
each observed delay as well as the bounds on the amount of censored
time in which events have been observed (required to adjust for
truncation). We will then combine these bounds with the model in
order to simulate data.
"""

bounds_df = DataFrame(
    pwindow_bounds = fill((1, 3), n),
    swindow_bounds = fill((1, 3), n),
    obs_time_bounds = fill((8, 12), n)
);

md"""
### Simulate from the double censored distribution for each
individual
"""

md"""
Using the double censored model, we simulate data by sampling from
the model using known true parameters. We use Turing's simulation
approach with DynamicPPL's `fix` function to set parameters to their
true values and sample from the prior predictive distribution. This
means we can use the same model for simulation and inference.
"""

md"""
We first create the base model which only specifies bounds to sample
for the observations processes. We'll use this for both simulation
and fitting:
"""

base_model = @with bounds_df begin
    CensoredDistributions_model(
        :pwindow_bounds,
        :swindow_bounds,
        :obs_time_bounds
    )
end

md"""
For simulation, fix the distribution parameters to known true
values:
"""

simulation_model = fix(
    base_model,
    (
        @varname(dist.mu) => meanlog,
        @varname(dist.sigma) => sdlog
    )
)

md"""
Now we can sample from the model using `rand` to get simulated
observations with their observation windows and relative observation
time:
"""

simulated_data = @chain simulation_model begin
    rand
    NamedTuple
    DataFrame
end;

md"""
### Visualise the simulated data

To make handling the data easier and later to speed up our models we
first create a dataframe with the data we just generated, aggregated
to unique combinations and count occurrences.
"""

simulated_counts = @chain simulated_data begin
    @transform :obs_upper = :obs .+ :swindows
    @groupby All()
    @combine :n = length(:pwindows)
end;

md"""
Now let's compare the samples with and without double interval
censoring to the true distribution. First let's calculate the
empirical CDF:
"""

empirical_cdf_obs = @with(simulated_counts, ecdf(:obs, weights = :n));

## Create a sequence of x values for the theoretical CDF
x_seq = @with simulated_counts begin
    range(
        minimum(:obs),
        stop = maximum(:obs) + 2,
        length = 100
    );
end;

## Calculate theoretical CDF using true log-normal distribution
theoretical_cdf = @chain x_seq begin
    cdf.(true_dist, _)
end;

## Generate uncensored samples from the true distribution
uncensored_samples = rand(true_dist, n);
empirical_cdf_uncensored = ecdf(uncensored_samples);

f = Figure()
ax = Axis(
    f[1, 1],
    title = "Censored vs Uncensored vs Theoretical CDF",
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
lines!(
    ax, x_seq, theoretical_cdf,
    label = "Theoretical CDF",
    color = :black, linewidth = 2
)
vlines!(
    ax, [mean(simulated_data.obs)],
    color = :blue, linestyle = :dash,
    label = "Censored mean", linewidth = 2
)
vlines!(
    ax, [mean(uncensored_samples)],
    color = :red, linestyle = :dash,
    label = "Uncensored mean", linewidth = 2
)
vlines!(
    ax, [mean(true_dist)],
    linestyle = :dash,
    label = "Theoretical mean",
    color = :black, linewidth = 2
)
axislegend(position = :rb)
f

md"""
## Fitting a naive model using Turing

We'll now fit a naive model that ignores the censoring process. This
model treats the observed delay data as if it came directly from the
uncensored delay distribution, providing a baseline for comparison.
"""

@model function naive_model()
    dist ~ to_submodel(latent_delay_dist())
    obs ~ weight(dist)
end

md"""
Now let's instantiate and condition this model using weighted
observations. We use a small constant to avoid issues at zero (a
hint that this model is misspecified) and condition directly using
NamedTuple format `(values = values, weights = counts)` which
enables joint observation conditioning.
"""

naive_mdl = @with simulated_counts begin
    condition(
        naive_model(),
        obs = (values = :obs .+ 1e-6, weights = :n)
    )
end

md"""
Now let's fit the conditioned model using the joint observation
pattern `(values = values, weights = counts)`.
"""

naive_fit = sample(
    naive_mdl, NUTS(), MCMCThreads(), 500, 4;
    chain_type = VNChain
);

summarystats(naive_fit)

plot_fit_with_truth(
    naive_fit,
    (; mu = meanlog, sigma = sdlog)
)

md"""
We see that the model has converged and the diagnostics look good.
However, just from the model posterior summary we see that we might
not be very happy with the fit. `mu` is smaller than the target
1.5 and `sigma` is larger than the target 0.75.
"""

md"""
## Fitting a truncation-adjusted interval model

Now let's fit an intermediate model that accounts for interval
censoring and right truncation but ignores the primary censoring
process. This provides a comparison point between the naive model
and the full model.
"""

@model function interval_only_model(
        swindow_bounds, obs_time_bounds
)
    swindows ~ product_distribution(
        [Uniform(sw[1], sw[2]) for sw in swindow_bounds]
    )
    obs_times ~ product_distribution(
        [Uniform(ot[1], ot[2]) for ot in obs_time_bounds]
    )

    dist ~ to_submodel(latent_delay_dist())

    icens_dists = map(obs_times, swindows) do D, sw
        truncated(
            interval_censored(dist, sw), upper = D
        )
    end
    obs ~ weight(icens_dists)
    return obs
end

md"""
Create the interval-only model with bounds, fix the window
parameters, and condition on observations
"""

interval_only_mdl = @with simulated_counts begin
    @chain interval_only_model(
        bounds_df.swindow_bounds,
        bounds_df.obs_time_bounds
    ) begin
        fix((
            @varname(swindows) => :swindows,
            @varname(obs_times) => :obs_times
        ))
        condition(
            obs = (values = :obs, weights = :n)
        )
    end
end;

md"""
Fit the interval-only model (*Note: `Turing.jl` supports a wide
range of fitting methods but here we use the No-U-turn sampler*):
"""

interval_only_fit = sample(
    interval_only_mdl, NUTS(), MCMCThreads(), 500, 4;
    chain_type = VNChain
);

summarystats(interval_only_fit)

md"""
Lets plot the posterior compared to the true values again. `to_submodel()`
automatically prefixes the LHS name to all variables in the inner model
(so `mu` becomes `dist.mu`). Using FlexiChains' `Prefixed` wrapper in
`plot_fit_with_truth` handles that transparently, so the same helper
works for both the plain prior chain and the prefixed posterior chain.
"""

plot_fit_with_truth(
    interval_only_fit,
    (; mu = meanlog, sigma = sdlog)
)

md"""
## Fitting the double censored model

Now we'll fit the full model that accounts for the censoring
process. Since the CensoredDistributions_model was defined earlier
and used for simulation, we'll reuse it for fitting. Here we fix
the censoring windows and observation time based on the observed
data and then condition on the weighted observations.
"""

CensoredDistributions_mdl = @with simulated_counts begin
    @chain base_model begin
        fix((
            @varname(pwindows) => :pwindows,
            @varname(swindows) => :swindows,
            @varname(obs_times) => :obs_times
        ))
        condition(
            obs = (values = :obs, weights = :n)
        )
    end
end;

CensoredDistributions_mdl()

md"""
Now we fit the model to recover the true parameters from the
synthetic data we generated earlier. This demonstrates the
package's ability to perform accurate parameter recovery when
the censoring process is properly modelled.
"""

CensoredDistributions_fit = sample(
    CensoredDistributions_mdl,
    NUTS(), MCMCThreads(), 1000, 4;
    chain_type = VNChain
);

summarystats(CensoredDistributions_fit)

plot_fit_with_truth(
    CensoredDistributions_fit,
    (; mu = meanlog, sigma = sdlog)
)

md"""
We see that the model has converged and the diagnostics look good.
We also see that the posterior means are near the true parameters
and the 90% credible intervals include the true parameters.
"""
