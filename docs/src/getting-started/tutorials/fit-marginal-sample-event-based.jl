md"""
# Fit marginal, sample event based

## Introduction

### What are we going to do in this how-to

A common workflow is to fit a delay model in its efficient **marginal** form
(the primary event integrated out, no extra latent dimensions), then to obtain
**event-based** draws: the full internal event times (the primary event time and
the observed time), either for fresh records or recovered for the records you
fit.

This works because the marginal and latent forms are one family sharing the same
parameters, together with the marginal-equals-latent equivalence (the latent
representation integrates to the marginal `logpdf`). The marginal-fit posterior
therefore drops straight into the latent form.

`predict_events` provides this in two flavours, dispatched by what you pass:

1. **Forward simulation (Turing-free):** `predict_events(d, ...)` simulates full
   event paths directly from a latent distribution via `rand`. No `@model`
   needed. Use this to generate fresh event paths from parameters (the
   new-record posterior-predictive path).
2. **Latent recovery (Turing):** `predict_events(chain, model)` recovers the
   observed records' integrated-out latent event times by running the latent
   form of the fitted model over the posterior chain.

### What might I need to know before starting

This how-to builds on [Fitting CensoredDistributions.jl modified distributions
with Turing.jl](@ref). It uses the marginal and latent submodels from
`primary_censored_model` and the [`latent`](@ref) wrapper.

## Packages used

We use Turing for probabilistic programming, Distributions for the delay
distribution, Random for reproducibility, and FlexiChains for the chain output.
"""

using CensoredDistributions
using Distributions
using Turing
using DynamicPPL: prefix, @varname
using FlexiChains: Parameter
using Random
using Statistics

md"""
## Simulate data with known latent event times

Each record has a primary event time drawn uniformly in the window `[0, 1]` and
an observed time equal to that primary plus a log-normal delay. We keep the true
primaries so we can check the recovery later.
"""

rng = MersenneTwister(42)

true_meanlog = 1.4;

true_sdlog = 0.5;

n = 60;

true_primaries = rand(rng, Uniform(0, 1), n);

observed = true_primaries .+ rand(rng, LogNormal(true_meanlog, true_sdlog), n);

md"""
## Fit the efficient marginal form

The marginal model scores each record through the marginal `primary_censored`
distribution; the primary event time is integrated out inside `logpdf`, so there
are no per-record latent dimensions to sample.
"""

@model function fit_marginal(y)
    mu ~ Normal(1.4, 0.5)
    sigma ~ truncated(Normal(0.5, 0.3); lower = 0.05)
    d = primary_censored(LogNormal(mu, sigma), Uniform(0, 1))
    for i in eachindex(y)
        obs ~ to_submodel(
            prefix(primary_censored_model(d, y[i]), Symbol(:rec, i)), false)
    end
end

chain = sample(rng, fit_marginal(observed), NUTS(), 200; progress = false)

md"""
The posterior concentrates near the true parameters.
"""

mu_post = vec(chain[Parameter(@varname(mu))]);

@assert abs(mean(mu_post) - true_meanlog) < 0.3

md"""
## Flavour 1: forward-simulate fresh event paths (Turing-free)

`predict_events(d, ...)` draws full event paths `[primary, observed]` directly
from a latent distribution. Build the latent distribution from a posterior draw
and simulate, with no model or conditioning. This is the new-record
posterior-predictive path.
"""

sigma_post = vec(chain[Parameter(@varname(sigma))]);

post_draw = (mu = mean(mu_post), sigma = mean(sigma_post));

ld = latent(primary_censored(
    LogNormal(post_draw.mu, post_draw.sigma), Uniform(0, 1)));

one_path = predict_events(ld; rng = MersenneTwister(1))

md"""
A single path is `[primary, observed]`: the primary lies in the window and the
observed time exceeds it (a positive delay).
"""

@assert 0 <= one_path[1] <= 1
@assert one_path[2] > one_path[1]

md"""
We can draw many paths at once, or push a set of posterior draws through the
latent form by rebuilding the distribution per draw.
"""

many_paths = predict_events(ld, 500; rng = MersenneTwister(2));

@assert all(p -> p[2] > p[1], many_paths)

build(p) = latent(primary_censored(
    LogNormal(p.mu, p.sigma), Uniform(0, 1)))

param_draws = [(mu = m, sigma = s)
               for (m, s) in zip(mu_post[1:50], sigma_post[1:50])];

per_draw_paths = predict_events(build, param_draws; rng = MersenneTwister(3));

@assert length(per_draw_paths) == length(param_draws)

md"""
## Flavour 2: recover the observed records' latent event times (Turing)

`predict_events(chain, model)` takes the marginal-fit `chain` and the **latent**
form of the same model carrying the observed data. It runs the latent model
conditioned on each posterior draw, re-sampling the integrated-out latents (here
the per-record primary event time). The latent model uses the same parameter
names as the marginal fit, so the posterior drops straight in.
"""

@model function latent_recovery(y)
    mu ~ Normal(1.4, 0.5)
    sigma ~ truncated(Normal(0.5, 0.3); lower = 0.05)
    d = latent(primary_censored(LogNormal(mu, sigma), Uniform(0, 1)))
    for i in eachindex(y)
        x ~ to_submodel(
            prefix(primary_censored_model(d, y[i]), Symbol(:rec, i)), false)
    end
end

events = predict_events(chain, latent_recovery(observed))

md"""
The recovered variables are the per-record latent primaries `rec_i.p` (only the
primary is sampled; the observed time is conditioned on the data).

### Consistency check

The marginal-equals-latent equivalence means every recovered primary must lie in
the window `[0, 1]` and reproduce the observed delay gap as a positive value
(`observed_i - p_i > 0`). We check this for every record.
"""

for i in 1:n
    p_i = vec(events[Parameter(@varname($(Symbol("rec", i)).p))])
    @assert all(0 .<= p_i .<= 1)
    @assert all(observed[i] .- p_i .> 0)
end

md"""
Note that recovery here gives the posterior over each latent primary *given the
observed total delay and the fitted parameters*; it is not an attempt to pin down
the exact simulated primary, which is not identified from the total alone. The
recovered primaries are nonetheless consistent: each stays in the window and
leaves a positive observed delay, as the marginal-equals-latent equivalence
requires.

## Summary

- Fit once in the cheap marginal form.
- `predict_events(d, ...)` forward-simulates fresh event paths from parameters,
  Turing-free.
- `predict_events(chain, model)` recovers the latent event times of the records
  you fit, conditioned on the data over the posterior.
- Both rely on the marginal and latent forms being one family that share
  parameters, so a marginal-fit posterior drops straight into the latent form.
"""
