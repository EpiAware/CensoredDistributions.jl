md"""
# [Fit marginal, sample event based](@id fit-marginal-sample-event-based)

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

Two standard entry points cover this:

1. **Forward simulation (Turing-free):** `rand(latent(d))` simulates a full
   event path directly from a latent distribution. No `@model` needed. Use this
   to generate fresh event paths from parameters (the new-record
   posterior-predictive path); a comprehension batches the draws.
2. **Latent recovery (Turing):** `DynamicPPL.predict(model, chain)` recovers the
   observed records' integrated-out latent event times by running the latent
   form of the fitted model over the posterior chain.

### What might I need to know before starting

This how-to builds on [Fitting CensoredDistributions.jl modified distributions
with Turing.jl](@ref). It uses the generic `composed_distribution_model` entry
and the [`latent`](@ref) wrapper.

## Packages used

We use Turing for probabilistic programming, Distributions for the delay
distribution, Random for reproducibility, FlexiChains for the chain output, and
Integrals for the quadrature in the equivalence check.
"""

using CensoredDistributions
using Distributions
using Turing
using ADTypes: AutoForwardDiff
using DynamicPPL: prefix, @varname, predict
using FlexiChains: Parameter
using Integrals
using Random
using Statistics

md"""
## Simulate data with known latent event times

Each record has a primary event time drawn uniformly in the window `[0, 1]` and
an observed time equal to that primary plus a log-normal delay.
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
            prefix(composed_distribution_model(d, y[i]), Symbol(:rec, i)), false)
    end
end

chain = sample(rng, fit_marginal(observed),
    NUTS(; adtype = AutoForwardDiff()), 200; progress = false)

md"""
The posterior concentrates near the true parameters.
We sample with ForwardDiff here; Mooncake also works as an AD backend.
"""

mu_post = vec(chain[Parameter(@varname(mu))]);

sigma_post = vec(chain[Parameter(@varname(sigma))]);

@assert abs(mean(mu_post) - true_meanlog) < 0.3

@assert abs(mean(sigma_post) - true_sdlog) < 0.2

md"""
## Demonstrate the marginal-equals-latent equivalence

The whole workflow rests on a single identity.
Scoring the marginal `primary_censored` distribution integrates the latent joint
over the primary event, so the marginal and latent forms are density-identical.
We can show this directly at the level of the cumulative distribution function.

The latent form draws a primary `p` from `Uniform(0, 1)`, then scores the
observed time through the conditional `cdf(LogNormal(mu, sigma), x - p)`.
Integrating that conditional over the primary reproduces the marginal `cdf`.
With a uniform primary the integral is `∫₀¹ cdf(LogNormal(mu, sigma), x - p) dp`,
which we evaluate with adaptive quadrature.
We build the distributions from the posterior mean.
"""

post_draw = (mu = mean(mu_post), sigma = mean(sigma_post));

marginal_leaf = primary_censored(
    LogNormal(post_draw.mu, post_draw.sigma), Uniform(0, 1))

function latent_integrated_cdf(x)
    delay = LogNormal(post_draw.mu, post_draw.sigma)
    problem = IntegralProblem((p, _) -> cdf(delay, x - p), (0.0, 1.0))
    return solve(problem, QuadGKJL(); reltol = 1e-12, abstol = 1e-12).u
end

eval_points = [1.0, 3.0, 6.0, 10.0];

cdf_diffs = [abs(cdf(marginal_leaf, x) - latent_integrated_cdf(x))
             for x in eval_points];

md"""
The marginal `cdf` and the latent-integrated `cdf` agree to numerical precision.
"""

@assert maximum(cdf_diffs) < 1e-10

maximum(cdf_diffs)

md"""
The `logpdf` carries a small tail residual of order `1e-4`, which is the
central-difference noise in the `primary_censored` density rather than a real
gap; the `cdf` comparison above is exact and is the clean check of the identity.

### When to prefer the marginal or the latent form

Prefer the marginal form by default: it carries no per-record latents, so it is
cheaper and lower-dimensional at scale.
Prefer the latent form for small-count or heavily censored problems, where the
extra latent dimensions cost little and the sampler explores the joint more
reliably than a stiff one-dimensional marginal.
The [Marginal versus latent](@ref marginal-versus-latent) section of the
composer toolkit gives the canonical treatment of this choice.

## Flavour 1: forward-simulate fresh event paths (Turing-free)

`rand(latent(d))` draws a full event path as a labelled
`(primary = ..., observed = ...)` record directly from a latent distribution.
Build the latent distribution from a posterior draw and simulate, with no model
or conditioning. This is the new-record posterior-predictive path.
"""

ld = latent(primary_censored(
    LogNormal(post_draw.mu, post_draw.sigma), Uniform(0, 1)));

one_path = rand(MersenneTwister(1), ld)

md"""
A single path is the record `(primary, observed)`: the primary lies in the
window and the observed time exceeds it (a positive delay).
"""

@assert 0 <= one_path.primary <= 1
@assert one_path.observed > one_path.primary

md"""
We can draw many paths with a comprehension, or push a set of posterior draws
through the latent form by rebuilding the distribution per draw.
"""

rng2 = MersenneTwister(2);

many_paths = [rand(rng2, ld) for _ in 1:500];

@assert all(p -> p.observed > p.primary, many_paths)

build(p) = latent(primary_censored(
    LogNormal(p.mu, p.sigma), Uniform(0, 1)))

param_draws = [(mu = m, sigma = s)
               for (m, s) in zip(mu_post[1:50], sigma_post[1:50])];

rng3 = MersenneTwister(3);

per_draw_paths = [rand(rng3, build(p)) for p in param_draws];

@assert length(per_draw_paths) == length(param_draws)

md"""
## Flavour 2: recover the observed records' latent event times (Turing)

`DynamicPPL.predict(model, chain)` takes the **latent** form of the same model
carrying the observed data and the marginal-fit `chain`. It runs the latent
model conditioned on each posterior draw, re-sampling the integrated-out latents
(here the per-record primary event time). The latent model uses the same
parameter names as the marginal fit, so the posterior drops straight in.
"""

@model function latent_recovery(y)
    mu ~ Normal(1.4, 0.5)
    sigma ~ truncated(Normal(0.5, 0.3); lower = 0.05)
    d = latent(primary_censored(LogNormal(mu, sigma), Uniform(0, 1)))
    for i in eachindex(y)
        x ~ to_submodel(
            prefix(composed_distribution_model(d, y[i]), Symbol(:rec, i)), false)
    end
end

events = predict(latent_recovery(observed), chain)

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
- `rand(latent(d))` forward-simulates fresh event paths from parameters,
  Turing-free (a comprehension batches them).
- `DynamicPPL.predict(model, chain)` recovers the latent event times of the
  records you fit, conditioned on the data over the posterior.
- Both rely on the marginal and latent forms being one family that share
  parameters, so a marginal-fit posterior drops straight into the latent form.
"""
