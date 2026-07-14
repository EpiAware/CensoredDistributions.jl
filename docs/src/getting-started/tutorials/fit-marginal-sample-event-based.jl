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
distribution, Random for reproducibility, and Integrals for the quadrature in
the equivalence check.
"""

using CensoredDistributions
using Distributions
using Turing
using ADTypes: AutoForwardDiff
using DynamicPPL: prefix, @varname, predict
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
    NUTS(; adtype = AutoForwardDiff()), 200; progress = false);

md"""
The posterior concentrates near the true parameters.
"""

mu_post = vec(chain[@varname(mu)]);

sigma_post = vec(chain[@varname(sigma)]);

# The posterior means sit close to the simulating truth.
recovery = [
    (parameter = :mu, truth = true_meanlog,
        posterior_mean = round(mean(mu_post); digits = 2)),
    (parameter = :sigma, truth = true_sdlog,
        posterior_mean = round(mean(sigma_post); digits = 2))]

md"""
## Demonstrate the target-marginal-latent equivalence

The whole workflow rests on a single identity, which we show once as a three-way
comparison of the cumulative distribution function at the posterior mean.

- The **target** is the distribution the data were simulated from: a primary `p`
  drawn from `Uniform(0, 1)` plus a `LogNormal(mu, sigma)` delay.
  Its `cdf` is the integral `∫₀¹ cdf(LogNormal(mu, sigma), x - p) dp`, which we
  evaluate with adaptive quadrature.
- The **marginal** form scores the `primary_censored` leaf, integrating the
  primary out inside `cdf`.
- The **latent** form carries the primary as an explicit dimension; its `cdf`
  comes from a Monte Carlo average of `cdf(LogNormal(mu, sigma), x - p)` over
  primaries drawn from the latent representation.

All three are the same distribution, so their `cdf` curves coincide.
"""

post_draw = (mu = mean(mu_post), sigma = mean(sigma_post));

marginal_leaf = primary_censored(
    LogNormal(post_draw.mu, post_draw.sigma), Uniform(0, 1))

latent_leaf = latent(marginal_leaf)

function target_cdf(x)
    delay = get_dist(marginal_leaf)
    problem = IntegralProblem((p, _) -> cdf(delay, x - p), (0.0, 1.0))
    return solve(problem, QuadGKJL(); reltol = 1e-12, abstol = 1e-12).u
end

function latent_cdf(x; rng = MersenneTwister(7), draws = 200_000)
    delay = get_dist(marginal_leaf)
    paths = (rand(rng, latent_leaf) for _ in 1:draws)
    return mean(cdf(delay, x - p.primary) for p in paths)
end

eval_points = [1.0, 3.0, 6.0, 10.0];

three_way = [(x = x, target = target_cdf(x),
                 marginal = cdf(marginal_leaf, x), latent = latent_cdf(x))
             for x in eval_points];

md"""
The target, marginal and latent `cdf` agree across the evaluation points.
The target-versus-marginal gap is at numerical precision (both integrate the
primary out exactly); the latent column carries only Monte Carlo noise from its
finite draws.
"""

three_way

# The largest target-versus-marginal and target-versus-latent gaps across the
# evaluation points quantify the agreement: the marginal gap sits at numerical
# precision, the latent gap at Monte Carlo noise.
equivalence_gaps = (
    target_vs_marginal = maximum(abs(r.target - r.marginal) for r in three_way),
    target_vs_latent = maximum(abs(r.target - r.latent) for r in three_way))

md"""
### When to prefer the marginal or the latent form

Prefer the marginal form by default: it carries no per-record latents, so it is
cheaper and lower-dimensional at scale.
Prefer the latent form for small-count or heavily censored problems, where the
extra latent dimensions cost little and the sampler may explore the joint more
reliably than a stiff one-dimensional marginal.
The [Marginal versus latent](@ref marginal-versus-latent) section of the
composer toolkit gives the canonical treatment of this choice.

## Flavour 1: forward-simulate fresh event paths (Turing-free)

`rand(latent(d))` draws a full event path as a labelled
`(primary = ..., observed = ...)` record directly from a latent distribution.
Build the latent distribution from a posterior draw and simulate, with no model
or conditioning. This is the new-record posterior-predictive path.
"""

one_path = rand(MersenneTwister(1), latent_leaf)

md"""
A single path is the record `(primary, observed)`: the primary lies in the
window and the observed time exceeds it (a positive delay).
"""

# The primary lies in the window and the observed time exceeds it.
one_path_check = (primary = round(one_path.primary; digits = 3),
    observed = round(one_path.observed; digits = 3),
    primary_in_window = 0 <= one_path.primary <= 1,
    observed_after_primary = one_path.observed > one_path.primary)

md"""
We draw many paths from a single latent leaf with a comprehension, or push a set
of posterior draws through the latent form by rebuilding the distribution per
draw.
The composer record path `rand(d, rows)` is the batched dual for a composed
object; a bare `latent` leaf has no batched `rand(leaf, n)` yet, so a single
leaf stays a comprehension here.
"""

rng2 = MersenneTwister(2);

many_paths = [rand(rng2, latent_leaf) for _ in 1:500];

# Every drawn path leaves a positive observed delay.
many_paths_summary = (n = length(many_paths),
    fraction_positive_delay = mean(p.observed > p.primary for p in many_paths))

build(p) = latent(primary_censored(
    LogNormal(p.mu, p.sigma), Uniform(0, 1)))

posterior_pairs = [(mu = m, sigma = s)
                   for (m, s) in zip(mu_post[1:50], sigma_post[1:50])];

rng3 = MersenneTwister(3);

per_draw_paths = [rand(rng3, build(p)) for p in posterior_pairs];

# One simulated path per posterior draw.
(draws = length(posterior_pairs), paths = length(per_draw_paths))

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

events = predict(latent_recovery(observed), chain);

md"""
The recovered variables are the per-record latent primaries `rec_i.p` (only the
primary is sampled; the observed time is conditioned on the data).

### Consistency check

The marginal-equals-latent equivalence means every recovered primary must lie in
the window `[0, 1]` and reproduce the observed delay gap as a positive value
(`observed_i - p_i > 0`). We check this for every record.
"""

recovered_in_window = trues(n)
recovered_positive_gap = trues(n)
for i in 1:n
    p_i = vec(events[@varname($(Symbol("rec", i)).p)])
    recovered_in_window[i] = all(0 .<= p_i .<= 1)
    recovered_positive_gap[i] = all(observed[i] .- p_i .> 0)
end

# Every record's recovered primaries stay in the window and leave a positive
# observed delay, as the equivalence requires.
consistency = (records = n,
    all_primaries_in_window = all(recovered_in_window),
    all_gaps_positive = all(recovered_positive_gap))

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
