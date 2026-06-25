md"""
# Fitting CensoredDistributions.jl modified distributions with Turing.jl

## Introduction

### What are we going to do in this exercise

We'll demonstrate how to use `CensoredDistributions.jl` in conjunction with
Turing.jl for Bayesian inference of epidemiological delay distributions.
We'll cover the following key points:

1. Defining a simple delay distribution as a composed object and deriving its
   priors from the parameter table.
2. Exploring the prior distribution of this model.
3. Building a composed distribution that incorporates double censoring and
   right truncation.
4. Generating synthetic data from that object using fixed parameters.
5. Fitting a naive model that ignores censoring.
6. Fitting a model that accounts for secondary event censoring and truncation
   but not primary event censoring.
7. Fitting the full model that accounts for double censoring and right
   truncation.
8. Aggregating to weighted counts so repeated observations score once.
9. Reading the posterior back through the standard composed interface and the
   FlexiChains output.

## What might I need to know before starting

This tutorial builds on the concepts introduced in [Getting Started with
CensoredDistributions.jl](@ref getting-started) and the composer reference,
[Composing censored distributions](@ref composer-toolkit), which introduces
[`compose`](@ref), [`params_table`](@ref), [`build_priors`](@ref), and the
marginal-versus-latent duality used here.

The Turing-facing functions ([`composed_distribution_model`](@ref),
[`composed_parameters_model`](@ref), [`update`](@ref)`(template, chain)`) come
from package extensions loaded once Turing and FlexiChains are available, so the
core package stays Turing-free.

We sample with Mooncake reverse-mode AD (`AutoMooncake`), the backend the other
composed-tree fitting tutorials use (see [Automatic differentiation
backends](@ref ad-backends) for the support matrix and per-backend benchmarks).
The composed tree differentiates under Mooncake with no special handling.

## Packages used
We use CairoMakie for plotting, Turing for probabilistic programming,
FlexiChains for working with MCMC output, DataFramesMeta for the data pipeline,
Random for reproducibility, and StatsBase for the empirical CDF.
"""

using CensoredDistributions
using Distributions
using Turing
using DynamicPPL: to_submodel, @varname
using FlexiChains: VNChain, parameters, rhat
using DataFramesMeta
using CairoMakie, PairPlots
using Random
using StatsBase
using Statistics
using ADTypes: AutoMooncake
import Mooncake

md"""
## Define the true parameters for generating synthetic data

We start by defining the number of samples and the true parameters of the
lognormal delay.
"""

n = 2000;

meanlog = 1.5;

sdlog = 0.75;

md"""
Now we can define the true delay distribution using Distributions.jl.
"""

true_dist = LogNormal(meanlog, sdlog);

md"""
## A reusable delay template

Every model in this tutorial fits the same single onset-to-report delay, so we
write one helper that wraps a delay leaf as a one-step [`Sequential`](@ref)
chain named `onset_report`. Naming the edge gives the composed object the clean
event names `onset` and `report`, the two columns each data row supplies. The
same composed object scores records and simulates them, so a model is built once
and used in both directions.
"""

delay_template(leaf) = sequential(:onset_report => leaf)

md"""
The full model layers primary censoring (a one-day primary-event window),
interval censoring (a one-day secondary window), and per-record right truncation
onto a lognormal delay. [`double_interval_censored`](@ref) builds the primary
plus interval leaf; the
truncation is applied per record at fit time through a reserved `obs_time`
field, so the leaf carries no fixed horizon.
"""

function full_leaf(mu, sig)
    double_interval_censored(LogNormal(mu, sig);
        primary_event = Uniform(0, 1), interval = 1.0)
end

full_template = delay_template(full_leaf(meanlog, sdlog))

md"""
The flat event names are the row columns.
"""

event_names(full_template)

md"""
## Priors

[`params_table`](@ref) lists every free delay parameter as a flat table, keyed
by edge path and parameter name, with the support a prior must respect. A
censored leaf is transparent here: only the inner lognormal's `mu` and `sigma`
appear, the censoring being fixed structure.
"""

param_table = params_table(full_template)

md"""
[`build_priors`](@ref) turns that table into the nested prior NamedTuple the
parameter model consumes, deriving a support-aware default for each row (an
unbounded prior for the location `mu`, a positive-truncated prior for the
positive scale `sigma`).
"""

full_priors = build_priors(param_table)

md"""
Each default prior centres on the template's own parameter value, and the
template here is built from the true `meanlog` and `sdlog`, so these illustrative
priors happen to sit on the values we recover.
That keeps the demo self-contained, but it does not drive the result.
With 2000 simulated records the likelihood dominates these wide priors, so the
recovery below reflects the data and the censoring model rather than the prior
mean; a real analysis would set priors from background knowledge, independent of
the parameters being estimated.
"""

md"""
## Prior predictive checks

The parameter block is the same for every fit.
[`composed_parameters_model`](@ref) samples the delay parameters from the priors
and rebuilds the same composed object. Sampling it under `Prior()` draws from
the prior over the delay parameters before seeing any data.
"""

@model function params_model(template, priors)
    delays ~ to_submodel(composed_parameters_model(template, priors))
    return delays
end

Random.seed!(123);

prior_chain = sample(params_model(full_template, full_priors), Prior(), 1000;
    chain_type = VNChain, progress = false)

md"""
To read the sampled `mu` and `sigma` off a chain, we pull every draw at once
with [`param_draws`](@ref)`(template, chain)`, which returns one nested
NamedTuple per draw keyed exactly like [`params`](@ref)`(template)`. We index the
`onset_report` edge to collect the two delay parameters into the column vectors
`PairPlots` wants. This same `draw_params` helper reads the prior chain and every
posterior chain below, with no per-draw `update` loop and no manual chain
indexing.
"""

function draw_params(template, chain; prefix = :delays)
    edges = param_draws(template, chain; prefix = prefix)
    return (mu = [e.onset_report.mu for e in edges],
        sigma = [e.onset_report.sigma for e in edges])
end

md"""
We overlay the prior draws against the true parameters with a `PairPlots.Truth`
layer; this shows what the model believes before seeing any data.
"""

truth_nt = (mu = meanlog, sigma = sdlog)

pairplot(
    PairPlots.Series(draw_params(full_template, prior_chain); label = "prior",
        color = (:grey, 0.5)),
    PairPlots.Truth(truth_nt, label = "True values"))

md"""
## Simulate from the double censored distribution

We simulate from the full composed object with the true parameters, so the
parameters we recover can be checked. `rand(d, rows)` walks the object once per
record and returns a labelled event record for each, the generative dual to the
batched scoring; the onset is the primary event origin (here zero) and the
report is the censored delay. We draw an observation window per record and keep
only reports that fall strictly before it, the right-truncation that the
`obs_time` field then adjusts for. Keeping `report < obs_time` matches the
conditioning the `obs_time` likelihood applies: a report interval-censored to
the day `obs_time` straddles the horizon, so the truncation retains only the
days that close before it, and the simulation and the fit then condition on the
same point.
"""

rng = MersenneTwister(123)

paths = rand(rng, full_template, fill((onset = missing, report = missing), n))

reports = [p.report for p in paths]

obs_times = rand(rng, DiscreteUniform(8, 12), n)

keep = reports .< obs_times;

md"""
We collect the kept records into a `DataFrame`. A `DataFrame` is a Tables.jl
source, so it passes straight into the vectorised
[`composed_distribution_model`](@ref). The `onset` column is the primary event
origin, `report` the observed delay, and `obs_time` the per-record truncation
horizon.
"""

simulated_data = DataFrame(onset = 0.0, report = reports[keep],
    obs_time = obs_times[keep])

first(simulated_data, 5)

md"""
### Aggregate to weighted counts

To speed up the fits we aggregate to unique `(report, obs_time)` combinations
and count occurrences. A reserved `count` row field weights each record's
contribution, so a repeated observation scores once with its multiplicity rather
than row by row.
"""

simulated_counts = @chain simulated_data begin
    @groupby(:report, :obs_time)
    @combine(:count = length(:report))
end

md"""
### Visualise the simulated data

Let's compare the censored samples to the true distribution. First we calculate
the empirical CDF of the censored observations, weighted by the counts.
"""

empirical_cdf_obs = @with(simulated_counts,
    ecdf(:report, weights = weights(:count)));

x_seq = @with simulated_counts begin
    range(minimum(:report), stop = maximum(:report) + 2, length = 100)
end;

md"""
The theoretical CDF comes from the true lognormal, and uncensored samples give a
second empirical reference.
"""

theoretical_cdf = cdf.(true_dist, x_seq);

uncensored_samples = rand(rng, true_dist, n);

empirical_cdf_uncensored = ecdf(uncensored_samples);

f = Figure()
ax = Axis(f[1, 1],
    title = "Censored vs Uncensored vs Theoretical CDF",
    ylabel = "Cumulative Probability", xlabel = "Delay")
scatter!(ax, x_seq, empirical_cdf_obs.(x_seq),
    label = "Empirical CDF (Censored)", color = :blue)
scatter!(ax, x_seq, empirical_cdf_uncensored.(x_seq),
    label = "Empirical CDF (Uncensored)", color = :red, marker = :cross)
lines!(ax, x_seq, theoretical_cdf, label = "Theoretical CDF",
    color = :black, linewidth = 2)
axislegend(position = :rb)
f

md"""
## The fitting model

Every fit shares one model: [`composed_parameters_model`](@ref) samples the
delay parameters from the priors and rebuilds the composed object, then the
vectorised [`composed_distribution_model`](@ref) scores the whole record table
in a single `~`. The table passes straight in as a Tables.jl source; each record
bakes in its own `obs_time` horizon and `count` weight. The three models below
differ only in the leaf they pass as the template.
"""

@model function fit_model(template, priors, data)
    delays ~ to_submodel(composed_parameters_model(template, priors))
    obs ~ to_submodel(composed_distribution_model(delays, data))
end

md"""
## Fitting a naive model using Turing

We first fit a naive model that ignores the censoring process. Its template is a
plain lognormal leaf, so the observed delays are treated as if they came
directly from the uncensored distribution, providing a baseline for comparison.
We add a small constant to the observations to avoid the lognormal's zero
boundary (a hint that this model is misspecified) and drop the `obs_time`
horizon, since the naive model also ignores truncation.
"""

naive_template = delay_template(LogNormal(meanlog, sdlog))

naive_priors = build_priors(params_table(naive_template))

naive_data = @chain simulated_counts begin
    @transform(:onset=0.0, :report=:report .+ 1e-6)
    @select(:onset, :report, :count)
end

naive_fit = sample(Xoshiro(1),
    fit_model(naive_template, naive_priors, naive_data),
    NUTS(0.8; adtype = AutoMooncake(; config = nothing)), MCMCThreads(), 300, 2;
    chain_type = VNChain, progress = false)

md"""
We read the fitted parameters back with [`update`](@ref)`(template, chain)`,
which reduces each parameter's draws (posterior mean by default) and rebuilds
the composed object, then list them with [`params_table`](@ref). No manual chain
indexing is needed.
"""

naive_recovered = params_table(
    update(naive_template, naive_fit; prefix = :delays))

md"""
Let's visualise the posterior alongside the true values.
"""

pairplot(
    PairPlots.Series(draw_params(naive_template, naive_fit); label = "naive",
        color = (:steelblue, 0.6)),
    PairPlots.Truth(truth_nt, label = "True values"))

md"""
We see that, just from the recovered parameters, we might not be very happy with
the fit. The naive model ignores the censoring and truncation that shaped the
data, so `mu` is pulled away from the target 1.5 and `sigma` away from 0.75.
"""

md"""
## Fitting a truncation-adjusted interval model

Now let's fit an intermediate model that accounts for interval censoring and
right truncation but ignores the primary censoring process. This provides a
comparison point between the naive model and the full model. Its template is an
[`interval_censored`](@ref) leaf, and the records carry their `obs_time` horizon
so the truncation is adjusted for.
"""

interval_template = delay_template(
    interval_censored(LogNormal(meanlog, sdlog), 1.0))

interval_priors = build_priors(params_table(interval_template))

censored_data = @chain simulated_counts begin
    @transform(:onset = 0.0)
    @select(:onset, :report, :obs_time, :count)
end

interval_fit = sample(Xoshiro(1),
    fit_model(interval_template, interval_priors, censored_data),
    NUTS(0.8; adtype = AutoMooncake(; config = nothing)), MCMCThreads(), 300, 2;
    chain_type = VNChain, progress = false)

interval_recovered = params_table(
    update(interval_template, interval_fit; prefix = :delays))

md"""
Plotting the posterior against the true values, the interval model recovers the
parameters more closely than the naive model, but still carries the bias from
ignoring the primary event window.
"""

pairplot(
    PairPlots.Series(draw_params(interval_template, interval_fit);
        label = "interval", color = (:steelblue, 0.6)),
    PairPlots.Truth(truth_nt, label = "True values"))

md"""
## Fitting the double censored model

Now we fit the full model that accounts for the whole censoring and truncation
process. It reuses the `full_template` we simulated from, scoring the same
truncation-carrying `censored_data` table. This demonstrates accurate parameter
recovery when the censoring process is properly modelled.
"""

full_fit = sample(Xoshiro(1),
    fit_model(full_template, full_priors, censored_data),
    NUTS(0.8; adtype = AutoMooncake(; config = nothing)), MCMCThreads(), 300, 2;
    chain_type = VNChain, progress = false)

full_recovered = params_table(update(full_template, full_fit; prefix = :delays))

md"""
And the corresponding pair plot with the true parameters overlaid.
"""

pairplot(
    PairPlots.Series(draw_params(full_template, full_fit); label = "full",
        color = (:orange, 0.6)),
    PairPlots.Truth(truth_nt, label = "True values"))

md"""
The full model concentrates near the true parameters. Collecting the three
recovered fits side by side shows the progression. The naive model is biased,
the interval model improves on it, and the full model recovers the truth.
"""

recovery = DataFrame(
    parameter = ["mu", "sigma"],
    truth = [meanlog, sdlog],
    naive = naive_recovered.value,
    interval = interval_recovered.value,
    full = full_recovered.value)

md"""
## Working with FlexiChains output

Because we asked Turing to return a `VNChain`, the posterior is keyed by
`VarName`s rather than flattened `Symbol`s, so specific quantities come out
without string munging. We can list the parameters in the chain directly: they
carry the submodel prefix `delays` and the edge name `onset_report`.
"""

parameters(full_fit)

md"""
The composed interface reads the fitted delay back as a distribution.
[`update`](@ref) rebuilds the object from the posterior, [`event`](@ref) fetches
the named leaf, and the overall [`mean`](@ref CensoredDistributions.mean)
reports the mean delay. For the per-event breakdown wrap with [`latent`](@ref):
the per-event labelled `NamedTuple` reports each event's mean, seeing through the
censored leaf to its inner free delay (keyed by [`event_names`](@ref)).
"""

fitted = update(full_template, full_fit; prefix = :delays)

fitted_leaf = event(fitted, :onset_report)

mean(fitted)

md"""
The per-event view is a labelled NamedTuple keyed by [`event_names`](@ref).
"""

mean(latent(fitted))

md"""
FlexiChains also extends Statistics functions over the chain. Here we ask for
the posterior mean of each parameter.
"""

mean(full_fit)

md"""
and the rhat convergence diagnostic across the two chains.
"""

rhat(full_fit)

md"""
Finally, because the chain is keyed by `VarName`, we can index into it with a
`@varname` to recover the raw samples for a single parameter as a matrix of
`(iter, chain)`. The name carries the submodel prefix `delays` and the edge name
`onset_report`.
"""

mu_samples = full_fit[@varname(delays.onset_report.mu)]

size(mu_samples)

md"""
## Fit without Turing

The whole log-density is already Turing-free: the same `params_table` /
`build_priors` / `update` toolchain and the per-record scalars assemble a
standard
[`LogDensityProblems`](https://github.com/tpapp/LogDensityProblems.jl) problem
over the flat parameter vector, so a composed model can be sampled by
AdvancedHMC / DynamicHMC / Pathfinder directly, with the DynamicPPL extension
demoted to one consumer among several.

[`as_logdensity`](@ref)`(template, priors, data)` builds the spec; the
[`flatten`](@ref) / [`unflatten`](@ref) codec (ordered by the
[`params_table`](@ref) row walk)
moves between the flat vector and the named, `update`-able `NamedTuple`, and the
`Bijectors` extension derives the unconstrained transform from the priors. The
flat layout and names match the `VarName`-keyed chain above, so a posterior is
interchangeable across backends. Loading `LogDensityProblems`, `Bijectors` and
an AD backend gives a standalone fit:

```julia
using CensoredDistributions, Distributions, Tables
using Bijectors, LogDensityProblems, LogDensityProblemsAD
using ADTypes: AutoForwardDiff
using AdvancedHMC

## `data` is a vector of event-vector records (one `[onset, report]` per row,
## as the composed `logpdf` scores); build it from the simulated records.
prob = as_logdensity(full_template, full_priors, data)
adprob = ADgradient(AutoForwardDiff(), prob)
D = LogDensityProblems.dimension(prob)

## Start from the template values pushed onto the unconstrained scale.
xc0 = collect(Tables.getcolumn(params_table(full_template), :value))
flatp = CensoredDistributions.flat_priors(prob)
z0 = [bijector(flatp[i])(xc0[i]) for i in eachindex(xc0)]

metric = DiagEuclideanMetric(D)
ham = Hamiltonian(metric, adprob)
integrator = Leapfrog(find_good_stepsize(ham, z0))
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric),
    StepSizeAdaptor(0.8, integrator))
samples, _ = sample(ham, kernel, z0, 1000, adaptor, 500)

## Map unconstrained draws back to named, constrained parameters via the codec.
post = [first(CensoredDistributions.to_constrained(prob, z))
        for z in samples[501:end]]
fitted = update(full_template, unflatten(full_template, mean(post)))
```

This path is covered end-to-end in the test suite (the LogDensityProblems
log-density equals the Turing log-joint on the same parameters, and an
AdvancedHMC fit recovers known parameters with no Turing in the path). A
runnable standalone-fit tutorial is tracked as follow-up.

## Summary

- One composed [`Sequential`](@ref) delay describes each record; the censored
  leaves ([`double_interval_censored`](@ref), [`interval_censored`](@ref)) build it,
  while [`params_table`](@ref) and [`build_priors`](@ref) derive support-aware
  priors.
- The same parameter block ([`composed_parameters_model`](@ref)) and vectorised
  record model ([`composed_distribution_model`](@ref)) fit every model; only the
  leaf changes, and a `DataFrame` of records scores in one `~`.
- `rand(d)` / `rand(latent(d))` simulates labelled event paths from the same
  object, so simulation and inference share one model.
- Reserved `obs_time` and `count` row fields apply per-record truncation and
  count weighting without changing the model.
- The naive model is misspecified; adding interval censoring and truncation
  improves it, and the full double-censored model recovers the truth.
- [`update`](@ref)`(template, chain)`, [`event`](@ref), and
  [`mean`](@ref CensoredDistributions.mean) read
  the fitted delay back onto the composed object with no manual chain indexing,
  while the FlexiChains output exposes the raw `VarName`-keyed samples.
- The same model fits without Turing: [`as_logdensity`](@ref) assembles a
  `LogDensityProblems` problem over the flat parameter vector (the
  [`flatten`](@ref) / [`unflatten`](@ref) codec + a prior-driven `Bijectors`
  transform), sampled directly by AdvancedHMC / DynamicHMC, with a posterior
  interchangeable with the Turing chain above.
"""
