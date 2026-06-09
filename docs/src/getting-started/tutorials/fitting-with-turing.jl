md"""
# Fitting CensoredDistributions.jl modified distributions with Turing.jl

## Introduction

### What are we going to do in this exercise

We fit one composed delay model to censored, right-truncated line-list data
with Turing.jl, then fit the *same* model two ways: in its cheap **marginal**
form and in its event-based **latent** form.
The marginal and latent forms are one family sharing the same parameters, so
they target the same posterior; the comparison shows how they differ in runtime
and in recovery at a small sampling budget.

We cover:

1. Building one composed distribution for the whole record with
   [`compose`](@ref) and the censored building blocks.
2. Deriving priors from the parameter table with [`params_table`](@ref) and
   [`build_priors`](@ref).
3. Simulating a line list from the model, so the true parameters are known.
4. Fitting the marginal form through the vectorised
   [`composed_distribution_model`](@ref) (a DataFrame scored in one `~`).
5. Fitting the latent form of the same object, sampling the intermediate event.
6. Comparing runtime and recovery, and reading the fitted delays back with
   [`update`](@ref) and [`edge_means`](@ref).

### What might I need to know before starting

This tutorial builds on [Getting Started with
CensoredDistributions.jl](@ref getting-started) and the composer reference, [The
composer toolkit](@ref composer-toolkit), which introduces [`compose`](@ref),
the marginal-versus-latent duality, and the parameter tooling used here.

The Turing-facing functions ([`composed_distribution_model`](@ref),
[`composed_parameters_model`](@ref), [`update`](@ref)`(template, chain)`) come
from package extensions loaded once Turing and FlexiChains are available, so
the core package stays Turing-free.

We sample with non-Enzyme AD: ForwardDiff here (see [Automatic differentiation
backends](@ref ad-backends) for the support matrix and per-backend benchmarks).

## Packages used
We use Turing for probabilistic programming, FlexiChains for the chain output,
DataFrames for the record table, CairoMakie with PairPlots for the posterior
overlays, and Random for reproducibility.
"""

using CensoredDistributions
using Distributions
using Turing
using DynamicPPL: to_submodel
using FlexiChains: VNChain
using DataFrames
using CairoMakie, PairPlots
using Random
using Statistics
using ADTypes: AutoForwardDiff

md"""
## One composed delay model

The record is an onset that leads to admission, then admission to death: a
two-step chain. Each step is a Gamma delay censored directly in the stack with
[`double_interval_censored`](@ref), giving a one-day primary-event window and a
one-day secondary (interval) window, so the censoring is part of the composed
object rather than a separate step.

The named chain steps give clean event slots: the tree's event names are
`onset, admit, death`, exactly the columns each data row supplies.
"""

dic(d) = double_interval_censored(d; primary_event = Uniform(0, 1),
    interval = 1.0)

template = compose((path = Sequential(
    (dic(Gamma(2.0, 1.5)), dic(Gamma(1.5, 2.0))),
    (:onset_admit, :admit_death)),))

md"""
The flat event names are the row columns; a missing column drives whether that
delay is conditioned on or marginalised for that record.
"""

CensoredDistributions.tree_event_names(template)

md"""
## Priors

[`params_table`](@ref) lists every free delay parameter as a flat table, keyed
by edge path and parameter name, with the support a prior must respect.
[`build_priors`](@ref) turns that table into the nested prior NamedTuple the
parameter model consumes, deriving a support-aware default for each row (a
positive-truncated prior for the positive shapes and scales here).
"""

param_table = params_table(template)

priors = build_priors(param_table)

md"""
## The fitting model

One model serves both fits. It samples the delay parameters from the priors
with [`composed_parameters_model`](@ref), rebuilding the same composed
structure, then scores the record table through the vectorised
[`composed_distribution_model`](@ref) in a single `~`.

The `wrap` argument controls the direction: `identity` keeps the bare composed
object (the **marginal** fit, the intermediate admission integrated out), and
[`latent`](@ref) wraps it (the **latent** fit, the intermediate admission
sampled as a `~` variable). Everything else is shared, so the two fits target
the same posterior.
"""

@model function delay_model(template, priors, rows; wrap = identity)
    delays ~ to_submodel(composed_parameters_model(template, priors))
    obs ~ to_submodel(composed_distribution_model(wrap(delays), rows))
end

md"""
## Simulate a line list

We simulate from the model with known true parameters, so recovery can be
checked. [`predict_events`](@ref) draws a full event path
`[onset, admit, death]` from the latent form of the truth; we keep all three
events as an observed record.
"""

true_params = (path = (onset_admit = (shape = 2.0, scale = 1.5),
    admit_death = (shape = 1.5, scale = 2.0)),)

truth = update(template, true_params)

n = 150;

sim_rows = let rng = MersenneTwister(20260609)
    map(1:n) do _
        s = predict_events(latent(truth); rng = rng)
        (onset = s.onset, admit = s.admit, death = s.death)
    end
end;

md"""
We collect the records into a `DataFrame`; the vectorised
[`composed_distribution_model`](@ref) consumes any Tables.jl source, so the
table passes straight in.
"""

data = DataFrame(sim_rows)

first(data, 5)

md"""
## Fit the marginal form

The marginal fit scores each record through the composed `logpdf` with the
intermediate admission integrated out, so there are no per-record latent
dimensions. We keep the budget modest (this page runs in a per-tutorial
subprocess) and time the run.
"""

marginal_time = @elapsed marginal_chain = sample(
    Xoshiro(1),
    delay_model(template, priors, data),
    NUTS(0.8; adtype = AutoForwardDiff()), 300;
    chain_type = VNChain, progress = false)

md"""
## Fit the latent form

The latent fit is the same model with the composed object wrapped in
[`latent`](@ref): the intermediate admission time is sampled as a `~` variable
per record and each segment scores against it. This adds dimensions but can mix
more robustly at a small budget, because the sampler walks the augmented joint
rather than a stiff marginal.
"""

latent_time = @elapsed latent_chain = sample(
    Xoshiro(1),
    delay_model(template, priors, data; wrap = latent),
    NUTS(0.8; adtype = AutoForwardDiff()), 300;
    chain_type = VNChain, progress = false)

md"""
## Compare runtime and recovery

We read the fitted delays off each chain with
[`update`](@ref)`(template, chain)` and [`edge_means`](@ref), which sees through
each censored leaf to its inner free delay mean. The true edge means are the
means of the underlying Gammas.
"""

true_means = edge_means(truth)

marginal_fit = update(template, marginal_chain; prefix = :delays)

latent_fit = update(template, latent_chain; prefix = :delays)

recovery = DataFrame(
    edge = ["onset_admit", "admit_death"],
    truth = [true_means.path.onset_admit, true_means.path.admit_death],
    marginal = [edge_means(marginal_fit).path.onset_admit,
        edge_means(marginal_fit).path.admit_death],
    latent = [edge_means(latent_fit).path.onset_admit,
        edge_means(latent_fit).path.admit_death])

md"""
Both fits recover the true edge means closely, as the marginal-equals-latent
equivalence requires: the two forms target the same posterior.
"""

md"""
At this small budget the latent form is the slower of the two per effective
sample at scale, since it carries one extra sampled event per record; the
marginal form is the cheap default. The latent form earns its cost when the
marginal integral is impractical, or when the augmented geometry mixes better
than a stiff marginal at small counts.
"""

runtime = DataFrame(
    form = ["marginal", "latent"],
    seconds = [marginal_time, latent_time])

md"""
## Posterior overlay

We summarise each posterior by its two edge means, the quantity reported for a
delay distribution, rather than by raw shape/scale samples. `mean_draws` reads
the edge means off every posterior draw through repeated
[`update`](@ref)`(template, chain; draw = i)` and [`edge_means`](@ref), with no
manual chain indexing.
"""

function mean_draws(chain)
    em = map(1:prod(size(chain))) do i
        edge_means(update(template, chain; prefix = :delays, draw = i))
    end
    return (onset_admit = [e.path.onset_admit for e in em],
        admit_death = [e.path.admit_death for e in em])
end

marginal_means = mean_draws(marginal_chain)

latent_means = mean_draws(latent_chain)

truth_nt = (onset_admit = true_means.path.onset_admit,
    admit_death = true_means.path.admit_death)

md"""
The two fits' posteriors over the delay means sit on top of each other and
bracket the true values, confirming that the marginal and latent forms target
the same posterior.
"""

pairplot(
    PairPlots.Series(marginal_means; label = "marginal",
        color = (:steelblue, 0.6)),
    PairPlots.Series(latent_means; label = "latent",
        color = (:orange, 0.6)),
    PairPlots.Truth(truth_nt, label = "True values"))

md"""
## Summary

- One composed distribution describes the whole record; [`compose`](@ref) and
  the censored leaves build it, while [`params_table`](@ref) and
  [`build_priors`](@ref) derive support-aware priors.
- The records score through the vectorised [`composed_distribution_model`](@ref)
  on a `DataFrame` in one `~`, and the same model fits and generates.
- The marginal and latent forms are one family on the same parameters: wrapping
  the composed object in [`latent`](@ref) switches a single fit between them,
  and both recover the truth.
- The marginal form is the cheap default; the latent form costs an extra
  sampled event per record but suits small counts or impractical marginal
  integrals.
- [`update`](@ref)`(template, chain)` and [`edge_means`](@ref) read the fitted
  delays back onto the composed object with no manual chain indexing.
"""
