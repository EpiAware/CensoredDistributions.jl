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

1. Building one composed distribution for the whole record from the censored
   building blocks (a [`Sequential`](@ref) chain, the composer
   [`compose`](@ref) lowers a chain to).
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
from package extensions loaded once Turing is available, so the core package
stays Turing-free. Turing returns a FlexiChains chain by default, so no
`chain_type` keyword is needed.

We sample with non-Enzyme AD: ForwardDiff here (see [Automatic differentiation
backends](@ref ad-backends) for the support matrix and per-backend benchmarks).

## Packages used
We use Turing for probabilistic programming, DataFramesMeta for the record
table, CairoMakie with PairPlots for the posterior overlays, and Random for
reproducibility.
"""

using CensoredDistributions
using Distributions
using Turing
using DynamicPPL: to_submodel, prefix
using DataFramesMeta
using CairoMakie, PairPlots
using Random
using Statistics
using ADTypes: AutoForwardDiff

md"""
## One composed delay model

The record is an onset that leads to admission, then admission to death: a
two-step chain. The chain is a [`Sequential`](@ref), the composer that
[`compose`](@ref) lowers a `Vector` of steps to; here we name the steps so the
event slots come out clean. Each step is a Gamma delay censored directly in the
stack with [`double_interval_censored`](@ref), giving a one-day primary-event
window and a one-day secondary (interval) window, so the censoring is part of
the composed object rather than a separate step.

The tree's event names are `onset, admit, death`, exactly the columns each data
row supplies.
"""

dic(d) = double_interval_censored(d; primary_event = Uniform(0, 1),
    interval = 1.0)

template = Sequential((dic(Gamma(2.0, 1.5)), dic(Gamma(1.5, 2.0))),
    (:onset_admit, :admit_death))

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
## The two fitting models

Both fits share the same parameter block: they sample the delay parameters from
the priors with [`composed_parameters_model`](@ref), rebuilding the same
composed `delays` object. They differ only in how that object scores the data.

The **marginal** model scores the whole record table through the vectorised
[`composed_distribution_model`](@ref) in a single `~`: the intermediate
admission is integrated out inside the composed `logpdf`, so there are no
per-record latent dimensions. A `DataFrame` is a Tables.jl source, so it passes
straight in.
"""

@model function marginal_model(template, priors, data)
    delays ~ to_submodel(composed_parameters_model(template, priors))
    obs ~ to_submodel(composed_distribution_model(delays, data))
end

md"""
The **latent** model wraps the same object in [`latent`](@ref) and scores one
record at a time, sampling the intermediate admission as a `~` variable per
record. The latent composer model takes one row, so we loop and `prefix` each
record's latents to keep the chain readable. The parameter block is identical,
so both models target the same posterior over `delays`.
"""

@model function latent_model(template, priors, rows)
    delays ~ to_submodel(composed_parameters_model(template, priors))
    ld = latent(delays)
    for i in eachindex(rows)
        obs ~ to_submodel(
            prefix(composed_distribution_model(ld, rows[i]), Symbol(:rec, i)),
            false)
    end
end

md"""
## Simulate a line list

We simulate from the model with known true parameters, so recovery can be
checked. [`update`](@ref) sets the truth on the composed object, then
[`predict_events`](@ref) walks it to draw a full event path
`[onset, admit, death]`; we keep all three events as an observed record.
"""

true_params = (onset_admit = (shape = 2.0, scale = 1.5),
    admit_death = (shape = 1.5, scale = 2.0))

truth = update(template, true_params)

n = 20;

sim_rows = let rng = MersenneTwister(20260609)
    map(1:n) do _
        s = predict_events(truth; rng = rng)
        (onset = s[1], admit = s[2], death = s[3])
    end
end;

md"""
We also collect the records into a `DataFrame` for the marginal fit; the
vectorised [`composed_distribution_model`](@ref) consumes any Tables.jl source,
so the table passes straight in.
"""

data = DataFrame(sim_rows)

first(data, 5)

md"""
## Fit the marginal form

The marginal fit scores the whole table in one `~`, the intermediate admission
integrated out, so there are no per-record latent dimensions. We keep the budget
modest (this page runs in a per-tutorial subprocess) and time the run.
"""

marginal_time = @elapsed marginal_chain = sample(
    Xoshiro(1),
    marginal_model(template, priors, data),
    NUTS(0.8; adtype = AutoForwardDiff()), 300;
    progress = false)

md"""
## Fit the latent form

The latent fit samples the intermediate admission time as a `~` variable per
record and scores each segment against it. This adds dimensions but can mix more
robustly at a small budget, because the sampler walks the augmented joint rather
than a stiff marginal. We give it the same budget and time it too.
"""

latent_time = @elapsed latent_chain = sample(
    Xoshiro(1),
    latent_model(template, priors, sim_rows),
    NUTS(0.8; adtype = AutoForwardDiff()), 300;
    progress = false)

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

marginal_means_fit = edge_means(marginal_fit)

latent_means_fit = edge_means(latent_fit)

recovery = DataFrame(
    edge = ["onset_admit", "admit_death"],
    truth = [true_means.onset_admit, true_means.admit_death],
    marginal = [marginal_means_fit.onset_admit,
        marginal_means_fit.admit_death],
    latent = [latent_means_fit.onset_admit, latent_means_fit.admit_death])

md"""
The marginal fit recovers the true edge means closely. The latent fit, carrying
many extra sampled events at this small budget, mixes less freely and sits a
little high, but tracks the same posterior the marginal targets: the two forms
are one family, and a larger budget closes the gap.
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
    return (onset_admit = [e.onset_admit for e in em],
        admit_death = [e.admit_death for e in em])
end

marginal_means = mean_draws(marginal_chain)

latent_means = mean_draws(latent_chain)

truth_nt = (onset_admit = true_means.onset_admit,
    admit_death = true_means.admit_death)

md"""
The marginal posterior over the delay means concentrates on the true values; the
latent posterior, with more sampled events to explore at this small budget, is
wider and shifted high. Both summarise the same family of parameters through the
same `update`/`edge_means` path.
"""

pairplot(
    PairPlots.Series(marginal_means; label = "marginal",
        color = (:steelblue, 0.6)),
    PairPlots.Series(latent_means; label = "latent",
        color = (:orange, 0.6)),
    PairPlots.Truth(truth_nt, label = "True values"))

md"""
## Summary

- One composed distribution describes the whole record; the censored leaves and
  a [`Sequential`](@ref) chain build it, while [`params_table`](@ref) and
  [`build_priors`](@ref) derive support-aware priors.
- The marginal fit scores the whole record table through the vectorised
  [`composed_distribution_model`](@ref) on a `DataFrame` in one `~`.
- The marginal and latent forms are one family on the same parameters: wrapping
  the composed object in [`latent`](@ref) scores each record's intermediate
  event as a latent. The marginal recovers the truth at this small budget; the
  latent needs a larger budget to match it.
- The marginal form is the cheap default; the latent form costs an extra
  sampled event per record but suits small counts or impractical marginal
  integrals.
- [`update`](@ref)`(template, chain)` and [`edge_means`](@ref) read the fitted
  delays back onto the composed object with no manual chain indexing.
"""
