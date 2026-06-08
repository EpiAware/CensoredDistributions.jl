md"""
# Andes hantavirus realtime delay estimation

## Introduction

This walkthrough replicates the delay layer of the
[epiforecasts/andv-linelist-analysis](https://github.com/epiforecasts/andv-linelist-analysis)
study of the 2018–19 Epuyén Andes virus (ANDV) outbreak
([Martínez et al. 2020, NEJM](https://doi.org/10.1056/NEJMoa2009040)) on the
composable CensoredDistributions.jl stack.

Andes virus is the one hantavirus with documented person-to-person
transmission, and the Epuyén cluster was a sustained human-to-human outbreak.
The study fits a line list of cases, each with an interval-censored exposure
window and symptom-onset date, and a source attribution linking a secondary
case to the case that infected it.
From these data it estimates the incubation period, the timing of each onward
transmission relative to its source's symptom onset, a weekly reproduction
number, and offspring dispersion.

The realtime question is what an analyst could infer *during* the outbreak,
before all chains have run their course.
At a cut-off date some infected people have not yet shown symptoms, so the line
list seen so far is a right-truncated sample of the eventual one.
A naive fit to the truncated data underestimates the delays, so the likelihood
must correct for the cut-off.
This page reproduces that delay-and-truncation correction; the reproduction
number and the offspring clustering stay in the user's own Turing model.

### Estimands

- the **incubation period** Inc, the delay from infection to symptom onset
  (LogNormal);
- the **transmission timing** ``\delta``, the gap from a source's symptom onset
  to its secondary's infection, which can be negative when transmission is
  pre-symptomatic (Normal);
- the realtime **right-truncation** correction at the observation cut-off,
  which differs between index and sourced cases;
- the offspring **completeness** ``p``, the chance a source's secondary has been
  observed by the cut-off, which thins the realtime reproduction number.

### Out of scope (stays user Turing)

The weekly ``R(t)`` random walk and the offspring-count / case-cluster
likelihood are ordinary Turing code in the original model and are not delay
distributions, so this page does not reproduce them.

## Packages used
"""

using CensoredDistributions
using Distributions
using Turing
using DynamicPPL: prefix, to_submodel
using DataFramesMeta
using Random
using Statistics
import ForwardDiff

md"""
## Two record types, two truncation denominators

An **index** case has no recorded source.
We observe its single incubation delay from a latent infection time to onset,
right-truncated because the case only enters the data if its onset falls on or
before the cut-off.
A record's delay distribution is the incubation period.

A **sourced** case is attributed to an earlier case.
We observe the gap from the *source's* onset to the *secondary's* onset, which
is the convolution of the signed transmission timing ``\delta`` and the
secondary's incubation period.
The secondary's own infection time is not recorded, so only the total chain
delay is seen, and the truncation denominator must use the convolution of the
two unobserved segments rather than either one alone.
Carrying the two denominators side by side without mixing them up is the job the
original `truncation_model` does by hand.

A whole record is ONE composed distribution: the right-truncated delay reaching
the observation.
[`truncate_to_horizon`](@ref) builds the single-delay form and
[`truncate_chain`](@ref) the convolved-chain form, dispatching the two
denominators from the chain and the mask of which intermediate events were
observed.
"""

## The composed natural-history delay for each record type, reused in the
## simulation and the fit.
index_delay(inc) = inc

sourced_delay(delta, inc) = convolve_distributions(delta, inc)

## The right-truncated record distribution at the remaining observation window
## `horizon - anchor`. Index records have one observed segment (single-delay
## denominator); sourced records have an unobserved intermediate event, so
## `truncate_chain` collapses the two segments to their convolution.
index_record(inc, window) = truncate_to_horizon(inc, window)

function sourced_record(delta, inc, window)
    truncate_chain((delta, inc), (false,), window)
end

md"""
The right-truncation is upper-only: it caps the upper bound at the remaining
window and adds no lower bound.
This matters for a LogNormal incubation period, whose support starts at zero:
an explicit `lower = 0` would make `truncated` differentiate
`logcdf(LogNormal, 0) = -Inf` and return a `NaN` gradient.
The upper-only form keeps the gradient finite, so the index path is safe under
automatic differentiation.
"""

let
    f = x -> logpdf(truncate_to_horizon(LogNormal(x[1], x[2]), 17.48), 5.41)
    ForwardDiff.gradient(f, [2.06, 0.41])
end

md"""
## Simulating a line list

We fix generating delays and simulate a line list to recover them.
The transmission timing is centred below zero so some secondaries are infected
before their source's onset, as the study finds.

For each case we draw its natural-history delay with a single `rand` on the
composed delay distribution, then apply the realtime keep-rule: a case enters
the data only if it has been observed by the `horizon`.
That keep-rule is exactly the right-truncation the likelihood corrects for, so
the simulation and the fit share the same generative process.
"""

inc_true = LogNormal(2.06, 0.41)

delta_true = Normal(-1.0, 2.0)

horizon = 45.0

function simulate(rng, inc, delta, horizon; n_index = 110, n_sourced = 110)
    rows = NamedTuple[]
    for _ in 1:n_index
        anchor = rand(rng, Uniform(0, 25))
        y = rand(rng, index_delay(inc))             # single rand
        anchor + y <= horizon || continue           # realtime keep-rule
        push!(rows, (; kind = "index", y, window = horizon - anchor))
    end
    for _ in 1:n_sourced
        src_onset = rand(rng, Uniform(0, 18))
        y = rand(rng, sourced_delay(delta, inc))    # single rand
        src_onset + y <= horizon || continue
        push!(rows, (; kind = "sourced", y, window = horizon - src_onset))
    end
    return DataFrame(rows)
end

rng = Random.MersenneTwister(2024)

linelist = simulate(rng, inc_true, delta_true, horizon)

first(linelist, 4)

md"""
## Compressing to unique combinations

At day-level resolution many cases share an event-time combination, so the
study weights one likelihood evaluation per unique combination rather than one
per case.
We compress the line list with DataFramesMeta, counting cases per unique
combination of record type, observed delay, and window, and carry the count in
a reserved `weight` field that the submodel applies to the likelihood.
"""

combos = @chain linelist begin
    @groupby :kind :y :window
    @combine :weight = length(:y)
end

index_rows = @chain combos begin
    @rsubset :kind == "index"
end

sourced_rows = @chain combos begin
    @rsubset :kind == "sourced"
end

(nrow(linelist), nrow(combos))

md"""
## Delay priors as a submodel

The delay-distribution priors live in a small prior-submodel, included via
`to_submodel`, so they are configurable without editing the fitting model: pass
different priors as arguments to swap them.
The submodel returns the realised incubation and transmission-timing
distributions, which the fit reuses for every record.
"""

@model function delay_priors(;
        mu_inc_prior = Normal(2.0, 0.5),
        sigma_inc_prior = truncated(Normal(0.0, 0.5); lower = 0),
        mu_delta_prior = Normal(0.0, 3.0),
        sigma_delta_prior = truncated(Normal(0.0, 2.0); lower = 0))
    mu_inc ~ mu_inc_prior
    sigma_inc ~ sigma_inc_prior
    mu_delta ~ mu_delta_prior
    sigma_delta ~ sigma_delta_prior
    return (; inc = LogNormal(mu_inc, sigma_inc),
        delta = Normal(mu_delta, sigma_delta))
end

md"""
## Fitting through one submodel per record

Each record's whole distribution is scored through ONE generic
[`composed_distribution_model`](@ref) call, which dispatches on the
distribution and manages the censoring and truncation likelihood internally; the
user model never writes a per-component `logpdf` or `@addlogprob!`.
The reserved `weight` in the row scales the contribution and `prefix` namespaces
each record's submodel.

For a sourced record the source's onset is the coupled origin: it anchors the
observed gap before the record's distribution is dispatched.
That cross-record wiring is the one part that lives in the loop, by design; the
delay, censoring, and truncation of every record go through its submodel.
"""

@model function hanta_delays(index_rows, sourced_rows; priors = delay_priors())
    p ~ to_submodel(priors)
    for r in eachrow(index_rows)
        d = index_record(p.inc, r.window)
        row = (; y = r.y, weight = r.weight)
        idx ~ to_submodel(
            prefix(composed_distribution_model(d, row),
                Symbol(:idx, rownumber(r))), false)
    end
    for r in eachrow(sourced_rows)
        d = sourced_record(p.delta, p.inc, r.window)
        row = (; y = r.y, weight = r.weight)
        src ~ to_submodel(
            prefix(composed_distribution_model(d, row),
                Symbol(:src, rownumber(r))), false)
    end
end

chn = sample(hanta_delays(index_rows, sourced_rows), NUTS(0.95), 600)

md"""
## Recovering the generating delays

We check that the credible intervals cover the generating values.
The transmission timing ``\delta`` is only seen through the convolved sourced
chain, where its contribution overlaps the incubation period, so its location
and the incubation scale are weakly identified at this sample size.
We therefore check coverage with a wide (99%) credible interval rather than the
point estimate, which is the honest statement for a weakly identified
convolution.
"""

function covers(sym, truth_value; q = (0.005, 0.995))
    draws = vec(chn[sym])
    lo, hi = quantile(draws, q[1]), quantile(draws, q[2])
    return (truth = truth_value, lo = round(lo; digits = 2),
        hi = round(hi; digits = 2), covered = lo ≤ truth_value ≤ hi)
end

recovery = (
    mu_inc = covers(Symbol("p.mu_inc"), 2.06),
    sigma_inc = covers(Symbol("p.sigma_inc"), 0.41),
    mu_delta = covers(Symbol("p.mu_delta"), -1.0),
    sigma_delta = covers(Symbol("p.sigma_delta"), 2.0))

md"""
## Ascertainment thinning

A source's secondary is only counted if its chain has completed by the cut-off,
so the realtime reproduction number is thinned by the completion probability
``p`` to give ``R_{\text{eff}} = R \cdot p``.
That ``p`` is the chain-completion CDF, `cdf(sourced_delay(δ, inc), window)`,
reusing the same convolved delay the sourced records are fitted through.
"""

completeness_probability(delta, inc, window) = cdf(sourced_delay(delta, inc), window)

let
    R = 1.8
    R * completeness_probability(delta_true, inc_true, horizon - 5.0)
end

md"""
!!! note "Pending integration: thinning helpers"
    A dedicated completeness/thinning helper and analytic `Convolved` moments
    are being added on a separate branch.
    Until then this page computes the thinning explicitly with `cdf` of the
    convolved delay; the swap to the named helpers is a follow-up.

The reproduction number ``R`` this multiplies, and the offspring-count
likelihood it feeds, are the user's Turing code and stay out of scope here.

## Summary

The realtime ANDV delay layer maps onto the rebuilt stack:

- each record is ONE composed right-truncated delay distribution
  ([`truncate_to_horizon`](@ref) for index cases,
  [`truncate_chain`](@ref) for sourced cases over a
  [`convolve_distributions`](@ref) chain);
- the delay priors are a configurable prior-submodel;
- each record is fitted through ONE submodel that owns its censoring and
  truncation likelihood, with the multiplicity in the reserved `weight` field;
- the coupled source onset is wired in the loop, the one cross-record piece;
- ascertainment thinning ``R_{\text{eff}} = R \cdot p`` reuses the convolved
  chain CDF.

The reproduction number random walk and the offspring clustering stay in the
user's Turing model.

!!! note "Pending integration"
    The per-record dispatch will move to the generic `composed_distribution_model`
    entry point and the simulation to a `predict_events` draw once those land;
    this page uses the current `double_interval_censored_model` submodel and a
    direct `rand` in the meantime.
"""
