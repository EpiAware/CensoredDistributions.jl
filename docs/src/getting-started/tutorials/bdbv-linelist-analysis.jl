md"""
# [Bundibugyo Ebola delays from the 2012 Isiro line list](@id bdbv-linelist-analysis)

## Introduction

The 2012 outbreak of Bundibugyo ebolavirus in Isiro, Haut-Uele, in the
Democratic Republic of the Congo is the only publicly available Bundibugyo line
list.
It was deposited as part of the seven-outbreak line list of Rosello et al.
(2015), and a later Bayesian re-analysis at
[epiforecasts/bdbv-linelist-analysis](https://github.com/epiforecasts/bdbv-linelist-analysis)
re-estimated the delay distributions after correcting an error in the
notification delay.

This page is a Bayesian workflow built on the composed CensoredDistributions.jl
stack.
We build one composed distribution for the whole case, simulate a synthetic line
list from it, fit that simulation and check the known parameters are recovered,
then fit the real line list and check the posterior matches the re-estimated
study delays.

Four delays describe a case's progression through detection and care.
Symptom onset is followed by hospital admission, by notification to the
surveillance system, and, for admitted cases, by a resolution that is either
death or discharge.
The onset-to-admission delay sets how long an infectious case circulates before
isolation; the admission-to-death and admission-to-discharge delays size the
bed-days and the clinical course; the onset-to-notification delay measures
surveillance lag.

Two features of the data shape the model.
Dates are recorded to the day, so each delay is doubly interval censored: the
onset day and the later event day are both windows rather than instants.
Resolution is a disjunction: an admitted case either dies or is
discharged, never both, and the fraction that die is the case-fatality ratio.
A [`resolve`](@ref) node keeps the resolution time and the outcome together,
and lets the case-fatality ratio depend on covariates.

This page assembles the case as one composed distribution.
How the composer stack is built, scored, and simulated from in general is
covered in the [composer toolkit tutorial](@ref composer-toolkit); here we use
the stack and focus on the science.

### Replication targets

The comparison targets are the re-estimated posterior delays, with uncertainty,
from the corrected `output/posterior_gamma.csv` of the analysis repo, not the
raw Rosello et al. (2015) means.
The notification delay was corrected there, so the match is posterior to
posterior: our credible interval against theirs.

| Delay | Study n | Posterior mean (d) | 95% CrI |
|---|---|---|---|
| onset → admission | 40 | 4.09 | 3.08, 5.46 |
| admission → death | 22 | 7.71 | 5.63, 10.44 |
| admission → discharge | 15 | 8.10 | 4.81, 13.82 |
| onset → notification | 38 | 20.37 | 13.64, 29.99 |

`Study n` is the study's count for each delay.
Our comparison table below reports our own count per delay, which for
admission-to-discharge is a smaller resolved-discharge subset (n = 11) because
the mutually exclusive outcome assignment moves a few ambiguous cases to death,
explained at the end of the comparison.

## Packages used

We use DataFramesMeta and CSV with Dates for the line-list pipeline, Turing for
inference, FlexiChains for the posterior, PairPlots and CairoMakie for the plots,
and CensoredDistributions for the composed delay model.
"""

using CSV, DataFramesMeta, Dates
using CensoredDistributions, Distributions
using CensoredDistributions: latent_primary_priors, latent_observed_logpdf
using Turing, Random, Statistics
using DynamicPPL: to_submodel, @varname
using FlexiChains: Parameter
using ADTypes: AutoMooncake
import Mooncake
using PairPlots, CairoMakie

md"""
## The delay model

A single composed distribution describes the whole case.
From onset there are two branches: a chain through admission to resolution, and
a direct delay to notification.
Resolution is a [`Resolve`](@ref) node over death and discharge.

Each delay is a Gamma censored directly in the stack: every leaf is built with
[`double_interval_censored`](@ref), a one-day primary-event window and a one-day
secondary window, so the censoring is part of the composed object rather than a
separate step.
The chain's steps are named so the event slots come out clean: the tree's event
names are `onset, admit, death, discharge, notif`, exactly the columns a record
row supplies.

`delay_tree` takes a case-fatality ratio so the same builder serves both
simulation (a known `cfr`) and the template for fitting (its value is a
placeholder; the fit drives the split per record through the covariate model
below).
"""

function double_interval_censored_delay(d)
    double_interval_censored(d;
        primary_event = Uniform(0, 1),
        interval = 1.0)
end

function delay_tree(; cfr = 0.5)
    resolution = resolve(
        :death => (double_interval_censored_delay(Gamma(2.0, 3.5)), cfr),
        :discharge => (double_interval_censored_delay(Gamma(1.0, 8.0)), 1 - cfr))
    admit_path = sequential(
        :onset_admit => double_interval_censored_delay(Gamma(1.2, 3.0)),
        :admit_resolution => resolution)
    return compose((admit_path = admit_path,
        onset_notif = double_interval_censored_delay(Gamma(0.7, 20.0))))
end

template = delay_tree()

md"""
The recursive `show` lays out the branching structure, with the resolve death
and discharge outcomes nested under the admission step.
"""

template

md"""
The flat event names are the columns a data row supplies; a row is matched to
them by name, and a missing column drives whether that delay is conditioned on
or marginalised for that case.
"""

event_names(template)

md"""
## Priors

[`params_table`](@ref) lists every free parameter of the composed delays as a
flat table.
The Gamma shapes and scales are the free delay parameters; the resolve node's
branch probabilities are not estimated as free parameters here, because the
case-fatality ratio is covariate-driven (below) and enters per record.
We read the table into a DataFrame and drop the branch-probability rows with a
single `@rsubset`, then [`build_priors`](@ref) turns the remaining rows into the
nested prior NamedTuple that [`composed_parameters_model`](@ref) consumes.
"""

param_inventory = DataFrame(params_table(template))

delay_table = @rsubset param_inventory !endswith(string(:edge), "branch_probs")

md"""
[`build_priors`](@ref) gives every row a support-derived default: a
positive-support Gamma shape or scale becomes a positive-truncated Normal centred
on the template value with a width that scales with its magnitude, so the data
dominate.
We override only the four Gamma shapes, recentring them near one (the shape of an
exponential-like delay) rather than on the template.
[`update`](@ref) edits the prior table by path, the same name path
[`update`](@ref)`(d, path => new_node)` uses on the tree, with a tuple of names
for a nested edge and a bare name for a top-level one, then a `(shape = prior,)`
NamedTuple of the fields to replace.
So we name only the four leaves we care about and every other parameter keeps
its support-derived default, a brms-style partial override.
"""

shape_prior = truncated(Normal(1.0, 1.5); lower = 0.05)

priors = update(build_priors(delay_table),
    (:admit_path, :onset_admit) => (shape = shape_prior,),
    (:admit_path, :admit_resolution, :death) => (shape = shape_prior,),
    (:admit_path, :admit_resolution, :discharge) => (shape = shape_prior,),
    :onset_notif => (shape = shape_prior,));

md"""
## The case-fatality ratio

Whether an admitted case dies is a logistic regression on health-worker status,
a probable (rather than confirmed) case definition, and standardised age.
The regression stays in plain Turing code; the resolve node only consumes the
resulting probability.
The per-case death probability is passed in through the reserved `:branch_probs`
row field, and the node conditions on the observed outcome, so the
death-versus-discharge split carries the case-fatality information without a
separate likelihood.

`death_prob` is the one place the covariates enter, shared by the fit and the
simulation below so the two cannot drift: it maps a coefficient NamedTuple and a
record to the case's death probability.
"""

logistic(z) = 1 / (1 + exp(-z))

function death_prob(β, r)
    logistic(β.β0 + β.β_hcw * r.hcw + β.β_def * r.probable +
             β.β_age * r.age_z)
end

# The event columns a record supplies to the scorer, dropping the covariate
# fields that only feed `death_prob`. A row keeps these five named events; the
# per-case `:branch_probs` is then `merge`d in.
function event_columns(r)
    (onset = r.onset, admit = r.admit, death = r.death,
        discharge = r.discharge, notif = r.notif)
end

md"""
## Fitting through the vectorised interface

The whole record set fits through one [`composed_distribution_model`](@ref) call
on a vector of rows.
Each row carries the event columns, the per-case `:branch_probs` from
`death_prob`, and (when present) a `weight`.
The resolve node self-dispatches on which outcome column is present, and
unobserved delays for a case are marginalised internally.
"""

@model function bdbv(template, priors, rows)
    delays ~ to_submodel(composed_parameters_model(template, priors))

    β0 ~ Normal(0, 1.5)
    β_hcw ~ Normal(0, 1)
    β_def ~ Normal(0, 1)
    β_age ~ Normal(0, 1)
    β = (; β0, β_hcw, β_def, β_age)

    obs_rows = map(rows) do r
        p = death_prob(β, r)
        merge(event_columns(r),
            (branch_probs = (death = p, discharge = 1 - p),))
    end
    obs ~ to_submodel(composed_distribution_model(delays, obs_rows))
end

md"""
## Step 1: simulate from the model

The composed distribution is the model's generative half: the `obs` likelihood in
`bdbv` walks it to read a record, and `rand(d)` walks the same object to draw a
full event path for a new case.
A single draw returns the named event record (a labelled `NamedTuple`), with
exactly one of death or discharge populated by the resolve node.
"""

rand(MersenneTwister(1), delay_tree(cfr = 0.6))

md"""
We build a synthetic line list with the same two pieces the model uses: the
shared `death_prob` maps each case's covariates through the known coefficients to
its case-fatality ratio, and a draw from `delay_tree(cfr = p)` is the same object
the `obs` block scores. Simulation and fit therefore cannot drift, and the known
delay parameters and coefficients are what the next step checks recovery against.
"""

sim_truth = (β0 = -0.2, β_hcw = -0.8, β_def = 1.0, β_age = 0.6)

sim_rows = let rng = MersenneTwister(20260609), n = 200
    map(1:n) do _
        cov = (hcw = rand(rng) < 0.25, probable = rand(rng) < 0.3,
            age_z = randn(rng))
        p = death_prob(sim_truth, cov)
        s = rand(rng, delay_tree(cfr = p))
        (onset = s.onset, admit = s.admit, death = s.death,
            discharge = s.discharge, notif = s.notif,
            hcw = cov.hcw, probable = cov.probable, age_z = cov.age_z)
    end
end

(n = length(sim_rows),
    deaths = count(r -> r.death !== missing, sim_rows),
    discharges = count(r -> r.discharge !== missing, sim_rows))

md"""
## Step 2: fit the simulation and check recovery

We fit the synthetic line list and read the posterior back onto the composed
object with [`update`](@ref), passing the fitted chain directly.
The likelihood is differentiated with Mooncake reverse mode (`AutoMooncake`),
used for both the simulation fit and the real fit.
"""

adbackend = AutoMooncake(; config = nothing)

sim_chain = sample(Xoshiro(1), bdbv(template, priors, sim_rows),
    NUTS(0.8; adtype = adbackend), MCMCThreads(), 500, 2;
    progress = false)

sim_fit = update(template, sim_chain; prefix = :delays);

md"""
The per-event [`mean`](@ref CensoredDistributions.mean)`(latent(fit))`
NamedTuple reads every delay mean off any fitted composed object at once, keyed
by [`event_names`](@ref) and seeing through each censored leaf to its inner free
delay. `flat_means` picks the four delays we report, and `delay_mean_draws`
applies it to every posterior draw, giving the posterior distribution of each
delay mean through repeated [`update`](@ref) on the chain (one draw at a time)
rather than any manual chain indexing.
"""

function flat_means(fit)
    em = mean(latent(fit))
    return (onset_admit = em.admit, admit_death = em.death,
        admit_discharge = em.discharge, onset_notif = em.notif)
end

function delay_mean_draws(chain)
    rows = map(1:prod(size(chain))) do i
        flat_means(update(template, chain; prefix = :delays, draw = i))
    end
    return (onset_admit = [r.onset_admit for r in rows],
        admit_death = [r.admit_death for r in rows],
        admit_discharge = [r.admit_discharge for r in rows],
        onset_notif = [r.onset_notif for r in rows])
end

md"""
We tabulate each known simulation value against its posterior mean and 95%
credible interval.
The case-fatality coefficients are recovered cleanly, the true value inside the
interval for each, and the onset-to-admission delay recovers tightly around its
truth.
The heavy-tailed delays are harder to pin down from a finite sample: the
onset-to-notification delay (Gamma shape 0.7) carries its mean in rare long
delays, so its posterior is wide, and the admission-to-discharge mean is wide on
the smaller resolved-as-discharge subset.
The wide intervals on the tail-sensitive delays preview the same weak
identification on the real line list.
"""

ci(v) = (mean = mean(v), lower = quantile(v, 0.025),
    upper = quantile(v, 0.975))

sim_delays = delay_mean_draws(sim_chain)

coef_draws(name) = vec(sim_chain[Parameter(name)])

recovery = let
    qs = [ci(sim_delays.onset_admit), ci(sim_delays.admit_death),
        ci(sim_delays.admit_discharge), ci(sim_delays.onset_notif),
        ci(coef_draws(@varname(β0))),
        ci(coef_draws(@varname(β_hcw))),
        ci(coef_draws(@varname(β_def))),
        ci(coef_draws(@varname(β_age)))]
    DataFrame(
        quantity = ["onset → admission mean", "admission → death mean",
            "admission → discharge mean", "onset → notification mean",
            "β0", "β_hcw", "β_def", "β_age"],
        truth = [round(mean(Gamma(1.2, 3.0)), digits = 2),
            round(mean(Gamma(2.0, 3.5)), digits = 2),
            round(mean(Gamma(1.0, 8.0)), digits = 2),
            round(mean(Gamma(0.7, 20.0)), digits = 2),
            sim_truth.β0, sim_truth.β_hcw, sim_truth.β_def, sim_truth.β_age],
        post_mean = [round(q.mean, digits = 2) for q in qs],
        post_lower = [round(q.lower, digits = 2) for q in qs],
        post_upper = [round(q.upper, digits = 2) for q in qs])
end

md"""
## The real line list

The bundled line list is the Bundibugyo subset (n = 52) of the Rosello et al.
(2015) eLife deposit, redistributed under CC-BY 4.0 (see the data folder's
`README`).
We read the dates, express every event as a day offset from each case's onset,
and apply the cleaning of the re-analysis: a small set of admission dates and one
notification date encode biologically impossible offsets and are set to missing.
"""

datadir = joinpath(@__DIR__, "data", "bdbv")

linelist = CSV.read(joinpath(datadir, "linelist.csv"), DataFrame;
    missingstring = ["NA", ""])

md"""
Admission-date offsets the Rosello deposit encodes as outliers (−89, −5, −4, −1,
and 328720 days from onset) are set to missing, as are negative offsets.
Death and discharge are made mutually exclusive by the recorded outcome, so each
resolved case carries exactly one of the two as its resolve outcome.
"""

admit_outliers = (-89, -5, -4, -1, 328720)

parse_date(x) = ismissing(x) ? missing : Date(string(x))

dayoff(a, b) = (ismissing(a) || ismissing(b)) ? missing :
               Float64(Dates.value(b - a))

drop_neg(x) = (!ismissing(x) && x < 0) ? missing : x

md"""
The pipeline parses the date columns, drops cases with no onset date, takes each
delay as a day offset from onset, then cleans the offsets.
Date parsing and outcome flags are whole-column transforms (`@transform`,
`@subset`); the offset and missing-data rules are genuinely per row, so they stay
`@rtransform`.
"""

clean = @chain linelist begin
    @transform begin
        :Date_of_onset_symp = parse_date.(:Date_of_onset_symp)
        :Date_hospital_discharge = parse_date.(:Date_hospital_discharge)
        :Date_of_notification = parse_date.(:Date_of_notification)
        :Date_of_Hospitalisation = parse_date.(:Date_of_Hospitalisation)
        :Date_of_Death = parse_date.(:Date_of_Death)
    end
    @subset .!ismissing.(:Date_of_onset_symp)
    @rtransform begin
        :admit = dayoff(:Date_of_onset_symp, :Date_of_Hospitalisation)
        :death = dayoff(:Date_of_onset_symp, :Date_of_Death)
        :discharge = dayoff(:Date_of_onset_symp, :Date_hospital_discharge)
        :notif = dayoff(:Date_of_onset_symp, :Date_of_notification)
    end
    @transform begin
        :is_hcw = :hcw .=== true
        :probable = :Case_definition .== "Probable"
        :died = :Outcome .== "Dead"
    end
    @rtransform begin
        :admit = (!ismissing(:admit) && :admit in admit_outliers) ?
                 missing : drop_neg(:admit)
        :death = drop_neg(:death)
        :discharge = drop_neg(:discharge)
        :notif = drop_neg(:notif)
    end
    @rtransform begin
        :death = (ismissing(:admit) || !:died) ? missing : :death
        :discharge = (ismissing(:admit) || :died) ? missing : :discharge
    end
end

md"""
Age is standardised, imputing the sample mean for the few missing ages, and each
case becomes one record row.
The standardisation uses whole-column statistics, so it is one more `@transform`
step before the rows are assembled.
"""

age_mean = mean(skipmissing(clean.Age))
age_sd = std(skipmissing(clean.Age))

clean = @transform clean :age_z = map(clean.Age) do a
    ismissing(a) ? 0.0 : (a - age_mean) / age_sd
end

real_rows = map(eachrow(clean)) do r
    (onset = 0.0, admit = r.admit, death = r.death,
        discharge = r.discharge, notif = r.notif,
        hcw = r.is_hcw, probable = r.probable, age_z = r.age_z)
end

n_obs(field) = count(r -> !ismissing(getproperty(r, field)), real_rows)

(n = length(real_rows), admit = n_obs(:admit), death = n_obs(:death),
    discharge = n_obs(:discharge), notif = n_obs(:notif))

md"""
## Step 3: fit the real line list

The same model and priors fit the real records.
"""

real_chain = sample(Xoshiro(20260609), bdbv(template, priors, real_rows),
    NUTS(0.8; adtype = adbackend), MCMCThreads(), 600, 2;
    progress = false)

real_fit = update(template, real_chain; prefix = :delays);

md"""
## Step 4: equivalence with the published study

### Priors and posteriors together

PairPlots shows the prior and posterior of the four delay means on one figure.
The prior draws come from running the same model in prior mode, so the prior
delay means pass through the very same delay-mean map as the posterior draws and
the shrinkage from prior to posterior is read directly.
"""

prior_chain = sample(Xoshiro(3), bdbv(template, priors, real_rows),
    Prior(), MCMCThreads(), 500, 2; progress = false)

prior_means = delay_mean_draws(prior_chain)

posterior_means = delay_mean_draws(real_chain)

pairplot(
    PairPlots.Series(prior_means; label = "prior",
        color = (:grey, 0.4)),
    PairPlots.Series(posterior_means; label = "posterior",
        color = (:steelblue, 0.6)))

md"""
The posteriors are tighter than the priors and sit at data-driven delay means.
The two easy delays (onset-to-admission, admission-to-death) shrink to narrow
posteriors; the onset-to-notification and admission-to-discharge means stay
comparatively wide, reflecting the heavy tail and the small discharge subset.

### Posterior versus re-estimated targets

The re-estimated targets are the corrected posterior delays from the analysis
repo, matched posterior to posterior.
They feed the single three-way comparison below, once the latent fit is in hand,
so the overlap of marginal, latent and target is shown once.
"""

targets = (
    onset_admit = (mean = 4.09, lower = 3.08, upper = 5.46),
    admit_death = (mean = 7.71, lower = 5.63, upper = 10.44),
    admit_discharge = (mean = 8.10, lower = 4.81, upper = 13.82),
    onset_notif = (mean = 20.37, lower = 13.64, upper = 29.99))

md"""
## Marginal versus latent

The fit above is the marginal form; the original Isiro re-analysis at
[epiforecasts/bdbv-linelist-analysis](https://github.com/epiforecasts/bdbv-linelist-analysis)
used the latent form instead, and the two are one family on the same parameters
(see [Marginal versus latent](@ref marginal-versus-latent) for the concept and
[Fit marginal, sample event based](@ref) for the how-to).
This section runs the latent form on the real records and checks the two agree.

The latent model is the marginal `bdbv` with one swap.
[`latent_segments`](@ref)`(delays)` lowers the same composed `delays` object into
the latent form, a [`Choose`](@ref) of single-edge [`latent`](@ref) chains, one
per delay segment, that samples each origin event and conditions the observed
time on it at the floored gap rather than integrating it out.
[`latent_records`](@ref)`(template, obs_rows)` turns the SAME record schema the
marginal model scores (the event columns plus the per-record `:branch_probs`)
into the per-segment rows the vectorised latent path consumes.
The whole record set then scores in one
`primaries ~ product_distribution(...)` statement plus one
[`latent_observed_logpdf`](@ref), with the death-versus-discharge split folded in
through each resolved segment's branch probability, so there is no per-record
submodel and no hand-rolled segment machinery.
Because admission is recorded for every resolved case here, no admission time is
sampled and the onset → admission and admission → resolution segments are
independent observed leaves; a single-edge chain is density-identical to the
marginal leaf, so the latent fit recovers the same delays as the marginal fit.

The latent model shares the parameter block, the case-fatality regression, the
priors and the data with the marginal `bdbv`, differing only by the
[`latent_segments`](@ref) / [`latent_records`](@ref) wrapper on the composed
object.
"""

@model function bdbv_latent(template, priors, rows)
    delays ~ to_submodel(composed_parameters_model(template, priors))

    β0 ~ Normal(0, 1.5)
    β_hcw ~ Normal(0, 1)
    β_def ~ Normal(0, 1)
    β_age ~ Normal(0, 1)
    β = (; β0, β_hcw, β_def, β_age)

    obs_rows = map(rows) do r
        p = death_prob(β, r)
        merge(event_columns(r),
            (branch_probs = (death = p, discharge = 1 - p),))
    end

    segments = latent_segments(delays)
    table = latent_records(template, obs_rows)

    primaries ~ product_distribution(latent_primary_priors(segments, table))
    Turing.@addlogprob! latent_observed_logpdf(segments, table, primaries)
end

md"""
We fit the latent form to the same real records, at a modest budget.
The posterior is read back with the same [`update`](@ref) and per-event
[`mean`](@ref CensoredDistributions.mean)`(latent(fit))` machinery as the
marginal fit, since both share the one parameter block.
"""

latent_chain = sample(Xoshiro(20260609), bdbv_latent(template, priors,
        real_rows),
    NUTS(0.8; adtype = adbackend), MCMCThreads(), 400, 2;
    progress = false)

latent_means = delay_mean_draws(latent_chain)

md"""
### Marginal versus latent versus target

The marginal fit, the latent fit and the re-estimated target are read into one
three-way comparison, shown once, with each delay's own record count `n`.
The delay means agree between the marginal and latent fits and both overlap the
published intervals.
The onset-to-admission and admission-to-death delays recover well, matching the
target closely in both mean and width.
The other two are weakly identified.
The onset-to-notification delay has a heavy tail (Gamma shape 0.7): the long
right tail that carries most of the mean is poorly constrained by 38 records, so
the posterior mean is uncertain and the interval is wide, the same tail the
simulation showed pulling the mean low even with clean data.
The admission-to-discharge delay is the smallest pathway: discharge is the
minority outcome and the mutually exclusive assignment by recorded outcome moves
a few ambiguous cases to death, so it sits on a smaller subset (n = 11) and its
interval is the widest of the four.
Read these two as covered-but-uncertain rather than precisely recovered.
"""

mvl_comparison = let
    rows = NamedTuple[]
    keys_ = (:onset_admit, :admit_death, :admit_discharge, :onset_notif)
    labels = ["onset → admission", "admission → death",
        "admission → discharge", "onset → notification"]
    counts = [n_obs(:admit), n_obs(:death), n_obs(:discharge), n_obs(:notif)]
    for (lab, k, nk) in zip(labels, keys_, counts)
        m = ci(getfield(posterior_means, k))
        l = ci(getfield(latent_means, k))
        t = getfield(targets, k)
        push!(rows,
            (delay = lab, n = nk,
                marginal_mean = round(m.mean, digits = 2),
                marginal_ci = "($(round(m.lower, digits = 2)), " *
                              "$(round(m.upper, digits = 2)))",
                latent_mean = round(l.mean, digits = 2),
                latent_ci = "($(round(l.lower, digits = 2)), " *
                            "$(round(l.upper, digits = 2)))",
                target_mean = t.mean,
                target_ci = "($(t.lower), $(t.upper))"))
    end
    DataFrame(rows)
end

md"""
The case-fatality coefficients are recovered the same way by both formulations,
which confirms the Bernoulli case-fatality term in the latent model and the
`:branch_probs` term in the marginal model are scoring the same split.
"""

cfr_comparison = let
    names = (@varname(β0), @varname(β_hcw), @varname(β_def), @varname(β_age))
    labels = ["β0", "β_hcw", "β_def", "β_age"]
    marg(n) = ci(vec(real_chain[Parameter(n)]))
    lat(n) = ci(vec(latent_chain[Parameter(n)]))
    DataFrame(
        coefficient = labels,
        marginal_mean = [round(marg(n).mean, digits = 2) for n in names],
        latent_mean = [round(lat(n).mean, digits = 2) for n in names])
end

md"""
The overlay puts the two formulations against the target on one axis, so the
agreement reads directly.
"""

mvl_fig = let
    f = Figure(size = (760, 380))
    ax = Axis(f[1, 1]; xlabel = "mean delay (days)",
        yticks = (1:4, reverse(mvl_comparison.delay)),
        title = "Marginal vs latent vs re-estimated target")
    for (k, lab) in enumerate(mvl_comparison.delay)
        y = 5 - k
        m = ci(getfield(posterior_means,
            (:onset_admit, :admit_death, :admit_discharge, :onset_notif)[k]))
        l = ci(getfield(latent_means,
            (:onset_admit, :admit_death, :admit_discharge, :onset_notif)[k]))
        t = getfield(targets,
            (:onset_admit, :admit_death, :admit_discharge, :onset_notif)[k])
        lines!(ax, [m.lower, m.upper], [y + 0.22, y + 0.22];
            color = :steelblue, linewidth = 6)
        scatter!(ax, [m.mean], [y + 0.22]; color = :steelblue, markersize = 11)
        lines!(ax, [l.lower, l.upper], [y, y]; color = :seagreen,
            linewidth = 6)
        scatter!(ax, [l.mean], [y]; color = :seagreen, markersize = 11)
        lines!(ax, [t.lower, t.upper], [y - 0.22, y - 0.22];
            color = :firebrick, linewidth = 6)
        scatter!(ax, [t.mean], [y - 0.22]; color = :firebrick,
            markersize = 11)
    end
    scatter!(ax, [NaN], [NaN]; color = :steelblue, label = "marginal")
    scatter!(ax, [NaN], [NaN]; color = :seagreen, label = "latent")
    scatter!(ax, [NaN], [NaN]; color = :firebrick, label = "target")
    axislegend(ax; position = :rb)
    f
end

md"""
The latent form matches the formulation of the original Isiro analysis and would
expose any case's sampled origin event; here every resolved record has its
admission recorded, so the two forms coincide and the sections above use the
marginal form.
See [Marginal versus latent](@ref marginal-versus-latent) for when to reach for
each.
"""

mvl_comparison

md"""
## Compound delays from the fitted distribution

The natural-history delay from onset to death is the convolution of the
onset-to-admission and admission-to-death delays.
We pull the two fitted edges straight off the updated composed object with
[`event`](@ref) (descending each delay's dotted name path in one call) and
convolve their inner delays, with no re-fitting.
"""

onset_to_death = let
    inner(leaf) = CensoredDistributions.free_leaf(leaf)
    convolve_distributions(
        inner(event(real_fit, :admit_path, :onset_admit)),
        inner(event(real_fit, :admit_path, :admit_resolution, :death)))
end

(mean = mean(onset_to_death), std = std(onset_to_death))

md"""
## Summary

- The whole case is one composed distribution: a chain from onset through
  admission to a [`Resolve`](@ref) resolution, plus a notification branch, with
  every leaf censored directly through [`double_interval_censored`](@ref).
- The case-fatality ratio is a logistic regression in plain Turing, fed to the
  resolve node per record through the reserved `:branch_probs` field.
- Priors come from [`params_table`](@ref) and [`build_priors`](@ref), which
  derives weakly informative defaults from each parameter's support; the records
  score through the vectorised [`composed_distribution_model`](@ref).
- The workflow simulates from the model with `rand(d)`, whose draws are unbiased
  now that the censoring is applied as `floor(target) - floor(origin)`, fits the
  real line list, and overlaps the re-estimated study delays within uncertainty
  for all four delays.
- The posterior is read back with [`update`](@ref) and the per-event
  [`mean`](@ref CensoredDistributions.mean)`(latent(fit))` NamedTuple, so delay
  means and the onset-to-death convolution come straight from the fitted object.
- The same model is fit in the marginal form and, through the package's
  vectorised latent path, the latent form that matches the original Isiro
  analysis; both recover the same delays and coefficients, so the marginal form
  is preferred for speed.
- Recovery is honest about identifiability, with onset-to-admission and
  admission-to-death recovering well, whereas the heavy-tailed
  onset-to-notification (Gamma shape 0.7) and the small-n admission-to-discharge
  (n = 11) are weakly identified with wide intervals, a tail effect the
  simulation reproduces.
"""
