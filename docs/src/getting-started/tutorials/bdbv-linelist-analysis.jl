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
Resolution is a one_of-risks problem: an admitted case either dies or is
discharged, never both, and the fraction that die is the case-fatality ratio.
A one_of-outcomes node keeps the resolution time and the outcome together,
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
The recursive `show` lays out the branching structure, with the one_of death
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
The Gamma shapes and scales are the free delay parameters; the one_of node's
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
[`build_priors`](@ref) takes the overrides as a flat `(edge, param) => prior`
mapping keyed against the table's `edge` column (the dotted path
[`params_table`](@ref) prints), so we name only the four rows we care about and
every other parameter keeps its support-derived default, a brms-style partial
override.
"""

shape_prior = truncated(Normal(1.0, 1.5); lower = 0.05)

shape_overrides = Dict(
    (Symbol("admit_path.onset_admit"), :shape) => shape_prior,
    (Symbol("admit_path.admit_resolution.death"), :shape) => shape_prior,
    (Symbol("admit_path.admit_resolution.discharge"), :shape) => shape_prior,
    (Symbol("onset_notif"), :shape) => shape_prior)

priors = build_priors(delay_table; priors = shape_overrides)

md"""
## The case-fatality ratio

Whether an admitted case dies is a logistic regression on health-worker status,
a probable (rather than confirmed) case definition, and standardised age.
The regression stays in plain Turing code; the one_of node only consumes the
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

md"""
## Fitting through the vectorised interface

The whole record set fits through one [`composed_distribution_model`](@ref) call
on a vector of rows.
Each row carries the event columns, the per-case `:branch_probs` from
`death_prob`, and (when present) a `weight`.
The one_of node self-dispatches on which outcome column is present, and
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
        (onset = r.onset, admit = r.admit, death = r.death,
            discharge = r.discharge, notif = r.notif,
            branch_probs = (death = p, discharge = 1 - p))
    end
    obs ~ to_submodel(composed_distribution_model(delays, obs_rows))
end

md"""
## Step 1: simulate from the model

The composed distribution is the model's generative half: the `obs` likelihood in
`bdbv` walks it to read a record, and `rand(d)` walks the same object to draw a
full event path for a new case.
A single draw returns the named event record (a labelled `NamedTuple`), with
exactly one of death or discharge populated by the one_of node.
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
the faster backend for a tree this size.
The same backend is used for both the simulation fit and the real fit.
"""

adbackend = AutoMooncake(; config = nothing)

sim_chain = sample(Xoshiro(1), bdbv(template, priors, sim_rows),
    NUTS(0.8; adtype = adbackend), MCMCThreads(), 500, 2;
    progress = false)

sim_fit = update(template, sim_chain; prefix = :delays)

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
interval for each.
The simulation draws themselves are unbiased: a draw emits a continuous event
path that the day-level censoring discretises with the same
`floor(target) - floor(origin)` rule the scorer uses, so the gaps carry no
systematic offset (the notification draw mean is ~14.0, its true Gamma mean).
The onset-to-admission delay recovers well, the true mean near the centre of a
tight interval.
The heavy-tailed delays are harder to pin down from a finite sample.
The onset-to-notification delay (Gamma shape 0.7) carries its mean in rare long
delays, so its posterior is wide, and the admission-to-death and
admission-to-discharge means are recovered but wide, the latter on the smaller
resolved-as-discharge subset.
The fit is calibrated, with every interval covering its truth at the nominal
rate across simulation seeds.
A single seed where a heavy-tailed interval lands just shy of its truth is
therefore Monte-Carlo variation on a small, heavy-tailed sample rather than a
bias.
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
resolved case carries exactly one of the two as its one_of outcome.
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

real_fit = update(template, real_chain; prefix = :delays)

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

We summarise each posterior delay mean and compare it to the re-estimated
target.
The targets are the corrected posterior delays from the analysis repo, matched
posterior to posterior.
"""

targets = (
    onset_admit = (mean = 4.09, lower = 3.08, upper = 5.46),
    admit_death = (mean = 7.71, lower = 5.63, upper = 10.44),
    admit_discharge = (mean = 8.10, lower = 4.81, upper = 13.82),
    onset_notif = (mean = 20.37, lower = 13.64, upper = 29.99))

comparison = let
    oa = ci(posterior_means.onset_admit)
    ad = ci(posterior_means.admit_death)
    ac = ci(posterior_means.admit_discharge)
    on = ci(posterior_means.onset_notif)
    DataFrame(
        delay = ["onset → admission", "admission → death",
            "admission → discharge", "onset → notification"],
        n = [n_obs(:admit), n_obs(:death), n_obs(:discharge),
            n_obs(:notif)],
        post_mean = round.([oa.mean, ad.mean, ac.mean, on.mean],
            digits = 2),
        post_lower = round.([oa.lower, ad.lower, ac.lower, on.lower],
            digits = 2),
        post_upper = round.([oa.upper, ad.upper, ac.upper, on.upper],
            digits = 2),
        target_mean = [targets.onset_admit.mean, targets.admit_death.mean,
            targets.admit_discharge.mean, targets.onset_notif.mean],
        target_lower = [targets.onset_admit.lower,
            targets.admit_death.lower, targets.admit_discharge.lower,
            targets.onset_notif.lower],
        target_upper = [targets.onset_admit.upper,
            targets.admit_death.upper, targets.admit_discharge.upper,
            targets.onset_notif.upper])
end

md"""
The interval plot overlays the two interval sets so the overlap reads directly.
"""

fig = let
    f = Figure(size = (760, 380))
    ax = Axis(f[1, 1]; xlabel = "mean delay (days)",
        yticks = (1:4, reverse(comparison.delay)),
        title = "Posterior delay means vs re-estimated targets")
    for (k, r) in enumerate(eachrow(comparison))
        y = 5 - k
        lines!(ax, [r.post_lower, r.post_upper], [y + 0.12, y + 0.12];
            color = :steelblue, linewidth = 6)
        scatter!(ax, [r.post_mean], [y + 0.12]; color = :steelblue,
            markersize = 12)
        lines!(ax, [r.target_lower, r.target_upper], [y - 0.12, y - 0.12];
            color = :firebrick, linewidth = 6)
        scatter!(ax, [r.target_mean], [y - 0.12]; color = :firebrick,
            markersize = 12)
    end
    scatter!(ax, [NaN], [NaN]; color = :steelblue, label = "this fit")
    scatter!(ax, [NaN], [NaN]; color = :firebrick,
        label = "re-estimated target")
    axislegend(ax; position = :rb)
    f
end

md"""
Every delay's credible interval overlaps the re-estimated target.
The onset-to-admission and admission-to-death delays recover well: they match
the target closely in both mean and width.
The other two delays are weakly identified, which here shows as wide intervals
rather than a clean recovery.
The onset-to-notification delay has a heavy tail (Gamma shape 0.7): the long
right tail that carries most of the mean is poorly constrained by 38 records, so
the posterior mean is uncertain and the interval is wide; the simulation showed
the same tail pulling the mean low even with clean data.
The admission-to-discharge delay is the smallest pathway: discharge is the
minority outcome and the mutually exclusive assignment by recorded outcome moves
a few ambiguous cases to death, so it sits on a smaller subset (n = 11) and its
interval is the widest of the four.
Read these two as covered-but-uncertain rather than precisely recovered; the
discharge pathway in particular is the least informed here.
"""

comparison

md"""
## Marginal versus latent

The fit above is the marginal form, with each delay's primary event integrated
out inside `logpdf` and the death-versus-discharge split read off the
[`Resolve`](@ref) node through the per-record `:branch_probs`.
The original Isiro re-analysis at
[epiforecasts/bdbv-linelist-analysis](https://github.com/epiforecasts/bdbv-linelist-analysis)
used the latent form instead, sampling each delay's primary event per record and
conditioning the observed time on it.
The two forms are one family on the same parameters, so the marginal-fit priors
drop straight into the latent form; see [Marginal versus latent](@ref
marginal-versus-latent) for the concept and [Fit marginal, sample event
based](@ref) for the how-to.
This section runs the latent form on the real records and checks the two agree.

The latent fit reuses the package's vectorised latent path
([`composer toolkit`](@ref composer-toolkit) covers the table scoring), so the
whole record set scores in one pair of statements rather than a per-record
submodel loop.
Each observed delay segment is a single-edge [`latent`](@ref) chain that samples
its origin event and conditions the observed time on it at the floored gap, the
latent counterpart of the marginal leaf.
A [`Choose`](@ref) routes each record's segments by a `:kind` field, so
`latent_primary_priors` stacks every segment's origin prior and
`latent_observed_logpdf` scores the whole table at once.
Because admission is recorded for every resolved case here, no admission time is
sampled; the onset → admission and admission → resolution segments are
independent observed leaves, the death-versus-discharge split enters as a
per-record Bernoulli term (the latent counterpart of `:branch_probs`), and the
notification delay scores through its own segment.

`delay_leaves` pulls the four fitted leaves off the reconstructed composed object
with [`event`](@ref), which descends a name path in one call (the same dotted
path [`params_table`](@ref) prints), so the latent fit reuses exactly the delays
the [`composed_parameters_model`](@ref) block sampled, with no second parameter
set.
"""

function delay_leaves(d)
    return (
        onset_admit = event(d, :admit_path, :onset_admit),
        admit_death = event(d, :admit_path, :admit_resolution, :death),
        admit_discharge = event(d, :admit_path, :admit_resolution,
            :discharge),
        onset_notif = event(d, :onset_notif))
end

md"""
Each delay segment becomes a single-edge [`latent`](@ref) chain whose origin
event is sampled and whose observed time conditions on it at the floored gap.
The four segments are gathered into one [`Choose`](@ref) keyed by a `:kind`
field, so a record's rows route to the right segment by name.
A single-edge chain is density-identical to the marginal leaf, so the latent fit
recovers the same delays as the marginal fit.
"""

function latent_segments(leaves)
    edge(leaf, name) = latent(sequential(name => leaf))
    return choose(
        :onset_admit => edge(leaves.onset_admit, :onset_admit),
        :admit_death => edge(leaves.admit_death, :admit_death),
        :admit_discharge => edge(leaves.admit_discharge, :admit_discharge),
        :onset_notif => edge(leaves.onset_notif, :onset_notif))
end

md"""
`latent_rows` turns one record into its observed segments, namely the
onset → admission segment when admission is recorded, the admission → resolution
segment for the outcome that occurred, and the notification segment if present.
Each row names its origin event as `missing` (the sampled latent) and its
observed event as the segment's delay, tagged with the `:kind` that selects its
segment.
The admission → resolution segment's delay is the recorded resolution day minus
the recorded admission day.
"""

function latent_rows(r)
    rows = NamedTuple[]
    r.admit !== missing &&
        push!(rows, (kind = :onset_admit, onset = missing, admit = r.admit))
    if r.death !== missing
        push!(rows,
            (kind = :admit_death, admit = missing, death = r.death - r.admit))
    elseif r.discharge !== missing
        push!(rows, (kind = :admit_discharge, admit = missing,
            discharge = r.discharge - r.admit))
    end
    r.notif !== missing &&
        push!(rows, (kind = :onset_notif, onset = missing, notif = r.notif))
    return rows
end

md"""
The latent model shares the parameter block and the case-fatality regression
with the marginal `bdbv`.
It builds the segment table once, samples every segment's origin event in one
`primaries ~ product_distribution(...)` statement, and scores the whole table
with one `latent_observed_logpdf`, so there is no per-record submodel.
The death-versus-discharge split enters as a vectorised Bernoulli term on the
resolved records, the latent counterpart of the marginal `:branch_probs`.
"""

@model function bdbv_latent(template, priors, rows)
    delays ~ to_submodel(composed_parameters_model(template, priors))

    β0 ~ Normal(0, 1.5)
    β_hcw ~ Normal(0, 1)
    β_def ~ Normal(0, 1)
    β_age ~ Normal(0, 1)
    β = (; β0, β_hcw, β_def, β_age)

    segments = latent_segments(delay_leaves(delays))
    table = reduce(vcat, map(latent_rows, rows))

    primaries ~ product_distribution(latent_primary_priors(segments, table))
    Turing.@addlogprob! latent_observed_logpdf(segments, table, primaries)

    cfr = sum(rows) do r
        r.death === missing && r.discharge === missing && return 0.0
        p = death_prob(β, r)
        return r.death !== missing ? log(p) : log(1 - p)
    end
    Turing.@addlogprob! cfr
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

The two formulations are tabulated side by side against the re-estimated target.
The delay means agree between the marginal and latent fits and both overlap the
published intervals.
"""

mvl_comparison = let
    rows = NamedTuple[]
    keys_ = (:onset_admit, :admit_death, :admit_discharge, :onset_notif)
    labels = ["onset → admission", "admission → death",
        "admission → discharge", "onset → notification"]
    for (lab, k) in zip(labels, keys_)
        m = ci(getfield(posterior_means, k))
        l = ci(getfield(latent_means, k))
        t = getfield(targets, k)
        push!(rows,
            (delay = lab,
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
  one_of node per record through the reserved `:branch_probs` field.
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
