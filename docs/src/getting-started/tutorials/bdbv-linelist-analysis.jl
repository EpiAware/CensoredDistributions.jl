md"""
# Bundibugyo Ebola delays from the 2012 Isiro line list

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
Resolution is a competing-risks problem: an admitted case either dies or is
discharged, never both, and the fraction that die is the case-fatality ratio.
A competing-outcomes node keeps the resolution time and the outcome together,
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

| Delay | n | Posterior mean (d) | 95% CrI |
|---|---|---|---|
| onset → admission | 40 | 4.09 | 3.08, 5.46 |
| admission → death | 22 | 7.71 | 5.63, 10.44 |
| admission → discharge | 15 | 8.10 | 4.81, 13.82 |
| onset → notification | 38 | 20.37 | 13.64, 29.99 |

## Packages used

We use DataFramesMeta and CSV with Dates for the line-list pipeline, Turing for
inference, FlexiChains for the posterior, PairPlots and CairoMakie for the plots,
and CensoredDistributions for the composed delay model.
"""

using CSV, DataFramesMeta, Dates
using CensoredDistributions, Distributions
using Turing, Random, Statistics
using DynamicPPL: to_submodel, @varname
using FlexiChains: Parameter
using ADTypes: AutoForwardDiff
using PairPlots, CairoMakie

md"""
## The delay model

A single composed distribution describes the whole case.
From onset there are two branches: a chain through admission to resolution, and
a direct delay to notification.
Resolution is a [`Competing`](@ref) node over death and discharge.

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

dic(d) = double_interval_censored(d; primary_event = Uniform(0, 1),
    interval = 1.0)

function delay_tree(; cfr = 0.5)
    resolution = Competing(
        :death => (dic(Gamma(2.0, 3.5)), cfr),
        :discharge => (dic(Gamma(1.0, 8.0)), 1 - cfr))
    admit_path = Sequential(
        (dic(Gamma(1.2, 3.0)), resolution),
        (:onset_admit, :admit_resolution))
    return compose((admit_path = admit_path,
        onset_notif = dic(Gamma(0.7, 20.0))))
end

template = delay_tree()

md"""
The recursive `show` lays out the branching structure, with the competing death
and discharge outcomes nested under the admission step.
"""

template

md"""
The flat event names are the columns a data row supplies; a row is matched to
them by name, and a missing column drives whether that delay is conditioned on
or marginalised for that case.
"""

CensoredDistributions.tree_event_names(template)

md"""
## Priors

[`params_table`](@ref) lists every free parameter of the composed delays as a
flat table.
The Gamma shapes and scales are the free delay parameters; the competing node's
branch probabilities are not estimated as free parameters here, because the
case-fatality ratio is covariate-driven (below) and enters per record.
We drop the branch-probability rows from the table, then [`build_priors`](@ref)
turns the remaining rows into the nested prior NamedTuple that
[`composed_parameters_model`](@ref) consumes.
"""

param_inventory = params_table(template)

is_cfr_row(edge) = endswith(string(edge), "branch_probs")

delay_table = let keep = [!is_cfr_row(e) for e in param_inventory.edge]
    (edge = param_inventory.edge[keep],
        param = param_inventory.param[keep],
        value = param_inventory.value[keep],
        support = param_inventory.support[keep])
end

md"""
[`build_priors`](@ref) gives every row a support-derived default: a
positive-support Gamma shape or scale becomes a positive-truncated Normal centred
on the template value with a width that scales with its magnitude, so the data
dominate.
We override only the four Gamma shapes, recentring them near one (the shape of an
exponential-like delay) rather than on the template, a brms-style partial
override that leaves the scales on their defaults.
"""

shape_prior = truncated(Normal(1.0, 1.5); lower = 0.05)

shape_overrides = (
    admit_path = (
        onset_admit = (shape = shape_prior,),
        admit_resolution = (death = (shape = shape_prior,),
            discharge = (shape = shape_prior,))),
    onset_notif = (shape = shape_prior,))

priors = build_priors(delay_table; priors = shape_overrides)

md"""
## The case-fatality ratio

Whether an admitted case dies is a logistic regression on health-worker status,
a probable (rather than confirmed) case definition, and standardised age.
The regression stays in plain Turing code; the competing node only consumes the
resulting probability.
The per-case death probability is passed in through the reserved `:branch_probs`
row field, and the node conditions on the observed outcome, so the
death-versus-discharge split carries the case-fatality information without a
separate likelihood.
"""

logistic(z) = 1 / (1 + exp(-z))

md"""
## Scoring through the vectorised interface

The whole record set scores through one [`composed_distribution_model`](@ref)
call on a vector of rows.
Each row carries the event columns, the per-case `:branch_probs`, and (when
present) a `weight`; the same object fits and generates.
The competing node self-dispatches on which outcome column is present, and
unobserved delays for a case are marginalised internally.
"""

@model function bdbv(template, priors, rows)
    delays ~ to_submodel(composed_parameters_model(template, priors))

    β0 ~ Normal(0, 1.5)
    β_hcw ~ Normal(0, 1)
    β_def ~ Normal(0, 1)
    β_age ~ Normal(0, 1)

    obs_rows = map(rows) do r
        p = logistic(β0 + β_hcw * r.hcw + β_def * r.probable +
                     β_age * r.age_z)
        (onset = r.onset, admit = r.admit, death = r.death,
            discharge = r.discharge, notif = r.notif,
            branch_probs = (death = p, discharge = 1 - p))
    end
    obs ~ to_submodel(composed_distribution_model(delays, obs_rows))
end

md"""
## Step 1: simulate from the model

The composed distribution is used in both directions: scoring walks it to read a
record's likelihood, and [`predict_events`](@ref) walks the same object to draw a
full event path for a new case.
A single draw returns the named event record, with exactly one of death or
discharge populated by the competing node.
"""

predict_events(delay_tree(cfr = 0.6); rng = MersenneTwister(1))

md"""
We build a synthetic line list this way.
Each simulated case has covariates and a known case-fatality ratio, set through
the `cfr` of its own draw, so the simulation and the fit share the same
mechanism.
The true delay parameters and regression coefficients are known, which is what
lets the next step check recovery.
"""

sim_truth = (β0 = -0.2, β_hcw = -0.8, β_def = 1.0, β_age = 0.6)

sim_rows = let rng = MersenneTwister(20260609), n = 200
    map(1:n) do _
        hcw = rand(rng) < 0.25
        probable = rand(rng) < 0.3
        age_z = randn(rng)
        p = logistic(sim_truth.β0 + sim_truth.β_hcw * hcw +
                     sim_truth.β_def * probable + sim_truth.β_age * age_z)
        s = predict_events(delay_tree(cfr = p); rng = rng)
        (onset = s.onset, admit = s.admit, death = s.death,
            discharge = s.discharge, notif = s.notif,
            hcw = hcw, probable = probable, age_z = age_z)
    end
end

(n = length(sim_rows),
    deaths = count(r -> r.death !== missing, sim_rows),
    discharges = count(r -> r.discharge !== missing, sim_rows))

md"""
## Step 2: fit the simulation and check recovery

We fit the synthetic line list and read the posterior back onto the composed
object with [`update`](@ref), passing the fitted chain directly.
[`edge_means`](@ref) then reads each delay's mean off the updated distribution,
so there is no manual chain indexing.

The likelihood is differentiated with forward mode (`AutoForwardDiff`).
Mooncake reverse mode is the preferred backend for a tree this size, but it
cannot compile a rule here: the tree's dotted edge-name handling compiles a
`Regex`, and Mooncake does not differentiate the try/catch inside regex
compilation, so we fall back to forward mode for this page.
The same backend is used for both the simulation fit and the real fit.
"""

adbackend = AutoForwardDiff()

sim_chain = sample(Xoshiro(1), bdbv(template, priors, sim_rows),
    NUTS(0.8; adtype = adbackend), MCMCThreads(), 600, 2;
    progress = false)

sim_fit = update(template, sim_chain; prefix = :delays)

md"""
[`edge_means`](@ref) reads every delay mean off any fitted composed object at
once, keyed by edge name and seeing through each censored leaf to its inner free
delay. `flat_means` names the four delays we report, and `delay_mean_draws`
applies it to every posterior draw, giving the posterior distribution of each
delay mean through repeated [`update`](@ref) on the chain (one draw at a time)
rather than any manual chain indexing.
"""

function flat_means(fit)
    em = edge_means(fit)
    return (onset_admit = em.admit_path.onset_admit,
        admit_death = em.admit_path.admit_resolution.death,
        admit_discharge = em.admit_path.admit_resolution.discharge,
        onset_notif = em.onset_notif)
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
The other three are harder.
The heavy-tailed onset-to-notification delay (Gamma shape 0.7) is the worst: its
mean is carried by rare long delays, so the posterior is pulled low (around 12
against a true 14) with a wide interval that the truth sits just above.
The admission-to-death mean comes back a little low for the same reason on a
lighter tail, and the admission-to-discharge mean is recovered but wide on the
smaller resolved-as-discharge subset.
So even with clean simulated data the two tail-sensitive delays are only weakly
identified, which previews the same split on the real line list.
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

datadir = joinpath(@__DIR__, "data")

linelist = CSV.read(joinpath(datadir, "linelist.csv"), DataFrame;
    missingstring = ["NA", ""])

md"""
Admission-date offsets the Rosello deposit encodes as outliers (−89, −5, −4, −1,
and 328720 days from onset) are set to missing, as are negative offsets.
Death and discharge are made mutually exclusive by the recorded outcome, so each
resolved case carries exactly one of the two as its competing outcome.
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
    NUTS(0.8; adtype = adbackend), MCMCThreads(), 800, 2;
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
    Prior(), MCMCThreads(), 1000, 2; progress = false)

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
## Compound delays from the fitted distribution

The natural-history delay from onset to death is the convolution of the
onset-to-admission and admission-to-death delays.
We take the two fitted edges from the updated composed object and convolve their
inner delays, with no re-fitting.
"""

onset_to_death = let
    ap = get_event(real_fit, :admit_path)
    res = get_event(ap, :admit_resolution)
    inner(leaf) = CensoredDistributions.free_leaf(leaf)
    convolve_distributions(inner(get_event(ap, :onset_admit)),
        inner(get_event(res, :death)))
end

(mean = mean(onset_to_death), std = std(onset_to_death))

md"""
## Summary

- The whole case is one composed distribution: a chain from onset through
  admission to a [`Competing`](@ref) resolution, plus a notification branch, with
  every leaf censored directly through [`double_interval_censored`](@ref).
- The case-fatality ratio is a logistic regression in plain Turing, fed to the
  competing node per record through the reserved `:branch_probs` field.
- Priors come from [`params_table`](@ref) and [`build_priors`](@ref), which
  derives weakly informative defaults from each parameter's support; the records
  score through the vectorised [`composed_distribution_model`](@ref).
- The workflow simulates from the model with [`predict_events`](@ref), whose
  draws are unbiased now that the censoring is applied as `floor(target) -
  floor(origin)`, fits the real line list, and overlaps the re-estimated study
  delays within uncertainty for all four delays.
- The posterior is read back with [`update`](@ref) applied to the fitted chain
  and [`edge_means`](@ref), so delay means and the onset-to-death convolution come
  straight from the fitted object.
- Recovery is honest about identifiability: onset-to-admission and
  admission-to-death recover well, whereas the heavy-tailed onset-to-notification
  (Gamma shape 0.7) and the small-n admission-to-discharge (n = 11) are weakly
  identified, with wide intervals; the simulation shows the heavy tail pulling
  the notification mean low even with clean data, a tail/identifiability effect
  rather than the old simulation artefact.
"""
