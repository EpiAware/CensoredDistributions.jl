# # Bundibugyo Ebola delays from the 2012 Isiro line list
#
# The 2012 outbreak of Bundibugyo ebolavirus in Isiro, Haut-Uele, in the
# Democratic Republic of the Congo is the only Bundibugyo line list that is
# publicly available.
# It was deposited as part of the seven-outbreak line list of Rosello et al.
# (2015), and a later Bayesian re-analysis at
# [epiforecasts/bdbv-linelist-analysis](https://github.com/epiforecasts/bdbv-linelist-analysis)
# re-estimated the delay distributions after correcting an error in the
# notification delay.
# This page fits the same delays with CensoredDistributions.jl and checks that
# the corrected estimates are recovered.
#
# Four delays describe a case's progression through detection and care.
# Symptom onset is followed by hospital admission, by notification to the
# surveillance system, and, for admitted cases, by a resolution that is either
# death or discharge.
# Each delay is informative for a different operational question.
# The onset-to-admission delay sets how long an infectious case circulates in
# the community before isolation.
# The admission-to-death and admission-to-discharge delays size the bed-days
# and the clinical course.
# The onset-to-notification delay measures surveillance lag.
#
# Two features of the data shape the model.
# Dates are recorded to the day, so each delay is doubly interval censored: the
# onset day and the later event day are both windows rather than instants.
# Resolution is a competing-risks problem: an admitted case either dies or is
# discharged, never both, and the fraction that die is the case-fatality ratio.
# Treating death and discharge as a single "time to resolution" would discard
# the outcome; treating them as independent delays would double count cases.
# A competing-outcomes node keeps the resolution time and the outcome together,
# and lets the case-fatality ratio depend on covariates.

# ## Data
#
# The bundled line list is the Bundibugyo subset (n = 52) of the Rosello et al.
# (2015) eLife deposit, redistributed here under CC-BY 4.0 (see the data
# folder's `README`).
# We read the dates, express every event as a day offset from each case's
# onset, and apply the cleaning of the re-analysis: a small set of admission
# dates and one notification date encode biologically impossible offsets and
# are set to missing.

using CSV, DataFramesMeta, Dates
using CensoredDistributions, Distributions
using Turing, Random, Statistics
using CairoMakie

datadir = joinpath(@__DIR__, "data")
ll = CSV.read(joinpath(datadir, "linelist.csv"), DataFrame;
    missingstring = ["NA", ""])

for c in (:Date_of_onset_symp, :Date_hospital_discharge,
    :Date_of_notification, :Date_of_Hospitalisation, :Date_of_Death)
    ll[!, c] = map(x -> ismissing(x) ? missing : Date(string(x)), ll[!, c])
end

# Admission-date offsets that the Rosello deposit encodes as outliers
# (-89, -5, -4, -1, and 328720 days from onset).
const ADMIT_OUTLIERS = (-89, -5, -4, -1, 328720)
dayoff(a, b) = (ismissing(a) || ismissing(b)) ? missing :
               Float64(Dates.value(b - a))

# Each case becomes one row of day offsets from onset, plus the covariates for
# the case-fatality model.
# Death and discharge are made mutually exclusive by the recorded outcome, so
# each resolved case carries exactly one of the two as its competing outcome.
clean = @chain ll begin
    @rsubset !ismissing(:Date_of_onset_symp)
    @rtransform begin
        :admit = dayoff(:Date_of_onset_symp, :Date_of_Hospitalisation)
        :death = dayoff(:Date_of_onset_symp, :Date_of_Death)
        :discharge = dayoff(:Date_of_onset_symp, :Date_hospital_discharge)
        :notif = dayoff(:Date_of_onset_symp, :Date_of_notification)
        :is_hcw = :hcw === true
        :probable = :Case_definition == "Probable"
        :died = :Outcome == "Dead"
    end
    @rtransform :admit = (!ismissing(:admit) &&
                          (:admit in ADMIT_OUTLIERS || :admit < 0)) ?
                         missing : :admit
    @rtransform begin
        :death = (!ismissing(:death) && :death < 0) ? missing : :death
        :discharge = (!ismissing(:discharge) && :discharge < 0) ?
                     missing : :discharge
        :notif = (!ismissing(:notif) && :notif < 0) ? missing : :notif
    end
    @rtransform begin
        :death = (ismissing(:admit) || !:died) ? missing : :death
        :discharge = (ismissing(:admit) || :died) ? missing : :discharge
    end
end

# Standardise age (mean 0, SD 1), imputing the sample mean for the few missing
# ages, then assemble the per-record event rows.
age_obs = collect(skipmissing(clean.Age))
age_mean = mean(age_obs)
age_sd = std(age_obs)

rows = map(eachrow(clean)) do r
    age_z = ismissing(r.Age) ? 0.0 : (r.Age - age_mean) / age_sd
    (onset = 0.0, admit = r.admit, death = r.death,
        discharge = r.discharge, notif = r.notif,
        hcw = r.is_hcw, probable = r.probable, age_z = age_z)
end

# The four delays are observed on overlapping but different subsets of cases.
n_obs(field) = count(r -> !ismissing(getproperty(r, field)), rows)
(n = length(rows), admit = n_obs(:admit), death = n_obs(:death),
    discharge = n_obs(:discharge), notif = n_obs(:notif))

# ## The delay model
#
# A single composed distribution describes the whole case.
# From onset there are two branches: a chain through admission to resolution,
# and a direct delay to notification.
# Resolution is a competing-outcomes node over death and discharge.
# Each delay is a Gamma, doubly interval censored with a one-day primary window
# and a one-day secondary window.

pwindow = 1.0
swindow = 1.0
function dic(d)
    double_interval_censored(d; primary_event = Uniform(0, pwindow),
        interval = swindow)
end

# The composed template uses bare Gamma delays so its parameters read as shape
# and scale; censoring is applied to the sampled delays during fitting.
# The chain groups the onset-to-admission edge and the admit-anchored
# resolution node; notification is a separate branch off onset.
template = compose((
    name = [:onset_admit, :admit_resolution, :onset_notif],
    dist = [Gamma(1.2, 3.0),
        Competing(:death => (Gamma(2.0, 3.5), 0.5),
            :discharge => (Gamma(1.0, 8.0), 0.5)),
        Gamma(0.7, 20.0)],
    chain = [1, 1, 0]))

# The recursive `show` lays out the branching structure, with the competing
# death and discharge outcomes nested under the admission step.
template

# The event names are the columns a data row supplies.
# A row is matched to them by name, so onset, admit, death, discharge, and
# notif each land in the right slot, and a missing column drives whether that
# delay is conditioned on or marginalised away for that case.
CensoredDistributions.tree_event_names(template)

# ## Priors
#
# `params_table` lists every free parameter of the composed delays as a table.
# Priors are defined against the same names, then turned into a submodel that
# samples the delays and rebuilds the composed distribution.
params_table(template)

priors = (
    onset_admit = (
        onset_admit = (shape = truncated(Normal(1.5, 1.0); lower = 0),
            scale = truncated(Normal(3.0, 2.0); lower = 0)),
        admit_resolution = (
            death = (shape = truncated(Normal(2.0, 1.0); lower = 0),
                scale = truncated(Normal(4.0, 2.0); lower = 0)),
            discharge = (shape = truncated(Normal(1.5, 1.0); lower = 0),
                scale = truncated(Normal(8.0, 4.0); lower = 0)))),
    onset_notif = (shape = truncated(Normal(1.0, 0.7); lower = 0),
        scale = truncated(Normal(15.0, 8.0); lower = 0)))

# Apply day-level double-interval censoring to every delay of a reconstructed
# template, keeping the same branching structure and outcome names.
function censor(d::CensoredDistributions.Sequential)
    CensoredDistributions.Sequential(map(censor, d.components), d.names)
end
function censor(d::CensoredDistributions.Parallel)
    CensoredDistributions.Parallel(map(censor, d.components), d.names)
end
function censor(c::CensoredDistributions.Competing)
    Competing((c.names .=> tuple.(map(censor, c.delays),
        c.branch_probs))...)
end
censor(d::UnivariateDistribution) = dic(d)

# ## The case-fatality ratio
#
# Whether an admitted case dies is modelled as a logistic regression on health
# worker status, a probable (rather than confirmed) case definition, and
# standardised age.
# The resolved probability enters the competing node per case: the node
# conditions on the observed outcome, so the death-versus-discharge split
# carries the case-fatality information without a separate likelihood.

_logistic(z) = 1 / (1 + exp(-z))

# ## Fitting
#
# Each record is scored through one submodel call.
# The competing node self-dispatches on which outcome column is present, the
# per-case death probability is passed in through the reserved `branch_probs`
# field, and unobserved delays for a case are marginalised internally.

@model function bdbv(template, priors, rows)
    delays ~ to_submodel(composed_parameters_model(template, priors))
    censored = censor(delays)

    β0 ~ Normal(0, 2)
    β_hcw ~ Normal(0, 1)
    β_def ~ Normal(0, 1)
    β_age ~ Normal(0, 1)

    for i in eachindex(rows)
        r = rows[i]
        p = _logistic(β0 + β_hcw * r.hcw + β_def * r.probable +
                      β_age * r.age_z)
        record = (onset = r.onset, admit = r.admit, death = r.death,
            discharge = r.discharge, notif = r.notif,
            branch_probs = (death = p, discharge = 1 - p))
        obs ~ to_submodel(
            prefix(composed_distribution_model(censored, record),
                Symbol("rec", i)), false)
    end
end

Random.seed!(20260608)
model = bdbv(template, priors, rows)
chain = sample(model, NUTS(0.8), MCMCThreads(), 1000, 4; progress = false)
nothing #hide

# ## Recovering the re-estimated delays
#
# The comparison targets are the posterior delay means and 95% credible
# intervals from the re-analysis, which corrected the notification delay.
# They are matched posterior to posterior: the package reproduces the corrected
# analysis when the intervals overlap, not only when the point estimates agree.

using FlexiChains: Prefixed
using DynamicPPL: @varname

# Posterior mean delay (shape times scale) of an edge, with a 95% credible
# interval across draws.
function delay_summary(shape_vn, scale_vn)
    shape = vec(chain[Prefixed(shape_vn)])
    scale = vec(chain[Prefixed(scale_vn)])
    means = shape .* scale
    return (mean = mean(means),
        lower = quantile(means, 0.025), upper = quantile(means, 0.975))
end

oa = delay_summary(@varname(delays.onset_admit.onset_admit.shape),
    @varname(delays.onset_admit.onset_admit.scale))
ad = delay_summary(
    @varname(delays.onset_admit.admit_resolution.death.shape),
    @varname(delays.onset_admit.admit_resolution.death.scale))
ac = delay_summary(
    @varname(delays.onset_admit.admit_resolution.discharge.shape),
    @varname(delays.onset_admit.admit_resolution.discharge.scale))
on = delay_summary(@varname(delays.onset_notif.shape),
    @varname(delays.onset_notif.scale))

# Re-estimated targets (posterior mean delay in days with 95% credible
# interval) from the bdbv-linelist-analysis Gamma fit.
targets = (
    onset_admit = (mean = 4.09, lower = 3.08, upper = 5.46),
    admit_death = (mean = 7.71, lower = 5.62, upper = 10.42),
    admit_discharge = (mean = 8.10, lower = 4.81, upper = 13.81),
    onset_notif = (mean = 20.37, lower = 13.64, upper = 29.94))

comparison = DataFrame(
    delay = ["onset to admission", "admission to death",
        "admission to discharge", "onset to notification"],
    n = [n_obs(:admit), n_obs(:death), n_obs(:discharge), n_obs(:notif)],
    posterior_mean = round.(
        [oa.mean, ad.mean, ac.mean, on.mean], digits = 2),
    posterior_lower = round.(
        [oa.lower, ad.lower, ac.lower, on.lower], digits = 2),
    posterior_upper = round.(
        [oa.upper, ad.upper, ac.upper, on.upper], digits = 2),
    target_mean = [targets.onset_admit.mean, targets.admit_death.mean,
        targets.admit_discharge.mean, targets.onset_notif.mean],
    target_lower = [targets.onset_admit.lower, targets.admit_death.lower,
        targets.admit_discharge.lower, targets.onset_notif.lower],
    target_upper = [targets.onset_admit.upper, targets.admit_death.upper,
        targets.admit_discharge.upper, targets.onset_notif.upper])

# Overlay the two interval sets to read the overlap directly.
fig = Figure(size = (760, 380))
ax = Axis(fig[1, 1]; xlabel = "mean delay (days)", yticks = (1:4,
        comparison.delay),
    title = "Posterior delay means vs re-estimated " *
            "targets")
ys = collect(1:4)
for (i, y) in enumerate(ys)
    lines!(ax, [comparison.posterior_lower[i], comparison.posterior_upper[i]],
        [y + 0.12, y + 0.12]; color = :steelblue, linewidth = 6)
    scatter!(ax, [comparison.posterior_mean[i]], [y + 0.12];
        color = :steelblue, markersize = 12)
    lines!(ax, [comparison.target_lower[i], comparison.target_upper[i]],
        [y - 0.12, y - 0.12]; color = :firebrick, linewidth = 6)
    scatter!(ax, [comparison.target_mean[i]], [y - 0.12];
        color = :firebrick, markersize = 12)
end
scatter!(ax, [NaN], [NaN]; color = :steelblue, label = "this fit")
scatter!(ax, [NaN], [NaN]; color = :firebrick, label = "re-estimated target")
axislegend(ax; position = :rb)
fig

# Every delay's credible interval overlaps the re-estimated target.
# The onset-to-admission and admission-to-death delays match closely in both
# mean and width.
# The admission-to-discharge delay sits on a smaller subset of cases here,
# because four resolved cases that carry a discharge date are assigned to death
# by the recorded outcome, so its interval is wider and shifted but still
# overlapping.
# The onset-to-notification delay is the weakest of the four: it is long and
# heavy tailed, and its posterior mean is a little below the target with a
# tighter upper tail, the expected behaviour for a sparse, skewed delay.

comparison

# ## Compound delays
#
# The natural-history delay from onset to death is the convolution of the
# onset-to-admission and admission-to-death delays.
# Its mean and variance follow from the convolution at the posterior mean
# delays, without re-fitting.
oa_post = Gamma(mean(vec(chain[Prefixed(@varname(delays.onset_admit.onset_admit.shape))])),
    mean(vec(chain[Prefixed(@varname(delays.onset_admit.onset_admit.scale))])))
ad_post = Gamma(
    mean(vec(chain[Prefixed(@varname(delays.onset_admit.admit_resolution.death.shape))])),
    mean(vec(chain[Prefixed(@varname(delays.onset_admit.admit_resolution.death.scale))])))
onset_to_death = convolve_distributions(oa_post, ad_post)
(mean = mean(onset_to_death), std = std(onset_to_death))

# ## Simulating from the same distribution
#
# The composed distribution is used in both directions.
# Scoring walks it to read the per-case likelihood; generation walks the same
# object with `rand` to draw a full set of event times for a new case.
# A single simulated record draws an admission delay, a competing resolution
# time with its outcome, and a notification delay.
function simulate_case(d; rng = Random.default_rng())
    chain_branch = get_event(d, :onset_admit)
    admit = rand(rng, get_event(chain_branch, :onset_admit))
    resolution = get_event(chain_branch, :admit_resolution)
    outcomes = event_names(resolution)
    outcome = outcomes[rand(rng, Distributions.Categorical(
        collect(resolution.branch_probs)))]
    resolve = admit + rand(rng, resolution)
    notif = rand(rng, get_event(d, :onset_notif))
    return (onset = 0.0, admit = admit, outcome = outcome,
        resolution = resolve, notif = notif)
end

simulate_case(censor(template); rng = MersenneTwister(20260608))

# ## Notes
#
# The competing-outcomes node carries the case-fatality information through the
# delay likelihood, so the death and discharge delays and the case-fatality
# ratio are fitted together rather than in separate steps.
# The discharge pathway is the least informed, both because discharge is the
# minority outcome among admitted cases and because the mutually exclusive
# assignment moves a few ambiguous cases to death.
# The onset-to-notification delay is long and sparse and should be read with
# its wide interval in mind.
