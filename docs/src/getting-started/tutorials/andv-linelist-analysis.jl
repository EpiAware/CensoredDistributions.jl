# # Real-time Andes virus delays from the Epuyén line list
#
# The 2018-19 Epuyén outbreak of Andes hantavirus (ANDV) in Argentina was the
# first documented sustained person-to-person spread of a hantavirus.
# A Bayesian re-analysis at
# [epiforecasts/andv-linelist-analysis](https://github.com/epiforecasts/andv-linelist-analysis)
# fits a joint model to the line list of Martínez et al. (2020) to estimate the
# incubation period, the transmission timing of each onward infection, and a
# time-varying reproduction number.
# This page fits the two delay distributions of that model with
# CensoredDistributions.jl and checks that the published estimates are
# recovered.
#
# Two delays describe how a case reaches symptom onset.
# The incubation period runs from a case's own infection to its symptom onset.
# It is the same biological process for every case, so it is SHARED across the
# whole outbreak.
# The transmission timing runs from a source's symptom onset to the moment that
# source infects the next case.
# It only exists for a case that has a human source, so it is estimated from the
# SOURCED cases alone.
#
# A case enters the model in one of two ways.
# A case observed from its own infection contributes its incubation period
# directly.
# A case observed from its source's onset contributes the SUM of the
# transmission timing and its own incubation period, because the case's own
# infection time is not observed and is integrated out.
# Both readings share the one incubation period.
#
# Two features of the data shape the model.
# Exposure and onset dates are recorded to the day, so every delay is doubly
# interval censored: the origin day and the onset day are both windows rather
# than instants.
# The outbreak is read in real time, so a case can only appear once its onset
# has happened by the analysis date; each delay is right-truncated at the time
# left between its origin and that date.
#
# The composed object below is built with the package's composer tools. The
# mechanics of `select`, `Sequential`, and the per-record scoring are covered in
# the [getting-started composer tutorial](@ref getting-started); here the focus
# is the delay model and the Bayesian workflow.

# ## Data
#
# The bundled line list is the Epuyén subset (n = 33 cases used here) encoded
# from Table S2 of Martínez et al. (2020), redistributed from the re-analysis
# repository under the MIT licence (see the data folder's `README`).
# We read the dates, express each delay as a day offset from its origin, and
# drop the two alternative-source sensitivity rows that the upstream main fit
# excludes.

using CSV, DataFramesMeta, Dates
using CensoredDistributions, Distributions
using Turing, Random, Statistics
using CairoMakie, PairPlots

datadir = joinpath(@__DIR__, "andv-data")
ll = CSV.read(joinpath(datadir, "linelist.csv"), DataFrame;
    missingstring = ["NA", ""])

# Keep the main-fit records (the `_alt` rows are sensitivity alternatives).
ll = @rsubset ll !endswith(string(:patient_id), "_alt")

# Day offsets from the earliest onset, and a lookup from a case id to its onset
# day so a sourced case can be anchored to its source's onset.
onset = Date.(string.(ll.onset_date))
t0 = minimum(onset)
day(d) = Float64(Dates.value(d - t0))
onset_day = day.(onset)
pid = string.(ll.patient_id)
onset_of = Dict(pid[i] => onset_day[i] for i in eachindex(pid))

# Exposure window (the recorded infection window) in day offsets.
parse_day(x) = ismissing(x) ? missing : day(Date(string(x)))
exp_lo = parse_day.(ll.exposure_lower)

# Real-time horizon: the analysis date, one week after the last recorded onset.
# Each record is right-truncated at the time from its origin to this horizon.
horizon = maximum(onset_day) + 7.0

# Each case becomes up to two records keyed by `:kind`.
# An `:index` record measures the incubation period from the case's own recorded
# exposure to its onset.
# A `:sourced` record measures onset relative to the source's onset, the
# transmission-timing-plus-incubation chain.
# Each record carries its delay, its `:obs_time` horizon, and the `:kind` the
# `select` routes on.
index_rows = NamedTuple[]
sourced_rows = NamedTuple[]
for i in eachindex(pid)
    if !ismissing(exp_lo[i])
        push!(index_rows, (kind = :index, delay = onset_day[i] - exp_lo[i],
            obs_time = horizon - exp_lo[i]))
    end
    src = ll.source_case[i]
    (ismissing(src) || src == "index") && continue
    s = first(split(string(src), "/"))  # first listed candidate source
    haskey(onset_of, s) || continue
    push!(sourced_rows,
        (kind = :sourced, delay = onset_day[i] - onset_of[s],
            obs_time = horizon - onset_of[s]))
end
rows = vcat(index_rows, sourced_rows)
(index = length(index_rows), sourced = length(sourced_rows))

# ## The delay model
#
# A single composed distribution describes a case from either origin.
# `inc` is the incubation period (a LogNormal), shared by both record types.
# `delta` is the transmission timing (a Normal, since a case can be infected
# shortly before or after its source's onset), used only by the sourced branch.
#
# The `:index` branch is the incubation period observed from the case's own
# infection.
# The `:sourced` branch is a `Sequential` chain of the transmission timing then
# the incubation period; the case's own infection sits between them and is not
# observed, so wrapping the whole chain in `double_interval_censored` integrates
# it out (the chain convolves).
# Each branch is doubly interval censored with a one-day primary window (the
# day-resolution origin) and a one-day secondary window (the day-resolution
# onset).
# `select` routes a record to its branch by the record's `:kind`.

pwindow = 1.0
swindow = 1.0

function delay_model(inc, delta)
    index_branch = double_interval_censored(inc;
        primary_event = Uniform(0, pwindow), interval = swindow)
    sourced_branch = double_interval_censored(Sequential(delta, inc);
        primary_event = Uniform(0, pwindow), interval = swindow)
    ## `select` is qualified because DataFramesMeta also exports the name.
    return CensoredDistributions.select(:index => index_branch,
        :sourced => sourced_branch; selector = :kind)
end

# The two branches at the upstream posterior means, shown as one selected
# disjunction.
delay_model(LogNormal(3.06, 0.32), Normal(0.17, 0.62))

# ## Priors
#
# The incubation period is shared between the two branches, so its parameters
# are a single TIED pair rather than one set per branch.
# We therefore sample `inc` and `delta` DIRECTLY from their priors inside the
# model and build the `select` from them, rather than asking
# `composed_parameters_model` to build independent priors per branch (which
# would duplicate the shared incubation period).
# The priors are those of the upstream model.
#
# `params_table` is still useful as an inventory of the free delay parameters.
# The sourced chain holds both delays, so listing it shows every free scalar:
# the transmission timing (step 1) and the incubation period (step 2). The index
# branch reuses the same incubation step, so there are four free parameters in
# all, not six.
params_table(Sequential(Normal(0.17, 0.62), LogNormal(3.06, 0.32)))

# The model samples the two delay parameter pairs, builds the `select`, and
# scores the whole table of records in one vectorised submodel.
# A record's `:obs_time` right-truncates its branch at the real-time horizon,
# and the `select` picks each record's branch by its `:kind`.
# The same submodel both scores observed delays and, with a missing-delay table,
# samples full delays, so it fits and generates from one definition.

@model function andv(rows)
    mu_inc ~ Normal(3.0, 0.5)
    sigma_inc ~ truncated(Normal(0.0, 0.5); lower = 0)
    mu_delta ~ Normal(0.0, 5.0)
    sigma_delta ~ truncated(Normal(0.0, 1.0); lower = 0)
    inc = LogNormal(mu_inc, sigma_inc)
    delta = Normal(mu_delta, sigma_delta)
    d = delay_model(inc, delta)
    obs ~ to_submodel(composed_distribution_model(d, rows))
    return d
end

# ## Simulate, fit, and recover
#
# Before touching the data we check the model can recover known parameters.
# A full delay path is drawn for each record straight from the composed object
# with `predict_events`, with no manual reconstruction, then the simulated
# delays are fitted and compared with the truth.

inc_true = LogNormal(3.06, 0.32)
delta_true = Normal(0.17, 0.62)
truth = delay_model(inc_true, delta_true)

Random.seed!(20260608)
sim_rows = NamedTuple[]
for _ in 1:40
    y = predict_events(CensoredDistributions._pick(truth, :index))
    push!(sim_rows, (kind = :index, delay = y, obs_time = 200.0))
end
for _ in 1:40
    y = predict_events(CensoredDistributions._pick(truth, :sourced))
    push!(sim_rows, (kind = :sourced, delay = y, obs_time = 200.0))
end

Random.seed!(20260608)
sim_chain = sample(andv(sim_rows), NUTS(0.95), MCMCThreads(), 300, 2;
    progress = false)
nothing #hide

# Posterior means against the simulating truth. The incubation period is
# recovered closely; the transmission timing is recovered in the right region
# but with a wide interval, the first sign of its weak identifiability.
sim_summary = DataFrame(
    parameter = ["mu_inc", "sigma_inc", "mu_delta", "sigma_delta"],
    truth = [3.06, 0.32, 0.17, 0.62],
    posterior_mean = [round(mean(vec(sim_chain[p])); digits = 3)
                      for p in (:mu_inc, :sigma_inc, :mu_delta, :sigma_delta)])

# ## Fitting the line list
#
# The same model is fitted to the real records.

Random.seed!(20260608)
chain = sample(andv(rows), NUTS(0.95), MCMCThreads(), 400, 4; progress = false)
nothing #hide

# ## Priors and posteriors
#
# `PairPlots.jl` shows the prior and the posterior of the four parameters on one
# figure. The incubation parameters concentrate tightly; the transmission-timing
# parameters barely move from their priors, which is the identifiability story
# made visible.
# The prior draws come straight from the four parameter priors declared in the
# model, so the overlay needs no extra fit.

Random.seed!(20260608)
priors = (mu_inc = Normal(3.0, 0.5),
    sigma_inc = truncated(Normal(0.0, 0.5); lower = 0),
    mu_delta = Normal(0.0, 5.0),
    sigma_delta = truncated(Normal(0.0, 1.0); lower = 0))
prior_draws = (; (p => rand(priors[p], 2000) for p in keys(priors))...)

pairnames = (:mu_inc, :sigma_inc, :mu_delta, :sigma_delta)
post_table = (; (p => vec(chain[p]) for p in pairnames)...)

fig = pairplot(
    PairPlots.Series(prior_draws, label = "prior", color = (:grey, 0.5)) =>
        (PairPlots.Scatter(markersize = 3), PairPlots.MarginHist()),
    PairPlots.Series(post_table, label = "posterior",
        color = (:steelblue, 0.6)) =>
        (PairPlots.Scatter(markersize = 3), PairPlots.MarginHist()))
fig

# ## Applying the posterior to the composed object
#
# The fitted parameters are pushed back onto the composed delay chain with
# `update`, so the post-fit delay object comes from the composer rather than
# from hand-built distributions.
# A `Sequential` template of the two delays is updated from the posterior means
# keyed by step name, returning the fitted transmission timing and incubation
# period as a single composed object.

template = Sequential(Normal(0.0, 1.0), LogNormal(1.0, 0.5))
posted = update(template,
    (step_1 = (mu = mean(vec(chain[:mu_delta])),
            sigma = mean(vec(chain[:sigma_delta]))),
        step_2 = (mu = mean(vec(chain[:mu_inc])),
            sigma = mean(vec(chain[:sigma_inc])))))
fitted_inc = get_event(posted, :step_2)

# Posterior mean incubation period (in days) and its 95% credible interval,
# read from the per-draw LogNormal means.
function post_inc(c, f)
    [f(LogNormal(c[:mu_inc][i], c[:sigma_inc][i]))
     for i in 1:length(c[:mu_inc])]
end
inc_means = post_inc(chain, mean)
inc_summary = (mean = mean(inc_means),
    lower = quantile(inc_means, 0.025), upper = quantile(inc_means, 0.975))

# ## Equivalence with the published estimates
#
# The comparison targets are the posterior means and 95% credible intervals from
# the re-analysis, read from its published posterior draws.
# They are matched posterior to posterior: the package reproduces the published
# analysis when the intervals overlap, not only when the point estimates agree.

function summ(c, p)
    v = vec(c[p])
    return (mean = mean(v), lower = quantile(v, 0.025),
        upper = quantile(v, 0.975))
end

## Published targets (posterior mean and 95% credible interval) from the
## andv-linelist-analysis joint fit.
targets = (
    mu_inc = (mean = 3.06, lower = 2.95, upper = 3.18),
    sigma_inc = (mean = 0.32, lower = 0.25, upper = 0.41),
    mu_delta = (mean = 0.17, lower = -0.16, upper = 0.49),
    sigma_delta = (mean = 0.62, lower = 0.45, upper = 0.84),
    inc_mean = (mean = 22.6, lower = 20.3, upper = 25.5))

here = (mu_inc = summ(chain, :mu_inc), sigma_inc = summ(chain, :sigma_inc),
    mu_delta = summ(chain, :mu_delta),
    sigma_delta = summ(chain, :sigma_delta), inc_mean = inc_summary)

labels = ["incubation log-mean", "incubation log-SD",
    "transmission-timing mean", "transmission-timing SD",
    "incubation mean (days)"]
keys_ = (:mu_inc, :sigma_inc, :mu_delta, :sigma_delta, :inc_mean)
comparison = DataFrame(
    quantity = labels,
    posterior_mean = round.([here[k].mean for k in keys_], digits = 2),
    posterior_lower = round.([here[k].lower for k in keys_], digits = 2),
    posterior_upper = round.([here[k].upper for k in keys_], digits = 2),
    target_mean = [targets[k].mean for k in keys_],
    target_lower = [targets[k].lower for k in keys_],
    target_upper = [targets[k].upper for k in keys_])

# Overlay the two interval sets so the overlap reads directly.
cfig = Figure(size = (760, 420))
ax = Axis(cfig[1, 1]; xlabel = "value",
    yticks = (1:5, comparison.quantity),
    title = "Posterior vs published targets")
for i in 1:5
    lines!(ax, [comparison.posterior_lower[i], comparison.posterior_upper[i]],
        [i + 0.12, i + 0.12]; color = :steelblue, linewidth = 6)
    scatter!(ax, [comparison.posterior_mean[i]], [i + 0.12];
        color = :steelblue, markersize = 12)
    lines!(ax, [comparison.target_lower[i], comparison.target_upper[i]],
        [i - 0.12, i - 0.12]; color = :firebrick, linewidth = 6)
    scatter!(ax, [comparison.target_mean[i]], [i - 0.12];
        color = :firebrick, markersize = 12)
end
scatter!(ax, [NaN], [NaN]; color = :steelblue, label = "this fit")
scatter!(ax, [NaN], [NaN]; color = :firebrick, label = "published target")
axislegend(ax; position = :rb)
cfig

# The incubation parameters and the incubation mean delay overlap the published
# targets closely in both location and width.
# The transmission-timing parameters overlap too, but their intervals are wide.
comparison

# ## The transmission timing is weakly identified
#
# The transmission timing `delta` is the gap between a source's onset and the
# next case's infection, but that infection is never observed: it is integrated
# out of the sourced branch.
# In these data the recorded exposure of a secondary case usually coincides with
# its source's symptom onset (the shared party and wake events), so there is
# little information left to separate the transmission timing from the
# incubation period.
# The result is a `delta` posterior that stays close to its prior, with a mean
# near zero and a credible interval that spans both signs, in agreement with the
# upstream finding that transmission clusters around source onset.
# We report it from several chains with the pooled credible interval rather than
# a single point estimate, and read it with its width in mind.
delta_diag = DataFrame(
    parameter = ["mu_delta", "sigma_delta"],
    posterior_mean = round.([here.mu_delta.mean, here.sigma_delta.mean],
        digits = 3),
    pooled_lower = round.([here.mu_delta.lower, here.sigma_delta.lower],
        digits = 3),
    pooled_upper = round.([here.mu_delta.upper, here.sigma_delta.upper],
        digits = 3))

# ## Scope
#
# The upstream model also estimates a time-varying reproduction number and the
# offspring (cluster) dispersion from the same line list; those parts are out of
# scope here, which fits only the two delay distributions.
