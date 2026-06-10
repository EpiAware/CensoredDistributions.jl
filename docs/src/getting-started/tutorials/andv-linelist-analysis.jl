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
# A case observed from its source's onset contributes a longer delay: the
# transmission timing from the source's onset to this case's own infection,
# followed by the incubation period from that infection to this case's onset.
# The case's own infection time sits between the two delays and is never
# observed.
# Both readings share the one incubation period.
#
# Two features of the data shape the model.
# Exposure and onset dates are recorded to the day, so every origin is a
# one-day window rather than an instant.
# The outbreak is read in real time, so a case can only appear once its onset
# has happened by the analysis date.
#
# ## A marginal model for both branches
#
# The unobserved infection time of a sourced case is marginalised out.
# Integrating over it makes the sourced delay the CONVOLUTION of the
# transmission timing and the incubation period: the source's onset to the
# case's onset, with the latent infection summed away.
# So both case types are single marginal delays selected by data.
# An `index` case scores the incubation period; a `sourced` case scores the
# convolved source-onset-to-onset delay.
# Each delay is double interval censored, with a one-day primary window (the
# day-resolution origin) and a one-day secondary window (the day-resolution
# onset), so the censoring is part of the delay object.
# A [`select_branch`](@ref) holds the two delays and routes each record to its
# branch by the row's `:kind` field, with the incubation period
# [`shared`](@ref) across both so it is one free parameter, not two.
#
# The convolution of a Normal transmission timing and a LogNormal incubation
# period has no closed form, so the sourced branch's likelihood runs a nested
# numerical integral.
# That makes it slower and less well conditioned than the index branch, which is
# why the transmission timing comes back only weakly identified below.
# A LATENT formulation that samples each infection time instead exists; the
# [composer toolkit tutorial](@ref composer-toolkit) shows how to turn the
# latent path on with [`latent`](@ref).
# This page uses the marginal double-interval-censored form for both branches.

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
using DynamicPPL: to_submodel
using ADTypes: AutoForwardDiff
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
horizon = maximum(onset_day) + 7.0

# Each case becomes up to two records, tagged by `:kind`.
# An `:index` record measures the incubation period from the case's own recorded
# exposure to its onset, with an `obs_time` horizon that right-truncates the
# delay so the real-time observation is respected.
# A `:sourced` record measures onset relative to the source's onset; its own
# infection time is marginalised, so the record is just the observed delay.
# Both kinds live in one table, routed by the `select_branch` selector.
rows = NamedTuple[]
for i in eachindex(pid)
    if !ismissing(exp_lo[i])
        push!(rows, (kind = :index, delay = onset_day[i] - exp_lo[i],
            obs_time = horizon - exp_lo[i]))
    end
    src = ll.source_case[i]
    (ismissing(src) || src == "index") && continue
    s = first(split(string(src), "/"))  # first listed candidate source
    haskey(onset_of, s) || continue
    push!(rows, (kind = :sourced, onset = onset_day[i] - onset_of[s]))
end
(index = count(r -> r.kind == :index, rows),
    sourced = count(r -> r.kind == :sourced, rows))

# ## The delay model
#
# `inc` is the incubation period (a LogNormal), shared by both record types.
# `delta` is the transmission timing (a Normal, since a case can be infected
# shortly before or after its source's onset), used only by the sourced branch.
#
# The index branch is the incubation period, double interval censored.
# The sourced branch is the marginal source-onset-to-onset delay:
# [`convolve_distributions`](@ref) sums the transmission timing and the
# incubation period, then the same double-interval censoring is applied.
# Both branches reuse the one `inc`, which a [`shared`](@ref) tag ties to a
# single free parameter across the `select_branch`.

pwindow = 1.0
swindow = 1.0

function dic(d)
    double_interval_censored(d;
        primary_event = Uniform(0, pwindow), interval = swindow)
end

# Build the two-branch selector for an incubation period and a transmission
# timing. The sourced branch convolves the two delays, marginalising the latent
# infection, and both branches share the one incubation period.
function andv_select(inc, delta)
    inc_s = shared(:inc, inc)
    return select_branch(
        :index => dic(inc_s),
        :sourced => dic(convolve_distributions(delta, inc_s)))
end

# The selector at the upstream posterior means.
andv_select(LogNormal(3.06, 0.32), Normal(0.17, 0.62))

# ## Priors
#
# The four parameters are sampled once inside the model: the incubation
# log-mean and log-SD, and the transmission-timing mean and SD. The priors are
# those of the upstream model.
#
# The model builds the selector from the sampled parameters, then scores the
# whole record table through the vectorised
# [`composed_distribution_model`](@ref): each record selects its branch by
# `:kind` and is scored marginally, with the index branch right-truncated at its
# `obs_time`. The same model both fits observed delays and simulates new ones.

@model function andv(rows)
    mu_inc ~ Normal(3.0, 0.5)
    sigma_inc ~ truncated(Normal(0.0, 0.5); lower = 0)
    mu_delta ~ Normal(0.0, 5.0)
    sigma_delta ~ truncated(Normal(0.0, 1.0); lower = 0)
    inc = LogNormal(mu_inc, sigma_inc)
    delta = Normal(mu_delta, sigma_delta)
    sel = andv_select(inc, delta)
    obs ~ to_submodel(composed_distribution_model(sel, rows))
end

# ## Simulate, fit, and recover
#
# Before touching the data we check the model can recover known parameters.
# [`predict_events`](@ref) draws a censored delay from each branch of the
# selector, so a simulated index record is a draw from the index branch and a
# simulated sourced record is a draw from the convolved sourced branch. The
# simulated delays are then fitted and compared with the truth.

inc_true = LogNormal(3.06, 0.32)
delta_true = Normal(0.17, 0.62)
sel_true = andv_select(inc_true, delta_true)

Random.seed!(20260608)
sim_rows = NamedTuple[]
for _ in 1:60
    push!(sim_rows,
        (kind = :index, delay = predict_events(sel_true; kind = :index),
            obs_time = 200.0))
end
for _ in 1:60
    push!(sim_rows,
        (kind = :sourced, onset = predict_events(sel_true; kind = :sourced)))
end

# The sourced branch runs a nested numerical integral per record, so each
# leapfrog step is more expensive than the latent alternative. We use a short
# warmup, a modest number of draws, and a higher acceptance target to keep the
# geometry stable while the doc build stays quick; a real analysis would use a
# longer run. The model is differentiated with `AutoForwardDiff` (Mooncake is
# the alternative supported backend; Enzyme is not used because it aborts
# uncatchably on the heterogeneous composer-tree recursion).
Random.seed!(20260608)
sim_chain = sample(andv(sim_rows),
    NUTS(150, 0.95; adtype = AutoForwardDiff()),
    MCMCThreads(), 150, 2; progress = false)
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
chain = sample(andv(rows),
    NUTS(150, 0.95; adtype = AutoForwardDiff()),
    MCMCThreads(), 150, 2; progress = false)
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
# The fitted parameters are pushed back onto a composed delay chain with
# [`update`](@ref), so the post-fit delay object comes from the composer rather
# than from hand-built distributions.
# A `Sequential` template of the two delays is updated from the posterior means
# keyed by step name, then [`edge_means`](@ref) reads both per-edge delay means
# off the fitted object in one call.

template = Sequential(Normal(0.0, 1.0), LogNormal(1.0, 0.5))
posted = update(template,
    (step_1 = (mu = mean(vec(chain[:mu_delta])),
            sigma = mean(vec(chain[:sigma_delta]))),
        step_2 = (mu = mean(vec(chain[:mu_inc])),
            sigma = mean(vec(chain[:sigma_inc])))))
fitted_edges = edge_means(posted)

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

# Overlay the two interval sets so the overlap reads directly. The four model
# parameters share a comparable scale, so they go on one axis; the incubation
# mean in days is kept in the table above rather than squashing the axis.
np = 4
cfig = Figure(size = (760, 380))
ax = Axis(cfig[1, 1]; xlabel = "value",
    yticks = (1:np, comparison.quantity[1:np]),
    title = "Posterior vs published targets")
for i in 1:np
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
# next case's infection, but that infection is never observed: it is marginalised
# out of the convolved sourced delay.
# In these data the recorded exposure of a secondary case usually coincides with
# its source's symptom onset (the shared party and wake events), so there is
# little information left to separate the transmission timing from the
# incubation period.
# The result is a `delta` posterior that stays close to its prior, with a mean
# near zero and a credible interval that spans both signs, in agreement with the
# upstream finding that transmission clusters around source onset.
# We report it from multiple chains as a pooled credible interval rather than a
# single point estimate, and read it with its width in mind.
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
