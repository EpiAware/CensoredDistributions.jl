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
# A case observed from its source's onset is a chain of two delays: the
# transmission timing from the source's onset to this case's own infection, then
# the incubation period from that infection to this case's onset.
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
# ## Marginal versus latent
#
# The unobserved infection time of a sourced case can be handled two ways.
# The MARGINAL form integrates it out, so the sourced delay becomes the
# convolution of the transmission timing and the incubation period.
# This is the textbook expression, but the convolution has no closed form for a
# Normal transmission timing and a LogNormal incubation period, so each
# likelihood evaluation runs a nested numerical integral that is slow and poorly
# conditioned at a small sampler budget.
# The LATENT form instead samples the infection time as a per-case parameter and
# scores the two delays directly, with no integral.
# A benchmark on these data found the latent form about five times faster in
# wall-clock and far cheaper per gradient, and it converged at the doc-build
# sampler budget where the marginal form did not.
# This page therefore uses the latent form as the primary fit.
#
# The two record types are routed by a single [`selecting`](@ref)
# disjunction keyed on a `:kind` field: an `:index` case scores its marginal
# incubation leaf, a `:sourced` case scores a latent two-edge chain whose middle
# infection event is sampled. One table carries both kinds, and the model loops
# over its rows letting `select` pick the branch per record, so the
# split-by-kind is data-driven rather than two hand-maintained tables.
# The composer tools used to build the branches are covered in the
# [composer toolkit tutorial](@ref composer-toolkit); here the focus is the
# delay model and the Bayesian workflow.

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
using CensoredDistributions: latent
using Turing, Random, Statistics
using DynamicPPL: prefix, to_submodel
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

# Each case becomes up to two records, tagged by a `:kind` field that a single
# `select` disjunction routes on.
# An `:index` record measures the incubation period from the case's own recorded
# exposure to its onset.
# A `:sourced` record measures onset relative to the source's onset; its own
# infection time is the latent the model samples.
# Both kinds live in one `records` table; the model loops over it and `select`
# picks the matching branch per row from the `:kind` field. A `:sourced` row
# carries the source onset as its origin event (`srconset`, anchored to zero), a
# `missing` `infection` event (the sampled latent), and the observed `onset`.
records = NamedTuple[]
for i in eachindex(pid)
    if !ismissing(exp_lo[i])
        push!(records, (kind = :index, delay = onset_day[i] - exp_lo[i],
            obs_time = horizon - exp_lo[i]))
    end
    src = ll.source_case[i]
    (ismissing(src) || src == "index") && continue
    s = first(split(string(src), "/"))  # first listed candidate source
    haskey(onset_of, s) || continue
    push!(records,
        (kind = :sourced, srconset = 0.0, infection = missing,
            onset = onset_day[i] - onset_of[s]))
end
(index = count(r -> r.kind == :index, records),
    sourced = count(r -> r.kind == :sourced, records))

# ## The delay model
#
# `inc` is the incubation period (a LogNormal), shared by both record types.
# `delta` is the transmission timing (a Normal, since a case can be infected
# shortly before or after its source's onset), used only by the sourced branch.
#
# The index branch is the incubation period observed from the case's own
# infection, doubly interval censored with a one-day primary window (the
# day-resolution exposure) and a one-day secondary window (the day-resolution
# onset).
# The sourced branch is a latent two-edge chain
# `srconset -> infection -> onset`: the transmission timing carries the source's
# onset to the case's own infection, then the incubation period carries that
# infection to the case's onset. The chain is wrapped in `latent`, so the middle
# infection event is sampled and the two delays are scored directly with no
# convolution.
# `selecting` routes a record to the branch its `:kind` names, so both kinds
# share one `inc` and one `delta` and are scored through a single object.

pwindow = 1.0
swindow = 1.0

# Index branch: a single doubly interval censored incubation delay.
function index_branch(inc)
    double_interval_censored(inc;
        primary_event = Uniform(0, pwindow), interval = swindow)
end

# Sourced branch: the latent transmission -> incubation chain, named so its
# events read srconset/infection/onset.
function sourced_branch(inc, delta)
    chain = Sequential(
        (primary_censored(delta, Uniform(0, pwindow)),
            primary_censored(inc, Uniform(0, pwindow))),
        (:srconset_infection, :infection_onset))
    return latent(chain)
end

# The routed delay model: one `select` over the two branches, keyed on `:kind`.
function delay_model(inc, delta)
    selecting(:index => index_branch(inc),
        :sourced => sourced_branch(inc, delta))
end

# The branches at the upstream posterior means.
delay_model(LogNormal(3.06, 0.32), Normal(0.17, 0.62))

# ## Priors
#
# The incubation parameters are a single TIED pair shared by both branches,
# rather than one set per branch, so `inc` and `delta` are sampled once inside
# the model. The priors are those of the upstream model.
#
# The model samples the two delay parameter pairs, builds one routed delay
# object, then loops over the combined `records` table. `select` reads each row's
# `:kind` and delegates to the matching branch: an `:index` row scores its
# marginal incubation leaf, a `:sourced` row scores the latent chain, sampling
# its own infection time and conditioning the observed onset on the shifted
# incubation period. The same definition fits observed delays and, with a missing
# event, samples the latent infection time.

@model function andv(records)
    mu_inc ~ Normal(3.0, 0.5)
    sigma_inc ~ truncated(Normal(0.0, 0.5); lower = 0)
    mu_delta ~ Normal(0.0, 5.0)
    sigma_delta ~ truncated(Normal(0.0, 1.0); lower = 0)
    inc = LogNormal(mu_inc, sigma_inc)
    delta = Normal(mu_delta, sigma_delta)

    ## One routed delay object scored per record; `select` picks the branch.
    d = delay_model(inc, delta)
    obs = Vector{Any}(undef, length(records))
    for (k, r) in enumerate(records)
        sub = prefix(composed_distribution_model(d, r), Symbol("rec_$k"))
        obs[k] ~ to_submodel(sub, false)
    end
    return d
end

# ## Simulate, fit, and recover
#
# Before touching the data we check the model can recover known parameters.
# A full delay path is drawn for each record from the latent form with
# `predict_events`, one latent-applied draw per record, then the simulated
# delays are fitted and compared with the truth.

inc_true = LogNormal(3.06, 0.32)
delta_true = Normal(0.17, 0.62)

Random.seed!(20260608)
sim_records = NamedTuple[]
for _ in 1:60
    path = predict_events(latent(primary_censored(inc_true,
        Uniform(0, pwindow))))
    push!(sim_records, (kind = :index, delay = path[2] - path[1],
        obs_time = 200.0))
end
for _ in 1:60
    seg = predict_events(latent(primary_censored(delta_true,
        Uniform(0, pwindow))))
    infection = seg[2]                       # source onset 0, infection latent
    case_onset = infection + rand(inc_true)  # infection -> case onset
    push!(sim_records, (kind = :sourced, srconset = 0.0, infection = missing,
        onset = case_onset))
end

# The latent likelihood scores the two delays directly, so each leapfrog step is
# cheap. We still use a short warmup, a modest number of draws, and a capped
# NUTS tree depth (`max_depth = 4`) to keep the doc build quick; a real analysis
# would use a longer run. The model is differentiated with `AutoForwardDiff`
# (Mooncake is the alternative supported backend; Enzyme is not used because it
# aborts uncatchably on the heterogeneous composer-tree recursion).
Random.seed!(20260608)
sim_chain = sample(andv(sim_records),
    NUTS(50, 0.9; max_depth = 4, adtype = AutoForwardDiff()),
    MCMCThreads(), 100, 2; progress = false)
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
chain = sample(andv(records),
    NUTS(50, 0.9; max_depth = 4, adtype = AutoForwardDiff()),
    MCMCThreads(), 100, 2; progress = false)
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
# `update`, so the post-fit delay object comes from the composer rather than
# from hand-built distributions.
# A `Sequential` template of the two delays is updated from the posterior means
# keyed by step name, then the per-event [`mean`](@ref)`(latent(posted))` Vector
# reads both delay means off the fitted object in one call.

template = Sequential(Normal(0.0, 1.0), LogNormal(1.0, 0.5))
posted = update(template,
    (step_1 = (mu = mean(vec(chain[:mu_delta])),
            sigma = mean(vec(chain[:sigma_delta]))),
        step_2 = (mu = mean(vec(chain[:mu_inc])),
            sigma = mean(vec(chain[:sigma_inc])))))
fitted_edges = mean(latent(posted))

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
# next case's infection, but that infection is never observed: it is the
# per-case latent the model samples.
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
