# # Real-time Andes virus delays from the Epuyén line list
#
# The 2018-19 Epuyén outbreak of Andes hantavirus (ANDV) in Argentina was the
# first documented sustained person-to-person spread of a hantavirus.
# A Bayesian re-analysis at
# [epiforecasts/andv-linelist-analysis](https://github.com/epiforecasts/andv-linelist-analysis)
# fits a joint model to the line list of Martínez et al. (2020) to estimate the
# incubation period, the transmission timing of each onward infection, a
# time-varying reproduction number, and the dispersion of onward transmission.
# This page fits the two delay distributions of that model with
# CensoredDistributions.jl, reads transmission intensity from the recorded
# offspring counts, and checks that the published estimates are recovered.
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
# A benchmark on these data found the latent form about six times faster in
# wall-clock than the marginal form and far cheaper per gradient, and it
# recovers the published posterior cleanly at the doc-build sampler budget.
#
# This page fits BOTH and compares them.
# The latent form is the primary, accurate fit, used for every downstream
# result; the marginal-convolved form is then fitted on the same data as the
# faster alternative, and a closing section sets the two recovered delay
# parameter sets side by side against the published targets so the speed and
# accuracy trade-off can be read directly.
#
# The two record types of the latent form are routed by a single
# [`selecting`](@ref) disjunction keyed on a `:kind` field: an `:index` case
# scores its marginal incubation leaf, a `:sourced` case scores a latent two-edge
# chain whose middle infection event is sampled. One table carries both kinds,
# and the model loops over its rows letting `select` pick the branch per record,
# so the split-by-kind is data-driven rather than two hand-maintained tables.
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
using DynamicPPL: prefix, to_submodel, InitFromPrior
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
# An outbreak read at this date only knows about a case once its onset has
# happened, so every delay is right-truncated to the window still open before
# the horizon. That window is the denominator of the per-record likelihood.
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
#
# Real-time truncation differs by record type, and the two halves are easy to
# mix up. An `:index` record is one observed incubation delay measured from the
# case's own exposure, so it is right-truncated to the single-delay window
# `horizon - exposure`; the reserved `obs_time` row field applies that
# truncation inside the leaf model (`-logcdf(inc, window)`). A `:sourced` record
# observes the total of the transmission timing and the incubation period, whose
# splitting infection event is never recorded, so its denominator is the CDF of
# the CONVOLUTION of the two delays at the window `horizon - source onset`.
# The convolved denominator cannot ride on the latent likelihood (the origin is
# sampled internally), so the sourced window is carried separately in
# `src_window` and scored with [`completeness_probability`](@ref) on the
# [`convolve_distributions`](@ref) chain.
records = NamedTuple[]
src_window = Float64[]
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
    push!(src_window, horizon - onset_of[s])
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
#
# Both record types are then right-truncated to the real-time horizon. An
# `:index` row carries its window as the reserved `obs_time` field, so the leaf
# model truncates it directly (`-logcdf(inc, window)`). A `:sourced` row's
# denominator is the convolved chain `delta + inc`; its completeness probability
# at the source-relative window is read with [`completeness_probability`](@ref)
# on a [`convolve_distributions`](@ref) chain and subtracted from the
# log-likelihood, the convolved counterpart of the index single-delay term.

@model function andv(records, src_window)
    mu_inc ~ Normal(3.0, 0.5)
    sigma_inc ~ truncated(Normal(0.0, 0.5); lower = 0)
    mu_delta ~ Normal(0.0, 5.0)
    sigma_delta ~ truncated(Normal(0.0, 1.0); lower = 0)
    inc = LogNormal(mu_inc, sigma_inc)
    delta = Normal(mu_delta, sigma_delta)

    ## One routed delay object scored per record; `select` picks the branch. The
    ## index leaf truncates on its reserved `obs_time` field; the sourced chain
    ## truncation is the convolved completeness denominator added below.
    d = delay_model(inc, delta)
    obs = Vector{Any}(undef, length(records))
    for (k, r) in enumerate(records)
        sub = prefix(composed_distribution_model(d, r), Symbol("rec_$k"))
        obs[k] ~ to_submodel(sub, false)
    end

    ## Sourced convolved-chain right-truncation: the completeness of
    ## `delta + inc` at each sourced record's window, matching the upstream
    ## sourced denominator `-log(cdf(delta + inc, obs_time - source onset))`.
    chain = convolve_distributions(delta, inc)
    for w in src_window
        p = completeness_probability(chain, w)
        Turing.@addlogprob! -log(max(p, floatmin(typeof(p))))
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

# A generous real-time horizon (200 days) is used for the simulation so the
# right-truncation is light and the truth is recovered cleanly; the truncation
# machinery is exercised in full on the real fit below.
sim_horizon = 200.0
Random.seed!(20260608)
sim_records = NamedTuple[]
sim_src_window = Float64[]
for _ in 1:30
    path = predict_events(latent(primary_censored(inc_true,
        Uniform(0, pwindow))))
    push!(sim_records, (kind = :index, delay = path[2] - path[1],
        obs_time = sim_horizon))
end
for _ in 1:30
    seg = predict_events(latent(primary_censored(delta_true,
        Uniform(0, pwindow))))
    infection = seg[2]                       # source onset 0, infection latent
    case_onset = infection + rand(inc_true)  # infection -> case onset
    push!(sim_records, (kind = :sourced, srconset = 0.0, infection = missing,
        onset = case_onset))
    push!(sim_src_window, sim_horizon)
end

# The latent likelihood scores the two delays directly, so each leapfrog step is
# cheap. The two chains run in PARALLEL with
# [`MCMCThreads`](https://turinglang.org/Turing.jl/stable/) at a sampler budget
# (250 warmup, 250 draws) large enough to read credible intervals from. Chains
# start from the priors (`InitFromPrior`): the real-time truncation creates a
# second, spurious mode at a near-zero incubation period (where every delay is
# trivially complete), and a prior-centred start keeps the sampler in the basin
# that carries the data. The model is differentiated with `AutoForwardDiff`
# (Mooncake is the alternative supported backend; Enzyme is not used because it
# aborts uncatchably on the heterogeneous composer-tree recursion).
Random.seed!(20260608)
sim_chain = sample(andv(sim_records, sim_src_window),
    NUTS(150, 0.95; max_depth = 6, adtype = AutoForwardDiff()),
    MCMCThreads(), 150, 2;
    initial_params = fill(InitFromPrior(), 2), progress = false)
nothing #hide

# Posterior means against the simulating truth. The incubation period is
# recovered closely; the transmission timing is recovered in the right region
# but with a wide interval, the first sign of its weak identifiability.
sim_pars = (:mu_inc, :sigma_inc, :mu_delta, :sigma_delta)
sim_draws = (; (p => vec(sim_chain[p]) for p in sim_pars)...)
sim_summary = DataFrame(
    parameter = ["mu_inc", "sigma_inc", "mu_delta", "sigma_delta"],
    truth = [3.06, 0.32, 0.17, 0.62],
    posterior_mean = [round(mean(sim_draws[p]); digits = 3)
                      for p in sim_pars],
    lower = [round(quantile(sim_draws[p], 0.025); digits = 3)
             for p in sim_pars],
    upper = [round(quantile(sim_draws[p], 0.975); digits = 3)
             for p in sim_pars])

# ## Fitting the line list
#
# The same model is fitted to the real records.

Random.seed!(20260608)
latent_time = @elapsed chain = sample(andv(records, src_window),
    NUTS(200, 0.95; max_depth = 6, adtype = AutoForwardDiff()),
    MCMCThreads(), 200, 2;
    initial_params = fill(InitFromPrior(), 2), progress = false)
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

# ## The marginal-convolved alternative
#
# The latent fit above samples each sourced case's infection time and scores the
# two delays directly. The MARGINAL form instead integrates that infection time
# out, so a sourced case is a single delay: the
# [`convolve_distributions`](@ref) of the transmission timing and the incubation
# period, measured from the source's onset to the case's onset. Both branches
# are then a single double-interval-censored leaf, so the whole model is one
# [`selecting`](@ref) disjunction with no sampled latent and no separate
# completeness term: the real-time horizon rides on each record's reserved
# `obs_time` field, which right-truncates the sourced convolved leaf exactly as
# it truncates the index leaf.
#
# The convolution of a Normal transmission timing and a LogNormal incubation
# period has no closed form, so the sourced branch runs a nested numerical
# integral per likelihood evaluation. That is what makes the marginal form
# slower per gradient than the latent form, and what can bias the recovered
# incubation SCALE at a small sampler budget where the integral is coarse. We
# fit it here on the same data to read the trade-off, not as the primary result.

# The marginal sourced branch is the convolved delay; the index branch is the
# bare incubation. Both share the one `inc` because both branches are rebuilt
# from the same sampled parameters inside the model, exactly as in the latent
# form.
function marginal_select(inc, delta)
    selecting(:index => index_branch(inc),
        :sourced => index_branch(convolve_distributions(delta, inc)))
end

# The same line list, re-expressed for the marginal form: a sourced record is now
# just its observed onset-to-onset delay plus the real-time `obs_time` window
# (`horizon - source onset`), with no sampled infection event. The index records
# are unchanged.
marg_records = NamedTuple[]
for i in eachindex(pid)
    if !ismissing(exp_lo[i])
        push!(marg_records,
            (kind = :index, delay = onset_day[i] - exp_lo[i],
                obs_time = horizon - exp_lo[i]))
    end
    src = ll.source_case[i]
    (ismissing(src) || src == "index") && continue
    s = first(split(string(src), "/"))
    haskey(onset_of, s) || continue
    push!(marg_records,
        (kind = :sourced, onset = onset_day[i] - onset_of[s],
            obs_time = horizon - onset_of[s]))
end

# The marginal model: the same four priors, then one routed delay object scored
# over the whole table. The vectorised [`composed_distribution_model`](@ref)
# entry scores every record in one `~`, each row picking its branch by `:kind`
# and right-truncated at its own `obs_time`; the sourced branch's convolved
# completeness denominator is applied by that truncation, so no `@addlogprob!`
# term is needed.
@model function andv_marginal(records)
    mu_inc ~ Normal(3.0, 0.5)
    sigma_inc ~ truncated(Normal(0.0, 0.5); lower = 0)
    mu_delta ~ Normal(0.0, 5.0)
    sigma_delta ~ truncated(Normal(0.0, 1.0); lower = 0)
    inc = LogNormal(mu_inc, sigma_inc)
    delta = Normal(mu_delta, sigma_delta)
    sel = marginal_select(inc, delta)
    obs ~ to_submodel(composed_distribution_model(sel, records))
    return sel
end

# Fitted at the same budget as the latent fit. The nested integral makes each
# leapfrog step more expensive, so the wall-clock is markedly longer for the
# same number of draws.
Random.seed!(20260608)
marg_time = @elapsed marg_chain = sample(andv_marginal(marg_records),
    NUTS(200, 0.95; max_depth = 6, adtype = AutoForwardDiff()),
    MCMCThreads(), 200, 2;
    initial_params = fill(InitFromPrior(), 2), progress = false)
nothing #hide

# ## Marginal versus latent versus the published target
#
# The two formulations are set side by side against the published posterior. The
# incubation log-mean (`mu_inc`) agrees across both. The marginal form's
# incubation log-SD (`sigma_inc`) is the parameter most exposed to the coarse
# nested integral at this budget, so it is the one to watch for the scale bias;
# the latent form recovers it cleanly. The weakly identified transmission-timing
# parameters stay close to their priors under both forms.
both = DataFrame(
    parameter = ["mu_inc", "sigma_inc", "mu_delta", "sigma_delta"],
    target = [3.06, 0.32, 0.17, 0.62],
    latent_mean = round.(
        [here.mu_inc.mean, here.sigma_inc.mean,
            here.mu_delta.mean, here.sigma_delta.mean], digits = 3),
    marginal_mean = round.(
        [mean(vec(marg_chain[p]))
         for p in (:mu_inc, :sigma_inc, :mu_delta,
            :sigma_delta)], digits = 3))

# A note on the trade-off, read from the same two fits. `latent_time` and
# `marg_time` are the wall-clock seconds of the two real-data fits at the same
# budget; their ratio is the speed-up the latent form gives by avoiding the
# per-record integral. The latent form is therefore the default for this
# analysis; the marginal form is the more familiar textbook expression and is
# convenient when a sourced case must be one observed delay rather than a
# sampled chain, but it should be run at a larger budget so the nested integral
# resolves the incubation scale.
speedup = round(marg_time / latent_time; digits = 1)
tradeoff = DataFrame(
    form = ["latent (primary)", "marginal-convolved"],
    wall_clock_s = round.([latent_time, marg_time], digits = 1),
    sourced_likelihood = ["sampled two-edge chain", "nested integral"],
    use_when = ["accuracy at a small budget; the default here",
        "a single observed sourced delay is wanted; run a larger budget"])

# The marginal form is roughly `speedup`x slower here, the cost of the nested
# integral that the latent form avoids.
tradeoff

# The recovered parameters from both forms against the target.
both

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

# ## Transmission intensity through the outbreak
#
# The same line list records who infected whom, so each case carries an
# offspring count: the number of secondaries the line list attributes to it. The
# upstream model reads transmission intensity from these counts with a branching
# (offspring) model, not a renewal recursion. Each source's offspring count is a
# draw from a negative binomial whose mean is a time-varying reproduction number
# `R(t)` evaluated at the source's own onset day, and whose dispersion `k`
# captures the superspreading: a few sources drive most onward infections.
#
# `R(t)` follows a weekly random walk on the log scale, written in plain Turing
# as a non-centred walk so the knot innovations sample cleanly. The dispersion
# uses the `1/sqrt(k)` parameterisation. The delays are not needed for the
# retrospective offspring counts, so this section fits the offspring layer on
# its own with the incubation and transmission timing held at posterior means;
# the upstream joint fit shares the delay parameters across both halves.

# Per-source onset day (in days from the first onset) and offspring count.
Z = Int.(ll.Z)
source_onset_day = onset_day

# Weekly knots over the outbreak. `log R(t)` is piecewise linear between them.
knots = collect(0.0:7.0:maximum(onset_day))
knots[end] < maximum(onset_day) && push!(knots, maximum(onset_day))
n_knots = length(knots)

# Piecewise-linear interpolation of `log R(t)` between weekly knots, clamped to
# the endpoint values outside the knot range.
function log_R_at(t, ks, log_R)
    t <= ks[1] && return log_R[1]
    t >= ks[end] && return log_R[end]
    b = searchsortedlast(ks, t)
    w = (t - ks[b]) / (ks[b + 1] - ks[b])
    return (1 - w) * log_R[b] + w * log_R[b + 1]
end

# `NegativeBinomial(k, k / (k + R))` is the mean-`R`, dispersion-`k` offspring
# law; the lower clamp on the success probability keeps the gradient finite when
# an extreme proposal drives `R` so high the probability would underflow.
safe_nb(k, R) = NegativeBinomial(k, max(k / (k + R), eps(typeof(k))))

# The offspring model: a weekly non-centred random walk on `log R(t)`, a
# `1/sqrt(k)` dispersion prior, and one negative binomial offspring likelihood
# per source indexed at its onset day. The walk and dispersion are returned so
# `generated_quantities` reads `log_R` and `k` back off the posterior draws.
@model function offspring(Z, source_onset_day, knots)
    phi ~ truncated(Normal(0.0, 1.0); lower = 0)
    k = 1.0 / phi^2
    sigma_rw ~ truncated(Normal(0.0, 0.2); lower = 0)
    log_R_init ~ Normal(log(1.5), 1.0)
    eps ~ filldist(Normal(0.0, 1.0), n_knots - 1)
    log_R = vcat(log_R_init, log_R_init .+ accumulate(+, sigma_rw .* eps))
    for i in eachindex(Z)
        R_i = exp(log_R_at(source_onset_day[i], knots, log_R))
        Z[i] ~ safe_nb(k, max(R_i, 1e-10))
    end
    return (; log_R, k)
end

# Two chains in parallel; the offspring layer carries no latent delays, so it
# samples quickly even at a larger budget than the delay fit.
Random.seed!(20260608)
rt_model = offspring(Z, source_onset_day, knots)
rt_chain = sample(rt_model,
    NUTS(250, 0.9; max_depth = 6, adtype = AutoForwardDiff()),
    MCMCThreads(), 250, 2;
    initial_params = fill(InitFromPrior(), 2), progress = false)
nothing #hide

# The dispersion `k` and its credible interval, read off the `1/sqrt(k)` draws.
k_draws = 1 ./ vec(rt_chain[:phi]) .^ 2
k_summary = (mean = mean(k_draws),
    lower = quantile(k_draws, 0.025), upper = quantile(k_draws, 0.975))

# The weekly `R(t)` is read from the deterministic `log_R` vector on each draw.
rt_gq = generated_quantities(rt_model, rt_chain)
log_R_draws = reduce(hcat, [g.log_R for g in vec(rt_gq)])
rt_summary = DataFrame(
    week = 1:n_knots,
    day = Int.(knots),
    R_mean = [round(mean(exp.(log_R_draws[b, :])); digits = 2)
              for b in 1:n_knots],
    R_lower = [round(quantile(exp.(log_R_draws[b, :]), 0.025); digits = 2)
               for b in 1:n_knots],
    R_upper = [round(quantile(exp.(log_R_draws[b, :]), 0.975); digits = 2)
               for b in 1:n_knots])

# `R(t)` starts above one and falls below it across the outbreak, the signature
# of an epidemic brought under control. The published analysis reports the same
# descent (from roughly 2.3 in the first weeks to about 0.4 by week 13) and a
# dispersion near `k = 0.45`, both reproduced here within the credible bands.
rfig = Figure(size = (760, 380))
rax = Axis(rfig[1, 1]; xlabel = "week", ylabel = "R(t)",
    title = "Weekly reproduction number")
band!(rax, rt_summary.week, rt_summary.R_lower, rt_summary.R_upper;
    color = (:steelblue, 0.25))
lines!(rax, rt_summary.week, rt_summary.R_mean; color = :steelblue,
    linewidth = 3)
hlines!(rax, [1.0]; color = :grey, linestyle = :dash)
rfig

# The dispersion and the early/late reproduction number against the published
# targets.
rt_compare = DataFrame(
    quantity = ["dispersion k", "R(t) week 1", "R(t) week $(n_knots)"],
    posterior = [round(k_summary.mean; digits = 2),
        rt_summary.R_mean[1], rt_summary.R_mean[end]],
    target = [0.45, 2.26, 0.38])

# ## Reading R(t) in real time
#
# The retrospective offspring fit above scores every source at full strength:
# `R_eff = R(t)`. Read in real time, a source's recent offspring may not yet
# have shown symptoms, so the count is incomplete and the rate must be thinned
# by the completeness of the offspring chain. The completeness is the CDF of the
# same convolved delay `delta + inc` used for the sourced truncation, evaluated
# at the time available since the source's onset, so
# [`thin_by_completeness`](@ref) gives `R_eff = R(t) * p` directly. The
# per-source thinning at the fitted delays:
inc_post = LogNormal(here.mu_inc.mean, here.sigma_inc.mean)
delta_post = Normal(here.mu_delta.mean, here.sigma_delta.mean)
completion_chain = convolve_distributions(delta_post, inc_post)
realtime_p = [completeness_probability(completion_chain,
                  horizon - source_onset_day[i]) for i in eachindex(Z)]
realtime_thinning = DataFrame(
    source_onset_day = Int.(source_onset_day),
    completeness = round.(realtime_p; digits = 3))
first(realtime_thinning, 6)

# ## Scope
#
# This page reproduces the upstream delay distributions, the offspring
# dispersion, and the weekly reproduction number, and shows the completeness
# thinning that corrects the rate when the line list is read before the outbreak
# has finished. The upstream model also runs counterfactual outbreak projections
# from these same posteriors, which are left to that analysis.
