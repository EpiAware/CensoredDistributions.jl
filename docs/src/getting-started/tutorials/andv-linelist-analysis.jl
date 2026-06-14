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
# A case enters the model in exactly one of two ways, never both.
# A case with a known human source is a chain of two observed delays: the
# transmission timing from the source's onset to this case's own infection, then
# the incubation period from that infection to this case's onset. In these data
# the case's own infection is its RECORDED EXPOSURE day, so the splitting event is
# observed (to the day) rather than latent, and both delays are scored directly.
# A case with no human source is the zoonotic index: it has no recorded exposure,
# so its infection has a broad prior and it contributes its incubation period
# alone, with no transmission edge.
# Each case therefore scores its incubation period exactly once, matching the
# upstream `latent_times_model` where every case carries one infection time and
# one incubation term.
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
# Exposure and onset are recorded to the day, so each delay's exact within-day
# position is unobserved. This day-resolution censoring can be handled two ways.
# The MARGINAL form integrates the within-day position out analytically, scoring
# each observed day-delay through a [`double_interval_censored`](@ref) leaf. The
# LATENT form instead samples each within-day position as a per-edge parameter and
# scores the delays conditionally on it.
# The two are the same model: the marginal leaf is the integral of the latent edge
# over the within-day primary, so they recover the same posterior. On these data
# the primary integral is analytical, so neither is appreciably faster; the choice
# is one of style.
#
# This page fits BOTH and compares them.
# The latent form is the primary fit, used for every downstream result; the
# marginal form is then fitted on the same data, and a closing section sets the
# two recovered delay parameter sets side by side against the published targets so
# the marginal-versus-latent agreement can be read directly. Both forms share the
# same mutually exclusive index/sourced split, so they fit the SAME 34 incubation
# terms and recover the same posterior.
#
# The split is keyed on a `:kind` field. A `:sourced` case is its two observed
# delays (transmission from source onset to the recorded exposure, then incubation
# from exposure to onset), and the single `:index` case is one latent incubation
# leaf off a broad-prior infection. Because the split is mutually exclusive, the
# index branch fits only the single zoonotic index case and the sourced branch
# fits the rest, so the incubation period is scored once per case.
# The composer tools used to build the branches are covered in the
# [composer toolkit tutorial](@ref composer-toolkit); here the focus is the
# delay model and the Bayesian workflow.

# ## Data
#
# The bundled line list is the Epuyén subset (n = 34 cases used here: one
# zoonotic index plus 33 human-sourced cases) encoded from Table S2 of Martínez
# et al. (2020), redistributed from the re-analysis repository under the MIT
# licence (see the data folder's `README`).
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

# Each case becomes EXACTLY ONE record, tagged by a `:kind` field the model routes
# on. The split is mutually exclusive: a case with a known human source is a
# `:sourced` record, and a case with no human source is the zoonotic `:index`
# record. A case is never both, so its incubation period is scored exactly once,
# matching the upstream model's single infection time per case.
#
# The key fact about these data is that every human-sourced case has a recorded
# EXPOSURE date, almost always a single day (the shared party, the wake, the
# household visit). The exposure date IS the case's own infection time, observed
# to the day. So a sourced case observes TWO delays whose splitting event (the
# infection) is recorded, not latent: the transmission timing from the source's
# onset to the exposure day, and the incubation period from the exposure day to
# the case's onset. Both delays are observed, so the incubation period is scored
# directly, once, from the recorded exposure. This is what the upstream model
# does when its `T_inf ~ Uniform(exp_lo, exp_hi)` collapses onto a one-day
# exposure window.
#
# A `:sourced` row therefore carries its two observed delays, `transmission`
# (exposure minus source onset) and `incubation` (onset minus exposure), and the
# real-time window `inc_obs = horizon - exposure` that right-truncates the
# incubation period. The single `:index` record (the zoonotic index, Patient 1,
# which has no human source and no recorded exposure date) is the only case whose
# infection is genuinely unobserved: it carries a `missing` `infection` event and
# its `onset` measured off a reference `inc_window` days earlier, so a broad-prior
# latent infection ranges across the plausible incubation span, exactly as the
# upstream index case draws `T_inf ~ Uniform(onset - inc_window, onset)`.
#
# Real-time truncation differs by record type, and the two halves are easy to mix
# up. A `:sourced` record's incubation period is one observed delay from the
# exposure day, so it is right-truncated to the single-delay window
# `horizon - exposure`. The single index case follows the upstream model, which
# applies no real-time truncation to the zoonotic index (its broad-prior infection
# already spans the open window).
#
# The broad prior on the zoonotic index's unobserved infection: it may have
# occurred up to `inc_window` days before its onset, matching the upstream
# `T_inf ~ Uniform(onset - 80, onset)`.
inc_window = 80.0

records = NamedTuple[]
for i in eachindex(pid)
    src = ll.source_case[i]
    if !ismissing(src) && src != "index"
        ## Human-sourced case: the recorded exposure day IS the observed
        ## infection, splitting the chain into two observed delays.
        s = first(split(string(src), "/"))  # first listed candidate source
        haskey(onset_of, s) || continue
        so = onset_of[s]
        ex = ismissing(exp_lo[i]) ? so + 1.0 : exp_lo[i]  # exposure = infection
        push!(records,
            (kind = :sourced, transmission = ex - so,
                incubation = onset_day[i] - ex, inc_obs = horizon - ex))
    else
        ## Zoonotic index (no human source): a latent incubation leaf off a
        ## broad-prior infection placed up to `inc_window` days before onset.
        push!(records, (kind = :index, infection = missing, onset = inc_window))
    end
end
(index = count(r -> r.kind == :index, records),
    sourced = count(r -> r.kind == :sourced, records))

# ## The delay model
#
# `inc` is the incubation period (a LogNormal), shared by both record types.
# `delta` is the transmission timing (a Normal, since a case can be infected
# shortly before or after its source's onset), used only by the sourced branch.
#
# The index branch is a single-edge latent incubation chain
# `infection -> onset`: the case's own infection is the sampled latent (its broad
# prior is a `Uniform(0, inc_window)` primary smear on the incubation leaf), and
# the incubation period carries that infection to the observed onset. This mirrors
# the upstream index case, which samples a broad-prior infection time and scores
# one incubation density off it, with no transmission edge.
# The sourced branch is a two-edge chain `srconset -> infection -> onset` whose
# middle infection event is OBSERVED (the exposure day): the transmission timing
# carries the source's onset to the recorded exposure, then the incubation period
# carries that exposure to the case's onset. Because the splitting infection is
# observed, both edges are scored directly against `delta` and `inc`, so the
# incubation period is scored once per sourced case off its exposure day. The two
# day-resolution edges differ only in how the within-day position of each event
# is handled, which is exactly the marginal-versus-latent choice fitted below.
# The model routes each record to its branch by its `:kind`, so both kinds share
# one `inc` and one `delta`. Because the index/sourced split is mutually
# exclusive, the index branch fits only the single zoonotic index case and the
# sourced branch fits the rest, so the incubation period is scored exactly 34
# times (once per case), not twice.

# Index branch: a single-edge latent incubation chain off a broad-prior
# infection. The incubation leaf carries a `Uniform(0, inc_window)` primary smear
# standing in for the unobserved infection position, and `latent` samples that
# infection so the incubation is scored as a bare density off it, the
# single-infection-time index case.
function index_branch(inc)
    return latent(Sequential(
        (primary_censored(inc, Uniform(0, inc_window)),),
        (:infection_onset,)))
end

# Sourced branch (LATENT form): the transmission -> incubation chain with the
# OBSERVED exposure as the middle event, wrapped in `latent` so each edge's
# within-day position (the primary event) is sampled and the two day-resolution
# delays are scored conditionally. The incubation edge is right-truncated to the
# real-time window `inc_obs` before the chain is built, so the sourced incubation
# carries the same single-delay completeness denominator as the upstream model.
function sourced_branch(inc, delta, inc_obs)
    inc_t = truncate_to_horizon(inc, inc_obs)
    chain = Sequential(
        (primary_censored(delta, Uniform(0, 1)),
            primary_censored(inc_t, Uniform(0, 1))),
        (:srconset_infection, :infection_onset))
    return latent(chain)
end

# The sourced LATENT record: the source onset is the anchor (`srconset = 0`), the
# exposure is the observed infection, and the onset is the observed end.
function sourced_record(r)
    return (srconset = 0.0, infection = r.transmission,
        onset = r.transmission + r.incubation)
end

# The index record routed through `index_branch`: only its event fields.
index_record(r) = (infection = r.infection, onset = r.onset)

# ## Priors
#
# The incubation parameters are a single TIED pair shared by both branches,
# rather than one set per branch, so `inc` and `delta` are sampled once inside
# the model. The priors are those of the upstream model.
#
# The model samples the two delay parameter pairs, then loops over the combined
# `records` table. A `:sourced` row scores its two observed delays through the
# latent chain (sampling each edge's within-day position), with the incubation
# edge right-truncated to its real-time window. The single `:index` row scores the
# latent incubation leaf, sampling its broad-prior infection time. The index case
# follows the upstream model and carries no real-time truncation (its broad-prior
# infection already spans the open window).

@model function andv(records)
    mu_inc ~ Normal(3.0, 0.5)
    sigma_inc ~ truncated(Normal(0.0, 0.5); lower = 0)
    mu_delta ~ Normal(0.0, 5.0)
    sigma_delta ~ truncated(Normal(0.0, 1.0); lower = 0)
    inc = LogNormal(mu_inc, sigma_inc)
    delta = Normal(mu_delta, sigma_delta)

    ## Each record scored by its own branch: a sourced case is a latent two-edge
    ## chain off its observed exposure; the index case is a latent incubation
    ## leaf off a broad-prior infection.
    idx = index_branch(inc)
    obs = Vector{Any}(undef, length(records))
    for (k, r) in enumerate(records)
        if r.kind == :sourced
            d = sourced_branch(inc, delta, r.inc_obs)
            sub = prefix(composed_distribution_model(d, sourced_record(r)),
                Symbol("rec_$k"))
        else
            sub = prefix(composed_distribution_model(idx, index_record(r)),
                Symbol("rec_$k"))
        end
        obs[k] ~ to_submodel(sub, false)
    end
    return delta
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
for _ in 1:4
    ## Latent index path off a broad-prior infection: an infection position drawn
    ## across the window plus the incubation period gives the observed onset.
    path = predict_events(latent(primary_censored(inc_true,
        Uniform(0, inc_window))))
    push!(sim_records, (kind = :index, infection = missing, onset = path[2]))
end
for _ in 1:30
    ## Source onset 0 -> exposure (transmission) -> onset (incubation), the
    ## exposure observed to the day, as in the real records.
    transmission = round(rand(delta_true))
    incubation = round(rand(truncated(inc_true; lower = 0.5)))
    push!(sim_records,
        (kind = :sourced, transmission = transmission,
            incubation = incubation, inc_obs = sim_horizon))
end

# Each sourced case scores two observed delays and the index case one latent
# leaf, so each leapfrog step is cheap. The two chains run in PARALLEL with
# [`MCMCThreads`](https://turinglang.org/Turing.jl/stable/) at a sampler budget
# (150 warmup, 150 draws) large enough to read credible intervals from. Chains
# start from the priors (`InitFromPrior`): the real-time truncation creates a
# second, spurious mode at a near-zero incubation period (where every delay is
# trivially complete), and a prior-centred start keeps the sampler in the basin
# that carries the data. The model is differentiated with `AutoForwardDiff`
# (Mooncake is the alternative supported backend; Enzyme is not used because it
# aborts uncatchably on the heterogeneous composer-tree recursion).
Random.seed!(20260608)
sim_chain = sample(andv(sim_records),
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
latent_time = @elapsed chain = sample(andv(records),
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
# keyed by step name, then the per-event
# [`mean`](@ref CensoredDistributions.mean)`(latent(posted))` Vector
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

# ## The marginal alternative
#
# The latent fit above samples each delay's within-day position (the primary
# event of each edge) and scores the day-resolution delays conditionally. The
# MARGINAL form instead integrates those within-day positions out analytically,
# so each sourced case is two double-interval-censored leaves scored directly on
# its observed delays: the incubation period from the exposure day to onset
# (right-truncated to the real-time window) and the transmission timing from the
# source's onset to the exposure day. Integrating the within-day primary out is
# exactly what [`double_interval_censored`](@ref) does, so the marginal leaves
# score the SAME likelihood as the latent edges. The two forms therefore fit the
# same model and recover the same posterior.
#
# The single zoonotic index case is handled identically to the latent form (the
# same latent incubation leaf off a broad-prior infection), so the two
# formulations differ only in how the 33 sourced cases' within-day positions are
# handled, integrated versus sampled. No case is scored twice: the index/sourced
# split is the same mutually exclusive split, so the incubation period is scored
# 34 times in both forms.
#
# Each [`double_interval_censored`](@ref) leaf integrates the within-day primary
# position out per likelihood evaluation. For this LogNormal incubation and
# Uniform primary that integral has an analytical solution, so the marginal form
# is not appreciably more expensive here; the two forms cost much the same and
# recover the same posterior. The point of this section is to show that
# equivalence, not a speed win: the marginal form trades sampled within-day edges
# for integrated leaves and lands on the same incubation estimate.

# The marginal sourced incubation leaf: the observed incubation period from the
# exposure day, primary- and interval-censored at day resolution and
# right-truncated to the real-time window. The transmission leaf is the same
# day-censored construction with no truncation. The index leaf is the latent
# broad-prior incubation, shared with the latent form.
function marginal_incubation(inc, window)
    return double_interval_censored(inc; primary_event = Uniform(0, 1),
        upper = window, interval = 1)
end
function marginal_transmission(delta)
    return double_interval_censored(delta; primary_event = Uniform(0, 1),
        interval = 1)
end

# The marginal model loops the records, scoring each sourced case's two observed
# delays through their double-censored leaves and the single index case through
# the shared latent incubation leaf. No `@addlogprob!` truncation term is needed:
# the incubation leaf's `upper` bound is the real-time completeness denominator.
@model function andv_marginal(records)
    mu_inc ~ Normal(3.0, 0.5)
    sigma_inc ~ truncated(Normal(0.0, 0.5); lower = 0)
    mu_delta ~ Normal(0.0, 5.0)
    sigma_delta ~ truncated(Normal(0.0, 1.0); lower = 0)
    inc = LogNormal(mu_inc, sigma_inc)
    delta = Normal(mu_delta, sigma_delta)

    idx = index_branch(inc)
    obs = Vector{Any}(undef, length(records))
    for (k, r) in enumerate(records)
        if r.kind == :sourced
            di = marginal_incubation(inc, r.inc_obs)
            sub = prefix(double_interval_censored_model(di, r.incubation),
                Symbol("rec_$k"))
            obs[k] ~ to_submodel(sub, false)
            dt = marginal_transmission(delta)
            tsub = prefix(double_interval_censored_model(dt, r.transmission),
                Symbol("tr_$k"))
            _t ~ to_submodel(tsub, false)
        else
            sub = prefix(composed_distribution_model(idx, index_record(r)),
                Symbol("rec_$k"))
            obs[k] ~ to_submodel(sub, false)
        end
    end
    return delta
end

# Fitted at the SAME budget as the latent fit, on the same data, so the
# comparison is a like-for-like posterior check rather than a budget artefact.
Random.seed!(20260608)
marg_time = @elapsed marg_chain = sample(andv_marginal(records),
    NUTS(200, 0.95; max_depth = 6, adtype = AutoForwardDiff()),
    MCMCThreads(), 200, 2;
    initial_params = fill(InitFromPrior(), 2), progress = false)
nothing #hide

# ## Marginal versus latent versus the published target
#
# The two formulations are set side by side against the published posterior. They
# fit the same model, so the recovered parameters agree within Monte Carlo error:
# the incubation log-mean (`mu_inc`) and log-SD (`sigma_inc`) match between forms
# and against the target. The weakly identified transmission-timing parameters
# stay close to their priors under both, so they differ between forms by more
# than the incubation parameters but remain in the same wide region.
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

# A note on the cost, read from the two real-data fits at the same budget. The
# fair comparison is the cost PER post-warmup draw: both forms draw 400
# post-warmup samples, so the wall-clock ratio is the per-draw cost ratio. On
# these data the analytical primary integral makes the two forms cost much the
# same per draw, so the choice between them is about expressiveness rather than
# speed: the latent form samples each within-day position and is convenient when
# the infection path itself is wanted, while the marginal form scores each delay
# as one integrated leaf and is the more familiar textbook expression.
latent_per_draw = latent_time / 400   # 200 draws x 2 chains
marg_per_draw = marg_time / 400       # 200 draws x 2 chains
cost_ratio = round(marg_per_draw / latent_per_draw; digits = 1)
tradeoff = DataFrame(
    form = ["latent", "marginal"],
    draws = [400, 400],
    wall_clock_s = round.([latent_time, marg_time], digits = 1),
    ms_per_draw = round.(
        [1000 * latent_per_draw, 1000 * marg_per_draw], digits = 1),
    sourced_likelihood = ["sampled within-day edges", "integrated leaves"])

# The marginal-to-latent per-draw cost ratio is about `cost_ratio` here; the two
# forms recover the same incubation posterior, so the choice is one of style, not
# accuracy or speed.
tradeoff

# The recovered parameters from both forms against the target.
both

# ## The transmission timing is weakly identified
#
# The transmission timing `delta` is the gap between a source's onset and the
# next case's infection (its recorded exposure day).
# In these data the recorded exposure of a secondary case usually coincides with
# its source's symptom onset (the shared party and wake events), so the
# transmission delay is near zero for most cases and there is little information
# to pin its spread.
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
