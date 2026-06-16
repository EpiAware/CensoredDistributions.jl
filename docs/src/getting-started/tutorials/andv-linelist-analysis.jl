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
# It is the same biological process for every case, so it is shared across the
# whole outbreak.
# The transmission timing runs from a source's symptom onset to the moment that
# source infects the next case.
# It only exists for a case that has a human source, so it is estimated from the
# sourced cases alone.
#
# A case enters the model in exactly one of two ways, never both.
# A case with a known human source is a chain of two observed delays: the
# transmission timing from the source's onset to this case's own infection, then
# the incubation period from that infection to this case's onset.
# In these data the case's own infection is its recorded exposure day, so the
# splitting event is observed to the day rather than unobserved, and both delays
# are scored directly.
# A case with no human source is the zoonotic index: it has no recorded exposure,
# so its infection has a broad prior and it contributes its incubation period
# alone, with no transmission edge.
# Each case therefore scores its incubation period exactly once, matching the
# upstream model where every case carries one infection time and one incubation
# term.
#
# This disjunction (a case is index or sourced, never both) is exactly what
# [`Select`](@ref) expresses: two independent named alternatives, picked per
# record by a data field rather than by a branch probability. We build the two
# alternatives with the composer front-ends and tie their shared incubation
# period across both with [`shared`](@ref), so one incubation parameter set is
# sampled once and reused by both branches.
#
# Two features of the data shape the model.
# Exposure and onset dates are recorded to the day, so every delay is doubly
# interval censored: the origin day and the later event day are both one-day
# windows rather than instants.
# The outbreak is read in real time, so a case only appears once its onset has
# happened by the analysis date, which right-truncates the sourced incubation
# period to the window still open before that date.
#
# How the composer stack is built, scored and simulated from in general is
# covered in the [composer toolkit tutorial](@ref composer-toolkit); here we use
# the stack and focus on the delay model and the Bayesian workflow.

# ## Packages used
#
# CSV and DataFramesMeta with Dates for the line-list pipeline, Turing for
# inference, FlexiChains for reading the posterior back onto the composed object,
# PairPlots and CairoMakie for the figures, and CensoredDistributions for the
# composed delay model.

using CSV, DataFramesMeta, Dates
using CensoredDistributions, Distributions
using CensoredDistributions: latent, shared
using Turing, Random, Statistics
using DynamicPPL: to_submodel, prefix, InitFromPrior, @varname
using FlexiChains: Parameter
using ADTypes: AutoMooncake
import Mooncake
using CairoMakie, PairPlots

# ## Data
#
# The bundled line list is the Epuyén subset (n = 34 cases used here: one
# zoonotic index plus 33 human-sourced cases) encoded from Table S2 of Martínez
# et al. (2020), redistributed from the re-analysis repository under the MIT
# licence (see the data folder's `README`).
# We read the dates, express each delay as a day offset from its origin, and
# drop the two alternative-source sensitivity rows that the upstream main fit
# excludes.

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
# happened, so the sourced incubation period is right-truncated to the window
# still open before the horizon.
horizon = maximum(onset_day) + 7.0

# The broad prior on the zoonotic index's unobserved infection: it may have
# occurred up to `inc_window` days before its onset, matching the upstream
# `T_inf ~ Uniform(onset - 80, onset)`.
inc_window = 80.0

# Each case becomes exactly one record. The split is mutually exclusive: a case
# with a known human source is a `:sourced` record, and the case with no human
# source is the zoonotic `:index` record. Because the split is mutually
# exclusive, the incubation period is scored exactly once per case.
#
# Every human-sourced case has a recorded exposure date, almost always a single
# day (the shared party, the wake, the household visit). The exposure date is the
# case's own infection time, observed to the day. So a sourced record carries two
# observed delays whose splitting event (the infection) is recorded, not
# unobserved: the transmission timing from the source's onset to the exposure
# day, and the incubation period from the exposure day to the case's onset. Both
# delays are anchored on the source's onset (`srconset = 0`), the exposure is the
# observed `infection`, and the case's `onset` is the observed end. The reserved
# `obs_time` field carries the real-time horizon so the record is right-truncated
# to the window still open at the analysis date.
#
# The single `:index` record (Patient 1, no human source and no recorded exposure)
# is the only case whose infection is genuinely unobserved: it carries a `missing`
# `infection` event and an `onset` measured `inc_window` days after a reference, so
# its broad-prior infection ranges across the plausible incubation span, exactly
# as the upstream index case draws `T_inf ~ Uniform(onset - inc_window, onset)`.
# The upstream model applies no real-time truncation to the zoonotic index, so the
# index record carries no `obs_time`.

index_rows = NamedTuple[]
sourced_rows = NamedTuple[]
for i in eachindex(pid)
    src = ll.source_case[i]
    if !ismissing(src) && src != "index"
        ## Human-sourced case: the recorded exposure day is the observed
        ## infection, splitting the chain into two observed delays.
        s = first(split(string(src), "/"))  # first listed candidate source
        haskey(onset_of, s) || continue
        so = onset_of[s]
        ex = ismissing(exp_lo[i]) ? so + 1.0 : exp_lo[i]  # exposure = infection
        push!(sourced_rows,
            (srconset = 0.0, infection = ex - so,
                onset = onset_day[i] - so, obs_time = horizon - so))
    else
        ## Zoonotic index (no human source): a broad-prior unobserved infection.
        push!(index_rows, (infection = missing, onset = inc_window))
    end
end
(index = length(index_rows), sourced = length(sourced_rows))

# ## The delay model
#
# The whole delay structure is one [`Select`](@ref) node over two named
# alternatives, built with the composer front-ends.
# `inc` is the incubation period (a LogNormal), shared by both alternatives;
# `delta` is the transmission timing (a Normal, since a case can be infected
# shortly before or after its source's onset), used only by the sourced branch.
# Each leaf is built with [`double_interval_censored`](@ref): the day-resolution
# data means every delay has a one-day primary-event window and a one-day
# secondary window, and the censoring is part of the composed object rather than
# a separate step.
#
# The index alternative is a single incubation leaf off a broad-prior infection:
# the leaf's primary event is a `Uniform(0, inc_window)` standing in for the
# unobserved within-window infection position, so the index case's infection is
# the sampled latent primary event and the incubation period carries it to the
# observed onset. The sourced alternative is a two-step [`Sequential`](@ref)
# chain `srconset → infection → onset` whose middle event is observed: the
# transmission timing carries the source's onset to the recorded exposure, then
# the incubation period carries that exposure to the case's onset.
#
# The incubation leaf in both alternatives is tagged [`shared`](@ref)`(:inc, …)`,
# so the prior/params interface treats the two occurrences as one free parameter:
# it is inventoried once, sampled once, and the one sampled value is placed in
# both branches. This is the single tied incubation period the upstream model
# uses.

double_day(d) = double_interval_censored(d; primary_event = Uniform(0, 1),
    interval = 1)

function delay_select(; inc = LogNormal(3.0, 0.3), delta = Normal(0.0, 1.0))
    index = compose((infection_onset = shared(:inc,
        primary_censored(inc, Uniform(0, inc_window))),))
    sourced = sequential(
        :srconset_infection => double_day(delta),
        :infection_onset => shared(:inc, double_day(inc)))
    return selecting(:index => index, :sourced => sourced)
end

template = delay_select()

# The recursive `show` lays out the disjunction: an `index` alternative and a
# `sourced` alternative, picked per record by the selector field.
template

# Each alternative's flat event names are the columns its records supply.
(index = event_names(event(template, :index)),
    sourced = event_names(event(template, :sourced)))

# ## Priors
#
# [`params_table`](@ref) lists every free parameter of the composed delays as a
# flat table. The shared incubation tag appears once (`inc`), and the
# transmission timing lives under the sourced branch.

param_inventory = params_table(template)

# [`build_priors`](@ref) turns the table into the nested prior NamedTuple that
# [`composed_parameters_model`](@ref) consumes, giving each parameter a
# support-derived default. We override the four with the upstream model's priors:
# the incubation log-mean and log-SD, and the transmission-timing mean and SD.

priors = build_priors(param_inventory;
    priors = (
        inc = (mu = Normal(3.0, 0.5),
            sigma = truncated(Normal(0.0, 0.5); lower = 0)),
        sourced = (srconset_infection = (mu = Normal(0.0, 5.0),
            sigma = truncated(Normal(0.0, 1.0); lower = 0)),)))

# ## The model
#
# The model samples the delay parameters once through
# [`composed_parameters_model`](@ref), then scores the records through the
# batched [`composed_distribution_model`](@ref). The index and sourced records
# have different event-slot counts (one delay versus two), so they score as two
# batches: the index records through the index alternative, the sourced records
# through the sourced alternative, each in a single `~` over the whole subset.
# Both batches read the same sampled `delays`, so the shared incubation period is
# fitted once across all 34 cases. The sourced records carry their `obs_time`
# horizon, so the batched scorer right-truncates each sourced delay to its
# real-time window through the package's truncation, with no manual completeness
# term.

@model function andv(template, index_rows, sourced_rows)
    delays ~ to_submodel(composed_parameters_model(template, priors))
    obs_index ~ to_submodel(
        prefix(
            composed_distribution_model(event(delays, :index), index_rows),
            :index), false)
    obs_sourced ~ to_submodel(
        prefix(
            composed_distribution_model(event(delays, :sourced), sourced_rows),
            :sourced), false)
    return delays
end

# ## Simulate, fit, and recover
#
# Before touching the data we check the model can recover known parameters.
# A full delay path is drawn for each record from the latent form with
# `rand(latent(d))`, one draw per record, then the simulated delays are fitted
# and compared with the truth. See
# [Marginal versus latent](@ref marginal-versus-latent) for the two forms and
# [Fit marginal, sample event based](@ref) for fitting marginally then drawing
# event paths.

inc_true = LogNormal(3.06, 0.32)
delta_true = Normal(0.17, 0.62)

# A generous real-time horizon (200 days) is used for the simulation so the
# right-truncation is light and the truth is recovered cleanly; the truncation
# machinery is exercised in full on the real fit below.
sim_horizon = 200.0
Random.seed!(20260608)
sim_index = NamedTuple[]
for _ in 1:4
    ## Index path off a broad-prior infection: an infection position drawn across
    ## the window plus the incubation period gives the observed onset. The latent
    ## leaf `rand` returns a `(primary, observed)` NamedTuple, so the observed
    ## onset is `path.observed`.
    path = rand(latent(primary_censored(inc_true, Uniform(0, inc_window))))
    push!(sim_index, (infection = missing, onset = path.observed))
end
sim_sourced = NamedTuple[]
for _ in 1:30
    ## Source onset 0 → exposure (transmission) → onset (incubation), the
    ## exposure observed to the day, as in the real records.
    transmission = round(rand(delta_true))
    incubation = round(rand(truncated(inc_true; lower = 0.5)))
    push!(sim_sourced,
        (srconset = 0.0, infection = transmission,
            onset = transmission + incubation, obs_time = sim_horizon))
end

# Each sourced record scores two observed delays and each index record one leaf,
# so each leapfrog step is cheap. The two chains run in parallel with
# [`MCMCThreads`](https://turinglang.org/Turing.jl/stable/) at a modest sampler
# budget (150 warmup, 150 draws) large enough to read credible intervals from.
# Chains start from the priors (`InitFromPrior`): the real-time truncation
# creates a second, spurious mode at a near-zero incubation period (where every
# delay is trivially complete), and a prior-centred start keeps the sampler in
# the basin that carries the data. The model is differentiated with Mooncake
# reverse mode (`AutoMooncake`), the preferred backend for a tree this size.
# Enzyme reverse is registered broken on these heterogeneous nested paths; the
# maths is AD-safe on the other backends.
Random.seed!(20260608)
sim_chain = sample(andv(template, sim_index, sim_sourced),
    NUTS(150, 0.95; max_depth = 6, adtype = AutoMooncake(; config = nothing)),
    MCMCThreads(), 150, 2;
    initial_params = fill(InitFromPrior(), 2), progress = false)
nothing #hide

# The fitted parameters are read back onto the composed object with
# [`update`](@ref), passing the chain directly so the post-fit delays come from
# the composer rather than from manual chain indexing. The per-event
# [`mean`](@ref CensoredDistributions.mean)`(latent(...))` NamedTuple then reads
# each delay's mean off the updated object, keyed by [`event_names`](@ref); here
# the sourced branch's two day-resolution edge means come straight off the
# recovered object.
sim_fit = update(template, sim_chain; prefix = :delays)
sim_edge_means = mean(latent(event(sim_fit, :sourced)))

# Posterior means against the simulating truth, read from the per-draw delay
# parameters. The incubation period is recovered closely; the transmission
# timing is recovered in the right region but with a wide interval, the first
# sign of its weak identifiability.
function delta_params(chain)
    mu = vec(chain[Parameter(@varname(delays.sourced.srconset_infection.mu))])
    sigma = vec(chain[Parameter(
        @varname(delays.sourced.srconset_infection.sigma))])
    return (mu = mu, sigma = sigma)
end
function inc_params(chain)
    mu = vec(chain[Parameter(@varname(delays.inc.mu))])
    sigma = vec(chain[Parameter(@varname(delays.inc.sigma))])
    return (mu = mu, sigma = sigma)
end

sim_inc = inc_params(sim_chain)
sim_delta = delta_params(sim_chain)
sim_draws = (mu_inc = sim_inc.mu, sigma_inc = sim_inc.sigma,
    mu_delta = sim_delta.mu, sigma_delta = sim_delta.sigma)
sim_pars = (:mu_inc, :sigma_inc, :mu_delta, :sigma_delta)
sim_summary = DataFrame(
    parameter = ["mu_inc", "sigma_inc", "mu_delta", "sigma_delta"],
    truth = [3.06, 0.32, 0.17, 0.62],
    posterior_mean = [round(mean(sim_draws[p]); digits = 3) for p in sim_pars],
    lower = [round(quantile(sim_draws[p], 0.025); digits = 3)
             for p in sim_pars],
    upper = [round(quantile(sim_draws[p], 0.975); digits = 3)
             for p in sim_pars])

# ## Fitting the line list
#
# The same model is fitted to the real records.

Random.seed!(20260608)
chain = sample(andv(template, index_rows, sourced_rows),
    NUTS(200, 0.95; max_depth = 6, adtype = AutoMooncake(; config = nothing)),
    MCMCThreads(), 200, 2;
    initial_params = fill(InitFromPrior(), 2), progress = false)
nothing #hide

fit = update(template, chain; prefix = :delays)

# ## Priors and posteriors
#
# `PairPlots.jl` shows the prior and the posterior of the four parameters on one
# figure. The incubation parameters concentrate tightly; the transmission-timing
# parameters barely move from their priors, which is the identifiability story
# made visible.
# The prior draws come from running the same model in prior mode
# (`sample(model, Prior(), …)`), so the overlay passes through the very same
# parameter block as the posterior and the shrinkage reads directly.

Random.seed!(20260608)
prior_chain = sample(andv(template, index_rows, sourced_rows),
    Prior(), MCMCThreads(), 300, 2; progress = false)

prior_inc = inc_params(prior_chain)
prior_delta = delta_params(prior_chain)
prior_draws = (mu_inc = prior_inc.mu, sigma_inc = prior_inc.sigma,
    mu_delta = prior_delta.mu, sigma_delta = prior_delta.sigma)

post_inc = inc_params(chain)
post_delta = delta_params(chain)
post_table = (mu_inc = post_inc.mu, sigma_inc = post_inc.sigma,
    mu_delta = post_delta.mu, sigma_delta = post_delta.sigma)

fig = pairplot(
    PairPlots.Series(prior_draws, label = "prior", color = (:grey, 0.5)) =>
        (PairPlots.Scatter(markersize = 3), PairPlots.MarginHist()),
    PairPlots.Series(post_table, label = "posterior",
        color = (:steelblue, 0.6)) =>
        (PairPlots.Scatter(markersize = 3), PairPlots.MarginHist()))
fig

# ## Reading the fitted delays off the composed object
#
# The fitted incubation period comes straight off the updated composed object,
# so the post-fit delay is the composer's leaf rather than a hand-built
# distribution. The incubation mean delay in days is read from the per-draw
# LogNormal means.
fitted_inc = CensoredDistributions.free_leaf(
    event(fit, :index, :infection_onset))

inc_means = [mean(LogNormal(post_inc.mu[i], post_inc.sigma[i]))
             for i in eachindex(post_inc.mu)]
inc_summary = (mean = mean(inc_means),
    lower = quantile(inc_means, 0.025), upper = quantile(inc_means, 0.975))

# ## Equivalence with the published estimates
#
# The comparison targets are the posterior means and 95% credible intervals from
# the re-analysis, read from its published posterior draws.
# They are matched posterior to posterior: the package reproduces the published
# analysis when the intervals overlap, not only when the point estimates agree.

function summ(v)
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

here = (mu_inc = summ(post_inc.mu), sigma_inc = summ(post_inc.sigma),
    mu_delta = summ(post_delta.mu), sigma_delta = summ(post_delta.sigma),
    inc_mean = inc_summary)

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
    NUTS(250, 0.9; max_depth = 6, adtype = AutoMooncake(; config = nothing)),
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
# convolved delay `delta + inc` (the same convolution the sourced truncation
# uses) evaluated at the time available since the source's onset. The package's
# [`thin_by_completeness`](@ref) returns the thinned `R_eff = R(t) * p` directly,
# and [`completeness_probability`](@ref) the underlying fraction `p`:
inc_post = LogNormal(here.mu_inc.mean, here.sigma_inc.mean)
delta_post = Normal(here.mu_delta.mean, here.sigma_delta.mean)
completion_chain = convolve_distributions(delta_post, inc_post)
R_week1 = rt_summary.R_mean[1]
realtime = map(eachindex(Z)) do i
    window = horizon - source_onset_day[i]
    p = completeness_probability(completion_chain, window)
    (source_onset_day = Int(source_onset_day[i]),
        completeness = round(p; digits = 3),
        R_eff_week1 = round(
            thin_by_completeness(R_week1, completion_chain,
                window); digits = 2))
end
realtime_thinning = DataFrame(realtime)
first(realtime_thinning, 6)

# ## Summary
#
# - The two ways a case enters the model (zoonotic index or human-sourced) are
#   one [`Select`](@ref) disjunction over named alternatives, picked per record
#   by the data rather than by a branch probability.
# - Each alternative is built with the composer front-ends and
#   [`double_interval_censored`](@ref) leaves; the incubation period is tied
#   across both with [`shared`](@ref), so one parameter set is sampled once and
#   reused, scoring the incubation period exactly once per case.
# - Priors come from [`params_table`](@ref) and [`build_priors`](@ref); the
#   records score through the batched [`composed_distribution_model`](@ref), and
#   the `obs_time` field drives the real-time right-truncation through the
#   package's truncation rather than a manual completeness term.
# - The posterior is read back with [`update`](@ref) applied to the fitted chain,
#   so the fitted delays come straight off the composed object.
# - The fit recovers the published incubation target
#   `LogNormal(mu_inc ≈ 3.06, sigma_inc ≈ 0.32)` and its mean delay; the weakly
#   identified transmission timing stays close to its prior, in agreement with
#   the upstream finding that transmission clusters around source onset.
# - The offspring layer reproduces the descending weekly reproduction number and
#   the superspreading dispersion, and the completeness thinning corrects the
#   rate when the line list is read before the outbreak has finished. The
#   upstream model also runs counterfactual outbreak projections from these same
#   posteriors, which are left to that analysis.
