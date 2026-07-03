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
# CensoredDistributions.jl jointly with the offspring branching process scored
# from the recorded offspring counts, and checks that the published estimates
# are recovered. Each source's expected offspring count is thinned by the
# completeness of its own composed delay, read off the same sampled object the
# delays are scored on, so the composition is load-bearing across the whole page.
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
# [`Choose`](@ref) expresses: two independent named alternatives, normally
# routed per record by the node's `:kind` selector in one scoring statement.
# The two alternatives here have different event-slot counts (the index case
# observes one delay, a sourced case two), and the batched marginal scorer
# scores a record table in one rectangular `product_distribution`, which needs
# every record to share a slot count; mixed lengths under one real-time horizon
# are deferred (see
# [#413](https://github.com/EpiAware/CensoredDistributions.jl/issues/413) and
# [#623](https://github.com/EpiAware/CensoredDistributions.jl/issues/623)). The
# split is known before fitting, so we score it the way the package recommends
# in the meantime: the records are sorted into an index batch and a sourced
# batch in Julia, and each fixed-length batch is scored against its named
# alternative. We build the two alternatives with the composer front-ends and
# tie their shared incubation period across both with [`shared`](@ref), so one
# incubation parameter set is sampled once and reused by both branches.
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
using CensoredDistributions: latent, shared, get_dist_recursive
using Turing, Random, Statistics
using DynamicPPL: to_submodel, prefix, InitFromPrior
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

datadir = joinpath(@__DIR__, "data", "andv")
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
        ## Its onset is the reference day `t0`, so the day offset placed here is
        ## `inc_window`: the onset sits that many days after the window start,
        ## not a coincidence with the prior width.
        index_onset = inc_window
        push!(index_rows, (infection = missing, onset = index_onset))
    end
end
(index = length(index_rows), sourced = length(sourced_rows))

# ## The delay model
#
# The whole delay structure is one [`Choose`](@ref) node over two named
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
# observed onset. The onset is recorded to the day, so the index leaf is
# [`double_interval_censored`](@ref) with the same one-day secondary window as
# every other delay, matching the upstream index case whose onset is drawn
# `Uniform(onset_lo, onset_hi)` over its one-day record. The sourced alternative
# is a two-step [`Sequential`](@ref) chain `srconset → infection → onset` whose
# middle event is observed: the transmission timing carries the source's onset to
# the recorded exposure, then the incubation period carries that exposure to the
# case's onset.
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
        double_interval_censored(inc; primary_event = Uniform(0, inc_window),
            interval = 1)),))
    sourced = sequential(
        :srconset_infection => double_day(delta),
        :infection_onset => shared(:inc, double_day(inc)))
    return choose(:index => index, :sourced => sourced)
end

template = delay_select()

# The recursive `show` lays out the disjunction: an `index` alternative and a
# `sourced` alternative, each scored as a batch against its named alternative.
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
# support-derived default. [`update`](@ref) then overrides the four with the
# upstream model's priors (the incubation log-mean and log-SD, and the
# transmission-timing mean and SD), addressing each leaf by its name path. The
# shared `inc` tag is a top-level name and the transmission timing sits under
# `sourced`.

priors = update(build_priors(param_inventory),
    :inc => (mu = Normal(3.0, 0.5),
        sigma = truncated(Normal(0.0, 0.5); lower = 0)),
    (:sourced, :srconset_infection) => (mu = Normal(0.0, 5.0),
        sigma = truncated(Normal(0.0, 1.0); lower = 0)));

# ## The offspring layer
#
# The same line list records who infected whom, so each source carries an
# offspring count: the number of secondaries the line list attributes to it.
# This is a branching process. Each source's offspring count is a negative
# binomial whose mean is a time-varying reproduction number `R(t)` evaluated at
# the source's own onset day, and whose dispersion `k` captures the
# superspreading (a few sources drive most onward infections). `R(t)` follows a
# weekly non-centred random walk on the log scale and `k` has a log-normal prior.
#
# Read in real time, a source's recent secondaries may not yet have shown
# symptoms, so its count is incomplete and the mean must be thinned by the
# completeness of that source's own offspring delay: the fraction of its
# secondaries whose symptom onset has happened by the analysis date. That delay
# is exactly the composed sourced chain, the transmission timing `delta` to the
# next infection then the incubation period `inc` to its onset, so the
# completeness is the CDF of `delta + inc` at the window still open since the
# source's onset. Reading that completeness off the same sampled delays the
# likelihood scores is what couples the offspring layer to the delays and lets
# the whole model be fitted jointly.

# Per-source onset day (days from the first onset) and offspring count.
Z = Int.(ll.Z)
source_onset_day = onset_day

# Thirteen weekly knots over the outbreak, matching the upstream `log_R` grid.
# `log R(t)` is piecewise linear between them; the few onsets past the last knot
# are clamped to its value by `log_R_at`.
knots = collect(0.0:7.0:(7.0 * 12))
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

# The offspring negative binomial built from a log mean and log dispersion: the
# success probability is `k / (k + exp(log_mu))`, computed as
# `inv(1 + exp(log_mu - log_k))` so it never evaluates `0/0` or `Inf/Inf`. The
# `log_mu - log_k` gap is clamped to a wide finite range so an extreme proposal
# cannot push the probability to exactly `0` or `1` (which a negative binomial
# rejects); the clamp bounds sit far outside any supported region, so they never
# bite the posterior. Here `log_mu` is the log-space completeness-thinned rate.
function nb_logmean(log_k, log_mu)
    k = exp(log_k)
    p = inv(1 + exp(clamp(log_mu - log_k, -30.0, 30.0)))
    return NegativeBinomial(k, p)
end

# Day-resolution log completeness of the offspring delay `delta + inc` at each
# window. The completeness `P(delta + inc <= w)` is written as a mixture over the
# transmission timing's daily PMF and the analytic incubation CDF,
# `sum_d P(delta = d) F_inc(w - d)`, so it carries no quadrature: it stays cheap
# and AD-safe inside the joint fit and matches the day resolution of the delay
# likelihood. The incubation CDF argument is floored at a small positive value so
# the LogNormal never takes `log` of a non-positive lag (a `NaN` reverse-mode
# gradient), and the completeness itself is floored so a recently seen source
# whose completeness underflows gives a large negative log rate rather than a
# rate of exactly zero (which would collapse the negative-binomial mean to a
# degenerate point mass).
delta_lags = collect(-20.0:20.0)
function log_completeness(delta, inc, windows)
    dpmf = cdf.(delta, delta_lags .+ 0.5) .- cdf.(delta, delta_lags .- 0.5)
    dpmf = dpmf ./ max(sum(dpmf), 1e-12)
    inc_cdf = cdf.(inc, max.(windows .- delta_lags', 1e-6))
    return log.(max.(inc_cdf * dpmf, 1e-12))
end

# ## The joint model
#
# One model scores the delays and the offspring counts together. It samples the
# delay parameters once through [`composed_parameters_model`](@ref), scores the
# index and sourced delay batches through the batched
# [`composed_distribution_model`](@ref) (the index records through the index
# alternative, the sourced records through the sourced alternative, each in one
# `~`; the sourced records carry their `obs_time` horizon so the scorer
# right-truncates each sourced delay to its real-time window), then scores the
# offspring counts. The per-source offspring rate is `R(t)` at the source's onset
# thinned by the completeness of that source's composed delay, and the
# completeness reads the transmission-timing and incubation cores straight off
# the sampled `delays` with [`get_dist_recursive`](@ref). The delays and the
# offspring layer therefore share one parameter block and are fitted jointly.

@model function andv(template, index_rows, sourced_rows,
        Z, source_onset, knots, horizon)
    delays ~ to_submodel(composed_parameters_model(template, priors))
    obs_index ~ to_submodel(
        prefix(
            composed_distribution_model(event(delays, :index), index_rows),
            :index), false)
    obs_sourced ~ to_submodel(
        prefix(
            composed_distribution_model(event(delays, :sourced), sourced_rows),
            :sourced), false)

    log_k ~ Normal(log(0.5), 0.7)
    sigma_rw ~ truncated(Normal(0.0, 0.2); lower = 0)
    log_R_init ~ Normal(log(1.5), 1.0)
    eps ~ filldist(Normal(0.0, 1.0), length(knots) - 1)
    log_R = vcat(log_R_init, log_R_init .+ accumulate(+, sigma_rw .* eps))

    ## The offspring completeness reads the transmission-timing and incubation
    ## cores off the sampled sourced chain, so the thinning is coupled to the
    ## same delays the records above are scored on.
    src = event(delays, :sourced)
    delta = get_dist_recursive(event(src, :srconset_infection))
    inc = get_dist_recursive(event(src, :infection_onset))
    log_p = log_completeness(delta, inc, horizon .- source_onset)
    for i in eachindex(Z)
        log_mu = log_R_at(source_onset[i], knots, log_R) + log_p[i]
        Z[i] ~ nb_logmean(log_k, log_mu)
    end
    return (; log_R, k = exp(log_k))
end

# ## Simulate, fit, and recover
#
# Before touching the data we check the joint model can recover known
# parameters: the two delays and the offspring dispersion and reproduction
# number together. A full delay path is drawn for each record from the latent
# form with `rand(latent(d))`, one draw per record, and an offspring count is
# drawn for each simulated source from the offspring negative binomial at a known
# reproduction number and dispersion; the simulated delays and counts are then
# fitted jointly and compared with the truth. The two alternatives have different
# event-slot counts and the simulation needs each record's latent primary event
# (the within-window infection time, the source's transmission timing) to build
# its real-time horizon, which the batched record sampler does not expose, so the
# two alternatives are simulated by hand here rather than through one
# `rand(template, rows)` call. See
# [Marginal versus latent](@ref marginal-versus-latent) for the two forms and
# [Fit marginal, sample event based](@ref) for fitting marginally then drawing
# event paths.

inc_true = LogNormal(3.06, 0.32)
delta_true = Normal(0.17, 0.62)

# A generous real-time horizon (200 days) is used for the simulation so the
# right-truncation is light and the truth is recovered; the truncation machinery
# is exercised in full on the real fit below. The day-binned `round` draws below
# match the model's day-resolution likelihood to the precision that matters
# here. `round(x)` and the leaf's `floor(x + Uniform(0, 1))` are both unbiased
# discretisations, so the simulated delays share the likelihood's mean, spread,
# and quantiles. The simulation uses 4 index and 30 sourced records, so both
# alternatives are exercised; the real fit has only 1 index record, so on the
# real data the index alternative is effectively a single observation and the
# incubation period is identified almost entirely from the sourced cases.
sim_horizon = 200.0
Random.seed!(20260608)
sim_index = NamedTuple[]
for _ in 1:4
    ## Index path off a broad-prior infection: an infection position drawn across
    ## the window plus the day-censored incubation period gives the observed
    ## onset. The latent leaf `rand` returns a `(primary, observed)` NamedTuple,
    ## so the observed onset is `path.observed`.
    path = rand(latent(double_interval_censored(inc_true;
        primary_event = Uniform(0, inc_window), interval = 1)))
    push!(sim_index, (infection = missing, onset = path.observed))
end
sim_sourced = NamedTuple[]
for _ in 1:30
    ## Source onset 0 → exposure (transmission) → onset (incubation), the
    ## exposure observed to the day, as in the real records. The incubation draw
    ## is floored at half a day so it never rounds to a zero-day delay (which
    ## would put onset on the exposure day); the LogNormal has negligible mass
    ## there, so this does not shift the recovered parameters.
    transmission = round(rand(delta_true))
    incubation = round(rand(truncated(inc_true; lower = 0.5)))
    push!(sim_sourced,
        (srconset = 0.0, infection = transmission,
            onset = transmission + incubation, obs_time = sim_horizon))
end

# Offspring counts for the simulation. Each simulated source sits at a spread of
# onset days across the outbreak and draws its count from the offspring negative
# binomial at a known reproduction number `R_true` and dispersion `k_true`,
# thinned by the completeness of the true delay at the generous simulation
# horizon. The thinning uses the same `log_completeness` the model applies, so a
# flat `R_true` is the truth the random-walk `R(t)` should recover.
R_true = 1.6
k_true = 0.5
sim_source_onset = collect(range(0.0, 84.0; length = length(sim_sourced)))
sim_log_p = log_completeness(delta_true, inc_true,
    sim_horizon .- sim_source_onset)
sim_Z = [rand(nb_logmean(log(k_true), log(R_true) + sim_log_p[i]))
         for i in eachindex(sim_source_onset)]

# Two chains run in parallel at a modest budget. The joint completeness coupling
# makes each gradient step heavier than the delays alone, so the budget is kept
# light for the doc build. Chains start from the priors (`InitFromPrior`): the
# real-time truncation creates a second, spurious mode at a near-zero incubation
# period (where every delay is trivially complete), and a prior-centred start
# keeps the sampler in the basin that carries the data. The model is
# differentiated with Mooncake reverse mode (`AutoMooncake`).
Random.seed!(20260608)
sim_model = andv(template, sim_index, sim_sourced, sim_Z, sim_source_onset,
    knots, sim_horizon)
sim_chain = sample(sim_model,
    NUTS(60, 0.95; max_depth = 6, adtype = AutoMooncake(; config = nothing)),
    MCMCThreads(), 60, 2;
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

# The posterior draws are shown against the simulating truth, the whole draw
# cloud per parameter rather than a single mean. The incubation truth sits
# inside its cloud; the transmission timing is recovered in the right region but
# with a broad cloud, the first sign of its weak identifiability.
# [`param_draws`](@ref)`(template, chain)` reads every draw off the chain as a
# vector of nested NamedTuples keyed like [`params`](@ref)`(template)`, so the
# four parameters come off by name (the shared incubation under `inc`, the
# transmission timing under the sourced branch) with no hand-written `@varname`
# index.
function flat_params(template, chain)
    edges = param_draws(template, chain; prefix = :delays)
    return (mu_inc = [e.inc.mu for e in edges],
        sigma_inc = [e.inc.sigma for e in edges],
        mu_delta = [e.sourced.srconset_infection.mu for e in edges],
        sigma_delta = [e.sourced.srconset_infection.sigma for e in edges])
end

sim_draws = flat_params(template, sim_chain)
sim_pars = (:mu_inc, :sigma_inc, :mu_delta, :sigma_delta)
sim_truth_vals = (mu_inc = 3.06, sigma_inc = 0.32,
    mu_delta = 0.17, sigma_delta = 0.62)

# Each parameter's posterior draws as a jittered cloud with the truth marked, so
# the recovery is read from the whole ensemble rather than a point summary.
sfig = Figure(size = (760, 380))
for (j, p) in enumerate(sim_pars)
    sax = Axis(sfig[1, j]; title = string(p),
        xticksvisible = false, xticklabelsvisible = false)
    d = sim_draws[p]
    scatter!(sax, 1.0 .+ 0.25 .* randn(length(d)), d;
        color = (:steelblue, 0.3), markersize = 4)
    hlines!(sax, [sim_truth_vals[p]]; color = :firebrick, linewidth = 2)
end
sfig

# The offspring parameters are recovered from the same joint fit. The dispersion
# `k` comes off the `log_k` draws and the flat simulating `R_true` off the mean
# of the recovered weekly `R(t)` (`generated_quantities` reads the deterministic
# `log_R` back per draw). The reproduction number lands close to its simulating
# value; the dispersion is recovered in the right region but only loosely, since
# a few dozen counts carry little information about superspreading. So the joint
# fit identifies the offspring layer alongside the delays.
sim_k = exp.(vec(sim_chain[:log_k]))
sim_logR = reduce(hcat, [g.log_R for g in vec(generated_quantities(
    sim_model, sim_chain))])
sim_offspring_recovery = (
    k = (truth = k_true, posterior = round(mean(sim_k); digits = 2)),
    R = (truth = R_true, posterior = round(mean(exp.(sim_logR)); digits = 2)))

# ## Fitting the line list
#
# The same joint model is fitted to the real records.

Random.seed!(20260608)
model = andv(template, index_rows, sourced_rows, Z, source_onset_day,
    knots, horizon)
chain = sample(model,
    NUTS(80, 0.95; max_depth = 6, adtype = AutoMooncake(; config = nothing)),
    MCMCThreads(), 80, 2;
    initial_params = fill(InitFromPrior(), 2), progress = false)
nothing #hide

fit = update(template, chain; prefix = :delays);

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
prior_chain = sample(model, Prior(), MCMCThreads(), 200, 2; progress = false)

prior_draws = flat_params(template, prior_chain)
post_draws = flat_params(template, chain)

fig = pairplot(
    PairPlots.Series(prior_draws, label = "prior", color = (:grey, 0.5)) =>
        (PairPlots.Scatter(markersize = 3), PairPlots.MarginHist()),
    PairPlots.Series(post_draws, label = "posterior",
        color = (:steelblue, 0.6)) =>
        (PairPlots.Scatter(markersize = 3), PairPlots.MarginHist()))
fig

# ## Reading the fitted delays off the composed object
#
# The incubation mean delay in days is read straight off the updated composed
# object rather than from a hand-built distribution.
# [`update`](@ref)`.(Ref(template), `[`param_draws`](@ref)`(template, chain))`
# rebuilds the composed delays once per draw, and
# [`mean`](@ref CensoredDistributions.mean)`(latent(event(...)))` returns the
# per-event means keyed by [`event_names`](@ref); the index alternative's
# `onset` entry is the incubation mean delay carried from the within-window
# infection to onset. Reading it per draw gives the posterior mean delay with
# its interval, with no manual per-draw `update` loop.
fits = update.(Ref(template), param_draws(template, chain; prefix = :delays))
inc_means = [mean(latent(event(f, :index))).onset for f in fits]
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

here = (mu_inc = summ(post_draws.mu_inc),
    sigma_inc = summ(post_draws.sigma_inc),
    mu_delta = summ(post_draws.mu_delta),
    sigma_delta = summ(post_draws.sigma_delta),
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
    target_upper = [targets[k].upper for k in keys_]);

# Overlay the two interval sets so the overlap reads directly. The four model
# parameters share a comparable scale, so they go on one axis; the incubation
# mean in days is on a separate panel rather than squashing that axis.
function interval_row!(axis, i, lo, hi, m, tlo, thi, tm)
    lines!(axis, [lo, hi], [i + 0.12, i + 0.12]; color = :steelblue,
        linewidth = 6)
    scatter!(axis, [m], [i + 0.12]; color = :steelblue, markersize = 12)
    lines!(axis, [tlo, thi], [i - 0.12, i - 0.12]; color = :firebrick,
        linewidth = 6)
    scatter!(axis, [tm], [i - 0.12]; color = :firebrick, markersize = 12)
end

np = 4
cfig = Figure(size = (760, 380))
ax = Axis(cfig[1, 1]; xlabel = "value",
    yticks = (1:np, comparison.quantity[1:np]),
    title = "Posterior vs published targets")
for i in 1:np
    interval_row!(ax, i, comparison.posterior_lower[i],
        comparison.posterior_upper[i], comparison.posterior_mean[i],
        comparison.target_lower[i], comparison.target_upper[i],
        comparison.target_mean[i])
end
scatter!(ax, [NaN], [NaN]; color = :steelblue, label = "this fit")
scatter!(ax, [NaN], [NaN]; color = :firebrick, label = "published target")
axislegend(ax; position = :rb)

iax = Axis(cfig[1, 2]; xlabel = "days",
    yticks = (1:1, ["incubation mean"]),
    title = "Incubation mean (days)")
interval_row!(iax, 1, comparison.posterior_lower[5],
    comparison.posterior_upper[5], comparison.posterior_mean[5],
    comparison.target_lower[5], comparison.target_upper[5],
    comparison.target_mean[5])
colsize!(cfig.layout, 2, Relative(0.35))
cfig

# The incubation parameters and the incubation mean delay overlap the published
# targets closely in both location and width.
# The transmission-timing parameters overlap too, but their intervals are wide.

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
# A `delta` straddling zero implies a sizeable pre-symptomatic fraction: the
# posterior probability that infection happens before the source's own onset,
# `P(delta < 0)`, read per draw from the fitted transmission-timing Normal.
presymptomatic = [cdf(Normal(post_draws.mu_delta[i],
                          post_draws.sigma_delta[i]), 0.0)
                  for i in eachindex(post_draws.mu_delta)]
presymptomatic_summary = (mean = round(mean(presymptomatic); digits = 2),
    lower = round(quantile(presymptomatic, 0.025); digits = 2),
    upper = round(quantile(presymptomatic, 0.975); digits = 2))

# ## Transmission intensity through the outbreak
#
# The offspring layer was fitted jointly with the delays in the model above, so
# the reproduction number `R(t)` and the dispersion `k` come straight off the
# same `chain`. Every source's expected offspring count was thinned inside that
# fit by the completeness of its own composed delay, so `R(t)` is read under
# real-time completeness rather than at full strength.

# The dispersion `k` and its credible interval, read off the joint `log_k` draws.
k_draws = exp.(vec(chain[:log_k]))
k_summary = (mean = mean(k_draws),
    lower = quantile(k_draws, 0.025), upper = quantile(k_draws, 0.975))

# The weekly `R(t)` is read from the deterministic `log_R` the joint model
# returns, recovered per draw with `generated_quantities`.
rt_gq = generated_quantities(model, chain)
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
# of an epidemic brought under control. The small offspring count (34 sources)
# and the light doc-build sampling budget make this a wide-banded estimate rather
# than a sharp reproduction of the upstream `R(t)`, which reports a descent from
# about 2.3 in the first weeks to roughly 0.4 by week 13 and a dispersion near
# `k = 0.45`. The published values fall inside the credible bands below.
R_draws = exp.(log_R_draws)
rt_median = [median(R_draws[b, :]) for b in 1:n_knots]
thin_idx = round.(Int, range(1, size(R_draws, 2); length = 60))
rfig = Figure(size = (760, 380))
rax = Axis(rfig[1, 1]; xlabel = "week", ylabel = "R(t)",
    title = "Weekly reproduction number")
band!(rax, rt_summary.week, rt_summary.R_lower, rt_summary.R_upper;
    color = (:steelblue, 0.2))
for s in thin_idx
    lines!(rax, rt_summary.week, R_draws[:, s];
        color = (:steelblue, 0.15), linewidth = 1)
end
lines!(rax, rt_summary.week, rt_median; color = :steelblue,
    linewidth = 3)
hlines!(rax, [1.0]; color = :grey, linestyle = :dash)
rfig

# The dispersion and the early/late reproduction number against the published
# targets, with the posterior 95% interval alongside.
rt_compare = DataFrame(
    quantity = ["dispersion k", "R(t) week 1", "R(t) week $(n_knots)"],
    posterior = [round(k_summary.mean; digits = 2),
        rt_summary.R_mean[1], rt_summary.R_mean[end]],
    posterior_lower = [round(k_summary.lower; digits = 2),
        rt_summary.R_lower[1], rt_summary.R_lower[end]],
    posterior_upper = [round(k_summary.upper; digits = 2),
        rt_summary.R_upper[1], rt_summary.R_upper[end]],
    target = [0.45, 2.26, 0.38])

# ## The completeness coupling
#
# The completeness that thinned each offspring term inside the joint fit is the
# CDF of the source's own composed delay `delta + inc` at the window still open
# since its onset. Reading it here off the posterior-mean delays with the same
# `log_completeness` the model uses shows what the
# fit applied: a source seen early in the outbreak has a wide window and is
# essentially complete, so its count is taken near full strength; a source seen
# close to the analysis date has a short window and a small completeness, so its
# rate is thinned the most and its offspring count carries less weight. This is
# the coupling that makes the joint fit a joint fit rather than two separate ones.
inc_post = LogNormal(here.mu_inc.mean, here.sigma_inc.mean)
delta_post = Normal(here.mu_delta.mean, here.sigma_delta.mean)
completeness = exp.(log_completeness(delta_post, inc_post,
    horizon .- source_onset_day))
completeness_by_source = DataFrame(
    source_onset_day = Int.(source_onset_day),
    completeness = round.(completeness; digits = 3))
first(completeness_by_source, 6)

# ## Summary
#
# - The two ways a case enters the model (zoonotic index or human-sourced) are
#   one [`Choose`](@ref) disjunction over named alternatives; the records are
#   pre-sorted into an index batch and a sourced batch and each batch is scored
#   against its named alternative.
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
# - The offspring branching process is fitted jointly with the delays in one
#   model: the delays and the negative-binomial offspring counts share a single
#   parameter block, and each source's expected offspring count is thinned by the
#   completeness of its own composed delay, read off the same sampled object with
#   [`get_dist_recursive`](@ref). The completeness is the day-resolution CDF of
#   the sourced chain `delta + inc`, computed as a mixture over the
#   transmission-timing PMF and the analytic incubation CDF so it carries no
#   quadrature and stays cheap and AD-safe under reverse-mode AD. The joint fit
#   recovers a descending weekly reproduction number and a superspreading
#   dispersion, with the published values inside its (wide, light-budget)
#   credible bands. The upstream model also runs counterfactual outbreak
#   projections from these posteriors, which are left to that analysis.
