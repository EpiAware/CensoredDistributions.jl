md"""
# [Stratified onset-to-test delays in the 2014-2016 Sierra Leone Ebola outbreak](@id ebola-stratified-delays)

## Introduction

The 2014-2016 Ebola virus disease epidemic in Sierra Leone produced one of the
largest line lists of any filovirus outbreak.
This page fits the symptom-onset-to-positive-test delay from that line list and
estimates how it varies by sex and across the fourteen districts, the
stratified delay analysis of the epidist R package
[Ebola vignette](https://epidist.epinowcast.org/articles/ebola.html) rebuilt on
the composed CensoredDistributions.jl stack.

The delay matters operationally: the gap from a case's symptom onset to the day
its sample tests positive is the lag before a confirmed case can be isolated and
its contacts traced, so a longer delay means more onward transmission before
detection.
A single pooled delay hides real heterogeneity.
Districts differed in laboratory access and surveillance intensity over the
epidemic, and a stratified estimate is what tells a responder where testing was
slow.

Two features of the data shape the model.
Both the onset date and the test date are recorded to the day, so each delay is
doubly interval censored: the true onset and the true test time are each a
one-day window rather than an instant.
The line list is also read at a fixed analysis date, so a case only enters once
its positive test has happened by that date; every delay is therefore
right-truncated to the window still open before the analysis horizon, exactly as
in the epidist vignette.

We follow the same Bayesian workflow as the other line-list pages on this site.
We build one composed onset-to-test delay, simulate a stratified line list from
it and check the known per-stratum delays are recovered, then fit the real
Sierra Leone line list with a partially pooled stratified model and read the
recovered district and sex delays back off the fitted object.

How the composer stack is built, scored, and simulated from in general is
covered in the [composer toolkit tutorial](@ref composer-toolkit); here we use
the stack and focus on the delay model and the Bayesian workflow.

### Replication target

The comparison target is the epidist vignette's sex-district model on the same
data.
That model puts a male fixed effect and district random effects on the log
mean of a LogNormal onset-to-test delay.
Its reported posterior is an intercept of 1.63 (95% CrI 1.54, 1.72) on the log
scale, a male effect of 0.04 (95% CrI 0.01, 0.07), and district random effects
spanning roughly Bombali (longest, about +0.26) to Kenema (shortest, about
-0.25).
We reproduce that structure and check our estimates sit in the same place.

## Packages used

We use CSV and DataFramesMeta with Dates for the line-list pipeline, Turing for
inference, FlexiChains for the posterior, CairoMakie for the plots, and
CensoredDistributions for the composed censored delay.
"""

using CSV, DataFramesMeta, Dates
using CensoredDistributions, Distributions
using Turing, Random, Statistics
using DynamicPPL: to_submodel
using FlexiChains: VNChain
import Mooncake
using ADTypes: AutoMooncake
using CairoMakie

md"""
## The delay

A single censored distribution describes one onset-to-test delay.
The delay itself is a LogNormal; [`double_interval_censored`](@ref) wraps it with
a one-day primary-event window (the day-resolution onset) and a one-day secondary
interval (the day-resolution test date), so the day-level censoring is part of
the object rather than applied as a separate step.

The onset-to-test gap is a single delay, so the model scores it as a bare
censored leaf rather than wrapping it in a one-edge chain.
A record supplies the observed gap in a `delay` field and carries its real-time
horizon in the reserved `obs_time` field, so the right-truncation is applied per
record by the package's truncation rather than baked into the object.

`delay_leaf` builds this censored delay for given LogNormal parameters.
The same builder serves simulation (a known `mu`, `sigma`) and the per-stratum
likelihood inside the fit (a stratified `mu`).
"""

function delay_leaf(mu, sigma)
    return double_interval_censored(LogNormal(mu, sigma);
        primary_event = Uniform(0, 1), interval = 1.0)
end

md"""
The template leaf carries the LogNormal delay and its day-level censoring.
"""

template = delay_leaf(1.6, 0.5)

#-

template

md"""
## Data

The bundled line list is the Sierra Leone Ebola line list (n = 8358) collated by
Fang et al. (2016) and shipped in the epidist R package as
`sierra_leone_ebola_data`, redistributed here under epidist's MIT licence (see
the data folder's `README`).
Each record carries a symptom-onset date, a positive-test date, a sex, and one of
fourteen districts.

We read the dates, take the onset-to-test delay in days, and anchor every date to
the earliest onset so the real-time horizon can be expressed as a day offset.
"""

datadir = joinpath(@__DIR__, "data", "ebola")

ll = CSV.read(joinpath(datadir, "linelist.csv"), DataFrame;
    missingstring = ["NA", ""]);

md"""
The epidist vignette restricts to the cases tested up to a fixed analysis date so
the right-truncation is a real, single horizon rather than a formality: a case
only enters the line list once its positive test has happened by that date, so
its delay is right-truncated to the window still open before it.
We use the same `2015-01-31` analysis cutoff.
We keep the records tested on or before the cutoff that carry both dates, a known
sex, and a non-negative delay, then express each onset day as an offset from the
earliest onset.
The sex column has a block of missing entries (about one in eight records); a
delay model stratified by sex has nothing to say about those, so they are
dropped rather than imputed.
"""

cutoff = Date("2015-01-31")

t0 = minimum(skipmissing(ll.date_of_symptom_onset))

dayoff(d) = Float64(Dates.value(d - t0))

clean = @chain ll begin
    @subset @byrow begin
        !ismissing(:date_of_symptom_onset)
        !ismissing(:date_of_sample_tested)
        !ismissing(:sex)
        :date_of_sample_tested >= :date_of_symptom_onset
        :date_of_sample_tested <= cutoff
    end
    @rtransform begin
        :onset_day = dayoff(:date_of_symptom_onset)
        :delay = Float64(Dates.value(:date_of_sample_tested -
                                     :date_of_symptom_onset))
        :male = :sex == "Male"
    end
end;

md"""
A handful of districts carry very few cases; we keep districts with at least
thirty records so each stratum's delay is identifiable, which retains the bulk of
the line list.
"""

district_counts = @chain clean begin
    @groupby(:district)
    @combine(:n = length(:delay))
    @subset(:n .>= 30)
end

districts = sort(district_counts.district)

clean = @subset clean @byrow :district in districts;

district_index = Dict(d => i for (i, d) in enumerate(districts))

n_district = length(districts)

md"""
The kept line list still runs to several thousand records, which is more than a
documentation-build MCMC needs to read the stratified delays.
We take a reproducible stratified subsample, capping each district at ninety
records (districts with fewer are kept whole), so every district stays
represented while the fit stays cheap enough to run in the docs build.
A production analysis would fit the full line list; the subsample is a build-time
convenience, not a modelling choice.
"""

clean = let cap = 90, rng = MersenneTwister(20260614)
    keep = Int[]
    for g in groupby(clean, :district)
        idx = parentindices(g)[1]
        n = length(idx)
        sel = n <= cap ? collect(1:n) : randperm(rng, n)[1:cap]
        append!(keep, idx[sel])
    end
    clean[sort(keep), :]
end;

md"""
The truncation horizon is the cutoff day expressed as an offset from the earliest
onset.
Because the test date is interval censored to the day, a case tested on the
cutoff day occupies the window `[delay, delay + 1)`, so the open window runs to
the end of the cutoff day: the horizon carries a `+ 1` day for the daily test
interval.
Without it a case tested exactly on the cutoff would have its censored interval
fall entirely at the truncation bound and score zero probability.
Each record's truncation bound is then `obs_time = horizon - onset_day`.
"""

horizon = Float64(Dates.value(cutoff - t0)) + 1.0

(n = nrow(clean), districts = n_district,
    male = count(clean.male), female = count(.!clean.male),
    horizon = horizon)

md"""
## Strata and records

The log mean delay varies by a `(male, district)` stratum, so each kept record
carries an integer stratum id that indexes the per-stratum delays the model
builds.
The strata are the observed `(male, district)` combinations; their count is the
number of distinct delays the model reads.
"""

strata = sort(unique([(r.male, district_index[r.district])
                      for r in eachrow(clean)]))

stratum_index = Dict(s => i for (i, s) in enumerate(strata))

n_stratum = length(strata)

md"""
Each kept record becomes a row carrying the observed gap in a `delay` field and
the reserved `obs_time` horizon `horizon - onset_day` that right-truncates the
record to the window still open at the analysis date.
Each record's integer stratum id is held alongside in `group`, which the grouped
scorer uses to pick that record's per-stratum delay.
"""

group = map(eachrow(clean)) do r
    stratum_index[(r.male, district_index[r.district])]
end;

records = map(eachrow(clean)) do r
    (delay = r.delay, obs_time = horizon - r.onset_day)
end;

md"""
## The stratified delay model

The log mean of the LogNormal delay is a partially pooled linear predictor:

```
mu_stratum = intercept + male_effect * is_male + district_effect[district]
```

The intercept is the female, baseline-district log mean delay; the male effect is
a fixed shift shared across districts; and the district effects are random
effects drawn from a single Normal whose scale `tau` is estimated, so districts
with little data are shrunk toward the overall mean (partial pooling) while
data-rich districts move freely.
The LogNormal scale `sigma` is shared across strata, matching the epidist
sex-district model.

The model samples the predictor coefficients and the shared `sigma`, then builds
one censored delay leaf per `(male, district)` stratum from that stratum's `mu`
(and the shared `sigma`) with `delay_leaf`.
The whole record table scores in one `~` through the grouped
[`composed_distribution_model`](@ref)`(ds, records; group)`: `ds` is the vector
of per-stratum delay leaves and `group` the integer stratum id per record, so
each record scores under its own stratum's leaf while the day-level censoring
and the per-record right-truncation (the reserved `obs_time` horizon) stay
inside the scorer.
The district effects are sampled non-centred (a standard-Normal `z` scaled by
`tau`) so the sampler sees a clean geometry.
"""

@model function ebola_stratified(records, group, strata)
    intercept ~ Normal(1.6, 0.5)
    male_effect ~ Normal(0.0, 0.5)
    sigma ~ truncated(Normal(0.0, 0.5); lower = 0.05)
    tau ~ truncated(Normal(0.0, 0.3); lower = 0)
    z ~ filldist(Normal(0.0, 1.0), n_district)
    district_effect = tau .* z

    ## One censored delay leaf per (male, district) stratum, built from that
    ## stratum's partially pooled log mean and the shared sigma.
    ds = map(strata) do (male, d)
        mu = intercept + male_effect * male + district_effect[d]
        delay_leaf(mu, sigma)
    end

    obs ~ to_submodel(composed_distribution_model(ds, records; group = group))
    return (; district_effect)
end

md"""
## Simulate a stratified line list and check recovery

Before touching the data we check the model recovers known per-stratum delays.
We draw a synthetic line list from the same censored leaf with a known
intercept, male effect, district effects, and `sigma`, applying the same
day-level censoring and real-time truncation the scorer assumes.
The primary-event time is added before flooring (`floor(delay + rand())`), so the
synthetic delay carries the same one-day primary-event window as the secondary
test-day interval, matching the double censoring the leaf encodes.
"""

sim_truth = (intercept = 1.6, male_effect = 0.1, sigma = 0.5, tau = 0.2)

sim_setup = let rng = MersenneTwister(20260614), n_per = 60, nd = n_district,
    sim_horizon = 60.0

    deffect = sim_truth.tau .* randn(rng, nd)
    out = NamedTuple[]
    keys = Tuple{Bool, Int}[]
    for d in 1:nd, _ in 1:n_per

        male = rand(rng) < 0.5
        onset = rand(rng) * (sim_horizon - 10)
        mu = sim_truth.intercept + sim_truth.male_effect * male + deffect[d]
        D = sim_horizon - onset
        ## Draw a continuous delay, censor to whole days, keep if observed in
        ## window; the leaf's truncation then corrects for the kept-only bias.
        delay = floor(rand(rng, LogNormal(mu, sim_truth.sigma)) + rand(rng))
        delay <= D || continue
        push!(out, (delay = delay, D = D))
        push!(keys, (male, d))
    end
    (rows = out, keys = keys, deffect = deffect)
end;

md"""
The synthetic strata are the observed `(male, district)` combinations of the
simulated line list, and each simulated record carries its own stratum id and
real-time horizon, exactly as the real records do.
"""

sim_strata = sort(unique(sim_setup.keys))

sim_stratum_index = Dict(s => i for (i, s) in enumerate(sim_strata))

sim_group = [sim_stratum_index[k] for k in sim_setup.keys];

sim_records = map(sim_setup.rows) do r
    (delay = r.delay, obs_time = r.D)
end;

(n = length(sim_records), districts = n_district, strata = length(sim_strata))

md"""
We fit the synthetic line list at a modest sampler budget (two chains, 200 warmup
and 200 draws each) with Mooncake reverse-mode AD, and check the known
parameters come back.
"""

adbackend = AutoMooncake(; config = nothing)

sim_chain = sample(Xoshiro(1),
    ebola_stratified(sim_records, sim_group, sim_strata),
    NUTS(200, 0.8; adtype = adbackend), MCMCThreads(), 200, 2;
    chain_type = VNChain, progress = false);

md"""
The fixed parameters are read straight off the chain and shown against the
simulating truth.
The plot draws each parameter's posterior as a spread of draws with its credible
interval, and marks the simulating truth; the intercept, the male effect, the
shared `sigma`, and the district scale `tau` each sit with the truth inside the
interval.
"""

ci(v) = (mean = mean(v), lower = quantile(v, 0.025),
    upper = quantile(v, 0.975))

draws(chain, sym) = vec(chain[sym])

sim_recovery_fig = let
    pars = (:intercept, :male_effect, :sigma, :tau)
    labels = ["intercept", "male effect", "sigma", "tau"]
    truths = (sim_truth.intercept, sim_truth.male_effect, sim_truth.sigma,
        sim_truth.tau)
    rng = MersenneTwister(1)
    f = Figure(size = (760, 360))
    for (k, p) in enumerate(pars)
        v = draws(sim_chain, p)
        q = ci(v)
        ax = Axis(f[fldmod1(k, 2)...]; title = labels[k])
        jitter = 0.04 .* randn(rng, length(v))
        scatter!(ax, v, jitter; color = (:steelblue, 0.25), markersize = 4)
        lines!(ax, [q.lower, q.upper], [0, 0]; color = :navy, linewidth = 4)
        scatter!(ax, [q.mean], [0]; color = :navy, markersize = 12)
        vlines!(ax, [truths[k]]; color = :firebrick, linewidth = 2)
        hideydecorations!(ax)
    end
    f
end

md"""
The district effects are returned by the model, so `generated_quantities` reads
the per-district random effect off every draw.
The plot shows each district's posterior interval against the simulating truth;
every simulated district effect lands inside its interval, so the
partial-pooling layer recovers the between-district spread it was given.
"""

function district_effect_mat(model, chain)
    gq = generated_quantities(model, chain)
    return reduce(hcat, [g.district_effect for g in vec(gq)])
end

sim_district_fig = let
    mat = district_effect_mat(
        ebola_stratified(sim_records, sim_group, sim_strata), sim_chain)
    qs = [ci(mat[d, :]) for d in 1:n_district]
    f = Figure(size = (760, 460))
    ax = Axis(f[1, 1]; xlabel = "district effect (log scale)",
        ylabel = "district index",
        title = "Recovered district effects vs simulating truth")
    for d in 1:n_district
        lines!(ax, [qs[d].lower, qs[d].upper], [d, d]; color = :steelblue,
            linewidth = 5)
        scatter!(ax, [qs[d].mean], [d]; color = :steelblue, markersize = 9)
    end
    scatter!(ax, sim_setup.deffect, 1:n_district; color = :firebrick,
        markersize = 11, marker = :diamond, label = "truth")
    axislegend(ax; position = :rt)
    f
end

md"""
## Fit the real line list

The same model and priors fit the real Sierra Leone records.
"""

real_chain = sample(Xoshiro(20260614),
    ebola_stratified(records, group, strata),
    NUTS(200, 0.8; adtype = adbackend), MCMCThreads(), 200, 2;
    chain_type = VNChain, progress = false);

md"""
## The fitted delays

### Fixed effects against the epidist target

The intercept and male effect are read off the posterior and compared with the
epidist sex-district model.
The intercept is the female baseline-district log mean delay and the male effect
is the male log shift; both are matched against the vignette's reported
posterior.
"""

target = (intercept = (mean = 1.63, lower = 1.54, upper = 1.72),
    male_effect = (mean = 0.04, lower = 0.01, upper = 0.07))

fixed_effects = let
    ic = ci(draws(real_chain, :intercept))
    me = ci(draws(real_chain, :male_effect))
    DataFrame(
        parameter = ["intercept (log mean)", "male effect (log)"],
        post_mean = round.([ic.mean, me.mean], digits = 3),
        post_lower = round.([ic.lower, me.lower], digits = 3),
        post_upper = round.([ic.upper, me.upper], digits = 3),
        target_mean = [target.intercept.mean, target.male_effect.mean],
        target_lower = [target.intercept.lower, target.male_effect.lower],
        target_upper = [target.intercept.upper, target.male_effect.upper])
end

md"""
The intercept sits close to the epidist value and the male effect is a small
positive shift in the same region, so men are estimated to wait slightly longer
from onset to test than women, the direction the vignette reports.

### Per-district mean delays

The recovered delay for each district is read off the fitted censored leaf.
For each posterior draw and district we rebuild the template with
[`update`](@ref) at that draw's intercept plus the district's random effect,
pull the district delay's inner LogNormal off the leaf with
[`free_leaf`](@ref CensoredDistributions.free_leaf), and take its mean, so the
reported delay comes straight from the composed object rather than from a
hand-built distribution.
We report the mean delay in days per district, ordered from fastest to slowest.
"""

function district_mean_draws(chain)
    ic = draws(chain, :intercept)
    sig = draws(chain, :sigma)
    gq = generated_quantities(
        ebola_stratified(records, group, strata), chain)
    deff = reduce(hcat, [g.district_effect for g in vec(gq)])
    means = Matrix{Float64}(undef, n_district, length(ic))
    for i in eachindex(ic)
        for d in 1:n_district
            ## Baseline-district delay: the female intercept plus this district's
            ## random effect, read through the censored leaf's LogNormal.
            leaf = CensoredDistributions.free_leaf(
                update(template, (mu = ic[i] + deff[d, i], sigma = sig[i])))
            means[d, i] = mean(leaf)
        end
    end
    return means
end

district_means = let means = district_mean_draws(real_chain)
    rows = map(1:n_district) do d
        q = ci(means[d, :])
        (district = districts[d], mean = q.mean, lower = q.lower,
            upper = q.upper)
    end
    sort(DataFrame(rows), :mean)
end;

md"""
The fastest and slowest districts differ by a few days in mean onset-to-test
delay, the same between-district spread the epidist random effects capture.
The interval plot shows each district's mean delay with its credible interval,
ordered fastest to slowest.
"""

district_fig = let df = district_means
    f = Figure(size = (760, 460))
    ax = Axis(f[1, 1]; xlabel = "mean onset-to-test delay (days)",
        yticks = (1:nrow(df), df.district),
        title = "Per-district onset-to-test delay (female baseline)")
    for (k, r) in enumerate(eachrow(df))
        lines!(ax, [r.lower, r.upper], [k, k]; color = :steelblue,
            linewidth = 6)
        scatter!(ax, [r.mean], [k]; color = :steelblue, markersize = 11)
    end
    ic = draws(real_chain, :intercept)
    sig = draws(real_chain, :sigma)
    pooled = mean(
        mean(CensoredDistributions.free_leaf(
            update(template, (mu = ic[i], sigma = sig[i]))))
    for i in eachindex(ic))
    vlines!(ax, [pooled]; color = :grey, linestyle = :dash)
    f
end

md"""
### Fit to the data

A posterior predictive check confirms the censored, truncated model reproduces
the observed delay distribution.
We overlay the empirical delay histogram with an ensemble of posterior
predictive densities, one per thinned posterior draw, each evaluated through the
same censored leaf the model fitted and averaged over the observed strata.
The ensemble shows a ribbon (its 5th-95th percentile across draws) and a sample
of individual draw curves, so the spread of the predictive is visible rather
than a single mean line.
"""

ppc_fig = let
    obs = [r.delay for r in records]
    edges = -0.5:1.0:(maximum(obs) + 0.5)
    grid = 0.0:1.0:maximum(obs)
    ic = draws(real_chain, :intercept)
    me = draws(real_chain, :male_effect)
    sig = draws(real_chain, :sigma)
    gq = generated_quantities(
        ebola_stratified(records, group, strata), real_chain)
    deff = reduce(hcat, [g.district_effect for g in vec(gq)])
    ## Each thinned draw gives one predictive density: the per-record censored
    ## pmf averaged over the observed strata. Records sharing a (male, district)
    ## stratum give the same delay for a given draw, so we count the strata once
    ## and weight by those counts rather than walking every record. The
    ## per-record truncation horizon enters through the maximal `obs_time`.
    Dmax = maximum(r.obs_time for r in records)
    stratum_counts = Dict{Int, Int}()
    for sid in group
        stratum_counts[sid] = get(stratum_counts, sid, 0) + 1
    end
    thin = 1:5:length(ic)
    ## One predictive density curve per posterior draw (columns = draws).
    curves = zeros(length(grid), length(thin))
    for (col, i) in enumerate(thin)
        for (sid, w) in stratum_counts
            (male, d) = strata[sid]
            mu = ic[i] + me[i] * male + deff[d, i]
            leaf = double_interval_censored(LogNormal(mu, sig[i]);
                primary_event = Uniform(0, 1), upper = Dmax, interval = 1.0)
            for (j, x) in enumerate(grid)
                curves[j, col] += w * pdf(leaf, x)
            end
        end
    end
    curves ./= length(records)
    lo = [quantile(curves[j, :], 0.05) for j in eachindex(grid)]
    hi = [quantile(curves[j, :], 0.95) for j in eachindex(grid)]

    f = Figure(size = (760, 400))
    ax = Axis(f[1, 1]; xlabel = "onset-to-test delay (days)",
        ylabel = "density",
        title = "Observed vs posterior predictive onset-to-test delay")
    hist!(ax, obs; bins = edges, normalization = :pdf,
        color = (:grey, 0.5), label = "observed")
    band!(ax, grid, lo, hi; color = (:firebrick, 0.25))
    for col in 1:min(20, size(curves, 2))
        lines!(ax, grid, curves[:, col]; color = (:firebrick, 0.2),
            linewidth = 1)
    end
    lines!(ax, Float64[], Float64[]; color = :firebrick, linewidth = 3,
        label = "posterior predictive draws")
    axislegend(ax; position = :rt)
    f
end

md"""
The posterior predictive ensemble brackets the observed histogram, so the
double-censored, right-truncated LogNormal is an adequate fit for the
onset-to-test delay across the strata.

## Summary

- The onset-to-test delay is one [`double_interval_censored`](@ref) LogNormal
  leaf: a one-day primary-event window and a one-day secondary interval, with the
  per-record right-truncation horizon supplied as the reserved `obs_time` row
  field, so the day-level censoring and the real-time truncation are handled by
  the composed object.
- The log mean delay is a partially pooled linear predictor: a female baseline
  intercept, a male fixed effect, and district random effects shrunk toward the
  mean through an estimated scale `tau`, with a shared LogNormal `sigma`.
- One censored delay leaf is built per `(male, district)` stratum with
  `delay_leaf`, and the whole record table scores in one `~` through the grouped
  [`composed_distribution_model`](@ref)`(ds, records; group)`, so the
  stratification enters through the integer stratum id while the censoring and
  truncation stay inside the scorer.
- The workflow simulates a stratified line list from the same leaf and recovers
  the intercept, male effect, `sigma`, `tau`, and the per-district effects within
  uncertainty, then fits the real Sierra Leone line list.
- The fitted delays are read back off the composed object with [`update`](@ref)
  and [`free_leaf`](@ref CensoredDistributions.free_leaf): the fixed effects sit
  close to the epidist sex-district model (intercept near 1.63, a small positive
  male effect), and the per-district mean delays reproduce the between-district
  spread the vignette's random effects capture.
- A posterior predictive check through the same censored leaf confirms the fit
  reproduces the observed delay distribution.
"""
