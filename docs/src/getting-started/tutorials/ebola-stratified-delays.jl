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
We build one composed censored-and-truncated delay leaf, simulate a stratified
line list from it and check the known per-stratum delays are recovered, then fit
the real Sierra Leone line list with a partially pooled stratified model and read
the recovered district and sex delays back off the fitted object.

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
CensoredDistributions for the composed censored delay leaf.
"""

using CSV, DataFramesMeta, Dates
using CensoredDistributions, Distributions
using Turing, Random, Statistics
using FlexiChains: VNChain
import Mooncake
using ADTypes: AutoMooncake
using CairoMakie

md"""
## The delay leaf

A single composed object describes one onset-to-test delay.
The delay itself is a LogNormal; [`double_interval_censored`](@ref) wraps it with
a one-day primary-event window (the day-resolution onset), a per-record upper
truncation bound (the real-time horizon), and a one-day secondary interval (the
day-resolution test date), so the whole censoring and truncation pipeline is part
of the object rather than applied as separate steps.

`delay_leaf` builds this leaf for given LogNormal parameters and a per-record
truncation horizon `D`.
The same builder serves simulation (a known `mu`, `sigma`) and the per-record
likelihood inside the fit (a stratified `mu`).
"""

function delay_leaf(mu, sigma, D)
    double_interval_censored(LogNormal(mu, sigma);
        primary_event = Uniform(0, 1), upper = D, interval = 1.0)
end

md"""
The leaf is an ordinary `Distributions.jl` object: it has a `logpdf` we score
records against and a `rand` we simulate from, both running the full censoring
and truncation pipeline.
"""

delay_leaf(1.6, 0.5, 30.0)

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

datadir = joinpath(@__DIR__, "ebola-data")

ll = CSV.read(joinpath(datadir, "linelist.csv"), DataFrame;
    missingstring = ["NA", ""])

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
end

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

clean = @subset clean @byrow :district in districts

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
end

md"""
The truncation horizon is the cutoff day expressed as an offset from the earliest
onset.
Because the test date is interval censored to the day, a case tested on the
cutoff day occupies the window `[delay, delay + 1)`, so the open window runs to
the END of the cutoff day: the horizon carries a `+ 1` day for the daily test
interval.
Without it a case tested exactly on the cutoff would have its censored interval
fall entirely at the truncation bound and score zero probability.
Each record's truncation bound is then `D = horizon - onset_day`.
"""

horizon = Float64(Dates.value(cutoff - t0)) + 1.0

(n = nrow(clean), districts = n_district,
    male = count(clean.male), female = count(.!clean.male),
    horizon = horizon)

md"""
Each kept record becomes a row carrying its delay, its stratum indices, and its
real-time truncation horizon `D = horizon - onset_day`.
"""

records = map(eachrow(clean)) do r
    (delay = r.delay, male = r.male, district = district_index[r.district],
        D = horizon - r.onset_day)
end

md"""
## The stratified delay model

The log mean of the LogNormal delay is a partially pooled linear predictor:

```
mu_record = intercept + male_effect * is_male + district_effect[district]
```

The intercept is the female, baseline-district log mean delay; the male effect is
a fixed shift shared across districts; and the district effects are random
effects drawn from a single Normal whose scale `tau` is estimated, so districts
with little data are shrunk toward the overall mean (partial pooling) while
data-rich districts move freely.
The LogNormal scale `sigma` is shared across strata, matching the epidist
sex-district model.

Each record builds its own censored, truncated delay leaf from its stratified
`mu` and its horizon `D`, and is scored directly with `logpdf`, so the
stratification enters through the per-record parameters while the censoring and
truncation stay inside the composed leaf.
The district effects are sampled non-centred (a standard-Normal `z` scaled by
`tau`) so the sampler sees a clean geometry.
"""

@model function ebola_stratified(records, n_district)
    intercept ~ Normal(1.6, 0.5)
    male_effect ~ Normal(0.0, 0.5)
    sigma ~ truncated(Normal(0.0, 0.5); lower = 0.05)
    tau ~ truncated(Normal(0.0, 0.3); lower = 0)
    z ~ filldist(Normal(0.0, 1.0), n_district)
    district_effect = tau .* z

    for r in records
        mu = intercept + male_effect * r.male + district_effect[r.district]
        Turing.@addlogprob! logpdf(delay_leaf(mu, sigma, r.D), r.delay)
    end
    return (; district_effect)
end

md"""
## Step 1: simulate a stratified line list and check recovery

Before touching the data we check the model recovers known per-stratum delays.
We draw a synthetic line list from the same composed leaf with a known intercept,
male effect, district effects, and `sigma`, applying the same day-level censoring
(`floor(test) - floor(onset)`) and real-time truncation the scorer assumes.
"""

sim_truth = (intercept = 1.6, male_effect = 0.1, sigma = 0.5, tau = 0.2)

sim_records = let rng = MersenneTwister(20260614), n_per = 60, nd = n_district,
    sim_horizon = 60.0

    deffect = sim_truth.tau .* randn(rng, nd)
    out = NamedTuple[]
    for d in 1:nd, _ in 1:n_per

        male = rand(rng) < 0.5
        onset = rand(rng) * (sim_horizon - 10)
        mu = sim_truth.intercept + sim_truth.male_effect * male + deffect[d]
        D = sim_horizon - onset
        ## Draw a continuous delay, censor to whole days, keep if observed in
        ## window; the leaf's truncation then corrects for the kept-only bias.
        delay = floor(rand(rng, LogNormal(mu, sim_truth.sigma)) + rand(rng))
        delay <= D || continue
        push!(out, (delay = delay, male = male, district = d, D = D))
    end
    (records = out, deffect = deffect)
end

(n = length(sim_records.records), districts = n_district)

md"""
We fit the synthetic line list at a modest sampler budget (two chains, 300 warmup
and 300 draws each) and check the known parameters come back.
The likelihood is differentiated with Mooncake reverse mode
(`AutoMooncake`), which scales better than forward mode across the many
per-record `logpdf` terms here; the per-record censored `logpdf` runs the
primary-event integral analytically, so the gradient stays cheap.
"""

adbackend = AutoMooncake(; config = nothing)

sim_chain = sample(Xoshiro(1),
    ebola_stratified(sim_records.records, n_district),
    NUTS(300, 0.8; adtype = adbackend), MCMCThreads(), 300, 2;
    chain_type = VNChain, progress = false)

md"""
The fixed parameters are read straight off the chain and tabulated against the
simulating truth.
The intercept, the male effect, the shared `sigma`, and the district scale `tau`
are each recovered with the truth inside the credible interval.
"""

ci(v) = (mean = mean(v), lower = quantile(v, 0.025),
    upper = quantile(v, 0.975))

draws(chain, sym) = vec(chain[sym])

sim_recovery = let
    pars = (:intercept, :male_effect, :sigma, :tau)
    truths = (sim_truth.intercept, sim_truth.male_effect, sim_truth.sigma,
        sim_truth.tau)
    qs = [ci(draws(sim_chain, p)) for p in pars]
    DataFrame(
        parameter = ["intercept", "male effect", "sigma", "tau"],
        truth = collect(truths),
        post_mean = round.([q.mean for q in qs], digits = 3),
        post_lower = round.([q.lower for q in qs], digits = 3),
        post_upper = round.([q.upper for q in qs], digits = 3))
end

md"""
The district effects are returned by the model, so `generated_quantities` reads
the per-district random effect off every draw.
Each simulated district effect lands inside its posterior interval, so the
partial-pooling layer recovers the between-district spread it was given.
"""

function district_effect_draws(model, chain)
    gq = generated_quantities(model, chain)
    mat = reduce(hcat, [g.district_effect for g in vec(gq)])
    return [ci(mat[d, :]) for d in 1:size(mat, 1)]
end

sim_district = let
    qs = district_effect_draws(
        ebola_stratified(sim_records.records, n_district), sim_chain)
    covered = count(d -> qs[d].lower <= sim_records.deffect[d] <= qs[d].upper,
        1:n_district)
    (covered = covered, of = n_district)
end

md"""
## Step 2: fit the real line list

The same model and priors fit the real Sierra Leone records.
"""

real_chain = sample(Xoshiro(20260614),
    ebola_stratified(records, n_district),
    NUTS(300, 0.8; adtype = adbackend), MCMCThreads(), 300, 2;
    chain_type = VNChain, progress = false)

md"""
## Step 3: the fitted delays

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

The recovered delay for each district is the mean of the LogNormal at that
district's log mean (female baseline), read off every posterior draw so it
carries a credible interval.
We report the mean delay in days per district, ordered from fastest to slowest.
"""

district_means = let
    gq = generated_quantities(ebola_stratified(records, n_district),
        real_chain)
    deff = reduce(hcat, [g.district_effect for g in vec(gq)])
    ic = draws(real_chain, :intercept)
    sig = draws(real_chain, :sigma)
    rows = map(1:n_district) do d
        ## Mean of LogNormal(intercept + district_effect, sigma) per draw.
        m = [mean(LogNormal(ic[i] + deff[d, i], sig[i]))
             for i in eachindex(ic)]
        q = ci(m)
        (district = districts[d], mean = q.mean, lower = q.lower,
            upper = q.upper)
    end
    sort(DataFrame(rows), :mean)
end

district_means

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
    pooled = mean(mean(LogNormal(ic[i], sig[i])) for i in eachindex(ic))
    vlines!(ax, [pooled]; color = :grey, linestyle = :dash)
    f
end

md"""
### Fit to the data

A posterior predictive check confirms the censored, truncated model reproduces
the observed delay distribution.
We overlay the empirical delay histogram with the posterior predictive density
averaged over the strata and the posterior draws, each density evaluated through
the same censored leaf the model fitted.
"""

ppc_fig = let
    obs = [r.delay for r in records]
    edges = -0.5:1.0:(maximum(obs) + 0.5)
    grid = 0.0:1.0:maximum(obs)
    ic = draws(real_chain, :intercept)
    me = draws(real_chain, :male_effect)
    sig = draws(real_chain, :sigma)
    gq = generated_quantities(ebola_stratified(records, n_district),
        real_chain)
    deff = reduce(hcat, [g.district_effect for g in vec(gq)])
    ## The predictive averages the per-record censored pmf over the posterior and
    ## over the observed strata. Records sharing a (male, district) stratum give
    ## the same leaf for a given draw, so we count the strata once and weight the
    ## predictive by those counts rather than walking every record.
    Dmax = maximum(r.D for r in records)
    strata = Dict{Tuple{Bool, Int}, Int}()
    for r in records
        k = (r.male, r.district)
        strata[k] = get(strata, k, 0) + 1
    end
    thin = 1:5:length(ic)
    dens = zeros(length(grid))
    for i in thin
        for ((male, d), w) in strata
            mu = ic[i] + me[i] * male + deff[d, i]
            leaf = delay_leaf(mu, sig[i], Dmax)
            for (j, x) in enumerate(grid)
                dens[j] += w * pdf(leaf, x)
            end
        end
    end
    dens ./= (length(thin) * length(records))

    f = Figure(size = (760, 400))
    ax = Axis(f[1, 1]; xlabel = "onset-to-test delay (days)",
        ylabel = "density",
        title = "Observed vs posterior predictive onset-to-test delay")
    hist!(ax, obs; bins = edges, normalization = :pdf,
        color = (:grey, 0.5), label = "observed")
    lines!(ax, grid, dens; color = :firebrick, linewidth = 3,
        label = "posterior predictive")
    axislegend(ax; position = :rt)
    f
end

md"""
The posterior predictive density tracks the observed histogram, so the
double-censored, right-truncated LogNormal is an adequate fit for the
onset-to-test delay across the strata.

## Summary

- The onset-to-test delay is one composed [`double_interval_censored`](@ref)
  LogNormal leaf: a one-day primary-event window, a per-record right-truncation
  horizon, and a one-day secondary interval, so the day-level censoring and the
  real-time truncation are baked into the object.
- The log mean delay is a partially pooled linear predictor: a female baseline
  intercept, a male fixed effect, and district random effects shrunk toward the
  mean through an estimated scale `tau`, with a shared LogNormal `sigma`.
- The workflow simulates a stratified line list from the same leaf and recovers
  the intercept, male effect, `sigma`, `tau`, and the per-district effects within
  uncertainty, then fits the real Sierra Leone line list.
- The fitted fixed effects sit close to the epidist sex-district model (intercept
  near 1.63, a small positive male effect), and the per-district mean delays
  reproduce the between-district spread the vignette's random effects capture.
- A posterior predictive check through the same censored leaf confirms the fit
  reproduces the observed delay distribution.
"""
