md"""
# Replicating the Bundibugyo Ebola delay model

## Introduction

This walkthrough rebuilds the delay model from the
[`epiforecasts/bdbv-linelist-analysis`](https://github.com/epiforecasts/bdbv-linelist-analysis)
study of the 2012 Isiro Bundibugyo Ebola (BDBV) outbreak, using
`CensoredDistributions.jl` and its Turing extension.
The aim is to show, component by component, how the package expresses the
pieces of a real line-list delay analysis, then to verify the expression by
simulating from known delays and recovering them.

The original analysis fits four atomic delay components at the day level:

- `d_oa` onset to admission,
- `d_ad` admission to death,
- `d_ac` admission to discharge,
- `d_on` onset to notification.

Each is doubly interval censored (the primary event occurs uniformly within a
day and the observed delay is recorded to the nearest day).
Around the delays the study layers stratification by health-care worker (HCW)
status, a death-pathway length-of-stay mixture, convolved onset to death
marginals, count weighting on repeated day-level delays, and a case-fatality
ratio (CFR) regression.

We map each of these onto the package below.
The mapping follows the component table in
[issue #322](https://github.com/EpiAware/CensoredDistributions.jl/issues/322):
the delay components, strata, mixture, convolution and weighting are all
expressed by the package, while the CFR stays as plain Turing code because it
is a covariate generalised linear model rather than a censoring construct.

### What stays outside the grammar

Two pieces of the original study are deliberately not package constructs.

The CFR is a logistic regression of outcome on covariates (HCW status, case
definition, standardised age).
It is a covariate generalised linear model, so it stays in user Turing code
with a plain `outcome ~ Bernoulli(logistic(Xβ))` term, sitting alongside the
delay model rather than inside it.

The original study also prototyped a shared-origin joint tree that linked the
delays through per-case latent onset and admission times.
That variant was abandoned because the No-U-turn sampler hit wedge-shaped
boundary geometry at the same-day-admission cases and produced hundreds of
divergent transitions.
We mention it for completeness but do not build it: the marginal formulation
used here is what the package supports and what Charniga et al. (2024)
recommend for retrospective complete-outbreak data.

## Packages used

We use Turing for probabilistic programming, DynamicPPL for the submodel
plumbing, Distributions for the delay families, FlexiChains for the posterior,
DataFramesMeta for the count aggregation, and Random for reproducibility.
We sample with ForwardDiff, matching the original analysis.
"""

using Turing
using DynamicPPL
using Distributions
using DataFramesMeta
using Random
using StatsBase
using FlexiChains
using FlexiChains: Prefixed
using CensoredDistributions
using ADTypes: AutoForwardDiff

md"""
## A delay family with a shared parametrisation

The original analysis supports three families (LogNormal, Gamma, Weibull) under
a common log-mean / log-shape parametrisation so the priors are comparable.
We follow the same idea but keep to the Gamma family, which is the model
comparison winner in the study, to keep the walkthrough short.

`build_delay_dist` maps a sampled log-mean and log-shape onto a Gamma
distribution with that mean.
"""

function build_delay_dist(log_mean, log_shape)
    shape = exp(log_shape)
    return Gamma(shape, exp(log_mean) / shape)
end

md"""
A small prior submodel samples the log-mean and log-shape and returns the
constructed delay distribution.
Using a submodel means the sample names are prefixed by the left-hand-side
variable when the submodel is included, so the four delay components stay
distinct in the posterior.
"""

@model function delay_prior(mean_loc)
    log_mean ~ Normal(mean_loc, 1.0)
    log_shape ~ Normal(0.0, 1.0)
    return build_delay_dist(log_mean, log_shape)
end

md"""
## Doubly interval censored delays

Each atomic delay is doubly interval censored.
The package builds this from a continuous delay distribution with
`double_interval_censored`: the default `primary_event = Uniform(0, 1)` places
the primary event uniformly within a one-day window, and `interval = 1.0`
records the delay to the nearest day.

```julia
double_interval_censored(Gamma(2.0, 1.5); interval = 1.0)
```

The resulting object is a univariate distribution whose `logpdf` integrates the
primary event out internally (the marginal default), so it scores natively
through a `~` statement or through the package's submodels.

## Count and multiplicity weighting

Day-level censoring means many cases share the same delay value.
The original analysis compresses each observation vector into its unique values
and their integer multiplicities, so each unique delay needs only one `logpdf`
call per evaluation, weighted by its count.

The package expresses this with the `weight` distribution wrapper:
`logpdf(weight(d, w), y) == w * logpdf(d, y)`.
Because `weight(d, w)` is itself a distribution, it drops into a `~` statement,
and it is exactly the `weight` keyword the censored submodels accept.

The helper below aggregates a delay vector into uniques and counts.
"""

function unique_counts(v)
    counts = Dict{eltype(v), Int}()
    for x in v
        counts[x] = get(counts, x, 0) + 1
    end
    u = collect(keys(counts))
    return u, [counts[x] for x in u]
end

md"""
## Stratification by health-care worker status

The original study fits HCW and non-HCW cases with a shared shape but a
log-mean shift for HCWs.
There is no special construct for this: the stratum loop is plain user code
that passes per-group parameters into the same `double_interval_censored`
builder.

We express both the per-delay submodel scoring and the strata in one model.
The model below fits the four atomic delays, applies an HCW log-mean shift to
each, and scores each stratum's count-weighted delays through
`double_interval_censored_model` (the package submodel for the marginal
double-censored pipeline).
"""

@model function bdbv_delays(data)
    ## Four atomic delay components, each a prior submodel returning a Gamma.
    ## The submodel prefixes the parameter names with the left-hand side, so
    ## the four components stay distinct (`dist_oa.log_mean`, and so on).
    dist_oa ~ to_submodel(delay_prior(log(3.0)))
    dist_ad ~ to_submodel(delay_prior(log(6.0)))
    dist_ac ~ to_submodel(delay_prior(log(13.0)))
    dist_on ~ to_submodel(delay_prior(log(7.0)))

    ## HCW log-mean shifts: HCW cases share the shape but shift the mean.
    β_oa_hcw ~ Normal(0.0, 0.5)
    β_ad_hcw ~ Normal(0.0, 0.5)
    β_ac_hcw ~ Normal(0.0, 0.5)
    β_on_hcw ~ Normal(0.0, 0.5)

    ## Stratum loop: one entry per (component, distribution, shift, subset).
    ## The non-HCW subset uses the baseline distribution; the HCW subset
    ## rebuilds the distribution with the shifted log-mean. Both are scored
    ## with count-weighted day-level delays through the package submodel.
    strata = (
        (:oa, dist_oa, β_oa_hcw, data.oa_n, data.oa_h),
        (:ad, dist_ad, β_ad_hcw, data.ad_n, data.ad_h),
        (:ac, dist_ac, β_ac_hcw, data.ac_n, data.ac_h),
        (:on, dist_on, β_on_hcw, data.on_n, data.on_h)
    )

    for (name, dist, β_hcw, obs_n, obs_h) in strata
        ## Non-HCW stratum, baseline distribution.
        if !isempty(obs_n)
            u, c = unique_counts(obs_n)
            dic = double_interval_censored(dist; interval = 1.0)
            for i in eachindex(u)
                y ~ to_submodel(
                    prefix(
                        double_interval_censored_model(
                            dic, u[i]; weight = c[i]),
                        Symbol(name, "_n_", i)),
                    false)
            end
        end
        ## HCW stratum, mean shifted by the HCW coefficient. We rebuild the
        ## Gamma from the shifted log-mean and the shared log-shape.
        if !isempty(obs_h)
            shifted = build_delay_dist(
                log(mean(dist)) + β_hcw, log(shape(dist)))
            u, c = unique_counts(obs_h)
            dic = double_interval_censored(shifted; interval = 1.0)
            for i in eachindex(u)
                y ~ to_submodel(
                    prefix(
                        double_interval_censored_model(
                            dic, u[i]; weight = c[i]),
                        Symbol(name, "_h_", i)),
                    false)
            end
        end
    end
end

md"""
## Simulate a line list

To verify the expression we simulate day-level delays from known Gamma
distributions, with an HCW mean shift, then fit and check we recover the
generating parameters.

The true means roughly match the study's anchors: a short onset to admission,
a longer admission to death and admission to discharge, and a notification
delay with a heavier tail.
"""

Random.seed!(20260519)

## True generating parameters (log scale).
truth = (
    oa = (log_mean = log(3.0), log_shape = log(2.0)),
    ad = (log_mean = log(6.0), log_shape = log(2.0)),
    ac = (log_mean = log(13.0), log_shape = log(1.5)),
    on = (log_mean = log(7.0), log_shape = log(1.2))
)

## HCW shifts on the log-mean.
hcw_shift = (oa = -0.3, ad = -0.1, ac = -0.1, on = -0.2)

md"""
We draw continuous delays from each Gamma, add a uniform primary-event offset
within the day, then floor to whole days to mimic day-level interval censoring.
"""

function simulate_delays(rng, p, n; shift = 0.0)
    base = build_delay_dist(p.log_mean + shift, p.log_shape)
    primary = rand(rng, Uniform(0, 1), n)
    delays = rand(rng, base, n) .+ primary
    return floor.(delays)
end

rng = Random.MersenneTwister(20260519)

n_n = 60   # non-HCW cases per component
n_h = 25   # HCW cases per component

sim = (
    oa_n = simulate_delays(rng, truth.oa, n_n),
    oa_h = simulate_delays(rng, truth.oa, n_h; shift = hcw_shift.oa),
    ad_n = simulate_delays(rng, truth.ad, n_n),
    ad_h = simulate_delays(rng, truth.ad, n_h; shift = hcw_shift.ad),
    ac_n = simulate_delays(rng, truth.ac, n_n),
    ac_h = simulate_delays(rng, truth.ac, n_h; shift = hcw_shift.ac),
    on_n = simulate_delays(rng, truth.on, n_n),
    on_h = simulate_delays(rng, truth.on, n_h; shift = hcw_shift.on)
)

md"""
A quick look at the simulated onset to admission delays for the two strata:
"""

(non_hcw_mean = mean(sim.oa_n), hcw_mean = mean(sim.oa_h))

md"""
## Fit and recover

We fit the delay model with the No-U-turn sampler and ForwardDiff, then check
the posterior log-means against the truth.
"""

delay_fit = sample(
    bdbv_delays(sim),
    NUTS(; adtype = AutoForwardDiff()), MCMCThreads(), 1000, 4;
    chain_type = VNChain
)

md"""
Compare the posterior log-mean for each component against the value used to
simulate.
The means should be close and the credible intervals should cover the truth.
"""

function recovery_table(chn, truth)
    rows = NamedTuple[]
    for (name, p) in pairs(truth)
        vn = Prefixed(Symbol("dist_", name, ".log_mean"))
        draws = vec(chn[vn])
        push!(rows,
            (
                component = name,
                truth = round(p.log_mean; digits = 3),
                post_mean = round(mean(draws); digits = 3),
                q025 = round(quantile(draws, 0.025); digits = 3),
                q975 = round(quantile(draws, 0.975); digits = 3)))
    end
    return DataFrame(rows)
end

recovery_table(delay_fit, truth)

md"""
The HCW shifts are recovered alongside the baselines:
"""

DataFrame(
    component = [:oa, :ad, :ac, :on],
    truth = collect(values(hcw_shift)),
    post_mean = [round(mean(vec(delay_fit[Prefixed(Symbol("β_", c, "_hcw"))]));
                     digits = 3)
                 for c in (:oa, :ad, :ac, :on)]
)

md"""
## The death-pathway mixture

After admission a case resolves to one of two competing outcomes: death (with
probability equal to the in-hospital fatality among admitted cases) or
discharge.
The original analysis builds the in-hospital length-of-stay marginal as a
per-draw mixture of the admission to death and admission to discharge delays,
weighted by the fatality probability.

The package expresses this directly as a `Competing` node, which lowers to a
`MixtureModel` over the branch delays weighted by the branch probabilities.
We build it from posterior mean delays and a fatality probability, then read
off the marginal length of stay.
"""

## Posterior mean Gamma for a component from the delay fit.
function fitted_gamma(chn, name)
    log_mean = mean(vec(chn[Prefixed(Symbol("dist_", name, ".log_mean"))]))
    log_shape = mean(vec(chn[Prefixed(Symbol("dist_", name, ".log_shape"))]))
    return build_delay_dist(log_mean, log_shape)
end

d_ad = fitted_gamma(delay_fit, :ad)   # admission to death
d_ac = fitted_gamma(delay_fit, :ac)   # admission to discharge

## In-hospital fatality among admitted cases.
p_die = 0.4

los_node = Competing(
    :death => (d_ad, p_die),
    :discharge => (d_ac, 1 - p_die))

los_mixture = as_mixture(los_node)
(los_mean = mean(los_mixture), los_median = median(los_mixture))

md"""
## Convolved onset to death marginal

The natural-history identity onset to death = (onset to admission) followed by
(admission to death) is recovered by convolving the two atomic components.
The package provides `convolve_distributions` for this; for a longer chain of
events `sequential_distribution` builds the same composition across several
delays at once.
"""

d_oa = fitted_gamma(delay_fit, :oa)   # onset to admission

onset_to_death = convolve_distributions(d_oa, d_ad)
(onset_to_death_mean = mean(onset_to_death),)

md"""
The onset to discharge marginal is the same construction with the discharge
delay:
"""

onset_to_discharge = convolve_distributions(d_oa, d_ac)
(onset_to_discharge_mean = mean(onset_to_discharge),)

md"""
A longer pathway can be assembled in one call.
`sequential_distribution` takes the ordered list of delays and represents the
whole chain, marginalising any unobserved intermediate events when scored
against an observation vector.
"""

chain = sequential_distribution([d_oa, d_ad])
chain isa CensoredDistributions.SequentialDistribution

md"""
## The case-fatality ratio stays in plain Turing

The CFR is a covariate logistic regression, not a censoring construct, so it is
not part of the package grammar (see issue #322).
It lives in user Turing code as a plain `Bernoulli` with a logistic link on the
covariates, sitting alongside the delay model.

Here it is on its own, fitted to simulated outcomes, to show the separation:
the delay model above and this block share no latent variables, exactly as in
the original study.
"""

## Numerically safe logistic, pinned away from {0, 1} for the Bernoulli domain.
safe_logistic(x) = clamp(inv(1 + exp(-x)), 1e-10, 1.0 - 1e-10)

@model function cfr_model(hcw, probable, age_z, outcome)
    β_0 ~ Normal(0.0, 2.0)
    β_hcw ~ Normal(0.0, 1.0)
    β_def ~ Normal(0.0, 1.0)
    β_age ~ Normal(0.0, 1.0)
    for i in eachindex(outcome)
        η = β_0 + β_hcw * hcw[i] + β_def * probable[i] + β_age * age_z[i]
        outcome[i] ~ Bernoulli(safe_logistic(η))
    end
end

md"""
Simulate covariates and outcomes from known coefficients, then fit:
"""

n_cfr = 200
hcw = rand(rng, Bernoulli(0.3), n_cfr) .* 1.0
probable = rand(rng, Bernoulli(0.4), n_cfr) .* 1.0
age_z = randn(rng, n_cfr)

cfr_truth = (β_0 = 0.2, β_hcw = -0.8, β_def = 0.5, β_age = 0.3)
η_true = cfr_truth.β_0 .+ cfr_truth.β_hcw .* hcw .+
         cfr_truth.β_def .* probable .+ cfr_truth.β_age .* age_z
outcome = rand.(rng, Bernoulli.(safe_logistic.(η_true)))

cfr_fit = sample(
    cfr_model(hcw, probable, age_z, outcome),
    NUTS(; adtype = AutoForwardDiff()), 1000;
    chain_type = VNChain
)

DataFrame(
    coefficient = [:β_0, :β_hcw, :β_def, :β_age],
    truth = collect(values(cfr_truth)),
    post_mean = [round(mean(vec(cfr_fit[Prefixed(c)])); digits = 3)
                 for c in (:β_0, :β_hcw, :β_def, :β_age)]
)

md"""
## Switching one delay to the latent formulation

The marginal default integrates the primary event out inside `logpdf`.
The same model can keep the primary event as a sampler-owned latent instead, by
flipping `mode = Latent()` on the `primary_censored` build; nothing else in the
model body changes, because the submodel dispatches on the resolved mode.

Below we score the onset to admission delays the latent way.
Each observation carries its own latent primary event, so the latent path
vectorises over records rather than weighting aggregated counts.
We fit on a small subset to keep the build quick.
"""

@model function latent_oa(ys)
    dist ~ to_submodel(delay_prior(log(3.0)))
    for i in eachindex(ys)
        d = primary_censored(dist, Uniform(0, 1); mode = Latent())
        y ~ to_submodel(
            prefix(primary_censored_model(d, ys[i]), Symbol("y_", i)),
            false)
    end
end

latent_subset = sim.oa_n[1:30] .+ 1e-6

latent_fit = sample(
    latent_oa(latent_subset),
    NUTS(; adtype = AutoForwardDiff()), 500;
    chain_type = VNChain
)

(
    latent_post_log_mean = round(
        mean(vec(latent_fit[Prefixed(@varname(dist.log_mean))]));
        digits = 3),
    truth = round(truth.oa.log_mean; digits = 3))

md"""
The latent fit recovers the same onset to admission log-mean as the marginal
default, demonstrating the marginal-versus-latent equivalence: switching
between them needs only the `mode` flag on the distribution.

## Summary

We rebuilt the BDBV delay model component by component:

- the four atomic delays via `double_interval_censored` and the marginal
  `double_interval_censored_model` submodel;
- HCW strata via a plain user loop passing per-group parameters;
- the death-pathway length-of-stay mixture as a `Competing` node lowering to a
  `MixtureModel`;
- the onset to death and onset to discharge marginals via
  `convolve_distributions`, with `sequential_distribution` for longer chains;
- count and multiplicity weighting through the `weight` distribution wrapper,
  passed as the submodel `weight` keyword;
- the CFR as a plain Turing `Bernoulli(logistic(Xβ))` term alongside the delay
  model, kept out of the grammar because it is a covariate regression.

The simulate-and-recover exercise confirms the package expression recovers the
generating delay parameters and HCW shifts, and the `Latent()` switch shows the
single-flag change between the marginal default and the latent formulation.
"""
