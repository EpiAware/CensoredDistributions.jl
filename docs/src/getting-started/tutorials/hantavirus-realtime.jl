md"""
# Real-time hantavirus delays: a per-record censoring walkthrough

## Introduction

This walkthrough rebuilds the **delay, truncation, and ascertainment layer**
of the Andes hantavirus (ANDV) real-time transmission model
([epiforecasts/andv-linelist-analysis](https://github.com/epiforecasts/andv-linelist-analysis))
using `CensoredDistributions.jl` and its Turing extension.
The aim is the *per-record loop* pattern: the user writes an ordinary Turing
model that loops over line-list records, and every censoring, truncation, and
completeness term comes from a package primitive rather than a hand-rolled
`@addlogprob!`.

The ANDV outbreak is a small cluster with a known transmission tree.
Each record is either an **index case** (zoonotic, no human source) or a
**sourced case** (infected by another line-list case).
At a real-time cut-off the observed delays are right-truncated, and offspring
counts are incomplete because some onward chains have not yet completed.
We reproduce three pieces of that model:

1. **Per-record right-truncation.** Index cases observe a single delay
   (incubation), so their truncation denominator is the single-delay CDF.
   Sourced cases observe the *sum* of the transmission delay and the secondary
   incubation period, with the intermediate infection event unobserved, so
   their denominator is the **convolved-chain** CDF.
2. **Completeness / ascertainment thinning.** A source's effective
   reproduction number is `R_eff = R * p`, where `p` is the probability the
   source's offspring chain has completed by the cut-off.
3. **The marginal vs latent switch.** The delay layer scores marginally by
   default; flipping one flag moves to the `Latent()` formulation where the
   primary event is a sampler-owned latent.
4. **The coupled latent origin.** A sourced case's latent infection time is
   coupled to its source's onset and fed into the per-record incubation delay
   through the `origin` keyword, so the user's loop owns the coupling prior.

### Out of scope (stays user Turing)

Two parts of the ANDV model are deliberately **not** package features and stay
in the user's Turing model:

- the **R(t) random walk** (a non-centred weekly walk on `log R`), and
- the **offspring-count / negative-binomial transmission-tree clustering**
  (`Z[i] ~ NegativeBinomial(k, ...)`).

We show where they plug in but do not implement them here.

### A remaining TODO

One package limitation remains.
The single-delay right-truncation lower bound triggers a NaN gradient for a
`LogNormal` index delay (pending the fix in
[#310](https://github.com/EpiAware/CensoredDistributions.jl/issues/310)), so
the index case uses an AD-safe `truncated(inc; upper = window)` workaround.
Everything else, including the source-to-offspring coupled latent origin, uses
package primitives.

## Packages used

We sample with `AutoForwardDiff` here, rather than the `AutoMooncakeForward`
backend used in the Turing fitting tutorial.
The convolved-chain truncation runs a numeric quadrature per record, and on
this small problem forward-mode `ForwardDiff` is the most reliable backend
through that quadrature; see
[Automatic differentiation backends](@ref ad-backends) for the support matrix.
"""

using Distributions
using Random
using Turing
using StatsBase: summarystats
using DynamicPPL
using DynamicPPL: to_submodel, prefix, InitFromParams
using FlexiChains: Prefixed
using CairoMakie
using CensoredDistributions
using ADTypes: AutoForwardDiff
import ForwardDiff

md"""
## A small hantavirus-style line list

We simulate a cluster: a handful of zoonotic index cases, each seeding a few
sourced cases.
The biological delays are:

- **Incubation period** `Inc ~ LogNormal`, from infection to symptom onset.
- **Transmission delay** `δ ~ Normal`, from a source's onset to the
  infection of its offspring. `δ` is **signed**: a negative draw is
  pre-symptomatic transmission (the offspring is infected before the source
  shows symptoms). `Convolved` supports negative-support components directly,
  so the signed delay needs no special handling.

The convolved chain delay a sourced case observes (source onset to offspring
onset) is `δ ⊕ Inc`.
"""

Random.seed!(2024)

## True delay parameters.
inc_meanlog = 1.6
inc_sdlog = 0.5
true_inc = LogNormal(inc_meanlog, inc_sdlog)

δ_mean = -1.0          # signed: pre-symptomatic transmission on average
δ_sd = 2.0
true_δ = Normal(δ_mean, δ_sd)

## Chain delay (source onset -> offspring onset) for sourced cases.
true_chain = convolve_distributions(true_δ, true_inc)

md"""
We lay out the cluster as a vector of records.
`source` is `0` for an index case, or the index of the source record otherwise.
Each record carries its source-onset time so we can build per-record windows;
index cases use their own (zoonotic) infection time as the anchor.
"""

## Index-case infection times (days from an arbitrary origin).
n_index = 4
index_infection = sort(rand(Uniform(0.0, 20.0), n_index))

records = NamedTuple[]
for i in 1:n_index
    t_inf = index_infection[i]
    t_onset = t_inf + rand(true_inc)
    push!(records,
        (; source = 0, t_inf, t_onset, source_onset = t_inf))
end

## Each index case seeds a Poisson(1.5) number of sourced cases.
for s in 1:n_index
    src_onset = records[s].t_onset
    for _ in 1:rand(Poisson(1.5))
        δ = rand(true_δ)
        t_inf = src_onset + δ
        t_onset = t_inf + rand(true_inc)
        push!(records,
            (; source = s, t_inf, t_onset, source_onset = src_onset))
    end
end

n_records = length(records)

md"""
The **real-time cut-off** is the day an analyst runs the model.
Anyone with onset after the cut-off is not yet in the line list; everyone
retained has their delay right-truncated at the remaining window.
"""

obs_time = maximum(r.t_onset for r in records) - 4.0

## Keep only records whose onset is on or before the cut-off (what an analyst
## would know).
observed = filter(r -> r.t_onset <= obs_time, records)
n_obs = length(observed)

md"""
For each retained record we measure:

- the **observed delay** the model scores, and
- the **remaining window** `obs_time - anchor`, the time still available
  before the cut-off, which sets the per-record truncation denominator.

Index cases are anchored at their infection time and observe a single
incubation delay.
Sourced cases are anchored at their **source's onset** and observe the chain
delay `δ ⊕ Inc`.
"""

function record_delay(r)
    if r.source == 0
        return r.t_onset - r.t_inf          # incubation only
    else
        return r.t_onset - r.source_onset   # chain: source onset -> onset
    end
end

function record_anchor(r)
    return r.source == 0 ? r.t_inf : r.source_onset
end

delays = [record_delay(r) for r in observed]
windows = [obs_time - record_anchor(r) for r in observed]
is_index = [r.source == 0 for r in observed]

md"""
## The per-record truncation denominators

This is the core of the real-time correction.
`truncate_chain` builds the right-truncated delay for one record, dispatching
the single-delay denominator against the convolved-chain denominator from the
chain segments and a mask of which intermediate events are observed:

- **Index case:** one segment `(Inc,)`, no internal boundary, so the
  denominator is `cdf(Inc, window)`.
- **Sourced case:** two segments `(δ, Inc)` with the intermediate infection
  event **unobserved** (`observed = (false,)`), so the segments collapse to
  `convolve_distributions(δ, Inc)` and the denominator is
  `cdf(δ ⊕ Inc, window)`.

This reproduces the two `@addlogprob!` truncation terms in the ANDV
`truncation_model` (index `-logcdf(Inc, ...)` vs sourced convolved
denominator) without a hand-written normaliser: the returned object is an
ordinary `truncated` distribution whose log-normaliser **is** the truncation
term.

!!! warning "TODO — single-delay truncation gradient"
    `truncate_chain`/`truncate_to_horizon` currently set the truncation lower
    bound to `minimum(dist)`. For a `LogNormal` index-case delay that is `0`,
    and `truncated` then differentiates `logcdf(LogNormal, 0) = -Inf`, giving a
    `NaN` gradient (the log-density itself is finite, so the bug only shows
    under automatic differentiation). The convolved-chain (sourced) path is
    unaffected because `minimum(Convolved) = -Inf`. Until the package
    right-truncation uses an upper-only bound, the index case below uses the
    AD-safe `truncated(inc_dist; upper = window)` workaround, which is the same
    one-sided right-truncation.
"""

function truncated_record(inc_dist, δ_dist, index::Bool, window::Real)
    if index
        ## TODO(CensoredDistributions): use
        ## `truncate_chain((inc_dist,), (), window)` once the single-delay
        ## lower bound no longer triggers a NaN gradient (see warning above).
        return truncated(inc_dist; upper = window)
    else
        return truncate_chain((δ_dist, inc_dist), (false,), window)
    end
end

md"""
A quick check that the single and convolved denominators differ, and that the
signed transmission delay flows through the convolved chain:
"""

let
    single = truncated_record(true_inc, true_δ, true, windows[1])
    chain = truncated_record(true_inc, true_δ, false, windows[end])
    (; single_logcdf = logcdf(true_inc, windows[1]),
        chain_logcdf = logcdf(true_chain, windows[end]),
        single_upper = single.upper, chain_upper = chain.upper)
end

md"""
## Completeness / ascertainment thinning

A source's onward chain is only counted once every event in it has happened by
the cut-off.
`completeness_probability(chain, window)` is the probability the chain has
completed, and `thin_by_completeness(R, chain, window)` returns the thinned
effective reproduction number `R_eff = R * p`.
This is exactly the ANDV offspring-completeness factor, built from the same
convolved chain delay used for truncation.
"""

function completeness(inc_dist, δ_dist, window::Real)
    return completeness_probability(
        convolve_distributions(δ_dist, inc_dist), window)
end

## Per-record completeness at the true parameters (illustrative).
completeness_at_truth = [completeness(true_inc, true_δ, w) for w in windows]

md"""
The thinned `R_eff` then feeds the **out-of-scope** offspring likelihood.
We show the shape of that step but do not fit it here:

```julia
## OUT OF SCOPE — stays user Turing (R(t) random walk + NB clustering):
##   R_i      = exp(log_R_at(onset_i, edges, log_R))   # R(t) random walk
##   R_eff    = thin_by_completeness(R_i, chain, window_i)  # package thinning
##   Z[i]    ~ NegativeBinomial(k, k / (k + R_eff))    # offspring clustering
```

Only the `thin_by_completeness` call is a package feature; the R(t) walk and
the negative-binomial offspring counts are the user's Turing model.

## The δ-bounded inner-truncation primitive

The ANDV controlled-outbreak counterfactual needs the joint probability that
transmission happened by an intervention horizon **and** the offspring is
still incubating at the observation horizon,
`P(δ ≤ Δ_q ∧ δ + Inc > Δ_p)`.
The inner `δ`-cap lives *inside* the convolution integral, which a plain
`truncated(Convolved(...))` cannot express.
`convolve_distributions(...; bounds = ...)` caps a component inside the
integral, so `cdf` of the bounded convolution gives the
`P(δ + Inc ≤ x ∧ δ ≤ upper_δ)` term directly:
"""

function pipeline_probability(inc_dist, δ_dist, Δ_q::Real, Δ_p::Real)
    q = cdf(δ_dist, Δ_q)
    iszero(q) && return zero(q)
    capped = convolve_distributions(
        δ_dist, inc_dist; bounds = [(-Inf, Δ_q), (-Inf, Inf)])
    return q - cdf(capped, Δ_p)
end

## Illustrative pipeline probability for one source.
pipeline_at_truth = pipeline_probability(true_inc, true_δ, 5.0, 3.0)

md"""
## The per-record loop model (marginal)

We now assemble the delay layer into a Turing model.
The model loops over records, building each record's right-truncated delay and
scoring it with `double_interval_censored_model`.
The incubation and transmission-timing parameters come from small reusable
submodels, mirroring the ANDV `incubation_model` / `transmission_delta_model`.

Marginal records share their delay distribution, so identical
`(index, window)` combinations could be collapsed and scored once with a
multiplicity `weight`.
Here the cluster is small, so we score one record per iteration for clarity;
the `weight` keyword is shown in the Turing fitting tutorial.
"""

@model function incubation_submodel()
    inc_μ ~ Normal(1.5, 0.5)
    inc_σ ~ truncated(Normal(0.0, 0.5); lower = 0)
    return LogNormal(inc_μ, inc_σ)
end

@model function transmission_submodel()
    δ_μ ~ Normal(0.0, 3.0)
    δ_σ ~ truncated(Normal(0.0, 2.0); lower = 0)
    return Normal(δ_μ, δ_σ)   # signed transmission delay
end

@model function delay_layer_marginal(delays, windows, is_index)
    inc ~ to_submodel(incubation_submodel(), false)
    δ ~ to_submodel(transmission_submodel(), false)

    for i in eachindex(delays)
        d = truncated_record(inc, δ, is_index[i], windows[i])
        obs ~ to_submodel(
            prefix(
                double_interval_censored_model(d, delays[i]),
                Symbol("obs", i)),
            false)
    end
end

md"""
We instantiate the model on the simulated, right-truncated line list and fit
with NUTS.
We start the sampler from a sensible point with `initial_params`, as the ANDV
model does, because a few extreme prior draws make the convolved-chain
quadrature numerically awkward.
The cluster is small, so a short run suffices to recover the delay parameters.
"""

marginal_mdl = delay_layer_marginal(delays, windows, is_index)

init_params = InitFromParams(
    (inc_μ = 1.5, inc_σ = 0.5, δ_μ = 0.0, δ_σ = 1.0))

marginal_fit = sample(
    marginal_mdl,
    NUTS(; adtype = AutoForwardDiff()), 600;
    initial_params = init_params, progress = false)

summarystats(marginal_fit)

md"""
### Recovery check

We compare the posterior for the incubation parameters against the truth.
The chain is a FlexiChains `VNChain`, so we pull each parameter with a
`Prefixed` `VarName` (the submodel prefixes `inc.` onto its variables).
"""

let
    μ = vec(marginal_fit[Prefixed(@varname(inc_μ))])
    σ = vec(marginal_fit[Prefixed(@varname(inc_σ))])
    (; inc_μ_post = (mean(μ), quantile(μ, (0.05, 0.95))),
        inc_μ_true = inc_meanlog,
        inc_σ_post = (mean(σ), quantile(σ, (0.05, 0.95))),
        inc_σ_true = inc_sdlog)
end

md"""
## The easy marginal vs latent switch

The package's design decision to marginalise the primary event or keep it as a
sampler-owned latent lives **on the distribution**, via its `mode`.
The submodel reads the resolved mode and dispatches; the model body does not
change.
Below we score the *index-case incubation delays* with the latent formulation:
the only change is adding `mode = Latent()` to the `primary_censored` build.

Each latent record samples its own primary event, so the latent path
vectorises over records (it cannot collapse duplicates with a multiplicity
`weight`).
We use a `Uniform` primary-event window for the latent demonstration, matching
the censored-submodel latent path.
"""

index_delays = delays[is_index]
pwindow = 1.0   # primary event window width

@model function index_layer_latent(index_delays, pwindow)
    inc ~ to_submodel(incubation_submodel(), false)

    for i in eachindex(index_delays)
        ## ONLY change from marginal: `mode = Latent()`.
        d = primary_censored(inc, Uniform(0, pwindow); mode = Latent())
        obs ~ to_submodel(
            prefix(
                primary_censored_model(d, index_delays[i]),
                Symbol("obs", i)),
            false)
    end
end

latent_mdl = index_layer_latent(index_delays, pwindow)

latent_fit = sample(
    latent_mdl,
    NUTS(; adtype = AutoForwardDiff()), 500;
    progress = false)

summarystats(latent_fit)

md"""
The latent fit recovers the same incubation location, demonstrating the
marginal-versus-latent equivalence: the marginal default integrates the
primary out inside `logpdf`, the latent path samples it, and switching needs
only the `mode` flag.

## Coupled latent origin

The ANDV latent-times model couples a sourced case's latent **infection time**
to its source's onset time: the offspring is infected after its source's onset
by the transmission delay `δ`, and shows symptoms after a further incubation
period.
That coupling needs a *per-record latent origin sampled in the user's loop* to
feed the per-record incubation delay.

The latent `primary_censored_model` takes an `origin` keyword for exactly this.
When `origin` is supplied the submodel scores **only** the conditional delay
`logpdf(Inc, T_onset − origin)` and draws no primary-event prior, so the user's
loop owns the coupled prior over the origin.
Here each sourced record samples its own latent infection time `T_inf`, scores
the transmission link `δ = T_inf − source_onset` (the user-owned coupled prior),
and passes `T_inf` as the incubation `origin`.
"""

@model function coupled_latent_sourced(observed)
    inc ~ to_submodel(incubation_submodel(), false)
    δ ~ to_submodel(transmission_submodel(), false)

    T = partype(inc)
    n = length(observed)
    T_inf = Vector{T}(undef, n)
    for i in 1:n
        onset = observed[i].t_onset
        ## Latent offspring infection time (before its own onset).
        T_inf[i] ~ Uniform(onset - 60.0, onset - 1e-3)

        if observed[i].source != 0
            ## USER owns the coupled prior: transmission link from the source's
            ## onset to this offspring's infection.
            link = T_inf[i] - observed[i].source_onset
            Turing.@addlogprob! logpdf(δ, link)
        end

        ## Incubation as the conditional delay, with the coupled infection time
        ## injected as `origin`; the submodel draws no primary-event prior.
        d = primary_censored(inc, Uniform(0, 1); mode = Latent())
        obs ~ to_submodel(
            prefix(
                primary_censored_model(d, onset; origin = T_inf[i]),
                Symbol("obs", i)),
            false)
    end
end

coupled_mdl = coupled_latent_sourced(observed)

coupled_fit = sample(
    coupled_mdl,
    NUTS(; adtype = AutoForwardDiff()), 400;
    initial_params = init_params, progress = false)

summarystats(coupled_fit)

md"""
The coupled fit recovers the incubation parameters while the user loop owns the
transmission-link prior, so the source-to-offspring coupling is expressed
without a hand-rolled incubation `@addlogprob!`.
The marginal delay layer earlier does not need this coupling: it scores the
*observed chain delay* directly with the convolved-chain truncation, which is
the marginalised form of the coupled latent.
"""

let
    μ = vec(coupled_fit[Prefixed(@varname(inc_μ))])
    (; inc_μ_post = (mean(μ), quantile(μ, (0.05, 0.95))),
        inc_μ_true = inc_meanlog)
end

md"""
## Summary

The real-time delay, truncation, and ascertainment layer of the ANDV model is
reproduced from package primitives in a per-record loop:

| ANDV component | Package primitive |
|---|---|
| Index right-truncation `-logcdf(Inc, ...)` | `truncate_chain((Inc,), (), window)` |
| Sourced convolved-chain truncation | `truncate_chain((δ, Inc), (false,), window)` |
| Convolved chain delay `δ ⊕ Inc` (signed δ) | `convolve_distributions(δ, Inc)` |
| Completeness factor `p` | `completeness_probability(chain, window)` |
| Offspring thinning `R_eff = R · p` | `thin_by_completeness(R, chain, window)` |
| δ-bounded pipeline probability | `convolve_distributions(...; bounds = ...)` |
| Marginal vs latent switch | `mode = Latent()` on `primary_censored` |
| Coupled latent origin | `primary_censored_model(d, y; origin = ...)` |

Out of scope (user Turing): the R(t) random walk and the negative-binomial
offspring-count clustering.
Remaining TODO: the single-delay right-truncation lower bound triggers a NaN
gradient (pending the package fix in
[#310](https://github.com/EpiAware/CensoredDistributions.jl/issues/310)), so
the index case uses the AD-safe `truncated(inc; upper = window)` workaround.
"""
