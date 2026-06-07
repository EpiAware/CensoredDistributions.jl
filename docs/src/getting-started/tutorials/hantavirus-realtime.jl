md"""
# Hantavirus realtime: per-record delay, truncation and ascertainment

## Introduction

This walkthrough rebuilds the delay, right-truncation and ascertainment layer
of the Epuyén Andes virus (ANDV) realtime model on the
`CensoredDistributions.jl` stack.
The original analysis lives in `epiforecasts/andv-linelist-analysis`.
We reproduce the parts that are delay distributions, and we keep the parts that
are not (the ``R(t)`` random walk and the offspring-count / case-cluster
machinery) in the user's own Turing model.

### What we cover

1. Per-record `NamedTuple` rows whose observed/missing pattern drives the
   per-record path.
2. The single-delay right-truncation denominator for index cases.
3. The convolved-chain right-truncation denominator for sourced cases.
4. The coupled latent origin for a sourced case, scored through the latent
   path (`PrimaryConditional` / `primary_conditional_logpdf`).
5. Completeness/ascertainment thinning ``R_{eff} = R \cdot p``.
6. The intended UI: each record is **one** composed distribution dispatched as
   a **submodel** that manages its own likelihood; the model loops over records
   and the loop only wires the cross-record coupled origin.
7. A simulate-and-recover exercise that draws event times with full-path
   `rand` and recovers the delay parameters.

### Out of scope (stays user Turing)

- The ``R(t)`` weekly random walk.
- The offspring-count / negative-binomial clustering likelihood.

These are ordinary Turing code in the original model; nothing in this package
replaces them, and this page does not reimplement them.

## What you should know first

This tutorial builds on
[Getting Started with CensoredDistributions.jl](@ref getting-started) and on
[Fitting with Turing.jl](@ref).
We sample with `ForwardDiff` for a short, self-contained build.

## Packages used
"""

using CensoredDistributions
using Distributions
using Turing
using DynamicPPL: prefix
using Random
using Statistics
using ADTypes: AutoForwardDiff

Random.seed!(2024)

md"""
## The realtime delay structure

The ANDV linelist has two record types.

An **index** case has no recorded source.
We observe a single incubation delay from its (latent) infection time to its
onset, and that delay is right-truncated: a case is only in the data if its
onset falls on or before the realtime cut-off.

A **sourced** case is attributed to an earlier case.
We observe the gap from the *source's* onset to the *secondary's* onset, which
is the convolution of the signed transmission delay ``\delta`` (infection of
the secondary relative to the source's onset) and the secondary's incubation
period.
The intermediate event (the secondary's infection time) is not recorded, so
the truncation denominator must use the convolution of the two unobserved
segments, not either segment alone.
Mixing these two truncation denominators up is the easy error the original
`truncation_model` guards against by carrying both forms side by side.

We encode the true delay parameters as ordinary `Distributions.jl` objects.
The transmission delay is signed (a secondary can be infected before the
source shows symptoms), so it is a `Normal` that can go negative.
"""

inc_true = LogNormal(2.06, 0.41)

delta_true = Normal(-1.0, 2.0)

horizon = 45.0

md"""
## Simulate event times with full-path `rand`

We simulate from the generative process directly, drawing each delay with
`rand`, then keep only the records whose onset falls on or before the
`horizon`.
That keep-rule is exactly the right-truncation the likelihood must correct
for.

Each record is a `NamedTuple` row.
The fields a record carries *are* its observed/missing pattern: an index row
carries an observed `delay` and its remaining `window`; a sourced row carries
the source onset, the secondary onset, and its `window`.
A loop over rows then dispatches on which fields are present, which is the
per-record-loop pattern the realtime model is built from.
"""

index_rows = NamedTuple[]
for _ in 1:80
    inc = rand(inc_true)
    anchor = rand(Uniform(0, 25))           # known infection-window anchor
    onset = anchor + inc
    onset <= horizon || continue            # right-truncated at the horizon
    push!(index_rows, (; delay = inc, window = horizon - anchor))
end

sourced_rows = NamedTuple[]
for _ in 1:80
    src_onset = rand(Uniform(0, 18))
    gap = rand(delta_true) + rand(inc_true) # convolved chain: delta + inc
    sec_onset = src_onset + gap
    sec_onset <= horizon || continue
    push!(sourced_rows,
        (; src_onset, sec_onset, window = horizon - src_onset))
end

(length(index_rows), length(sourced_rows))

md"""
## Right-truncation: single-delay versus convolved-chain

The package builds the right-truncation denominator for one record and
dispatches the single-delay case against the convolved-chain case.

For an **index** case the recorded delay is a single segment, so the
denominator is `cdf(inc, window)`.
[`truncate_to_horizon`](@ref) returns the right-truncated object whose
log-normaliser is exactly that ``-\log \text{cdf}`` term, so scoring the
observed delay through its `logpdf` reproduces the per-record correction with
no PPL dependency.

The truncation is upper-only: it caps the upper bound at `window` and adds no
lower bound.
This matters for AD on a `LogNormal` index delay.
An earlier version set `lower = minimum(dist) = 0`, and `truncated` then
differentiated `logcdf(LogNormal, 0) = -Inf`, giving a `NaN` gradient.
With the upper-only fix the gradient is finite:
"""

import ForwardDiff

let
    f = x -> logpdf(truncate_to_horizon(LogNormal(x[1], x[2]), horizon), 5.41)
    ForwardDiff.gradient(f, [2.06, 0.41])   # finite, no NaN
end

md"""
For a **sourced** case the intermediate event is unobserved, so the
denominator must be `cdf` of the *convolution* of the two segments.
[`truncate_chain`](@ref) picks the right form from a chain of segments and a
mask of which internal splitting events were observed.
With `observed = (false,)` the splitting event between ``\delta`` and the
incubation period is not recorded, so the chain collapses to the convolution
and the denominator becomes the convolved-chain CDF.
"""

let
    index = truncate_chain((inc_true,), (), horizon)
    sourced = truncate_chain((delta_true, inc_true), (false,), horizon)
    (logpdf(index, 5.41), logpdf(sourced, 5.41))
end

md"""
The convolution itself is [`convolve_distributions`](@ref), and its `cdf` is
the chain-completion probability used below for ascertainment thinning.
The signed `delta_true` convolves and the result is AD-safe.
"""

chain_delay = convolve_distributions(delta_true, inc_true)

cdf(chain_delay, horizon)

md"""
## Coupled latent origin (the latent path)

For a sourced case the source's onset is a shared, coupled origin: the
secondary's infection time is the source onset plus the signed transmission
delay, and its onset is that plus the incubation period.
When you want the origin sampler-owned rather than integrated out, use the
latent flow.

A single [`primary_censored`](@ref) leaf is marginalisable, so by default it is
the univariate marginal.
Wrapping it with [`latent`](@ref) selects the multivariate representation over
`[primary, observed]`: `rand` produces the internal times, and `logpdf` is the
primary prior plus the conditional of the observed time given the primary.
"""

let
    node = primary_censored(inc_true, Uniform(0, 1))
    ld = latent(node)
    rand(ld)                                 # [primary, observed]
end

md"""
The conditional that the latent path scores is
[`PrimaryConditional`](@ref): given a realised primary ``p`` it shifts the
delay, so `logpdf(PrimaryConditional(d, p), y) = logpdf(get_dist(d), y - p)`.
The named entry point is [`primary_conditional_logpdf`](@ref).
For a coupled origin shared across records (a source's onset feeding an
offspring's infection), the caller owns the coupled prior: it samples the
shared origin in its own model and scores each record with
`primary_conditional_logpdf(d, p_shared, y)` directly.
This is the escape hatch the design calls out for a shared latent that the
leaf's own `p ~ get_primary_event(d)` cannot express.
"""

let
    d = primary_censored(inc_true, Uniform(0, 1))
    primary_conditional_logpdf(d, 0.3, 2.7)  # logpdf(inc_true, 2.7 - 0.3)
end

md"""
## Ascertainment / completeness thinning

In realtime, an offspring is only counted if its full chain has completed by
the horizon, so the per-source rate is thinned by the completion probability
``p`` to give ``R_{eff} = R \cdot p``.
That ``p`` is the chain-completion CDF, `cdf(convolve_distributions(δ, inc),
horizon - T_onset)`, exactly the convolved-chain denominator from above.

!!! note "Flagged gap: no thinning helper in the stack"
    A dedicated `thin_by_completeness` / `completeness_probability` helper is
    **not** present in this stack, so we compute the thinning explicitly with
    `R * cdf(convolve_distributions(...), horizon)`.
    A small follow-up could add the named helper; the explicit form below is
    the same quantity and is AD-safe.

The completeness probability is differentiable in the delay parameters, so it
can sit inside a Turing model where ``R`` is sampled:
"""

function completeness_probability(delta, inc, window)
    cdf(convolve_distributions(delta, inc), window)
end

R_eff(R, delta, inc, window) = R * completeness_probability(delta, inc, window)

let
    R = 1.8
    R_eff(R, delta_true, inc_true, horizon - 5.0)
end

md"""
The thinning is AD-safe in the delay parameters:
"""

let
    fp = x -> log(completeness_probability(
        Normal(x[1], 2.0), LogNormal(x[2], 0.41), horizon))
    ForwardDiff.gradient(fp, [-1.0, 2.06])   # finite, no NaN
end

md"""
The ``R`` that this multiplies, and the offspring-count likelihood it feeds,
are the user's Turing code and stay out of scope here.

## The per-record submodel UI

The intended usage is **not** hand-rolled per-component dispatch.
Each record is one composed distribution `d`, and the model dispatches to it as
a *submodel* that manages its own likelihood internally:

```julia
for i in eachindex(rows)
    obs[i] ~ to_submodel(
        prefix(double_interval_censored_model(d_i, rows[i].y), Symbol(:r, i)))
end
```

[`double_interval_censored_model`](@ref) is the univariate submodel
constructor: it takes the *whole* composed distribution as `d` and scores the
observation against it, optionally weighted by the row's multiplicity.
Because the per-record right-truncated delay is itself a univariate composed
distribution (the [`truncate_to_horizon`](@ref) object for an index record, the
[`truncate_chain`](@ref) object for a sourced record), passing it as `d` lets
the submodel own the marginal/condition/convolve likelihood.
We never write the `logpdf` or an `@addlogprob!` by hand.

`prefix(...)` namespaces each record's submodel so the variables stay distinct
and groupable in the chain.

The one piece that stays in the user loop is the **cross-record coupling**.
For a sourced record the observed quantity is the gap from the *source's* onset
to the secondary's onset, so the source onset (the coupled origin) is wired
into the offspring's record as the gap anchor before the record's distribution
is dispatched.
This loop wiring of the coupled origin is exactly the user responsibility the
design assigns to the caller; everything else is the submodel's job.

## Recover the delay parameters

We put the per-record submodel loop together and recover the delay parameters
from the simulated rows.
Index records dispatch their incubation delay to the single-delay
right-truncation distribution; sourced records dispatch their source-to-
secondary gap (anchored on the coupled source onset) to the convolved-chain
right-truncation distribution.
"""

@model function hanta_delays(index_rows, sourced_rows)
    mu_inc ~ Normal(2.0, 0.5)
    sigma_inc ~ truncated(Normal(0.0, 0.5); lower = 0)
    mu_delta ~ Normal(0.0, 3.0)
    sigma_delta ~ truncated(Normal(0.0, 2.0); lower = 0)
    inc = LogNormal(mu_inc, sigma_inc)
    delta = Normal(mu_delta, sigma_delta)

    ## index records: the single-delay right-truncation distribution, dispatched
    ## as a submodel that owns its likelihood.
    for i in eachindex(index_rows)
        r = index_rows[i]
        d = truncate_to_horizon(inc, r.window)
        idx ~ to_submodel(
            prefix(double_interval_censored_model(d, r.delay),
                Symbol(:idx, i)), false)
    end

    ## sourced records: the convolved-chain right-truncation distribution,
    ## dispatched as a submodel. The loop only wires the coupled source onset:
    ## the observed gap is anchored on the source's onset (cross-record
    ## coupling), then the record's distribution owns the likelihood.
    for i in eachindex(sourced_rows)
        r = sourced_rows[i]
        d = truncate_chain((delta, inc), (false,), r.window)
        gap = r.sec_onset - r.src_onset
        src ~ to_submodel(
            prefix(double_interval_censored_model(d, gap),
                Symbol(:src, i)), false)
    end
end

mdl = hanta_delays(index_rows, sourced_rows)

chn = sample(mdl, NUTS(0.95; adtype = AutoForwardDiff()), 600)

md"""
The posterior means sit near the true values used to simulate the data
(`inc_true = LogNormal(2.06, 0.41)`, `delta_true = Normal(-1.0, 2.0)`):
"""

(mu_inc = mean(chn[:mu_inc]),
    sigma_inc = mean(chn[:sigma_inc]),
    mu_delta = mean(chn[:mu_delta]),
    sigma_delta = mean(chn[:sigma_delta]))

md"""
## Summary

We rebuilt the realtime ANDV delay layer on the package primitives: per-record
`NamedTuple` rows, [`truncate_to_horizon`](@ref) for the single-delay
denominator, [`truncate_chain`](@ref) for the convolved-chain denominator, the
coupled latent origin through [`PrimaryConditional`](@ref) /
[`primary_conditional_logpdf`](@ref), and explicit ``R_{eff} = R \cdot p``
ascertainment thinning via the convolved-chain CDF.
The model dispatches each record's whole distribution as a
[`double_interval_censored_model`](@ref) submodel that owns its likelihood, and
the loop only wires the cross-record coupled origin, never a hand-rolled
per-component `logpdf` or `@addlogprob!`.
The ``R(t)`` random walk and the offspring-count clustering stay in the user's
Turing model and are not part of this package.
"""
