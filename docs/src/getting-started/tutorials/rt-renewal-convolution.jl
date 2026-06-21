md"""
# [An Rt renewal model with delay convolution](@id rt-renewal-convolution)

## Introduction

### What are we going to do in this exercise

We build a small EpiNow2-style renewal model and fit a time-varying
reproduction number to two observed streams at once: reported cases and
deaths.
Both streams come from the same latent infections through a shared incubation
period to symptom onset, after which they branch: onset to report for cases and
onset to death for deaths.
Each branch is also thinned, because the two streams disagree about scale.
Only a fraction of infections is ever reported (the ascertainment), and only a
small fraction ends in death (the infection fatality ratio).

The renewal recursion itself is user-side; the package's job is the
*observation layer*.
We compose one branched delay stack with shared incubation and two reporting
branches, then push the latent infection series through it with a single
[`convolve_distributions`](@ref) call that returns both event streams at once,
each already thinned.

We cover:

1. A pure-Julia forward demo: pick a true Rt, ascertainment and IFR, run the
   renewal step, and convolve infections through the combined stack to simulate
   observed cases and deaths.
2. A Turing fit: the same renewal and combined-stack convolution recomputed
   inside a `@model`, fitting Rt, ascertainment and IFR to the simulated cases
   and deaths, and assessing how well each is recovered.
3. A posterior-predictive check: push the posterior back through the same
   forward map and overlay the fitted case and death series, with a band, on
   the data.

### What might I need to know before starting

This tutorial builds on [Getting Started with
CensoredDistributions.jl](@ref getting-started) and the composer reference,
[Composing censored distributions](@ref composer-toolkit), which introduces
[`Sequential`](@ref) chains, [`compose`](@ref) and
[`convolve_distributions`](@ref).
We do not re-explain the composer basics here; this page is about using a
branched, thinned delay stack as a renewal observation layer.

[`convolve_distributions`](@ref) is AD-safe: the discretised delay PMFs depend
differentiably on the delay parameters, the vector convolution is linear, and a
[`thin`](@ref) factor is carried through as a forward multiplier, so gradients
flow through the whole stack and it can be called directly inside a `@model`.
We sample with Mooncake reverse-mode AD (see [Automatic differentiation
backends](@ref ad-backends) for the support matrix).

## Packages used

We use Distributions for the delay distributions, Turing and Mooncake for the
fit, FlexiChains for the chain output, AlgebraOfGraphics with CairoMakie for the
plots, DataFramesMeta for the plotting tables, and Random and Statistics for
reproducibility and summaries.
"""

using CensoredDistributions
using Distributions
using Turing
using Mooncake
using FlexiChains: VNChain
using AlgebraOfGraphics
using CairoMakie
using DataFramesMeta
using Random
using Statistics
using ADTypes: AutoMooncake

md"""
## The model

Infections follow a renewal process driven by a time-varying reproduction
number ``R_t`` and a generation-interval PMF ``g``:

```math
I_t = R_t \sum_{s \ge 1} g_s \, I_{t-s}.
```

Both observed streams share the infection-to-onset incubation period, then
branch.
For cases the branch is onset to report and the count is thinned by the
ascertainment ``\alpha``; for deaths the branch is onset to death and the count
is thinned by the infection fatality ratio ``\rho``:

```math
\text{cases}_t = \alpha \,(I \circledast d_{\text{case}})_t, \qquad
\text{deaths}_t = \rho \,(I \circledast d_{\text{death}})_t,
```

where ``\circledast`` is the causal discrete convolution and
``d_{\text{case}}`` and ``d_{\text{death}}`` are the two branch delays, each the
shared incubation convolved with its branch tail.
Thinning matters because the two streams disagree about scale: most infections
never appear as reported cases, and only a small fraction end in death, so a
model that ignores ascertainment and IFR cannot match both counts with one
infection series.

### The generation interval

The generation interval is a short discrete PMF over positive lags.
We discretise a Gamma with [`double_interval_censored`](@ref), which applies
primary-event censoring and truncation before interval censoring, then read the
day-bin masses with [`discretise_pmf`](@ref) so each entry is a proper bin
integral rather than a point-mass evaluation.
We drop the zero lag so an infection can only generate new infections from the
next day on, and renormalise the remaining positive lags to a PMF.
"""

gen_dist = double_interval_censored(Gamma(2.5, 1.3); upper = 14.0,
    interval = 1.0)

gi_max = 12

gen_pmf = CensoredDistributions.discretise_pmf(gen_dist, gi_max)

g = let masses = pdf(gen_pmf, 1:gi_max)
    masses ./ sum(masses)
end

md"""
### The renewal step

A plain loop applies the renewal recursion: each day's infections are
``R_t`` times the generation-weighted sum of recent infections.
We seed the first few days so the recursion has history to build on.
"""

function renewal(Rt, g, I0; seed_days = length(g))
    n = length(Rt)
    I = zeros(eltype(Rt), n)
    I[1:seed_days] .= I0
    for t in (seed_days + 1):n
        acc = zero(eltype(Rt))
        for s in 1:min(length(g), t - 1)
            acc += g[s] * I[t - s]
        end
        I[t] = Rt[t] * acc
    end
    return I
end

md"""
### The branched observation stack

Each delay stage is a continuous distribution discretised with
[`double_interval_censored`](@ref), which censors the primary event, truncates,
and interval-censors to a daily PMF in one call.
The incubation period is shared; the case and death branches are its tails.
We build the whole observation layer as one composed stack with
[`compose`](@ref)`(incubation; cases = ..., deaths = ...)`: a shared origin that
fans out into named branches.
Each branch is wrapped in [`thin`](@ref) so its convolved count carries the
branch's scale (ascertainment for cases, IFR for deaths).
[`thin`](@ref) is transparent to `logpdf` and materialises only under
convolution, so the same object scores a delay and scales a count series.
"""

incubation = double_interval_censored(Gamma(1.8, 1.4); upper = 20.0,
    interval = 1.0)

onset_report = double_interval_censored(Gamma(1.5, 1.2); upper = 20.0,
    interval = 1.0)

onset_death = double_interval_censored(Gamma(3.0, 4.0); upper = 40.0,
    interval = 1.0)

md"""
A small helper builds the combined stack from the three delays and the two
scales, so the forward demo and the fit construct the observation layer the same
way.
The scales enter through [`thin`](@ref), so the helper takes plain numbers in
the demo and sampled latents in the fit with no change.
"""

function observation_stack(incubation, onset_report, onset_death, alpha, rho)
    return compose(incubation;
        cases = thin(onset_report, alpha),
        deaths = thin(onset_death, rho))
end

md"""
Pushing an infection series through the combined stack with a single
[`convolve_distributions`](@ref) call and `events = (:cases, :deaths)` returns a
`NamedTuple` of both streams, each discretised, convolved and thinned in one
pass.
Calling the convolution once per stream would rebuild the shared incubation
twice and lose the shared-origin structure, so we do it once.
"""

function expected_streams(stack, infections)
    return convolve_distributions(stack, infections;
        events = (:cases, :deaths))
end

md"""
## Pure-Julia forward demo

We pick a true Rt path (a piecewise level that rises, dips below one and
recovers), a true ascertainment and a true IFR, run the renewal, and convolve
through the combined stack.
This exercises the full observation machinery with no Turing in sight.

The Rt path changes level on the same day boundaries the fitted block model
uses (every 18 days), so the four blocks can represent the truth exactly and
the per-block truth we score against later is not a simplification.
"""

n_days = 70

true_Rt = vcat(fill(1.6, 18), fill(0.7, 18), fill(1.3, 18),
    fill(1.0, n_days - 54))

true_alpha = 0.3

true_rho = 0.012

I0 = 10.0

infections = renewal(true_Rt, g, I0)

md"""
One [`convolve_distributions`](@ref) call over the combined stack returns the
expected cases and deaths together, each already thinned by its branch scale.
"""

true_stack = observation_stack(incubation, onset_report, onset_death,
    true_alpha, true_rho)

expected = expected_streams(true_stack, infections)

expected_cases = expected.cases

expected_deaths = expected.deaths

md"""
The expected counts are Poisson means; we draw observed counts to act as the
data the fit will see.
We use one RNG for the data here and a separate one for the sampler later, so
the simulated dataset and the fit are seeded independently.
"""

rng = MersenneTwister(20260610)

cases_obs = rand.(rng, Poisson.(expected_cases .+ 1e-6))

deaths_obs = rand.(rng, Poisson.(expected_deaths .+ 1e-6))

md"""
The simulated streams share the same infection wave but sit at very different
scales: ascertainment pins the case counts well below infections, and the IFR
pins deaths far lower still.
The death stream also lags the case stream, because its delay is longer.
We assemble a tidy table and draw it with AlgebraOfGraphics, faceting on the
three series so each sits on its own panel.
"""

forward_df = vcat(
    DataFrame(day = 1:n_days, value = infections, kind = "infections"),
    DataFrame(day = 1:n_days, value = expected_cases, kind = "cases (mean)"),
    DataFrame(day = 1:n_days, value = Float64.(cases_obs),
        kind = "cases (obs)"),
    DataFrame(day = 1:n_days, value = expected_deaths, kind = "deaths (mean)"),
    DataFrame(day = 1:n_days, value = Float64.(deaths_obs),
        kind = "deaths (obs)"))

forward_means = @subset forward_df endswith.(:kind, "(mean)") .|
                                   (:kind .== "infections")

forward_obs = @subset forward_df endswith.(:kind, "(obs)")

forward_plot = (
    data(forward_means) *
    mapping(:day, :value, layout = :kind) * visual(Lines) +
    data(forward_obs) *
    mapping(:day, :value, layout = :kind) * visual(Scatter, markersize = 5))

draw(forward_plot;
    facet = (; linkyaxes = :none),
    figure = (; size = (800, 500)),
    axis = (; xlabel = "day", ylabel = "count",
        title = "Forward simulation"))

md"""
## The Turing fit

The model puts priors on the reproduction number, the ascertainment and the
IFR, then rebuilds the same combined stack inside the `@model` with the sampled
scales threaded through [`thin`](@ref) and recomputes both expected streams with
the same single [`convolve_distributions`](@ref) call used in the forward demo.
The sampled `alpha` and `rho` are AD duals; [`thin`](@ref) carries them as
forward factors and the convolution stays AD-safe, so nothing special is needed.
We give Rt a small piecewise level per block, so the parameter count stays low.
The block layout (`n_blocks`, `block_len`, `block_of`) is defined once below and
read as globals inside the `@model` and the predictive map, to keep the renewal
and observation code shared across the forward demo and the fit.
"""

n_blocks = 4

block_len = ceil(Int, n_days / n_blocks)

block_of(t) = min(n_blocks, fld(t - 1, block_len) + 1)

@model function rt_renewal(cases, deaths, g, incubation, onset_report,
        onset_death, I0)
    n = length(cases)
    R_block ~ filldist(truncated(Normal(1.0, 0.5); lower = 0.1), n_blocks)
    alpha ~ Beta(2, 5)
    rho ~ Beta(1, 50)

    Rt = [R_block[block_of(t)] for t in 1:n]
    infections = renewal(Rt, g, I0)
    stack = observation_stack(incubation, onset_report, onset_death, alpha, rho)
    expected = expected_streams(stack, infections)

    for t in 1:n
        cases[t] ~ Poisson(expected.cases[t] + 1e-6)
        deaths[t] ~ Poisson(expected.deaths[t] + 1e-6)
    end
end

md"""
We fit with NUTS on Mooncake, taking a small number of post-warmup draws across
two parallel chains, enough to show the well-identified parameters recover.
Most of the runtime is the one-off gradient compilation rather than the draws,
so the page sits with the heavy fitting tutorials.
"""

model = rt_renewal(cases_obs, deaths_obs, g, incubation, onset_report,
    onset_death, I0)

chain = sample(Xoshiro(1), model,
    NUTS(0.8; adtype = AutoMooncake(; config = nothing)),
    MCMCThreads(), 100, 2; chain_type = VNChain, progress = false);

md"""
## Recovery

We pull the posterior parameters from the chain with FlexiChains.
Indexing the chain by a parameter name returns every draw across all chains in
one access; we collect the three model parameters into a single NamedTuple of
draw vectors.
The block-level Rt is a vector parameter, so its draws are vectors.
The well-identified ascertainment and the weakly-identified IFR have no figure
of their own, so we report each against the truth.
"""

param_names = (:R_block, :alpha, :rho)

draws = (; (p => vec(chain[p]) for p in param_names)...);

recovery = (
    ascertainment = (truth = true_alpha,
        posterior = round(mean(draws.alpha); digits = 3)),
    ifr = (truth = true_rho, posterior = round(mean(draws.rho); digits = 4)))

md"""
The block-level Rt and the ascertainment recover well: the block levels track
the true rise, dip and recovery to within a few percent, and the ascertainment
lands near its true value despite never being observed directly.
The IFR is only weakly identified.
The simulated death stream is thin (around ten deaths over the whole period),
so it carries little information about ``\rho``; the posterior sits between the
data and the `Beta(1, 50)` prior and recovers the right order of magnitude
rather than a sharp point estimate.
Fitting both streams together still identifies the case-side scale that a
single stream could not, while the death-side scale stays uncertain.
We plot the true Rt path against a sample of posterior Rt trajectories, one per
draw, so the spread across draws is visible rather than a single mean line.
"""

rt_draws = [vcat(DataFrame(day = 1:n_days,
                Rt = [draws.R_block[i][block_of(t)] for t in 1:n_days],
                draw = i)) for i in eachindex(draws.alpha)]

rt_traj_df = vcat(rt_draws...)

rt_plot = (
    data(rt_traj_df) *
    mapping(:day, :Rt, group = :draw => nonnumeric) *
    visual(Lines, color = (:steelblue, 0.15)) +
    data(DataFrame(day = 1:n_days, Rt = true_Rt)) *
    mapping(:day, :Rt) * visual(Lines, color = :black, linewidth = 2))

draw(rt_plot;
    figure = (; size = (800, 350)),
    axis = (; xlabel = "day", ylabel = "Rt",
        title = "Rt recovery: posterior trajectories vs truth"))

md"""
## Posterior-predictive fit to data

Recovering the parameters is one check; a stronger one is whether the fitted
model reproduces the data it was fit to.
We push the posterior back through the same forward map used in the demo and the
`@model`: for each draw we rebuild the Rt path, run the renewal, rebuild the
observation stack with that draw's ascertainment and IFR, and convolve to
expected cases and deaths.
This reuses [`convolve_distributions`](@ref) through the `observation_stack` /
`expected_streams` helpers with no new machinery.
"""

post_draws = [(R_block = draws.R_block[i], alpha = draws.alpha[i],
                  rho = draws.rho[i]) for i in eachindex(draws.alpha)];

function predict_streams(draw)
    Rt = [draw.R_block[block_of(t)] for t in 1:n_days]
    infections = renewal(Rt, g, I0)
    stack = observation_stack(incubation, onset_report, onset_death,
        draw.alpha, draw.rho)
    return expected_streams(stack, infections)
end

pred = [predict_streams(d) for d in post_draws];

md"""
For each draw we resample the Poisson observation layer, `rand(Poisson(mean))`,
so the band is a true posterior-predictive interval that includes observation
noise rather than a credible interval on the expected count alone.
This matters for the deaths panel, where the counts are 0 to 2 and the
observation noise dominates: a band on the mean alone would be far too narrow to
contain integer counts.
We summarise the predictive draws by their pointwise median and a 95% band, then
overlay the simulated observed counts the fit saw.
A model that fits well has the observed points sitting inside its predictive
band.
"""

pred_rng = MersenneTwister(20260611)

function band_df(series, kind)
    mat = reduce(hcat, series)
    draws_pp = [rand(pred_rng, Poisson(max(mat[t, d], 0.0) + 1e-6))
                for t in 1:n_days, d in axes(mat, 2)]
    return DataFrame(day = 1:n_days, kind = kind,
        med = [median(draws_pp[t, :]) for t in 1:n_days],
        lo = [quantile(draws_pp[t, :], 0.025) for t in 1:n_days],
        hi = [quantile(draws_pp[t, :], 0.975) for t in 1:n_days])
end

pp_band = vcat(
    band_df([p.cases for p in pred], "cases"),
    band_df([p.deaths for p in pred], "deaths"))

pp_obs = vcat(
    DataFrame(day = 1:n_days, kind = "cases", value = Float64.(cases_obs)),
    DataFrame(day = 1:n_days, kind = "deaths", value = Float64.(deaths_obs)))

pp_plot = (
    data(pp_band) *
    mapping(:day, :lo, :hi, layout = :kind) * visual(Band, alpha = 0.3) +
    data(pp_band) *
    mapping(:day, :med, layout = :kind) * visual(Lines) +
    data(pp_obs) *
    mapping(:day, :value, layout = :kind) * visual(Scatter, markersize = 5))

draw(pp_plot;
    facet = (; linkyaxes = :none),
    figure = (; size = (800, 350)),
    axis = (; xlabel = "day", ylabel = "count",
        title = "Posterior-predictive fit"))

md"""
The fitted median tracks both observed series and the band covers the scatter,
so the model reproduces the case and death streams it was fit to, at their very
different scales, from one shared infection process.

The same pattern carries to the case studies: after fitting, push the posterior
back through the model's forward map, summarise the predicted series by a median
and a band across draws, and overlay the observed data to check the fit visually
rather than only by parameter recovery.

## Summary

- The renewal recursion is user-side; the package supplies the observation
  layer as one branched delay stack built with [`compose`](@ref) and pushed
  through [`convolve_distributions`](@ref) in a single call that returns every
  requested event stream.
- Two streams share one infection series and one incubation period, then branch:
  onset to report for cases and onset to death for deaths, each scaled by a
  [`thin`](@ref) factor carried in the stack.
- Thinning matters because the streams sit at different scales; fitting both at
  once identifies the case-side ascertainment, while the IFR stays only weakly
  identified from the thin death stream.
- The combined stack is AD-safe, so the same renewal, single convolution and
  in-stack thinning run inside a Turing `@model` with Mooncake reverse-mode AD
  across parallel chains; Rt and the ascertainment recover well, while the IFR
  is only weakly identified from the thin death stream.
"""
