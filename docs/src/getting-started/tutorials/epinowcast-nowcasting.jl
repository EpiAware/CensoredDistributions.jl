md"""
# [An epinowcast-style hazard nowcasting model](@id epinowcast-nowcasting)

## Introduction

### What are we going to do in this exercise

We build a nowcasting model in the style of
[epinowcast](https://package.epinowcast.org/) and fit it with Turing.
Counts arrive with a reporting delay, so the most recent reference dates look
artificially low: their late reports have not happened yet.
Nowcasting corrects this right-truncation by modelling the delay and inferring
the counts that are still to be reported.

The model has five parts, each mapped onto a CensoredDistributions tool:

1. An **expectation process**: a log-normal random walk for the expected final
   counts ``\lambda_t`` over reference date ``t``.
2. A **branched reporting delay**: infection to symptom onset, then a branch from
   onset to a reported **case** and onset to a reported **death**, built as one
   shared-origin [`compose`](@ref) stack of
   [`double_interval_censored`](@ref) delays.
3. A **discrete-time reporting hazard** with **reference-date** and
   **report-date** effects: each branch delay is discretised to a PMF, turned
   into a hazard, and reshaped by a reference-date random walk (slow drift in
   reporting speed) and a report-date day-of-week term, using
   [`reference_report_matrix`](@ref).
4. An **ascertainment layer**: the case stream is under-reported, thinned by a
   reported fraction ``\rho`` with [`thin`](@ref).
5. **Real-time right-truncation**: only cells whose report date ``t + d`` is at
   or before "now" are observed. We also show this truncation acting on the
   composed two-delay delay itself, with [`truncate_to_horizon`](@ref) and
   [`completeness_probability`](@ref).

We then fit the model in Turing and show that it recovers the expectation path,
the hazard effects, the ascertainment fraction and the right-truncated counts,
including a posterior nowcast of the not-yet-reported totals and a fit-to-data
view of the reference-by-report count structure.

### What might I need to know before starting

This tutorial builds on [Getting Started with
CensoredDistributions.jl](@ref getting-started), the composer reference
([Composing censored distributions](@ref composer-toolkit)) and the renewal
observation layer ([An Rt renewal model with delay
convolution](@ref)).
We reuse the branched [`compose`](@ref) stack from the renewal tutorial and add
the discrete-time hazard layer on top.

### The epinowcast model

epinowcast models the reporting delay as a discrete-time hazard.
The hazard ``h_{t,d}`` is the probability that a case with reference date ``t``
is reported at delay ``d`` given it has not been reported by delay ``d - 1``.
Writing the hazard on the logit scale lets the delay distribution vary smoothly
by reference date and carry a calendar report-date pattern while staying a proper
PMF per reference date:

```math
\operatorname{logit}(h_{t,d}) = \gamma_{t,d} + \delta_{t} + \epsilon_{t + d},
```

where ``\gamma_{t,d}`` is the baseline hazard of the discretised delay,
``\delta_t`` is a reference-date effect (a random walk in reporting speed) and
``\epsilon_{t+d}`` is a report-date effect (a day-of-week term, indexed by the
calendar report date ``t + d``).
The hazard reconstructs the per-reference-date delay PMF ``p_{t,d}``, and the
expected count in cell ``(t, d)`` is ``\mathbb{E}[n_{t,d}] = \lambda_t \,
p_{t,d}``.
This is exactly what [`reference_report_matrix`](@ref) computes; see the
[model definition](https://package.epinowcast.org/articles/model.html).

## Packages used

We use Distributions for the delays, Turing and Mooncake for the fit,
FlexiChains for the chain output, AlgebraOfGraphics with CairoMakie for the
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
## The reporting model

### The branched delay stack

Both streams share the infection-to-onset incubation period, then branch: onset
to case report and onset to death.
We build the whole delay as one shared-origin [`compose`](@ref) stack of
[`double_interval_censored`](@ref) delays, exactly as in the renewal tutorial.
[`double_interval_censored`](@ref) applies primary-event censoring, truncation and daily
interval censoring in one call, so each branch is a daily-resolution delay.
"""

incubation = double_interval_censored(
    Gamma(1.8, 1.4); upper = 20.0, interval = 1.0)

onset_case = double_interval_censored(
    Gamma(1.5, 1.2); upper = 20.0, interval = 1.0)

onset_death = double_interval_censored(
    Gamma(3.0, 4.0); upper = 30.0, interval = 1.0)

delay_stack = compose(incubation; case = onset_case, death = onset_death)

md"""
The maximum reporting delay caps the number of delay bins we track per reference
date.
We read each branch's total delay PMF (incubation convolved with the branch tail)
straight out of the stack by pushing a unit impulse through
[`convolve_distributions`](@ref): a one-at-time-zero series convolved with a delay
PMF returns that PMF, and `events = (:case, :death)` returns both branch PMFs at
once.
This reuses the same shared-origin convolution the renewal tutorial uses, so the
baseline delay is exactly the stack's delay.
"""

max_delay = 14

impulse = let s = zeros(max_delay + 1)
    s[1] = 1.0
    s
end

branch_pmfs = convolve_distributions(delay_stack, impulse;
    events = (:case, :death))

base_case_pmf = branch_pmfs.case ./ sum(branch_pmfs.case)

base_death_pmf = branch_pmfs.death ./ sum(branch_pmfs.death)

md"""
We also keep the composed two-delay delay for each stream as a single
[`Convolved`](@ref) chain, the incubation chained with the branch tail.
[`convolve_distributions`](@ref) chains the two daily-censored delays into one
total delay.
We truncate and thin this composed delay below, showing right-truncation and
ascertainment acting on the composed stack itself rather than only on the count
matrix.
"""

composed_case = convolve_distributions(incubation, onset_case)

composed_death = convolve_distributions(incubation, onset_death)

md"""
### The expectation process

The expected final counts follow a log-normal random walk over reference date,
the epinowcast default:

```math
\log \lambda_t \sim \mathrm{Normal}(\log \lambda_{t-1}, \sigma).
```

A small helper turns an initial level, a vector of standardised innovations and
a step size into the expected-count path, so the forward demo and the fit build
``\lambda`` the same way.
"""

function lognormal_rw(log_lambda0, z, sigma)
    n = length(z) + 1
    log_lambda = Vector{eltype(z)}(undef, n)
    log_lambda[1] = log_lambda0
    for t in 2:n
        log_lambda[t] = log_lambda[t - 1] + sigma * z[t - 1]
    end
    return exp.(log_lambda)
end

md"""
### The hazard effects

Two effect families reshape the baseline delay hazard.
The reference-date effect ``\delta_t`` is itself a random walk: reporting speed
drifts over reference date.
The report-date effect ``\epsilon_s`` is a day-of-week term indexed by the
calendar report date ``s = t + d``, a length-seven vector of logit-hazard
shifts.
[`reference_report_matrix`](@ref) takes the reference effect as a per-reference
vector and the report effect as a callable `report_date -> shift`, applies them
through the hazard and forms the expected-count matrix, zeroing any cell whose
report date is after `now`.

Unlike the renewal layer's [`convolve_distributions`](@ref), which applies one
shared delay PMF across all reference dates, the nowcasting hazard gives each
reference date its own delay PMF (that is the whole point of a reference-date
effect), so the output is a reference-by-report matrix rather than a convolved
series.
The baseline PMF still comes from the same delay-discretisation machinery
(`_delay_pmf`, read above through a unit-impulse convolution); the hazard layer
only reshapes that PMF per reference date.
"""

function expected_matrix(expected, base_pmf, ref_effect, dow, now)
    report_effect = s -> dow[mod1(s, 7)]
    return reference_report_matrix(expected, base_pmf;
        reference_effects = ref_effect, report_effect = report_effect,
        now = now)
end

md"""
## Pure-Julia forward simulation

We pick a true expectation path, a true reference-effect random walk and a true
day-of-week pattern, and form the full (untruncated) and right-truncated
reference-by-report matrices for both streams.
"""

n_days = 35

now = n_days        # the real-time horizon: we observe up to day `n_days`

rng = MersenneTwister(20260611)

md"""
### Truncation on the composed two-delay stack

The hazard layer truncates the *count matrix*: a cell whose report date `t + d`
sits past `now` is dropped.
A complementary truncation acts on the *composed delay itself*.
The remaining window for reference date `t` is `now - t`, the time left before
the real-time horizon.
[`truncate_to_horizon`](@ref) right-truncates the composed two-delay chain to that
window, returning an ordinary `truncated` distribution whose normaliser is the
right-truncation correction `-logcdf(composed, window)`.
Because the intermediate onset event is not separately reported, the denominator
is the cdf of the *convolution* of the two delays, not of either delay alone,
which is the harder of the two right-truncation cases the package distinguishes.
The δ-bounded [`truncate_to_window`](@ref) restricts instead to a finite report
window `[now - t - δ, now - t]`, the reports that land in the last `δ` days.
"""

recent_window = now - (n_days - 5)

case_recent_trunc = truncate_to_horizon(composed_case, recent_window)

case_recent_window = truncate_to_window(composed_case, recent_window, 7.0)

md"""
The completeness probability of the composed two-delay chain is the fraction of a
reference date's reports that can have arrived by the horizon: it is the cdf of
the convolved incubation-plus-tail delay at the remaining window.
This is the stack-level analogue of the matrix `now` truncation, expressed once
on the composed delay rather than cell by cell, and it falls towards zero for the
most recent reference dates whose composed delay has barely had time to elapse.
"""

stack_completeness = [completeness_probability(composed_case, max(now - t, 0))
                      for t in 1:n_days]

md"""
The expectation random walk gives a smooth wave of expected final counts; deaths
sit far below cases through a fixed fraction (an infection-fatality-style scale).
"""

true_sigma = 0.08

true_z = randn(rng, n_days - 1)

true_lambda_case = lognormal_rw(log(120.0), true_z, true_sigma)

true_death_frac = 0.02

true_lambda_death = true_death_frac .* true_lambda_case

md"""
### Ascertainment

Not every case is reported.
We add an ascertainment layer: the observed cases are a thinned fraction of the
true cases, the same under-reporting EpiNow2 represents with a reporting fraction
on the observation model.
[`thin`](@ref) is the package's forward thinning op: `thin(delay, rho)` attaches
a fixed factor `rho` that [`convolve_distributions`](@ref) multiplies into the
delay's count series.
We thin the composed case delay and read the ascertainment factor back out of its
parameters, then apply the same factor to the case expectation, so the thinned
case rate in every cell is `true_rho` times the true rate.
Deaths are taken as fully ascertained, the usual assumption that deaths are
reported close to completely, so only the case stream is thinned.
"""

true_rho = 0.6

thinned_case_delay = thin(composed_case, true_rho)

ascertainment_factor = last(params(thinned_case_delay))

md"""
The reference-date effect is a slow random walk on the logit hazard (reporting
gradually speeds up), and the report-date effect is a weekday pattern: reports
dip at the weekend and rebound on Mondays.
"""

true_ref_effect = cumsum(vcat(0.0, 0.03 .* randn(rng, n_days - 1)))

## Mon..Sun shifts on the logit hazard.
true_dow = [0.4, 0.2, 0.0, -0.1, -0.2, -0.6, -0.5]

md"""
The forward matrices are the expected counts per (reference, report) cell.
The truncated matrices zero every cell whose report date is past `now`: these are
the data the nowcast will see, missing the late reports of the most recent
reference dates.
"""

case_full = expected_matrix(true_lambda_case, base_case_pmf, true_ref_effect,
    true_dow, nothing)

death_full = expected_matrix(true_lambda_death, base_death_pmf, true_ref_effect,
    true_dow, nothing)

case_trunc = expected_matrix(true_lambda_case, base_case_pmf, true_ref_effect,
    true_dow, now)

death_trunc = expected_matrix(true_lambda_death, base_death_pmf,
    true_ref_effect, true_dow, now)

md"""
Ascertainment thins the case rate by `ascertainment_factor`, the factor read out
of the thinned composed delay above.
The case matrices below are the *observed* (ascertained) case rates; the death
matrices are unthinned.
"""

case_full_obs = ascertainment_factor .* case_full

case_trunc_obs = ascertainment_factor .* case_trunc

md"""
We draw observed counts per cell as Poisson around the truncated, ascertained
expected matrix: this is the reporting triangle, one count per (reference date,
delay) that has actually been reported by `now`.
"""

case_obs = rand.(rng, Poisson.(case_trunc_obs .+ 1e-6))

death_obs = rand.(rng, Poisson.(death_trunc .+ 1e-6))

md"""
The count seen so far for each reference date is the row sum of the observed
triangle; the true final count is the row sum of the full (untruncated)
ascertained expected matrix.
For the most recent reference dates the seen count falls well below the truth,
because their late reports are still missing: that gap is what the nowcast fills.
"""

case_seen = vec(sum(case_obs; dims = 2))

case_truth = vec(sum(case_full_obs; dims = 2))

truncation_df = vcat(
    DataFrame(day = 1:n_days, count = case_truth, kind = "true final"),
    DataFrame(day = 1:n_days, count = Float64.(case_seen),
        kind = "seen by now"))

truncation_plot = data(truncation_df) *
                  mapping(:day, :count, color = :kind => "") *
                  visual(Lines)

draw(truncation_plot;
    figure = (; size = (800, 350)),
    axis = (; xlabel = "reference date", ylabel = "case count",
        title = "Right-truncation: seen vs true final counts"))

md"""
The two truncation routes track the same right-truncation, from different sides.
With the hazard effects switched off, the fraction of each reference date's
expected counts that lands at or before `now` is the row sum of the truncated
matrix over the row sum of the full matrix.
We compare that matrix-level fraction with the stack-level
[`completeness_probability`](@ref) of the composed two-delay delay at the same
remaining window.
Both saturate at one for old reference dates and fall towards zero for the most
recent ones, so the truncation pattern is the same.
They differ in level: the matrix fraction is built from the delay PMF
renormalised over the capped `max_delay` support, while the composed-delay cdf
keeps the full untruncated tail, so the stack-level completeness sits below the
matrix fraction for recent dates (`max_completeness_gap` measures the largest
difference).
"""

base_full = expected_matrix(true_lambda_case, base_case_pmf,
    zeros(n_days), zeros(7), nothing)

base_trunc = expected_matrix(true_lambda_case, base_case_pmf,
    zeros(n_days), zeros(7), now)

matrix_completeness = vec(sum(base_trunc; dims = 2)) ./
                      vec(sum(base_full; dims = 2))

max_completeness_gap = maximum(abs.(matrix_completeness .- stack_completeness))

md"""
## The Turing fit

The model rebuilds the same expectation random walk and hazard matrices inside a
`@model`, with the expectation innovations, the random-walk step, the
reference-effect innovations and the day-of-week pattern all sampled.
Each observed triangle cell is Poisson around the model's right-truncated
expected matrix, so the likelihood is exactly the right-truncation contribution:
unobserved (future) cells carry no term.

The whole hazard layer is AD-safe vector arithmetic, so the matrices recompute
under Mooncake reverse-mode AD with no special handling.
We share one reference-effect random walk and one day-of-week pattern across both
streams (they report through the same system), and give the death stream its own
expectation through a sampled death fraction.

One numerical guard: a right-truncated cell has expected mass exactly zero, and
an extreme `log`-scale random-walk proposal can push the expectation to a very
large value, so the product can momentarily evaluate to `Inf * 0 = NaN` before
NUTS rejects the step. A small `safe_rate` floor keeps every Poisson rate finite
and positive so the gradient stays defined; it does not change the model at any
sensible parameter value.
"""

safe_rate(x) = isfinite(x) ? max(x, 1e-6) : 1e-6

@model function epinowcast_model(case_obs, death_obs, base_case_pmf,
        base_death_pmf, now)
    n = size(case_obs, 1)

    ## Expectation random walk for cases.
    log_lambda0 ~ Normal(log(120.0), 0.5)
    sigma_rw ~ truncated(Normal(0.0, 0.15); lower = 0.0)
    z ~ filldist(Normal(0.0, 1.0), n - 1)
    lambda_case = lognormal_rw(log_lambda0, z, sigma_rw)

    ## Deaths are a sampled fraction of the case expectation.
    death_frac ~ Beta(1, 40)
    lambda_death = death_frac .* lambda_case

    ## Case ascertainment: an informative prior on the reported fraction, as
    ## EpiNow2 uses for under-reporting. Cases identify only `rho * lambda`, so
    ## the prior on the expectation level and this prior together separate the
    ## reported fraction from the underlying expectation.
    rho ~ Beta(4, 4)

    ## Reference-date hazard effect: its own slow random walk.
    sigma_ref ~ truncated(Normal(0.0, 0.1); lower = 0.0)
    zref ~ filldist(Normal(0.0, 1.0), n - 1)
    ref_effect = cumsum(vcat(zero(eltype(zref)), sigma_ref .* zref))

    ## Report-date day-of-week hazard effect: seven independent logit-hazard
    ## shifts, one per weekday, with no sum-to-zero constraint. The final-delay
    ## hazard is pinned to one inside the matrix, which anchors the overall
    ## level and identifies the weekday shifts relative to it.
    dow ~ filldist(Normal(0.0, 0.5), 7)
    report_effect = s -> dow[mod1(s, 7)]

    ## Right-truncated expected matrices, one per stream. Cases are thinned by
    ## the ascertainment fraction `rho`; deaths are fully ascertained.
    case_exp = rho .* reference_report_matrix(lambda_case, base_case_pmf;
        reference_effects = ref_effect, report_effect = report_effect,
        now = now)
    death_exp = reference_report_matrix(lambda_death, base_death_pmf;
        reference_effects = ref_effect, report_effect = report_effect,
        now = now)

    ## Likelihood over the observed (report date ≤ now) cells only.
    for t in 1:n, d in 1:size(case_obs, 2)

        if t + (d - 1) <= now
            case_obs[t, d] ~ Poisson(safe_rate(case_exp[t, d]))
            death_obs[t, d] ~ Poisson(safe_rate(death_exp[t, d]))
        end
    end
end

md"""
We fit with NUTS on Mooncake, sampling a short run that is enough to recover the
expectation path and the hazard effects at this size.
"""

model = epinowcast_model(
    case_obs, death_obs, base_case_pmf, base_death_pmf, now)

chain = sample(Xoshiro(1), model,
    NUTS(0.8; adtype = AutoMooncake(; config = nothing)),
    MCMCThreads(), 300, 4; chain_type = VNChain, progress = false)

md"""
## Recovery

### The expectation path and the nowcast

We reconstruct the posterior expectation random walk from the sampled initial
level and innovations, and the posterior reference effect and day-of-week pattern
the same way, then form the full (untruncated) expected matrices per posterior
draw.
The full matrix is thinned by the posterior ascertainment fraction `rho`, so the
nowcast is on the same reported-case scale as the observations.
The row sums of the thinned full matrix are the nowcast: the model's estimate of
the final reported counts including the reports still to come.
We summarise by the posterior mean and a 90% credible band.
"""

draw_keys = (:log_lambda0, :sigma_rw, :z, :death_frac, :rho,
    :sigma_ref, :zref, :dow)

draws = (; (k => chain[k] for k in draw_keys)...)

n_draws = length(draws.log_lambda0)

function nowcast_draw(i)
    lambda = lognormal_rw(draws.log_lambda0[i], draws.z[i], draws.sigma_rw[i])
    ref = cumsum(vcat(0.0, draws.sigma_ref[i] .* draws.zref[i]))
    M = expected_matrix(lambda, base_case_pmf, ref, draws.dow[i], nothing)
    return draws.rho[i] .* vec(sum(M; dims = 2))
end

nowcast_mat = reduce(hcat, (nowcast_draw(i) for i in 1:n_draws))

nowcast_mean = vec(mean(nowcast_mat; dims = 2))

nowcast_lo = [quantile(nowcast_mat[t, :], 0.05) for t in 1:n_days]

nowcast_hi = [quantile(nowcast_mat[t, :], 0.95) for t in 1:n_days]

md"""
The nowcast tracks the true final counts across the whole window, and in
particular lifts the most recent reference dates back up from their truncated
"seen" values to their true level, with the credible band widening where more of
the count is still unreported.
"""

nowcast_df = vcat(
    DataFrame(day = 1:n_days, count = case_truth, kind = "true final"),
    DataFrame(day = 1:n_days, count = Float64.(case_seen),
        kind = "seen by now"),
    DataFrame(day = 1:n_days, count = nowcast_mean, kind = "nowcast mean"))

band_df = DataFrame(day = 1:n_days, lo = nowcast_lo, hi = nowcast_hi)

nowcast_plot = (
    data(band_df) * mapping(:day, :lo, :hi) * visual(Band; alpha = 0.2) +
    data(nowcast_df) * mapping(:day, :count, color = :kind => "") *
    visual(Lines))

draw(nowcast_plot;
    figure = (; size = (800, 380)),
    axis = (; xlabel = "reference date", ylabel = "case count",
        title = "Nowcast vs truncated observations"))

md"""
### The hazard effects

We recover the day-of-week report effect (the posterior-mean logit-hazard shift
per weekday) and compare it to the truth.
The weekend dip and Monday rebound come through, identified purely from how the
reporting triangle fills in by report date.
"""

post_dow = vec(mean(reduce(hcat, draws.dow); dims = 2))

dow_recovery = (truth = round.(true_dow; digits = 2),
    posterior = round.(post_dow; digits = 2))

md"""
### Ascertainment recovery

The ascertainment fraction is recovered from the case counts.
The posterior mean and 90% interval of `rho` are reported against the truth: the
prior is centred at 0.5 but the data pull it towards the true 0.6, so the
reported fraction is learnt rather than imposed.
"""

post_rho = vec(draws.rho)

rho_recovery = (truth = true_rho,
    posterior_mean = round(mean(post_rho); digits = 3),
    lo = round(quantile(post_rho, 0.05); digits = 3),
    hi = round(quantile(post_rho, 0.95); digits = 3))

md"""
### The reference-by-report structure

A direct check on the delay model is the fit to the reporting triangle itself.
We compare the observed counts by delay (summed over reference dates) with the
posterior-mean expected counts by delay, the marginal delay profile the hazard
model implies.
"""

post_case_exp = let M = zeros(n_days, max_delay + 1)
    for i in 1:n_draws
        lambda = lognormal_rw(
            draws.log_lambda0[i], draws.z[i], draws.sigma_rw[i])
        ref = cumsum(vcat(0.0, draws.sigma_ref[i] .* draws.zref[i]))
        M .+= draws.rho[i] .*
              expected_matrix(lambda, base_case_pmf, ref, draws.dow[i], now)
    end
    M ./ n_draws
end

obs_by_delay = vec(sum(case_obs; dims = 1))

exp_by_delay = vec(sum(post_case_exp; dims = 1))

delay_df = vcat(
    DataFrame(delay = 0:max_delay, count = Float64.(obs_by_delay),
        kind = "observed"),
    DataFrame(delay = 0:max_delay, count = exp_by_delay,
        kind = "posterior mean"))

delay_plot = (
    data(@subset delay_df :kind .== "observed") *
    mapping(:delay, :count) * visual(Scatter, markersize = 6) +
    data(@subset delay_df :kind .== "posterior mean") *
    mapping(:delay, :count) * visual(Lines))

draw(delay_plot;
    figure = (; size = (800, 350)),
    axis = (; xlabel = "report delay (days)", ylabel = "case count",
        title = "Delay profile: observed vs posterior mean"))

md"""
## Summary

- The model is the epinowcast nowcasting model assembled from
  CensoredDistributions tools: a log-normal random-walk expectation, a shared-
  origin branched [`compose`](@ref) stack of [`double_interval_censored`](@ref) delays for
  the case and death streams, a discrete-time reporting hazard with reference-
  date and report-date effects, and real-time right-truncation.
- The hazard layer is the one new piece: [`reference_report_matrix`](@ref) turns
  a baseline delay PMF into a per-reference-date hazard, reshapes it by a
  reference-date random walk and a report-date day-of-week term, and forms the
  right-truncated expected-count matrix ``\lambda_t \, p_{t,d}``.
- Right-truncation is shown two ways: per cell on the count matrix (the `now`
  argument the fit uses) and per record on the composed two-delay delay, where
  [`truncate_to_horizon`](@ref) and [`completeness_probability`](@ref) act on the
  [`convolve_distributions`](@ref) chain of incubation and branch tail. Both
  routes show the same falling truncation pattern, differing in level because the
  matrix PMF is renormalised over the capped delay support.
- Cases are under-reported: an ascertainment layer thins the case stream by a
  reported fraction `rho`, built with [`thin`](@ref), the same observation
  thinning EpiNow2 uses for under-reporting.
- The whole layer is AD-safe, so the expectation, the hazard effects, the
  ascertainment fraction and the right-truncated counts recompute inside a Turing
  `@model` and fit with Mooncake reverse-mode AD.
- The fit recovers the expectation path, the day-of-week reporting pattern, the
  delay profile and the ascertainment fraction, and the nowcast lifts the most
  recent right-truncated reference dates back to their true final reported counts
  with a credible band that widens where more of the count is still to be
  reported.

### What is deferred

- The parametric baseline hazard is fixed here (a chosen delay distribution per
  branch); jointly inferring the delay parameters with the hazard effects (the
  full epinowcast ``\mu_{g,t}``, ``\upsilon_{g,t}`` regressions) is a natural
  extension.
- The streams are modelled separately after sharing the reference and report
  effects; a [`competing`](@ref) onset branch (case versus death as competing
  outcomes) would tie the branch probability to a case-fatality term.
- The composed-stack truncation is shown alongside the matrix truncation as a
  cross-check; folding the per-record stack truncation directly into the
  likelihood (rather than the matrix `now`) is a natural next step.
"""
