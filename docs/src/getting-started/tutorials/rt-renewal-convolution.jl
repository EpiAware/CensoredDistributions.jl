md"""
# An Rt renewal model with delay convolution

## Introduction

### What are we going to do in this exercise

We build a small EpiNow2-style renewal model and fit a time-varying
reproduction number to two observed streams at once: reported cases and
deaths.
Both streams come from the same latent infections, but each is seen through
its own delay and its own thinning: cases are under-ascertained (only a
fraction of infections are ever reported), and deaths occur at the infection
fatality ratio.

The renewal recursion itself is user-side; the package's job is the
*observation layer*.
We push the latent infection series through a composed delay stack with
[`convolve_distributions`](@ref)`(stack, series)`, which discretises the
stack's delay to a PMF and convolves the series with it, returning expected
event counts at the same times.

We cover:

1. A pure-Julia forward demo: pick a true Rt, ascertainment and IFR, run the
   renewal step, convolve infections through the case and death delay stacks,
   thin each stream, and simulate observed cases and deaths.
2. A Turing fit: the same renewal + [`convolve_distributions`](@ref) +
   thinning recomputed inside a `@model`, fitting Rt, ascertainment and IFR to
   the simulated cases and deaths, and recovering the truth.

### What might I need to know before starting

This tutorial builds on [Getting Started with
CensoredDistributions.jl](@ref getting-started) and the composer reference,
[The composer toolkit](@ref composer-toolkit), which introduces
[`Sequential`](@ref) chains, [`compose`](@ref) and
[`convolve_distributions`](@ref).
We do not re-explain the composer basics here; this page is about using a
convolved delay stack as a renewal observation layer.

[`convolve_distributions`](@ref) is AD-safe: the discretised delay PMF depends
differentiably on the delay parameters and the vector convolution is linear,
so gradients flow through it and it can be called directly inside a `@model`
with no special wrapper.
We sample with non-Enzyme AD: ForwardDiff here (see [Automatic
differentiation backends](@ref ad-backends) for the support matrix).

## Packages used

We use Distributions for the delay distributions, Turing for the fit,
FlexiChains for the chain output, CairoMakie for plots, and Random and
Statistics for reproducibility and summaries.
"""

using CensoredDistributions
using Distributions
using Turing
using FlexiChains: Parameter
using CairoMakie
using Random
using Statistics
using ADTypes: AutoForwardDiff

md"""
## The model

Infections follow a renewal process driven by a time-varying reproduction
number ``R_t`` and a generation-interval PMF ``g``:

```math
I_t = R_t \sum_{s \ge 1} g_s \, I_{t-s}.
```

Each observed stream is the infection series seen through a reporting delay and
scaled by a thinning probability.
For cases the delay runs infection to symptom onset to report, and only a
fraction ``\alpha`` (the ascertainment) of infections is ever reported.
For deaths the delay runs infection to death, and the scale is the infection
fatality ratio ``\rho``:

```math
\text{cases}_t = \alpha \,(I \circledast d_{\text{case}})_t, \qquad
\text{deaths}_t = \rho \,(I \circledast d_{\text{death}})_t,
```

where ``\circledast`` is the causal discrete convolution that
[`convolve_distributions`](@ref)`(stack, series)` computes.
Thinning matters because the two streams disagree about scale: most infections
never appear as reported cases, and only a small fraction end in death, so a
model that ignores ascertainment and IFR cannot match both counts with one
infection series.

### The generation interval

The generation interval is a short discrete PMF over positive lags; we
discretise a Gamma with [`interval_censored`](@ref) and drop the zero lag, so
an infection can only generate new infections from the next day on.
"""

gen_dist = interval_censored(Gamma(2.5, 1.3), 1.0)

gi_max = 12

g = let raw = [pdf(gen_dist, Float64(s)) for s in 1:gi_max]
    raw ./ sum(raw)
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
### The observation delay stacks

Each stream's delay is a composed [`Sequential`](@ref) stack.
The case stack is infection to onset, then onset to report; the death stack is
the single infection-to-death delay.
[`convolve_distributions`](@ref)`(stack, infections)` discretises the stack's
total delay to a PMF and convolves the infection series with it, giving
expected counts before thinning.
"""

case_delay = Sequential((Gamma(1.8, 1.4), Gamma(1.5, 1.2)),
    (:infection_onset, :onset_report))

death_delay = Gamma(3.0, 4.0)

md"""
## Pure-Julia forward demo

We pick a true Rt path (a short piecewise level that dips below one and
recovers), a true ascertainment and a true IFR, run the renewal, convolve
through each delay stack and thin.
This exercises the full observation machinery with no Turing in sight.
"""

n_days = 60

true_Rt = vcat(fill(1.6, 20), fill(0.8, 20), fill(1.2, n_days - 40))

true_alpha = 0.3

true_rho = 0.012

I0 = 10.0

infections = renewal(true_Rt, g, I0)

md"""
Convolve the infection series through each delay stack, then thin by the
ascertainment and the IFR.
The convolution returns the stack's end-point event by default, the reported
case time for the case stack and the death time for the death stack.
"""

expected_cases = true_alpha .* convolve_distributions(case_delay, infections)

expected_deaths = true_rho .* convolve_distributions(death_delay, infections)

md"""
The expected counts are Poisson means; we draw observed counts to act as the
data the fit will see.
"""

rng = MersenneTwister(20260610)

cases_obs = rand.(rng, Poisson.(expected_cases))

deaths_obs = rand.(rng, Poisson.(expected_deaths))

md"""
The simulated streams share the same infection wave but sit at very different
scales: ascertainment pins the case counts well below infections, and the IFR
pins deaths far lower still.
The death stream also lags the case stream, because its delay is longer.
"""

let fig = Figure(size = (800, 500))
    ax1 = Axis(fig[1, 1]; ylabel = "infections", title = "Forward simulation")
    lines!(ax1, 1:n_days, infections; color = :black)

    ax2 = Axis(fig[2, 1]; ylabel = "cases")
    scatter!(ax2, 1:n_days, cases_obs; color = :steelblue, markersize = 6)
    lines!(ax2, 1:n_days, expected_cases; color = :steelblue)

    ax3 = Axis(fig[3, 1]; xlabel = "day", ylabel = "deaths")
    scatter!(ax3, 1:n_days, deaths_obs; color = :firebrick, markersize = 6)
    lines!(ax3, 1:n_days, expected_deaths; color = :firebrick)
    fig
end

md"""
## The Turing fit

The model puts priors on the reproduction number, the ascertainment and the
IFR, then recomputes expected cases and deaths with the *same* renewal,
[`convolve_distributions`](@ref) and thinning used in the forward demo.
We give Rt a small piecewise level per 20-day block, so the parameter count
stays low and the subprocess build is fast.
The convolution runs directly inside the `@model`; it is AD-safe, so ForwardDiff
differentiates through it with no special handling.
"""

n_blocks = 3

block_of(t) = min(n_blocks, fld(t - 1, 20) + 1)

@model function rt_renewal(cases, deaths, g, case_delay, death_delay, I0)
    n = length(cases)
    R_block ~ filldist(truncated(Normal(1.0, 0.5); lower = 0.1), n_blocks)
    alpha ~ Beta(2, 5)
    rho ~ Beta(1, 50)

    Rt = [R_block[block_of(t)] for t in 1:n]
    infections = renewal(Rt, g, I0)
    exp_cases = alpha .* convolve_distributions(case_delay, infections)
    exp_deaths = rho .* convolve_distributions(death_delay, infections)

    for t in 1:n
        cases[t] ~ Poisson(exp_cases[t] + 1e-6)
        deaths[t] ~ Poisson(exp_deaths[t] + 1e-6)
    end
end

md"""
We fit with NUTS on ForwardDiff and a modest budget, which is enough to
recover the block-level Rt, the ascertainment and the IFR at this size.
"""

model = rt_renewal(cases_obs, deaths_obs, g, case_delay, death_delay, I0)

chain = sample(Xoshiro(1), model, NUTS(0.8; adtype = AutoForwardDiff()), 400;
    progress = false)

md"""
## Recovery

We read posterior means for the block-level Rt, the ascertainment and the IFR
and compare them with the truth.
The true block Rt are the levels we set: 1.6, 0.8 and 1.2.
"""

R_draws = vec(chain[Parameter(@varname(R_block))])

post_R = [mean(getindex.(R_draws, b)) for b in 1:n_blocks]

post_alpha = mean(vec(chain[Parameter(@varname(alpha))]))

post_rho = mean(vec(chain[Parameter(@varname(rho))]))

true_R_block = [true_Rt[1], true_Rt[21], true_Rt[41]]

recovery = (
    R_block = (truth = true_R_block, posterior = post_R),
    ascertainment = (truth = true_alpha, posterior = post_alpha),
    ifr = (truth = true_rho, posterior = post_rho))

md"""
The fit recovers all three: the block-level Rt tracks the true dip-and-recover
path, and the ascertainment and IFR land near their true scales despite never
being observed directly.
The two streams together identify the scales that a single stream could not.
"""

let fig = Figure(size = (800, 350))
    ax = Axis(fig[1, 1]; xlabel = "day", ylabel = "Rt",
        title = "Rt recovery")
    lines!(ax, 1:n_days, true_Rt; color = :black, label = "truth")
    post_Rt = [post_R[block_of(t)] for t in 1:n_days]
    lines!(ax, 1:n_days, post_Rt; color = :steelblue, label = "posterior mean")
    hlines!(ax, [1.0]; color = :grey, linestyle = :dash)
    axislegend(ax)
    fig
end

md"""
## Summary

- The renewal recursion is user-side; the package supplies the observation
  layer through [`convolve_distributions`](@ref)`(stack, series)`, which pushes
  an infection series through a composed delay stack to expected event counts.
- Two streams come from one infection series: cases through an
  infection-onset-report [`Sequential`](@ref) stack thinned by ascertainment,
  and deaths through an infection-death delay thinned by the IFR.
- Thinning matters because the streams sit at different scales; fitting both at
  once identifies the ascertainment and IFR that a single stream cannot.
- [`convolve_distributions`](@ref) is AD-safe, so the same renewal, convolution
  and thinning run inside a Turing `@model` and recover Rt, ascertainment and
  IFR with ForwardDiff.
- A dedicated Turing entry point for the convolution observation layer is a
  separate follow-up; today the convolution is called directly in the model.
"""
