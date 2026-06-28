md"""
# [A susceptibility-depleting renewal model](@id renewal-susceptibility)

## Introduction

### What are we going to do in this exercise

We build an SIR-style renewal model with a depleting susceptible pool and follow
it through to observed reported cases.
Infections follow the renewal recurrence modulated by the remaining susceptible
fraction; we then push the infections through an observation delay to reported
counts, simulate data, and fit the model with Turing to recover the
reproduction-number level and the susceptible-pool size.

Unlike the [Rt renewal with delay convolution](@ref rt-renewal-convolution)
tutorial, which hand-rolls the renewal loop, here the recurrence is a single
[`renewal`](@ref) call with a [`susceptibility_depletion`](@ref) modulator, and
the fit uses the [`renewal_model`](@ref) submodel so the renewal fits like the
rest of the stack.

We cover:

1. A forward simulation: a generation interval, an Rt path, a susceptible pool,
   and an observation delay, run through [`renewal`](@ref) and
   [`observe_renewal`](@ref) to expected cases.
2. A Turing fit: [`renewal_model`](@ref) samples the Rt path and the
   susceptible-pool size, and the observed counts are scored against the
   reported series.
3. Recovery: pull the posterior and check the Rt level and the pool size.

### What might I need to know before starting

This builds on [Getting Started](@ref getting-started) and the
[Rt renewal with delay convolution](@ref rt-renewal-convolution) tutorial, which
introduces the generation interval and the
[`convolve_distributions`](@ref) observation layer.

The renewal recurrence `I[t] = R_t · (S[t]/N) · Σ_s g_s I[t-s]`,
`S[t] = S[t-1] − I[t]`, is the [`susceptibility_depletion`](@ref) modulator.
Modulators compose with [`combine_modulators`](@ref), so a transmissibility or
immunity-waning term stacks on top with no change to the call.

## Packages used

We use Distributions for the delay families, Turing and Mooncake for the fit,
FlexiChains for the chain output, and Random and Statistics for reproducibility
and summaries.
"""

using CensoredDistributions
using Distributions
using Turing
using Turing: @varname
using Mooncake
using FlexiChains: VNChain
using Random
using Statistics
using ADTypes: AutoMooncake

md"""
## The generation interval

The generation interval is a short discrete PMF over positive lags, built as a
truncated Gamma read off in day bins with [`interval_censored`](@ref).
We drop the zero lag (`lower = 1`) so an infection generates from the next day
on, and the day-bin masses already sum to one over the positive lags.
"""

gi_max = 12

gen_dist = interval_censored(
    truncated(Gamma(2.5, 1.3); lower = 1.0, upper = Float64(gi_max)), 1.0)

g = pdf(gen_dist, 1:gi_max)

md"""
## The forward simulation

We pick a true Rt path that rises, dips below one and recovers, a susceptible
pool `N`, and a seed.
[`renewal`](@ref) with [`susceptibility_depletion`](@ref) runs the SIR-style
recurrence; the susceptible fraction bends the epidemic down as the pool runs
out.
"""

n_days = 90

true_Rt = vcat(fill(1.8, 25), fill(0.8, 25), fill(1.3, n_days - 50))

true_N = 1.0e4

I0 = 10.0

infections = renewal(true_Rt, g, I0;
    modulator = susceptibility_depletion(true_N))

md"""
## The observation layer

Infections are reported after an incubation-to-onset delay, and only a fraction
are ascertained.
We build the delay with [`double_interval_censored`](@ref) and carry the
ascertainment through [`thin`](@ref), then push the infections through with
[`observe_renewal`](@ref), the renewal-to-observation bridge.
"""

true_ascertainment = 0.4

onset_delay = thin(
    double_interval_censored(Gamma(1.8, 1.4); upper = 20.0, interval = 1.0),
    true_ascertainment)

expected_cases = observe_renewal(infections, onset_delay)

md"""
The expected counts are Poisson means; we draw observed counts as the data the
fit will see.
"""

rng = MersenneTwister(20260626)

cases_obs = rand.(rng, Poisson.(expected_cases .+ 1.0e-6))

md"""
## The Turing fit

[`renewal_model`](@ref) takes the Rt path and the modulator priors, runs the
renewal, and returns the infections.
We report them through the same observation delay and score the observed counts.
The `make_modulator` closure maps the sampled `logN` to a
[`susceptibility_depletion`](@ref) modulator, so the fit and the simulation
share the recurrence.
Here we hold Rt at its known path (passed as a plain vector, so no Rt parameters
enter the chain) and estimate the susceptible-pool size, the quantity the
depletion adds over the bare renewal.
The pool prior is bounded: at very large `N` the susceptible fraction saturates
near one and the depletion gradient vanishes, so a bounded prior keeps the
sampler on the identifiable slope.
The [Rt renewal with delay convolution](@ref rt-renewal-convolution) tutorial
covers estimating the Rt path itself.
"""

mod_priors = (N = truncated(Normal(1.0e4, 3.0e3); lower = 4.0e3,
    upper = 2.5e4),)

make_mod = p -> susceptibility_depletion(p.N)

@model function renewal_fit(g, I0, Rt, mod_priors, make_mod, delay, cases)
    infections ~ to_submodel(renewal_model(g, I0, Rt;
        modulator_priors = mod_priors, make_modulator = make_mod))
    expected = observe_renewal(infections, delay)
    cases ~ product_distribution(Poisson.(expected .+ 1.0e-6))
    return infections
end

model = renewal_fit(g, I0, true_Rt, mod_priors, make_mod, onset_delay,
    cases_obs)

chain = sample(Xoshiro(1), model,
    NUTS(0.95; adtype = AutoMooncake(; config = nothing)),
    200; chain_type = VNChain, progress = false);

md"""
## Recovery

We read the posterior susceptible-pool size back from the chain.
The pool size is namespaced under the `renewal_model` submodel prefix, and the
posterior mean recovers the data-generating value.
"""

recovered_N = mean(chain[@varname(infections.N.x)])

md"""
The recovered pool size sits near the true `N`, and the renewal fits through the
same [`renewal`](@ref) recurrence the simulation used — the
susceptibility-depletion story carried end to end from incidence to reported
cases.

This is the population-renewal side of the individual-path correspondence the
recurrent multi-state work (#545) and the convolve-loop population view (#759)
build out; those are separate from this scalar-incidence renewal step.
"""
