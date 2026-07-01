md"""
# [Recurrent multi-state transitions: waning and reinfection](@id recurrent-multistate)

## Introduction

The composed stack builds a finite, acyclic event tree: each event happens at
most once and times accumulate forward.
That cannot represent a cycle, where a path returns to a state and takes an
unbounded, random number of steps.
Waning immunity with reinfection
(`susceptible -> infected -> recovered -> susceptible -> ...`) and hospital
cycling (`admitted -> ward -> ICU -> ward -> ...`) are exactly that.

This tutorial uses the renewal-over-states model [`recur`](@ref), which lifts
the no-cycles limit while reusing the existing verbs: each state owns a
[`compete`](@ref) racing-hazard node over its outgoing edges, a path is the
unfolded renewal of those sojourns, and the per-step scoring is the same
cause-resolved term the package already computes.

### What are we going to do in this exercise

1. Build a waning-immunity reinfection model with a back edge that the acyclic
   grammar cannot express.
2. Simulate cyclic paths and read the jump chain.
3. Score a path and recover the sojourn parameters with Turing.
4. Show the memoryless CTMC fast path [`ctmc`](@ref) for panel data.

### What might I need to know before starting

This builds on the competing-outcomes ideas in
[A branching-process-like natural history with competing outcomes](@ref
branching-competing).
The semi-Markov sojourn here is the same clock-reset holding time that page
documents, now arranged in a cycle.
"""

# We work with the package, Distributions, a seeded RNG, and Turing for the fit.

using CensoredDistributions
using Distributions
using Random
using Turing
using DynamicPPL
using Statistics

# ## A reinfection cycle
#
# Each non-absorbing state maps to its outgoing edges. A single edge is a bare
# `dest => sojourn`; several edges build a racing-hazard `compete` node. The
# `recovered -> susceptible` edge closes the loop, and `dead` is absorbing.

model = recur(
    :susceptible => (:infected => Gamma(2.0, 4.0)),
    :infected => (:recovered => Gamma(2.0, 3.0), :dead => Gamma(2.0, 8.0)),
    :recovered => (:susceptible => Gamma(2.0, 30.0)))

# The transient states have outgoing edges; `dead` is the only absorbing state.

(transient = CensoredDistributions.transient_states(model),
    absorbing = CensoredDistributions.absorbing_states(model))

# ## Simulate cyclic paths
#
# `rand` runs the renewal: from each state it draws the winning next-state and
# its sojourn, resets the clock, and stops at an absorbing state or the horizon.
# A path that returns to `susceptible` is a genuine cycle.

path = rand(MersenneTwister(545), model; horizon = 120.0)

# The visited states show the cycle (susceptible appears more than once before
# any absorption).

CensoredDistributions.visited_states(path)

# ## Score a path
#
# The path log-density sums the per-sojourn competing-risks terms, plus the
# survival of an unfinished final sojourn if the horizon censored it.

logpdf(model, path)

# ## Fit the sojourn parameters with Turing
#
# `recurrent_states_model` samples each state's edge parameters from priors and
# returns the rebuilt model; the path log-likelihood is added with
# `@addlogprob!`. We simulate training paths from the model above and recover
# its sojourn scales.

Random.seed!(11)
paths = [rand(model; horizon = 200.0) for _ in 1:300]

priors = (
    susceptible = (infected = (shape = truncated(Normal(2, 0.5); lower = 0),
        scale = truncated(Normal(4, 1.5); lower = 0)),),
    infected = (
        recovered = (shape = truncated(Normal(2, 0.5); lower = 0),
            scale = truncated(Normal(3, 1); lower = 0)),
        dead = (shape = truncated(Normal(2, 0.5); lower = 0),
            scale = truncated(Normal(8, 2); lower = 0))),
    recovered = (susceptible = (shape = truncated(Normal(2, 0.5); lower = 0),
        scale = truncated(Normal(30, 6); lower = 0)),))

@model function fit_reinfection(template, priors, paths)
    m ~ to_submodel(recurrent_states_model(template, priors))
    for p in paths
        DynamicPPL.@addlogprob! logpdf(m, p)
    end
end

Random.seed!(1)
chain = sample(
    fit_reinfection(model, priors, paths), NUTS(), 500; progress = false)

# The posterior means recover the data-generating sojourn scales (truths 4, 3,
# 8, 30). The chain names are namespaced by state and edge.

(infect = mean(chain[Symbol("m.susceptible.infected.scale")]),
    recover = mean(chain[Symbol("m.infected.recovered.scale")]),
    die = mean(chain[Symbol("m.infected.dead.scale")]),
    wane = mean(chain[Symbol("m.recovered.susceptible.scale")]))

# ## The memoryless CTMC fast path
#
# When every sojourn is exponential, the process is a continuous-time Markov
# chain. `ctmc` builds the generator matrix, and the closed-form
# `P(t) = exp(Q t)` scores panel data (the state observed at fixed visit times,
# the transition times unknown) that the semi-Markov path cannot.

cmodel = ctmc(
    :susceptible => (:infected => 0.25),
    :infected => (:recovered => 0.33, :dead => 0.12),
    :recovered => (:susceptible => 0.033))

# A panel observation scores through the same `logpdf` front door: it dispatches
# on the observation shape, so a `(time, state)` panel multiplies the per-gap
# transition probabilities while a `(from, to, dwell)` jump chain scores the
# exact term. No bespoke panel-scoring name is needed.

panel = [(0.0, :susceptible), (5.0, :infected), (9.0, :recovered),
    (40.0, :susceptible)]
logpdf(cmodel, panel)

# `recur` reaches the same fast path automatically: when every edge is
# `Exponential` and every state races (no fixed-probability split), `recur`
# returns a [`CTMCStates`](@ref), so the memoryless model needs no separate
# constructor call.

auto = recur(
    :susceptible => (:infected => Exponential(1 / 0.25)),
    :infected => (:recovered => Exponential(1 / 0.33),
        :dead => Exponential(1 / 0.12)),
    :recovered => (:susceptible => Exponential(1 / 0.033)))
auto isa CTMCStates

# ## Repeating one sojourn across many states
#
# A progression through several identical stages (e.g. an Erlang-style chain of
# equal-rate steps) repeats one sojourn rather than spelling each step out.
# [`compose`](@ref)`(dist, n)` builds the repeated chain in one call, which then
# drops in as an edge sojourn like any other delay.

stagewise = compose(Gamma(2.0, 3.0), 3)
event_names(stagewise)

# ## Summary
#
# - [`recur`](@ref) builds a renewal-over-states model that admits cycles, with
#   each edge sojourn any `UnivariateDistribution`.
# - `rand` simulates a cyclic path and `logpdf` scores it by reusing the
#   competing-risks term; the path likelihood fits with Turing through
#   [`recurrent_states_model`](@ref).
# - [`ctmc`](@ref) is the memoryless fast path: exponential sojourns, a
#   generator matrix, and panel-data scoring via `exp(Q t)`. `logpdf` is the one
#   scoring front door for both panel and jump-chain data, and an
#   all-exponential [`recur`](@ref) dispatches to the CTMC automatically.
# - [`compose`](@ref)`(dist, n)` repeats one sojourn into an `n`-step chain.
