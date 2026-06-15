md"""
# [A branching-process-like natural history with competing outcomes](@id branching-competing)

## Introduction

Individual-based and branching-process simulators build a per-case event
timeline by hand: each individual gets an infection time, and later events
(onset, severity, recovery or death, reporting) are drawn as delays anchored on
an earlier event. Some events occur only with a probability, some outcomes are
mutually exclusive, and which-and-when can be hazard-driven.

The composed stack expresses that whole pattern as one distribution, sampled and
scored from a single object.

### What are we going to do in this exercise

We build one natural-history object for a case and exercise the enriched
competing-outcome composition:

1. Map each modelling concept onto a composed primitive:

| Modelling concept | Composed primitive |
|---|---|
| event as a delay from a prior event | [`Sequential`](@ref) step |
| several events from one anchor | [`Parallel`](@ref) |
| an event that only sometimes occurs | a no-event [`competing`](@ref) branch |
| competing risks, which-and-when hazard-driven | racing-hazard [`competing`](@ref) |
| aggregate ascertainment on expected counts | [`thin`](@ref) on the forward series |

2. Build a toy single-type branching process as a stand-in for any
   individual-based simulator.
3. Assemble the per-case natural history as one composed distribution and
   sample event timelines from it.
4. Score the composed object in a Turing model and recover the parameters.

### What might I need to know before starting

This tutorial builds on [Getting Started with
CensoredDistributions.jl](@ref getting-started) and the composer reference,
[Composing censored distributions](@ref composer-toolkit).
The toy branching process below is a stand-in for any individual-based
simulator: it is illustrative only and uses plain `Distributions`.

## Packages used
"""

using CensoredDistributions
using Distributions
using Turing
using ADTypes: AutoForwardDiff
using DynamicPPL: prefix
using Random
using Statistics

md"""
## A toy single-type branching process

A minimal stand-in: each case has offspring `~ Poisson(R)`, and each offspring is
infected a generation-interval delay after its parent. We grow a few generations
and keep every case's infection time. This is illustrative; a real simulator's
transmission engine stays in the simulator.
"""

rng = MersenneTwister(2024)

R = 1.6
gen_interval = Gamma(2.0, 2.5)

function toy_branching(rng, R, gen_interval, n_generations)
    infection_times = [0.0]
    current = [0.0]
    for _ in 1:n_generations
        next = Float64[]
        for t in current
            noffspring = rand(rng, Poisson(R))
            for _ in 1:noffspring
                push!(next, t + rand(rng, gen_interval))
            end
        end
        append!(infection_times, next)
        current = next
        isempty(current) && break
    end
    return infection_times
end

infection_times = toy_branching(rng, R, gen_interval, 9)
length(infection_times)

md"""
## One composed object per case

Each case's natural history is one composed object anchored on its infection:

- `infection -> onset`: an incubation delay (a [`Sequential`](@ref) step).
- `onset -> report`: made OPTIONAL with a no-event [`competing`](@ref) branch — a
  case is reported with probability `ρ`, else no report time is written
  (Feature 1).
- `onset -> {death, recover}`: a racing-hazard [`competing`](@ref) off onset; the
  first of the two latent delays wins, and which-and-when is coupled (Feature 2).

We anchor the object on a latent primary (the sub-unit timing of infection) so a
single `rand` draws the whole event path.
"""

ρ = 0.6                       # report probability
incubation = primary_censored(LogNormal(1.5, 0.4), Uniform(0, 1))
report = competing(:report => (Gamma(2.0, 1.5), ρ),
    :none => (NoEvent(), 1 - ρ))
severity = competing(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))

natural_history = compose((onset = incubation,
    reporting = report,
    severity = severity))

md"""
The object names its events; `event_names` is the flat per-record key space.
"""

event_names(natural_history)

md"""
## Simulate a line list

`rand` the object once per case, anchored on the case's infection time. The
draw is a `NamedTuple` keyed by the event names; a non-occurring report leaves
its slot `missing`, and exactly one of `death` / `recover` is filled (the racing
winner). We shift each case's path by its infection time to a real line list.
"""

function simulate_case(rng, history, infection_time)
    draw = rand(rng, history)
    return map(v -> v === missing ? missing : v + infection_time, draw)
end

linelist = [simulate_case(rng, natural_history, t) for t in infection_times]
linelist[1:3]

md"""
A case with no report shows the `missing` report slot; a death case has a
`death` time and a missing `recover` (and vice versa).
"""

n_reported = count(r -> r.report !== missing, linelist)
n_death = count(r -> r.death !== missing, linelist)
n_recover = count(r -> r.recover !== missing, linelist)
(; n_cases = length(linelist), n_reported, n_death, n_recover)

md"""
## The racing-hazard derived split

For the racing-hazard severity node the winning probability of each cause is
DERIVED from the hazards, not a free parameter. `winning_probabilities` returns
the cause split, and the simulated death fraction matches it within Monte Carlo
error.
"""

wp = winning_probabilities(severity)

(; derived_death = wp.death,
    simulated_death = n_death / (n_death + n_recover))

md"""
## Forward view: per-outcome expected counts

Pushing an incidence series through the object with
[`convolve_distributions`](@ref) gives the per-outcome expected-count streams. For
the racing-hazard node each outcome's stream is the cause-resolved sub-density
`f_j ∏_{k≠j} S_k`, sub-stochastic and NOT renormalised: its total mass equals the
derived winning probability. The forward death-fraction agrees with the
simulation.
"""

series = zeros(80)
series[1] = 1.0
fwd = convolve_distributions(severity, series; events = (:death, :recover))

(; forward_death_mass = sum(fwd.death),
    forward_recover_mass = sum(fwd.recover),
    derived_death = wp.death)

md"""
The three views of the same object agree: the simulated argmin-cause frequency,
the derived winning probability, and the forward stream mass.
"""

md"""
## Score back: recover the racing-hazard delays with Turing

With the SAME object we score the line list and recover the severity delays in a
small Turing model. We rebuild the natural-history object from sampled
parameters and score each record through `composed_distribution_model`, which
self-dispatches on which outcome slot is observed (and handles the optional
report). The records are anchored at zero here (the infection time is the known
anchor), so we re-centre each record on its own infection time.
"""

records = [map(v -> v === missing ? missing : v - t, r)
           for (r, t) in zip(linelist, infection_times)]

@model function fit_severity(records)
    death_shape ~ truncated(Normal(2.0, 0.5); lower = 0.2)
    recover_shape ~ truncated(Normal(3.0, 0.5); lower = 0.2)
    severity = competing(:death => Gamma(death_shape, 3.0),
        :recover => Gamma(recover_shape, 2.0))
    history = compose((onset = incubation,
        reporting = report,
        severity = severity))
    for i in eachindex(records)
        obs ~ to_submodel(
            prefix(composed_distribution_model(history, records[i]),
                Symbol(:rec, i)), false)
    end
end

chain = sample(rng, fit_severity(records),
    NUTS(; adtype = AutoForwardDiff()), 200; progress = false)

md"""
The posterior means concentrate near the true racing-hazard shapes (`2.0` for
death, `3.0` for recover) within Monte Carlo error.
"""

(; death_shape = mean(chain[:death_shape]),
    recover_shape = mean(chain[:recover_shape]))

md"""
## Non-terminal whole-tree outcomes

A competing branch is not limited to a single leaf delay: an outcome may be a
WHOLE composer subtree (Feature 3). Here the `death` outcome carries its own
sub-chain `death -> burial`, so winning the death cause unfolds a further event,
while `recover` stays a leaf. The composer outcome contributes its subtree's
event slots, so `event_names` interleaves the sub-chain's `burial` event where
the death outcome sits.
"""

burial = primary_censored(Gamma(1.5, 1.0), Uniform(0, 1))
death_chain = Sequential((Gamma(2.0, 3.0), burial), (:onset_death, :death_burial))
severity_tree = competing(:death => (death_chain, 0.4),
    :recover => (Gamma(3.0, 2.0), 0.6))

history_tree = compose((onset = incubation, severity = severity_tree))
event_names(history_tree)

md"""
A single `rand` draws the whole path: a death case fills the `death` and
`burial` slots (its sub-chain), a recover case fills `recover` (the others
`missing`). The drawn record scores straight back through `logpdf`.
"""

tree_draw = rand(rng, history_tree)
(; tree_draw, tree_logpdf = logpdf(history_tree, tree_draw))

md"""
## Mapping back to a hand-rolled layer

Each step of this tutorial maps onto a row of the table in the introduction:

- the toy branching process is the simulator's transmission engine (kept
  separate);
- the per-case `compose(...)` object is the natural-history / observation layer a
  simulator would otherwise hand-roll;
- the no-event `competing` branch is the per-case detection / asymptomatic gate;
- the racing-hazard `competing` is the competing-risk severity outcome;
- `convolve_distributions(object, series)` is the aggregate forward observation
  layer.

A branching-process developer can replace a bespoke per-event delay / observation
layer with one composed object that both simulates each case and scores the
resulting line list.
"""
