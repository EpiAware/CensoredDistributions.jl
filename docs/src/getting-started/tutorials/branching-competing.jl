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
| day-resolution event recorded as a date | a [`double_interval_censored`](@ref) leaf |
| an event that only sometimes occurs | a no-event [`competing`](@ref) branch |
| competing risks, which-and-when hazard-driven | racing-hazard [`competing`](@ref) |
| an outcome that continues into a further chain | a [`competing`](@ref) outcome holding a subtree |

2. Run a single-type branching process (a Galton-Watson process with a
   generation-interval delay) to get a set of infection times.
3. Assemble the per-case natural history as one composed distribution and
   sample event timelines from it.
4. Score the composed object in a Turing model and recover the parameters.

### What might I need to know before starting

This tutorial builds on [Getting Started with
CensoredDistributions.jl](@ref getting-started) and the composer reference,
[Composing censored distributions](@ref composer-toolkit).
The branching process below is the transmission engine: it draws infection
times, and the composed object handles every per-case event after infection.
The observed delays (incubation, reporting) are recorded to the day, so their
leaves are [`double_interval_censored`](@ref) — primary censoring on the
unobserved sub-day timing plus daily interval censoring on the recorded date.

## Packages used
"""

using CensoredDistributions
using Distributions
using Turing
using ADTypes: AutoForwardDiff
using DynamicPPL: to_submodel
using Random
using Statistics

md"""
## A single-type branching process

The transmission engine is a single-type branching process (a Galton-Watson
process): each case draws its offspring count from `Poisson(R)`, and each
offspring is infected a generation-interval delay after its parent. We grow it
for a fixed number of generations from a single seed and keep every case's
infection time. This is a genuine branching process, not a placeholder: the same
`infection_times` would come from any individual-based transmission model.
"""

rng = MersenneTwister(2024)

R = 1.6
gen_interval = Gamma(2.0, 2.5)

function branching_process(rng, R, gen_interval, n_generations)
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

infection_times = branching_process(rng, R, gen_interval, 9)
length(infection_times)

md"""
## One composed object per case

Each case's natural history is one composed object anchored on its infection:

- `infection -> onset`: an incubation delay. Onset is recorded as a date, so the
  leaf is [`double_interval_censored`](@ref): primary censoring on the unobserved
  sub-day infection timing, then daily interval censoring on the recorded onset
  date.
- `onset -> report`: a reporting delay, also recorded to the day
  ([`double_interval_censored`](@ref)), and made OPTIONAL with a no-event
  [`competing`](@ref) branch — a case is reported with probability `ρ`, else no
  report time is written.
- `onset -> {death, recover}`: a racing-hazard [`competing`](@ref); the first of
  the two latent delays wins, and which-and-when is coupled. These are the
  cause-specific latent times that drive the competing-risk hazards, so they
  stay continuous — the derived winning split below integrates over them.

A single `rand` draws the whole event path.
"""

ρ = 0.6                       # report probability
incubation = double_interval_censored(LogNormal(1.5, 0.4); interval = 1)
report = competing(
    :report => (double_interval_censored(Gamma(2.0, 1.5); interval = 1), ρ),
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
parameters and score the WHOLE line list in one `~` through the batch
[`composed_distribution_model`](@ref) — pass the vector of records and the
competing node self-dispatches per record on which outcome slot is observed (and
handles the optional report and the missing slots internally), with no manual
per-record loop. The records are anchored at zero here (the infection time is
the known anchor), so we re-centre each record on its own infection time.
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
    obs ~ to_submodel(composed_distribution_model(history, records))
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
## Outcomes that continue into a further chain

A competing outcome is not limited to a single leaf delay: an outcome can carry a
WHOLE composer subtree, so winning that outcome unfolds further events. Here the
`death` outcome carries its own sub-chain `death -> burial` (the burial date is
recorded to the day, a [`double_interval_censored`](@ref) leaf), while `recover`
stays a single leaf. The outcome contributes its subtree's event slots, so
`event_names` shows the sub-chain's `burial` event where the death outcome sits.
"""

burial = double_interval_censored(Gamma(1.5, 1.0); interval = 1)
death_chain = Sequential((Gamma(2.0, 3.0), burial),
    (:onset_death, :death_burial))
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

- the branching process is the simulator's transmission engine (kept separate);
- the per-case `compose(...)` object is the natural-history / observation layer a
  simulator would otherwise hand-roll;
- the [`double_interval_censored`](@ref) leaves are the day-resolution recorded
  events (onset and report dates);
- the no-event `competing` branch is the per-case detection / asymptomatic gate;
- the racing-hazard `competing` is the competing-risk severity outcome;
- a `competing` outcome holding a subtree is an outcome that opens a further
  event chain (death then burial);
- `convolve_distributions(object, series)` is the aggregate forward observation
  layer.

A branching-process developer can replace a bespoke per-event delay / observation
layer with one composed object that both simulates each case and scores the
resulting line list.
"""
