# Recurrent / cyclic multi-state transitions (#545)

Status: implemented.
The maintainer approved approach B (renewal over states) as the default with
approach A (CTMC) as a memoryless fast path; both are built.
This page keeps the design comparison as the rationale and records what shipped.

## The gap

The composed object today is a finite, acyclic event tree.
Each event happens at most once and times accumulate forward through
`sequential` / `parallel` / `resolve` / `compete`.
That cannot represent a recurrent or back-and-forth transition, where a path
forms a cycle and takes an unbounded, random number of steps.
The motivating epidemiological cases are waning immunity with reinfection
(`recovered -> susceptible -> infected -> ...`) and hospital cycling
(`admitted -> ward -> ICU -> ward -> ...`).

The existing branching-competing tutorial already names this limit honestly in
its "What this is not" box: no cycles, no calendar-time intensities.
This proposal scopes how to lift the no-cycles limit.

## Two approaches

### A. Continuous-time Markov chain (generator matrix)

Represent the system as a CTMC with a transition-intensity matrix `Q`.
State holding times are exponential, the next state is categorical with
probabilities `q_ij / -q_ii`, and the path likelihood is the standard
jump-chain-plus-sojourn product (the `msm` model).

Strengths.
The memoryless structure gives closed-form transition-probability matrices
`P(t) = exp(Qt)`, which makes panel data (state observed at fixed visit times,
transition times unknown) tractable by matrix exponential.
A compact, familiar object for survival/biostatistics readers.

Weaknesses.
Holding times are forced exponential.
Epidemiological dwell times are rarely exponential (incubation, hospital
length-of-stay), and the package's whole reason to exist is non-exponential,
censored delays.
Recovering non-exponential sojourns needs phase-type expansion (extra latent
sub-states via the linear-chain trick), which inflates the state space and
sits awkwardly beside the existing leaf-as-distribution idiom.
A generator matrix is a parallel engine: it does not reuse `compete`,
`sequential` or the censoring wrappers, so it duplicates the grammar rather
than extending it.

### B. Renewal over states (semi-Markov, recommended)

Represent the system as a state graph where each non-absorbing state owns a
`compete(...)` node over its outgoing edges (the cause-specific sojourn
distributions).
A path is an unfolded renewal: from the current state, draw the winning
next-state and its dwell time from that state's competing-risks node, reset the
clock on entry to the next state, and repeat until an absorbing state or a
horizon.
Cycles are allowed because an edge may point back to an already-visited state.

Strengths.
The sojourn on each edge is ANY `UnivariateDistribution` (tenet 1), so
non-exponential, censored, modified delays compose unchanged.
Every per-step quantity is the existing verb machinery: the cause-resolved log
sub-density `log f_j(t) + sum_{k != j} log S_k(t)` is exactly what the
`Compete` node already computes, the winning-cause draw is `rand_outcome`, and
the derived split is `probs`.
The clock-reset sojourn is the semi-Markov holding time the multi-state framing
tutorial already documents.
The likelihood factorises over completed sojourns, so it is AD-safe and fits
with the same tooling the package uses elsewhere.
The single new idea over today's grammar is the loop that keeps taking steps;
nothing else is new.

Weaknesses.
The semi-Markov jump-chain likelihood needs the transition TIMES (a line
list), so panel data (state-at-visit only) needs an extra marginalisation the
CTMC gets for free via `P(t)`.
A still-running final sojourn is right-censored and must be scored as a
survival term rather than a density (a production item, not a blocker).
The CTMC's `exp(Qt)` closed forms are unavailable, so panel-likelihood and some
analytic quantities need numeric work.

## Recommendation: B, the renewal-over-states mode

It extends the existing idiom rather than bolting on a parallel CTMC engine.
The semi-Markov sojourn IS the package's core object (a censored,
non-exponential delay), the per-step scoring IS the existing `Compete` node,
and the only addition is unfolding the steps in a loop.
This matches north-star tenet 5 (renewal / branching are compositions: the
convolution / racing primitive applied recursively over time) and keeps a
cyclic model on the same leaf-as-distribution footing as everything else.
The CTMC's one genuine advantage (panel data via `exp(Qt)`) is a narrower use
case for this package's line-list-first data, and the memoryless restriction it
imposes is exactly the restriction the package exists to avoid.
A CTMC fast path for the memoryless special case can be added later behind the
same state-graph front-end if panel data becomes a priority.

## What shipped

The renewal-over-states default lives in `src/composers/recurrent/`:

- `recur(...)` builds a `RecurrentStates` model: each state owns a `Compete` /
  `Resolve` / lone-edge transition node, and a back edge gives a cycle. Any
  `UnivariateDistribution` (plain, censored or composed) is a valid edge
  sojourn.
- `rand` simulates a `StatePath` of `(from, to, dwell)` jumps to absorption or a
  horizon; `logpdf` scores a path by reusing the `Compete` cause-resolved
  sub-density, and adds the survival term of a horizon-censored final sojourn
  (right-censoring).
- `ctmc(...)` builds the memoryless `CTMCStates` fast path: a generator matrix
  with closed-form `transition_probability` `P(t) = exp(Q t)`, scored through the
  dispatching `logpdf` front door (a `(time, state)` panel uses `exp(Q t)`, a
  `(from, to, dwell)` jump chain the exponential-sojourn term). The matrix
  exponential is AD-safe. An all-exponential `recur(...)` auto-dispatches to this
  representation, and `ctmc(::RecurrentStates)` converts an existing model.
- `recurrent_states_model(template, priors)` (DynamicPPL extension) samples each
  state's edge priors through the shared `_params_submodel` and rebuilds the
  model, so a cyclic model fits with `@addlogprob! logpdf(m, path)` like the
  rest of the stack.

Tests cover construction / validation, cyclic simulation, scoring, censoring,
the CTMC panel likelihood, and a Turing fit that recovers the sojourn scales.
The four AD-relevant paths (the reinfection path, a horizon-censored path, the
CTMC jump-chain and the CTMC panel `exp(Qt)`) are in the ADFixtures
`:recurrent` scenario group, so the full per-backend matrix (ForwardDiff,
ReverseDiff, Mooncake forward+reverse, Enzyme forward+reverse) exercises them,
with the genuine compiled-backend gaps registered broken (Mooncake reverse on
the `Compete`-node path; Enzyme reverse on the `Dict`-backed rebuild; Enzyme
forward on the matrix exponential). The
[Recurrent multi-state transitions](@ref recurrent-multistate) tutorial works a
reinfection cycle end to end.

The self-contained design `prototypes/recurrent_states_545.jl` is kept as the
original proof of concept.

## Deferred production scope

These extend the shipped feature and are not yet implemented:

- panel observation for the SEMI-Markov (non-exponential) path: it needs a
  transition-probability recursion the exponential CTMC gets free via `exp(Qt)`;
- calendar-time-varying (piecewise-constant) edge intensities;
- a richer event-record schema so a cyclic model reads back through
  `event_names` / `params_table` / `update(d, chain)` the way the acyclic tree
  does (the per-state priors front-door is in via `recurrent_states_model`);
- ODE lowering of these representations (tracked separately in #758).
