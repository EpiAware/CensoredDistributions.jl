# ============================================================================
# Path simulation and scoring for RecurrentStates
# ============================================================================
#
# A realisation of a `RecurrentStates` model is a PATH: a sequence of jumps.
# `StatePath` is the self-describing record (the cyclic analogue of the acyclic
# tree's named event record): the start state, the vector of `(from, to, dwell)`
# jumps, the calendar time accumulated, and whether the path was right-CENSORED
# at a horizon (it hit `horizon` mid-sojourn) rather than ABSORBED (it reached
# an absorbing state) or truncated at the jump cap.
#
# `rand` simulates a path; `logpdf` scores one. Scoring sums the per-sojourn
# transition log sub-densities -- exactly the `Compete` / `Resolve` cause-
# resolved term -- plus, for a horizon-censored path, the survival term of the
# final, still-running sojourn (the probability no edge fired by the remaining
# time). The sum factorises over steps because each clock resets, so the path
# likelihood is AD-safe and differentiates w.r.t. the edge parameters.

@doc """

A realisation of a [`RecurrentStates`](@ref) model: a state path.

`StatePath` records the simulated or observed path through the states: the
`start` state, the `jumps` taken (each a `(from, to, dwell)` `NamedTuple`), the
total `elapsed` calendar time, and the `stop` reason. `stop` is `:absorbed`
(reached an absorbing state), `:censored` (a horizon cut the final sojourn
short, so the last state is still running), or `:maxjumps` (hit the jump cap).

A `:censored` path carries the unfinished final state in `censored_state` and
the remaining time the final sojourn survived in `censored_for`, which
[`logpdf`](@ref) scores as a survival term.

# Fields
- `start`: the state the path began in.
- `jumps`: the completed jumps, each `(from, to, dwell)`.
- `elapsed`: total time across the completed jumps.
- `stop`: why the path ended (`:absorbed` / `:censored` / `:maxjumps`).
- `censored_state`: the unfinished state for a `:censored` path, else `nothing`.
- `censored_for`: the time the final sojourn survived for a `:censored` path,
  else `0`.

# See also
- [`RecurrentStates`](@ref): the model this realises.
"""
struct StatePath{T, J}
    start::Symbol
    jumps::J
    elapsed::T
    stop::Symbol
    censored_state::Union{Symbol, Nothing}
    censored_for::T
end

function StatePath(start::Symbol, jumps, elapsed::T, stop::Symbol;
        censored_state = nothing, censored_for = zero(elapsed)) where {T}
    return StatePath{T, typeof(jumps)}(
        start, jumps, elapsed, stop, censored_state, T(censored_for))
end

# The sequence of states visited, including the start and any censored tail.
function visited_states(p::StatePath)
    states = Symbol[p.start]
    for j in p.jumps
        push!(states, j.to)
    end
    return states
end

Base.length(p::StatePath) = length(p.jumps)

function Base.show(io::IO, p::StatePath)
    print(io, "StatePath(", p.start)
    for j in p.jumps
        print(io, " -", round(j.dwell; digits = 2), "-> ", j.to)
    end
    p.stop == :censored && print(io, " [censored in ", p.censored_state,
        " after ", round(p.censored_for; digits = 2), "]")
    print(io, ")")
    return nothing
end

# --- simulate (`rand`) ------------------------------------------------------

@doc """

Simulate a path from a [`RecurrentStates`](@ref) model.

Starting in the model's `start` state, repeatedly draw the winning next-state
and its sojourn from the current state's transition node, reset the clock on
entry, and stop at an absorbing state, when `horizon` would be exceeded
(right-censoring the final sojourn), or at the `max_jumps` cap. Returns a
[`StatePath`](@ref).

# Arguments
- `rng`: random number generator (the no-`rng` method uses the global default).
- `m`: the [`RecurrentStates`](@ref) model to simulate.

# Keyword Arguments
- `horizon`: the calendar-time cap; a sojourn that would cross it leaves the
  path `:censored` in its current state (default `Inf`, run to absorption).
- `max_jumps`: the maximum number of jumps before stopping with `:maxjumps`
  (default `10_000`); guards an unbounded cycle with no horizon.

# Examples
```@example
using CensoredDistributions, Distributions, Random

model = recur(
    :well => (:ill => Gamma(2.0, 5.0)),
    :ill => (:well => Gamma(2.0, 3.0), :dead => Gamma(2.0, 10.0)))
rand(MersenneTwister(1), model; horizon = 60.0)
```

# See also
- [`StatePath`](@ref): the returned record.
- [`logpdf`](@ref): score a path.
"""
function Base.rand(rng::AbstractRNG, m::RecurrentStates;
        horizon::Real = Inf, max_jumps::Int = 10_000)
    jumps = NamedTuple{(:from, :to, :dwell), Tuple{Symbol, Symbol, Float64}}[]
    state = m.start
    clock = 0.0
    for _ in 1:max_jumps
        is_absorbing(m, state) && return StatePath(
            m.start, jumps, clock, :absorbed)
        node = m.nodes[state]
        dest, dwell = _draw_edge(rng, node)
        if clock + dwell > horizon
            return StatePath(m.start, jumps, clock, :censored;
                censored_state = state, censored_for = horizon - clock)
        end
        push!(jumps, (from = state, to = dest, dwell = Float64(dwell)))
        clock += dwell
        state = dest
    end
    return StatePath(m.start, jumps, clock, :maxjumps)
end

Base.rand(m::RecurrentStates; kwargs...) = rand(default_rng(), m; kwargs...)

# Draw the winning `(dest, dwell)` from a transition node: a one_of node races /
# resolves through `rand_outcome`; a lone edge draws its single sojourn.
_draw_edge(rng::AbstractRNG, c::AbstractOneOf) = rand_outcome(rng, c)
_draw_edge(rng::AbstractRNG, p::Pair) = (p.first, rand(rng, p.second))

# --- score (`logpdf`) -------------------------------------------------------

@doc """

Log-density of a path under a [`RecurrentStates`](@ref) model.

Sums the per-sojourn transition log sub-densities of the observed jumps: a
racing-hazard step scores `log f_to(t) + sum_{k != to} log S_k(t)` (the
[`Compete`](@ref) cause-resolved term), a fixed-split step its branch-weighted
mixture term, and a lone edge its plain `logpdf`. A `:censored` path adds the
survival term of its unfinished final sojourn (the probability no edge fired in
the remaining time). The clock resets each step, so the sum factorises and is
AD-safe in the edge parameters.

The path may be a [`StatePath`](@ref) (e.g. from [`rand`](@ref)) or any iterable
of `(from, to, dwell)` jumps.

# Arguments
- `m`: the [`RecurrentStates`](@ref) model.
- `path`: a [`StatePath`](@ref) or an iterable of `(from, to, dwell)` jumps.

# Examples
```@example
using CensoredDistributions, Distributions, Random

model = recur(
    :well => (:ill => Gamma(2.0, 5.0)),
    :ill => (:well => Gamma(2.0, 3.0), :dead => Gamma(2.0, 10.0)))
path = rand(MersenneTwister(1), model; horizon = 60.0)
logpdf(model, path)
```

# See also
- [`StatePath`](@ref): the path record.
- [`rand`](@ref): simulate a path.
"""
function logpdf(m::RecurrentStates, path::StatePath)
    total = _jumps_logpdf(m, path.jumps)
    # A horizon-censored path adds the survival of its unfinished final sojourn:
    # the probability the current state had not yet transitioned by the
    # remaining time, the standard right-censoring contribution.
    if path.stop == :censored && path.censored_state !== nothing
        node = m.nodes[path.censored_state]
        total += _edge_logsurvival(node, path.censored_for)
    end
    return total
end

# Score an iterable of bare `(from, to, dwell)` jumps (no censoring tail).
logpdf(m::RecurrentStates, jumps) = _jumps_logpdf(m, jumps)

function _jumps_logpdf(m::RecurrentStates, jumps)
    total = 0.0
    for j in jumps
        haskey(m.nodes, j.from) || throw(ArgumentError(
            "the path visits state $(j.from), which has no outgoing edges " *
            "in the model (it is absorbing)"))
        total = total + _transition_logpdf(m.nodes[j.from], j.to, j.dwell)
    end
    return total
end

# The transition log sub-density of taking edge `dest` after dwell `t`.
# A racing-hazard node reuses the `Compete` cause-resolved term; a fixed split
# the `Resolve` conditioned term; a lone edge its plain `logpdf`.
function _transition_logpdf(c::Compete, dest::Symbol, t::Real)
    j = _edge_index(c, dest)
    return _hazard_cause_logpdf(c, j, t)
end

function _transition_logpdf(c::Resolve, dest::Symbol, t::Real)
    j = _edge_index(c, dest)
    return _one_of_condition_logpdf(c.branch_probs, c.delays[j], t, j)
end

function _transition_logpdf(p::Pair, dest::Symbol, t::Real)
    dest === p.first || throw(ArgumentError(
        "the path takes edge to $(dest) from a state whose only edge is to " *
        "$(p.first)"))
    return logpdf(p.second, t)
end

# The log survival of a node at `t`: the probability NO edge has fired by `t`.
# A racing node is the product survival `sum_k log S_k(t)`; a fixed split the
# log of `1 - sum_k p_k F_k(t)`; a lone edge `logccdf`. The survival terms route
# through the AD-safe `_logccdf_ad_safe` / `_cdf_ad_safe` helpers so a Gamma's
# shape / scale differentiates (the stock `logccdf(::Gamma)` / `cdf(::Gamma)`
# have no `Dual` rule and break ForwardDiff on the censored-survival term).
_edge_logsurvival(c::Compete, t::Real) = _hazard_logsurvival(c, t)
function _edge_logsurvival(c::Resolve, t::Real)
    # `1 - sum_k p_k F_k(t)`: the probability the resolution time exceeds `t`.
    surv = one(float(t)) -
           sum(ntuple(k -> c.branch_probs[k] * _cdf_ad_safe(c.delays[k], t),
        length(c.branch_probs)))
    return log(max(surv, zero(surv)))
end
_edge_logsurvival(p::Pair, t::Real) = _logccdf_ad_safe(p.second, t)

# Index of the edge leading to `dest` in a one_of node; errors if absent.
function _edge_index(c::AbstractOneOf, dest::Symbol)
    j = findfirst(==(dest), component_names(c))
    j === nothing && throw(ArgumentError(
        "the path takes an edge to $(dest), which is not an outgoing edge of " *
        "this state (edges: $(join(string.(component_names(c)), ", ")))"))
    return j
end
