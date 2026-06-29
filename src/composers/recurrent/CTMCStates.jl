# ============================================================================
# CTMCStates: the memoryless (generator-matrix) fast path
# ============================================================================
#
# The default `RecurrentStates` mode is semi-Markov: any sojourn distribution,
# scored from the jump chain plus sojourn times (a line list). When every
# sojourn is EXPONENTIAL the process is a continuous-time Markov chain, and two
# things become available that the general semi-Markov path cannot offer:
#
#   - a closed-form transition-probability matrix `P(t) = exp(Q t)`, so PANEL
#     data (the state observed at fixed visit times, the transition times
#     UNKNOWN) is scored directly by multiplying `P(Δt)` entries -- no
#     marginalisation over the hidden jump chain;
#   - the exact jump-chain likelihood is the same exponential-sojourn special
#     case of the semi-Markov term, so a CTMC scores a line list too.
#
# `CTMCStates` holds the generator matrix `Q` (off-diagonal `q_ij >= 0` the rate
# of `i -> j`, diagonal `q_ii = -sum_{j != i} q_ij`) over an ordered state set.
# It is the memoryless FAST PATH, built with `ctmc(...)`; for non-exponential
# dwell times use the semi-Markov `recur(...)` default.

@doc raw"""

A continuous-time Markov chain over states: the memoryless fast path.

`CTMCStates` holds a transition-intensity (generator) matrix `Q` over an ordered
set of `states`. Off-diagonal `Q[i, j]` is the rate of the `i -> j` transition
and each diagonal is `Q[i, i] = -sum_{j != i} Q[i, j]`. Holding times are
exponential, so the transition-probability matrix is `P(t) = exp(Q t)` in closed
form. This makes PANEL data (the state observed at fixed visit times, transition
times unknown) tractable through [`transition_probability`](@ref), and the exact
jump-chain likelihood is the exponential-sojourn special case of the semi-Markov
[`RecurrentStates`](@ref) path.

Build it with the [`ctmc`](@ref) verb. For non-exponential sojourns use the
semi-Markov [`recur`](@ref) default instead.

# Fields
- `states`: the ordered state names (a tuple of `Symbol`s); the `Q` rows /
  columns follow this order.
- `Q`: the generator matrix (rows sum to zero).

# See also
- [`ctmc`](@ref): the constructor verb.
- [`RecurrentStates`](@ref): the semi-Markov default for non-exponential dwells.
- [`transition_probability`](@ref): the `exp(Q t)` panel-data kernel.
"""
struct CTMCStates{S, M}
    "Ordered state names; the `Q` rows / columns follow this order."
    states::S
    "The generator matrix (rows sum to zero)."
    Q::M

    function CTMCStates(states::S, Q::M) where {S <: Tuple, M <: AbstractMatrix}
        n = length(states)
        size(Q) == (n, n) || throw(ArgumentError(
            "the generator matrix must be $(n)x$(n) for $(n) states; got " *
            "$(size(Q))"))
        all(s -> s isa Symbol, states) ||
            throw(ArgumentError("every state name must be a Symbol"))
        _validate_generator(Q)
        return new{S, M}(states, Q)
    end
end

# A generator matrix has non-negative off-diagonals and zero-sum rows.
function _validate_generator(Q::AbstractMatrix)
    n = size(Q, 1)
    for i in 1:n
        rowsum = zero(eltype(Q))
        for j in 1:n
            if i != j
                Q[i, j] >= -1e-9 || throw(ArgumentError(
                    "off-diagonal generator rates must be non-negative; " *
                    "Q[$i, $j] = $(Q[i, j])"))
                rowsum += Q[i, j]
            end
        end
        isapprox(Q[i, i], -rowsum; atol = 1e-6) || throw(ArgumentError(
            "generator row $i must sum to zero (diagonal = -sum of " *
            "off-diagonals); got Q[$i, $i] = $(Q[i, i]), -rowsum = $(-rowsum)"))
    end
    return nothing
end

@doc raw"""

Build a [`CTMCStates`](@ref) memoryless model from `from => (to => rate, ...)`
transition-rate specifications.

Each argument is `from => transitions`, where `transitions` lists the outgoing
`to => rate` edges of state `from` (a single edge or a tuple / NamedTuple of
them). The state order is the order of first appearance (each `from`, then any
`to`-only absorbing states). The generator matrix is assembled with each
diagonal set to minus its row sum.

# Arguments
- `specs`: `from => transitions` pairs; `transitions` is a single `to => rate`
  edge or a tuple / NamedTuple of `to => rate` edges (`rate >= 0`).

# Examples
```@example
using CensoredDistributions

# An illness-death CTMC: well <-> ill, ill -> dead.
model = ctmc(
    :well => (:ill => 0.2),
    :ill => (:well => 0.3, :dead => 0.1))
transition_probability(model, 5.0)
```

# See also
- [`CTMCStates`](@ref): the model type.
- [`recur`](@ref): the semi-Markov default for non-exponential dwells.
"""
function ctmc(specs::Pair...)
    isempty(specs) && throw(ArgumentError("ctmc needs at least one state spec"))
    # State order: each `from` in spec order, then any absorbing `to` states.
    order = Symbol[]
    for (from, _) in specs
        from isa Symbol ||
            throw(ArgumentError("each state name must be a Symbol; got $from"))
        from in order || push!(order, from)
    end
    # Collect `(from, to, rate)` edges, keeping the rate element type so a
    # `Dual`/tracked rate flows into `Q` (the CTMC fit differentiates the
    # generator). The matrix element type is promoted from the rates.
    edges = Any[]
    for (from, transitions) in specs
        for (to, rate) in _ctmc_edges(transitions)
            to isa Symbol ||
                throw(ArgumentError("each edge destination must be a Symbol"))
            rate >= 0 || throw(ArgumentError(
                "transition rate $(from) -> $(to) must be non-negative; " *
                "got $rate"))
            to in order || push!(order, to)
            push!(edges, (from, to, rate))
        end
    end
    states = Tuple(order)
    idx = Dict(s => i for (i, s) in enumerate(states))
    n = length(states)
    T = isempty(edges) ? Float64 :
        promote_type(Float64, mapreduce(e -> typeof(e[3]), promote_type, edges))
    # `T` comes from `typeof` over the untyped `edges` vector, so inference widens
    # it to an abstract `Type`; assert the result is a 2-D `Matrix{T}` so the
    # `CTMCStates(states, Q)` call below dispatches against `M <: AbstractMatrix`
    # without a spurious higher-dimensional `Array` branch.
    Q = zeros(T, n, n)::Matrix{T}
    for (from, to, rate) in edges
        Q[idx[from], idx[to]] += rate
    end
    for i in 1:n
        Q[i, i] = -sum(Q[i, j] for j in 1:n if j != i; init = zero(T))
    end
    return CTMCStates(states, Q)
end

# Normalise a state's transition spec to an iterable of `to => rate` edges.
_ctmc_edges(p::Pair) = (p,)
_ctmc_edges(t::Tuple) = t
_ctmc_edges(nt::NamedTuple) = _nt_pairs(nt)

# --- accessors --------------------------------------------------------------

state_index(m::CTMCStates, s::Symbol) = findfirst(==(s), m.states)

@doc raw"""

The transition-probability matrix `P(t) = exp(Q t)` of a [`CTMCStates`](@ref)
model over a time gap `t`.

`P[i, j]` is the probability of being in state `j` a time `t` after being in
state `i`, marginalising over every intermediate jump. This is the kernel a
panel-data likelihood multiplies, and is available because the holding times are
exponential (the memoryless fast path).

# Arguments
- `m`: the [`CTMCStates`](@ref) model.
- `t`: the non-negative time gap.

# Examples
```@example
using CensoredDistributions

model = ctmc(:well => (:ill => 0.2), :ill => (:well => 0.3, :dead => 0.1))
P = transition_probability(model, 2.0)
sum(P; dims = 2)  # each row sums to one
```

# See also
- [`CTMCStates`](@ref): the model type.
- [`logpdf`](@ref): the panel-data and jump-chain likelihoods that use this.
"""
function transition_probability(m::CTMCStates, t::Real)
    t >= 0 || throw(ArgumentError("the time gap must be non-negative; got $t"))
    return _matrix_exp(m.Q .* t)
end

# Matrix exponential via scaling-and-squaring with a Taylor inner series. A
# small dependency-free implementation (LinearAlgebra is not a hard dep of the
# package); accurate for the small generators these models build and AD-friendly
# (plain `+`/`*`, so rates differentiate through it). The squaring count `s` is
# control flow derived from the NUMERIC norm (a plain Float64), so a `Dual`/
# tracked entry differentiates through the series and the squaring without `s`
# itself becoming a tracked quantity.
function _matrix_exp(A::AbstractMatrix)
    n = size(A, 1)
    T = promote_type(eltype(A), Float64)
    # Scale so the norm is small, sum the Taylor series, then square back. The
    # scaling count and the convergence test read the AD-stripped `_primal`
    # value (the same core hook the censoring layer uses), so a `Dual`/tracked
    # entry never flows into the integer `s` or the `Bool` break.
    nrm = float(_primal(maximum(sum(abs, A; dims = 2))))
    s = max(0, ceil(Int, log2(nrm + eps())))
    B = A ./ (2^s)
    E = _eye(T, n)
    term = _eye(T, n)
    for k in 1:30
        term = (term * B) ./ k
        E = E .+ term
        float(_primal(maximum(abs, term))) < 1e-15 && break
    end
    for _ in 1:s
        E = E * E
    end
    return E
end

# An identity matrix of element type `T` without importing LinearAlgebra.
function _eye(::Type{T}, n::Int) where {T}
    M = zeros(T, n, n)
    for i in 1:n
        M[i, i] = one(T)
    end
    return M
end

# --- panel-data likelihood --------------------------------------------------

@doc raw"""

Log-likelihood of a panel observation under a [`CTMCStates`](@ref) model.

A panel observation is the state seen at a sequence of visit times, with the
transition times UNKNOWN between visits. Its log-likelihood is
`sum_v log P(Δt_v)[s_v, s_{v+1}]`, where `P(Δt) = exp(Q Δt)` marginalises
over every hidden jump in the gap. This is the CTMC's advantage over the
semi-Markov path, which needs the transition times.

# Arguments
- `m`: the [`CTMCStates`](@ref) model.
- `panel`: an iterable of `(time, state)` observations, in increasing time
  order.

# Examples
```@example
using CensoredDistributions

model = ctmc(:well => (:ill => 0.2), :ill => (:well => 0.3, :dead => 0.1))
panel = [(0.0, :well), (3.0, :ill), (7.0, :well)]
panel_logpdf(model, panel)
```

# See also
- [`transition_probability`](@ref): the `exp(Q Δt)` kernel summed here.
- [`logpdf`](@ref): the exact jump-chain likelihood (transition times known).
"""
function panel_logpdf(m::CTMCStates, panel)
    obs = collect(panel)
    length(obs) >= 2 || throw(ArgumentError(
        "a panel needs at least two (time, state) observations"))
    total = 0.0
    for v in 1:(length(obs) - 1)
        (t0, s0) = obs[v]
        (t1, s1) = obs[v + 1]
        Δt = t1 - t0
        Δt >= 0 || throw(ArgumentError(
            "panel visit times must be non-decreasing; got Δt = $Δt"))
        i = state_index(m, s0)
        j = state_index(m, s1)
        (i === nothing || j === nothing) && throw(ArgumentError(
            "a panel observation names a state not in the model"))
        P = transition_probability(m, Δt)
        total += log(P[i, j])
    end
    return total
end

# --- exact jump-chain likelihood --------------------------------------------

@doc """

Log-density of an exactly-observed jump chain under a [`CTMCStates`](@ref).

When the transition times ARE known (a line list, not panel data) the CTMC
likelihood is the exponential-sojourn special case of the semi-Markov path: each
step contributes `log q_{from,to} - q_from * dwell` where `q_from` is the total
exit rate of `from`. A [`StatePath`](@ref) or any iterable of `(from, to,
dwell)` jumps is accepted; a `:censored` path adds the survival `-q_state *
remaining` of its unfinished final sojourn.

# Arguments
- `m`: the [`CTMCStates`](@ref) model.
- `path`: a [`StatePath`](@ref) or an iterable of `(from, to, dwell)` jumps.

# See also
- [`panel_logpdf`](@ref): the panel-data likelihood (transition times unknown).
"""
function logpdf(m::CTMCStates, path::StatePath)
    total = _ctmc_jumps_logpdf(m, path.jumps)
    if path.stop == :censored && path.censored_state !== nothing
        i = state_index(m, path.censored_state)
        total += -(-m.Q[i, i]) * path.censored_for
    end
    return total
end

logpdf(m::CTMCStates, jumps) = _ctmc_jumps_logpdf(m, jumps)

function _ctmc_jumps_logpdf(m::CTMCStates, jumps)
    total = 0.0
    for j in jumps
        i = state_index(m, j.from)
        k = state_index(m, j.to)
        (i === nothing || k === nothing) && throw(ArgumentError(
            "a jump names a state not in the model"))
        q_ik = m.Q[i, k]
        q_ik > 0 || throw(ArgumentError(
            "the model has no $(j.from) -> $(j.to) transition (rate zero)"))
        total += log(q_ik) - (-m.Q[i, i]) * j.dwell
    end
    return total
end

# --- simulate ---------------------------------------------------------------

@doc """

Simulate a path from a [`CTMCStates`](@ref) model (Gillespie / exponential
sojourns).

From each state, draw an exponential holding time at the total exit rate and the
next state from the rate-weighted categorical, stopping at an absorbing state
(zero exit rate), when `horizon` would be exceeded, or at the `max_jumps` cap.
Returns a [`StatePath`](@ref), the same record the semi-Markov path returns.

# Keyword Arguments
- `start`: the state to begin in (default: the first state).
- `horizon`: the calendar-time cap (default `Inf`).
- `max_jumps`: the jump cap (default `10_000`).

# See also
- [`CTMCStates`](@ref): the model type.
- [`rand`](@ref CensoredDistributions.RecurrentStates): the semi-Markov draw.
"""
function Base.rand(rng::AbstractRNG, m::CTMCStates;
        start::Union{Symbol, Nothing} = nothing,
        horizon::Real = Inf, max_jumps::Int = 10_000)
    origin = start === nothing ? first(m.states) : start
    state = origin
    jumps = NamedTuple{(:from, :to, :dwell), Tuple{Symbol, Symbol, Float64}}[]
    clock = 0.0
    for _ in 1:max_jumps
        i = state_index(m, state)
        exit_rate = -m.Q[i, i]
        exit_rate <= 0 && return StatePath(origin, jumps, clock, :absorbed)
        dwell = rand(rng, Exponential(1 / exit_rate))
        if clock + dwell > horizon
            return StatePath(origin, jumps, clock, :censored;
                censored_state = state, censored_for = horizon - clock)
        end
        dest = m.states[_sample_rate_target(rng, m.Q, i, exit_rate)]
        push!(jumps, (from = state, to = dest, dwell = dwell))
        clock += dwell
        state = dest
    end
    return StatePath(origin, jumps, clock, :maxjumps)
end

Base.rand(m::CTMCStates; kwargs...) = rand(default_rng(), m; kwargs...)

# Sample the next state index from row `i`'s rate-weighted categorical.
function _sample_rate_target(rng::AbstractRNG, Q::AbstractMatrix, i::Int,
        exit_rate::Real)
    n = size(Q, 2)
    u = rand(rng) * exit_rate
    acc = 0.0
    for j in 1:n
        j == i && continue
        acc += Q[i, j]
        u <= acc && return j
    end
    # Fall through to the last off-diagonal target on a rounding edge.
    return findlast(j -> j != i && Q[i, j] > 0, 1:n)
end

function Base.show(io::IO, m::CTMCStates)
    print(io, "CTMCStates(", length(m.states), " states)")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", m::CTMCStates)
    println(io, "CTMCStates continuous-time Markov chain")
    println(io, "  states: ", join(string.(m.states), ", "))
    for i in 1:length(m.states), j in 1:length(m.states)

        i != j && m.Q[i, j] > 0 &&
            println(io, "  ", m.states[i], " -> ",
                m.states[j], " @ rate ", round(m.Q[i, j]; digits = 4))
    end
    return nothing
end
