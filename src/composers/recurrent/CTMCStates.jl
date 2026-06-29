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
    # State order: each `from` in spec order, then any `to`-only absorbing
    # states. This pass reads only the state Symbols, never a rate value.
    order = Symbol[]
    for (from, transitions) in specs
        from isa Symbol ||
            throw(ArgumentError("each state name must be a Symbol; got $from"))
        from in order || push!(order, from)
        for (to, _rate) in _ctmc_edges(transitions)
            to isa Symbol ||
                throw(ArgumentError("each edge destination must be a Symbol"))
            to in order || push!(order, to)
        end
    end
    states = Tuple(order)
    idx = Dict(s => i for (i, s) in enumerate(states))
    n = length(states)
    # Generator element type from the rate TYPES only -- a `Dual`/tracked rate
    # widens `T` without its value ever entering an untyped container. The
    # earlier code collected `(from, to, rate)` into `edges = Any[]` and read the
    # rates back from that `Vector{Any}`. Boxing the active rate into the untyped
    # array dropped its AD identity, and worst of all SILENTLY: Enzyme FORWARD
    # returned a WRONG (finite) gradient and Enzyme REVERSE aborted on the `ctmc`
    # MethodInstance (`EnzymeNoTypeError`, "copy untyped data"). Assembling `Q`
    # straight from the typed spec tuple keeps the rate dataflow traceable for
    # ForwardDiff / ReverseDiff / Mooncake (forward AND reverse), which all now
    # match the ForwardDiff reference. Enzyme still cannot compile this builder
    # (`EnzymeInternalError`, an upstream Enzyme compiler bug on the
    # heterogeneous `Pair...` iteration / runtime-typed `Q`), but it now FAILS
    # LOUDLY rather than silently returning a wrong gradient -- so both CTMC
    # scenarios are honestly registered Enzyme-broken in the AD fixtures rather
    # than trusted (see `test/ADFixtures` `backend_broken_scenarios`).
    T = _ctmc_rate_type(Float64, specs...)
    # Assert a 2-D `Matrix{T}` so `CTMCStates(states, Q)` dispatches against
    # `M <: AbstractMatrix` without a spurious higher-dimensional `Array` branch.
    Q = zeros(T, n, n)::Matrix{T}
    for (from, transitions) in specs
        i = idx[from]
        for (to, rate) in _ctmc_edges(transitions)
            rate >= 0 || throw(ArgumentError(
                "transition rate $(from) -> $(to) must be non-negative; " *
                "got $rate"))
            Q[i, idx[to]] += rate
        end
    end
    for i in 1:n
        Q[i, i] = -sum(Q[i, j] for j in 1:n if j != i; init = zero(T))
    end
    return CTMCStates(states, Q)
end

# Promote the generator element type from the edge rate TYPES, recursing over
# the spec tuple so no active rate value is ever stored in an untyped container
# (see `ctmc` for why that boxing breaks Enzyme). `typeof` is non-differentiable,
# so reading each rate here contributes no tangent.
_ctmc_rate_type(T::Type) = T
function _ctmc_rate_type(T::Type, spec::Pair, rest::Pair...)
    for (_to, rate) in _ctmc_edges(spec.second)
        T = promote_type(T, typeof(rate))
    end
    return _ctmc_rate_type(T, rest...)
end

# Normalise a state's transition spec to an iterable of `to => rate` edges.
_ctmc_edges(p::Pair) = (p,)
_ctmc_edges(t::Tuple) = t
_ctmc_edges(nt::NamedTuple) = _nt_pairs(nt)

@doc raw"""

Convert a memoryless [`RecurrentStates`](@ref) model to its [`CTMCStates`](@ref)
generator-matrix representation.

A renewal-over-states model is a continuous-time Markov chain exactly when every
edge sojourn is `Exponential` and every state RACES its edges (a [`Compete`](@ref)
node or a lone edge, never a fixed-probability [`Resolve`](@ref) split). For such
a model the per-edge rate is `1 / scale` of its `Exponential`, and the generator
`Q` collects those rates. This is the conversion [`recur`](@ref) performs
automatically, exposed here for a `RecurrentStates` built or edited by hand. It
errors if the model is not memoryless.

# Arguments
- `m`: a memoryless [`RecurrentStates`](@ref) (all-`Exponential`, all-racing).

# Examples
```@example
using CensoredDistributions, Distributions

# Built directly as a RecurrentStates, then converted to the CTMC fast path.
rs = CensoredDistributions.RecurrentStates(
    Dict(:well => (:ill => Exponential(5.0)),
        :ill => CensoredDistributions.Compete(:well => Exponential(3.0),
            :dead => Exponential(10.0))), :well)
ctmc(rs)
```

# See also
- [`recur`](@ref): builds a `RecurrentStates`, auto-dispatching to a CTMC when
  the model is memoryless.
- [`CTMCStates`](@ref): the generator-matrix model this returns.
"""
function ctmc(m::RecurrentStates)
    _is_memoryless(m) || throw(ArgumentError(
        "a RecurrentStates converts to a CTMC only when every edge sojourn is " *
        "Exponential and every state races its edges (a Compete node or a lone " *
        "edge, not a fixed-probability Resolve); use the semi-Markov recur(...) " *
        "path for non-exponential or fixed-split models"))
    order = _ctmc_state_order(m)
    idx = Dict(s => i for (i, s) in enumerate(order))
    n = length(order)
    # Promote the matrix element type from the edge rates so a `Dual`/tracked
    # scale differentiates through the generator (mirrors `ctmc(specs...)`).
    T = Float64
    for node in values(m.nodes)
        for d in _edge_dists(node)
            T = promote_type(T, typeof(_exp_rate(d)))
        end
    end
    Q = zeros(T, n, n)
    for (state, node) in m.nodes
        for (dest, d) in zip(_edge_dests(node), _edge_dists(node))
            Q[idx[state], idx[dest]] += _exp_rate(d)
        end
    end
    for i in 1:n
        Q[i, i] = -sum(Q[i, j] for j in 1:n if j != i; init = zero(T))
    end
    return CTMCStates(Tuple(order), Q)
end

# The exit rate of an `Exponential` edge: `1 / scale` (`params(Exponential) =
# (θ,)` the scale). Kept AD-safe by `inv` on the scalar scale.
_exp_rate(d::Exponential) = inv(params(d)[1])

# Whether a RecurrentStates is memoryless (an all-`Exponential`, all-racing CTMC).
function _is_memoryless(m::RecurrentStates)
    return all(_node_is_exp_race, values(m.nodes))
end

# A node is an exponential race when it is a lone `Exponential` edge or a
# `Compete` of `Exponential`s. A `Resolve` (fixed split) is never a CTMC, and a
# non-`Exponential` sojourn keeps the model semi-Markov.
_node_is_exp_race(p::Pair) = p.second isa Exponential
_node_is_exp_race(c::Compete) = all(d -> d isa Exponential, c.delays)
_node_is_exp_race(::Any) = false

# A deterministic CTMC state order from a RecurrentStates: the start state, then
# the remaining transient states sorted, then the absorbing states sorted.
function _ctmc_state_order(m::RecurrentStates)
    return unique(vcat([m.start], transient_states(m), absorbing_states(m)))
end

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

# --- observation likelihood (dispatching front door) ------------------------

@doc raw"""

Log-likelihood of an observation under a [`CTMCStates`](@ref) model.

`logpdf` is the single front door for both observation kinds a CTMC scores; it
DISPATCHES on the shape of `obs`, so no bespoke per-kind scoring name is needed:

- a PANEL — an iterable of `(time, state)` pairs (the state seen at fixed visit
  times, the transition times UNKNOWN) — scores
  `sum_v log P(Δt_v)[s_v, s_{v+1}]`, where `P(Δt) = exp(Q Δt)` marginalises over
  every hidden jump in the gap (the CTMC's advantage over the semi-Markov path);
- a JUMP CHAIN — a [`StatePath`](@ref) or an iterable of `(from, to, dwell)`
  jumps (the transition times KNOWN, a line list) — scores the exact
  exponential-sojourn term `sum log q_{from,to} - q_from * dwell`.

A panel observation is recognised by its element shape (a `(time, state)` pair),
a jump by its `(from, to, dwell)` fields, so the same `logpdf(model, data)` call
covers both.

# Arguments
- `m`: the [`CTMCStates`](@ref) model.
- `obs`: a panel (iterable of `(time, state)` pairs, in increasing time order) or
  a jump chain (a [`StatePath`](@ref) or iterable of `(from, to, dwell)` jumps).

# Examples
```@example
using CensoredDistributions

model = ctmc(:well => (:ill => 0.2), :ill => (:well => 0.3, :dead => 0.1))
panel = [(0.0, :well), (3.0, :ill), (7.0, :well)]
logpdf(model, panel)            # panel: transition times unknown
jumps = [(from = :well, to = :ill, dwell = 4.0)]
logpdf(model, jumps)            # jump chain: transition times known
```

# See also
- [`transition_probability`](@ref): the `exp(Q Δt)` kernel the panel sums.
- [`RecurrentStates`](@ref): the semi-Markov default for non-exponential dwells.
"""
function logpdf(m::CTMCStates, obs)
    o = collect(obs)
    (!isempty(o) && _is_panel_obs(first(o))) &&
        return _ctmc_panel_logpdf(m, o)
    return _ctmc_jumps_logpdf(m, o)
end

# Whether an observation element is a PANEL `(time, state)` point rather than a
# `(from, to, dwell)` jump. A panel point is a 2-tuple ending in a state Symbol,
# or a NamedTuple carrying a `:state` field; a jump NamedTuple has no `:state`.
_is_panel_obs(o::Tuple) = length(o) == 2 && o[2] isa Symbol
_is_panel_obs(o::NamedTuple) = haskey(o, :state)
_is_panel_obs(::Any) = false

function _ctmc_panel_logpdf(m::CTMCStates, panel)
    obs = collect(panel)
    length(obs) >= 2 || throw(ArgumentError(
        "a panel needs at least two (time, state) observations"))
    total = 0.0
    for v in 1:(length(obs) - 1)
        (t0, s0) = _panel_point(obs[v])
        (t1, s1) = _panel_point(obs[v + 1])
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

# Read a `(time, state)` panel point from a 2-tuple or a `(time, state)`
# NamedTuple, so the panel worker is order-independent for the NamedTuple form.
_panel_point(o::Tuple) = (o[1], o[2])
_panel_point(o::NamedTuple) = (o.time, o.state)

# --- exact jump-chain likelihood --------------------------------------------

@doc """

Log-density of an exactly-observed jump chain ([`StatePath`](@ref) overload).

When the transition times ARE known (a line list, not panel data) the CTMC
likelihood is the exponential-sojourn special case of the semi-Markov path: each
step contributes `log q_{from,to} - q_from * dwell` where `q_from` is the total
exit rate of `from`. This [`StatePath`](@ref) overload adds, for a `:censored`
path, the survival `-q_state * remaining` of its unfinished final sojourn; a bare
iterable of `(from, to, dwell)` jumps routes through the general `logpdf` front
door.

# Arguments
- `m`: the [`CTMCStates`](@ref) model.
- `path`: a [`StatePath`](@ref).

# See also
- [`CTMCStates`](@ref): the model type; its `logpdf` front door dispatches panel
  data and bare jump chains on the observation shape.
"""
function logpdf(m::CTMCStates, path::StatePath)
    total = _ctmc_jumps_logpdf(m, path.jumps)
    if path.stop == :censored && path.censored_state !== nothing
        i = state_index(m, path.censored_state)
        total += -(-m.Q[i, i]) * path.censored_for
    end
    return total
end

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
