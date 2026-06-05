# ============================================================================
# Recursive event-tree / nesting layer (#298)
#
# `event_tree(root, edges)` declares a tree of events whose directed edges are
# delay distributions from a parent event to a child event. A single
# `EventTree` distribution then evaluates the joint log density of a per-record
# observation (one entry per event, each a concrete time or `Missing`) as a
# scalar sum that dispatches on the per-record missingness:
#
# - an OBSERVED node fixes its event time; every outgoing edge becomes an
#   independent conditional factor `logpdf(delay, child_time - node_time)`,
#   factorising the subtrees rooted at the observed node;
# - a MISSING (latent) interior node is marginalised by integrating over its
#   event time once, jointly across every edge that touches it (the incoming
#   edge density times the outgoing-subtree kernels), so a node shared by
#   several edges is integrated a single time and never double-counted.
#
# This composes the existing primitives rather than re-deriving quadrature: the
# per-edge conditional factors reuse the delay distributions' own `logpdf`, the
# latent-node marginalisation reuses the same fixed-node Gauss-Legendre rule
# (`gl_integrate`) the convolution and shared-origin layers use, and a chain of
# latent nodes folds into `convolve_distributions` when its interior is a simple
# run. `rand` returns the full event-time path over the tree.
#
# The two structural primitives compose RECURSIVELY: a child subtree may itself
# branch (parallel / shared-origin) or chain (sequential), and a latent node's
# integration kernel recurses into its children's kernels, so a shared interior
# latent node couples its whole downstream subtree under one joint integral.
#
# Only CONJUNCTIVE (all-children-occur) nodes are modelled here. DISJUNCTIVE
# (competing-outcome) nodes carrying a branch probability are #300; the edge
# representation leaves a clean extension point (a per-edge field can carry a
# branch weight without changing the node/observation layout).
# ============================================================================

@doc """

A directed edge of an [`EventTree`](@ref): a delay distribution from a parent
event to a child event.

The edge stores the names of its `parent` and `child` events and the `delay`
distribution of the time from the parent event to the child event. Any
continuous `UnivariateDistribution` is admissible (a bare delay, a
`truncated` delay, or a censoring-wrapped delay), so edges are heterogeneous.

# See also
- [`EventTree`](@ref): the tree this edge belongs to
- [`event_tree`](@ref): constructor that assembles edges into a tree
"""
struct EventEdge{D <: UnivariateDistribution}
    "Name of the parent (origin) event."
    parent::Symbol
    "Name of the child (destination) event."
    child::Symbol
    "Delay distribution of the time from the parent event to the child event."
    delay::D
end

@doc """

Joint distribution of a tree of events linked by delay distributions.

`EventTree` models a set of named events arranged as a tree rooted at one
origin event. Each directed edge carries the delay distribution from its
parent event to its child event, and the joint density of an observation
factorises over the tree:

```math
f(\\text{event times})
  = f_{\\text{root}}(t_{\\text{root}})
    \\prod_{(\\text{parent} \\to \\text{child})}
      f_{\\text{delay}}(t_{\\text{child}} - t_{\\text{parent}}),
```

with the root prior supplied by the `primary_event` (the origin censoring
window) or taken as an improper flat prior when the root time is observed.

# Missingness dispatch

The observation is one time per event (a concrete value or `Missing`).
[`logpdf`](@ref) returns a scalar sum whose terms are selected per record from
the missingness pattern:

- an **observed** node fixes its time, so each outgoing edge contributes the
  independent conditional factor ``\\log f_{\\text{delay}}(t_{\\text{child}} -
  t_{\\text{node}})`` and the subtrees below it factorise;
- a **missing** interior node is **latent**: its event time is marginalised by
  one integral over the incoming-edge density times the kernels of its
  outgoing subtrees, so a node shared by several edges is integrated jointly,
  once, and never double-counted.

A latent node whose children are themselves latent nests the integration, so a
connected block of latent nodes is integrated as one joint marginal over the
shared latents.

# Shared interior nodes

An interior event that feeds several edges (for example an admission that both
precedes death/discharge and is itself reached from onset) is a single node in
the tree. When it is observed it is conditioned on once and its outgoing edges
factorise; when it is missing it is the integration variable of one marginal
that couples every edge incident to it. Either way the shared node contributes
exactly once, which is the correctness property a flat list of independent
delays cannot express.

# Automatic differentiation

`logpdf` is safe to differentiate with the observation passed as a constant.
The missingness pattern drives only the control flow that groups observed and
latent nodes; the differentiated arithmetic sees only the concrete observed
times and the delay parameters, so no `Union{Missing}` type ever enters the
gradient tape. This is verified across every supported backend.

# Fields

`root` is the name of the origin event, `edges` the tuple of [`EventEdge`](@ref)
delays, `children` a map from each event name to the indices of its outgoing
edges, `events` the ordered tuple of all event names, `primary_event` the root
censoring window (or `nothing`), `interval` the secondary interval-censoring
width (or `nothing`), `horizon` the right-truncation cut-off (or `nothing`),
`solver` the latent-node quadrature rule and `force_numeric` the flag forcing
numeric integration.

# See also
- [`event_tree`](@ref): constructor function
- [`logpdf`](@ref): per-record missingness-dispatched evaluation
- [`primary_censored`](@ref): the single-edge case this generalises
- [`sequential_distribution`](@ref): the chain (single-path) special case
"""
struct EventTree{E <: Tuple, C, EV <: Tuple, P, I, H, S} <:
       Distribution{Multivariate, Continuous}
    "Name of the root (origin) event."
    root::Symbol
    "Tuple of the directed delay edges."
    edges::E
    "Map from each event name to the indices of its outgoing edges."
    children::C
    "Ordered tuple of every event name in the tree."
    events::EV
    "Root censoring window (primary event distribution), or `nothing`."
    primary_event::P
    "Secondary interval-censoring width, or `nothing`."
    interval::I
    "Right-truncation horizon (observation cut-off), or `nothing`."
    horizon::H
    "Quadrature solver for latent-node marginalisation."
    solver::S
    "Whether to force numeric integration even when a faster path exists."
    force_numeric::Bool
end

# Build the children adjacency map (event name -> outgoing edge indices) and the
# ordered event-name tuple from the root and edges. Validates that the graph is
# a tree rooted at `root`: every non-root event has exactly one parent and is
# reachable from the root, with no cycles.
function _build_tree_topology(root::Symbol, edges::Tuple)
    children = Dict{Symbol, Vector{Int}}()
    parents = Dict{Symbol, Symbol}()
    names = Symbol[root]

    for (i, e) in enumerate(edges)
        e isa EventEdge ||
            throw(ArgumentError("every edge must be an EventEdge"))
        haskey(parents, e.child) &&
            throw(ArgumentError(
                "event $(e.child) has more than one parent; the structure " *
                "must be a tree (shared children are not conjunctive nodes)"))
        e.child == root &&
            throw(ArgumentError("the root event cannot be a child"))
        parents[e.child] = e.parent
        push!(get!(children, e.parent, Int[]), i)
        e.parent in names || push!(names, e.parent)
        e.child in names || push!(names, e.child)
    end

    # Every non-root event must be reachable from the root by walking parents.
    for name in names
        name == root && continue
        haskey(parents, name) ||
            throw(ArgumentError(
                "event $name has no parent edge and is not the root"))
        # Walk to the root, detecting cycles.
        seen = Set{Symbol}()
        cur = name
        while cur != root
            cur in seen &&
                throw(ArgumentError("the edges contain a cycle at $cur"))
            push!(seen, cur)
            haskey(parents, cur) ||
                throw(ArgumentError(
                    "event $cur does not connect to the root $root"))
            cur = parents[cur]
        end
    end

    return children, Tuple(names)
end

@doc """

Build an [`EventTree`](@ref) from a root event and a list of delay edges.

Each edge is a `parent => child => delay` (or an [`EventEdge`](@ref)) declaring
the delay distribution from a parent event to a child event. The edges must
form a tree rooted at `root`: every non-root event has exactly one parent edge
and is reachable from the root. An interior event that several edges descend
*from* is a shared node; an event with two incoming edges (two parents) is
rejected, since that is not a conjunctive tree node.

The distribution is data-free: which events are observed is decided per record
from the observation passed to [`logpdf`](@ref), not at construction.

# Arguments
- `root`: the name (`Symbol`) of the origin event.
- `edges`: the delay edges, each given as `parent => (child => delay)` (a
  nested `Pair`) or directly as an [`EventEdge`](@ref). `parent` and `child`
  are `Symbol`s and `delay` is a continuous `UnivariateDistribution`.

# Keyword Arguments
- `primary_event`: root censoring window applied to the origin event. Defaults
  to `nothing` (no primary censoring).
- `interval`: secondary interval-censoring width. Defaults to `nothing`.
- `horizon`: right-truncation horizon (observation cut-off). Defaults to
  `nothing`.
- `solver`: quadrature rule for latent-node marginalisation. Defaults to
  `GaussLegendre(; n = 64)`.
- `force_numeric`: force numeric integration even where a faster path exists.
  Defaults to `false`.

# Examples
```@example
using CensoredDistributions, Distributions

# The bdbv tree: onset feeds admission and notification; admission feeds
# death and discharge. Admission is the shared interior node.
tree = event_tree(:onset,
    [:onset => :admit => Gamma(2.0, 1.0),
     :onset => :notif => LogNormal(1.0, 0.5),
     :admit => :death => Gamma(1.5, 1.0),
     :admit => :disch => Gamma(2.0, 1.5)])

# A per-case observation: onset and admission observed, the rest a mix of
# observed and missing. Missing events are marginalised.
obs = (onset = 0.0, admit = 2.0, death = 5.0, disch = missing, notif = 1.5)
logpdf(tree, obs)
```

# See also
- [`EventTree`](@ref): the distribution type
- [`logpdf`](@ref): per-record missingness-dispatched evaluation
"""
function event_tree(root::Symbol, edges;
        primary_event = nothing, interval = nothing, horizon = nothing,
        solver = GaussLegendre(; n = 64), force_numeric::Bool = false)
    edge_tuple = Tuple(_as_edge(e) for e in edges)
    length(edge_tuple) >= 1 ||
        throw(ArgumentError("event_tree needs at least one edge"))
    children, names = _build_tree_topology(root, edge_tuple)
    return EventTree(
        root, edge_tuple, children, names, primary_event, interval, horizon,
        solver, force_numeric)
end

# Coerce a `parent => (child => delay)` nested pair to an `EventEdge`. An
# `EventEdge` passes through unchanged.
_as_edge(e::EventEdge) = e
function _as_edge(e::Pair{Symbol, <:Pair{Symbol, <:UnivariateDistribution}})
    return EventEdge(e.first, e.second.first, e.second.second)
end
function _as_edge(e)
    throw(ArgumentError(
        "each edge must be `parent => child => delay` or an EventEdge; " *
        "got $(typeof(e))"))
end

# ---------------------------------------------------------------------------
# Interface: events / length / eltype / params
# ---------------------------------------------------------------------------

# The event-time vector has one entry per event.
Base.length(d::EventTree) = length(d.events)

@doc "

The ordered tuple of every event name in an [`EventTree`](@ref).

The order matches the entries of the event-time vector returned by
[`rand`](@ref) and accepted by [`logpdf`](@ref) when a positional vector is
used. See also: [`event_tree`](@ref).
"
event_names(d::EventTree) = d.events

function Base.eltype(::Type{<:EventTree{E}}) where {E <: Tuple}
    return mapreduce(e -> eltype(fieldtype(e, :delay)), promote_type,
        fieldtypes(E))
end

# Numeric element type carrying any AD tangent in the distribution parameters,
# recovered from the delay params (and the root prior) rather than the sample
# `eltype`, which would drop a Dual. Mirrors the ParallelPrimaryCensored helper.
function _tree_eltype(d::EventTree)
    T = float(mapreduce(
        e -> _parallel_param_eltype(e.delay), promote_type, d.edges))
    if d.primary_event !== nothing
        T = promote_type(T, float(_parallel_param_eltype(d.primary_event)))
    end
    return T
end

params(d::EventTree) = map(e -> params(e.delay), d.edges)

# ---------------------------------------------------------------------------
# Observation access: map a (named or positional) observation to event times
# ---------------------------------------------------------------------------

# Return the observed time for event `name` (a value or `missing`) from an
# observation given as a NamedTuple, an AbstractDict, or a positional vector in
# `event_names(d)` order. Reading the (possibly `Missing`) entry happens here,
# kept apart from the differentiated arithmetic.
function _obs_get(d::EventTree, observation::NamedTuple, name::Symbol)
    return haskey(observation, name) ? getproperty(observation, name) : missing
end
function _obs_get(d::EventTree, observation::AbstractDict, name::Symbol)
    return get(observation, name, missing)
end
function _obs_get(d::EventTree, observation::AbstractVector, name::Symbol)
    i = findfirst(==(name), d.events)
    i === nothing && return missing
    return observation[i]
end

# ---------------------------------------------------------------------------
# logpdf: recursive per-node missingness dispatch returning a scalar
# ---------------------------------------------------------------------------

@doc """

Compute the joint log probability density of an event-tree observation.

`observation` gives one time per event, as a `NamedTuple`, an `AbstractDict`
keyed by event name, or a positional `AbstractVector` in
[`event_names`](@ref) order. Each entry is a concrete time (the event is
observed) or `Missing` (the event is unobserved). Events absent from a
`NamedTuple`/`AbstractDict` are treated as `Missing`.

The result is a scalar sum dispatching on the missingness pattern: observed
nodes are conditioned on and their subtrees factorise, while a latent (missing)
interior node is marginalised by one integral that jointly couples every edge
touching it. A latent node whose children are also latent nests the
integration, so a connected block of latent nodes is one joint marginal.

Missingness drives only control flow; the differentiated arithmetic sees only
concrete observed times, so this is safe to differentiate with `observation`
held constant.

See also: [`EventTree`](@ref), [`pdf`](@ref)
"""
function logpdf(d::EventTree, observation)
    T = _tree_eltype(d)

    root_time = _obs_get(d, observation, d.root)
    if root_time === missing
        # Latent root: marginalise it jointly with any latent descendants by
        # integrating over the root prior times the root's subtree kernel.
        d.primary_event === nothing && throw(ArgumentError(
            "a missing root needs a primary_event prior to marginalise over"))
        return _tree_latent_logpdf(
            d, observation, d.root, d.primary_event, T)
    end

    # Observed root: its prior is the primary-event density (flat / dropped
    # when there is no primary_event), and each outgoing subtree factorises.
    rt = convert(T, root_time)
    lp = d.primary_event === nothing ? zero(T) :
         convert(T, logpdf(d.primary_event, rt))
    lp += _tree_observed_node_logpdf(d, observation, d.root, rt, T)
    return lp
end

# Sum of the outgoing-edge log contributions for an OBSERVED node at time
# `node_time`. Each outgoing edge is handled by `_tree_edge_logpdf`, which
# conditions on or marginalises its child subtree.
function _tree_observed_node_logpdf(
        d::EventTree, observation, name::Symbol, node_time::T, ::Type{T}) where {T}
    lp = zero(T)
    for ei in get(d.children, name, Int[])
        lp += _tree_edge_logpdf(d, observation, d.edges[ei], node_time, T)
    end
    return lp
end

# Log contribution of one edge whose PARENT is observed at `parent_time`. If the
# child is observed, condition (the edge delay density at the observed gap) and
# recurse into the child's own outgoing subtrees. If the child is latent,
# marginalise it: integrate the edge delay times the child's subtree kernel over
# the child event time.
function _tree_edge_logpdf(
        d::EventTree, observation, edge::EventEdge, parent_time::T,
        ::Type{T}) where {T}
    child_time = _obs_get(d, observation, edge.child)
    if child_time === missing
        # Latent child anchored at an observed parent: a one-dimensional (or,
        # with latent descendants, nested) marginal over the child time. The
        # child prior is the (horizon-truncated) edge delay shifted by the
        # parent time.
        prior = _ShiftedDelay(_tree_horizon_delay(d, edge.delay), parent_time)
        return _tree_latent_logpdf(d, observation, edge.child, prior, T)
    end
    ct = convert(T, child_time)
    # Condition: the edge delay density at the observed gap, plus the child's
    # own outgoing subtrees (which factorise off the now-observed child).
    lp = _tree_edge_conditional_logpdf(d, edge.delay, ct - parent_time, T)
    lp += _tree_observed_node_logpdf(d, observation, edge.child, ct, T)
    return lp
end

# Conditional log density of an edge delay at an observed gap, applying the
# tree's secondary interval censoring when set. The horizon truncation is
# applied at the marginal level (see `truncated(::EventTree, ...)`); here only
# the interval width wraps the bare delay so an observed window is scored as a
# discrete interval rather than a point.
function _tree_edge_conditional_logpdf(
        d::EventTree, delay::UnivariateDistribution, gap, ::Type{T}) where {T}
    base = _tree_horizon_delay(d, delay)
    if d.interval === nothing
        return convert(T, logpdf(base, gap))
    end
    return convert(T, logpdf(interval_censored(base, d.interval), gap))
end

# A delay distribution shifted by a fixed origin time: the prior on a latent
# child event whose observed parent sits at `origin`. Used as the integration
# prior for a latent node; `pdf`/`logpdf`/support all shift by `origin` so the
# latent node's time (not its delay) is the integration variable.
struct _ShiftedDelay{D <: UnivariateDistribution, T <: Real}
    delay::D
    origin::T
end
_shifted_pdf(s::_ShiftedDelay, t) = pdf(s.delay, t - s.origin)
_shifted_logpdf(s::_ShiftedDelay, t) = logpdf(s.delay, t - s.origin)
_shifted_min(s::_ShiftedDelay) = s.origin + minimum(s.delay)
_shifted_max(s::_ShiftedDelay) = s.origin + maximum(s.delay)

# Prior accessors unified over a real distribution (latent root: the
# primary_event) and a `_ShiftedDelay` (latent child anchored at its parent).
_prior_pdf(p::UnivariateDistribution, t) = pdf(p, t)
_prior_pdf(p::_ShiftedDelay, t) = _shifted_pdf(p, t)
_prior_min(p::UnivariateDistribution) = float(minimum(p))
_prior_min(p::_ShiftedDelay) = float(_shifted_min(p))
_prior_max(p::UnivariateDistribution) = float(maximum(p))
_prior_max(p::_ShiftedDelay) = float(_shifted_max(p))

# Marginalise a LATENT node `name` whose prior over its event time is `prior`.
#
# The node may itself have outgoing edges whose children are observed or latent;
# the integrand at a candidate node time `t` is
#   prior(t) * Π_{outgoing edges} subtree_kernel(edge, t),
# where the subtree kernel for an edge with an OBSERVED child is the edge delay
# density at the (now fixed) gap times that child's own subtree factor, and for
# a LATENT child is the child's nested marginal kernel. Integrating once over
# `t` couples every edge incident to the shared node, so a node reached by one
# incoming edge and feeding several outgoing edges is integrated jointly and a
# single time. Leaf latent nodes (no outgoing edges) integrate their prior to
# one, contributing nothing.
function _tree_latent_logpdf(
        d::EventTree, observation, name::Symbol, prior, ::Type{T}) where {T}
    lower = _prior_min(prior)
    upper = _prior_max(prior)

    kids = get(d.children, name, Int[])

    # A latent leaf: its prior integrates to one over its full support, so the
    # marginal contributes log 1 = 0. (The node is unobserved and childless,
    # so it carries no likelihood information.)
    isempty(kids) && return zero(T)

    # Tighten the integration window to where every outgoing subtree kernel can
    # be non-zero, so the quadrature spends its nodes on the supported region.
    for ei in kids
        edge = d.edges[ei]
        ct = _obs_get(d, observation, edge.child)
        ct === missing && continue
        # Observed child at ctv: the gap ct - t must lie in the delay support,
        # i.e. t in [ct - max(delay), ct - min(delay)].
        ctv = convert(T, ct)
        lower = max(lower, ctv - float(maximum(edge.delay)))
        upper = min(upper, ctv - float(minimum(edge.delay)))
    end
    upper > lower || return convert(T, -Inf)

    # Density-weighted integrand over the node time t: the prior times the
    # product of the outgoing subtree kernels (each a PROBABILITY DENSITY in t).
    integrand = let d = d, observation = observation, name = name, prior = prior,
        kids = kids, T = T

        t -> begin
            val = _prior_pdf(prior, t)
            val <= 0 && return zero(val)
            for ei in kids
                val *= _tree_subtree_kernel(d, observation, d.edges[ei], t, T)
                val <= 0 && return zero(val)
            end
            return val
        end
    end

    integral = gl_integrate(integrand, lower, upper, _tree_gl(d))
    return integral <= 0 ? convert(T, -Inf) : convert(T, log(integral))
end

# Density kernel (in the PARENT node time `t`) of one outgoing edge's subtree.
# Observed child: the edge delay density at the gap times the child's own
# subtree factor (an ordinary number, the child being fixed). Latent child:
# the child's nested marginal density as a function of `t`, computed by an
# inner integral over the child event time. The nesting is what makes a block
# of connected latent nodes one joint marginal.
function _tree_subtree_kernel(
        d::EventTree, observation, edge::EventEdge, t, ::Type{T}) where {T}
    child_time = _obs_get(d, observation, edge.child)
    if child_time === missing
        return _tree_latent_kernel(d, observation, edge, t, T)
    end
    ct = convert(T, child_time)
    gap = ct - t
    dens = _tree_edge_conditional_density(d, edge.delay, gap)
    dens <= 0 && return zero(dens)
    # The observed child's own outgoing subtrees factorise off it (fixed time),
    # contributing a constant log factor that we fold in as a density multiplier.
    sub = _tree_observed_node_logpdf(d, observation, edge.child, ct, T)
    return dens * exp(sub)
end

# Conditional density of an edge delay at an observed gap (interval-censored
# when the tree carries a secondary interval width), the density companion of
# `_tree_edge_conditional_logpdf`.
function _tree_edge_conditional_density(
        d::EventTree, delay::UnivariateDistribution, gap)
    base = _tree_horizon_delay(d, delay)
    if d.interval === nothing
        return pdf(base, gap)
    end
    return pdf(interval_censored(base, d.interval), gap)
end

# The edge delay used in scoring, right-truncated to the observation horizon
# when the tree carries one. A delay longer than the horizon could not have
# been observed, so each edge delay is conditioned on falling within
# `[lower, horizon]`. With no horizon the bare delay is used.
function _tree_horizon_delay(d::EventTree, delay::UnivariateDistribution)
    d.horizon === nothing && return delay
    return truncated(delay; upper = d.horizon)
end

# Nested marginal density (in the parent time `t`) of an edge whose child is
# LATENT: integrate the edge delay (shifted to anchor at `t`) times the child's
# own subtree kernels over the child event time. A latent leaf child integrates
# its delay to one, so the kernel is one and the edge drops from the joint.
function _tree_latent_kernel(
        d::EventTree, observation, edge::EventEdge, t, ::Type{T}) where {T}
    grandkids = get(d.children, edge.child, Int[])
    isempty(grandkids) && return one(promote_type(typeof(t), T))

    prior = _ShiftedDelay(_tree_horizon_delay(d, edge.delay), t)
    lower = _shifted_min(prior)
    upper = _shifted_max(prior)
    for ei in grandkids
        gk = d.edges[ei]
        ctc = _obs_get(d, observation, gk.child)
        ctc === missing && continue
        ctv = convert(T, ctc)
        lower = max(lower, ctv - float(maximum(gk.delay)))
        upper = min(upper, ctv - float(minimum(gk.delay)))
    end
    upper > lower || return zero(promote_type(typeof(t), T))

    inner = let d = d, observation = observation, prior = prior, grandkids = grandkids,
        T = T

        u -> begin
            val = _shifted_pdf(prior, u)
            val <= 0 && return zero(val)
            for ei in grandkids
                val *= _tree_subtree_kernel(d, observation, d.edges[ei], u, T)
                val <= 0 && return zero(val)
            end
            return val
        end
    end
    return gl_integrate(inner, lower, upper, _tree_gl(d))
end

# Fixed Gauss-Legendre rule for the tree's latent-node quadrature. Reuses the
# convolution layer's shared rule so latent marginalisation and convolution
# integrate on the same AD-safe scaffold.
_tree_gl(d::EventTree) = _CONVOLVED_GL

@doc "

Compute the joint probability density of an event-tree observation.

See also: [`logpdf`](@ref)
"
function pdf(d::EventTree, observation)
    return exp(logpdf(d, observation))
end

# ---------------------------------------------------------------------------
# Right-truncation as a `truncated` method on the tree
# ---------------------------------------------------------------------------

@doc """

Right-truncate an [`EventTree`](@ref) to an observation horizon.

Returns a copy of `d` carrying `horizon` as its right-truncation cut-off. Each
edge delay is conditioned on falling within `[lower, horizon]` (a delay longer
than the observation window could not have been seen), applied per edge inside
[`logpdf`](@ref) via [`truncated`](@ref) on the edge delay. This is the
per-edge form the rest of the stack uses; a joint shared-origin denominator is
out of scope here.

# Arguments
- `d`: the tree to right-truncate.
- `horizon`: the observation cut-off time.

# See also
- [`EventTree`](@ref): the distribution type
- [`logpdf`](@ref): applies the per-record truncation denominator
"""
function truncated(d::EventTree, horizon::Real)
    return EventTree(
        d.root, d.edges, d.children, d.events, d.primary_event, d.interval,
        float(horizon), d.solver, d.force_numeric)
end

# ---------------------------------------------------------------------------
# rand: full event-time path over the tree
# ---------------------------------------------------------------------------

@doc """

Sample a full event-time path for an [`EventTree`](@ref).

Draws the root event time from the `primary_event` prior (zero when there is no
primary event), then walks the tree from the root adding a draw of each edge's
delay to its parent's time. Returns a `NamedTuple` of event name to event time.

See also: [`EventTree`](@ref)
"""
function Base.rand(rng::AbstractRNG, d::EventTree)
    T = float(eltype(d))
    times = Dict{Symbol, T}()
    times[d.root] = d.primary_event === nothing ? zero(T) :
                    convert(T, rand(rng, d.primary_event))
    _tree_sample_subtree!(rng, d, d.root, times, T)
    return NamedTuple{d.events}(Tuple(times[n] for n in d.events))
end

Base.rand(d::EventTree) = rand(default_rng(), d)

# Recursively sample each child's event time as its parent's time plus an edge
# delay draw, in tree order.
function _tree_sample_subtree!(
        rng::AbstractRNG, d::EventTree, name::Symbol, times, ::Type{T}) where {T}
    for ei in get(d.children, name, Int[])
        edge = d.edges[ei]
        times[edge.child] = times[name] + convert(T, rand(rng, edge.delay))
        _tree_sample_subtree!(rng, d, edge.child, times, T)
    end
    return times
end

sampler(d::EventTree) = d
