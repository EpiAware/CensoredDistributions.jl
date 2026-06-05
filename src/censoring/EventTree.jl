# ============================================================================
# Recursive event-tree / nesting layer (#298)
#
# `primary_censored(edges, primary_event)` declares a tree of events whose
# directed edges are delay distributions from a parent event to a child event.
# The `edges` argument is a Tables.jl edge list: any Tables.jl source with the
# columns `parent`, `child`, and `delay`, one row per edge. The root event (the
# single node that is never a child) is inferred from the edge list and takes
# `primary_event` as its censoring window. A single `EventTree` distribution
# then evaluates the joint log density of a per-record observation (one entry
# per event, each a concrete time or `Missing`) as a scalar sum that dispatches
# on the per-record missingness:
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
- [`primary_censored`](@ref): constructor that assembles an edge list into a
  tree
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
delays, `events` the ordered tuple of all event names, `primary_event` the root
censoring window (or `nothing`), `interval` the secondary interval-censoring
width (or `nothing`), `horizon` the right-truncation cut-off (or `nothing`),
and `solver` the latent-node quadrature rule. The parent-to-children adjacency
is recomputed on demand from `edges` rather than stored, which keeps the struct
fully differentiable.

# See also
- [`primary_censored`](@ref): constructor (an edge list builds the tree; a
  single delay recovers the single-edge case this generalises)
- [`logpdf`](@ref): per-record missingness-dispatched evaluation
- [`sequential_distribution`](@ref): the chain (single-path) special case
"""
struct EventTree{E <: Tuple, EV <: Tuple, P, I, H, S} <:
       Distribution{Multivariate, Continuous}
    "Name of the root (origin) event."
    root::Symbol
    "Tuple of the directed delay edges."
    edges::E
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
end

# Outgoing edge indices of the event `name`, computed on the fly from the edge
# tuple. Storing the adjacency as a `Dict` field instead defeats Enzyme's type
# analysis (the `Dict{Symbol, Vector{Int}}` embedded in the differentiated
# struct produces a "bad enzyme_type" failure); scanning the small edge tuple
# keeps the struct a plain tuple-of-distributions that every backend handles.
function _tree_children(d::EventTree, name::Symbol)
    idx = Int[]
    @inbounds for i in eachindex(d.edges)
        d.edges[i].parent === name && push!(idx, i)
    end
    return idx
end

# Validate that the edges form a single tree and INFER its root: every event
# has at most one parent, exactly one event has no parent (the root), and every
# event is reachable from that root with no cycles. A shared interior node (one
# parent, several children) is allowed and is integrated once; an event with
# two incoming edges (two parents) is rejected, since that is not a conjunctive
# tree node. Returns the inferred root and the ordered event-name tuple. The
# adjacency is recomputed on the fly by `_tree_children`, so it is not stored.
function _build_tree_topology(edges::Tuple)
    parents = Dict{Symbol, Symbol}()
    names = Symbol[]

    for e in edges
        e isa EventEdge ||
            throw(ArgumentError("every edge must be an EventEdge"))
        haskey(parents, e.child) &&
            throw(ArgumentError(
                "event $(e.child) has more than one parent; the structure " *
                "must be a tree (shared children are not conjunctive nodes)"))
        parents[e.child] = e.parent
        e.parent in names || push!(names, e.parent)
        e.child in names || push!(names, e.child)
    end

    # The root is the single event that is never a child.
    roots = filter(n -> !haskey(parents, n), names)
    length(roots) == 1 ||
        throw(ArgumentError(
            "the edge list must have exactly one root (an event that is " *
            "never a child); found $(length(roots)): $(Tuple(roots))"))
    root = roots[1]

    # Every non-root event must reach the root by walking parents, with no
    # cycles. Catching a cycle here also rejects a self-loop edge.
    for name in names
        name == root && continue
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

    # Place the root first so the event order starts at the origin.
    ordered = Symbol[root]
    for n in names
        n == root || push!(ordered, n)
    end
    return root, Tuple(ordered)
end

@doc """

Build an [`EventTree`](@ref) from a Tables.jl edge list and a primary event.

`edges` is any Tables.jl source with the columns `parent`, `child`, and
`delay`, one row per directed edge. Each row declares the `delay` distribution
of the time from its `parent` event to its `child` event. The root event (the
single event that is never a child) is inferred from the edge list and takes
`primary_event` as its censoring window.

The edge list must form one tree: every event has at most one parent, exactly
one event has no parent (the root), and every event is reachable from the root
with no cycles. An interior event that several rows descend *from* is a shared
node, integrated exactly once; an event named as `child` by two rows (two
parents) is rejected, since that is not a conjunctive tree node.

The distribution is data-free: which events are observed is decided per record
from the observation passed to [`logpdf`](@ref), not at construction.

# Arguments
- `edges`: a Tables.jl source with columns `parent` (`Symbol`), `child`
  (`Symbol`), and `delay` (a continuous `UnivariateDistribution`), one row per
  edge. A `NamedTuple` of column vectors, a vector of `NamedTuple` rows, or a
  `DataFrame` all work.
- `primary_event`: the root censoring window (primary event distribution)
  applied to the origin event.

# Keyword Arguments
- `interval`: secondary interval-censoring width. Defaults to `nothing`.
- `horizon`: right-truncation horizon (observation cut-off). Defaults to
  `nothing`.
- `solver`: quadrature rule for latent-node marginalisation. Defaults to
  `GaussLegendre(; n = 64)`.

# Examples
```@example
using CensoredDistributions, Distributions

# The bdbv tree as an edge list: onset feeds admission and notification;
# admission feeds death and discharge. Admission is the shared interior node.
edges = (parent = [:onset, :onset, :admit, :admit],
    child = [:admit, :notif, :death, :disch],
    delay = [Gamma(2.0, 1.0), LogNormal(1.0, 0.5),
        Gamma(1.5, 1.0), Gamma(2.0, 1.5)])
tree = primary_censored(edges, Uniform(0.0, 1.0))

# A per-case observation: onset and admission observed, the rest a mix of
# observed and missing. Missing events are marginalised.
obs = (onset = 0.0, admit = 2.0, death = 5.0, disch = missing, notif = 1.5)
logpdf(tree, obs)
```

# See also
- [`EventTree`](@ref): the distribution type
- [`logpdf`](@ref): per-record missingness-dispatched evaluation
"""
function primary_censored(edges, primary_event::UnivariateDistribution;
        interval = nothing, horizon = nothing,
        solver = GaussLegendre(; n = 64))
    edge_tuple = _edges_from_table(edges)
    length(edge_tuple) >= 1 ||
        throw(ArgumentError("the edge list needs at least one edge"))
    root, names = _build_tree_topology(edge_tuple)
    return EventTree(
        root, edge_tuple, names, primary_event, interval, horizon, solver)
end

# Read a Tables.jl edge list into a tuple of `EventEdge`s. Requires the
# `parent`, `child`, and `delay` columns; each row becomes one edge. Reading
# happens here, kept apart from the differentiated arithmetic.
function _edges_from_table(edges)
    Tables.istable(edges) ||
        throw(ArgumentError(
            "edges must be a Tables.jl source with columns parent, child, " *
            "and delay; got $(typeof(edges))"))
    rows = Tables.rows(edges)
    cols = Tables.columnnames(rows)
    for required in (:parent, :child, :delay)
        required in cols ||
            throw(ArgumentError(
                "the edge list is missing the `$required` column"))
    end
    return Tuple(_edge_from_row(row) for row in rows)
end

# Coerce one Tables.jl row to an `EventEdge`, checking the field kinds.
function _edge_from_row(row)
    parent = Tables.getcolumn(row, :parent)
    child = Tables.getcolumn(row, :child)
    delay = Tables.getcolumn(row, :delay)
    parent isa Symbol ||
        throw(ArgumentError("the `parent` entry must be a Symbol"))
    child isa Symbol ||
        throw(ArgumentError("the `child` entry must be a Symbol"))
    delay isa UnivariateDistribution ||
        throw(ArgumentError(
            "the `delay` entry must be a UnivariateDistribution"))
    return EventEdge(parent, child, delay)
end

# ---------------------------------------------------------------------------
# Interface: events / length / eltype / params
# ---------------------------------------------------------------------------

# The event-time vector has one entry per event.
Base.length(d::EventTree) = length(d.events)

@doc "

Return the ordered tuple of every event name in an [`EventTree`](@ref).

The order matches the entries returned by [`rand`](@ref) and accepted by
[`logpdf`](@ref) when a positional observation vector is used.

# Examples
```@example
using CensoredDistributions, Distributions

edges = (parent = [:onset], child = [:admit], delay = [Gamma(2.0, 1.0)])
tree = primary_censored(edges, Uniform(0.0, 1.0))
event_names(tree)
```

# See also
- [`primary_censored`](@ref): constructor function
- [`EventTree`](@ref): the distribution type
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
function logpdf(d::EventTree, observation::NamedTuple)
    return _eventtree_logpdf(d, observation)
end
function logpdf(d::EventTree, observation::AbstractDict)
    return _eventtree_logpdf(d, observation)
end
# A positional observation vector (event_names order). The explicit
# `AbstractVector` signature resolves the ambiguity with the generic
# `Distributions.logpdf(::Distribution{ArrayLikeVariate}, ::AbstractArray)`.
function logpdf(d::EventTree, observation::AbstractVector)
    return _eventtree_logpdf(d, observation)
end

function _eventtree_logpdf(d::EventTree, observation)
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
    for ei in _tree_children(d, name)
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

    kids = _tree_children(d, name)

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
    grandkids = _tree_children(d, edge.child)
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

# Fixed Gauss-Legendre rule for the tree's latent-node quadrature. The user's
# `solver` (an Integrals.jl `GaussLegendre`) already carries reference nodes and
# weights on `[-1, 1]`; wrap them in the `_GL` rule the convolution layer's
# `gl_integrate` consumes, so latent marginalisation runs on the same AD-safe
# fixed-node scaffold. Falls back to the shared `_CONVOLVED_GL` for any solver
# without explicit nodes/weights.
function _tree_gl(d::EventTree)
    s = d.solver
    if hasproperty(s, :nodes) && hasproperty(s, :weights) &&
       !isempty(s.nodes)
        return _GL(s.nodes, s.weights)
    end
    return _CONVOLVED_GL
end

@doc "

Compute the joint probability density of an event-tree observation.

See also: [`logpdf`](@ref)
"
pdf(d::EventTree, observation::NamedTuple) = exp(logpdf(d, observation))
pdf(d::EventTree, observation::AbstractDict) = exp(logpdf(d, observation))
pdf(d::EventTree, observation::AbstractVector) = exp(logpdf(d, observation))

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
        d.root, d.edges, d.events, d.primary_event, d.interval,
        float(horizon), d.solver)
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
    for ei in _tree_children(d, name)
        edge = d.edges[ei]
        times[edge.child] = times[name] + convert(T, rand(rng, edge.delay))
        _tree_sample_subtree!(rng, d, edge.child, times, T)
    end
    return times
end

sampler(d::EventTree) = d

# ---------------------------------------------------------------------------
# show: a readable tree of parent -> child edges and their delays
# ---------------------------------------------------------------------------

@doc """

Print an [`EventTree`](@ref) as an indented tree of its `parent -> child`
edges, each annotated with the edge delay distribution. The root line records
the primary event, and the interval, horizon, and solver settings are summarised
when set.

See also: [`EventTree`](@ref), [`primary_censored`](@ref)
"""
function Base.show(io::IO, ::MIME"text/plain", d::EventTree)
    println(io, "EventTree with $(length(d.events)) events")
    pe = d.primary_event === nothing ? "none" : string(d.primary_event)
    println(io, "  root: $(d.root) (primary_event: $pe)")
    _tree_show_subtree(io, d, d.root, "  ")
    extras = String[]
    d.interval === nothing || push!(extras, "interval = $(d.interval)")
    d.horizon === nothing || push!(extras, "horizon = $(d.horizon)")
    isempty(extras) || print(io, "  (", join(extras, ", "), ")")
    return nothing
end

# Recursively print each outgoing edge of `name` as `parent -> child: delay`,
# indenting one level per tree depth.
function _tree_show_subtree(io::IO, d::EventTree, name::Symbol, indent::String)
    kids = _tree_children(d, name)
    for (k, ei) in enumerate(kids)
        edge = d.edges[ei]
        last = k == length(kids)
        branch = last ? "└─ " : "├─ "
        println(io, indent, branch,
            "$(edge.parent) -> $(edge.child): $(edge.delay)")
        child_indent = indent * (last ? "   " : "│  ")
        _tree_show_subtree(io, d, edge.child, child_indent)
    end
    return nothing
end
