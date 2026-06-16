# ============================================================================
# Censored composers: recursive nested-tree scoring
# ============================================================================
#
# Split out of `censored_specialisations.jl` (the shared recovery helpers /
# trait detection live there and are included first). Generic nested-tree
# scoring; the `Competing`/`Select` slice scoring and per-record rebuilds live
# in `censored_competing.jl`.

# ---------------------------------------------------------------------------
# Recursive nested-composer scoring
# ---------------------------------------------------------------------------
#
# A composer whose branch/step is ITSELF a `Sequential` / `Parallel` is an
# arbitrary, irregular event tree (e.g. bdbv `onset -> {admit -> {death,
# discharge}, notif}`). The flat one-level scoring above only handles leaf
# steps/branches; the recursion here scores the tree at every depth, each node
# consuming its contiguous slice of the flat event vector.
#
# The flat event vector lays out one entry per LEAF event in depth-first order,
# preceded by the root origin `E_0`. A nested composer's ORIGIN is the parent's
# event at its position (the origin is shared between parent and child, so a
# nested composer does NOT re-introduce its own origin slot); it then consumes
# one slot per leaf event in its subtree (`_nleaves`). `_subtree_logpdf` walks
# the tree returning `(logprob, next_idx)`: the subtree log density and the index
# one past the last event it consumed. Dispatch on the node type drives the
# recursion (no runtime type lookup), and the components are processed by
# head/tail tuple recursion so the compiler specialises each child type, keeping
# the walk type-stable and AD-safe.

# The DATA value type for a tree walk: the event vector's non-missing element
# type. Used only to convert the observed event VALUES (data) and the `zero`
# seed; the per-edge log densities keep their own (possibly AD-tracked) type and
# flow into the UNANNOTATED accumulator, so an AD `Dual`/tracked value from the
# leaf params propagates through the sum without being narrowed.
#
# Extracted by a `@generated` function that strips `Missing` at COMPILE time and
# splices the bare type in as a constant. A plain runtime `nonmissingtype(eltype(
# events))` crashes Mooncake's reverse-mode codegen (uncatchable `signal 4`) when
# applied to the tracked event vector; folding it into a generated constant keeps
# `nonmissingtype` off the runtime (and so differentiated) path entirely. The
# `@generated` body has bound parameters, so it is also Aqua-clean.
_tree_acc_type(d, events) = _data_value_type(eltype(events))
@generated function _data_value_type(::Type{E}) where {E}
    S = nonmissingtype(E)
    bare = (S === Union{} || S === Missing) ? Float64 : S
    return :($bare)
end

# The node whose origin primary event seeds the root of a `Sequential` tree: the
# first step (its origin E_0 is the latent primary). Kept as a tiny helper so the
# dispatch site reads clearly.
_first_origin_node(d::Sequential) = d.components[1]

# --- Recursive tree scoring via per-node runtime loops ------------------------
#
# `_tree_score` walks the tree and sums each edge's contribution, recursing into
# a nested composer step/branch through a dispatched function call. The walk over
# a node's children is a RUNTIME `for` loop over the component tuple (mirroring
# the generic `_composite_logpdf` in `nesting.jl`), NOT a head/tail type
# recursion: the compiled AD backends (Mooncake) differentiate a per-node runtime
# loop plus dispatched recursion, but choke on deep tuple-splatting head/tail
# recursion that carries the tracked leaf distributions. The mutual recursion
# (Sequential -> Parallel -> Sequential) goes through dispatch, the same shape
# the generic composer logpdf already differentiates.
#
# The flat event vector is laid out depth-first: entry 1 is the root origin E_0,
# then one entry per LEAF event in traversal order. A node's ORIGIN is the
# parent's event (a SHARED slot), while its own leaf events occupy a CONTIGUOUS
# slice; the walk carries both `origin_idx` (the origin event) and `event_start`
# (where this node's own leaf events begin) so an irregular tree scores at every
# depth. Each node returns `(logprob, next_event_idx, terminal_idx)`.

# The event-index layout is computed by PURE-INTEGER helpers (`_child_nleaves`
# from `nesting.jl`, plus `_terminal_offset` below) that touch only the tree
# structure, never the leaf params, so the scoring recursion can return a plain
# SCALAR log density. Returning a scalar (not an index-carrying tuple) is what
# keeps the compiled AD backends (Mooncake) happy: a recursion returning a tuple
# that mixes a tracked log density with inactive `Int` indices hits Mooncake's
# mixed-activity codegen, while a scalar-returning recursion over the same
# unrolled-tuple loop differentiates exactly as the generic `_composite_logpdf`
# does (notwithstanding for Enzyme on deeply mixed trees).

# Offset (relative to a node's `event_start`) of the TERMINAL event a following
# chain step hangs off. A leaf edge terminates at its own (single) event; a
# Sequential at its last leaf event; a Parallel at its shared origin, which is the
# PARENT's event (offset -1 relative to the branch's own first event). Pure Int.
_terminal_offset(::UnivariateDistribution) = 0
# A `Competing` is a terminal node (the chain does not continue through a single
# outcome); like a `Parallel` its terminal for a following step is the shared
# origin it hangs off, offset -1 from its own first (outcome) event slot.
_terminal_offset(::AbstractCompeting) = -1
_terminal_offset(d::Sequential) = _seq_terminal_offset(d.components)
_terminal_offset(::Parallel) = -1
# A nested `Select` swaps in ONE alternative per row; the alternatives share one
# event-slot width (checked by `_event_child_nleaves`), so the Select's terminal
# offset is that of its (common) alternative. A following chain step hangs off the
# same terminal regardless of which alternative routes.
_terminal_offset(d::Select) = _terminal_offset(_flat_select_alternative(d))
function _seq_terminal_offset(components::Tuple)
    # The last step's terminal, measured from the chain's own first event. Uses
    # the EVENT-slot count (a `Competing` step spans one slot per outcome).
    off = 0
    @inbounds for i in 1:(length(components) - 1)
        off += _event_child_nleaves(components[i])
    end
    last = components[end]
    return off + _terminal_offset(last)
end

# Top entry: origin at 1, first event at 2; returns the scalar log density. The
# return type is NOT `T` (the event-data type): the per-edge log densities carry
# any AD `Dual`/tracked type from the leaf params, so the result widens past the
# data type and must not be asserted to `T`.
#
# A per-record observation `horizon` (default `nothing`) threads down to any
# nested `Competing`/`HazardCompeting` node so it right-truncates its conditioned
# branch at the remaining window from its anchor, mirroring the top-level
# `_maybe_truncate` (#517). `horizon === nothing` is byte-identical to the
# untruncated walk; the plain leaf/chain edges ignore the horizon (their
# whole-compose right-truncation is applied by the flat scorer's collapsed
# denominator, not here).
function _nested_tree_logpdf(d::Union{Sequential, Parallel}, events,
        primary, ::Type{T}, horizon = nothing) where {T}
    return _tree_score(d, events, 1, 2, primary, T, horizon)
end

# Sequential: the first step hangs off `origin_idx` with the (latent) primary,
# each later step off the previous step's terminal event. The first step is
# PEELED off so the primary lives only on it (a runtime `i == 1 ? primary :
# nothing` would make the primary a `Union{P, Nothing}` value and defeat
# inference); the rest are walked with `for c in tail` (Julia unrolls a `for`
# over a tuple, so the heterogeneous steps stay type-stable). The running event
# cursor and origin index are PURE-INTEGER bookkeeping computed from
# `_child_nleaves` / `_terminal_offset`, so each `_tree_step` returns a scalar.
function _tree_score(d::Sequential, events, origin_idx::Int, event_start::Int,
        primary, ::Type{T}, horizon = nothing) where {T}
    comps = d.components
    first_step = first(comps)
    acc = _tree_step(first_step, events, origin_idx, event_start, primary, T,
        horizon)
    o_idx = event_start + _terminal_offset(first_step)
    ev_idx = event_start + _event_child_nleaves(first_step)
    @inbounds for step in Base.tail(comps)
        acc += _tree_step(step, events, o_idx, ev_idx, nothing, T, horizon)
        o_idx = ev_idx + _terminal_offset(step)
        ev_idx += _event_child_nleaves(step)
    end
    return acc
end

# Parallel: every branch hangs off the SHARED origin `origin_idx`; each branch's
# own events continue from the running cursor. The branch walk is
# `for b in components` (unrolled, type-stable); the cursor is pure-Int
# bookkeeping, so each `_tree_step` returns a scalar.
#
# Within a NESTED tree the per-branch scoring is independent given the shared
# origin: an observed origin conditions each branch on the declared edge, a
# missing origin marginalises each LEAF branch against the latent primary via the
# analytic `primary_censored` (handled in `_tree_step`). The coupled 1-D origin
# QUADRATURE (the flat-`Parallel` shared-origin marginal) is deliberately NOT
# reachable from here: its `gl_integrate` path hard-crashes the compiled AD
# backends (Mooncake/Enzyme) uncatchably even when not taken at runtime (they
# build a rule for every reachable branch), so pulling it into the nested path
# would break AD for the whole tree. A flat (non-nested) `Parallel` with a missing
# shared origin still gets the coupled integral through its own `logpdf` method.
function _tree_score(d::Parallel, events, origin_idx::Int, event_start::Int,
        primary, ::Type{T}, horizon = nothing) where {T}
    acc = zero(T)
    ev_idx = event_start
    @inbounds for branch in d.components
        acc += _tree_step(branch, events, origin_idx, ev_idx, primary, T,
            horizon)
        ev_idx += _event_child_nleaves(branch)
    end
    return acc
end

# A nested-composer step/branch recurses on a FRESH sub-event view
# `[origin, its_leaf_events...]` (origin at 1, first event at 2), rather than the
# full event vector with offset indices. Passing a shrinking sub-view per level
# mirrors the generic `_composite_logpdf` recursion (which the compiled AD
# backends differentiate); recursing on the full shared array with offsets
# instead trips Mooncake's reverse-mode codegen on the repeated aliasing. The
# sub-view is built from the constant event vector (index control flow only), so
# the differentiated arithmetic still sees only the leaf params. The nested
# (declared-edge) semantics are preserved -- this is `_tree_score`, not the flat
# `logpdf` -- so an observed origin conditions each child on its declared edge.
function _tree_step(step::Union{Sequential, Parallel}, events, o_idx::Int,
        ev_idx::Int, primary, ::Type{T}, horizon = nothing) where {T}
    # The sub-view spans the node's EVENT slots (a `Competing` step contributes
    # one slot per outcome), so use the event-slot count, not `length`.
    sub = _subevent_slice(events, o_idx, ev_idx, _event_nleaves(step.components))
    return _tree_score(step, sub, 1, 2, primary, T, horizon)
end

# The `[origin, leaf_events...]` sub-event vector for a nested node: its origin
# (the parent's event) followed by the node's own `n` leaf-event slice. Always a
# freshly gathered `Vector{eltype(events)}` (one concrete return type, so the
# recursive `_tree_score` call stays type-stable); the gather reads only the
# constant event values, so the differentiated arithmetic is unaffected.
function _subevent_slice(events, o_idx::Int, ev_idx::Int, n::Int)
    out = Vector{eltype(events)}(undef, n + 1)
    out[1] = events[o_idx]
    @inbounds for k in 1:n
        out[k + 1] = events[ev_idx + k - 1]
    end
    return out
end
# A nested `Select` on the tree path routes to ONE alternative and scores it as
# that edge. The numeric event-vector path carries no row selector (the selector
# is a Symbol, not an event time), so this DETERMINISTIC default commits to the
# FIRST alternative -- the data-free value-vector round-trip, where a constructed
# flat vector must score back through `logpdf` without a selector. The DATA path
# does NOT reach here: the per-record build resolves the selector into the tree
# (`_resolve_selects`) before scoring, replacing the Select with the routed
# alternative, so a real record routes by `row[selector]` and never silently
# scores alternative 1. The chosen alternative is scored through its own
# `_tree_step`, so a leaf conditions on its declared censoring and a composer
# alternative recurses.
function _tree_step(step::Select, events, o_idx::Int, ev_idx::Int,
        primary, ::Type{T}, horizon = nothing) where {T}
    return _tree_step(_flat_select_alternative(step), events, o_idx, ev_idx,
        primary, T, horizon)
end

function _tree_step(step::UnivariateDistribution, events, o_idx::Int,
        ev_idx::Int, primary, ::Type{T}, horizon = nothing) where {T}
    o = events[o_idx]
    y = events[ev_idx]
    if o !== missing && y !== missing
        # Condition on the edge's declared censoring at the observed gap. Only the
        # DATA gap is converted to `T`; the log density keeps its own (AD-tracked)
        # type so a leaf-param gradient is not narrowed.
        return logpdf(step, convert(T, y) - convert(T, o))
    end
    if o === missing && y !== missing && primary !== nothing
        # Root origin latent: marginalise E_0 against this edge's core.
        return logpdf(primary_censored(_marginal_core(step), primary),
            convert(T, y))
    end
    # A missing endpoint contributes no factor.
    return zero(T)
end
