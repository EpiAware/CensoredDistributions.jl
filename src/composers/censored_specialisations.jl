# ============================================================================
# Censored specialisations of the generic composers
# ============================================================================
#
# The generic `Sequential` / `Parallel` / `Competing` composers score a
# value vector with one entry per step/branch. When their internal nodes are our
# censored distributions (`primary_censored` / `interval_censored` /
# `double_interval_censored`), per-record marginalisation is AUTOMATIC and
# data-driven, selected by MULTIPLE DISPATCH on the value vector's element type
# (a `Missing`-admitting event vector) and on the node types. There is no
# runtime predicate, no `mode` keyword, and no new node-type hierarchy.
#
# Evaluating a censored chain against an EVENT vector
# `[E_0, E_1, ..., E_k]` (one entry per event, each a value or `missing`):
#   - the origin's primary censoring is ALWAYS applied (first segment);
#   - an OBSERVED intermediate -> CONDITION on its (censored) value: the
#     adjacent delay is an independent factor at the observed gap;
#   - an UNOBSERVED intermediate -> MARGINALISE by CONVOLVING the adjacent
#     delays and DROPPING that intermediate's censoring (the latent is a
#     continuous time, not a windowed observation), via `convolve_distributions`
#     over the continuous cores recovered with `get_dist_recursive`.
# For `Parallel` with censored branches the shared origin couples the branches:
# a missing origin entry is marginalised by one 1-D origin integral, a present
# origin conditions. `Competing` already lowers to a `MixtureModel`
# (`as_mixture`), so it needs no event-vector specialisation here.

# ---------------------------------------------------------------------------
# Origin primary-event recovery
# ---------------------------------------------------------------------------

# The primary event distribution censoring a node's origin, or `nothing` when
# the node carries no primary censoring. Recurses through the censoring wrappers
# (`Truncated`, `IntervalCensored`, `Weighted`) so a `double_interval_censored`
# origin still surfaces its primary event. Dispatch on the node type is the
# whole selection: a plain delay returns `nothing` and so keeps the generic
# (uncensored) treatment.
_origin_primary_event(d::PrimaryCensored) = d.primary_event
_origin_primary_event(d::Truncated) = _origin_primary_event(d.untruncated)
_origin_primary_event(d::IntervalCensored) = _origin_primary_event(d.dist)
_origin_primary_event(d::Weighted) = _origin_primary_event(d.dist)
_origin_primary_event(::UnivariateDistribution) = nothing
# A nested multivariate composer is not a primary-censored origin leaf, so it
# surfaces no origin primary event (the censored treatment is resolved within
# the child). `Competing` is univariate and already hits the fallback above.
_origin_primary_event(::Union{Sequential, Parallel}) = nothing
# A nested `Select` resolves censoring within the committed alternative (the
# first on the flat path); a `Latent` resolves it within its wrapped node.
# Neither surfaces a flat origin primary event.
_origin_primary_event(::Select) = nothing
_origin_primary_event(d::Latent) = _origin_primary_event(d.dist)

# The continuous delay core of a (possibly censored) node, for marginalisation:
# strip every censoring layer so a marginalised run convolves only continuous
# delays, never a discrete/windowed object. A `Convolved` node is already a
# continuous sum and is left intact for the fold.
_marginal_core(d::UnivariateDistribution) = get_dist_recursive(d)
_marginal_core(d::Convolved) = d

# The secondary interval censoring a leaf carries (its `IntervalCensored` layer),
# recovered THROUGH the `Truncated` / `Weighted` wrappers, or `nothing` when the
# leaf has none (uncensored or primary-only). The sim walk draws a continuous
# delay from `_marginal_core` and then applies this interval to the RECORDED
# event times, matching the scorer that floors each gap to its day. A
# primary-only leaf returns `nothing`, so it stays continuous (no spurious
# flooring). Dispatch on the node type is the whole selection.
_leaf_interval(d::IntervalCensored) = d
_leaf_interval(d::Truncated) = _leaf_interval(d.untruncated)
_leaf_interval(d::Weighted) = _leaf_interval(d.dist)
_leaf_interval(::UnivariateDistribution) = nothing

# The BARE continuous core of an edge for a SAMPLED-endpoint latent edge (#453):
# every censoring layer stripped (the same `_marginal_core` the marginal scorer
# uses), so NO primary AND no secondary interval. When an edge's predecessor or
# target is a sampled continuous latent, the marginal form scores the edge on this
# bare core — `_parallel_conditional_logpdf` conditions each flat branch on
# `_marginal_core`, and `_sequential_segment` convolves the `_marginal_core`s
# across a sampled-intermediate run. Re-applying ANY censoring (primary or the
# secondary interval) on a sampled-endpoint edge would diverge from the marginal
# and double-count the within-window uncertainty already represented by the
# sampled continuous time. Shared by the per-record latent scoring (the DynamicPPL
# ext) and the vectorised endpoint-observed chain path so the two latent forms and
# the marginal all agree. An OBSERVED-to-OBSERVED edge keeps its declared
# censoring (#419) and never reaches here.
_bare_latent_edge(edge) = _marginal_core(edge)

# Apply a leaf's secondary interval to a recorded (continuous) event time: floor
# to the regular width or snap to the arbitrary boundary, mirroring
# `rand(::IntervalCensored)`. A `nothing` interval leaves the value continuous.
_apply_leaf_interval(value, ::Nothing) = value
function _apply_leaf_interval(value, d::IntervalCensored)
    is_regular_intervals(d) &&
        return floor_to_interval(value, interval_width(d))
    return find_interval_boundary(value, d.boundaries)
end

# The secondary interval of the edge that ANCHORS a composer's origin: the first
# step of a `Sequential` (recursing into a nested first step), the shared-origin
# edge of a `Parallel`. The recorded origin slot is discretised to this interval
# so a first-level gap (its event minus the origin) reflects `floor(target) -
# floor(origin)` and is unbiased; a primary-only origin edge returns `nothing`,
# leaving the origin continuous.
_origin_interval(d::Sequential) = _origin_interval(d.components[1])
_origin_interval(d::Parallel) = _shared_origin_interval(d.components)
_origin_interval(d::AbstractCompeting) = _shared_origin_interval(d.delays)
# A nested `Select` / `Latent` anchors its origin within the routed alternative /
# wrapped node; the alternatives share one origin interval, so the first is
# representative. Mirrors the `_tree_primary_event` recursion so a Parallel whose
# origin-anchor branch is itself a `Select`/`Latent` still discretises its origin
# slot (reached only when that branch is the first censored one, e.g. a
# single-branch `compose((x = select(...),))`, issue #436).
_origin_interval(d::Select) = _shared_origin_interval(d.alternatives)
_origin_interval(d::Latent) = _origin_interval(d.dist)
_origin_interval(d::UnivariateDistribution) = _leaf_interval(d)

# The shared origin interval of a `Parallel`/`Competing`: the branches hang off
# one origin, so they must agree on its interval; the first non-`nothing` is
# returned (a mismatch is a malformed shared origin, but the scorer/data already
# assume one shared origin so this stays a simple first-found).
function _shared_origin_interval(components)
    @inbounds for c in components
        iv = _origin_interval(c)
        iv === nothing || return iv
    end
    return nothing
end

# Whether a component is itself a nested composer (a branch/step that recurses)
# rather than a leaf edge. Dispatch on the type, so the recursion is selected at
# compile time with no runtime type lookup. A `Competing` is univariate (a single
# marginal time-to-resolution leaf), so it is NOT a nested composer here.
_is_nested_composer(::Union{Sequential, Parallel}) = true
_is_nested_composer(::UnivariateDistribution) = false

# Whether any of a composer's components is itself a nested composer. Selects the
# recursive tree walk over the flat one-level scoring. Resolved on the component
# TYPES (the `Tuple` element types), so it is a compile-time constant and the
# branch on it is eliminated, keeping the scoring type-stable.
_any_nested_composer(components::Tuple) = any(_is_nested_composer, components)

# A SINGLETON trait carrying the nested/flat choice in its TYPE, so the nested and
# flat scoring become separate dispatched methods. A plain `if _any_nested_...`
# branch leaves BOTH branches in one method body, and the compiled AD backends
# (Mooncake) build a rule for every reachable branch -- including the flat
# `convolve_distributions` marginalisation path that hard-crashes them
# uncatchably. Splitting by dispatch keeps the flat path out of the nested
# method entirely, so a nested tree differentiates without dragging in the
# crashing convolution code.
#
# The trait is selected by DISPATCH on the component `Tuple` TYPE, returning a
# concrete singleton -- NOT a runtime `cond ? _Nested() : _Flat()` ternary, whose
# result is a `Union{_Nested, _Flat}` value that forces a dynamic dispatch on the
# trait. Mooncake's reverse-mode codegen crashes (uncatchable `signal 4`) on that
# union-typed dispatch, so the trait must resolve to a single concrete type at
# compile time.
struct _Nested end
struct _Flat end
_nested_trait(components::C) where {C <: Tuple} = _nested_trait(C)
@generated function _nested_trait(::Type{C}) where {C <: Tuple}
    # A `Competing` component also forces the tree path: its multi-slot outcome
    # layout is scored by `_tree_step(::Competing)`, not the flat segment-grouped
    # scorer. A `Select` component likewise forces the tree path: a nested Select
    # routes per row to one of its alternatives (scored by `_tree_step(::Select)`),
    # not by the flat one-level segment scoring that would silently commit to a
    # single alternative.
    has_nested = any(
        t -> t <: Sequential || t <: Parallel || t <: AbstractCompeting ||
             t <: Select,
        fieldtypes(C))
    return has_nested ? :(_Nested()) : :(_Flat())
end

# Whether a composer's components carry primary censoring (an origin primary
# event), checked on the component types. `true` selects the censored
# full-event-path `rand`; `false` keeps the generic per-leaf-value `rand`.
function _is_censored_composer(components::Tuple)
    any(c -> _origin_primary_event(c) !== nothing, components)
end

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
_terminal_offset(d::Select) = _terminal_offset(first(d.alternatives))
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
function _nested_tree_logpdf(d::Union{Sequential, Parallel}, events,
        primary, ::Type{T}) where {T}
    return _tree_score(d, events, 1, 2, primary, T)
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
        primary, ::Type{T}) where {T}
    comps = d.components
    first_step = first(comps)
    acc = _tree_step(first_step, events, origin_idx, event_start, primary, T)
    o_idx = event_start + _terminal_offset(first_step)
    ev_idx = event_start + _event_child_nleaves(first_step)
    @inbounds for step in Base.tail(comps)
        acc += _tree_step(step, events, o_idx, ev_idx, nothing, T)
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
        primary, ::Type{T}) where {T}
    acc = zero(T)
    ev_idx = event_start
    @inbounds for branch in d.components
        acc += _tree_step(branch, events, origin_idx, ev_idx, primary, T)
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
        ev_idx::Int, primary, ::Type{T}) where {T}
    # The sub-view spans the node's EVENT slots (a `Competing` step contributes
    # one slot per outcome), so use the event-slot count, not `length`.
    sub = _subevent_slice(events, o_idx, ev_idx, _event_nleaves(step.components))
    return _tree_score(step, sub, 1, 2, primary, T)
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
        primary, ::Type{T}) where {T}
    return _tree_step(first(step.alternatives), events, o_idx, ev_idx,
        primary, T)
end

function _tree_step(step::UnivariateDistribution, events, o_idx::Int,
        ev_idx::Int, primary, ::Type{T}) where {T}
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

# --- competing outcome-slice layout (leaf OR composer subtree, #466 F3) -------
#
# Each competing outcome occupies a CONTIGUOUS slice of the event vector starting
# at `ev_idx`: a LEAF outcome is one slot, a NON-TERMINAL outcome whose payload is
# a composer subtree is its whole subtree's event-slot width (`_event_child_
# nleaves`, matching `_walk_edge!`'s emitted names and `_event_child_nleaves`'s
# count). `_competing_outcome_start(delays, k)` is the 1-based offset of outcome
# `k`'s slice within the node's outcome region (outcome 1 at 0). Pure-Int
# bookkeeping over the constant tree structure, so it stays off the AD'd values.
function _competing_outcome_start(delays::Tuple, k::Int)
    off = 0
    @inbounds for j in 1:(k - 1)
        off += _competing_outcome_slots(delays[j])
    end
    return off
end

# Whether ANY slot of outcome `k`'s slice (width `w`, beginning at absolute index
# `start`) is observed (non-missing). A composer outcome is "observed" when any of
# its subtree slots is present (e.g. a `death => chain` record fills one of the
# chain's leaf events); a leaf outcome is observed when its single slot is.
function _competing_outcome_observed(events, start::Int, w::Int)
    @inbounds for s in 0:(w - 1)
        events[start + s] === missing || return true
    end
    return false
end

# Resolve WHICH competing outcome a record observes by scanning each outcome's
# slice for a present slot, erroring if two distinct outcomes are both observed (a
# record resolves to exactly one outcome). Returns `(obs_i, obs_start, obs_w)`
# with `obs_i == 0` when no outcome is observed. The anchor is the parent event
# `events[o_idx]` (a competing outcome's slice hangs off it). Pure control flow
# over the constant event vector.
function _resolve_competing_outcome(delays::Tuple, names::Tuple, events,
        ev_idx::Int)
    n = length(delays)
    obs_i = 0
    obs_start = 0
    obs_w = 0
    off = 0
    @inbounds for k in 1:n
        w = _competing_outcome_slots(delays[k])
        start = ev_idx + off
        if _competing_outcome_observed(events, start, w)
            obs_i == 0 || throw(ArgumentError(
                "a nested Competing record may observe at most one outcome; " *
                "got outcomes $(names[obs_i]) and $(names[k])"))
            obs_i = k
            obs_start = start
            obs_w = w
        end
        off += w
    end
    return obs_i, obs_start, obs_w
end

# A nested `Competing` node SELF-DISPATCHES on which OUTCOME a record observes,
# mirroring the top-level `Competing` self-dispatch but anchored at the parent
# event. Its outcome slices begin at `ev_idx`; the anchor (the parent origin) is
# `events[o_idx]`. The numeric event-vector path uses the node's STORED branch
# probabilities (the per-record `branch_probs` override is a ROW input, applied by
# the DynamicPPL extension which re-anchors a nested Competing with row context).
# Exactly one outcome observed -> score that branch's mixture weight `log p_k`
# plus its payload term (a leaf conditions on its delay at the gap; a composer
# subtree recurses through `_tree_score` on the outcome's slice, anchored at the
# shared parent origin); no outcome observed -> contributes no factor (the
# resolved-but-unknown-outcome encoding for a NESTED Competing is deferred).
function _tree_step(step::Competing, events, o_idx::Int, ev_idx::Int,
        primary, ::Type{T}) where {T}
    return _competing_tree_logpdf(step, step.branch_probs, o_idx,
        events, ev_idx, primary, T)
end

# Score a nested `Competing` against its outcome slices given the anchor INDEX
# `o_idx` and branch probabilities `probs` (stored or row-overridden). Pure
# Turing-free arithmetic shared by the numeric path and the DynamicPPL extension.
# A no-event branch's slot is a PRESENCE marker (its value is not a time): a
# non-missing no-event slot is an OBSERVED non-occurrence scoring `log q` (no
# delay term), a missing no-event slot is a latent non-occurrence contributing
# nothing. An observed real LEAF outcome conditions on that branch; a COMPOSER
# outcome scores `log p_k + _tree_score(subtree, slice)` (the mixture weight plus
# its subtree's own censored event-vector density anchored at the parent origin).
function _competing_tree_logpdf(c::Competing, probs, o_idx::Int, events,
        ev_idx::Int, primary, ::Type{T}) where {T}
    obs_i,
    obs_start,
    obs_w = _resolve_competing_outcome(c.delays, c.names, events, ev_idx)
    obs_i == 0 && return zero(T)
    delay = c.delays[obs_i]
    # An OBSERVED non-occurrence scores the no-event mass `log q` alone.
    _is_no_event(delay) && return log(probs[obs_i])
    o = events[o_idx]
    return log(probs[obs_i]) +
           _competing_outcome_payload_logpdf(delay, o, o_idx, events,
        obs_start, obs_w, primary, T)
end

# The payload term of an observed competing outcome: a LEAF conditions on its
# (declared-censored) delay at the gap from the anchor; a COMPOSER subtree scores
# its own censored event-vector density on the outcome's slice, anchored at the
# parent origin (shared like a nested-composer origin). The mixture weight
# `log p_k` is added by the caller, so this is the conditional `f(payload | k)`.
function _competing_outcome_payload_logpdf(delay::UnivariateDistribution, o,
        o_idx::Int, events, obs_start::Int, obs_w::Int, primary, ::Type{T}
) where {T}
    y = events[obs_start]
    o === missing && throw(ArgumentError(
        "a nested Competing with an observed outcome needs an observed anchor " *
        "(its parent event); got a missing anchor"))
    return logpdf(delay, convert(T, y) - convert(T, o))
end

# A COMPOSER outcome: recurse through `_tree_score` on the `[origin, slice...]`
# sub-event vector, the outcome's resolution sharing the parent origin (the
# subtree origin slot). The subtree's leaf events occupy `obs_w` slots from
# `obs_start`; the parent origin anchors them. The subtree may itself be censored,
# so its own origin primary (if any) seeds the recursion; the parent-level primary
# is passed through for a sampled-origin sub-chain.
function _competing_outcome_payload_logpdf(
        delay::Union{Sequential, Parallel}, o, o_idx::Int, events,
        obs_start::Int, obs_w::Int, primary, ::Type{T}) where {T}
    sub = _subevent_slice(events, o_idx, obs_start, obs_w)
    sub_primary = _subtree_origin_primary(delay, primary)
    return _tree_score(delay, sub, 1, 2, sub_primary, T)
end

# A `Select` outcome routes to its committed (first) alternative on the numeric
# path (the data path resolves Selects out before scoring); score that.
function _competing_outcome_payload_logpdf(delay::Select, o, o_idx::Int, events,
        obs_start::Int, obs_w::Int, primary, ::Type{T}) where {T}
    return _competing_outcome_payload_logpdf(first(delay.alternatives), o,
        o_idx, events, obs_start, obs_w, primary, T)
end

# A nested `Competing` outcome (a competing node as a competing branch): recurse
# through the competing scorer on the outcome's slice, anchored at the parent
# origin (the inner competing's outcomes hang off the same shared anchor).
function _competing_outcome_payload_logpdf(delay::Competing, o, o_idx::Int,
        events, obs_start::Int, obs_w::Int, primary, ::Type{T}) where {T}
    sub = _subevent_slice(events, o_idx, obs_start, obs_w)
    return _competing_tree_logpdf(delay, delay.branch_probs, 1, sub, 2,
        primary, T)
end

function _competing_outcome_payload_logpdf(delay::HazardCompeting, o, o_idx::Int,
        events, obs_start::Int, obs_w::Int, primary, ::Type{T}) where {T}
    sub = _subevent_slice(events, o_idx, obs_start, obs_w)
    return _hazard_competing_tree_logpdf(delay, 1, sub, 2, primary, T)
end

# The origin primary event seeding a COMPOSER competing outcome's subtree. A
# censored subtree surfaces its own shared latent origin (e.g. a chain whose first
# step is `primary_censored`); a plain subtree has none and inherits the parent's
# primary (a sampled-origin sub-chain off the competing anchor). Mirrors the
# `_tree_score` root seeding.
function _subtree_origin_primary(d::Sequential, parent_primary)
    p = _origin_primary_event(_first_origin_node(d))
    return p === nothing ? parent_primary : p
end
function _subtree_origin_primary(d::Parallel, parent_primary)
    p = _shared_primary_event(d.components)
    return p === nothing ? parent_primary : p
end

# A nested racing-hazard `HazardCompeting` node SELF-DISPATCHES on which OUTCOME a
# record observes, anchored at the parent event `events[o_idx]`. Exactly one
# outcome observed -> its payload term (a leaf scores the cause-resolved
# sub-density `log f_j(t) + Σ_{k≠j} log S_k(t)`; a composer subtree the racing
# survival PLUS the subtree's own event-vector density); no outcome observed ->
# no factor (a fully latent record). The winning probability is DERIVED, so there
# is no branch-probability term.
function _tree_step(step::HazardCompeting, events, o_idx::Int, ev_idx::Int,
        primary, ::Type{T}) where {T}
    return _hazard_competing_tree_logpdf(step, o_idx, events, ev_idx, primary, T)
end

function _hazard_competing_tree_logpdf(step::HazardCompeting, o_idx::Int, events,
        ev_idx::Int, primary, ::Type{T}) where {T}
    obs_i,
    obs_start,
    obs_w = _resolve_competing_outcome(step.delays, step.names, events, ev_idx)
    obs_i == 0 && return zero(T)
    o = events[o_idx]
    return _hazard_outcome_payload_logpdf(step, obs_i, o, o_idx, events,
        obs_start, obs_w, primary, T)
end

# A LEAF racing outcome: the cause-resolved sub-density at the observed gap. The
# log density carries any AD `Dual`/tracked type from the racing delays' params,
# so it is NOT narrowed to the data type `T` (only the `obs_gap`, data, is `T`).
function _hazard_outcome_payload_logpdf(step::HazardCompeting, obs_i::Int, o,
        o_idx::Int, events, obs_start::Int, obs_w::Int, primary, ::Type{T}
) where {T}
    return _hazard_outcome_payload(step, obs_i, step.delays[obs_i], o, o_idx,
        events, obs_start, obs_w, primary, T)
end

function _hazard_outcome_payload(step::HazardCompeting, obs_i::Int,
        delay::UnivariateDistribution, o, o_idx::Int, events, obs_start::Int,
        obs_w::Int, primary, ::Type{T}) where {T}
    y = events[obs_start]
    o === missing && throw(ArgumentError(
        "a nested HazardCompeting with an observed outcome needs an observed " *
        "anchor (its parent event); got a missing anchor"))
    return _hazard_cause_logpdf(step, obs_i, convert(T, y) - convert(T, o))
end

# A COMPOSER racing outcome (#466 F3): the racing SURVIVAL up to the subtree's
# resolution PLUS the subtree's own censored event density. The cause-resolved
# weighting of a non-terminal racing branch is its survival contribution; the
# subtree then scores the within-branch path conditional on that cause winning.
# The anchor for the subtree is the parent origin (shared).
function _hazard_outcome_payload(step::HazardCompeting, obs_i::Int,
        delay::Union{Sequential, Parallel}, o, o_idx::Int, events,
        obs_start::Int, obs_w::Int, primary, ::Type{T}) where {T}
    sub = _subevent_slice(events, o_idx, obs_start, obs_w)
    sub_primary = _subtree_origin_primary(delay, primary)
    return _tree_score(delay, sub, 1, 2, sub_primary, T)
end

# --- per-record branch-probability override for a nested Competing -----
#
# A per-record `branch_probs` override is a ROW input (the covariate CFR
# `logistic(Xβ)` flows in per record), so it cannot ride the numeric event
# vector. Rather than thread it through the AD-sensitive tree recursion, the
# DynamicPPL extension REBUILDS the tree with the (single) Competing node's
# probabilities replaced for the record, then scores the rebuilt tree through the
# normal numeric path (whose `_tree_step(::Competing)` reads the now-overridden
# stored probs). The override's element type is preserved, so a `logistic(Xβ)`
# `Dual` flows through the rebuilt node and differentiates.

# Count the `Competing` nodes anywhere in a composed tree, so a single per-record
# `branch_probs` field is rejected as ambiguous when more than one node exists.
_count_competing(c::Competing) = 1
_count_competing(::UnivariateDistribution) = 0
function _count_competing(d::Union{Sequential, Parallel})
    return sum(_count_competing, d.components; init = 0)
end

# Rebuild a composed tree with the single `Competing` node's branch probabilities
# replaced by `probs` (already coerced to outcome order and validated). Errors if
# the tree has no Competing node or more than one (the single `branch_probs` row
# field is then ambiguous). Pure, Turing-free; the new probs keep their
# element type.
function _override_competing_outcome_probs(d, probs)
    n = _count_competing(d)
    n == 1 || throw(ArgumentError(
        "a per-record `branch_probs` override needs exactly one Competing node " *
        "in the tree; found $n"))
    return _replace_competing(d, probs)
end

_replace_competing(c::Competing, probs) = Competing(c.names, c.delays, probs)
_replace_competing(d::UnivariateDistribution, probs) = d
function _replace_competing(d::Sequential, probs)
    return Sequential(map(c -> _replace_competing(c, probs), d.components),
        d.names)
end
function _replace_competing(d::Parallel, probs)
    return Parallel(map(c -> _replace_competing(c, probs), d.components),
        d.names)
end

# --- per-record nested-Select routing ---------------------------------------
#
# A nested `Select` routes per record by the row's selector field
# (`row[selector]`). On the DATA path the per-record build RESOLVES every nested
# Select into the tree, replacing each with its routed alternative for the record,
# then scores the resolved (Select-free) tree through the normal numeric path. The
# selector VALUE is data (a Symbol), so resolving it out of the tree before the
# differentiated scoring keeps the AD path free of the routing control flow and
# mirrors the per-record `branch_probs` rebuild for a nested Competing. A row
# missing a needed selector field errors here, so a data record can never silently
# score alternative 1.

# Count the nested `Select` nodes anywhere in a composed tree, so a tree with no
# Select skips the resolution rebuild entirely.
_count_selects(::Select) = 1
_count_selects(::UnivariateDistribution) = 0
function _count_selects(d::Union{Sequential, Parallel})
    return sum(_count_selects, d.components; init = 0)
end

# Resolve every nested `Select` in `d` to its routed alternative for `row`,
# returning the rebuilt (Select-free) tree. A row missing a Select's selector
# field errors clearly (no silent commit to alternative 1). A tree with no nested
# Select is returned unchanged. The chosen alternative may itself nest a Select,
# so the rebuild recurses into it.
_resolve_selects(d, row::NamedTuple) = _resolve_selects_node(d, row)

_resolve_selects_node(d::UnivariateDistribution, ::NamedTuple) = d
function _resolve_selects_node(d::Select, row::NamedTuple)
    haskey(row, d.selector) || throw(ArgumentError(
        "a nested Select needs its selector field $(repr(d.selector)) on the " *
        "record to route; the row has no such field, so it cannot pick an " *
        "alternative (it would otherwise silently score the first one)"))
    kind = row[d.selector]
    kind isa Symbol || throw(ArgumentError(
        "the nested Select selector field $(repr(d.selector)) must hold a " *
        "Symbol naming the alternative; got $(typeof(kind))"))
    return _resolve_selects_node(_pick(d, kind), row)
end
function _resolve_selects_node(d::Sequential, row::NamedTuple)
    return Sequential(map(c -> _resolve_selects_node(c, row), d.components),
        d.names)
end
function _resolve_selects_node(d::Parallel, row::NamedTuple)
    return Parallel(map(c -> _resolve_selects_node(c, row), d.components),
        d.names)
end

# ---------------------------------------------------------------------------
# Sequential: per-record missingness dispatch over a censored chain
# ---------------------------------------------------------------------------

# A `Sequential` chain of `k` steps spans `k + 1` events `E_0, ..., E_k`. Scored
# against an EVENT vector (one entry per event, `missing` admitted) the chain
# marginalises unobserved intermediates and conditions on observed ones. The
# element-type dispatch (`>: Missing`) keeps the all-concrete one-value-per-step
# generic path untouched: a `Vector{Float64}` of step gaps still hits the
# generic `logpdf`, while a `Vector{Union{Missing, Float64}}` event vector hits
# this specialisation.

@doc raw"

Log density of a [`Sequential`](@ref) chain scored against an EVENT vector.

`events` has one entry per event ``E_0, \dots, E_k`` (length `length(d) + 1`);
each entry is a value (observed) or `missing` (unobserved). The chain is grouped
into segments by the missingness pattern and the log density is the sum of the
per-segment log-densities at the observed gaps:

- the origin's primary censoring (from the first step) is always applied to the
  first segment;
- an unobserved intermediate is marginalised by convolving the continuous cores
  of the delays it spans and dropping their censoring;
- an observed intermediate conditions: its adjacent delay is an independent
  factor at the observed gap.

Missingness drives only the control flow grouping the segments; the
differentiated arithmetic sees only concrete observed values, so this is safe to
differentiate with `events` held constant.

See also: [`Sequential`](@ref), [`Parallel`](@ref)
"
function logpdf(d::Sequential, events::AbstractVector{T}) where {T >: Missing}
    # The flat event vector carries one entry per EVENT slot plus the root
    # origin. A nested composer contributes its whole subtree; a `Competing`
    # contributes one slot per OUTCOME, so the count is
    # `_event_nleaves(d.components) + 1`, not `length(d) + 1` (the value layout).
    n = _event_nleaves(d.components) + 1
    length(events) == n || throw(DimensionMismatch(
        "a Sequential event vector needs $n entries " *
        "(one per event), got $(length(events))"))
    # Dispatch on the nested/flat trait so the flat and nested scoring are
    # separate compiled methods (see `_nested_trait`): a chain with a nested
    # composer step recurses through the tree, a flat chain keeps the one-level
    # segment-grouped scoring, and neither method body carries the other's code.
    return _seq_event_logpdf(_nested_trait(d.components), d, events)
end

function _seq_event_logpdf(::_Nested, d::Sequential, events)
    return _nested_tree_logpdf(d, events,
        _origin_primary_event(_first_origin_node(d)),
        _tree_acc_type(d, events))
end

function _seq_event_logpdf(::_Flat, d::Sequential, events)
    return _seq_event_logpdf_h(d, events, nothing)
end

# Flat `Sequential` scoring with an OPTIONAL per-record observation horizon
# `horizon` (real-time right-truncation). The WHOLE composed distribution is
# truncated per record at the horizon (the same combine-then-censor semantics as
# wrapping a chain in `double_interval_censored`): the record is included only if
# its LAST OBSERVED event occurred by the horizon `D`. Resolved semantics (#366,
# maintainer decision): whole-compose TOTAL truncation.
#
#   numerator   = the untruncated factorised per-segment density (unchanged): an
#                 observed intermediate conditions on its own edge at the observed
#                 gap; an unobserved-intermediate run convolves its cores.
#   denominator = a SINGLE term `-logcdf(C, window)` where `C` is the convolution
#                 of all components from the origin to the LAST OBSERVED event
#                 (the origin primary reapplied) and `window = D - origin`.
#
# This is the standard real-time right-truncation (a record is in the sample iff
# its last observed event has occurred by `D`) and is well-defined for ALL
# observed patterns, so it covers BOTH the endpoint-observed hanta index/sourced
# shape (a single collapsed total) and the observed-intermediate case (the
# factorised numerator over the single conv-to-last-observed denominator). With a
# single observed segment the denominator's `C` IS that segment, so the result
# reduces to `truncate_to_horizon(seg, window)`. `horizon === nothing` leaves the
# scoring untruncated.
function _seq_event_logpdf_h(d::Sequential, events, horizon)
    horizon === nothing && return _seq_event_logpdf_untrunc(d, events)
    obs_idx, obs_val = _observed_indices_values(events)
    length(obs_idx) >= 2 || throw(ArgumentError(
        "a Sequential event vector needs at least two observed events"))
    primary = _origin_primary_event(d.components[1])
    # Numerator: the untruncated factorised per-segment density. The origin
    # primary is reapplied only to the segment that actually starts at the latent
    # origin E_0 (`obs_idx[j] == 1`), matching `_sequential_segment`'s `a == 1`
    # contract and the vectorised `_build_seq_bundle` run.
    total = zero(promote_type(eltype(obs_val), float(eltype(d))))
    for j in 1:(length(obs_idx) - 1)
        seg = _sequential_segment(
            d.components, obs_idx[j], obs_idx[j + 1],
            obs_idx[j] == 1 ? primary : nothing)
        gap = obs_val[j + 1] - obs_val[j]
        total += logpdf(seg, gap)
    end
    # Denominator: a single conv-to-last-observed right-truncation term. `C` is
    # the convolution of every component from the origin to the LAST observed
    # event, truncated at the remaining window from the observed origin. With one
    # observed segment `C` is that segment, so this matches the endpoint-observed
    # term `-logcdf(seg, window)`. The origin primary is reapplied only when the
    # first observed event IS the latent origin E_0 (`obs_idx[1] == 1`), matching
    # `_sequential_segment`'s `a == 1` contract and the vectorised
    # `_build_seq_bundle` run; a denominator anchored at an observed intermediate
    # (origin unobserved) is an exact-time anchor, not the latent primary.
    last_seg = _sequential_segment(
        d.components, obs_idx[1], obs_idx[end],
        obs_idx[1] == 1 ? primary : nothing)
    window = horizon - obs_val[1]
    # A non-positive window (the horizon already passed the observed origin) is an
    # empty-support truncation: the record cannot have been observed, so the whole
    # contribution is `-Inf`, matching the `truncate_to_horizon` empty-support
    # guard rather than the `+Inf` a bare `total - logcdf(., minimum)` would give.
    window <= minimum(last_seg) &&
        return convert(typeof(total), -Inf)
    return total - logcdf(last_seg, window)
end

# The untruncated flat-chain scoring (the original per-segment factorisation).
function _seq_event_logpdf_untrunc(d::Sequential, events)
    # Pre-pass on the constant event vector: collect the observed event indices
    # and their concrete values. The `Union{Missing}` entries are read only
    # here, so the differentiated arithmetic below touches only concrete gaps.
    obs_idx, obs_val = _observed_indices_values(events)
    length(obs_idx) >= 2 || throw(ArgumentError(
        "a Sequential event vector needs at least two observed events"))

    primary = _origin_primary_event(d.components[1])
    total = zero(promote_type(eltype(obs_val), float(eltype(d))))
    for j in 1:(length(obs_idx) - 1)
        # Reapply the origin primary only to the segment starting at the latent
        # origin E_0 (`obs_idx[j] == 1`); a leading run anchored at an observed
        # event is an exact-time anchor, matching the vectorised build.
        seg = _sequential_segment(
            d.components, obs_idx[j], obs_idx[j + 1],
            obs_idx[j] == 1 ? primary : nothing)
        gap = obs_val[j + 1] - obs_val[j]
        total += logpdf(seg, gap)
    end
    return total
end

@doc "

Density of a [`Sequential`](@ref) chain scored against an EVENT vector.

See also: [`logpdf`](@ref)
"
function pdf(d::Sequential, events::AbstractVector{T}) where {T >: Missing}
    return exp(logpdf(d, events))
end

# Walk the (constant) event vector and return the observed event indices and
# their concrete values. Kept separate from the arithmetic so the
# `Union{Missing}` handling is pure control flow.
function _observed_indices_values(events)
    idx = Int[]
    val = Float64[]
    for i in eachindex(events)
        o = events[i]
        o === missing && continue
        push!(idx, i)
        push!(val, Float64(o))
    end
    return idx, val
end

# Segment distribution spanning observed events at 1-based event indices `a` and
# `b`. The steps linking them are `components[a:(b - 1)]`.
#
# A SINGLE observed-bounded edge conditions on its OWN declared censoring: the
# edge's gap is scored through the edge distribution AS DECLARED ("condition
# on its (censored) value"). A `primary_censored` / `double_interval_censored` /
# `interval_censored` edge therefore keeps that censoring (real linelist data is
# day-resolution, so a day-observed edge delay is interval-censored, not an exact
# continuous time); a plain continuous edge is conditioned on the continuous
# value, which is correct for that edge.
#
# A RUN of two or more steps (one or more UNOBSERVED intermediate events between
# the two observed endpoints) is marginalised: the spanned steps' continuous
# cores are convolved and the intermediates' own censoring is dropped (the latent
# intermediate is a continuous time, not a windowed observation). When a primary
# event is supplied (the origin segment, whose origin E_0 is the latent primary)
# it is reapplied to the convolved core via `primary_censored`.
function _sequential_segment(components, a, b, primary)
    run = components[a:(b - 1)]
    # Single observed-bounded edge: respect the edge's declared censoring.
    length(run) == 1 && return run[1]
    # Unobserved-intermediate run: convolve the continuous cores; the origin
    # segment reapplies the (latent) primary event.
    core = convolve_distributions(map(_marginal_core, collect(run)))
    primary === nothing && return core
    return primary_censored(core, primary)
end

# ---------------------------------------------------------------------------
# Parallel: shared-origin marginalisation over censored branches
# ---------------------------------------------------------------------------

# When the branches of a `Parallel` are censored they share one latent origin
# (the common primary event), so the branch observations are coupled through
# that origin. Scored against an event vector `[O, Y_1, ..., Y_n]` (origin slot
# first), a missing origin marginalises by one 1-D integral over the shared
# origin and a present origin conditions. A missing branch drops from the joint.
# The branch primaries must agree (one shared origin); they are checked once.

@doc raw"

Log density of a [`Parallel`](@ref) of censored branches scored against the
shared-origin event vector ``[O, Y_1, \dots, Y_n]``.

The branches share one latent origin ``O`` (their common primary event), so the
observations are coupled: ``Y_i = O + D_i`` with independent branch delays
``D_i``. The first entry is the shared origin and the rest are the branch
observations (each a value or `missing`):

- a missing origin is marginalised by a single one-dimensional integral over the
  shared origin against the present-branch delay densities,
  ``\int f_O(o) \prod_{i \text{ present}} f_{D_i}(y_i - o)\, do``;
- a present origin conditions: ``\log f_O(o) + \sum_{i} \log f_{D_i}(y_i - o)``;
- a missing branch drops from the joint.

The branch primaries must agree (one shared origin). Missingness drives only
control flow, so this differentiates with the event vector held constant.

See also: [`Parallel`](@ref), [`Sequential`](@ref)
"
function logpdf(d::Parallel, events::AbstractVector{T}) where {T >: Missing}
    # One entry per EVENT slot plus the shared origin; a tree branch contributes
    # its whole subtree's events and a `Competing` one slot per OUTCOME,
    # so the length is `_event_nleaves(d.components) + 1`.
    n = _event_nleaves(d.components) + 1
    length(events) == n || throw(DimensionMismatch(
        "a Parallel event vector needs $n entries " *
        "(shared origin then one per branch), got $(length(events))"))
    # Dispatch on the nested/flat trait so the nested tree recursion and the flat
    # shared-origin (coupled-integral) scoring are separate compiled methods (see
    # `_nested_trait`); the flat coupled `gl_integrate` path stays out of the
    # nested method that the compiled AD backends differentiate.
    return _par_event_logpdf(_nested_trait(d.components), d, events)
end

function _par_event_logpdf(::_Nested, d::Parallel, events)
    return _nested_tree_logpdf(d, events,
        _shared_primary_event(d.components), _tree_acc_type(d, events))
end

function _par_event_logpdf(::_Flat, d::Parallel, events)
    primary = _shared_primary_event(d.components)
    # Plain branches: no shared primary to integrate, so the origin must be
    # observed; condition each branch on its gap and drop missing branches.
    primary === nothing && return _par_plain_logpdf(d, events)

    cores = map(_marginal_core, d.components)
    # Promote over the PARAMETER types of the cores and primary (which carry any
    # AD Dual from the leaf params; a distribution's `eltype` is its variate type,
    # not its parameter type, so it would drop the Dual) plus the event vector,
    # so the marginal/condition arithmetic stays on the differentiated type.
    T2 = promote_type(_event_eltype(events), _param_eltype(primary),
        map(_param_eltype, cores)...)
    origin = events[1]
    if origin === missing
        return _parallel_marginal_logpdf(primary, cores, events, T2)
    end
    return _parallel_conditional_logpdf(primary, cores, events, T2)
end

# Plain-branch Parallel sharing an exactly-observed continuous origin: condition
# each present branch on its gap `y_i - o` (no primary prior, no integral); drop
# missing branches. A missing origin cannot be marginalised without a
# distribution for it, so it is rejected (promote it to a primary or go latent).
function _par_plain_logpdf(d::Parallel, events)
    origin = events[1]
    origin === missing && throw(ArgumentError(
        "a plain-branch Parallel needs an observed shared origin to condition " *
        "on; a missing continuous origin cannot be marginalised without a " *
        "distribution for it (declare a primary event or pass a latent origin)"))
    T = promote_type(_event_eltype(events),
        map(_param_eltype, d.components)...)
    o = events[1]
    o isa Real || return convert(T, -Inf)
    isnan(o) && return convert(T, NaN)
    lp = zero(T)
    @inbounds for i in 2:length(events)
        y = events[i]
        y === missing && continue
        isnan(y) && return convert(T, NaN)
        u = y - o
        branch = d.components[i - 1]
        insupport(branch, u) || return convert(T, -Inf)
        lp += convert(T, logpdf(branch, u))
    end
    return lp
end

@doc "

Density of a [`Parallel`](@ref) of censored branches scored against the
shared-origin event vector.

See also: [`logpdf`](@ref)
"
function pdf(d::Parallel, events::AbstractVector{T}) where {T >: Missing}
    return exp(logpdf(d, events))
end

# ---------------------------------------------------------------------------
# Per-record observation-horizon right-truncation (hanta)
# ---------------------------------------------------------------------------
#
# `event_logpdf(d, events; horizon)` is the horizon-aware event-vector log
# density. With `horizon === nothing` it is exactly the censored composer
# `logpdf(d, events)` (back-compat). With a per-record `horizon` the WHOLE
# composed distribution is right-truncated at that observation time for the
# record, the same combine-then-censor direction as wrapping the compose in
# `double_interval_censored` (`wrap.jl`) but with the upper bound supplied PER
# RECORD rather than baked in. For a `Sequential` the record's observed total
# (origin -> terminal) is truncated at `horizon - origin` (the endpoint-observed
# hanta index/sourced shape); for a `Parallel` each branch endpoint is truncated
# at `horizon - origin` off the shared origin. `truncate_to_horizon` /
# `_truncate_window` are the implementation primitive (upper-only, AD-safe, with
# the non-positive-window empty-support guard); they are NOT a user-facing
# per-segment interface here.

@doc raw"

Horizon-aware event-vector log density of a censored composer.

`event_logpdf(d, events; horizon)` scores `events` exactly as
`logpdf(d, events)` when `horizon === nothing`. With a per-record `horizon` the
WHOLE composed distribution is right-truncated at that observation time for the
record, the same combine-then-censor direction as wrapping the compose in
[`double_interval_censored`](@ref) but with the upper bound supplied per record.

For a [`Sequential`](@ref) the truncation is whole-compose TOTAL truncation
(#366): the factorised per-segment numerator is divided by a single
``F(\text{window})`` denominator, where the denominator delay is the convolution
of every component from the origin to the LAST OBSERVED event and
``\text{window} = \text{horizon} - \text{origin}``. A record is then included
only if its last observed event occurred by the horizon. This is well-defined for
all observed patterns, so observed intermediates are scored (their numerator
factorises, the single conv-to-last-observed denominator truncates), not rejected.
For a [`Parallel`](@ref) each branch endpoint is truncated at
``\text{horizon} - \text{origin}`` off the shared origin. The truncation
contributes the ``-\log F(\text{window})`` correction, upper-only and AD-safe.

# Arguments
- `d`: a censored [`Sequential`](@ref) or [`Parallel`](@ref) composer.
- `events`: the flat event vector `[E_0, ..., E_k]` (value or `missing` each).

# Keyword Arguments
- `horizon`: the per-record observation horizon, or `nothing` for no truncation.

# Examples
```@example
using CensoredDistributions, Distributions

seq = Sequential(
    primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
    primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
# Endpoint-observed record (intermediate unobserved): the total is truncated.
ev = Vector{Union{Missing, Float64}}([0.0, missing, 5.0])
CensoredDistributions.event_logpdf(seq, ev; horizon = 8.0)
```

# See also
- [`double_interval_censored`](@ref): the whole-compose wrap with a baked-in
  bound
- [`truncate_to_horizon`](@ref): the truncation primitive
"
function event_logpdf(
        d::Sequential, events::AbstractVector{T}; horizon = nothing
) where {T >: Missing}
    horizon === nothing && return logpdf(d, events)
    # A nested chain has no single observed origin/terminal total; whole-compose
    # truncation is defined on the flat chain (the andv shape).
    _nested_trait(d.components) isa _Flat || throw(ArgumentError(
        "per-record horizon truncation is defined for a flat Sequential " *
        "chain; a nested tree has no single observed total to truncate"))
    return _seq_event_logpdf_h(d, events, horizon)
end

function event_logpdf(
        d::Parallel, events::AbstractVector{T}; horizon = nothing
) where {T >: Missing}
    horizon === nothing && return logpdf(d, events)
    _nested_trait(d.components) isa _Flat || throw(ArgumentError(
        "per-record horizon truncation is defined for a flat Parallel set; " *
        "a nested tree has no single observed total to truncate"))
    return _par_event_logpdf_h(d, events, horizon)
end

# A univariate leaf record's horizon truncation: the leaf is observed from the
# (zero) origin, so the window is the horizon itself.
function event_logpdf(d::UnivariateDistribution, x::Real; horizon = nothing)
    horizon === nothing && return logpdf(d, x)
    return logpdf(truncate_to_horizon(d, horizon), x)
end

# Flat `Parallel` with a horizon: every branch endpoint shares the observed
# origin, so each is right-truncated at `horizon - origin` (each branch is its own
# endpoint, no intermediate, so the whole-compose meaning is unambiguous). A
# missing (marginalised) origin has no observed anchor, so truncation would
# require integrating the denominator over the origin; that is out of scope and
# rejected with a clear error.
function _par_event_logpdf_h(d::Parallel, events, horizon)
    origin = events[1]
    origin === missing && throw(ArgumentError(
        "per-record horizon truncation needs an observed shared origin; a " *
        "missing (marginalised) origin has no anchor to truncate from"))
    primary = _shared_primary_event(d.components)
    primary === nothing && throw(ArgumentError(
        "Parallel shared-origin scoring needs censored branches with a " *
        "primary event; got plain branches"))
    o = Float64(origin)
    cores = map(_marginal_core, d.components)
    T2 = promote_type(_event_eltype(events), _param_eltype(primary),
        map(_param_eltype, cores)...)
    insupport(primary, o) || return convert(T2, -Inf)
    lp = convert(T2, logpdf(primary, o))
    window = horizon - o
    @inbounds for i in eachindex(cores)
        y = events[i + 1]
        y === missing && continue
        seg = truncate_to_horizon(cores[i], window)
        u = Float64(y) - o
        lp += convert(T2, logpdf(seg, u))
    end
    return lp
end

# Element type of an event vector, treating the `missing`-only / mixed slots as
# Float64 so the arithmetic stays concrete.
function _event_eltype(events)
    (eltype(events) === Missing) ? Float64 :
    promote_type(Float64, nonmissingtype(eltype(events)))
end

# Numeric (parameter) type of a distribution, recovered from its flattened
# `params`. A distribution's `eltype` is its variate type (`Float64`) and would
# drop an AD `Dual` carried by the parameters, so the arithmetic type is taken
# from the parameters instead. `Uniform`'s bounds are plain `Float64`; a censored
# delay's shape/scale carry the differentiated `Dual`.
function _param_eltype(d)
    return mapreduce(typeof, promote_type, _flatten_params(params(d));
        init = Float64)
end

# Flatten the (possibly nested) `params` of a distribution to its scalar leaves,
# so `_param_eltype` promotes over every parameter. The composed `params` is now
# a name-keyed `NamedTuple`, so a NamedTuple is flattened over its values
# just like a tuple; a bare scalar is its own leaf.
_flatten_params(t::Tuple) = mapreduce(_flatten_params, (a, b) -> (a..., b...), t;
    init = ())
function _flatten_params(nt::NamedTuple)
    return mapreduce(_flatten_params, (a, b) -> (a..., b...), values(nt);
        init = ())
end
_flatten_params(x) = (x,)

# The single shared primary event of a `Parallel`'s branches, or `nothing` when
# no branch carries primary censoring. Errors if the branches disagree on the
# primary event (a shared origin must be unique).
function _shared_primary_event(components::Tuple)
    primary = nothing
    @inbounds for c in components
        p = _origin_primary_event(c)
        p === nothing && continue
        if primary === nothing
            primary = p
        else
            primary == p || throw(ArgumentError(
                "Parallel shared-origin branches must share one primary " *
                "event; got $(primary) and $(p)"))
        end
    end
    return primary
end

# Present (non-missing) branch indices and their concrete observed values from
# the event vector `[O, Y_1, ...]`. Returns `nothing` values when a present
# observation is NaN so the caller propagates it.
function _parallel_present(events, ::Type{T}) where {T}
    present = Int[]
    vals = T[]
    @inbounds for i in 2:length(events)
        y = events[i]
        y === missing && continue
        isnan(y) && return present, nothing
        push!(present, i - 1)
        push!(vals, convert(T, y))
    end
    return present, vals
end

# Concrete origin: log f_O(o) + Σ_present log f_{D_i}(y_i - o). Missing branches
# drop out.
function _parallel_conditional_logpdf(primary, cores, events, ::Type{T}) where {T}
    o = events[1]
    o isa Real || return convert(T, -Inf)
    isnan(o) && return convert(T, NaN)
    insupport(primary, o) || return convert(T, -Inf)

    lp = convert(T, logpdf(primary, o))
    @inbounds for i in eachindex(cores)
        y = events[i + 1]
        y === missing && continue
        isnan(y) && return convert(T, NaN)
        u = y - o
        insupport(cores[i], u) || return convert(T, -Inf)
        lp += convert(T, logpdf(cores[i], u))
    end
    return lp
end

# Missing origin: marginalise the shared origin over the present branches by a
# single 1-D Gauss-Legendre integral. Missing branches contribute no factor and
# do not narrow the origin window.
function _parallel_marginal_logpdf(primary, cores, events, ::Type{T}) where {T}
    present, yvals = _parallel_present(events, T)
    yvals === nothing && return convert(T, NaN)
    isempty(present) && return zero(T)

    lo = float(minimum(primary))
    hi = float(maximum(primary))
    @inbounds for k in eachindex(present)
        i = present[k]
        a = float(minimum(cores[i]))
        b = float(maximum(cores[i]))
        y = yvals[k]
        lo = max(lo, y - b)
        hi = min(hi, y - a)
    end
    hi > lo || return convert(T, -Inf)

    val = gl_integrate(lo, hi, _PRIMARY_GL) do o
        p = pdf(primary, o)
        @inbounds for k in eachindex(present)
            p *= pdf(cores[present[k]], yvals[k] - o)
        end
        p
    end
    val <= 0 && return convert(T, -Inf)
    return convert(T, log(val))
end

# ---------------------------------------------------------------------------
# Full-path simulation: `rand` returns every event time / internal time
# ---------------------------------------------------------------------------
#
# For simulate-and-recover (the case studies) `rand` must produce the COMPLETE
# set of event times, not a summary. The composer-level `rand` delegates to
# `_composer_rand`, which branches on whether the composer is censored (its
# components carry an origin primary event). A plain composer keeps the generic
# per-leaf-value realisation; a censored composer simulates the full
# event-time path including the latent origin draw.

# A composer is simulated through three regimes, selected by structure:
#   - NESTED tree (a child is itself a composer) or a `Competing` child: walk the
#     tree sharing one latent origin draw, sampling each `Competing` outcome, and
#     return a NAMED event record keyed by `_flat_event_names` (the missing
#     unsampled Competing outcomes round-trip back through `logpdf`);
#   - FLAT censored chain/set: the full event-time path `[E_0, ...]` vector
#     (back-compat, the original censored-composer shape);
#   - plain composer: the generic per-leaf-value realisation.

# Sequential: dispatch on the nested/flat trait, then on whether the flat chain
# carries primary censoring.
function _composer_rand(rng::AbstractRNG, d::Sequential)
    return _composer_rand(_nested_trait(d.components), rng, d)
end

# Parallel: same trait split as Sequential.
function _composer_rand(rng::AbstractRNG, d::Parallel)
    return _composer_rand(_nested_trait(d.components), rng, d)
end

# A CENSORED nested tree draws the full named path (a shared latent origin to
# sample). A PLAIN nested tree has no origin distribution, so it keeps the generic
# per-leaf realisation (a Competing child stays its marginal time-to-resolution).
function _composer_rand(::_Nested, rng::AbstractRNG,
        d::Union{Sequential, Parallel})
    _tree_primary_event(d) === nothing && return _composite_rand(
        rng, d.components, float(eltype(d)))
    return _tree_event_record(rng, d)
end

# Flat censored chain: the full event-time path `[E_0, E_1, ..., E_k]` with `E_0`
# the latent origin draw and each subsequent time the previous plus a
# continuous-core delay draw, so every internal event time is in the result. A
# plain chain keeps the generic per-leaf realisation.
function _composer_rand(::_Flat, rng::AbstractRNG, d::Sequential)
    _is_censored_composer(d.components) || return _composite_rand(
        rng, d.components, float(eltype(d)))
    primary = _origin_primary_event(d.components[1])
    cores = map(_marginal_core, d.components)
    T = float(promote_type(_param_eltype(primary),
        map(_param_eltype, cores)...))
    times = Vector{T}(undef, length(cores) + 1)
    times[1] = convert(T, rand(rng, primary))
    @inbounds for i in eachindex(cores)
        times[i + 1] = times[i] + rand(rng, cores[i])
    end
    # Discretise the RECORDED times to each step's secondary interval, leaving the
    # continuous chaining above untouched so a chained anchor keeps its sub-day
    # jitter (flooring it would re-introduce the recovery bias).
    return _discretise_flat_seq(times, d.components)
end

# Apply each step's secondary interval to the recorded chain times AFTER the
# continuous walk. Slot `i + 1` is step `i`'s event; the origin slot is floored
# to the origin edge's interval so a first-level gap is unbiased (a primary-only
# origin edge leaves it continuous).
function _discretise_flat_seq(times, components::Tuple)
    out = copy(times)
    out[1] = _apply_leaf_interval(times[1], _origin_interval(components[1]))
    @inbounds for i in eachindex(components)
        out[i + 1] = _apply_leaf_interval(times[i + 1], _leaf_interval(
            components[i]))
    end
    return out
end

# Flat censored set: the shared-origin event-time vector `[O, Y_1, ..., Y_n]`
# with `O` the single shared origin draw and `Y_i = O + D_i`. Plain branches keep
# the generic per-leaf realisation.
function _composer_rand(::_Flat, rng::AbstractRNG, d::Parallel)
    primary = _is_censored_composer(d.components) ?
              _shared_primary_event(d.components) : nothing
    primary === nothing && return _composite_rand(
        rng, d.components, float(eltype(d)))
    cores = map(_marginal_core, d.components)
    T = float(promote_type(_param_eltype(primary),
        map(_param_eltype, cores)...))
    out = Vector{T}(undef, length(cores) + 1)
    o = convert(T, rand(rng, primary))
    out[1] = o
    @inbounds for i in eachindex(cores)
        out[i + 1] = o + rand(rng, cores[i])
    end
    # Discretise each branch endpoint to its own secondary interval; the shared
    # origin slot stays continuous (the latent primary).
    return _discretise_flat_par(out, d.components)
end

# Apply each branch's secondary interval to the recorded endpoints AFTER the
# continuous draw. Slot `i + 1` is branch `i`'s endpoint; the shared origin slot
# is floored to the shared origin edge's interval (unbiased first-level gaps).
function _discretise_flat_par(out, components::Tuple)
    res = copy(out)
    res[1] = _apply_leaf_interval(out[1], _shared_origin_interval(components))
    @inbounds for i in eachindex(components)
        res[i + 1] = _apply_leaf_interval(out[i + 1], _leaf_interval(
            components[i]))
    end
    return res
end

# ---------------------------------------------------------------------------
# Nested/Competing-aware full-path simulation: a NAMED event record
# ---------------------------------------------------------------------------
#
# For an irregular tree (e.g. bdbv `onset -> {admit -> Competing(death,
# discharge), notif}`) the flat vector shapes above cannot express the
# structure: branches share one origin, a `Competing` resolves to ONE outcome,
# and the slots are named. `_tree_event_record` walks the tree sharing a single
# latent origin draw, samples each `Competing` outcome via its branch
# probabilities, and returns a `NamedTuple` keyed by `_flat_event_names` — the
# root origin then one slot per leaf event in depth-first order. An UNSAMPLED
# Competing outcome is `missing`, so the record feeds straight back into the
# censored composer `logpdf` (one observed outcome slot, the rest missing).

# Draw the full named event record of a nested tree: one shared origin draw at
# the root, then the depth-first leaf-event slots filled by the tree walk.
# Fill the typed event vector of a nested tree: one shared origin draw at the
# root, then the depth-first leaf-event slots from the tree walk. The vector is
# `Vector{Union{Missing, T}}` with a concrete sampled-time type `T` (an unsampled
# Competing outcome stays `missing`), so the walk is type-stable.
function _tree_event_vector(rng::AbstractRNG, d::Union{Sequential, Parallel})
    primary = _tree_primary_event(d)
    T = _tree_rand_type(d, primary)
    out = Vector{Union{Missing, T}}(missing, _event_nleaves(d.components) + 1)
    origin = convert(T, rand(rng, primary))
    out[1] = origin
    _tree_rand!(out, rng, d, origin, 2, T)
    # Discretise the RECORDED event times to each leaf's secondary interval as a
    # SEPARATE pass: the continuous walk above leaves every chaining anchor with
    # its sub-day jitter (a later step hangs off the continuous terminal), and
    # flooring an anchor mid-walk would re-introduce the recovery bias. The pass
    # mirrors the same depth-first slot layout. The origin slot is floored to the
    # origin edge's interval so first-level gaps are unbiased (a primary-only
    # origin edge keeps it continuous), then each leaf slot to its own interval.
    out[1] = _apply_leaf_interval(out[1], _origin_interval(d))
    _discretise_event_record!(out, d, 2)
    return out
end

# Draw the full NAMED event record of a nested tree: the typed event vector keyed
# by `_flat_event_names`, mirroring the same flat layout the scorer consumes.
function _tree_event_record(rng::AbstractRNG, d::Union{Sequential, Parallel})
    out = _tree_event_vector(rng, d)
    enames = _flat_event_names(d)
    return NamedTuple{enames}(Tuple(out))
end

# The latent origin distribution seeding the whole tree: a `Sequential`'s origin
# is its first step's primary (recursing into a nested first step); a `Parallel`'s
# is its branches' shared primary (recursing into a nested branch). `nothing` for
# a plain (uncensored) tree.
#
# The Parallel case must RECURSE into each branch rather than only checking each
# branch's `_origin_primary_event` (a leaf-only probe that returns `nothing` for a
# nested composer): a single-branch `compose((x = nested_censored,))` is a
# Parallel-of-one whose only branch is itself a censored composer, and the whole
# tree shares one latent origin found inside that branch. Probing leaf-only would
# miss it, fall through to the plain per-leaf-value `rand`, and error on the
# nested record (issue #436).
_tree_primary_event(d::Sequential) = _tree_primary_event(d.components[1])
_tree_primary_event(d::Parallel) = _shared_tree_primary_event(d.components)
_tree_primary_event(d::UnivariateDistribution) = _origin_primary_event(d)
# A `Competing` branch shares the parent origin and resolves to one outcome; its
# own origin primary (if censored) is any outcome's primary (they share it). A
# `Select` shares the origin of its alternatives. Both recurse so a nested
# censored child still surfaces the tree's shared latent origin.
_tree_primary_event(d::AbstractCompeting) = _shared_tree_primary_event(d.delays)
_tree_primary_event(d::Select) = _shared_tree_primary_event(d.alternatives)
_tree_primary_event(d::Latent) = _tree_primary_event(d.dist)

# The single shared primary event across a tuple of (possibly nested) branches,
# recursing into each via `_tree_primary_event`. Mirrors `_shared_primary_event`
# but descends into nested composer branches, so a Parallel-of-one over a nested
# censored child still finds the shared origin. Branches that disagree on the
# primary event are rejected (a shared origin must be unique).
#
# Implemented as a HEAD/TAIL tuple recursion (not a `primary = nothing`
# accumulator loop): the loop form leaves `primary::Union{Nothing, P}`, which
# older compilers (CI's `lts`/`1`) fail to constant-fold, making
# `_tree_primary_event` type-unstable and breaking `@inferred` on the sampling
# walk. The recursion dispatches `_combine_primary` on whether each side is a
# concrete primary or `nothing`, so the compiler resolves the return type per
# step (the same pattern `_tree_core_eltype` uses).
_shared_tree_primary_event(::Tuple{}) = nothing
function _shared_tree_primary_event(components::Tuple)
    return _combine_primary(_tree_primary_event(first(components)),
        _shared_tree_primary_event(Base.tail(components)))
end

# Combine a branch's primary with the rest's shared primary. A `nothing` side
# yields the other; two concrete primaries must agree (a shared origin is unique).
_combine_primary(a::Nothing, b) = b
_combine_primary(a, b::Nothing) = a
_combine_primary(a::Nothing, ::Nothing) = nothing
function _combine_primary(a, b)
    a == b || throw(ArgumentError(
        "Parallel shared-origin branches must share one primary " *
        "event; got $(a) and $(b)"))
    return a
end

# The sampled event-time element type: promote the primary and every leaf delay
# core's parameter type so an AD/tracked param flows through (matching the
# scoring path's type handling).
function _tree_rand_type(d, primary)
    return float(promote_type(_param_eltype(primary), _tree_core_eltype(d)))
end

# Promote a child's leaf-delay parameter types. `map` + splatting `promote_type`
# over the fixed-length component tuple (the same shape the flat `_composer_rand`
# uses) keeps the type computation inferable, where a `mapreduce` with an `init`
# does not constant-fold here.
_tree_core_eltype(d::AbstractCompeting) = promote_type(map(_param_eltype, d.delays)...)
_tree_core_eltype(d::UnivariateDistribution) = _param_eltype(_marginal_core(d))
function _tree_core_eltype(d::Union{Sequential, Parallel})
    return promote_type(map(_tree_core_eltype, d.components)...)
end
# A nested `Select` promotes over EVERY alternative's core param type, so the
# sampled-time element type covers whichever alternative routes for a record (the
# default-alternative `rand` still falls within this promoted type).
_tree_core_eltype(d::Select) = promote_type(map(_tree_core_eltype,
    d.alternatives)...)

# Fill the event slots of composer `d` hanging off the absolute `origin` time,
# starting at flat index `event_start`. Returns the next free index. A
# `Sequential` threads the terminal time step to step; a `Parallel` hangs every
# branch off the shared origin. The walk over children is a runtime loop over the
# component tuple, mirroring the scoring recursion.
function _tree_rand!(out, rng::AbstractRNG, d::Sequential, origin, event_start,
        ::Type{T}) where {T}
    idx = event_start
    o = origin
    @inbounds for step in d.components
        idx, term = _tree_rand_step!(out, rng, step, o, idx, T)
        o = term
    end
    return idx
end

function _tree_rand!(out, rng::AbstractRNG, d::Parallel, origin, event_start,
        ::Type{T}) where {T}
    idx = event_start
    @inbounds for branch in d.components
        idx, _ = _tree_rand_step!(out, rng, branch, origin, idx, T)
    end
    return idx
end

# Sample one step/branch hanging off the absolute `origin` time, filling slots
# from `idx`. Returns `(next_idx, terminal_time)`: the terminal is what a
# following chain step hangs off (its own event for a leaf, the last step for a
# Sequential, the shared origin for a Parallel/Competing, mirroring
# `_terminal_offset`).
function _tree_rand_step!(out, rng::AbstractRNG,
        step::Union{Sequential, Parallel}, origin, idx, ::Type{T}) where {T}
    next = _tree_rand!(out, rng, step, origin, idx, T)
    return next, _tree_subtree_terminal(out, step, origin, idx)
end

function _tree_rand_step!(out, rng::AbstractRNG, step::Competing, origin, idx,
        ::Type{T}) where {T}
    # Draw the resolved outcome from the branch probabilities, fill only its
    # outcome's slice; the other outcomes' slices stay missing so the record
    # scores as the conditioned (one-outcome) Competing. A no-event win leaves
    # EVERY slot missing (no time recorded), so the record scores as a latent
    # non-occurrence. A NON-TERMINAL (composer) outcome recurses into its subtree,
    # which shares the parent `origin` as its anchor (#466 Feature 3).
    i = _sample_branch(rng, step.branch_probs)
    start = idx + _competing_outcome_start(step.delays, i)
    if !_is_no_event(step.delays[i])
        _competing_outcome_rand!(out, rng, step.delays[i], origin, start, T)
    end
    return idx + _event_child_nleaves(step), origin
end

# Sample the chosen competing outcome's payload into `out` from index `start`,
# hanging off the parent `origin`. A LEAF outcome fills one slot with
# `origin + delay`; a COMPOSER outcome walks its subtree (sharing `origin` as the
# subtree origin) so the subtree's leaf-event slots are filled.
function _competing_outcome_rand!(out, rng::AbstractRNG,
        delay::UnivariateDistribution, origin, start::Int, ::Type{T}) where {T}
    out[start] = origin + convert(T, rand(rng, _marginal_core(delay)))
    return nothing
end

function _competing_outcome_rand!(out, rng::AbstractRNG,
        delay::Union{Sequential, Parallel}, origin, start::Int,
        ::Type{T}) where {T}
    _tree_rand!(out, rng, delay, origin, start, T)
    return nothing
end

function _competing_outcome_rand!(out, rng::AbstractRNG, delay::Select, origin,
        start::Int, ::Type{T}) where {T}
    return _competing_outcome_rand!(out, rng, first(delay.alternatives), origin,
        start, T)
end

# A nested `Competing` / `HazardCompeting` outcome recurses through its own
# `_tree_rand_step!`, anchored at the parent origin.
function _competing_outcome_rand!(out, rng::AbstractRNG,
        delay::AbstractCompeting, origin, start::Int, ::Type{T}) where {T}
    _tree_rand_step!(out, rng, delay, origin, start, T)
    return nothing
end

# Racing-hazard `HazardCompeting`: draw a latent racing time per cause, fill the
# argmin-cause's slice; the others stay missing so the record scores as the
# cause-resolved sub-density. A LEAF cause's racing time is a draw from its delay
# (filling its one slot at the min time); a NON-TERMINAL (composer) cause's racing
# time is its subtree's marginal time-to-resolution, and on a win its subtree is
# walked (#466 Feature 3). The terminal for a following step is the shared origin.
function _tree_rand_step!(out, rng::AbstractRNG, step::HazardCompeting, origin,
        idx, ::Type{T}) where {T}
    n = _n_branches(step)
    best_i = 1
    best_t = _hazard_outcome_racing_time(rng, step.delays[1], T)
    @inbounds for k in 2:n
        t = _hazard_outcome_racing_time(rng, step.delays[k], T)
        if t < best_t
            best_t = t
            best_i = k
        end
    end
    start = idx + _competing_outcome_start(step.delays, best_i)
    _hazard_outcome_rand!(out, rng, step.delays[best_i], origin, best_t, start, T)
    return idx + _event_child_nleaves(step), origin
end

# The racing time of one outcome (for the argmin draw): a leaf delay draws its own
# time; a composer subtree draws its marginal time-to-resolution (the gap from a
# zero origin to the subtree's terminal event), so its first-resolution time races
# the leaf causes. The draw is from a fresh zero-origin walk so the racing time is
# independent of where the cause finally lands.
function _hazard_outcome_racing_time(rng, delay::UnivariateDistribution, ::Type{T}
) where {T}
    convert(T, rand(rng, _marginal_core(delay)))
end
function _hazard_outcome_racing_time(rng, delay::Union{Sequential, Parallel},
        ::Type{T}) where {T}
    z = zero(T)
    nslots = _event_nleaves(delay.components)
    tmp = Vector{Union{Missing, T}}(missing, nslots + 1)
    tmp[1] = z
    _tree_rand!(tmp, rng, delay, z, 2, T)
    return _subtree_resolution_time(delay, tmp, T)
end

# The marginal resolution time of a zero-origin subtree walk: a `Sequential`
# resolves at its last leaf event (its single terminal); a `Parallel` resolves
# when its LAST branch endpoint fires (the latest of its filled slots), so it
# races by the time the whole fan-out has resolved.
function _subtree_resolution_time(d::Sequential, tmp, ::Type{T}) where {T}
    term = tmp[2 + _terminal_offset(d)]
    return term === missing ? convert(T, Inf) : convert(T, term)
end
function _subtree_resolution_time(d::Parallel, tmp, ::Type{T}) where {T}
    best = convert(T, -Inf)
    @inbounds for k in 2:length(tmp)
        v = tmp[k]
        v === missing && continue
        v > best && (best = convert(T, v))
    end
    return best == convert(T, -Inf) ? convert(T, Inf) : best
end

# Fill the winning racing outcome's slice. A LEAF cause fills its single slot at
# the (absolute) winning time; a COMPOSER cause walks its subtree off the parent
# origin (its leaf events fill the slice; the winning racing time is implicit in
# the drawn sub-times).
function _hazard_outcome_rand!(out, rng::AbstractRNG,
        delay::UnivariateDistribution, origin, best_t, start::Int,
        ::Type{T}) where {T}
    out[start] = origin + best_t
    return nothing
end
function _hazard_outcome_rand!(out, rng::AbstractRNG,
        delay::Union{Sequential, Parallel}, origin, best_t, start::Int,
        ::Type{T}) where {T}
    _tree_rand!(out, rng, delay, origin, start, T)
    return nothing
end

# Sample an outcome index from the branch probabilities by inverse-CDF (a single
# uniform draw), so no `Categorical` dependency is pulled in. The last outcome
# absorbs any rounding so an index is always returned.
function _sample_branch(rng::AbstractRNG, probs)
    u = rand(rng)
    c = zero(float(eltype(probs)))
    @inbounds for i in 1:(length(probs) - 1)
        c += probs[i]
        u <= c && return i
    end
    return length(probs)
end

function _tree_rand_step!(out, rng::AbstractRNG, step::UnivariateDistribution,
        origin, idx, ::Type{T}) where {T}
    y = origin + convert(T, rand(rng, _marginal_core(step)))
    out[idx] = y
    return idx + 1, y
end

# A nested `Select` samples its DEFAULT (first) alternative on the data-free
# simulation path, matching the deterministic default `_tree_step(::Select)`
# scores, so a sampled record round-trips through `logpdf`. The data path resolves
# the Select before sampling, so a routed record samples its chosen alternative.
function _tree_rand_step!(out, rng::AbstractRNG, step::Select, origin, idx,
        ::Type{T}) where {T}
    return _tree_rand_step!(out, rng, first(step.alternatives), origin, idx, T)
end

# The terminal event time of a nested subtree just filled into `out` from
# `idx`: the shared origin for a Parallel (every branch hangs off it), the last
# leaf event for a Sequential. `_terminal_offset` gives that slot relative to the
# subtree's first event index (the same offset the scorer uses).
_tree_subtree_terminal(out, ::Parallel, origin, idx) = origin
function _tree_subtree_terminal(out, d::Sequential, origin, idx)
    return out[idx + _terminal_offset(d)]
end

# --- Post-walk discretisation of a tree event record --------------------------
#
# After the continuous tree walk fills `out`, apply each LEAF's secondary
# interval to its own recorded slot in a SEPARATE depth-first pass mirroring the
# walk's slot layout. Run after the whole walk so every chaining anchor keeps its
# sub-day jitter; flooring an anchor mid-walk would re-introduce the bias. The
# origin slot is never touched (the latent primary stays continuous). Returns the
# next free index, like `_tree_rand!`.
function _discretise_event_record!(out, d::Sequential, event_start::Int)
    idx = event_start
    @inbounds for step in d.components
        idx = _discretise_step!(out, step, idx)
    end
    return idx
end

function _discretise_event_record!(out, d::Parallel, event_start::Int)
    idx = event_start
    @inbounds for branch in d.components
        idx = _discretise_step!(out, branch, idx)
    end
    return idx
end

# Discretise one step/branch's slots, advancing the cursor by its event-slot
# count (a Competing spans one slot per outcome, #333).
function _discretise_step!(out, step::Union{Sequential, Parallel}, idx::Int)
    return _discretise_event_record!(out, step, idx)
end

function _discretise_step!(out, step::AbstractCompeting, idx::Int)
    n = _n_branches(step)
    off = 0
    @inbounds for k in 1:n
        delay = step.delays[k]
        w = _competing_outcome_slots(delay)
        _discretise_competing_outcome!(out, delay, idx + off, w)
        off += w
    end
    return idx + off
end

# Discretise ONE competing outcome's slice. A LEAF outcome floors its single slot
# to the leaf's interval; a COMPOSER outcome discretises its subtree's slots
# through the subtree's own depth-first pass (#466 Feature 3).
function _discretise_competing_outcome!(out, delay::UnivariateDistribution,
        start::Int, ::Int)
    out[start] === missing ||
        (out[start] = _apply_leaf_interval(out[start], _leaf_interval(delay)))
    return nothing
end
function _discretise_competing_outcome!(out, delay::Union{Sequential, Parallel},
        start::Int, ::Int)
    _discretise_event_record!(out, delay, start)
    return nothing
end
function _discretise_competing_outcome!(out, delay::Select, start::Int, w::Int)
    return _discretise_competing_outcome!(out, first(delay.alternatives), start,
        w)
end
function _discretise_competing_outcome!(out, delay::AbstractCompeting,
        start::Int, ::Int)
    _discretise_step!(out, delay, start)
    return nothing
end

function _discretise_step!(out, step::UnivariateDistribution, idx::Int)
    out[idx] === missing ||
        (out[idx] = _apply_leaf_interval(out[idx], _leaf_interval(step)))
    return idx + 1
end

# A nested `Select` discretises its DEFAULT alternative's slot(s), matching the
# default the simulation walk filled.
function _discretise_step!(out, step::Select, idx::Int)
    return _discretise_step!(out, first(step.alternatives), idx)
end
