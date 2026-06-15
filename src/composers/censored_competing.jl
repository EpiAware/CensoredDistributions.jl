# ============================================================================
# Censored composers: competing-outcome slice scoring + per-record rebuilds
# ============================================================================
#
# Split out of `censored_specialisations.jl`. The competing outcome-slice
# layout, the nested-`Competing`/`HazardCompeting` tree scoring, the per-record
# branch-probability override, and the per-record nested-`Select` routing.
# Builds on the generic nested-tree scoring in `censored_scoring_tree.jl`.

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
