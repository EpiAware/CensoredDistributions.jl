# ============================================================================
# Censored composers: one_of-outcome slice scoring + per-record rebuilds
# ============================================================================
#
# Split out of `censored_specialisations.jl`. The one_of outcome-slice
# layout, the nested-`Resolve`/`Compete` tree scoring, the per-record
# branch-probability override, and the per-record nested-`Choose` routing.
# Builds on the generic nested-tree scoring in `censored_scoring_tree.jl`.

# --- one_of outcome-slice layout (leaf OR composer subtree) ---------------
#
# Each one_of outcome occupies a CONTIGUOUS slice of the event vector starting
# at `ev_idx`: a LEAF outcome is one slot, a NON-TERMINAL outcome whose payload is
# a composer subtree is its whole subtree's event-slot width (`_event_child_
# nleaves`, matching `_walk_edge!`'s emitted names and `_event_child_nleaves`'s
# count). `_one_of_outcome_start(delays, k)` is the 1-based offset of outcome
# `k`'s slice within the node's outcome region (outcome 1 at 0). Pure-Int
# bookkeeping over the constant tree structure, so it stays off the AD'd values.
function _one_of_outcome_start(delays::Tuple, k::Int)
    off = 0
    @inbounds for j in 1:(k - 1)
        off += _one_of_outcome_slots(delays[j])
    end
    return off
end

# Whether ANY slot of outcome `k`'s slice (width `w`, beginning at absolute index
# `start`) is observed (non-missing). A composer outcome is "observed" when any of
# its subtree slots is present (e.g. a `death => chain` record fills one of the
# chain's leaf events); a leaf outcome is observed when its single slot is.
function _one_of_outcome_observed(events, start::Int, w::Int)
    @inbounds for s in 0:(w - 1)
        events[start + s] === missing || return true
    end
    return false
end

# Resolve WHICH one_of outcome a record observes by scanning each outcome's
# slice for a present slot, erroring if two distinct outcomes are both observed (a
# record resolves to exactly one outcome). Returns `(obs_i, obs_start, obs_w)`
# with `obs_i == 0` when no outcome is observed. The anchor is the parent event
# `events[o_idx]` (a one_of outcome's slice hangs off it). Pure control flow
# over the constant event vector.
function _resolve_one_of_outcome(delays::Tuple, names::Tuple, events,
        ev_idx::Int)
    n = length(delays)
    obs_i = 0
    obs_start = 0
    obs_w = 0
    off = 0
    @inbounds for k in 1:n
        w = _one_of_outcome_slots(delays[k])
        start = ev_idx + off
        if _one_of_outcome_observed(events, start, w)
            obs_i == 0 || throw(ArgumentError(
                "a nested Resolve record may observe at most one outcome; " *
                "got outcomes $(names[obs_i]) and $(names[k])"))
            obs_i = k
            obs_start = start
            obs_w = w
        end
        off += w
    end
    return obs_i, obs_start, obs_w
end

# A nested `Resolve` node SELF-DISPATCHES on which OUTCOME a record observes,
# mirroring the top-level `Resolve` self-dispatch but anchored at the parent
# event. Its outcome slices begin at `ev_idx`; the anchor (the parent origin) is
# `events[o_idx]`. The numeric event-vector path uses the node's STORED branch
# probabilities (the per-record `branch_probs` override is a ROW input, applied by
# the DynamicPPL extension which re-anchors a nested Resolve with row context).
# Exactly one outcome observed -> score that branch's mixture weight `log p_k`
# plus its payload term (a leaf conditions on its delay at the gap; a composer
# subtree recurses through `_tree_score` on the outcome's slice, anchored at the
# shared parent origin); no outcome observed -> contributes no factor (the
# resolved-but-unknown-outcome encoding for a NESTED Resolve is deferred).
#
# A per-record observation `horizon` (default `nothing`) RIGHT-TRUNCATES the
# conditioned branch at the remaining window from the anchor, mirroring the
# top-level `_maybe_truncate` so a nested Resolve honours the same real-time
# right-truncation as the top-level node. With `horizon === nothing` the
# scoring is byte-identical to the untruncated path.
function _tree_step(step::Resolve, events, o_idx::Int, ev_idx::Int,
        primary, ::Type{T}, horizon = nothing) where {T}
    return _one_of_tree_logpdf(step, step.branch_probs, o_idx,
        events, ev_idx, primary, T, horizon)
end

# Score a nested `Resolve` against its outcome slices given the anchor INDEX
# `o_idx` and branch probabilities `probs` (stored or row-overridden). Pure
# Turing-free arithmetic shared by the numeric path and the DynamicPPL extension.
# A no-event branch's slot is a PRESENCE marker (its value is not a time): a
# non-missing no-event slot is an OBSERVED non-occurrence scoring `log q` (no
# delay term), a missing no-event slot is a latent non-occurrence contributing
# nothing. An observed real LEAF outcome conditions on that branch; a COMPOSER
# outcome scores `log p_k + _tree_score(subtree, slice)` (the mixture weight plus
# its subtree's own censored event-vector density anchored at the parent origin).
#
# A per-record `horizon` (default `nothing`) right-truncates the branch density at
# the remaining window from the anchor (`horizon - events[o_idx]`), matching the
# top-level `_maybe_truncate` semantics.
function _one_of_tree_logpdf(c::Resolve, probs, o_idx::Int, events,
        ev_idx::Int, primary, ::Type{T}, horizon = nothing) where {T}
    obs_i,
    obs_start,
    obs_w = _resolve_one_of_outcome(c.delays, c.names, events, ev_idx)
    obs_i == 0 && return zero(T)
    delay = c.delays[obs_i]
    # An OBSERVED non-occurrence scores the no-event mass `log q` alone.
    _is_no_event(delay) && return log(probs[obs_i])
    o = events[o_idx]
    return log(probs[obs_i]) +
           _one_of_outcome_payload_logpdf(delay, o, o_idx, events,
        obs_start, obs_w, primary, T, horizon)
end

# The remaining observation window for a nested Resolve outcome anchored at `o`
# (the parent origin): `horizon - o`, the time left to observe the branch from its
# anchor. The top-level Resolve node is anchored at origin 0, so its window IS
# the horizon; a nested node hangs off a non-zero anchor, so the horizon is
# shifted by the anchor exactly as the flat `Parallel`/`Sequential` horizon paths
# compute `horizon - origin`. `nothing` horizon -> `nothing` window (no
# truncation). The anchor is already validated non-missing by the caller.
_one_of_window(::Nothing, o) = nothing
function _one_of_window(horizon, o)
    t = _horizon_time(horizon)
    return t - convert(typeof(t), o)
end

# Right-truncate a branch `delay` at the per-record window, or return it unchanged
# when no horizon applies, mirroring the top-level `_maybe_truncate`. The threaded
# `horizon` carrier rides along so a δ-bounded horizon δ-bounds the branch; a plain
# horizon (or `nothing` δ) is the upper-only `truncated(delay; upper = window)`.
_one_of_truncate(delay, ::Nothing, horizon) = delay
function _one_of_truncate(delay, window, horizon)
    return _truncate_horizon(delay, window, horizon)
end

# The payload term of an observed one_of outcome: a LEAF conditions on its
# (declared-censored) delay at the gap from the anchor; a COMPOSER subtree scores
# its own censored event-vector density on the outcome's slice, anchored at the
# parent origin (shared like a nested-composer origin). The mixture weight
# `log p_k` is added by the caller, so this is the conditional `f(payload | k)`.
# A per-record `horizon` (default `nothing`) right-truncates a LEAF branch at the
# remaining window from the anchor; the composer-subtree / nested-Resolve
# payloads thread it on into their own recursion.
function _one_of_outcome_payload_logpdf(delay::UnivariateDistribution, o,
        o_idx::Int, events, obs_start::Int, obs_w::Int, primary, ::Type{T},
        horizon = nothing) where {T}
    y = events[obs_start]
    o === missing && throw(ArgumentError(
        "a nested Resolve with an observed outcome needs an observed anchor " *
        "(its parent event); got a missing anchor"))
    branch = _one_of_truncate(delay, _one_of_window(horizon, o), horizon)
    return logpdf(branch, convert(T, y) - convert(T, o))
end

# A COMPOSER outcome: recurse through `_tree_score` on the `[origin, slice...]`
# sub-event vector, the outcome's resolution sharing the parent origin (the
# subtree origin slot). The subtree's leaf events occupy `obs_w` slots from
# `obs_start`; the parent origin anchors them. The subtree may itself be censored,
# so its own origin primary (if any) seeds the recursion; the parent-level primary
# is passed through for a sampled-origin sub-chain. A per-record `horizon` threads
# on into the subtree recursion so a Resolve nested inside this subtree also
# truncates; the subtree shares the parent anchor, so the same absolute
# horizon applies.
function _one_of_outcome_payload_logpdf(
        delay::Union{Sequential, Parallel}, o, o_idx::Int, events,
        obs_start::Int, obs_w::Int, primary, ::Type{T},
        horizon = nothing) where {T}
    sub = _subevent_slice(events, o_idx, obs_start, obs_w)
    sub_primary = _subtree_origin_primary(delay, primary)
    return _tree_score(delay, sub, 1, 2, sub_primary, T, horizon)
end

# A `Choose` outcome routes to its committed (first) alternative on the numeric
# path (the data path resolves Choose nodes out before scoring); score that.
function _one_of_outcome_payload_logpdf(delay::Choose, o, o_idx::Int, events,
        obs_start::Int, obs_w::Int, primary, ::Type{T},
        horizon = nothing) where {T}
    return _one_of_outcome_payload_logpdf(_flat_choose_alternative(delay), o,
        o_idx, events, obs_start, obs_w, primary, T, horizon)
end

# A nested `Resolve` outcome (a one_of node as a one_of branch): recurse
# through the one_of scorer on the outcome's slice, anchored at the parent
# origin (the inner one_of's outcomes hang off the same shared anchor). The
# per-record `horizon` threads on so the inner Resolve truncates too.
function _one_of_outcome_payload_logpdf(delay::Resolve, o, o_idx::Int,
        events, obs_start::Int, obs_w::Int, primary, ::Type{T},
        horizon = nothing) where {T}
    sub = _subevent_slice(events, o_idx, obs_start, obs_w)
    return _one_of_tree_logpdf(delay, delay.branch_probs, 1, sub, 2,
        primary, T, horizon)
end

function _one_of_outcome_payload_logpdf(delay::Compete, o, o_idx::Int,
        events, obs_start::Int, obs_w::Int, primary, ::Type{T},
        horizon = nothing) where {T}
    sub = _subevent_slice(events, o_idx, obs_start, obs_w)
    return _hazard_one_of_tree_logpdf(delay, 1, sub, 2, primary, T, horizon)
end

# The origin primary event seeding a COMPOSER one_of outcome's subtree. A
# censored subtree surfaces its own shared latent origin (e.g. a chain whose first
# step is `primary_censored`); a plain subtree has none and inherits the parent's
# primary (a sampled-origin sub-chain off the one_of anchor). Mirrors the
# `_tree_score` root seeding.
function _subtree_origin_primary(d::Sequential, parent_primary)
    p = _origin_primary_event(_first_origin_node(d))
    return p === nothing ? parent_primary : p
end
function _subtree_origin_primary(d::Parallel, parent_primary)
    p = _shared_primary_event(d.components)
    return p === nothing ? parent_primary : p
end

# A nested racing-hazard `Compete` node SELF-DISPATCHES on which OUTCOME a
# record observes, anchored at the parent event `events[o_idx]`. Exactly one
# outcome observed -> its payload term (a leaf scores the cause-resolved
# sub-density `log f_j(t) + Σ_{k≠j} log S_k(t)`; a composer subtree the racing
# survival PLUS the subtree's own event-vector density); no outcome observed ->
# no factor (a fully latent record). The winning probability is DERIVED, so there
# is no branch-probability term.
function _tree_step(step::Compete, events, o_idx::Int, ev_idx::Int,
        primary, ::Type{T}, horizon = nothing) where {T}
    return _hazard_one_of_tree_logpdf(step, o_idx, events, ev_idx, primary, T,
        horizon)
end

function _hazard_one_of_tree_logpdf(step::Compete, o_idx::Int, events,
        ev_idx::Int, primary, ::Type{T}, horizon = nothing) where {T}
    obs_i,
    obs_start,
    obs_w = _resolve_one_of_outcome(step.delays, step.names, events, ev_idx)
    obs_i == 0 && return zero(T)
    o = events[o_idx]
    return _hazard_outcome_payload_logpdf(step, obs_i, o, o_idx, events,
        obs_start, obs_w, primary, T, horizon)
end

# A LEAF racing outcome: the cause-resolved sub-density at the observed gap. The
# log density carries any AD `Dual`/tracked type from the racing delays' params,
# so it is NOT narrowed to the data type `T` (only the `obs_gap`, data, is `T`).
function _hazard_outcome_payload_logpdf(step::Compete, obs_i::Int, o,
        o_idx::Int, events, obs_start::Int, obs_w::Int, primary, ::Type{T},
        horizon = nothing) where {T}
    return _hazard_outcome_payload(step, obs_i, step.delays[obs_i], o, o_idx,
        events, obs_start, obs_w, primary, T, horizon)
end

function _hazard_outcome_payload(step::Compete, obs_i::Int,
        delay::UnivariateDistribution, o, o_idx::Int, events, obs_start::Int,
        obs_w::Int, primary, ::Type{T}, horizon = nothing) where {T}
    y = events[obs_start]
    o === missing && throw(ArgumentError(
        "a nested Compete with an observed outcome needs an observed " *
        "anchor (its parent event); got a missing anchor"))
    return _hazard_cause_logpdf(step, obs_i, convert(T, y) - convert(T, o))
end

# A COMPOSER racing outcome: the cause-resolved density of a
# NON-TERMINAL racing branch is its subtree's own censored event density TIMES the
# survival of the OTHER causes up to the branch's resolution time `t`, mirroring
# the leaf formula `f_j(t) ∏_{k≠j} S_k(t)`. The subtree replaces the within-branch
# density `f_j(t)`; the cross-cause survival `Σ_{k≠j} logccdf_k(t)` weights it by
# the racing causes that did NOT fire by `t`. The resolution time `t` is the gap
# from the parent origin to the subtree's terminal/latest event (matching
# `_subtree_resolution_time` on the rand path). AD-safe (the survival rides
# `_logccdf_ad_safe` in log space; only the `t` gap is narrowed to the data type).
# The anchor for the subtree is the parent origin (shared).
function _hazard_outcome_payload(step::Compete, obs_i::Int,
        delay::Union{Sequential, Parallel}, o, o_idx::Int, events,
        obs_start::Int, obs_w::Int, primary, ::Type{T},
        horizon = nothing) where {T}
    sub = _subevent_slice(events, o_idx, obs_start, obs_w)
    sub_primary = _subtree_origin_primary(delay, primary)
    subtree = _tree_score(delay, sub, 1, 2, sub_primary, T, horizon)
    t = _hazard_subtree_resolution_gap(delay, sub, T)
    return subtree + _hazard_cross_cause_logsurvival(step, obs_i, t)
end

# The cross-cause survival `Σ_{k≠j} logccdf_k(t)` of a racing node: the log
# survival of every cause OTHER than the winning `j` at the resolution time `t`.
# Each term rides `_logccdf_ad_safe` so the non-winning leaf params propagate
# their `Dual`/tracked type (no `float` strip). Unlike `_hazard_cause_logpdf`'s
# `≠ j` sum (all-leaf, where probing the skipped cause's survival is harmless),
# the WINNING cause `j` here may be a COMPOSER whose survival routes through the
# non-AD-safe `Convolved` CDF, so its term is NEVER evaluated: the winner is
# skipped in the loop rather than zeroed in place. The accumulator is seeded with
# `zero(t)` (no survival probe) and promoted by the `+=` of each non-winning term,
# so it never touches the winner's survival and an empty non-winning set (a
# degenerate single-cause node) still returns a typed zero.
function _hazard_cross_cause_logsurvival(step::Compete, j::Int, t)
    n = _n_branches(step)
    acc = zero(t)
    @inbounds for k in 1:n
        k == j && continue
        acc += _logccdf_ad_safe(step.delays[k], t)
    end
    return acc
end

# --- standalone disjunction-node event-record scoring -----------------------
#
# A STANDALONE `Resolve` / `Compete` (one sampled on its own, not nested in a
# `compose(...)` tree) scores the SAME named event record its `rand` produces: a
# `NamedTuple` keyed by `_flat_event_names(c) = (:event_1, c.names...)`, the
# positional origin slot then one slot per outcome with the fired outcome's time
# present and the others `missing`. The record is matched BY NAME to the flat
# event vector, then scored through the SAME outcome-slice scorer the in-tree path
# uses (`_one_of_tree_logpdf` / `_hazard_one_of_tree_logpdf` with the origin at
# slot 1 and the outcomes from slot 2), so a standalone draw round-trips through
# `logpdf(c, rand(c))` identically to the in-tree draw of the same node. A
# standalone node has no upstream latent origin and no observation horizon, so
# `primary`/`horizon` are `nothing`. This is the disjunction sibling of the
# `logpdf(::Sequential/Parallel, ::NamedTuple)` record scorer.

# Score a standalone Resolve outcome record (a single labelled draw). A column
# table (a `NamedTuple` of vectors) is a MULTI-record source, summed per row.
function logpdf(c::Resolve, x::NamedTuple)
    Tables.istable(x) && return _one_of_table_logpdf(c, x)
    _is_nonterminal(c) && _nonterminal_marginal_error("logpdf")
    events = _row_event_vector_by_name(_flat_event_names(c), x)
    return _one_of_tree_logpdf(c, c.branch_probs, 1, events, 2, nothing,
        Float64)
end

# Score a standalone Compete outcome record (the winning probability is DERIVED
# from the hazards, so there is no branch-probability term).
function logpdf(c::Compete, x::NamedTuple)
    Tables.istable(x) && return _one_of_table_logpdf(c, x)
    _is_nonterminal(c) && _nonterminal_marginal_error("logpdf")
    events = _row_event_vector_by_name(_flat_event_names(c), x)
    return _hazard_one_of_tree_logpdf(c, 1, events, 2, nothing, Float64)
end

# Score a TABLE / vector of standalone disjunction-node records: the SUM of each
# record's single-record log density. A standalone one_of node carries no
# per-record covariate routing (no Choose selector, no horizon), so the table is
# scored by summing the per-row scorer directly rather than through the
# `record_distributions` per-record assembly the composer tables use.
function logpdf(c::AbstractOneOf, rows::AbstractVector{<:NamedTuple})
    return sum(logpdf(c, r) for r in rows)
end

function _one_of_table_logpdf(c::AbstractOneOf, table)
    return sum(logpdf(c, r) for r in Tables.namedtupleiterator(table))
end

# The log marginal SURVIVAL of a COMPOSER racing branch at `t`: the probability
# the branch has NOT yet resolved by `t`, matching the marginal resolution time
# `_hazard_outcome_racing_time` draws on the rand path. A `Sequential` resolves at
# the SUM of its components' resolution times, so its survival is that of the
# convolved marginal resolution-time distribution (each component reduced via
# `_branch_marginal`, convolved in series; `logccdf` gives `log P(T > t)`). A
# `Parallel` resolves when its LATEST endpoint fires, so its survival is
# `S_max(t) = 1 - ∏_i F_i(t)`, i.e. `log1mexp(Σ_i logcdf_i(t))`. Both let a
# composer branch race a leaf cause as a LOSER (its survival weights another
# cause's win) just as the leaf path's `_hazard_cause_logpdf` already queries each
# cause's survival; the marginal cores carry the leaf params for AD (the
# convolution CDF is analytic where the cores admit it, else AD-safe numeric).
function _logccdf_ad_safe(d::Sequential, t::Real)
    conv = convolved(map(_branch_marginal, d.components))
    return logccdf(conv, t)
end
function _logccdf_ad_safe(d::Parallel, t::Real)
    n = length(d.components)
    s = sum(ntuple(k -> logcdf(_branch_marginal(d.components[k]), t), n))
    return log1mexp(s)
end

# The marginal resolution-time delay of a racing branch's component, used to build
# the branch survival: a leaf strips to its `_marginal_core`; a nested `Sequential`
# reduces to the convolution of its components. A nested `Parallel`/`Resolve`
# sub-component has no closed marginal resolution time here, so it errors clearly
# rather than scoring a wrong survival.
_branch_marginal(d::UnivariateDistribution) = _marginal_core(d)
function _branch_marginal(d::Sequential)
    return convolved(map(_branch_marginal, d.components))
end
function _branch_marginal(d::Union{Parallel, AbstractOneOf})
    throw(ArgumentError(
        "a racing-branch survival convolution does not support a nested " *
        "$(nameof(typeof(d))) sub-component; only leaf delays and Sequential " *
        "sub-chains have a closed marginal resolution time here"))
end

# The resolution time of an observed composer outcome as the gap from the parent
# origin (`sub[1]`) to the subtree's terminal/latest event, mirroring
# `_subtree_resolution_time` on the rand path: a `Sequential` resolves at its
# terminal leaf (`_terminal_offset`), a `Parallel` when its latest filled slot
# fires. Pure data arithmetic narrowed to the event type `T`; a missing terminal
# slot (a partially observed branch) yields `Inf`, whose cross-cause survival is
# `-Inf` (an impossible record), keeping the score consistent.
function _hazard_subtree_resolution_gap(d::Sequential, sub, ::Type{T}) where {T}
    o = sub[1]
    term = sub[2 + _terminal_offset(d)]
    (o === missing || term === missing) && return convert(T, Inf)
    return convert(T, term) - convert(T, o)
end
function _hazard_subtree_resolution_gap(d::Parallel, sub, ::Type{T}) where {T}
    o = sub[1]
    o === missing && return convert(T, Inf)
    best = convert(T, -Inf)
    @inbounds for k in 2:length(sub)
        v = sub[k]
        v === missing && continue
        v > best && (best = convert(T, v))
    end
    best == convert(T, -Inf) && return convert(T, Inf)
    return best - convert(T, o)
end

# --- per-record branch-probability override for a nested Resolve -----
#
# A per-record `branch_probs` override is a ROW input (the covariate CFR
# `logistic(Xβ)` flows in per record), so it cannot ride the numeric event
# vector. Rather than thread it through the AD-sensitive tree recursion, the
# DynamicPPL extension REBUILDS the tree with the (single) Resolve node's
# probabilities replaced for the record, then scores the rebuilt tree through the
# normal numeric path (whose `_tree_step(::Resolve)` reads the now-overridden
# stored probs). The override's element type is preserved, so a `logistic(Xβ)`
# `Dual` flows through the rebuilt node and differentiates.

# Count the `Resolve` (mixture) nodes anywhere in a composed tree, so a single
# per-record `branch_probs` field is rejected as ambiguous when more than one node
# exists. The walk RECURSES through every place a node can nest: a composer's
# `components`, an `AbstractOneOf`'s outcome `delays` (a Resolve/Choose can
# legitimately nest inside a one_of outcome subtree, `_is_one_of_branch`),
# a `Choose`'s `alternatives`, and a `Latent`'s inner `dist`. A `Resolve` counts
# ONE and ALSO recurses into its own delays (a Resolve nested as another
# Resolve's outcome). A `Compete` is NOT branch-prob-overridable (its
# winning probabilities are derived), so it counts zero but still recurses into its
# delays. A plain leaf counts zero. (Previously the `::UnivariateDistribution`
# fallback swallowed an `AbstractOneOf` — `AbstractOneOf <:
# UnivariateDistribution` — silently UNDER-counting a nested Resolve and
# bypassing the `n == 1` ambiguity guard; `Choose`/`Latent` are multivariate, so
# they hit no method and errored. Both are now handled.)
_count_one_of(c::Resolve) = 1 + _count_one_of_in(c.delays)
_count_one_of(c::Compete) = _count_one_of_in(c.delays)
_count_one_of(::UnivariateDistribution) = 0
function _count_one_of(d::Union{Sequential, Parallel})
    return _count_one_of_in(d.components)
end
_count_one_of(d::Choose) = _count_one_of_in(d.alternatives)
_count_one_of(d::Latent) = _count_one_of(d.dist)

# Sum `_count_one_of` over a tuple of children (components / delays /
# alternatives). HEAD/TAIL recursion avoids the `Any`-inference widening a
# `sum`/`mapreduce` over a heterogeneous tuple hits on the CI compilers.
_count_one_of_in(::Tuple{}) = 0
function _count_one_of_in(xs::Tuple)
    return _count_one_of(first(xs)) + _count_one_of_in(Base.tail(xs))
end

# Rebuild a composed tree with the single `Resolve` node's branch probabilities
# replaced by `probs` (already coerced to outcome order and validated). Errors if
# the tree has no Resolve node or more than one (the single `branch_probs` row
# field is then ambiguous). Pure, Turing-free; the new probs keep their
# element type.
function _override_one_of_outcome_probs(d, probs)
    n = _count_one_of(d)
    n == 1 || throw(ArgumentError(
        "a per-record `branch_probs` override needs exactly one Resolve node " *
        "in the tree; found $n"))
    return _replace_one_of(d, probs)
end

# Rebuild every node, replacing the (single) `Resolve`'s probs. Recurses through
# the SAME nesting `_count_one_of` walks (composer components, AbstractOneOf
# delays, Choose alternatives, Latent inner dist), so a Resolve nested inside a
# one_of-outcome subtree or a Choose alternative is reached. A `Resolve` also
# rebuilds its own delays (a nested Resolve outcome is reached). A
# `Compete` is not overridable but rebuilds its delays (a Resolve may
# nest inside one of its causes). A `Choose` rebuilds its alternatives; a `Latent`
# its inner dist. The `n == 1` guard in `_override_one_of_outcome_probs` ensures
# exactly one `Resolve` exists, so `probs` is applied to that single node.
function _replace_one_of(c::Resolve, probs)
    Resolve(c.names,
        map(d -> _replace_one_of(d, probs), c.delays), probs)
end
function _replace_one_of(c::Compete, probs)
    return Compete(c.names,
        map(d -> _replace_one_of(d, probs), c.delays))
end
_replace_one_of(d::UnivariateDistribution, probs) = d
function _replace_one_of(d::Sequential, probs)
    return Sequential(map(c -> _replace_one_of(c, probs), d.components),
        d.names)
end
function _replace_one_of(d::Parallel, probs)
    return Parallel(map(c -> _replace_one_of(c, probs), d.components),
        d.names)
end
function _replace_one_of(d::Choose, probs)
    return Choose(
        d.names, map(a -> _replace_one_of(a, probs),
            d.alternatives), d.selector)
end
_replace_one_of(d::Latent, probs) = Latent(_replace_one_of(d.dist, probs))

# --- per-record nested-Choose routing ---------------------------------------
#
# A nested `Choose` routes per record by the row's selector field
# (`row[selector]`). On the DATA path the per-record build RESOLVES every nested
# Choose into the tree, replacing each with its routed alternative for the record,
# then scores the resolved (Choose-free) tree through the normal numeric path. The
# selector VALUE is data (a Symbol), so resolving it out of the tree before the
# differentiated scoring keeps the AD path free of the routing control flow and
# mirrors the per-record `branch_probs` rebuild for a nested Resolve. A row
# missing a needed selector field errors here, so a data record can never silently
# score alternative 1.

# Count the nested `Choose` nodes anywhere in a composed tree, so a tree with no
# Choose skips the resolution rebuild entirely. RECURSES through the same nesting
# `_count_one_of` walks: composer components, an `AbstractOneOf`'s outcome
# `delays` (a Choose can nest inside a one_of-outcome subtree), a Choose's own
# alternatives, and a Latent's inner dist. (Previously a nested `AbstractOneOf`
# hit the `::UnivariateDistribution` fallback — they share that supertype — so a
# Choose inside a one_of outcome was never counted and the resolution rebuild
# was skipped, leaving an unresolved Choose to silently score its first
# alternative; and a top-level `Latent` hit no method.)
_count_chooses(c::Choose) = 1 + _count_chooses_in(c.alternatives)
_count_chooses(::UnivariateDistribution) = 0
function _count_chooses(d::Union{Sequential, Parallel})
    return _count_chooses_in(d.components)
end
_count_chooses(c::AbstractOneOf) = _count_chooses_in(c.delays)
_count_chooses(d::Latent) = _count_chooses(d.dist)

_count_chooses_in(::Tuple{}) = 0
function _count_chooses_in(xs::Tuple)
    return _count_chooses(first(xs)) + _count_chooses_in(Base.tail(xs))
end

# Resolve every nested `Choose` in `d` to its routed alternative for `row`,
# returning the rebuilt (Choose-free) tree. A row missing a Choose's selector
# field errors clearly (no silent commit to alternative 1). A tree with no nested
# Choose is returned unchanged. The chosen alternative may itself nest a Choose,
# so the rebuild recurses into it. Recurses through the SAME nesting as
# `_count_chooses` (composer components, AbstractOneOf delays, Latent inner),
# so a Choose nested inside a one_of-outcome subtree is resolved out.
_pick_choose(d, row::NamedTuple) = _choose_alternative_node(d, row)

_choose_alternative_node(d::UnivariateDistribution, ::NamedTuple) = d
function _choose_alternative_node(d::Choose, row::NamedTuple)
    haskey(row, d.selector) || throw(ArgumentError(
        "a nested Choose needs its selector field $(repr(d.selector)) on the " *
        "record to route; the row has no such field, so it cannot pick an " *
        "alternative (it would otherwise silently score the first one)"))
    kind = row[d.selector]
    kind isa Symbol || throw(ArgumentError(
        "the nested Choose selector field $(repr(d.selector)) must hold a " *
        "Symbol naming the alternative; got $(typeof(kind))"))
    return _choose_alternative_node(_pick(d, kind), row)
end
function _choose_alternative_node(d::Sequential, row::NamedTuple)
    return Sequential(map(c -> _choose_alternative_node(c, row), d.components),
        d.names)
end
function _choose_alternative_node(d::Parallel, row::NamedTuple)
    return Parallel(map(c -> _choose_alternative_node(c, row), d.components),
        d.names)
end
function _choose_alternative_node(c::Resolve, row::NamedTuple)
    return Resolve(c.names,
        map(d -> _choose_alternative_node(d, row), c.delays), c.branch_probs)
end
function _choose_alternative_node(c::Compete, row::NamedTuple)
    return Compete(c.names,
        map(d -> _choose_alternative_node(d, row), c.delays))
end
function _choose_alternative_node(d::Latent, row::NamedTuple)
    return Latent(_choose_alternative_node(d.dist, row))
end
