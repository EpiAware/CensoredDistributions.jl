# ============================================================================
# Censored composers: full-path simulation (`rand`) + discretisation
# ============================================================================
#
# Split out of `censored_specialisations.jl`. The censored composer `rand`
# returns the COMPLETE event-time path (flat vector and the nested/Competing-
# aware NAMED record), plus the post-walk interval discretisation.

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
    return _competing_outcome_rand!(out, rng, _flat_select_alternative(delay),
        origin, start, T)
end

# A nested `Competing` / `HazardCompeting` outcome recurses through its own
# `_tree_rand_step!`, anchored at the parent origin.
function _competing_outcome_rand!(out, rng::AbstractRNG,
        delay::AbstractCompeting, origin, start::Int, ::Type{T}) where {T}
    _tree_rand_step!(out, rng, delay, origin, start, T)
    return nothing
end

# Racing-hazard `HazardCompeting`: draw a latent racing realisation per cause, fill
# the argmin-cause's slice; the others stay missing so the record scores as the
# cause-resolved sub-density. A LEAF cause's racing time is a draw from its delay
# (filling its one slot at the min time); a NON-TERMINAL (composer) cause's racing
# realisation is a full zero-origin subtree walk whose marginal time-to-resolution
# races the leaf causes. On a COMPOSER-cause win the SAME realisation that produced
# the winning racing time is recorded (re-anchored on the real origin), so the
# recorded subtree IS the race-winning draw rather than a second independent walk
# (#466 Feature 3; fixes the rand/likelihood mismatch where the scorer reads the
# recorded resolution time). The terminal for a following step is the shared origin.
function _tree_rand_step!(out, rng::AbstractRNG, step::HazardCompeting, origin,
        idx, ::Type{T}) where {T}
    n = _n_branches(step)
    best_i = 1
    best_t, best_real = _hazard_outcome_racing_draw(rng, step.delays[1], T)
    @inbounds for k in 2:n
        t, real = _hazard_outcome_racing_draw(rng, step.delays[k], T)
        if t < best_t
            best_t = t
            best_i = k
            best_real = real
        end
    end
    start = idx + _competing_outcome_start(step.delays, best_i)
    _hazard_outcome_rand!(out, rng, step.delays[best_i], origin, best_t,
        best_real, start, T)
    return idx + _event_child_nleaves(step), origin
end

# The racing draw of one outcome (for the argmin): the racing TIME plus the
# REALISATION that produced it, so the winner's recorded subtree IS the race-
# winning draw. A leaf delay draws its own time; its "realisation" is the time
# itself (a leaf re-derives `origin + best_t` and ignores the carried realisation).
# A composer subtree draws a full zero-origin walk into `tmp`, returning its
# marginal time-to-resolution AND the realised `tmp` slots, so on a win the carried
# `tmp` is re-anchored on the real origin instead of drawing a fresh independent
# subtree. The draw is a fresh zero-origin walk, racing the leaf causes.
function _hazard_outcome_racing_draw(rng, delay::UnivariateDistribution,
        ::Type{T}) where {T}
    t = convert(T, rand(rng, _marginal_core(delay)))
    return t, t
end
function _hazard_outcome_racing_draw(rng, delay::Union{Sequential, Parallel},
        ::Type{T}) where {T}
    z = zero(T)
    nslots = _event_nleaves(delay.components)
    tmp = Vector{Union{Missing, T}}(missing, nslots + 1)
    tmp[1] = z
    _tree_rand!(tmp, rng, delay, z, 2, T)
    return _subtree_resolution_time(delay, tmp, T), tmp
end

# The racing TIME of one outcome alone (the argmin key), dropping the realisation
# `_hazard_outcome_racing_draw` carries for the recorded-subtree fidelity. Kept as
# a thin wrapper for the MC racing-frequency checks (which only need the time).
function _hazard_outcome_racing_time(rng, delay, ::Type{T}) where {T}
    return first(_hazard_outcome_racing_draw(rng, delay, T))
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

# Fill the winning racing outcome's slice from the realisation that WON the race.
# A LEAF cause fills its single slot at the (absolute) winning time (it ignores the
# carried realisation, which for a leaf is just the time). A COMPOSER cause copies
# the realised zero-origin subtree `real` (the SAME walk whose resolution time won
# the race) into the slice, re-anchored on the real `origin`, so the recorded
# subtree IS the race-winning draw and its resolution time matches `best_t` — the
# density the scorer reads off the recorded slot. (Previously a second independent
# `_tree_rand!` here drew an unrelated subtree, so samples did not follow the
# assigned likelihood.) `rng` is unused on this path now (no fresh draw).
function _hazard_outcome_rand!(out, rng::AbstractRNG,
        delay::UnivariateDistribution, origin, best_t, real, start::Int,
        ::Type{T}) where {T}
    out[start] = origin + best_t
    return nothing
end
function _hazard_outcome_rand!(out, rng::AbstractRNG,
        delay::Union{Sequential, Parallel}, origin, best_t, real, start::Int,
        ::Type{T}) where {T}
    _reanchor_subtree!(out, real, origin, start, T)
    return nothing
end

# Copy a zero-origin realised subtree `real` (slot 1 is the zero origin, slots
# 2..end its depth-first leaf events) into `out` from index `start`, re-anchored on
# the real `origin`: each filled leaf slot becomes `origin + (zero-origin time)`.
# A missing slot (an unfired branch in a partially resolving subtree) stays missing,
# preserving the realised missingness pattern the scorer conditions on.
function _reanchor_subtree!(out, real, origin, start::Int, ::Type{T}) where {T}
    @inbounds for k in 2:length(real)
        v = real[k]
        out[start + (k - 2)] = v === missing ? missing : origin + convert(T, v)
    end
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
    return _tree_rand_step!(out, rng, _flat_select_alternative(step), origin,
        idx, T)
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
    return _discretise_competing_outcome!(out, _flat_select_alternative(delay),
        start, w)
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
    return _discretise_step!(out, _flat_select_alternative(step), idx)
end
