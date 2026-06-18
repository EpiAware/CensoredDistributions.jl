# ============================================================================
# Tree EVENT names and by-name row -> event-vector mapping
# ============================================================================
#
# A composed distribution carries two distinct name spaces:
#
#   - EDGE names (`k` of them for a `k`-edge composer): the `names` field used by
#     `params_table` / `composed_parameters_model`. PARAMETERS belong to edges
#     (an edge IS a delay distribution with parameters), so edge names key the
#     parameter inventory.
#   - EVENT names (`k + 1` of them): the origin event plus one target event per
#     edge, in the SAME flat depth-first order as the scored event vector
#     `[E_0, E_1, ..., E_k]`. OBSERVATIONS are events (a linelist column is an
#     event time: onset, admit, death, ...), so event names key a data ROW.
#
# Both name spaces are retained; they describe different things. This file
# derives the EVENT-name layout from the existing EDGE names and maps a
# NamedTuple row to the flat event vector BY NAME, so a reordered row scores
# identically to an in-order one and a name the tree does not have errors
# clearly. Positional fallback (`:event_i`) applies only when an edge name is a
# positional default (`:step_i` / `:branch_i`), i.e. when the front-end was given
# no real names to derive events from.
#
# Turing-free and distributions-led: this reads structure + names only.

# --- edge-name -> (origin, target) event names ------------------------------

# Whether `s` is `prefix` immediately followed by one or more ASCII digits and
# nothing else (the positional-default shape `prefix_1`, `prefix_2`, ...). Plain
# string scan, NO `Regex`: a compiled `Regex` uses a try/catch that Mooncake
# reverse cannot differentiate, and this runs on the AD'd scoring path (it is
# reached from `_flat_event_names` inside the differentiated `logpdf`). See #409.
function _has_positional_suffix(s::AbstractString, prefix::AbstractString)
    startswith(s, prefix) || return false
    rest = SubString(s, ncodeunits(prefix) + 1)
    isempty(rest) && return false
    return all(c -> '0' <= c <= '9', rest)
end

# Whether an edge name is a positional default (`:step_i` / `:branch_i`),
# carrying no real event names to derive an origin/target split from.
function _is_positional_edge_name(name::Symbol)
    s = string(name)
    return _has_positional_suffix(s, "step_") ||
           _has_positional_suffix(s, "branch_")
end

# Split an underscore-joined edge name `:onset_admit` into its `(:onset, :admit)`
# origin/target event names. A name with no single internal split (or a
# positional default) has no derivable split and returns `nothing`, so the caller
# falls back to positional event names. This is the UNDERSCORED ("_" separator)
# EVENT/VALUE namespace, distinct from the DOTTED ("." separator) parameter-path
# namespace (`_join_path` / `_split_edge` in introspection.jl).
function _split_edge_name(name::Symbol)
    _is_positional_edge_name(name) && return nothing
    s = string(name)
    parts = split(s, '_')
    length(parts) == 2 || return nothing
    (isempty(parts[1]) || isempty(parts[2])) && return nothing
    return (Symbol(parts[1]), Symbol(parts[2]))
end

# --- flat EVENT-name layout for a tree --------------------------------------
#
# `_flat_event_names(d)` returns the tuple of event names matching the flat event
# vector `[E_0, E_1, ..., E_k]`: entry 1 is the root origin, then one name per
# LEAF event in depth-first order, exactly the layout `_tree_score` /
# `_event_vector` consume. Built by appending into a `Symbol[]` and freezing to a
# tuple, mirroring the `params_table` pre-order walk. Edge names are read from the
# PARENT composer's `names` field (a leaf edge does not store its own name), so
# each child is visited paired with its edge name.

# `_flat_event_names(d)` is the internal worker behind the public
# [`event_names`](@ref) (flat) accessor: the tuple of event names matching the
# scored event vector `[E_0, E_1, ..., E_k]`, the root origin event followed by
# one target event per edge in depth-first order. Event names are DERIVED from
# the composer's edge names (an edge `:onset_admit` gives origin `:onset` and
# target `:admit`); an edge with a positional default name (`:step_i` /
# `:branch_i`) contributes the positional event name `:event_i` instead. These
# EVENT names key a data ROW (a linelist column is an event time), distinct from
# the EDGE names ([`component_names`](@ref) / the parameter inventory). A row
# passed to `composed_distribution_model` is matched to the event vector BY these
# names, so field order does not matter.
function _flat_event_names(d::Union{Sequential, Parallel})
    names = Symbol[]
    counter = Ref(0)
    origin = _root_origin_name(d, counter)
    push!(names, origin)
    _walk_targets!(names, d, origin, counter)
    return Tuple(names)
end

# A standalone `Resolve` node has only a positional origin; its OUTCOME event
# names anchor at the parent event when nested (see `_walk_edge!` below), so on
# its own it exposes the origin plus one slot per outcome named by its outcomes.
_flat_event_names(c::AbstractOneOf) = (:event_1, c.names...)

# The root origin event name E_0: derived from the FIRST edge's name split, else
# positional. For a `Sequential` the first edge is `components[1]`; for a
# `Parallel` it is the first branch. A nested first child recurses to its own
# first edge.
function _root_origin_name(d::Union{Sequential, Parallel}, counter)
    name1 = component_names(d)[1]
    pair = _edge_origin_pair(name1, d.components[1])
    pair === nothing && return _next_event_name(counter)
    return pair
end

# The ORIGIN event name implied by an edge: the first half of a split edge name,
# recursing into a nested child's first edge. `nothing` when no name splits.
function _edge_origin_pair(edge_name::Symbol, child::UnivariateDistribution)
    split = _split_edge_name(edge_name)
    return split === nothing ? nothing : split[1]
end
function _edge_origin_pair(
        edge_name::Symbol, child::Union{Sequential, Parallel})
    return _root_origin_name_or_nothing(child)
end
# A nested `Choose` as a first edge derives its origin from the EDGE name split
# (a leaf alternative) or its default alternative's own first edge (a composer
# alternative); the alternatives share the slot layout, so the default names it.
function _edge_origin_pair(edge_name::Symbol, child::Choose)
    return _edge_origin_pair(edge_name, _flat_choose_alternative(child))
end
function _root_origin_name_or_nothing(d::Union{Sequential, Parallel})
    name1 = component_names(d)[1]
    return _edge_origin_pair(name1, d.components[1])
end

# Append the TARGET event(s) of each edge of composer `d` hanging off `origin`.
# A `Sequential` threads the terminal event forward step to step; a `Parallel`
# hangs every branch off the shared origin.
function _walk_targets!(names, d::Sequential, origin::Symbol, counter)
    prev = origin
    enames = component_names(d)
    for i in eachindex(d.components)
        prev = _walk_edge!(names, enames[i], d.components[i], prev, counter)
    end
    return nothing
end

function _walk_targets!(names, d::Parallel, origin::Symbol, counter)
    enames = component_names(d)
    for i in eachindex(d.components)
        _walk_edge!(names, enames[i], d.components[i], origin, counter)
    end
    return nothing
end

# Append one edge's target event(s) and return the edge's TERMINAL event name
# (what a following chain step hangs off). A leaf edge pushes its single target
# (the second half of its split name, else positional); a nested composer
# recurses and returns its terminal (its last leaf for a chain, the shared origin
# for a parallel, mirroring `_terminal_offset`).
function _walk_edge!(names, edge_name::Symbol, child::UnivariateDistribution,
        origin::Symbol, counter)
    split = _split_edge_name(edge_name)
    target = split === nothing ? _next_event_name(counter) : split[2]
    push!(names, target)
    return target
end

function _walk_edge!(names, edge_name::Symbol,
        child::Union{Sequential, Parallel}, origin::Symbol, counter)
    _walk_targets!(names, child, origin, counter)
    return _nested_terminal_name(child, names, origin)
end

# A nested `Resolve` edge contributes EVENT name(s) per OUTCOME, anchored at the
# parent `origin`. A LEAF outcome (a plain delay) is one event slot named by the
# outcome: the death/discharge columns of a record are each their own slot, so the
# observed outcome is identified by which slot is present. A NON-TERMINAL outcome
# whose payload is a composer SUBTREE (#466 Feature 3) instead emits the SUBTREE's
# event names, anchored at the outcome's resolution event (the subtree origin),
# sharing that slot exactly like a nested-composer origin: the outcome's resolution
# IS the subtree origin, so the subtree's `_walk_targets!` hangs off it rather than
# introducing a fresh origin slot. The edge/parameter names are unaffected (params
# still belong to the Resolve outcomes, see `params_table`). A Resolve is a
# terminal node for a FOLLOWING chain step (the chain does not continue through a
# single outcome), so its terminal name is the shared origin it hangs off.
function _walk_edge!(names, edge_name::Symbol, child::AbstractOneOf,
        origin::Symbol, counter)
    for k in eachindex(child.names)
        _walk_one_of_outcome!(names, child.names[k], child.delays[k],
            origin, counter)
    end
    return origin
end

# Append the event name(s) of ONE one_of outcome. A leaf outcome pushes its
# single name; a composer outcome walks its subtree anchored at the outcome's
# resolution event (the subtree origin). The outcome name itself is NOT pushed for
# a composer outcome: that name labels the resolution event, which IS the parent
# anchor shared into the subtree (no extra slot), so the subtree's own target
# events fill the outcome's slice.
function _walk_one_of_outcome!(names, oname::Symbol,
        delay::UnivariateDistribution, origin::Symbol, counter)
    push!(names, oname)
    return nothing
end

function _walk_one_of_outcome!(names, oname::Symbol,
        delay::Union{Sequential, Parallel}, origin::Symbol, counter)
    _walk_targets!(names, delay, oname, counter)
    return nothing
end

# A `Choose` outcome routes to one alternative of a shared event-slot width; its
# default alternative names the slot(s), anchored at the outcome's resolution.
function _walk_one_of_outcome!(names, oname::Symbol, delay::Choose,
        origin::Symbol, counter)
    return _walk_one_of_outcome!(names, oname,
        _flat_choose_alternative(delay), origin, counter)
end

# A nested `Resolve` outcome (a one_of node as a one_of branch) recurses
# through the one_of walk, anchored at the outcome's resolution event.
function _walk_one_of_outcome!(names, oname::Symbol, delay::AbstractOneOf,
        origin::Symbol, counter)
    _walk_edge!(names, oname, delay, oname, counter)
    return nothing
end

# A nested `Choose` edge contributes the event name(s) of its DEFAULT (first)
# alternative: the alternatives share one event-slot width, so the slot layout is
# the same whichever routes, and the default names the slot for `event_names`
# / `rand`. A leaf alternative pushes the split target of the EDGE name (so a
# `:admit_death` Choose edge still names its slot `:death`); a composer
# alternative recurses through its own walk.
function _walk_edge!(names, edge_name::Symbol, child::Choose,
        origin::Symbol, counter)
    return _walk_edge!(names, edge_name, _flat_choose_alternative(child),
        origin, counter)
end

_nested_terminal_name(::Parallel, names, origin::Symbol) = origin
_nested_terminal_name(::Sequential, names, origin::Symbol) = names[end]

# Allocate the next positional event name `:event_i`.
function _next_event_name(counter)
    counter[] += 1
    return Symbol(:event_, counter[])
end

# Whether a tuple of event names is the all-positional default layout (so a row
# is matched positionally, the documented fallback rather than by name).
function _all_positional_event_names(enames::Tuple)
    return all(n -> _has_positional_suffix(string(n), "event_"), enames)
end

# --- pure row -> event-vector / reserved-field parsing ----------------------
#
# These map a `NamedTuple` table row to the flat event vector and read the
# reserved (non-event) fields. They are PURE and Turing-free (data only), so they
# live in the core and are shared by BOTH the per-record `composed_distribution_
# model` (the DynamicPPL extension) and the vectorised `record_distributions`,
# keeping a single source of truth for the by-name row matching.

# Reserved row fields that are NOT events: a multiplicity weight (`weight` /
# `count`), a per-record observation horizon (`obs_time`, the hanta
# right-truncation observation time D), a per-record δ-bounded observation-window
# width (`obs_window`, which adds a LOWER edge a width δ below the horizon, giving
# the finite window `[obs_time - δ, obs_time]`), and a per-record Resolve
# branch-probability override (`branch_probs`) that rides a nested-Resolve tree
# row and is excluded from by-name event matching.
const _RESERVED_ROW_FIELDS = (
    :weight, :count, :obs_time, :obs_window, :branch_probs)

# The event values of a row in field order, dropping the reserved weight/count
# fields, as a `Vector{Union{Missing, Float64}}` (one entry per event, `missing`
# admitted). The `Missing`-admitting element type keeps the censored composer
# `logpdf` specialisation selected even for an all-observed row. This is
# the POSITIONAL fallback, used only when a composer carries no derivable event
# names (its edges are positional defaults).
function _row_event_vector(row::NamedTuple)
    ks = filter(k -> !(k in _RESERVED_ROW_FIELDS), keys(row))
    out = Vector{Union{Missing, Float64}}(undef, length(ks))
    for (i, k) in enumerate(ks)
        v = row[k]
        out[i] = v === missing ? missing : Float64(v)
    end
    return out
end

# The event vector for a composer `d` from a `row`, matched to the tree's flat
# EVENT names BY NAME: `row.onset, row.admit, row.death` land in their
# slots regardless of field order, `missing` fields drive the dispatch, and a
# reserved field is excluded. When the tree's event names are all positional
# defaults (`:event_i`), the row is matched POSITIONALLY (the fallback).
function _row_event_vector(d::Union{Sequential, Parallel}, row::NamedTuple)
    enames = _flat_event_names(d)
    _all_positional_event_names(enames) && return _row_event_vector(row)
    return _row_event_vector_by_name(enames, row)
end

# Build the by-name event vector: validate every non-reserved row field is a
# known event, then place each event by name (a missing required event errors).
function _row_event_vector_by_name(enames::Tuple, row::NamedTuple)
    for k in keys(row)
        k in _RESERVED_ROW_FIELDS && continue
        k in enames || throw(ArgumentError(
            "row field $(repr(k)) is not an event of this tree; expected " *
            "events $(collect(enames)) (reordering is allowed; names are not)"))
    end
    out = Vector{Union{Missing, Float64}}(undef, length(enames))
    for (i, name) in enumerate(enames)
        haskey(row, name) || throw(ArgumentError(
            "row is missing required event $(repr(name)); expected events " *
            "$(collect(enames))"))
        v = row[name]
        out[i] = v === missing ? missing : Float64(v)
    end
    return out
end

# The multiplicity weight carried by a row: an explicit `kw_weight` wins,
# otherwise a reserved `weight`/`count` field, otherwise `nothing` (unweighted).
function _row_weight_field(row::NamedTuple, kw_weight)
    kw_weight === nothing || return kw_weight
    haskey(row, :weight) && return row.weight
    haskey(row, :count) && return row.count
    return nothing
end

# The per-record observation horizon carried by a row's reserved fields. The
# upper edge is the `obs_time` horizon D (hanta): present and non-missing right-
# truncates the record at the horizon; absent or missing means no truncation
# (back-compat). A reserved `obs_window` field adds the δ-bounded LOWER edge: with
# `obs_time = D` and `obs_window = δ` the record is truncated to the finite window
# `[D - δ, D]`, returned as a `WindowedHorizon` carrier so the δ threads through
# the same scoring paths the plain horizon does. With no (or missing) `obs_window`
# a plain `Real` horizon is returned, byte-identical to the upper-only form.
function _row_horizon_field(row::NamedTuple)
    haskey(row, :obs_time) || return nothing
    h = row.obs_time
    h === missing && return nothing
    δ = _row_window_field(row)
    return δ === nothing ? h : WindowedHorizon(h, δ)
end

# The δ-bounded observation-window width carried by a row's reserved `obs_window`
# field, or `nothing` (upper-only) when absent or missing.
function _row_window_field(row::NamedTuple)
    haskey(row, :obs_window) || return nothing
    δ = row.obs_window
    return δ === missing ? nothing : δ
end
