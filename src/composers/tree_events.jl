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

# Whether an edge name is a positional default (`:step_i` / `:branch_i`),
# carrying no real event names to derive an origin/target split from.
function _is_positional_edge_name(name::Symbol)
    s = string(name)
    return occursin(r"^step_\d+$", s) || occursin(r"^branch_\d+$", s)
end

# Split an underscore-joined edge name `:onset_admit` into its `(:onset, :admit)`
# origin/target event names. A name with no single internal split (or a
# positional default) has no derivable split and returns `nothing`, so the caller
# falls back to positional event names.
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

# A standalone `Competing` node has only a positional origin; its OUTCOME event
# names anchor at the parent event when nested (see `_walk_edge!` below), so on
# its own it exposes the origin plus one slot per outcome named by its outcomes.
_flat_event_names(c::Competing) = (:event_1, c.names...)

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

# A nested `Competing` edge contributes one EVENT name per OUTCOME,
# anchored at the parent `origin`: the death/discharge columns of a record are
# each their own event slot, so the observed outcome is identified by which slot
# is present. The outcome names replace the single opaque resolution event; the
# edge/parameter names are unaffected (params still belong to the Competing
# outcomes, see `params_table`). A Competing is a terminal node (the chain does
# not continue through a single outcome), so its terminal name for a following
# step is the shared origin it hangs off.
function _walk_edge!(names, edge_name::Symbol, child::Competing,
        origin::Symbol, counter)
    for name in child.names
        push!(names, name)
    end
    return origin
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
    return all(n -> occursin(r"^event_\d+$", string(n)), enames)
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
# right-truncation observation time D), and a per-record Competing branch-
# probability override (`branch_probs`) that rides a nested-Competing tree
# row and is excluded from by-name event matching.
const _RESERVED_ROW_FIELDS = (:weight, :count, :obs_time, :branch_probs)

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

# The per-record observation horizon D carried by a row's reserved `obs_time`
# field (hanta): present and non-missing right-truncates the record at the
# horizon; absent or missing means no truncation (back-compat).
function _row_horizon_field(row::NamedTuple)
    haskey(row, :obs_time) || return nothing
    h = row.obs_time
    return h === missing ? nothing : h
end
