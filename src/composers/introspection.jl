# ============================================================================
# Prior introspection for composed distributions (Option A)
# ============================================================================
#
# After `compose(structure)`, these helpers read the composed distribution's
# free parameters so a user can define priors discoverably, against the
# structure rather than by hand-matching `lm_oa`/`ls_oa`/... :
#
# - `params(d)` (extended on the composers): a NESTED, NAME-keyed structure
#   mirroring the tree, leaves delegating to the standard/extended
#   `Distributions.params`.
# - `params_table(d)`: that structure FLATTENED to a Tables.jl table, one row
#   per scalar parameter, columns `edge | param | value | support` (the support
#   being the edge distribution's variate support, the domain a prior respects).
# - `event_names(d)` / `event(d, name)`: list the edge/event names of a
#   composed distribution and fetch a child by name.
#
# Names come from the composers' `names` field, which every `compose` front-end
# threads through (NamedTuple keys, table `name` column, matrix `names=`).
#
# IMPLEMENTATION NOTE (type stability): the `show` and `params`/`params_table`
# traversals are HAND-ROLLED, type-stable recursion over the component tuples,
# NOT AbstractTrees iterators (whose traversal is not type-stable for the
# heterogeneous composer tree). A thin AbstractTrees INTERFACE
# (`children`/`nodevalue`/`printnode`) is provided at the bottom purely so
# EXTERNAL code can traverse our trees; our own paths never use it.
#
# Distributions-led: this reads structure + `params` + `support`; it is not a
# model generator and stays Turing-free.

# --- node headers ----------------------------------------------------------

# Header label for a composer node (its TYPE plus a count).
_node_header(d::Sequential) = "Sequential ($(length(d.components)) steps)"
_node_header(d::Parallel) = "Parallel ($(length(d.components)) branches)"
_node_header(c::Competing) = "Competing ($(_n_branches(c)) outcomes)"

# Is a child a composer (has named children) or a leaf?
_is_composer_dist(::Union{Sequential, Parallel, Competing}) = true
_is_composer_dist(::Any) = false

# Named children of a composer as `(name, child[, note])` triples. The note is
# an extra print annotation (a `Competing` outcome's branch probability), empty
# otherwise. Hand-rolled and type-stable (a tuple comprehension over the
# constant-length component tuple).
function _named_children(d::Union{Sequential, Parallel})
    names = component_names(d)
    return ntuple(i -> (names[i], d.components[i], ""), length(d.components))
end
function _named_children(c::Competing)
    return ntuple(length(c.names)) do i
        (c.names[i], c.delays[i], "p = $(c.branch_probs[i])")
    end
end

# --- recursive indented-tree show (hand-rolled, type-stable) ---------------
#
# A nested composed distribution prints as ONE indented tree, recursing into
# every child so the whole structure is visible at once. Shared `├─ / └─` glyphs
# and indentation: a header line for the root, then each child indented one
# level, composer children recursing and leaf delays printed inline. The compact
# `show(io, d)` one-liners on each type are kept for inline/array display.

# Entry point shared by the three composer `show(::MIME"text/plain")` methods.
function _show_composer_tree(io::IO, d)
    println(io, _node_header(d))
    _show_children(io, d, "")
    return nothing
end

# Print the named children of `node` under `prefix`. Each child gets a `├─ `
# connector (`└─ ` for the last); a composer child recurses with an extended
# prefix (`│  ` for non-last siblings, three spaces for the last).
function _show_children(io::IO, node, prefix::String)
    children = _named_children(node)
    n = length(children)
    for i in 1:n
        last = i == n
        connector = last ? "└─ " : "├─ "
        name, child, note = children[i]
        label = isempty(note) ? "$(name): " : "$(name) ($(note)): "
        if _is_composer_dist(child)
            println(io, prefix, connector, label, _node_header(child))
            _show_children(io, child, prefix * (last ? "   " : "│  "))
        else
            println(io, prefix, connector, label, child)
        end
    end
    return nothing
end

# --- nested name-keyed params (hand-rolled, type-stable) --------------------

@doc "

Nested, name-keyed parameters of a composed distribution.

Returns a `NamedTuple` keyed by the node names, each value the `params` of that
child (recursing into nested composers; a leaf delegates to its standard/
extended `Distributions.params`). A `Competing` node contributes a name-keyed
NamedTuple of its outcomes plus a `branch_probs` entry. This nested form is for
prior introspection via [`params_table`](@ref); a composed distribution
reconstructs through [`compose`](@ref), not through `Distribution(params...)`.

See also: [`params_table`](@ref), [`event_names`](@ref), [`event`](@ref)
"
function _composed_params(d::Union{Sequential, Parallel})
    names = component_names(d)
    vals = map(_child_params, d.components)
    return NamedTuple{names}(vals)
end

_child_params(c::Union{Sequential, Parallel}) = _composed_params(c)
_child_params(c::Competing) = _competing_params(c)
_child_params(c) = params(c)

# A `Competing` node's nested params: each outcome name -> its delay's params,
# plus a `branch_probs` entry carrying the (free) outcome probabilities.
function _competing_params(c::Competing)
    outcome_vals = map(params, c.delays)
    outcomes = NamedTuple{c.names}(outcome_vals)
    return merge(outcomes, (; branch_probs = c.branch_probs))
end

# --- censoring-transparent leaves ------------------------------------------
#
# A composed leaf may itself be a censored delay (e.g. a
# `double_interval_censored(Gamma(...))`, i.e. an
# `IntervalCensored{Truncated{PrimaryCensored{Gamma}}}`). The censoring bounds
# (primary event, truncation, interval) are FIXED structure, not free
# parameters; only the inner delay's parameters (the `Gamma` shape/scale) are
# free. `_free_leaf` peels the fixed censoring off to the inner free delay, and
# `_rewrap_leaf` rebuilds the same censoring around a new inner delay. The
# introspection (`params_table`, names) and reconstruction layers go through
# these, so a censored leaf is transparent: its rows show only the inner free
# params and it round-trips by re-censoring the rebuilt delay. A plain leaf is
# the identity for both. The public `Distributions.params` is unchanged.

# Innermost free delay of a (possibly censored) leaf.
free_leaf(leaf) = leaf
free_leaf(d::PrimaryCensored) = free_leaf(d.dist)
free_leaf(d::IntervalCensored) = free_leaf(d.dist)
free_leaf(d::Truncated) = free_leaf(d.untruncated)

# Rebuild the SAME censoring around a new inner delay `inner`. Mirrors
# `free_leaf`: each wrapper recurses inwards then re-applies its fixed spec.
rewrap_leaf(leaf, inner) = inner
function rewrap_leaf(d::PrimaryCensored, inner)
    return PrimaryCensored(rewrap_leaf(d.dist, inner), d.primary_event, d.method)
end
function rewrap_leaf(d::IntervalCensored, inner)
    return IntervalCensored(rewrap_leaf(d.dist, inner), d.boundaries)
end
function rewrap_leaf(d::Truncated, inner)
    return truncated(rewrap_leaf(d.untruncated, inner); lower = d.lower,
        upper = d.upper)
end

# --- parameter-name introspection for leaves -------------------------------

# Best-effort scalar parameter NAMES for a leaf distribution, matched
# positionally to `params(leaf)`. Distributions.jl exposes parameter values via
# `params` but not their names generically, so common families are mapped
# explicitly; anything else falls back to `:param_1, :param_2, ...`.
_param_names(::Distributions.Normal) = (:mu, :sigma)
_param_names(::Distributions.LogNormal) = (:mu, :sigma)
_param_names(::Distributions.Gamma) = (:shape, :scale)
_param_names(::Distributions.Weibull) = (:shape, :scale)
_param_names(::Distributions.Exponential) = (:scale,)
_param_names(::Distributions.Uniform) = (:lower, :upper)
_param_names(::Any) = ()

# Names for the inner free delay's `params` tuple, padding with positional
# fallbacks so every value has a label even when the family is unmapped. A
# censored leaf delegates to its free delay (`free_leaf`), so the censoring
# bounds never appear.
function _leaf_param_names(leaf)
    inner = free_leaf(leaf)
    vals = params(inner)
    base = _param_names(inner)
    n = length(vals)
    return ntuple(n) do i
        i <= length(base) ? base[i] : Symbol(:param_, i)
    end
end

# --- params_table (hand-rolled pre-order walk) -----------------------------

# A thin wrapper over the flat column table so `params_table(d)` prints as an
# actual table (matching its name) rather than as a bare `NamedTuple` of vectors,
# while staying a first-class Tables.jl source. It forwards the whole Tables.jl
# column interface to the wrapped `NamedTuple`, so `Tables.istable`,
# `Tables.columns`, `Tables.getcolumn` and `DataFrame(tbl)` all work unchanged,
# and `getproperty` forwards `tbl.edge`/`tbl.param`/... to the columns. Only the
# `show(::MIME"text/plain")` is customised, to render a padded ASCII table.

@doc "

A Tables.jl column table of a composed distribution's free parameters.

The value [`params_table`](@ref) returns: a Tables.jl source (a column table)
that prints as a padded `edge | param | value | support` table. It is a thin
wrapper over a `NamedTuple` of equal-length column vectors, forwarding the whole
Tables.jl column interface and column access (`tbl.edge`, `tbl.param`, ...), so
`Tables.istable`, `Tables.columns`, `Tables.getcolumn`, `DataFrame(tbl)` and
[`build_priors`](@ref) all consume it unchanged; only its display is customised.

See also: [`params_table`](@ref), [`build_priors`](@ref).
"
struct ParamsTable{C <: NamedTuple}
    columns::C
end

# Tables.jl source interface: a column table, delegating to the wrapped columns.
Tables.istable(::Type{<:ParamsTable}) = true
Tables.columnaccess(::Type{<:ParamsTable}) = true
Tables.columns(t::ParamsTable) = getfield(t, :columns)
Tables.columnnames(t::ParamsTable) = keys(getfield(t, :columns))
Tables.getcolumn(t::ParamsTable, i::Int) = getfield(t, :columns)[i]
Tables.getcolumn(t::ParamsTable, nm::Symbol) = getfield(t, :columns)[nm]
Tables.schema(t::ParamsTable) = Tables.schema(getfield(t, :columns))
Tables.rowaccess(::Type{<:ParamsTable}) = true
Tables.rows(t::ParamsTable) = Tables.rows(getfield(t, :columns))

# Forward column access (`tbl.edge`, `tbl.param`, ...) to the wrapped columns so
# the table reads like the NamedTuple it wraps.
Base.getproperty(t::ParamsTable, nm::Symbol) = getfield(t, :columns)[nm]
Base.propertynames(t::ParamsTable) = keys(getfield(t, :columns))

# The number of rows (every column is equal length).
function _nrows(t::ParamsTable)
    cols = getfield(t, :columns)
    return isempty(cols) ? 0 : length(first(cols))
end

# A compact one-liner for inline / array display.
function Base.show(io::IO, t::ParamsTable)
    print(io, "ParamsTable($(_nrows(t)) rows)")
    return nothing
end

# A padded ASCII table for `text/plain` display, so `params_table(d)` renders as
# an actual table. Columns are `edge | param | value | support`; each cell is the
# `string` of the value, columns padded to their widest cell (header included).
function Base.show(io::IO, ::MIME"text/plain", t::ParamsTable)
    cols = getfield(t, :columns)
    names = collect(keys(cols))
    n = _nrows(t)
    println(io, "params_table ($n rows)")
    isempty(names) && return nothing
    # Stringify every cell, then size each column to its widest entry.
    cells = [string.(getindex(cols, nm)) for nm in names]
    headers = string.(names)
    widths = [maximum(length, vcat(headers[j], cells[j]); init = 0)
              for j in eachindex(names)]
    pad(s, w) = s * " "^(w - length(s))
    row(parts) = "  " * join((pad(parts[j], widths[j])
        for j in eachindex(parts)), "  ")
    println(io, row(headers))
    println(io, "  " * join(("─"^w for w in widths), "  "))
    for i in 1:n
        line = row([cells[j][i] for j in eachindex(names)])
        i == n ? print(io, line) : println(io, line)
    end
    return nothing
end

@doc "

Flatten a composed distribution's parameters into a prior-definition table.

`params_table(d)` returns a Tables.jl column table (a [`ParamsTable`](@ref)
wrapping a `NamedTuple` of equal-length column vectors, so
`Tables.istable(params_table(d))` is `true` and it prints as a padded table);
wrap it in `DataFrame` for a DataFrame. It has one row per scalar free parameter
of the composed distribution `d`, with columns:

- `edge`: the dotted path of names to the parameter's edge/leaf (e.g.
  `:onset_admit`, or `:resolution.branch_probs` inside a `Competing`).
- `param`: the parameter name (e.g. `:mu`, `:sigma`; positional `:param_i` where
  the family is unmapped).
- `value`: the current parameter value.
- `support`: the `(minimum, maximum)` variate support of that edge's
  distribution, the domain a prior over the edge must respect (from `minimum`/
  `maximum`/`support`).

Define priors against the rows of this table instead of hand-matching parameter
names. Built from [`params`](@ref) (nested, name-keyed values) plus the edge
distributions' support.

# Examples
```@example
using CensoredDistributions, Distributions

tree = compose((onset_admit = LogNormal(1.5, 0.4),
    admit_death = Gamma(2.0, 1.0)))
tbl = params_table(tree)
tbl.edge  # a column; wrap the table in `DataFrame(tbl)` for a DataFrame
```

# See also
- [`params`](@ref): the nested name-keyed values
- [`event_names`](@ref), [`event`](@ref): name introspection
"
function params_table(d::Union{Sequential, Parallel, Competing, Select})
    edges = Symbol[]
    params_col = Symbol[]
    values = Any[]
    supports = Any[]
    seen = Set{Symbol}()
    _walk_rows!(edges, params_col, values, supports, seen, d, ())
    return ParamsTable((edge = edges, param = params_col,
        value = values, support = supports))
end

# Pre-order walk over the composer tree. `path` is the tuple of names from the
# root to the current node. A composer recurses into its named children; a
# `Competing` additionally emits its branch-probability rows; a leaf emits one
# row per scalar parameter. Hand-rolled (not AbstractTrees) to stay type-stable.
function _walk_rows!(edges, params_col, values, supports, seen,
        d::Union{Sequential, Parallel}, path)
    names = component_names(d)
    for (name, child) in zip(names, d.components)
        _walk_rows!(edges, params_col, values, supports, seen, child,
            (path..., name))
    end
    return nothing
end

# A `Select`'s alternatives each contribute their own rows; a tag shared across
# alternatives is deduped via `seen`, so a parameter tied across the index and
# sourced branches is inventoried once.
function _walk_rows!(edges, params_col, values, supports, seen,
        d::Select, path)
    for (name, alt) in zip(d.names, d.alternatives)
        _walk_rows!(edges, params_col, values, supports, seen, alt,
            (path..., name))
    end
    return nothing
end

function _walk_rows!(edges, params_col, values, supports, seen,
        c::Competing, path)
    for (name, delay) in zip(c.names, c.delays)
        _walk_rows!(edges, params_col, values, supports, seen, delay,
            (path..., name))
    end
    sup = (zero(eltype(c.branch_probs)), one(eltype(c.branch_probs)))
    edge = _join_path((path..., :branch_probs))
    for (k, p) in enumerate(c.branch_probs)
        push!(edges, edge)
        push!(params_col, Symbol(c.names[k]))
        push!(values, p)
        push!(supports, sup)
    end
    return nothing
end

# Leaf distribution: one row per scalar FREE parameter. A censored leaf shows
# only its inner free delay's params and that delay's support (the censoring
# bounds are fixed structure, see `free_leaf`). A shared-tagged leaf
# (`_shared_tag`) is inventoried ONCE under its TAG as the edge: the first
# occurrence emits the rows, later occurrences with the same tag are skipped so
# the tied parameter is listed once.
function _walk_rows!(edges, params_col, values, supports, seen, leaf, path)
    tag = _shared_tag(leaf)
    tag !== nothing && tag in seen && return nothing
    inner = free_leaf(leaf)
    pnames = _leaf_param_names(leaf)
    vals = params(inner)
    sup = (minimum(inner), maximum(inner))
    edge = tag === nothing ? _join_path(path) : tag
    tag === nothing || push!(seen, tag)
    for (pname, v) in zip(pnames, vals)
        push!(edges, edge)
        push!(params_col, pname)
        push!(values, v)
        push!(supports, sup)
    end
    return nothing
end

# Join a name path to a single dotted `Symbol` (e.g. `(:a, :b)` -> `:a.b`); a
# single-element path keeps its bare name.
_join_path(path::Tuple) = Symbol(join(string.(path), "."))

# --- update: nested NamedTuple -> reconstructed distribution ----------------

# Reconstruct a (possibly censored) leaf from a new inner free delay built out
# of `vals`, re-applying the fixed censoring. Mirrors the extension's
# `_reconstruct_leaf` but is Turing-free; argument checks are kept on (this is
# building a concrete distribution, not a gradient hot path).
function _update_leaf(leaf, vals::Tuple)
    inner = free_leaf(leaf)
    ctor = Base.typename(typeof(inner)).wrapper
    return rewrap_leaf(leaf, ctor(vals...))
end

@doc "

Update a composed distribution's free parameters from a nested `NamedTuple`.

`update(d, params)` returns a new distribution of the SAME structure as `d` with
its free parameters replaced by the values in `params`. The `params` NamedTuple
mirrors the tree: a [`Sequential`](@ref)/[`Parallel`](@ref) is keyed by its edge
names, a leaf by its parameter names (as in [`params_table`](@ref)'s `param`
column), and a [`Competing`](@ref) by its outcome names plus an optional
`branch_probs` entry. A censored leaf is transparent: supply only the inner
delay's parameters and the censoring is carried through.

Pair with [`chain_to_params`](@ref) to read posterior means or a single draw
from a Turing chain into the right NamedTuple, so `update(template, means)`
returns a ready-to-`rand`/inspect distribution.

# Arguments
- `d`: the composed distribution (or bare leaf) to update.
- `params`: a nested NamedTuple of new parameter values keyed like `d`.

# Examples
```@example
using CensoredDistributions, Distributions

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
tree2 = update(tree, (onset_admit = (shape = 3.0, scale = 1.5),
    admit_death = (mu = 0.7, sigma = 0.5)))
event(tree2, :onset_admit)
```

# See also
- [`params_table`](@ref): the flat inventory whose `param` names key the leaves
- [`chain_to_params`](@ref): build the NamedTuple from a fitted chain
"
function update(d::Union{Sequential, Parallel, Competing, Select},
        params::NamedTuple)
    return _update(d, params, params)
end

function update(leaf, params::NamedTuple)
    return _update(leaf, params, params)
end

# `_update` is the recursive worker. The whole top-level `params` is threaded down
# as the `shared` source: a shared-tagged leaf is keyed at the TOP level by its
# tag (matching `params_table`'s tag edge), so every occurrence reads the one
# entry; per-node keys are validated against the per-occurrence params with the
# shared tags excluded.

function _update(d::Union{Sequential, Parallel}, params::NamedTuple, shared)
    names = component_names(d)
    _check_child_keys(params, names, nameof(typeof(d)), shared)
    parts = ntuple(length(names)) do i
        _update(d.components[i], _child_params(params, names[i]), shared)
    end
    return _rebuild(d, parts)
end

# A `Select` updates each alternative; a tag shared across alternatives reads one
# entry from `shared` and is placed in every occurrence.
function _update(d::Select, params::NamedTuple, shared)
    _check_child_keys(params, d.names, :Select, shared)
    alts = ntuple(length(d.names)) do i
        _update(d.alternatives[i], _child_params(params, d.names[i]), shared)
    end
    return Select(d.names, alts, d.selector)
end

function _update(c::Competing, params::NamedTuple, shared)
    _check_child_keys(params, c.names, :Competing, shared; optional = (:branch_probs,))
    delays = ntuple(length(c.names)) do i
        _update(c.delays[i], _child_params(params, c.names[i]), shared)
    end
    probs = if haskey(params, :branch_probs)
        bp = params.branch_probs
        _check_update_keys(bp, c.names, Symbol("Competing branch_probs"))
        ntuple(i -> bp[c.names[i]], length(c.names))
    else
        c.branch_probs
    end
    return Competing(c.names, delays, probs)
end

# Leaf: take the new parameter values in `_leaf_param_names` order and rebuild. A
# shared-tagged leaf reads its values from the top-level `shared` entry under its
# tag, so every occurrence of the tag updates from the one entry.
function _update(leaf, params::NamedTuple, shared)
    tag = _shared_tag(leaf)
    leaf_params = tag === nothing ? params : _shared_entry(shared, tag, leaf)
    pnames = _leaf_param_names(leaf)
    _check_update_keys(leaf_params, pnames, nameof(typeof(leaf)))
    vals = ntuple(i -> leaf_params[pnames[i]], length(pnames))
    return _update_leaf(leaf, vals)
end

# A child's per-occurrence params: a shared-tagged child carries no per-occurrence
# entry (its values live at the top level under its tag), so an absent key is fine
# and an empty NamedTuple is threaded down (the leaf then reads `shared`).
function _child_params(params::NamedTuple, name::Symbol)
    return haskey(params, name) ? params[name] : NamedTuple()
end

# The top-level shared entry for a tag, erroring clearly when it is absent.
function _shared_entry(shared::NamedTuple, tag::Symbol, leaf)
    haskey(shared, tag) || throw(ArgumentError(
        "update(...) is missing the shared parameter $(repr(tag)) " *
        "(a `shared($(repr(tag)), ...)` leaf needs a top-level `$tag` entry)"))
    return shared[tag]
end

# Validate a composer node's child keys. A child key may be ABSENT (a branch
# whose only params are shared carries no per-occurrence entry; the leaf reads
# the top-level shared entry), so missing names are tolerated; an UNEXPECTED key
# (not a child name, a shared tag, or an `optional`) errors. With no shared tags
# and no all-shared branch this is the same exact-cover check as before.
function _check_child_keys(params::NamedTuple, names::Tuple, what, shared;
        optional::Tuple = ())
    allowed = (names..., optional..., keys(shared)...)
    extra_keys = filter(k -> !(k in allowed), keys(params))
    isempty(extra_keys) || throw(ArgumentError(
        "update($what, ...) has unexpected keys $(collect(extra_keys)); " *
        "expected $(collect(names))"))
    return nothing
end

# Validate `params` covers exactly `expected` (plus any `optional` keys) at the
# current node, with a clear error naming the node.
function _check_update_keys(params::NamedTuple, expected::Tuple, what;
        optional::Tuple = ())
    have = keys(params)
    missing_keys = filter(k -> !(k in have), expected)
    extra_keys = filter(k -> !(k in expected) && !(k in optional), have)
    isempty(missing_keys) || throw(ArgumentError(
        "update($what, ...) is missing $(collect(missing_keys)); " *
        "expected $(collect(expected))"))
    isempty(extra_keys) || throw(ArgumentError(
        "update($what, ...) has unexpected keys $(collect(extra_keys)); " *
        "expected $(collect(expected))"))
    return nothing
end

function _check_update_keys(params, ::Tuple, what; optional::Tuple = ())
    throw(ArgumentError(
        "update($what, ...) expects a NamedTuple; got $(typeof(params))"))
end

# `_rebuild` for the composers (mirrors the extension's helper, kept core-side so
# `update` is Turing-free).
_rebuild(d::Sequential, components::Tuple) = Sequential(components, d.names)
_rebuild(d::Parallel, components::Tuple) = Parallel(components, d.names)

# --- build_priors: params_table + flat priors -> nested NamedTuple ----------

# Split a dotted edge `Symbol` (`:a.b`) back into its name path (`(:a, :b)`).
function _split_edge(edge::Symbol)
    parts = split(string(edge), '.')
    return Tuple(Symbol.(parts))
end

# Insert `value` at the `(path..., leaf)` location of a nested `Dict` tree,
# creating intermediate `Dict`s as needed. Used to assemble the nested prior
# structure from flat table rows before freezing to NamedTuples.
function _nest_insert!(tree::Dict, path::Tuple, leaf::Symbol, value)
    node = tree
    for k in path
        node = get!(node, k, Dict{Symbol, Any}())::Dict{Symbol, Any}
    end
    node[leaf] = value
    return nothing
end

# Freeze a nested `Dict{Symbol}` tree into nested `NamedTuple`s (leaves, the
# prior objects, are left untouched).
_freeze_tree(x) = x
function _freeze_tree(d::Dict{Symbol})
    ks = Tuple(keys(d))
    return NamedTuple{ks}(map(k -> _freeze_tree(d[k]), ks))
end

# --- parameter-derived default priors (brms-style family defaults) ----------
#
# The default prior is classified from the PARAMETER's own natural domain, not
# the leaf's variate support: a location-family delay (`Normal`, `Affine(Normal)`)
# has unbounded variate support, but its scale parameter still lives on the
# positive half-line, so a `minimum(dist)`/`maximum(dist)` rule would wrongly
# give it an unconstrained prior with mass on negative scale.

# Location parameters live on the whole line (a `Normal`/`LogNormal` `mu`, a
# `Uniform` bound), so they get an unconstrained default.
function _is_location_param(p::Symbol)
    p === :mu || p === :location || p === :loc || p === :lower || p === :upper
end

# Scale/shape/rate-type parameters are positive by construction (the `sigma` of a
# `Normal`/`LogNormal`, the `shape`/`scale` of a `Gamma`/`Weibull`, the `scale`
# of an `Exponential`, and the common positive parameter names of related
# families), so they get a positive-truncated default regardless of the leaf's
# variate support.
function _is_positive_param(p::Symbol)
    p === :sigma || p === :scale || p === :rate || p === :shape ||
        p === :alpha || p === :beta || p === :theta || p === :nu ||
        p === :k || p === :df
end

@doc "

Pick a default prior for a parameter row, brms-style.

`default_prior(row)` is the per-row default [`build_priors`](@ref) uses for rows
the user does not override. `row` is a `(; edge, param, value, support)`
NamedTuple (a [`params_table`](@ref) row); the prior family follows the
parameter's own natural domain (classified by name), not the leaf's variate
support:

- a probability parameter, support `[0, 1]` (a `branch_probs` row) ->
  `Uniform(0, 1)`.
- a scale/shape/rate-type parameter (`:sigma`, `:scale`, `:shape`, `:rate`, ...)
  -> `truncated(Normal(value, scale); lower = 0)`, positive by construction even
  for a location-family delay (a `Normal`/`Affine(Normal)` `sigma`).
- a location parameter (`:mu`, `:location`, a `Uniform` bound) ->
  `Normal(value, scale)`, unconstrained since the location lives on the whole
  line even for a positive-support delay.
- otherwise, an unmapped name falls back to the variate support: a non-negative
  support -> `truncated(Normal(value, scale); lower = 0)`, else
  `Normal(value, scale)`.

The spread `scale` defaults to `max(abs(value), 1)`, a weakly-informative width
that scales with the parameter's magnitude.

# Arguments
- `row`: a [`params_table`](@ref) row `(; edge, param, value, support)`.

# Examples
```@example
using CensoredDistributions, Distributions

# A positive scale parameter -> a positive-truncated default.
default_prior((; edge = :onset_admit, param = :scale,
    value = 1.0, support = (0.0, Inf)))
```

# See also
- [`build_priors`](@ref): assembles the nested prior NamedTuple, using this as
  the per-row default and accepting overrides.
"
function default_prior(row)
    lo, hi = row.support
    scale = max(abs(float(row.value)), one(float(row.value)))
    if lo == 0 && hi == 1
        return Distributions.Uniform(0, 1)
    elseif _is_positive_param(row.param)
        return Distributions.truncated(
            Distributions.Normal(row.value, scale); lower = 0)
    elseif _is_location_param(row.param)
        return Distributions.Normal(row.value, scale)
    elseif lo >= 0 && isinf(hi)
        return Distributions.truncated(
            Distributions.Normal(row.value, scale); lower = 0)
    else
        return Distributions.Normal(row.value, scale)
    end
end

@doc "

Assemble the nested prior `NamedTuple` from a [`params_table`](@ref) inventory.

`build_priors(table; priors, default)` turns the flat parameter table into the
nested `NamedTuple` that [`composed_parameters_model`](@ref) (and [`update`](@ref))
expect, so users define priors against the flat table rows rather than by hand-
matching the tree.

For each row the prior is chosen in order:
1. a user `priors` override for that `(edge, param)`, if present, else
2. `default(row)`, the per-row default (support-derived [`default_prior`](@ref)
   unless a different `default` function is given).

By default every row gets a sensible support-derived prior, so
`build_priors(params_table(tree))` alone yields a complete prior NamedTuple. A
user overrides only the parameters they care about (brms-style partial override)
through `priors`.

`row` is a `NamedTuple` `(; edge, param, value, support)` (the table's columns
for that row), so a custom `default` can pick a prior from the parameter's
`support`.

# Arguments
- `table`: a [`params_table`](@ref) inventory (any Tables.jl column table with
  `edge`, `param`, `value`, `support` columns).

# Keyword Arguments
- `priors`: per-parameter overrides, either a `(edge, param) => prior` mapping
  (e.g. a `Dict`) or a nested `NamedTuple` keyed like the tree
  (`(onset_admit = (shape = prior,),)`); only the listed parameters are
  overridden (default: empty).
- `default`: a function `row -> prior` for rows not overridden (default:
  [`default_prior`](@ref), deriving the prior family from the parameter's
  support).

# Examples
```@example
using CensoredDistributions, Distributions

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
tbl = params_table(tree)
# Support-derived defaults everywhere, overriding only one parameter.
nested = build_priors(tbl;
    priors = (onset_admit = (shape = truncated(Normal(2, 0.5); lower = 0),),))
nested.onset_admit.shape
```

# See also
- [`params_table`](@ref): the flat inventory keyed against.
- [`default_prior`](@ref): the support-derived per-row default.
- [`composed_parameters_model`](@ref), [`update`](@ref): consume the result.
"
function build_priors(table; priors = Dict{Tuple{Symbol, Symbol}, Any}(),
        default = default_prior)
    edges = Tables.getcolumn(table, :edge)
    params_col = Tables.getcolumn(table, :param)
    values = Tables.getcolumn(table, :value)
    supports = Tables.getcolumn(table, :support)
    tree = Dict{Symbol, Any}()
    for i in eachindex(edges)
        edge = edges[i]
        param = params_col[i]
        ovr = _prior_override(priors, edge, param)
        prior = if ovr !== nothing
            ovr
        elseif default !== nothing
            row = (; edge = edge, param = param,
                value = values[i], support = supports[i])
            default(row)
        else
            throw(ArgumentError(
                "no prior for ($edge, $param) and no default supplied"))
        end
        _nest_insert!(tree, _split_edge(edge), param, prior)
    end
    return _freeze_tree(tree)
end

# A user override for `(edge, param)`, or `nothing` if none. Accepts a mapping
# keyed by the `(edge, param)` pair (a `Dict`) or a nested `NamedTuple` keyed
# like the tree (descend the edge path, then the param). Missing keys return
# `nothing` so the row falls through to the default.
function _prior_override(priors::NamedTuple, edge::Symbol, param::Symbol)
    node = priors
    for name in _split_edge(edge)
        node isa NamedTuple && haskey(node, name) || return nothing
        node = node[name]
    end
    node isa NamedTuple && haskey(node, param) || return nothing
    return node[param]
end

function _prior_override(priors, edge::Symbol, param::Symbol)
    key = (edge, param)
    return haskey(priors, key) ? priors[key] : nothing
end

# --- name introspection ----------------------------------------------------

@doc "

The FLAT event names of a composed distribution.

`event_names(d)` returns the tuple of event names in the SAME flat depth-first
layout as a `rand(latent(d))` draw and the per-event
[`mean`](@ref)`(latent(d))`/`var(latent(d))`/`std(latent(d))`: the root origin
event followed by one target event per leaf edge.
An inner composer's events are exposed, so `compose((path = [a, b],))` lists the
inner `(:onset, ...)` events rather than just the `(:path,)` edge. Event names
are derived from the edge names (an edge `:onset_admit` gives origin `:onset` and
target `:admit`); a positional default edge contributes `:event_i`. These EVENT
names key a data ROW, distinct from the nested EDGE/child structure of
[`event_tree`](@ref) (whose first level is the top-level child names).

# Examples
```@example
using CensoredDistributions, Distributions

tree = compose((onset_admit = LogNormal(1.5, 0.4),
    admit_death = Gamma(2.0, 1.0)))
event_names(tree)
```

# See also
- [`event_tree`](@ref): the NESTED tree of event names
- [`event`](@ref): fetch a child or subtree by name path
- [`params_table`](@ref): the parameter table
"
event_names(d::Union{Sequential, Parallel, Competing}) = _flat_event_names(d)
# A `Select` has no single flat layout (the active alternative is data-selected),
# so its flat event names are its alternative names.
event_names(d::Select) = d.names

@doc "

The NESTED tree of event names of a composed distribution.

`event_tree(d)` returns the event-name structure as data: a nested `NamedTuple`
keyed by child name down to the leaves, mirroring the tree. Its FIRST level is
the top-level child names (the old top-level `event_names` result); a
[`Sequential`](@ref)/[`Parallel`](@ref)/[`Select`](@ref) child recurses to its
own nested NamedTuple, a [`Competing`](@ref) child to its outcome names, and a
leaf to its own name. Pair with [`event_names`](@ref) for the FLAT per-event
layout that matches `rand`/`mean`/`var`.

# Examples
```@example
using CensoredDistributions, Distributions

tree = compose((admit_path = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4))),
    onset_recover = Gamma(3.0, 1.0)))
event_tree(tree)
```

# See also
- [`event_names`](@ref): the FLAT per-event names
- [`event`](@ref): fetch a child or subtree by name path
"
function event_tree(d::Union{Sequential, Parallel})
    names = component_names(d)
    vals = ntuple(i -> _event_tree_child(names[i], d.components[i]),
        length(names))
    return NamedTuple{names}(vals)
end

function event_tree(c::Competing)
    vals = ntuple(i -> _event_tree_child(c.names[i], c.delays[i]),
        length(c.names))
    return NamedTuple{c.names}(vals)
end

function event_tree(d::Select)
    vals = ntuple(i -> _event_tree_child(d.names[i], d.alternatives[i]),
        length(d.names))
    return NamedTuple{d.names}(vals)
end

# A composer child recurses to its own nested NamedTuple; a leaf is keyed by its
# parent under its own name, so its value is just that name (the leaf event).
function _event_tree_child(::Symbol, c::Union{Sequential, Parallel, Competing, Select})
    event_tree(c)
end
_event_tree_child(name::Symbol, ::Any) = name

# Direct-child lookup by a single (un-dotted) name. Internal so the public
# `event` can split a dotted path before descending.
function _event_child(d::Union{Sequential, Parallel}, name::Symbol)
    names = component_names(d)
    idx = findfirst(==(name), names)
    idx === nothing && throw(KeyError(name))
    return d.components[idx]
end

function _event_child(c::Competing, name::Symbol)
    idx = findfirst(==(name), c.names)
    idx === nothing && throw(KeyError(name))
    return c.delays[idx]
end

function _event_child(d::Select, name::Symbol)
    idx = findfirst(==(name), d.names)
    idx === nothing && throw(KeyError(name))
    return d.alternatives[idx]
end

@doc "

Fetch a composed distribution's child (event/edge), or descend a name path.

`event(d, path...)` returns the sub-distribution of `d` at the named location: a
single `Symbol` fetches a direct child (a branch of a [`Parallel`](@ref), a step
of a [`Sequential`](@ref), an outcome delay of a [`Competing`](@ref), or an
alternative of a [`Select`](@ref)); multiple `Symbol`s, or a single dotted-path
`Symbol` (`:admit_path.admit_death`, as in [`params_table`](@ref)'s `edge`
column), descend the tree one name per step. Throws a `KeyError` if a name along
the path is not a child at that level.

# Arguments
- `d`: the composed distribution to look up a child of (or descend into).
- `path`: one or more edge/event names (`Symbol`s) from `d` down to the target,
  or a single dotted-path `Symbol`.

# Examples
```@example
using CensoredDistributions, Distributions

tree = compose((admit_path = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4))),
    onset_recover = Gamma(3.0, 1.0)))
event(tree, :onset_recover)
event(tree, :admit_path, :admit_death)
```

# See also
- [`event_names`](@ref): list a node's flat event names
- [`event_tree`](@ref): the nested tree of event names
"
function event(d)
    # No name is an error: `event` needs at least one name to fetch.
    throw(ArgumentError("event needs at least one name"))
end

# A single name: a dotted `Symbol` (`:a.b`) splits into its steps and descends;
# a bare name is a single direct-child lookup.
function event(d, name::Symbol)
    steps = _split_edge(name)
    node = d
    for step in steps
        node = _event_child(node, step)
    end
    return node
end

# Two or more names descend the tree one step per name.
function event(d, name1::Symbol, name2::Symbol, rest::Symbol...)
    node = _event_child(d, name1)
    for name in (name2, rest...)
        node = _event_child(node, name)
    end
    return node
end

# --- AbstractTrees interop (interface only; NOT used internally) -----------
#
# Thin AbstractTrees interface so EXTERNAL code can walk a composed tree with
# the standard `children`/`nodevalue`/`printnode`/`print_tree`. Our own `show`,
# `params`, `params_table` and the `logpdf`/`rand`/AD hot paths deliberately do
# NOT use these (AbstractTrees traversal is not type-stable for heterogeneous
# trees). Nodes are wrapped in `ComposerNode` to carry the edge name; a leaf has
# no children.

struct ComposerNode{D}
    name::Symbol
    dist::D
    "Print annotation (a `Competing` outcome's branch probability), else empty."
    note::String
end
ComposerNode(name::Symbol, dist) = ComposerNode(name, dist, "")

# Root wrapper for a composed distribution (unnamed root).
_root_node(d) = ComposerNode(:root, d)

function AbstractTrees.children(node::ComposerNode)
    _is_composer_dist(node.dist) || return ()
    return map(t -> ComposerNode(t[1], t[2], t[3]), _named_children(node.dist))
end

AbstractTrees.nodevalue(node::ComposerNode) = node.dist

function AbstractTrees.printnode(io::IO, node::ComposerNode)
    name = node.name === :root ? "" :
           isempty(node.note) ? "$(node.name): " :
           "$(node.name) ($(node.note)): "
    if _is_composer_dist(node.dist)
        print(io, name, _node_header(node.dist))
    else
        print(io, name, node.dist)
    end
    return nothing
end
