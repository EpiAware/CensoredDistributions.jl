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
# NOT generic tree iterators (whose traversal is not type-stable for the
# heterogeneous composer tree). Structure introspection for external code is
# provided by `event_names`/`event_tree`/`event`, not a tree-walking interface.
#
# Distributions-led: this reads structure + `params` + `support`; it is not a
# model generator and stays Turing-free.

# --- node headers ----------------------------------------------------------

# Header label for a composer node (its TYPE plus a count).
_node_header(d::Sequential) = "Sequential ($(length(d.components)) steps)"
_node_header(d::Parallel) = "Parallel ($(length(d.components)) branches)"
_node_header(c::Resolve) = "Resolve ($(_n_branches(c)) outcomes)"
function _node_header(c::Compete)
    return "Compete ($(_n_branches(c)) racing outcomes)"
end
# A `Choose` is not `<: AbstractOneOf` (no shared origin, no branch
# probabilities, `.alternatives` not `.delays`), so it has its own header.
function _node_header(d::Choose)
    return "Choose ($(_n_alternatives(d)) alternatives, " *
           "selector = $(repr(d.selector)))"
end

# Is a child a composer (has named children) or a leaf? `Choose` is deliberately
# NOT `<: AbstractOneOf`, so it is listed explicitly to take part in the shared
# tree render (recursing into a nested `Choose` rather than printing it flat).
_is_composer_dist(::Union{Sequential, Parallel, AbstractOneOf, Choose}) = true
_is_composer_dist(::Any) = false

# Named children of a composer as `(name, child[, note])` triples. The note is
# an extra print annotation (a `Resolve` outcome's branch probability), empty
# otherwise. Hand-rolled and type-stable (a tuple comprehension over the
# constant-length component tuple).
function _named_children(d::Union{Sequential, Parallel})
    names = component_names(d)
    return ntuple(i -> (names[i], d.components[i], ""), length(d.components))
end
function _named_children(c::Resolve)
    return ntuple(length(c.names)) do i
        (c.names[i], c.delays[i], "p = $(c.branch_probs[i])")
    end
end
# A racing-hazard node has no per-outcome branch probability (it is derived), so
# its children carry the `racing` annotation instead.
function _named_children(c::Compete)
    return ntuple(length(c.names)) do i
        (c.names[i], c.delays[i], "racing")
    end
end
# A `Choose` alternative carries no branch probability (the active one is chosen
# by the row's selector value), so its children take no annotation.
function _named_children(d::Choose)
    return ntuple(i -> (d.names[i], d.alternatives[i], ""),
        _n_alternatives(d))
end

# --- recursive indented-tree show (hand-rolled, type-stable) ---------------
#
# A nested composed distribution prints as ONE indented tree, recursing into
# every child so the whole structure is visible at once. Shared `├─ / └─` glyphs
# and indentation: a header line for the root, then each child indented one
# level, composer children recursing and leaf delays printed inline. The compact
# `show(io, d)` one-liners on each type are kept for inline/array display.

# Entry point shared by the composer `show(::MIME"text/plain")` methods
# (`Sequential`, `Parallel`, `Resolve`, `Compete`, `Choose`).
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

# --- opt-in detailed inspection --------------------------------------------
#
# `show` is deliberately compact (structure plus short leaf labels); `inspect`
# is the explicit opt-in for the full nested detail, recursing the same tree but
# printing each leaf's full `text/plain` show (every field, including a leaf's
# solver) under an indented prefix.

@doc "

Print a composed distribution's full nested detail.

`inspect(io, d)` walks the same tree as `show` but prints each leaf's full
`text/plain` representation (every field, including a censored leaf's solver),
so it is the opt-in companion to the compact structural `show`. A composer node
prints its header and recurses; a leaf prints its detailed multi-line show
indented under its name. Writes to `io` (default `stdout`) and returns nothing.

# Arguments
- `io`: the IO stream to print to (default `stdout`).
- `d`: the composed distribution (or bare leaf) to inspect.

# Examples
```@example
using CensoredDistributions, Distributions

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
inspect(tree)
```

# See also
- [`event_tree`](@ref): the nested tree of event names
- [`params_table`](@ref): the flat parameter inventory
"
function inspect(io::IO, d)
    if _is_composer_dist(d)
        println(io, _node_header(d))
        _inspect_children(io, d, "")
    else
        _inspect_leaf(io, d, "")
    end
    return nothing
end

inspect(d) = inspect(stdout, d)

# Recurse the composer tree like `_show_children`, but print each leaf's full
# `text/plain` detail (rather than its compact one-line label) indented under
# its name.
function _inspect_children(io::IO, node, prefix::String)
    children = _named_children(node)
    n = length(children)
    for i in 1:n
        last = i == n
        connector = last ? "└─ " : "├─ "
        name, child, note = children[i]
        label = isempty(note) ? "$(name): " : "$(name) ($(note)): "
        child_prefix = prefix * (last ? "   " : "│  ")
        if _is_composer_dist(child)
            println(io, prefix, connector, label, _node_header(child))
            _inspect_children(io, child, child_prefix)
        else
            println(io, prefix, connector, label)
            _inspect_leaf(io, child, child_prefix)
        end
    end
    return nothing
end

# Print a leaf's detail, unwrapping any censoring layers onto their own lines so
# every component (delay, primary event, truncation, interval, solver) is
# visible, while still summarising a quadrature solver by its type and node
# COUNT rather than dumping its node and weight arrays.
function _inspect_leaf(io::IO, leaf, prefix::String)
    for line in _leaf_detail_lines(leaf)
        println(io, prefix, line)
    end
    return nothing
end

# The detail lines for a (possibly censored) leaf, peeling each censoring
# wrapper into a labelled line and recursing into its inner distribution. A
# plain leaf is its own single compact line.
function _leaf_detail_lines(d::PrimaryCensored)
    return vcat("PrimaryCensored",
        _indent_field("dist", _leaf_detail_lines(get_dist(d))),
        "  primary_event: $(d.primary_event)", "  method: $(d.method)")
end

function _leaf_detail_lines(d::IntervalCensored)
    spec = is_regular_intervals(d) ? "interval: $(d.boundaries)" :
           "boundaries: $(length(d.boundaries)) edges"
    return vcat("IntervalCensored",
        _indent_field("dist", _leaf_detail_lines(d.dist)), "  $spec")
end

function _leaf_detail_lines(d::Truncated)
    return vcat("Truncated",
        _indent_field("dist", _leaf_detail_lines(d.untruncated)),
        "  lower: $(d.lower)", "  upper: $(d.upper)")
end

_leaf_detail_lines(leaf) = [sprint(show, leaf)]

# Indent a field's value lines under `  <name>: `, putting the first value line
# on the label line and aligning any continuation lines beneath it.
function _indent_field(name::String, lines::Vector{String})
    isempty(lines) && return ["  $name: "]
    head = "  $name: $(first(lines))"
    tail = ["    " * l for l in lines[2:end]]
    return vcat(head, tail)
end

# --- nested name-keyed params (hand-rolled, type-stable) --------------------

@doc "

Nested, name-keyed parameters of a composed distribution.

Returns a `NamedTuple` keyed by the node names, each value the `params` of that
child (recursing into nested composers; a leaf delegates to its standard/
extended `Distributions.params`). A `Resolve` node contributes a name-keyed
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
_child_params(c::Resolve) = _one_of_params(c)
_child_params(c::Compete) = _hazard_one_of_params(c)
_child_params(c::Choose) = _choose_params(c)
_child_params(c) = params(c)

# A racing-hazard node's nested params: each outcome name -> its delay's params.
# There is NO `branch_probs` entry (the winning probability is derived).
function _hazard_one_of_params(c::Compete)
    return NamedTuple{c.names}(map(params, c.delays))
end

# A `Resolve` node's nested params: each outcome name -> its delay's params,
# plus a `branch_probs` entry carrying the (free) outcome probabilities.
function _one_of_params(c::Resolve)
    outcome_vals = map(params, c.delays)
    outcomes = NamedTuple{c.names}(outcome_vals)
    return merge(outcomes, (; branch_probs = c.branch_probs))
end

# A `Choose` node's nested params: a NamedTuple keyed by the alternative NAMES,
# each value the alternative's own `_child_params` (recursing into a nested
# composer, delegating to `params` at a leaf). The public `params(::Choose)`
# stays positional (mirroring `params(::Resolve)`), but this name-keyed form is
# what the nested params tree threads when a `Choose` is a child, so a `Choose`
# under a `Sequential`/`Parallel` yields a name-keyed subtree rather than a
# positional tuple. Per-branch params are namespaced per alternative
# (`index.…`/`sourced.…`); a tag shared across alternatives via `shared(:tag,...)`
# still appears once per occurrence here and is inventoried/sampled once by
# `params_table`/the prior model.
function _choose_params(d::Choose)
    vals = map(_child_params, d.alternatives)
    return NamedTuple{d.names}(vals)
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

# Whether a leaf distribution constructor `ctor` accepts a `check_args` keyword for
# the sampled value tuple `vals`. Used by the DynamicPPL extension's leaf
# reconstruction to skip the argument check (so a sampler probing an out-of-support
# point yields `-Inf` rather than throwing mid-gradient) only where the family
# supports it. Pure reflection returning a `Bool` (constant w.r.t. the params), so a
# zero-derivative primitive: the `CensoredDistributionsMooncakeExt` registers a
# Mooncake `@zero_adjoint` for it so Mooncake reverse never traces its underlying
# `jl_gf_invoke_lookup` foreigncall (which Mooncake on Julia LTS cannot
# differentiate), keeping the reconstruction AD-safe on every backend and Julia
# version.
function _ctor_has_check_args(ctor, vals::Tuple)
    return hasmethod(ctor, typeof(vals), (:check_args,))
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
# bounds never appear. A `thin(d, p)` leaf appends a trailing `:thin` name for
# its reporting weight, the free parameter the `ThinOp` carries.
function _leaf_param_names(leaf)
    inner = free_leaf(leaf)
    vals = params(inner)
    base = _param_names(inner)
    n = length(vals)
    delay_names = ntuple(n) do i
        i <= length(base) ? base[i] : Symbol(:param_, i)
    end
    return _thin_factor(leaf) === nothing ? delay_names :
           (delay_names..., :thin)
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
  `:onset_admit`, or `:resolution.branch_probs` inside a `Resolve`).
- `param`: the parameter name (e.g. `:mu`, `:sigma`; positional `:param_i` where
  the family is unmapped).
- `value`: the current parameter value.
- `support`: the `(minimum, maximum)` variate support of that edge's
  distribution, the domain a prior over the edge must respect (from `minimum`/
  `maximum`/`support`).

Define priors against the rows of this table instead of hand-matching parameter
names. Built from [`params`](@ref) (nested, name-keyed values) plus the edge
distributions' support.

For a [`Choose`](@ref) node the alternatives' independent per-branch params are
namespaced per alternative (`index.…` / `sourced.…`), one row-group per
alternative. A parameter tied across alternatives via [`shared`](@ref)`(:tag,
...)` is inventoried ONCE under its `tag` edge and sampled once, so a value tied
across the index and sourced branches appears as a single row-group.

A bare leaf distribution (no composer wrapping it) is also accepted; its rows
carry an empty `edge`, so [`build_priors`](@ref) keys the priors flat by
parameter name.

A [`thin`](@ref)`(d, p)` leaf surfaces an extra `:thin` row for its reporting
probability `p` (after the inner delay's rows, support `[0, 1]`), since `p` is a
free parameter that enters the per-record likelihood, so it can be given a prior
and overridden like the delay parameters.

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
function params_table(
        d::Union{Sequential, Parallel, AbstractOneOf, Choose})
    edges = Symbol[]
    params_col = Symbol[]
    values = Any[]
    supports = Any[]
    seen = Set{Symbol}()
    _walk_rows!(edges, params_col, values, supports, seen, d, ())
    return ParamsTable((edge = edges, param = params_col,
        value = values, support = supports))
end

# A bare leaf (an uncomposed distribution) inventories its own free parameters
# under an empty edge, so the front-door `build_priors(leaf)` keys the priors
# flat by parameter name.
function params_table(d::Distributions.Distribution)
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
# `Resolve` additionally emits its branch-probability rows; a leaf emits one
# row per scalar parameter. Hand-rolled recursion to stay type-stable.
function _walk_rows!(edges, params_col, values, supports, seen,
        d::Union{Sequential, Parallel}, path)
    names = component_names(d)
    for (name, child) in zip(names, d.components)
        _walk_rows!(edges, params_col, values, supports, seen, child,
            (path..., name))
    end
    return nothing
end

# A `Choose`'s alternatives each contribute their own rows; a tag shared across
# alternatives is deduped via `seen`, so a parameter tied across the index and
# sourced branches is inventoried once.
function _walk_rows!(edges, params_col, values, supports, seen,
        d::Choose, path)
    for (name, alt) in zip(d.names, d.alternatives)
        _walk_rows!(edges, params_col, values, supports, seen, alt,
            (path..., name))
    end
    return nothing
end

function _walk_rows!(edges, params_col, values, supports, seen,
        c::Resolve, path)
    for (name, delay) in zip(c.names, c.delays)
        _is_no_event(delay) && continue
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

# A racing-hazard node emits only its outcome delays' parameter rows; there is
# NO branch-probability block (the winning probability is derived, not free).
function _walk_rows!(edges, params_col, values, supports, seen,
        c::Compete, path)
    for (name, delay) in zip(c.names, c.delays)
        _walk_rows!(edges, params_col, values, supports, seen, delay,
            (path..., name))
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
    sup = (minimum(inner), maximum(inner))
    factor = _thin_factor(leaf)
    # The inner delay params take the delay support; a trailing `:thin` weight
    # takes the `[0, 1]` probability support so its default prior is bounded.
    vals = factor === nothing ? params(inner) : (params(inner)..., factor)
    sups = factor === nothing ? ntuple(_ -> sup, length(vals)) :
           (ntuple(_ -> sup, length(vals) - 1)...,
        (zero(factor), one(factor)))
    edge = tag === nothing ? _join_path(path) : tag
    tag === nothing || push!(seen, tag)
    for (pname, v, s) in zip(pnames, vals, sups)
        push!(edges, edge)
        push!(params_col, pname)
        push!(values, v)
        push!(supports, s)
    end
    return nothing
end

# Join a name path to a single dotted `Symbol` (e.g. `(:a, :b)` -> `:a.b`); a
# single-element path keeps its bare name. This is the DOTTED ("." separator)
# PARAMETER-PATH namespace (params_table edges / priors), distinct from the
# UNDERSCORED ("_" separator) event/value namespace (`_join_value_path`,
# `_split_edge_name`).
_join_path(path::Tuple) = Symbol(join(string.(path), "."))

# --- update: nested NamedTuple -> reconstructed distribution ----------------

# Reconstruct a (possibly censored) leaf from a new inner free delay built out
# of `vals`, re-applying the fixed censoring. Mirrors the extension's
# `_reconstruct_leaf` but is Turing-free; argument checks are kept on (this is
# building a concrete distribution, not a gradient hot path). For a thinned leaf
# the trailing value is the new `thin` weight: it is split off, the inner delay
# rebuilt from the remaining values, and the new factor routed into the op via
# `_set_thin_factor`.
function _update_leaf(leaf, vals::Tuple)
    inner = free_leaf(leaf)
    ctor = Base.typename(typeof(inner)).wrapper
    if _thin_factor(leaf) === nothing
        return rewrap_leaf(leaf, ctor(vals...))
    end
    delay_vals = vals[1:(end - 1)]
    rebuilt = rewrap_leaf(leaf, ctor(delay_vals...))
    return _set_thin_factor(rebuilt, vals[end])
end

@doc "

Update a composed distribution's free parameters from a nested `NamedTuple`.

`update(d, params)` returns a new distribution of the SAME structure as `d` with
its free parameters replaced by the values in `params`. The `params` NamedTuple
mirrors the tree: a [`Sequential`](@ref)/[`Parallel`](@ref) is keyed by its edge
names, a leaf by its parameter names (as in [`params_table`](@ref)'s `param`
column), and a [`Resolve`](@ref) by its outcome names plus an optional
`branch_probs` entry. A censored leaf is transparent: supply only the inner
delay's parameters and the censoring is carried through. A [`thin`](@ref) leaf
takes its delay parameters plus a trailing `thin` weight (as in
[`params_table`](@ref)), routed back into the reporting probability.

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
- [`update`](@ref)`(d, path => new_node)`: replace whole nodes (same shape)
- [`prune`](@ref), [`splice`](@ref): topology edits that change the shape
"
function update(d::Union{Sequential, Parallel, AbstractOneOf, Choose},
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

# A `Choose` updates each alternative; a tag shared across alternatives reads one
# entry from `shared` and is placed in every occurrence.
function _update(d::Choose, params::NamedTuple, shared)
    _check_child_keys(params, d.names, :Choose, shared)
    alts = ntuple(length(d.names)) do i
        _update(d.alternatives[i], _child_params(params, d.names[i]), shared)
    end
    return Choose(d.names, alts, d.selector)
end

function _update(c::Resolve, params::NamedTuple, shared)
    _check_child_keys(params, c.names, :Resolve, shared; optional = (:branch_probs,))
    delays = ntuple(length(c.names)) do i
        _update(c.delays[i], _child_params(params, c.names[i]), shared)
    end
    probs = if haskey(params, :branch_probs)
        bp = params.branch_probs
        _check_update_keys(bp, c.names, Symbol("Resolve branch_probs"))
        ntuple(i -> bp[c.names[i]], length(c.names))
    else
        c.branch_probs
    end
    return Resolve(c.names, delays, probs)
end

# A racing-hazard node updates each outcome delay; there is no `branch_probs`
# block to update (the winning probability is derived).
function _update(c::Compete, params::NamedTuple, shared)
    _check_child_keys(params, c.names, :Compete, shared)
    delays = ntuple(length(c.names)) do i
        _update(c.delays[i], _child_params(params, c.names[i]), shared)
    end
    return Compete(c.names, delays)
end

# A no-event marker carries no parameters, so `update` leaves it unchanged.
_update(d::NoEvent, ::NamedTuple, shared) = d

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
# The DOTTED ("." separator) PARAMETER-PATH namespace (inverse of `_join_path`),
# distinct from the UNDERSCORED event/value namespace (`_split_edge_name`).
function _split_edge(edge::Symbol)
    parts = split(string(edge), '.')
    return Tuple(Symbol.(parts))
end

# The nesting path for an edge in `build_priors`. A bare-leaf row carries an
# empty edge, which nests at the top level (a flat, param-keyed NamedTuple)
# rather than under an empty name segment.
_edge_path(edge::Symbol) = edge === Symbol("") ? () : _split_edge(edge)

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

- a probability parameter, support `[0, 1]` (a `branch_probs` row or a
  [`thin`](@ref) reporting weight) -> `Uniform(0, 1)`.
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

Assemble the nested prior `NamedTuple` for a composed distribution.

`build_priors(tree; priors, default)` is the front-door: it takes a composed
distribution (or a bare leaf) directly, calls [`params_table`](@ref) internally,
and assembles the nested `NamedTuple` that [`composed_parameters_model`](@ref)
(and [`update`](@ref)) expect, so the common path is one call rather than
`build_priors(params_table(tree))`.

`build_priors(table; priors, default)` takes the flat parameter table itself,
for the override workflow where the user inspects or edits the table first. Both
forms share the same keyword surface and assembly rule.

For each row the prior is chosen in order:
1. a user `priors` override for that `(edge, param)`, if present, else
2. `default(row)`, the per-row default (support-derived [`default_prior`](@ref)
   unless a different `default` function is given).

By default every row gets a sensible support-derived prior, so
`build_priors(params_table(tree))` alone yields a complete prior NamedTuple. To
recentre only the parameters you care about (brms-style partial override), edit
the result with [`update`](@ref)`(priors, path => fields)`, addressing each leaf
by the same name path the tree [`update`](@ref) uses.

`row` is a `NamedTuple` `(; edge, param, value, support)` (the table's columns
for that row), so a custom `default` can pick a prior from the parameter's
`support`.

# Arguments
- `tree`: a composed distribution from [`compose`](@ref) (or a bare leaf); the
  front-door form builds its [`params_table`](@ref) internally. Equivalently a
  [`params_table`](@ref) inventory itself (any Tables.jl column table with
  `edge`, `param`, `value`, `support` columns) for the table form.

# Keyword Arguments
- `priors`: per-parameter overrides, either a `(edge, param) => prior` mapping
  (e.g. a `Dict`) or a nested `NamedTuple` keyed like the tree
  (`(onset_admit = (shape = prior,),)`); only the listed parameters are
  overridden (default: empty). Prefer editing the assembled table with
  [`update`](@ref).
- `default`: a function `row -> prior` for rows not overridden (default:
  [`default_prior`](@ref), deriving the prior family from the parameter's
  support).

# Examples
```@example
using CensoredDistributions, Distributions

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
# Front-door: one call straight from the tree, support-derived defaults
# everywhere, then recentre one parameter by path.
nested = update(build_priors(tree),
    :onset_admit => (shape = truncated(Normal(2, 0.5); lower = 0),))
nested.onset_admit.shape
```

# See also
- [`params_table`](@ref): the flat inventory keyed against.
- [`default_prior`](@ref): the support-derived per-row default.
- [`update`](@ref): recentre priors by path on the assembled table.
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
        _nest_insert!(tree, _edge_path(edge), param, prior)
    end
    return _freeze_tree(tree)
end

# Front-door: take a composed distribution (or a bare leaf) directly, build its
# `params_table` internally, then forward to the table method. The whole keyword
# surface (`priors`, `default`) is forwarded unchanged, so the two entry points
# behave identically.
function build_priors(
        tree::Union{Sequential, Parallel, AbstractOneOf, Choose,
            Distributions.Distribution}; kwargs...)
    return build_priors(params_table(tree); kwargs...)
end

@doc "

Build the nested prior `NamedTuple` straight from a composed distribution.

`param_priors(tree; priors, default)` is a thin convenience over
[`build_priors`](@ref)`(`[`params_table`](@ref)`(tree))`: it reads the parameter
inventory of the composed distribution (or bare leaf) `tree` and assembles the
nested prior `NamedTuple` in one call, forwarding the same keyword surface. It
adds no prior logic of its own.

# Arguments
- `tree`: a composed distribution from [`compose`](@ref) (or a bare leaf).

# Keyword Arguments
- `priors`: per-parameter overrides, either a `(edge, param) => prior` mapping
  or a nested `NamedTuple` keyed like the tree; only the listed parameters are
  overridden (default: empty). Prefer editing the result with [`update`](@ref).
- `default`: a function `row -> prior` for rows not overridden (default:
  [`default_prior`](@ref)).

# Examples
```@example
using CensoredDistributions, Distributions

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
priors = param_priors(tree)
priors.onset_admit.shape
```

# See also
- [`build_priors`](@ref): the underlying assembly (table or front-door form).
- [`params_table`](@ref): the parameter inventory read internally.
- [`update`](@ref): recentre individual priors by path.
"
function param_priors(tree; kwargs...)
    return build_priors(tree; kwargs...)
end

# A user override for `(edge, param)`, or `nothing` if none. Accepts a mapping
# keyed by the `(edge, param)` pair (a `Dict`) or a nested `NamedTuple` keyed
# like the tree (descend the edge path, then the param). Missing keys return
# `nothing` so the row falls through to the default.
function _prior_override(priors::NamedTuple, edge::Symbol, param::Symbol)
    node = priors
    for name in _edge_path(edge)
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

# --- update: override fields of a nested prior NamedTuple --------------------
#
# `build_priors` returns the nested prior NamedTuple; `update` edits it in the
# same path convention the tree `update` uses, so a user recentres a handful of
# priors without rebuilding the whole structure or reaching for a flat
# `(edge, param) => prior` Dict.

# Merge `fields` into the leaf prior group at `path`, walking the nested
# NamedTuple by name and rebuilding the spine on the way back up. An empty path
# merges into `node` itself; an unknown name or a path that runs past a leaf
# group errors (mirroring the tree-update `_child_index` behaviour).
function _override_priors(node::NamedTuple, path::Tuple, fields::NamedTuple)
    if isempty(path)
        for f in keys(fields)
            haskey(node, f) || throw(ArgumentError(
                "update(priors, ...): no prior field $(repr(f)); " *
                "have $(collect(keys(node)))"))
        end
        return merge(node, fields)
    end
    name = first(path)
    haskey(node, name) || throw(ArgumentError(
        "update(priors, ...): no prior named $(repr(name)); " *
        "have $(collect(keys(node)))"))
    child = node[name]
    child isa NamedTuple || throw(ArgumentError(
        "update(priors, ...): path runs past prior leaf at $(repr(name))"))
    edited = _override_priors(child, Base.tail(path), fields)
    return merge(node, NamedTuple{(name,)}((edited,)))
end

@doc "

Override prior fields of a [`build_priors`](@ref) table by path.

`update(priors, path => fields, ...)` returns a new nested prior `NamedTuple`
with the per-field priors in each `fields` `NamedTuple` merged in at the leaf
addressed by `path`, leaving every other prior at its [`build_priors`](@ref)
default. A `path` is a `Symbol` (a top-level name) or a tuple of names (a nested
path), the SAME address convention [`update`](@ref)`(d, path => new_node)` uses
on a composed distribution. `fields` is a `NamedTuple` of the parameter priors
to replace at that leaf (e.g. `(shape = prior,)`, `(mu = ..., sigma = ...)`); an
unknown path or field errors clearly.

This replaces the flat `(edge, param) => prior` override Dict. Define the
support-derived defaults with `build_priors(table)`, then recentre only the
priors you care about with `update`.

# Arguments
- `priors`: a nested prior `NamedTuple` from [`build_priors`](@ref).
- `edits`: one or more `path => fields` pairs, `path` a `Symbol` or tuple of
  names and `fields` a `NamedTuple` of per-parameter prior overrides.

# Examples
```@example
using CensoredDistributions, Distributions

inner = compose((death = Gamma(1.5, 1.0), discharge = Gamma(2.0, 1.5)))
tree = compose((admit_resolution = inner, onset_notif = Gamma(2.0, 1.0)))
shape_prior = truncated(Normal(1.0, 1.5); lower = 0.05)
priors = update(build_priors(params_table(tree)),
    (:admit_resolution, :death) => (shape = shape_prior,),
    :onset_notif => (shape = shape_prior,))
priors.onset_notif.shape
```

# See also
- [`build_priors`](@ref): assembles the nested prior NamedTuple this edits.
- [`update`](@ref)`(d, path => new_node)`: the same path convention on a tree.
"
function update(priors::NamedTuple, edits::Pair...)
    out = priors
    for (path, fields) in edits
        out = _override_priors(out, _as_path(path), fields)
    end
    return out
end

# --- name introspection ----------------------------------------------------

@doc "

The per-record key names of a composed distribution.

`event_names(d)` returns the tuple of names keying one drawn/scored record,
EXACTLY the keys of `rand(d)`, `rand(latent(d))`, and the `NamedTuple` a
`logpdf(d, ::NamedTuple)` accepts, in the same order. The names follow the
RECORD SCHEMA the distribution actually realises:

  - A CENSORED composer (one carrying a primary-censoring event) realises the
    flat EVENT path — the root origin event then one target event per leaf edge
    — so its keys are the flat event names. These are derived from the edge
    names (an edge `:onset_admit` gives origin `:onset` and target `:admit`); a
    positional default edge contributes `:event_i`.
  - A PLAIN (uncensored) composer realises one value per leaf edge with no
    latent origin, so its keys are the BRANCH/edge names themselves
    (`compose((a = ..., b = ...))` keys on `(:a, :b)`, a nested chain joining
    with `_`). The edge names are NOT split here: a plain branch named `:a`
    appears as `:a`, matching `keys(rand(d))`.

An inner composer's events are exposed, so a nested `compose((path = [a, b],))`
flattens the inner layout rather than stopping at the `(:path,)` edge. These
record keys are distinct from the nested EDGE/child structure of
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
function event_names(d::Union{Sequential, Parallel})
    return _output_names(d)
end
# A standalone disjunction (`Resolve` / `Compete`) realises the flat event
# record directly (a positional origin slot then one slot per outcome), so its
# record keys ARE the flat event names — `event_names == keys(rand(c))` already.
event_names(c::AbstractOneOf) = _flat_event_names(c)
# A `Choose` has no single flat layout (the active alternative is data-selected),
# so its flat event names are its alternative names.
event_names(d::Choose) = d.names

@doc "

The NESTED tree of event names of a composed distribution.

`event_tree(d)` returns the event-name structure as data: a nested `NamedTuple`
keyed by child name down to the leaves, mirroring the tree. Its FIRST level is
the top-level child names (the old top-level `event_names` result); a
[`Sequential`](@ref)/[`Parallel`](@ref)/[`Choose`](@ref) child recurses to its
own nested NamedTuple, a [`Resolve`](@ref) child to its outcome names, and a
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

function event_tree(c::AbstractOneOf)
    vals = ntuple(i -> _event_tree_child(c.names[i], c.delays[i]),
        length(c.names))
    return NamedTuple{c.names}(vals)
end

function event_tree(d::Choose)
    vals = ntuple(i -> _event_tree_child(d.names[i], d.alternatives[i]),
        length(d.names))
    return NamedTuple{d.names}(vals)
end

# A composer child recurses to its own nested NamedTuple; a leaf is keyed by its
# parent under its own name, so its value is just that name (the leaf event).
function _event_tree_child(
        ::Symbol, c::Union{Sequential, Parallel, AbstractOneOf, Choose})
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

function _event_child(c::AbstractOneOf, name::Symbol)
    idx = findfirst(==(name), c.names)
    idx === nothing && throw(KeyError(name))
    return c.delays[idx]
end

function _event_child(d::Choose, name::Symbol)
    idx = findfirst(==(name), d.names)
    idx === nothing && throw(KeyError(name))
    return d.alternatives[idx]
end

@doc "

Fetch a composed distribution's child (event/edge), or descend a name path.

`event(d, path...)` returns the sub-distribution of `d` at the named location: a
single `Symbol` fetches a direct child (a branch of a [`Parallel`](@ref), a step
of a [`Sequential`](@ref), an outcome delay of a [`Resolve`](@ref), or an
alternative of a [`Choose`](@ref)); multiple `Symbol`s, or a single dotted-path
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
