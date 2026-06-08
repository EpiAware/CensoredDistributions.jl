# ============================================================================
# Prior introspection for composed distributions (#351, Option A)
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
# - `event_names(d)` / `get_event(d, name)`: list the edge/event names of a
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
# model generator (#328) and stays Turing-free.

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

Nested, name-keyed parameters of a composed distribution (#351).

Returns a `NamedTuple` keyed by the node names, each value the `params` of that
child (recursing into nested composers; a leaf delegates to its standard/
extended `Distributions.params`). A `Competing` node contributes a name-keyed
NamedTuple of its outcomes plus a `branch_probs` entry. This nested form is for
prior introspection via [`params_table`](@ref); a composed distribution
reconstructs through [`compose`](@ref), not through `Distribution(params...)`.

See also: [`params_table`](@ref), [`event_names`](@ref), [`get_event`](@ref)
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

# Names for the full `params(leaf)` tuple, padding with positional fallbacks so
# every value has a label even when the family is unmapped or `params` carries
# extra (wrapper) fields.
function _leaf_param_names(leaf)
    vals = params(leaf)
    base = _param_names(leaf)
    n = length(vals)
    return ntuple(n) do i
        i <= length(base) ? base[i] : Symbol(:param_, i)
    end
end

# --- params_table (hand-rolled pre-order walk) -----------------------------

@doc "

Flatten a composed distribution's parameters into a prior-definition table.

`params_table(d)` returns a Tables.jl-compatible column table with one row per
scalar free parameter of the composed distribution `d`, with columns:

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
tbl.edge
```

# See also
- [`params`](@ref): the nested name-keyed values
- [`event_names`](@ref), [`get_event`](@ref): name introspection
"
function params_table(d::Union{Sequential, Parallel, Competing})
    edges = Symbol[]
    params_col = Symbol[]
    values = Any[]
    supports = Any[]
    _walk_rows!(edges, params_col, values, supports, d, ())
    return (edge = edges, param = params_col,
        value = values, support = supports)
end

# Pre-order walk over the composer tree. `path` is the tuple of names from the
# root to the current node. A composer recurses into its named children; a
# `Competing` additionally emits its branch-probability rows; a leaf emits one
# row per scalar parameter. Hand-rolled (not AbstractTrees) to stay type-stable.
function _walk_rows!(edges, params_col, values, supports,
        d::Union{Sequential, Parallel}, path)
    names = component_names(d)
    for (name, child) in zip(names, d.components)
        _walk_rows!(edges, params_col, values, supports, child, (path..., name))
    end
    return nothing
end

function _walk_rows!(edges, params_col, values, supports, c::Competing, path)
    for (name, delay) in zip(c.names, c.delays)
        _walk_rows!(edges, params_col, values, supports, delay, (path..., name))
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

# Leaf distribution: one row per scalar parameter, all sharing the leaf's
# variate support.
function _walk_rows!(edges, params_col, values, supports, leaf, path)
    pnames = _leaf_param_names(leaf)
    vals = params(leaf)
    sup = (minimum(leaf), maximum(leaf))
    edge = _join_path(path)
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

# --- name introspection ----------------------------------------------------

@doc "

List the event/edge names of a composed distribution (#351).

`event_names(d)` returns the tuple of top-level child names: branch names for a
[`Parallel`](@ref), step names for a [`Sequential`](@ref), outcome names for a
[`Competing`](@ref). Pair with [`get_event`](@ref) to fetch a child by name and
[`params_table`](@ref) to see its parameters.

# Examples
```@example
using CensoredDistributions, Distributions

tree = compose((onset_admit = LogNormal(1.5, 0.4),
    admit_death = Gamma(2.0, 1.0)))
event_names(tree)
```

# See also
- [`get_event`](@ref): fetch a child by name
- [`params_table`](@ref): the parameter table
"
event_names(d::Union{Sequential, Parallel, Competing}) = component_names(d)

@doc "

Fetch a composed distribution's child (event/edge) by name (#351).

`get_event(d, name)` returns the child distribution labelled `name` (a branch of
a [`Parallel`](@ref), a step of a [`Sequential`](@ref), or an outcome delay of a
[`Competing`](@ref)). Throws a `KeyError` if `name` is not one of
[`event_names`](@ref).

# Arguments
- `d`: the composed distribution to look up a child of.
- `name`: the edge/event name (`Symbol`) of the child to fetch.

# Examples
```@example
using CensoredDistributions, Distributions

tree = compose((onset_admit = LogNormal(1.5, 0.4),
    admit_death = Gamma(2.0, 1.0)))
get_event(tree, :admit_death)
```

# See also
- [`event_names`](@ref): list the names
"
function get_event(d::Union{Sequential, Parallel}, name::Symbol)
    names = component_names(d)
    idx = findfirst(==(name), names)
    idx === nothing && throw(KeyError(name))
    return d.components[idx]
end

function get_event(c::Competing, name::Symbol)
    idx = findfirst(==(name), c.names)
    idx === nothing && throw(KeyError(name))
    return c.delays[idx]
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
