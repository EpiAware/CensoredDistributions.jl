# ============================================================================
# Intervention: structural edits on a composed distribution tree
# ============================================================================
#
# `update` rebuilds a tree's LEAVES with new parameter values; intervention is
# the same recursive reconstruction with a node EDIT instead of a param update:
# replace a named node's distribution, swap a child, cut a branch (a `Competing`
# arm or a `Select` alternative, renormalising probs), or splice a before/after
# change at a named node. Each op walks the tree by name path and rebuilds only
# the touched spine, so the result is a fresh, valid composed distribution that
# scores (`logpdf`) and `rand`s. The walk is hand-rolled and type-stable, reusing
# the `component_names` / `_rebuild` / `Competing` / `Select` machinery rather
# than adding tree types. Distributions-led: a node-to-node edit, Turing-free.
#
# Paths are tuples of edge names from the root (e.g.
# `(:admit_path, :admit_resolution, :death)`); a single `Symbol` is a one-step
# path. The op family mirrors the `apply(op, node)` shape from the node-transform
# protocol: each public function resolves a path then applies a leaf op.

# --- path-walk core ---------------------------------------------------------
#
# `_edit_at(node, path, op)` walks `path` from `node`, applying `op(target)` at
# the addressed node and rebuilding the spine on the way back up. An empty path
# applies `op` to `node` itself; otherwise `_edit_step` dispatches on the
# composer type to find the named child, recurse, and rebuild with the edited
# child swapped in.

function _edit_at(node, path::Tuple, op)
    isempty(path) && return op(node)
    return _edit_step(node, path, op)
end

function _edit_step(d::Union{Sequential, Parallel}, path::Tuple, op)
    names = component_names(d)
    idx = _child_index(names, first(path), nameof(typeof(d)))
    parts = ntuple(length(names)) do i
        i == idx ? _edit_at(d.components[i], Base.tail(path), op) :
        d.components[i]
    end
    return _rebuild(d, parts)
end

function _edit_step(c::Competing, path::Tuple, op)
    idx = _child_index(c.names, first(path), :Competing)
    delays = ntuple(length(c.names)) do i
        i == idx ? _edit_at(c.delays[i], Base.tail(path), op) : c.delays[i]
    end
    return Competing(c.names, delays, c.branch_probs)
end

function _edit_step(d::Select, path::Tuple, op)
    idx = _child_index(d.names, first(path), :Select)
    alts = ntuple(length(d.names)) do i
        i == idx ? _edit_at(d.alternatives[i], Base.tail(path), op) :
        d.alternatives[i]
    end
    return Select(d.names, alts, d.selector)
end

# A leaf has no children: a non-empty path into it is an error.
function _edit_step(leaf, path::Tuple, op)
    throw(ArgumentError(
        "intervene path runs past a leaf at $(repr(first(path))); " *
        "$(nameof(typeof(leaf))) has no named children"))
end

# Index of `name` in a node's name tuple, erroring clearly when absent.
function _child_index(names::Tuple, name::Symbol, what)
    idx = findfirst(==(name), names)
    idx === nothing && throw(ArgumentError(
        "intervene($what, ...): no child named $(repr(name)); " *
        "have $(collect(names))"))
    return idx
end

_as_path(p::Symbol) = (p,)
_as_path(p::Tuple) = p

# --- intervene: replace a node ----------------------------------------------

@doc "

Replace named nodes of a composed distribution with new distributions.

`intervene(d, path => new_node, ...)` returns a new composed distribution of the
same outer structure as `d` with the node addressed by each `path` replaced by
`new_node`. A `path` is a `Symbol` (a top-level child) or a tuple of edge names
from the root (e.g. `(:admit_path, :admit_resolution, :death)`); `new_node` may
be a leaf distribution or a nested composer. This is the node-replace
intervention: a structural edit reusing the same recursive reconstruction as
[`update`](@ref), so the result scores and `rand`s.

# Arguments
- `d`: the composed distribution to edit.
- `edits`: one or more `path => new_node` pairs.

# Examples
```@example
using CensoredDistributions, Distributions

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
tree2 = intervene(tree, :admit_death => Gamma(3.0, 1.5))
get_event(tree2, :admit_death)
```

# See also
- [`cut_branch`](@ref): drop a `Competing` arm or `Select` alternative
- [`splice`](@ref): insert a before/after step at a node
- [`update`](@ref): replace free parameters rather than whole nodes
"
function intervene(d::Union{Sequential, Parallel, Competing, Select},
        edits::Pair...)
    out = d
    for (path, new_node) in edits
        out = _edit_at(out, _as_path(path), _ -> new_node)
    end
    return out
end

@doc "

Swap a named child of a node for a new distribution.

`swap_child(d, parent_path, name => new_node)` replaces the child `name` of the
node at `parent_path` with `new_node`. Sugar over [`intervene`](@ref) with the
child name appended to the parent path; `parent_path` is `()` for the root.

# Arguments
- `d`: the composed distribution to edit.
- `parent_path`: path (a `Symbol`, tuple, or `()`) to the node whose child is
  swapped.
- `edit`: a `name => new_node` pair naming the child and its replacement.

# Examples
```@example
using CensoredDistributions, Distributions

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
tree2 = swap_child(tree, (), :onset_admit => Gamma(4.0, 1.0))
get_event(tree2, :onset_admit)
```

# See also
- [`intervene`](@ref): replace a node addressed by full path
"
function swap_child(d::Union{Sequential, Parallel, Competing, Select},
        parent_path, edit::Pair)
    name, new_node = edit
    full = (_as_path(parent_path)..., name)
    return intervene(d, full => new_node)
end

# --- cut_branch: drop a Competing arm / Select alternative / step -----------

@doc "

Drop a branch from a composed distribution.

`cut_branch(d, path)` removes the node addressed by `path` from its parent. A
[`Competing`](@ref) arm is removed and the remaining branch probabilities are
renormalised to sum to one; a [`Select`](@ref) alternative or a
[`Sequential`](@ref)/[`Parallel`](@ref) step is removed. The parent must keep at
least the minimum number of children (two for `Competing`/`Select`, one for
`Sequential`/`Parallel`). The result is a valid composed distribution that
scores and `rand`s.

# Arguments
- `d`: the composed distribution to edit.
- `path`: path (a `Symbol` or tuple of edge names) to the branch to drop.

# Examples
```@example
using CensoredDistributions, Distributions

node = competing(:death => (Gamma(1.5, 1.0), 0.3),
    :disch => (Gamma(2.0, 1.5), 0.5),
    :transfer => (Gamma(1.0, 1.0), 0.2))
tree = compose((resolution = node, onset = Gamma(1.0, 1.0)))
tree2 = cut_branch(tree, (:resolution, :transfer))
event_names(get_event(tree2, :resolution))
```

# See also
- [`intervene`](@ref): replace a node rather than drop it
"
function cut_branch(d::Union{Sequential, Parallel, Competing, Select}, path)
    p = _as_path(path)
    isempty(p) && throw(ArgumentError("cut_branch needs a non-empty path"))
    parent_path = p[1:(end - 1)]
    name = p[end]
    return _edit_at(d, parent_path, parent -> _drop_child(parent, name))
end

# Remove the child `name` from a composer, rebuilding the node without it.
function _drop_child(d::Union{Sequential, Parallel}, name::Symbol)
    names = component_names(d)
    idx = _child_index(names, name, nameof(typeof(d)))
    length(names) >= 2 || throw(ArgumentError(
        "cut_branch: $(nameof(typeof(d))) needs at least one remaining child"))
    keep = filter(!=(idx), 1:length(names))
    parts = Tuple(d.components[i] for i in keep)
    kept_names = Tuple(names[i] for i in keep)
    return _rebuild_named(d, parts, kept_names)
end

function _drop_child(c::Competing, name::Symbol)
    idx = _child_index(c.names, name, :Competing)
    length(c.names) >= 3 || throw(ArgumentError(
        "cut_branch: Competing needs at least two remaining outcomes"))
    keep = filter(!=(idx), 1:length(c.names))
    kept_probs = Tuple(c.branch_probs[i] for i in keep)
    total = sum(kept_probs)
    total > 0 || throw(ArgumentError(
        "cut_branch: remaining Competing branch probabilities sum to zero"))
    probs = map(p -> p / total, kept_probs)
    return Competing(Tuple(c.names[i] for i in keep),
        Tuple(c.delays[i] for i in keep), probs)
end

function _drop_child(d::Select, name::Symbol)
    idx = _child_index(d.names, name, :Select)
    length(d.names) >= 3 || throw(ArgumentError(
        "cut_branch: Select needs at least two remaining alternatives"))
    keep = filter(!=(idx), 1:length(d.names))
    return Select(Tuple(d.names[i] for i in keep),
        Tuple(d.alternatives[i] for i in keep), d.selector)
end

function _drop_child(leaf, name::Symbol)
    throw(ArgumentError(
        "cut_branch: $(nameof(typeof(leaf))) has no child to drop"))
end

# `_rebuild` taking explicit names (the dropped-child case changes the name set).
_rebuild_named(::Sequential, parts::Tuple, names::Tuple) = Sequential(parts, names)
_rebuild_named(::Parallel, parts::Tuple, names::Tuple) = Parallel(parts, names)

# --- splice: insert a before/after step at a node ---------------------------

@doc "

Splice before/after steps around a node in a composed distribution.

`splice(d, path; before, after)` replaces the node at `path` with a
[`Sequential`](@ref) chain of `before`, the original node, then `after` (any of
which may be omitted). This inserts a change-point step around the addressed node
without rebuilding the rest of the tree, e.g. an extra delay before a branch or a
follow-up step after it. The result is a valid composed distribution that scores
and `rand`s.

# Arguments
- `d`: the composed distribution to edit.
- `path`: path (a `Symbol` or tuple of edge names) to the node to wrap.

# Keyword Arguments
- `before`: a `name => dist` step inserted before the node (default: none).
- `after`: a `name => dist` step inserted after the node (default: none).

# Examples
```@example
using CensoredDistributions, Distributions

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
tree2 = splice(tree, :admit_death; after = :death_report => Gamma(1.0, 2.0))
event_names(get_event(tree2, :admit_death))
```

# See also
- [`intervene`](@ref): replace a node outright
"
function splice(d::Union{Sequential, Parallel, Competing, Select}, path;
        before = nothing, after = nothing)
    p = _as_path(path)
    isempty(p) && throw(ArgumentError("splice needs a non-empty path"))
    name = p[end]
    (before === nothing && after === nothing) && throw(ArgumentError(
        "splice needs a `before` and/or `after` step"))
    return _edit_at(d, p, node -> _spliced(node, name, before, after))
end

# Build the spliced Sequential around `node`, naming the original step `name`.
function _spliced(node, name::Symbol, before, after)
    pre = before === nothing ? () : (before,)
    post = after === nothing ? () : (after,)
    steps = (pre..., name => node, post...)
    dists = Tuple(s.second for s in steps)
    names = Tuple(s.first for s in steps)
    return Sequential(dists, names)
end
