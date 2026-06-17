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
# scores (`logpdf`) and `rand`s. The walk is hand-rolled and type-stable,
# reusing `component_names` / `_rebuild` / `Competing` / `Select` rather
# than adding tree types. Distributions-led: a node-to-node edit, Turing-free.
#
# Paths are tuples of edge names from the root (e.g.
# `(:admit_path, :admit_resolution, :death)`); a single `Symbol` is a one-step
# path. The op family mirrors the `apply(op, node)` shape from the
# node-transform protocol: a public function resolves a path then applies an op.

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
        "edit path runs past a leaf at $(repr(first(path))); " *
        "$(nameof(typeof(leaf))) has no named children"))
end

# Index of `name` in a node's name tuple, erroring clearly when absent.
function _child_index(names::Tuple, name::Symbol, what)
    idx = findfirst(==(name), names)
    idx === nothing && throw(ArgumentError(
        "edit($what, ...): no child named $(repr(name)); " *
        "have $(collect(names))"))
    return idx
end

# Normalise an edit-verb address to a name-path tuple, accepting the SAME forms
# as the read-side [`event`](@ref): a bare `Symbol` (a one-step path), a dotted
# `Symbol` (`:a.b`, split via `_split_edge` into `(:a, :b)`), or a tuple of edge
# names. So `event(d, addr)` reads and `update(d, addr => x)` writes the same
# address.
_as_path(p::Symbol) = _split_edge(p)
_as_path(p::Tuple) = p

# --- update: replace a node (structural edit) -------------------------------
#
# `update` is the single verb for both kinds of SHAPE-PRESERVING edit: a value
# update keyed by a nested NamedTuple (the `update(d, ::NamedTuple)` method in
# `introspection.jl`) and node replacement keyed by `path => new_node` pairs
# (here). The two methods dispatch on the second argument at the public
# boundary, so the type-stable scoring path (the NamedTuple worker `_update`) is
# untouched by the structural-edit path.

@doc "

Replace named nodes of a composed distribution with new distributions.

`update(d, path => new_node, ...)` returns a new composed distribution of the
SAME outer structure as `d` with the node addressed by each `path` replaced by
`new_node`. A `path` is a `Symbol` (a top-level child), a dotted `Symbol`
(`:admit_path.admit_resolution.death`, as in [`event`](@ref) /
[`params_table`](@ref)), or a tuple of edge names from the root (e.g.
`(:admit_path, :admit_resolution, :death)`); the same address [`event`](@ref)
READS is the one this WRITES. `new_node` may be a leaf distribution or a nested
composer. This shares the recursive reconstruction with the value-update method
[`update`](@ref)`(d, params::NamedTuple)`, so the result scores and `rand`s. It
preserves the tree SHAPE; for shape changes use [`prune`](@ref) or
[`splice`](@ref).

# Arguments
- `d`: the composed distribution to edit.
- `edits`: one or more `path => new_node` pairs.

# Examples
```@example
using CensoredDistributions, Distributions

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
tree2 = update(tree, :admit_death => Gamma(3.0, 1.5))
event(tree2, :admit_death)
```

# See also
- [`prune`](@ref): drop a `Competing` arm or `Select` alternative (changes shape)
- [`splice`](@ref): insert a before/after step at a node (changes shape)
- [`update`](@ref)`(d, params::NamedTuple)`: replace free parameter values
"
function update(d::Union{Sequential, Parallel, Competing, Select},
        edits::Pair...)
    out = d
    for (path, new_node) in edits
        out = _edit_at(out, _as_path(path), _ -> new_node)
    end
    return out
end

# `intervene` was the node-replace verb; it is now the `update(d, ::Pair...)`
# method. `swap_child` was sugar (parent path + child name); rebuild the full
# path and call `update`.
@deprecate intervene(d::Union{Sequential, Parallel, Competing, Select},
    edits::Pair...) update(d, edits...)

@doc "
    intervene(d, edits...)

Deprecated alias of [`update`](@ref)`(d, edits...)`. Use `update` instead.

# Arguments
- `d`: the composed distribution to edit.
- `edits`: one or more `path => new_node` pairs.

# Examples
```@example
using CensoredDistributions, Distributions

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
update(tree, :admit_death => Gamma(3.0, 1.5))
```

# See also
- [`update`](@ref): the current verb.
" intervene

@doc "
    swap_child(d, parent_path, edit)

Deprecated alias for replacing a child by parent path. Use
[`update`](@ref)`(d, (parent_path..., name) => new)` instead.

# Arguments
- `d`: the composed distribution to edit.
- `parent_path`: the path to the parent node.
- `edit`: a `name => new_node` pair naming the child to replace.

# Examples
```@example
using CensoredDistributions, Distributions

tree = compose((resolution = compose((death = Gamma(1.5, 1.0),)),))
update(tree, (:resolution, :death) => Gamma(3.0, 1.5))
```

# See also
- [`update`](@ref): the current verb.
"
function swap_child(d::Union{Sequential, Parallel, Competing, Select},
        parent_path, edit::Pair)
    Base.depwarn(
        "`swap_child(d, parent_path, name => new)` is deprecated; use " *
        "`update(d, (parent_path..., name) => new)`.", :swap_child)
    name, new_node = edit
    full = (_as_path(parent_path)..., name)
    return update(d, full => new_node)
end

# --- prune: drop a Competing arm / Select alternative / step ----------------

@doc "

Drop a branch from a composed distribution (a topology edit).

`prune(d, path)` removes the node addressed by `path` from its parent, CHANGING
the tree shape. A [`Competing`](@ref) arm is removed and the remaining branch
probabilities are renormalised to sum to one; a [`Select`](@ref) alternative or
a [`Sequential`](@ref)/[`Parallel`](@ref) step is removed. The parent must keep
at least the minimum number of children (two for `Competing`/`Select`, one for
`Sequential`/`Parallel`). The result is a valid composed distribution that
scores and `rand`s. `path` accepts the same forms as [`event`](@ref): varargs
`Symbol`s, a dotted `Symbol`, or a tuple of edge names.

`prune` and [`splice`](@ref) are the two topology edits (they change the tree
shape); [`update`](@ref) keeps the same shape and replaces contents.

# Arguments
- `d`: the composed distribution to edit.
- `path`: the branch to drop, as varargs `Symbol`s, a dotted `Symbol`, or a
  tuple of edge names.

# Examples
```@example
using CensoredDistributions, Distributions

node = competing(:death => (Gamma(1.5, 1.0), 0.3),
    :disch => (Gamma(2.0, 1.5), 0.5),
    :transfer => (Gamma(1.0, 1.0), 0.2))
tree = compose((resolution = node, onset = Gamma(1.0, 1.0)))
tree2 = prune(tree, :resolution, :transfer)
event_names(event(tree2, :resolution))
```

# See also
- [`splice`](@ref): insert a before/after step at a node (the other topology edit)
- [`update`](@ref): replace a node or its values (keeps the shape)
"
function prune(d::Union{Sequential, Parallel, Competing, Select}, path::Symbol)
    # A single `Symbol` goes through `_as_path` (so a dotted `:a.b` splits); two
    # or more `Symbol`s are the literal varargs path.
    return _prune_path(d, _as_path(path))
end

function prune(d::Union{Sequential, Parallel, Competing, Select},
        name1::Symbol, name2::Symbol, rest::Symbol...)
    return _prune_path(d, (name1, name2, rest...))
end

function prune(d::Union{Sequential, Parallel, Competing, Select}, path::Tuple)
    return _prune_path(d, _as_path(path))
end

function _prune_path(d, p::Tuple)
    isempty(p) && throw(ArgumentError("prune needs a non-empty path"))
    parent_path = p[1:(end - 1)]
    name = p[end]
    return _edit_at(d, parent_path, parent -> _drop_child(parent, name))
end

# `cut_branch` was the drop-a-branch verb; it is now `prune`.
@deprecate cut_branch(d::Union{Sequential, Parallel, Competing, Select}, path) prune(
    d, path)

@doc "
    cut_branch(d, path)

Deprecated alias of [`prune`](@ref)`(d, path)`. Use `prune` instead.

# Arguments
- `d`: the composed distribution to edit.
- `path`: the branch to drop, as a `Symbol`, dotted `Symbol`, or tuple of edge
  names.

# Examples
```@example
using CensoredDistributions, Distributions

node = competing(:death => (Gamma(1.5, 1.0), 0.3),
    :disch => (Gamma(2.0, 1.5), 0.5),
    :transfer => (Gamma(1.0, 1.0), 0.2))
tree = compose((resolution = node, onset = Gamma(1.0, 1.0)))
prune(tree, :resolution, :transfer)
```

# See also
- [`prune`](@ref): the current verb.
" cut_branch

# Remove the child `name` from a composer, rebuilding the node without it.
function _drop_child(d::Union{Sequential, Parallel}, name::Symbol)
    names = component_names(d)
    idx = _child_index(names, name, nameof(typeof(d)))
    length(names) >= 2 || throw(ArgumentError(
        "prune: $(nameof(typeof(d))) needs at least one remaining child"))
    keep = filter(!=(idx), 1:length(names))
    parts = Tuple(d.components[i] for i in keep)
    kept_names = Tuple(names[i] for i in keep)
    return _rebuild_named(d, parts, kept_names)
end

function _drop_child(c::Competing, name::Symbol)
    idx = _child_index(c.names, name, :Competing)
    length(c.names) >= 3 || throw(ArgumentError(
        "prune: Competing needs at least two remaining outcomes"))
    keep = filter(!=(idx), 1:length(c.names))
    kept_probs = Tuple(c.branch_probs[i] for i in keep)
    total = sum(kept_probs)
    total > 0 || throw(ArgumentError(
        "prune: remaining Competing branch probabilities sum to zero"))
    probs = map(p -> p / total, kept_probs)
    return Competing(Tuple(c.names[i] for i in keep),
        Tuple(c.delays[i] for i in keep), probs)
end

function _drop_child(d::Select, name::Symbol)
    idx = _child_index(d.names, name, :Select)
    length(d.names) >= 3 || throw(ArgumentError(
        "prune: Select needs at least two remaining alternatives"))
    keep = filter(!=(idx), 1:length(d.names))
    return Select(Tuple(d.names[i] for i in keep),
        Tuple(d.alternatives[i] for i in keep), d.selector)
end

function _drop_child(leaf, name::Symbol)
    throw(ArgumentError(
        "prune: $(nameof(typeof(leaf))) has no child to drop"))
end

# `_rebuild` taking explicit names (a dropped child changes the name set).
function _rebuild_named(::Sequential, parts::Tuple, names::Tuple)
    return Sequential(parts, names)
end
function _rebuild_named(::Parallel, parts::Tuple, names::Tuple)
    return Parallel(parts, names)
end

# --- splice: insert a before/after step at a node ---------------------------

@doc "

Splice before/after steps around a node in a composed distribution (a topology
edit).

`splice(d, path; before, after)` replaces the node at `path` with a
[`Sequential`](@ref) chain of `before`, the original node, then `after` (any of
which may be omitted). This inserts a change-point step around the addressed
node without rebuilding the rest of the tree, e.g. an extra delay before a
branch or a follow-up step after it, CHANGING the tree shape. The result is a
valid composed distribution that scores and `rand`s. `path` accepts the same
forms as [`event`](@ref): varargs `Symbol`s, a dotted `Symbol`, or a tuple of
edge names.

`splice` and [`prune`](@ref) are the two topology edits (they change the tree
shape); [`update`](@ref) keeps the same shape and replaces contents.

# Arguments
- `d`: the composed distribution to edit.
- `path`: the node to wrap, as varargs `Symbol`s, a dotted `Symbol`, or a tuple
  of edge names.

# Keyword Arguments
- `before`: a `name => dist` step inserted before the node (default: none).
- `after`: a `name => dist` step inserted after the node (default: none).

# Examples
```@example
using CensoredDistributions, Distributions

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
tree2 = splice(tree, :admit_death; after = :death_report => Gamma(1.0, 2.0))
event_names(event(tree2, :admit_death))
```

# See also
- [`prune`](@ref): drop a branch (the other topology edit)
- [`update`](@ref): replace a node or its values (keeps the shape)
"
function splice(d::Union{Sequential, Parallel, Competing, Select},
        path::Symbol; before = nothing, after = nothing)
    return _splice_path(d, _as_path(path); before, after)
end

function splice(d::Union{Sequential, Parallel, Competing, Select},
        name1::Symbol, name2::Symbol, rest::Symbol...;
        before = nothing, after = nothing)
    return _splice_path(d, (name1, name2, rest...); before, after)
end

function splice(d::Union{Sequential, Parallel, Competing, Select},
        path::Tuple; before = nothing, after = nothing)
    return _splice_path(d, _as_path(path); before, after)
end

function _splice_path(d, p::Tuple; before, after)
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
