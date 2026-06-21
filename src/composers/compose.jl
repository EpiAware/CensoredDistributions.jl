# ============================================================================
# compose: the friendly front-end constructor for the composer stack
# ============================================================================
#
# `compose` is a CONSTRUCTOR over the [`Sequential`](@ref) / [`Parallel`](@ref)
# composers: it does NOT introduce a new monolithic tree type. Three
# friendly inputs all lower to the SAME nested composer stack:
#
# - a `NamedTuple` (named, recursive): a `Parallel` over the named children; a
#   child that is itself a `NamedTuple` nests as a `Parallel`, a child that is a
#   `Vector`/`Tuple` of distributions nests as a `Sequential`, a bare
#   distribution is a leaf;
# - a Tables.jl table with `name` and `dist` columns: a `Parallel` over the rows
#   (the column-table equivalent of a flat `NamedTuple`);
# - a nested `Matrix` of distributions: rows are `Parallel` branches and the
#   columns within a row are `Sequential` steps (a branching grid).
#
# The mappings are chosen so the three inputs build identical stacks for the
# same structure, which the tests assert by `==` on the composed objects.

@doc "

Build a nested composer stack from a friendly front-end input.

`compose` lowers a NamedTuple, a Tables.jl table, or a nested matrix to the same
[`Sequential`](@ref) / [`Parallel`](@ref) stack. It is a constructor over the
composers, not a new tree type.

# Arguments
- `input`: the front-end to lower, one of the three forms below.

# Inputs

- `NamedTuple` (named, recursive): a [`Parallel`](@ref) over the named children.
  A child that is itself a `NamedTuple` nests as a `Parallel`, a child that is a
  `Vector` or `Tuple` of distributions nests as a [`Sequential`](@ref), and a
  bare `UnivariateDistribution` is a leaf branch. The same flat structure may be
  given as `:name => child` Pairs (`compose(:a => d1, :b => d2)`) for data-
  driven or computed branch names; the two spellings build the same stack.
- Tables.jl table with `name` and `dist` columns: a [`Parallel`](@ref) over the
  rows, the column-table equivalent of a flat `NamedTuple`. An optional `chain`
  column folds rows sharing a non-zero group id into a [`Sequential`](@ref)
  branch, and an optional `compete`/`prob` column pair folds rows sharing a
  non-zero `compete` id into a [`Resolve`](@ref) node whose `prob` entries are
  the branch probabilities (each in ``[0, 1]`` and summing to one per group).
- nested `Matrix` of distributions: rows are [`Parallel`](@ref) branches and the
  columns within a row are [`Sequential`](@ref) steps. This orientation is
  canonical, so a one-column matrix is parallel leaf branches (one row each) and
  a one-row matrix is a single [`Sequential`](@ref) chain (the lone row's
  columns), not a `Parallel`.

# Contract

`compose` ALWAYS returns a composer, never a bare univariate leaf.
A single branch stays a [`Parallel`](@ref)-of-one and a single step a
one-element [`Sequential`](@ref); the wrapper is never collapsed away.
A bare leaf is used directly at the SCORING layer, where
[`record_distributions`](@ref) and [`composed_distribution_model`](@ref) accept
a bare `UnivariateDistribution`, so callers do not need `compose` to pass one
through.

# Examples
```@example
using CensoredDistributions, Distributions

# A regular 2x2 grid built three ways, all equal.
nt = (r1 = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4)],
    r2 = [Gamma(1.0, 1.0), Gamma(3.0, 1.0)])
table = (name = [:a, :b, :c, :d],
    dist = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4),
        Gamma(1.0, 1.0), Gamma(3.0, 1.0)],
    chain = [1, 1, 2, 2])
mat = [Gamma(2.0, 1.0) LogNormal(0.5, 0.4); Gamma(1.0, 1.0) Gamma(3.0, 1.0)]
compose(nt) == compose(table) == compose(mat)
```

# See also
- [`Sequential`](@ref), [`Parallel`](@ref), [`Resolve`](@ref): the composers
"
function compose end

# --- NamedTuple front-end --------------------------------------------------
# A NamedTuple maps to a Parallel over its values, each value lowered by
# `_compose_child`. The keys become the branch NAMES, threaded into the
# `Parallel` so `params`/`params_table`/`show` are name-keyed (Option A).
# Structurally this still matches the table and matrix forms (`==` ignores
# names); only the labels differ.
#
# A column table is also a NamedTuple, so a NamedTuple carrying `name`/`dist`
# column vectors is routed to the Tables.jl path instead, letting one
# `(name, dist, chain)` column table build the same stack as a structural
# NamedTuple.
function compose(nt::NamedTuple)
    _is_column_table(nt) && return _compose_table(nt)
    children = map(_compose_child, Tuple(nt))
    return Parallel(children, keys(nt))
end

# --- Pairs front-end -------------------------------------------------------
# `compose(:a => d1, :b => d2)` is the Pairs spelling of the NamedTuple
# front-end, for data-driven or computed branch names (a NamedTuple needs
# literal field names). It lowers to a NamedTuple and the same `Parallel` build,
# so the two spellings are `==`. Each pair's name must be a Symbol.
function compose(branch1::Pair, branches::Pair...)
    all_pairs = (branch1, branches...)
    names = Tuple(b.first for b in all_pairs)
    all(n -> n isa Symbol, names) ||
        throw(ArgumentError("each compose branch name must be a Symbol"))
    dists = Tuple(b.second for b in all_pairs)
    return compose(NamedTuple{names}(dists))
end

# --- shared-origin front-end ----------------------------------------------
# `compose(origin; branch = ...)` shares `origin` across the named branches: the
# branches fan out from one origin, so the result is a Sequential whose last step
# is a Parallel of the branch tails. Convolving the stack returns one series per
# branch, each delayed by `origin` convolved with the branch tail (e.g. a shared
# incubation, then a reporting branch and a death branch).
function compose(origin::Union{UnivariateDistribution, Sequential, Parallel,
            Choose};
        branches...)
    isempty(branches) &&
        throw(ArgumentError("compose(origin; branches...) needs ≥1 branch"))
    nt = NamedTuple(branches)
    tails = map(_compose_child, Tuple(nt))
    return Sequential((_compose_child(origin), Parallel(tails, keys(nt))))
end

# A NamedTuple is treated as a column table when it has `name` and `dist`
# fields that are both vectors (the column-table shape), AND those vectors
# carry the column ROLES of a real table: the `:dist` column holds
# distributions and the `:name` column holds row LABELS (not distributions).
# This disambiguates a genuine `(name, dist)` table from a structural
# NamedTuple whose user-chosen branch keys happen to be `:name`/`:dist`
# carrying distribution vectors, e.g. `(name = [d1, d2], dist = [d3, d4])` —
# two named chain branches, not a table.
function _is_column_table(nt::NamedTuple)
    haskey(nt, :name) && haskey(nt, :dist) &&
        nt.name isa AbstractVector && nt.dist isa AbstractVector &&
        all(d -> d isa UnivariateDistribution, nt.dist) &&
        !any(n -> n isa UnivariateDistribution, nt.name)
end

# Lower a single front-end value to a composer child. A nested NamedTuple
# recurses (carrying its own keys); a bare vector/tuple of composables becomes a
# Sequential with default `:step_i` names (a plain vector has no names to carry).
# A pre-built composer value (Sequential/Parallel) drops in unchanged, so a
# `compose(...)` result nests as a child and a `Sequential((...), names)` value
# keeps readable step names. A `Resolve` is a UnivariateDistribution leaf and is
# covered by the first method.
_compose_child(d::UnivariateDistribution) = d
_compose_child(c::Union{Sequential, Parallel, Choose}) = c
_compose_child(nt::NamedTuple) = compose(nt)
function _compose_child(v::Union{AbstractVector, Tuple})
    all(_is_composable, v) ||
        throw(ArgumentError(
            "a sequential child must hold UnivariateDistributions or " *
            "composers"))
    return Sequential(map(_compose_child, Tuple(v)))
end

# --- nested Matrix front-end -----------------------------------------------
# A matrix maps to a Parallel over its rows (branches), each row a Sequential
# over its columns (chain steps). A row with a single entry collapses to that
# bare leaf, so a one-column matrix is parallel leaf branches (one row each)
# and a one-row matrix is a single Sequential chain (the lone row's columns),
# matching the NamedTuple/table forms for the same structure.
#
# Names thread through optional keyword arguments (Option A): `names`
# labels the row branches and `step_names` labels the columns within each
# multi-step row. Both fall back to positional defaults (`:branch_i` /
# `:step_j`) when omitted, so the matrix form still works name-free.
function compose(m::AbstractMatrix{<:UnivariateDistribution};
        names = nothing, step_names = nothing)
    nrows, ncols = size(m)
    (nrows >= 1 && ncols >= 1) ||
        throw(ArgumentError("the matrix needs at least one row and column"))
    branch_names = _coerce_names(names, :branch, nrows)
    col_names = ncols == 1 ? nothing : _coerce_names(step_names, :step, ncols)
    branches = ntuple(nrows) do i
        steps = Tuple(m[i, j] for j in 1:ncols)
        ncols == 1 ? steps[1] : Sequential(steps, col_names)
    end
    return Parallel(branches, branch_names)
end

# --- Tables.jl table front-end ---------------------------------------------
# A table with `name` and `dist` columns maps to a Parallel over its rows, the
# column-table equivalent of a flat NamedTuple of leaves. An optional `chain`
# column groups consecutive rows that share a non-zero group id into one
# Sequential branch, so a table can also express the nested chain a NamedTuple
# encodes with a vector value. An optional `compete`/`prob` column pair folds
# the rows sharing a non-zero `compete` group id into one `Resolve` node (the
# `prob` entries its branch probabilities), so the table can also express a
# one_of-outcome set. The generic method accepts any Tables.jl source (a
# column table is also matched by the NamedTuple method, which delegates here);
# the `_compose_table` worker does the shared build.
function compose(table)
    Tables.istable(table) ||
        throw(ArgumentError(
            "compose expects a NamedTuple, a Tables.jl table with `name` and " *
            "`dist` columns, or a nested Matrix; got $(typeof(table))"))
    return _compose_table(table)
end

function _compose_table(table)
    cols = Tables.columns(table)
    names = Tables.columnnames(cols)
    (:name in names && :dist in names) ||
        throw(ArgumentError("the table needs `name` and `dist` columns"))
    dists = Tables.getcolumn(cols, :dist)
    row_names = Tables.getcolumn(cols, :name)
    all(d -> d isa UnivariateDistribution, dists) ||
        throw(ArgumentError(
            "every `dist` entry must be a UnivariateDistribution"))
    # A `prob` column only makes sense alongside `compete`, which marks the rows
    # the probabilities apply to; reject it alone rather than silently ignoring.
    (:prob in names && !(:compete in names)) &&
        throw(ArgumentError(
            "a `prob` column needs a `compete` column to mark its outcome set"))
    if :compete in names
        return _compose_table_one_of(dists, row_names,
            Tables.getcolumn(cols, :compete),
            :prob in names ? Tables.getcolumn(cols, :prob) : nothing,
            :chain in names ? Tables.getcolumn(cols, :chain) : nothing)
    end
    if :chain in names
        return _compose_table_chained(
            dists, row_names, Tables.getcolumn(cols, :chain))
    end
    # Flat table: each row is a branch, the `name` column its branch name.
    return Parallel(Tuple(dists), _coerce_names(row_names, :branch, length(dists)))
end

# Fold the rows sharing a non-zero `compete` group id into one `Resolve` node
# (its `prob` entries the branch probabilities); rows with a zero/`missing`
# `compete` id stay ordinary leaf branches (or, with a `chain` column, fold into
# Sequential branches by chain id). Branches appear in first-seen order — the
# row order of each group's first member — so the Parallel reads down the table,
# named by that first row, mirroring `_compose_table_chained`.
function _compose_table_one_of(dists, row_names, compete, prob, chain)
    all(g -> g === missing || g >= 0, compete) || throw(ArgumentError(
        "`compete` group ids must be non-negative or missing"))
    # One first-seen pass assigns each row a branch KEY: a `compete:id` for a
    # one_of group, else `chain:id` for a chained leaf (a zero/`missing`
    # `compete` AND `chain` both make a fresh singleton key). The branch order is
    # the keys' first appearance, and `members` holds each key's rows in order.
    order = Any[]
    members = Dict{Any, Vector{Int}}()
    leaf_counter = 0
    has_compete = false
    for i in eachindex(dists)
        c = compete[i]
        if !(c === missing || c == 0)
            has_compete = true
            key = (:compete, Int(c))
        else
            ch = chain === nothing ? missing : chain[i]
            key = (ch === missing || ch == 0) ? (:leaf, leaf_counter -= 1) :
                  (:chain, Int(ch))
        end
        key in order || push!(order, key)
        push!(get!(members, key, Int[]), i)
    end
    has_compete || throw(ArgumentError(
        "the `compete` column marks no one_of rows (all zero/missing)"))
    branches = map(order) do key
        idx = members[key]
        if key[1] === :compete
            _one_of_from_rows(dists, row_names, prob, idx, key[2])
        elseif length(idx) == 1
            dists[idx[1]]
        else
            steps = Tuple(Symbol(row_names[i]) for i in idx)
            Sequential(Tuple(dists[i] for i in idx), steps)
        end
    end
    branch_names = Tuple(Symbol(row_names[members[key][1]]) for key in order)
    return Parallel(Tuple(branches), branch_names)
end

# Build one `Resolve` node from a compete group's rows: `name => (dist, prob)`
# per row. The constructor validates the branch probabilities sum to one and lie
# in `[0, 1]`; a missing `prob` in a compete row is an error (it is required).
function _one_of_from_rows(dists, row_names, prob, idx, gid)
    prob === nothing && throw(ArgumentError(
        "a `compete` group needs a `prob` column of branch probabilities"))
    outcomes = map(idx) do i
        p = prob[i]
        p === missing && throw(ArgumentError(
            "row $(row_names[i]) is in compete group $gid but has a missing " *
            "`prob`"))
        Symbol(row_names[i]) => (dists[i], p)
    end
    return Resolve(outcomes...)
end

# Group rows by the `chain` column: rows sharing a non-zero group id fold into
# one Sequential branch (in row order); a zero/`missing` group is a leaf branch.
# Branches appear in first-seen group order, matching the NamedTuple value order.
# Each branch is named by the FIRST row of its group; the steps within a chained
# branch are named by their own rows' `name` entries (Option A).
function _compose_table_chained(dists, row_names, groups)
    # Group ids must be non-negative: a zero/`missing` group is a unique leaf,
    # to which a fresh negative id is assigned, so a negative user group would
    # collide with those auto-generated leaf ids.
    all(g -> g === missing || g >= 0, groups) ||
        throw(ArgumentError("`chain` group ids must be non-negative or missing"))
    order = Int[]              # group ids in first-seen order (0 -> unique leaf)
    members = Dict{Int, Vector{Int}}()
    leaf_counter = 0
    for i in eachindex(dists)
        g = groups[i]
        gid = (g === missing || g == 0) ? (leaf_counter -= 1) : Int(g)
        gid in order || push!(order, gid)
        push!(get!(members, gid, Int[]), i)
    end
    branches = map(order) do gid
        idx = members[gid]
        if length(idx) == 1
            dists[idx[1]]
        else
            step_names = Tuple(Symbol(row_names[i]) for i in idx)
            Sequential(Tuple(dists[i] for i in idx), step_names)
        end
    end
    # A chained branch takes the name of its first row; a leaf branch its own.
    branch_names = Tuple(Symbol(row_names[members[gid][1]]) for gid in order)
    return Parallel(Tuple(branches), branch_names)
end
