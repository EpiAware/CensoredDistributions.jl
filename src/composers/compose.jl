# ============================================================================
# compose: the friendly front-end constructor for the composer stack
# ============================================================================
#
# `compose` is a CONSTRUCTOR over the [`Sequential`](@ref) / [`Parallel`](@ref)
# composers (#329): it does NOT introduce a new monolithic tree type. Three
# friendly inputs all lower to the SAME nested composer stack:
#
# - a `NamedTuple` (named, recursive): a `Parallel` over the named children; a
#   child that is itself a `NamedTuple` nests as a `Parallel`, a child that is a
#   `Vector`/`Tuple` of distributions nests as a `Sequential`, a bare
#   distribution is a leaf;
# - a Tables.jl table with `name` and `dist` columns: a `Parallel` over the rows
#   (the column-table equivalent of a flat `NamedTuple`);
# - a nested `Matrix` of distributions: rows are `Sequential` steps and the
#   columns within a row are `Parallel` branches (a branching grid).
#
# The mappings are chosen so the three inputs build identical stacks for the
# same structure, which the tests assert by `==` on the composed objects.

@doc raw"

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
  bare `UnivariateDistribution` is a leaf branch.
- Tables.jl table with `name` and `dist` columns: a [`Parallel`](@ref) over the
  rows, the column-table equivalent of a flat `NamedTuple`.
- nested `Matrix` of distributions: rows are [`Parallel`](@ref) branches and the
  columns within a row are [`Sequential`](@ref) steps, so a one-column matrix is
  parallel leaf branches and a one-row matrix is a single chain.

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
- [`Sequential`](@ref), [`Parallel`](@ref), [`Competing`](@ref): the composers
"
function compose end

# --- NamedTuple front-end --------------------------------------------------
# A NamedTuple maps to a Parallel over its values, each value lowered by
# `_compose_child`. The names label branches for the user but do not change the
# stack, so this matches the table and matrix forms structurally.
#
# A column table is also a NamedTuple, so a NamedTuple carrying `name`/`dist`
# column vectors is routed to the Tables.jl path instead, letting one
# `(name, dist, chain)` column table build the same stack as a structural
# NamedTuple.
function compose(nt::NamedTuple)
    _is_column_table(nt) && return _compose_table(nt)
    return Parallel(map(_compose_child, Tuple(nt)))
end

# A NamedTuple is treated as a column table when it has `name` and `dist` fields
# that are both vectors (the column-table shape), not nested branch values.
function _is_column_table(nt::NamedTuple)
    haskey(nt, :name) && haskey(nt, :dist) &&
        nt.name isa AbstractVector && nt.dist isa AbstractVector
end

# Lower a single NamedTuple value to a composer child.
_compose_child(d::UnivariateDistribution) = d
_compose_child(nt::NamedTuple) = compose(nt)
function _compose_child(v::Union{AbstractVector, Tuple})
    all(x -> x isa UnivariateDistribution, v) ||
        throw(ArgumentError(
            "a sequential child must hold UnivariateDistributions"))
    return Sequential(Tuple(v))
end

# --- nested Matrix front-end -----------------------------------------------
# A matrix maps to a Parallel over its rows (branches), each row a Sequential
# over its columns (chain steps). A row with a single entry collapses to that
# bare leaf, so a one-column matrix is one chain folded into a single Parallel
# branch and a one-row matrix is parallel leaf branches, matching the
# NamedTuple/table forms for the same structure.
function compose(m::AbstractMatrix{<:UnivariateDistribution})
    nrows, ncols = size(m)
    (nrows >= 1 && ncols >= 1) ||
        throw(ArgumentError("the matrix needs at least one row and column"))
    branches = ntuple(nrows) do i
        steps = Tuple(m[i, j] for j in 1:ncols)
        ncols == 1 ? steps[1] : Sequential(steps)
    end
    return Parallel(branches)
end

# --- Tables.jl table front-end ---------------------------------------------
# A table with `name` and `dist` columns maps to a Parallel over its rows, the
# column-table equivalent of a flat NamedTuple of leaves. An optional `chain`
# column groups consecutive rows that share a non-zero group id into one
# Sequential branch, so a table can also express the nested chain a NamedTuple
# encodes with a vector value. The generic method accepts any Tables.jl source
# (a column table is also matched by the NamedTuple method, which delegates
# here); the `_compose_table` worker does the shared build.
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
    all(d -> d isa UnivariateDistribution, dists) ||
        throw(ArgumentError(
            "every `dist` entry must be a UnivariateDistribution"))
    if :chain in names
        return _compose_table_chained(dists, Tables.getcolumn(cols, :chain))
    end
    return Parallel(Tuple(dists))
end

# Group rows by the `chain` column: rows sharing a non-zero group id fold into
# one Sequential branch (in row order); a zero/`missing` group is a leaf branch.
# Branches appear in first-seen group order, matching the NamedTuple value order.
function _compose_table_chained(dists, groups)
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
        length(idx) == 1 ? dists[idx[1]] :
        Sequential(Tuple(dists[i] for i in idx))
    end
    return Parallel(Tuple(branches))
end
