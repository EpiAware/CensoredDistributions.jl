# ============================================================================
# compose: the friendly front-end constructor for the composer stack
# ============================================================================
#
# `compose` is a constructor over the [`Sequential`](@ref) / [`Parallel`](@ref)
# composers: it does not introduce a new monolithic tree type. Two
# friendly inputs both lower to the same nested composer stack:
#
# - a `NamedTuple` (named, recursive): a `Parallel` over the named children; a
#   child that is itself a `NamedTuple` nests as a `Parallel`, a child that is a
#   `Vector`/`Tuple` of distributions nests as a `Sequential`, a bare
#   distribution is a leaf;
# - an explicit Tables.jl table with `name` and `dist` columns: a `Parallel`
#   over the rows (the column-table equivalent of a flat `NamedTuple`).
#
# The mappings are chosen so the two inputs build identical stacks for the
# same structure, which the tests assert by `==` on the composed objects.

@doc "

Build a nested composer stack from a friendly front-end input.

`compose` lowers a NamedTuple or an explicit Tables.jl table to the same
[`Sequential`](@ref) / [`Parallel`](@ref) stack. It is a constructor over the
composers, not a new tree type.

# Arguments
- `input`: the front-end to lower, one of the two forms below.

# Inputs

- `NamedTuple` (named, recursive): a [`Parallel`](@ref) over the named children.
  A child that is itself a `NamedTuple` nests as a `Parallel`, a child that is a
  `Vector` or `Tuple` of distributions nests as a [`Sequential`](@ref), and a
  bare `UnivariateDistribution` is a leaf branch. The same flat structure may be
  given as `:name => child` Pairs (`compose(:a => d1, :b => d2)`) for data-
  driven or computed branch names; the two spellings build the same stack.
- explicit Tables.jl table with `name` and `dist` columns: a [`Parallel`](@ref)
  over the rows, the column-table equivalent of a flat `NamedTuple`. An optional
  `chain` column folds rows sharing a non-zero group id into a
  [`Sequential`](@ref) branch, and an optional `compete`/`prob` column pair folds
  rows sharing a non-zero `compete` id into a [`Resolve`](@ref) node whose `prob`
  entries are the branch probabilities (each in ``[0, 1]`` and summing to one per
  group). A NamedTuple is always read structurally, so a column-table source must
  be passed as a non-NamedTuple Tables.jl source (e.g. a row table or DataFrame).

# Contract

`compose` always returns a composer, never a bare univariate leaf.
A single branch stays a [`Parallel`](@ref)-of-one and a single step a
one-element [`Sequential`](@ref); the wrapper is never collapsed away.
A bare leaf is used directly at the scoring layer, where
[`record_distributions`](@ref) and [`composed_distribution_model`](@ref) accept
a bare `UnivariateDistribution`, so callers do not need `compose` to pass one
through.

# Examples
```@example
using CensoredDistributions, Distributions

# A regular 2x2 grid built two ways, both equal. The table is a row table
# (a Tables.jl source); a column-table NamedTuple would lower structurally.
nt = (r1 = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4)],
    r2 = [Gamma(1.0, 1.0), Gamma(3.0, 1.0)])
table = [(name = :a, dist = Gamma(2.0, 1.0), chain = 1),
    (name = :b, dist = LogNormal(0.5, 0.4), chain = 1),
    (name = :c, dist = Gamma(1.0, 1.0), chain = 2),
    (name = :d, dist = Gamma(3.0, 1.0), chain = 2)]
compose(nt) == compose(table)
```

# See also
- [`Sequential`](@ref), [`Parallel`](@ref), [`Resolve`](@ref): the composers
"
function compose end

# --- NamedTuple front-end --------------------------------------------------
# A NamedTuple maps to a Parallel over its values, each value lowered by
# `_compose_child`. The keys become the branch names, threaded into the
# `Parallel` so `params`/`params_table`/`show` are name-keyed (Option A).
# Structurally this still matches the table form (`==` ignores names); only the
# labels differ.
#
# A NamedTuple is always read structurally, even one whose keys happen to be
# `:name`/`:dist`: there is no column-table auto-detection. To build from a
# `(name, dist, chain)` column table, pass a non-NamedTuple Tables.jl source
# (a row table or DataFrame) to the `compose(table)` method.
function compose(nt::NamedTuple)
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

# --- repeated-leaf front-end -----------------------------------------------
# `compose(dist, n)` repeats the same distribution (or pre-built composer
# subtree) `n` times, the clean replacement for spelling out `n` identical
# steps / branches by hand. The default builds a [`Sequential`](@ref) chain of
# `n` identical steps (`:step_1 … :step_n`); `chain = false` builds a
# [`Parallel`](@ref) of `n` identical branches (`:branch_1 … :branch_n`). The
# repeated child is one shared object, so a single edit (or one prior keyed by
# the repeated leaf) covers every copy.
@doc raw"""

Repeat the same distribution `n` times into a composer stack.

`compose(dist, n)` is the count form of [`compose`](@ref): it repeats `dist` (a
leaf or a pre-built composer subtree) `n` times, instead of writing out `n`
identical steps or branches by hand. The default builds a [`Sequential`](@ref)
chain (`:step_1 … :step_n`); `chain = false` builds a [`Parallel`](@ref) of `n`
identical branches (`:branch_1 … :branch_n`).

# Arguments
- `dist`: the distribution or composer subtree to repeat.
- `n`: the repeat count (`n >= 1`).

# Keyword Arguments
- `chain`: `true` (default) repeats into a [`Sequential`](@ref) chain; `false`
  into a [`Parallel`](@ref) branch set.

# Examples
```@example
using CensoredDistributions, Distributions

# A three-step chain of one shared delay (e.g. a fixed-rate progression).
compose(Gamma(2.0, 1.0), 3)

# Three independent branches off one origin instead.
compose(Gamma(2.0, 1.0), 3; chain = false)
```

# See also
- [`compose`](@ref): the NamedTuple / table front-end.
- [`sequential`](@ref), [`parallel`](@ref): the structural composers this builds.
"""
function compose(dist::Union{UnivariateDistribution, Sequential, Parallel,
            Choose},
        n::Integer; chain::Bool = true)
    n >= 1 || throw(ArgumentError("compose(dist, n) needs n >= 1; got $n"))
    child = _compose_child(dist)
    components = ntuple(_ -> child, n)
    return chain ? Sequential(components) : Parallel(components)
end

# --- Tables.jl table front-end ---------------------------------------------
# A table with `name` and `dist` columns maps to a Parallel over its rows, the
# column-table equivalent of a flat NamedTuple of leaves. An optional `chain`
# column groups consecutive rows that share a non-zero group id into one
# Sequential branch, so a table can also express the nested chain a NamedTuple
# encodes with a vector value. An optional `compete`/`prob` column pair folds
# the rows sharing a non-zero `compete` group id into one `Resolve` node (the
# `prob` entries its branch probabilities), so the table can also express a
# one_of-outcome set. This generic method accepts any Tables.jl source except a
# NamedTuple, which the more specific NamedTuple method always reads structurally
# (there is no column-table auto-detection); the `_compose_table` worker does the
# table build.
function compose(table)
    Tables.istable(table) ||
        throw(ArgumentError(
            "compose expects a NamedTuple or a Tables.jl table with `name` " *
            "and `dist` columns; got $(typeof(table))"))
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
    # One first-seen pass assigns each row a branch key: a `compete:id` for a
    # one_of group, else `chain:id` for a chained leaf (a zero/`missing`
    # `compete` and `chain` both make a fresh singleton key). The branch order is
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
# Each branch is named by the first row of its group; the steps within a chained
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
