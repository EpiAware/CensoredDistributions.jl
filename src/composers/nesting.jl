# Shared nesting machinery for the composers, defined once both composer types
# exist so the `Union{Sequential, Parallel}` methods resolve. A realisation of
# any composer is a FLAT vector of leaf values; a nested child contributes its
# own flat sub-vector, so nesting is pure concatenation and that nesting is the
# tree. These helpers do the flat-slice recursion shared by `Sequential` and
# `Parallel`. This layer adds NO censored-internal behaviour.

# A composable child is any univariate distribution (a leaf or a `Competing`), a
# nested `Sequential` / `Parallel` / `Select`, or a `latent`-wrapped node. Used to
# validate composer components and `Select` alternatives. A `Latent` is a
# Multivariate node over `[primary, observed]`, so it is admitted explicitly here
# rather than through the univariate clause; this lets a `Select` carry a latent
# alternative branch (the index-vs-sourced split's sourced chain).
_is_composable(::UnivariateDistribution) = true
_is_composable(::Union{Sequential, Parallel}) = true
_is_composable(::Select) = true
_is_composable(::Latent) = true
_is_composable(::Any) = false

# Default positional names for a composer node, used when the front-end (or a
# positional constructor) supplies none. `_default_names(:step, 3)` is
# `(:step_1, :step_2, :step_3)`; the prefix is `:step` for `Sequential` and
# `:branch` for `Parallel`. Built as a typed tuple so the names field stays
# concretely typed.
function _default_names(prefix::Symbol, n::Int)
    return ntuple(i -> Symbol(prefix, :_, i), n)
end

# Coerce a user-supplied names collection (a tuple/vector of Symbols, or
# `nothing` for "use defaults") to a Symbol tuple of the right length. Used by
# the `compose` front-ends so every input format threads names through.
_coerce_names(::Nothing, prefix::Symbol, n::Int) = _default_names(prefix, n)
function _coerce_names(names, ::Symbol, n::Int)
    length(names) == n || throw(ArgumentError(
        "supplied $(length(names)) names for $n components"))
    return Tuple(Symbol(x) for x in names)
end

# Number of flat leaf values a child contributes: one for a univariate leaf,
# its own leaf count for a nested composer.
_child_nleaves(::UnivariateDistribution) = 1
_child_nleaves(c::Union{Sequential, Parallel}) = length(c)
# A nested `Select` swaps in ONE alternative of fixed width, so it occupies a
# fixed flat slot only when every alternative has the same leaf count. The
# common width is the nested Select's leaf count; disagreeing widths cannot
# share one flat slot and error (a `length(::Select)` has no single answer).
function _child_nleaves(c::Select)
    n = _child_nleaves(first(c.alternatives))
    widths = map(_child_nleaves, c.alternatives)
    all(==(n), widths) || throw(ArgumentError(
        "a nested Select needs every alternative to have the same leaf count " *
        "to occupy a fixed flat slot; got $(widths)"))
    return n
end
# A latent alternative scores `[primary, observed]` (two slots); the flat
# value-vector layout collapses it to its marginal leaf count, since a nested
# Select's flat slot carries observed values, not the latent primary.
_child_nleaves(c::Latent) = _child_nleaves(c.dist)

# Total leaf count over a tuple of children. A HEAD/TAIL recursion, NOT
# `sum(_child_nleaves, components)`: `sum(f, ::Tuple)` over a heterogeneous tuple
# is inferred `Any` on the CI compilers (`lts`/`1`) -- it lowers to a generic
# `mapreduce` whose accumulator type the older inference cannot resolve -- which
# poisons every downstream `Vector{...}(undef, _nleaves(...) + 1)` constructor
# (its length argument becomes `Any`, so the constructed array type widens to
# `Any` and the whole sampling/scoring path infers `Any`). Julia 1.12 happens to
# constant-fold the `sum` and so masks the regression locally. The recursion
# below resolves to a concrete `Int` per step on every supported version.
_nleaves(::Tuple{}) = 0
function _nleaves(components::Tuple)
    _child_nleaves(first(components)) + _nleaves(Base.tail(components))
end

# Number of EVENT slots a child contributes to the flat EVENT vector.
# Distinct from `_child_nleaves` (the generic VALUE-vector layout): a `Competing`
# node contributes ONE value (its marginal time-to-resolution) to the value
# vector but exposes one EVENT slot PER OUTCOME so a record's death/discharge
# columns each land in their own slot and the observed outcome is identified
# positionally (self-dispatch). Every other child contributes the same count
# as `_child_nleaves`, so the value and event layouts coincide for Competing-free
# trees and `length`/the generic value path are untouched.
_event_child_nleaves(c) = _child_nleaves(c)
_event_child_nleaves(c::Competing) = _n_branches(c)
_event_child_nleaves(c::Union{Sequential, Parallel}) = _event_nleaves(c.components)
# A nested `Select` occupies its (common) alternative's EVENT-slot width: every
# alternative must expose the same number of event slots to share one flat slot,
# so the chosen alternative for a row lands in the same slice whichever it is.
function _event_child_nleaves(c::Select)
    n = _event_child_nleaves(first(c.alternatives))
    widths = map(_event_child_nleaves, c.alternatives)
    all(==(n), widths) || throw(ArgumentError(
        "a nested Select needs every alternative to expose the same number of " *
        "event slots to occupy a fixed flat slot; got $(widths)"))
    return n
end

# Total EVENT-slot count over a tuple of children (the flat event vector minus
# its shared origin). HEAD/TAIL recursion for the same reason as `_nleaves`:
# `sum(_event_child_nleaves, ::Tuple)` infers `Any` on the CI compilers and
# widens the `Vector{Union{Missing, T}}(missing, _event_nleaves(...) + 1)`
# constructor in `_tree_event_vector` to `Any`, breaking `@inferred` on the
# sampling walk on every version except the one that constant-folds it (1.12).
_event_nleaves(::Tuple{}) = 0
function _event_nleaves(components::Tuple)
    _event_child_nleaves(first(components)) +
    _event_nleaves(Base.tail(components))
end

# Sum the per-child log-densities over the matching flat slices of `x`. A leaf
# consumes one scalar; a nested composer consumes a `_child_nleaves`-long slice
# and recurses. The offset walk is pure control flow over the constant index, so
# the differentiated arithmetic sees only concrete values (AD-safe).
function _composite_logpdf(components::Tuple, x::AbstractVector)
    total = zero(eltype(x))
    offset = 0
    @inbounds for c in components
        n = _child_nleaves(c)
        total += _child_logpdf(c, x, offset, n)
        offset += n
    end
    return total
end

_child_logpdf(c::UnivariateDistribution, x, offset, ::Int) = logpdf(c, x[offset + 1])
# A nested child scores its own contiguous slice of the value vector; a `@view`
# avoids a copy and differentiates on every supported backend.
function _child_logpdf(c::Union{Sequential, Parallel}, x, offset, n::Int)
    logpdf(c, @view x[(offset + 1):(offset + n)])
end
# A nested `Select` in the data-free flat value-vector path commits to its FIRST
# alternative (a deterministic default so flat `logpdf`/`rand` round-trip); the
# selector-driven choice lives in the row/record path, not the flat path.
function _child_logpdf(c::Select, x, offset, n::Int)
    return _child_logpdf(_flat_select_alternative(c), x, offset, n)
end
# A latent alternative on the flat path scores through its marginal node (the
# flat slot carries observed values; the latent primary is integrated out).
function _child_logpdf(c::Latent, x, offset, n::Int)
    return _child_logpdf(c.dist, x, offset, n)
end

# The alternative a nested Select commits to on the data-free flat path: the
# first. The row/record path overrides this by the row's selector value.
_flat_select_alternative(c::Select) = first(c.alternatives)

# Concatenate the per-child draws into one flat vector of element type `T`.
function _composite_rand(rng::AbstractRNG, components::Tuple, ::Type{T}) where {T}
    out = Vector{T}(undef, _nleaves(components))
    offset = 0
    @inbounds for c in components
        n = _child_nleaves(c)
        _child_rand!(out, offset, rng, c)
        offset += n
    end
    return out
end

function _child_rand!(out, offset, rng::AbstractRNG, c::UnivariateDistribution)
    out[offset + 1] = rand(rng, c)
    return nothing
end
function _child_rand!(
        out, offset, rng::AbstractRNG, c::Union{Sequential, Parallel})
    sub = rand(rng, c)
    @inbounds for k in eachindex(sub)
        out[offset + k] = sub[k]
    end
    return nothing
end
# A nested `Select` samples its FIRST alternative on the flat path, matching the
# committed alternative the flat `_child_logpdf` scores.
function _child_rand!(out, offset, rng::AbstractRNG, c::Select)
    return _child_rand!(out, offset, rng, _flat_select_alternative(c))
end
# A latent alternative samples its observed value through its marginal node on
# the flat path (the latent primary is not part of the flat slot).
function _child_rand!(out, offset, rng::AbstractRNG, c::Latent)
    return _child_rand!(out, offset, rng, c.dist)
end

# The recursive indented-tree printing and the `params`/`params_table` traversal
# share the AbstractTrees.jl interface defined in `introspection.jl`
# (`ComposerNode`, `children`, `printnode`, `_node_header`, `_show_composer_tree`).
