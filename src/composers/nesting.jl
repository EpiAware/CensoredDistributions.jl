# Shared nesting machinery for the composers, defined once both composer types
# exist so the `Union{Sequential, Parallel}` methods resolve. A realisation of
# any composer is a FLAT vector of leaf values; a nested child contributes its
# own flat sub-vector, so nesting is pure concatenation and that nesting is the
# tree. These helpers do the flat-slice recursion shared by `Sequential` and
# `Parallel`. This layer adds NO censored-internal behaviour (#329).

# A composable child is any univariate distribution (a leaf or a `Competing`) or
# a nested `Sequential` / `Parallel` / `Select`. Used to validate composer
# components and `Select` alternatives.
_is_composable(::UnivariateDistribution) = true
_is_composable(::Union{Sequential, Parallel}) = true
_is_composable(::Select) = true
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

# Total leaf count over a tuple of children.
_nleaves(components::Tuple) = sum(_child_nleaves, components)

# Number of EVENT slots a child contributes to the flat EVENT vector (#333).
# Distinct from `_child_nleaves` (the generic VALUE-vector layout): a `Competing`
# node contributes ONE value (its marginal time-to-resolution) to the value
# vector but exposes one EVENT slot PER OUTCOME so a record's death/discharge
# columns each land in their own slot and the observed outcome is identified
# positionally (#329 self-dispatch). Every other child contributes the same count
# as `_child_nleaves`, so the value and event layouts coincide for Competing-free
# trees and `length`/the generic value path are untouched.
_event_child_nleaves(c) = _child_nleaves(c)
_event_child_nleaves(c::Competing) = _n_branches(c)
_event_child_nleaves(c::Union{Sequential, Parallel}) = _event_nleaves(c.components)

# Total EVENT-slot count over a tuple of children (the flat event vector minus
# its shared origin).
_event_nleaves(components::Tuple) = sum(_event_child_nleaves, components)

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

# The recursive indented-tree printing and the `params`/`params_table` traversal
# share the AbstractTrees.jl interface defined in `introspection.jl`
# (`ComposerNode`, `children`, `printnode`, `_node_header`, `_show_composer_tree`).
