# Shared nesting machinery for the composers, defined once both composer types
# exist so the `Union{Sequential, Parallel}` methods resolve. A realisation of
# any composer is a FLAT vector of leaf values; a nested child contributes its
# own flat sub-vector, so nesting is pure concatenation and that nesting is the
# tree. These helpers do the flat-slice recursion shared by `Sequential` and
# `Parallel`. This layer adds NO censored-internal behaviour (#329).

# A composable child is any univariate distribution (a leaf or a `Competing`) or
# a nested `Sequential` / `Parallel`. Used to validate composer components.
_is_composable(::UnivariateDistribution) = true
_is_composable(::Union{Sequential, Parallel}) = true
_is_composable(::Any) = false

# Number of flat leaf values a child contributes: one for a univariate leaf,
# its own leaf count for a nested composer.
_child_nleaves(::UnivariateDistribution) = 1
_child_nleaves(c::Union{Sequential, Parallel}) = length(c)

# Total leaf count over a tuple of children.
_nleaves(components::Tuple) = sum(_child_nleaves, components)

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
