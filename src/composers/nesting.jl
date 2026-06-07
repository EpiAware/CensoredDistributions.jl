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

# ---------------------------------------------------------------------------
# Recursive indented-tree printing for the composers
# ---------------------------------------------------------------------------
#
# A nested composed distribution prints as ONE indented tree, recursing into
# every child so the whole structure is visible at once. The three composers
# share the same `├─ / └─` glyphs and indentation via `_show_tree`: a header
# line for the node, then each child indented one level, with composer children
# recursing and leaf distributions printed inline. The compact `show(io, d)`
# one-liners on each type are kept for inline/array display.

# Header label for a composer node (the node TYPE, plus a count).
_node_header(d::Sequential) = "Sequential ($(length(d.components)) steps)"
_node_header(d::Parallel) = "Parallel ($(length(d.components)) branches)"
_node_header(c::Competing) = "Competing ($(_n_branches(c)) outcomes)"

# Child labels: a composer child has no inline label (it recurses); a leaf is
# shown by its compact `repr`. `Sequential`/`Parallel` children are unnamed
# (positional); `Competing` labels each child with its outcome name and prob.
_is_composer(::Union{Sequential, Parallel, Competing}) = true
_is_composer(::Any) = false

# Print `node`'s subtree to `io`. `prefix` is the accumulated indentation for
# this node's children; the root call passes an empty prefix. Each child gets a
# `├─ ` connector (or `└─ ` for the last), and a composer child recurses with an
# extended prefix (`│  ` for non-last siblings, spaces for the last).
function _show_tree(io::IO, node, prefix::String)
    children, labels = _tree_children(node)
    n = length(children)
    for i in 1:n
        last = i == n
        connector = last ? "└─ " : "├─ "
        child = children[i]
        label = labels[i]
        if _is_composer(child)
            head = isempty(label) ? _node_header(child) :
                   "$(label): $(_node_header(child))"
            println(io, prefix, connector, head)
            _show_tree(io, child, prefix * (last ? "   " : "│  "))
        else
            line = isempty(label) ? string(child) : "$(label): $(child)"
            println(io, prefix, connector, line)
        end
    end
    return nothing
end

# Children and their inline labels for each composer. `Sequential`/`Parallel`
# children are positional (no label); `Competing` labels each by outcome name
# and branch probability.
function _tree_children(d::Union{Sequential, Parallel})
    return collect(d.components), fill("", length(d.components))
end
function _tree_children(c::Competing)
    labels = ["$(c.names[k]) (p = $(c.branch_probs[k]))"
              for k in 1:_n_branches(c)]
    return collect(c.delays), labels
end
