# Structural equality for the composers (Option A): two front-ends that
# build the same nested stack compare equal even if their node names differ.
# Names are metadata labelling the structure for `params`/`params_table`/`show`,
# so the NamedTuple, table, and matrix `compose` forms stay structurally `==`
# while each carries its own names. `==`/`hash` therefore compare only
# `components` for `Sequential`/`Parallel` (ignoring the `names` field); use
# `component_names` to compare names explicitly. `Resolve` keeps its names in
# `==`/`hash`, as those names are intrinsic outcome identities, not relaxable
# structure metadata.

Base.:(==)(a::Sequential, b::Sequential) = a.components == b.components
Base.:(==)(a::Parallel, b::Parallel) = a.components == b.components
function Base.:(==)(a::Resolve, b::Resolve)
    return a.names == b.names && a.delays == b.delays &&
           a.branch_probs == b.branch_probs
end
# A racing-hazard node has no branch probabilities (derived), so its identity is
# its names and racing delays. A mixture and a racing-hazard node are never equal.
function Base.:(==)(a::Compete, b::Compete)
    return a.names == b.names && a.delays == b.delays
end
Base.:(==)(::Resolve, ::Compete) = false
Base.:(==)(::Compete, ::Resolve) = false
Base.:(==)(::NoEvent, ::NoEvent) = true
function Base.:(==)(a::Choose, b::Choose)
    return a.names == b.names && a.alternatives == b.alternatives &&
           a.selector == b.selector
end

Base.hash(d::Sequential, h::UInt) = hash(d.components, hash(:Sequential, h))
Base.hash(d::Parallel, h::UInt) = hash(d.components, hash(:Parallel, h))
function Base.hash(c::Resolve, h::UInt)
    return hash(c.branch_probs,
        hash(c.delays, hash(c.names, hash(:Resolve, h))))
end
function Base.hash(c::Compete, h::UInt)
    return hash(c.delays, hash(c.names, hash(:Compete, h)))
end
Base.hash(::NoEvent, h::UInt) = hash(:NoEvent, h)
function Base.hash(d::Choose, h::UInt)
    return hash(d.selector,
        hash(d.alternatives, hash(d.names, hash(:Choose, h))))
end
