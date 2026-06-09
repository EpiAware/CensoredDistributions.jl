# STRUCTURAL equality for the composers (Option A): two front-ends that
# build the same nested stack compare equal even if their node NAMES differ.
# Names are metadata labelling the structure for `params`/`params_table`/`show`,
# so the NamedTuple, table, and matrix `compose` forms stay structurally `==`
# while each carries its own names. `==`/`hash` therefore compare only
# `components` for `Sequential`/`Parallel` (ignoring the `names` field); use
# `component_names` to compare names explicitly. `Competing` keeps its names in
# `==`/`hash`, as those names are intrinsic outcome identities, not relaxable
# structure metadata.

Base.:(==)(a::Sequential, b::Sequential) = a.components == b.components
Base.:(==)(a::Parallel, b::Parallel) = a.components == b.components
function Base.:(==)(a::Competing, b::Competing)
    return a.names == b.names && a.delays == b.delays &&
           a.branch_probs == b.branch_probs
end
function Base.:(==)(a::Select, b::Select)
    return a.names == b.names && a.alternatives == b.alternatives &&
           a.selector == b.selector
end

Base.hash(d::Sequential, h::UInt) = hash(d.components, hash(:Sequential, h))
Base.hash(d::Parallel, h::UInt) = hash(d.components, hash(:Parallel, h))
function Base.hash(c::Competing, h::UInt)
    return hash(c.branch_probs,
        hash(c.delays, hash(c.names, hash(:Competing, h))))
end
function Base.hash(d::Select, h::UInt)
    return hash(d.selector,
        hash(d.alternatives, hash(d.names, hash(:Select, h))))
end
