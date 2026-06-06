# Structural equality for the composers, so two front-ends that build the same
# nested stack compare equal. Distributions.jl gives no structural `==` for these
# wrappers, and the front-end tests assert the NamedTuple, table, and matrix
# forms all produce the identical stack. Equality is defined field-wise over the
# components (recursing into nested composers); `hash` is kept consistent.

Base.:(==)(a::Sequential, b::Sequential) = a.components == b.components
Base.:(==)(a::Parallel, b::Parallel) = a.components == b.components
function Base.:(==)(a::Competing, b::Competing)
    return a.names == b.names && a.delays == b.delays &&
           a.branch_probs == b.branch_probs
end

Base.hash(d::Sequential, h::UInt) = hash(d.components, hash(:Sequential, h))
Base.hash(d::Parallel, h::UInt) = hash(d.components, hash(:Parallel, h))
function Base.hash(c::Competing, h::UInt)
    return hash(c.branch_probs,
        hash(c.delays, hash(c.names, hash(:Competing, h))))
end
