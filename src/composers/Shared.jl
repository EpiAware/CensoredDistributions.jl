# ============================================================================
# Shared: a name-tagged sub-distribution tied across composed branches
# ============================================================================
#
# `shared(:inc, dist)` tags a leaf with a NAME so two occurrences carrying the
# same tag are ONE free parameter, not two. A parameter can appear in several
# branches of a tree (e.g. an incubation `inc` in BOTH the index and sourced
# branches of a `selecting`); without a tie the prior/params interface would
# inventory, sample and update each occurrence independently and duplicate the
# shared parameter. A `Shared` tag lets the interface dedup BY NAME: inventory
# once, sample once, place the one sampled value in every occurrence.
#
# The wrapper is TRANSPARENT in the hot path: every `Distributions` method
# delegates to the wrapped leaf, so `logpdf`/`rand`/`cdf`/... are unchanged and
# AD flows straight through. Only the introspection (`params_table`), the
# reconstruction (`update`) and the Turing parameter submodel
# (`composed_parameters_model`) read the tag, deduping occurrences by it.

@doc "

A name-tagged leaf tied across the branches of a composed distribution.

`Shared` wraps a leaf distribution with a `tag` (a `Symbol`) marking it as a
shared parameter group. Two `Shared` leaves carrying the SAME tag are treated as
the SAME free parameter by the prior/params interface: [`params_table`](@ref)
lists the group's parameters ONCE (deduped by tag),
[`composed_parameters_model`](@ref) samples the group ONCE and places the sampled
values in every occurrence, and [`update`](@ref) updates all occurrences from one
entry. The wrapper is transparent to scoring and sampling (every distribution
method delegates to the wrapped leaf), so it only changes how parameters are
inventoried, sampled and reconstructed.

# Fields
- `tag`: the shared-parameter group name (`Symbol`).
- `dist`: the wrapped leaf distribution.

# See also
- [`shared`](@ref): constructor over a name and a distribution.
- [`params_table`](@ref), [`update`](@ref): dedup occurrences by tag.
"
struct Shared{D <: UnivariateDistribution} <:
       UnivariateDistribution{ValueSupport}
    "The shared-parameter group name (`Symbol`)."
    tag::Symbol
    "The wrapped leaf distribution."
    dist::D
end

@doc "

Tag a leaf distribution as a shared parameter group named `name`.

`shared(name, dist)` marks `dist` as a tied parameter so multiple occurrences of
the same `name` in a composed distribution are handled ONCE by the prior/params
interface (inventoried, sampled and updated as a single free parameter), with the
shared value placed in every occurrence. The result is transparent to scoring and
sampling.

# Arguments
- `name`: the shared-parameter group name (`Symbol`).
- `dist`: the leaf distribution to tag.

# Examples
```@example
using CensoredDistributions, Distributions

# The same incubation `inc` tied across two branches of a `selecting`.
inc = shared(:inc, Gamma(2.0, 1.0))
d = selecting(:index => inc,
    :sourced => compose((src = LogNormal(0.5, 0.4), inc = inc)))
event_names(d)
```

# See also
- [`Shared`](@ref): the tagged-leaf type.
"
shared(name::Symbol, dist::UnivariateDistribution) = Shared(name, dist)

# The shared tag of a (possibly censored) leaf, or `nothing` when untagged. The
# tag survives censoring wrappers so `shared(:inc, double_interval_censored(...))`
# and a bare `shared(:inc, Gamma(...))` both report `:inc`.
_shared_tag(leaf) = nothing
_shared_tag(d::Shared) = d.tag
_shared_tag(d::PrimaryCensored) = _shared_tag(d.dist)
_shared_tag(d::IntervalCensored) = _shared_tag(d.dist)
_shared_tag(d::Truncated) = _shared_tag(d.untruncated)

# `Shared` is transparent: every distribution method delegates to the wrapped
# leaf, so the hot path (logpdf/rand/cdf/quantile/...) is unchanged and AD flows
# straight through. Only the introspection/reconstruction layers read the tag.
get_dist(d::Shared) = d.dist
free_leaf(d::Shared) = free_leaf(d.dist)
rewrap_leaf(d::Shared, inner) = Shared(d.tag, rewrap_leaf(d.dist, inner))

# The tag does not change the realisation type, so the element type is the
# wrapped leaf's (keeps a composed tree's `eltype`/`rand` element type correct).
Base.eltype(::Type{<:Shared{D}}) where {D} = eltype(D)
minimum(d::Shared) = minimum(d.dist)
maximum(d::Shared) = maximum(d.dist)
insupport(d::Shared, x::Real) = insupport(d.dist, x)
params(d::Shared) = params(d.dist)

logpdf(d::Shared, x::Real) = logpdf(d.dist, x)
pdf(d::Shared, x::Real) = pdf(d.dist, x)
cdf(d::Shared, x::Real) = cdf(d.dist, x)
logcdf(d::Shared, x::Real) = logcdf(d.dist, x)
ccdf(d::Shared, x::Real) = ccdf(d.dist, x)
logccdf(d::Shared, x::Real) = logccdf(d.dist, x)
quantile(d::Shared, p::Real) = quantile(d.dist, p)
Base.rand(rng::AbstractRNG, d::Shared) = rand(rng, d.dist)

@doc "

Print a [`Shared`](@ref) tagged leaf as its tag and wrapped distribution.

See also: [`shared`](@ref)
"
function Base.show(io::IO, d::Shared)
    print(io, "shared(", repr(d.tag), ", ", d.dist, ")")
    return nothing
end

# --- shared-tag collection (for dedup in params/sampling) -------------------

# Collect the FIRST-occurrence leaf per shared tag in pre-order, as a
# `tag => leaf` ordered pairs vector. The first occurrence defines the tag's free
# parameters (its inner family) for the prior table and the sampling submodel;
# later occurrences reuse the one sampled value. Used by `composed_parameters_model`
# to sample each shared group once.
function _collect_shared(d)
    acc = Pair{Symbol, Any}[]
    seen = Set{Symbol}()
    _collect_shared!(acc, seen, d)
    return acc
end

function _collect_shared!(acc, seen, d::Union{Sequential, Parallel})
    for c in d.components
        _collect_shared!(acc, seen, c)
    end
    return nothing
end
function _collect_shared!(acc, seen, d::Select)
    for a in d.alternatives
        _collect_shared!(acc, seen, a)
    end
    return nothing
end
function _collect_shared!(acc, seen, c::Competing)
    for g in c.delays
        _collect_shared!(acc, seen, g)
    end
    return nothing
end
function _collect_shared!(acc, seen, leaf)
    tag = _shared_tag(leaf)
    (tag === nothing || tag in seen) && return nothing
    push!(seen, tag)
    push!(acc, tag => leaf)
    return nothing
end
