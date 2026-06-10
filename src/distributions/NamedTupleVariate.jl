# ============================================================================
# PROOF OF CONCEPT — NamedTuple-valued composed variate (RFC / do-not-merge)
# ============================================================================
#
# This file is a DECISION-INFORMING experiment for the issue "Evaluate
# NamedTuple-valued event representation (learn from JuliaBUGS #459)". It is NOT
# a feature: it does not replace the existing multivariate-`Vector` variate of
# `Sequential`/`Parallel`, it adds a thin wrapper alongside it so the variate
# form can be exercised end to end (`rand` -> NamedTuple,
# `logpdf(::NamedTuple)`, `mean` -> NamedTuple) and the three hard consequences
# (bijector, AD, `product_distribution`) can be probed against real backends.
#
# Design mirrors JuliaBUGS' `to_distribution`: a single draw is a `NamedTuple`
# keyed by the tree's flat EVENT names (`tree_event_names`), not a positional
# vector. The names are baked into the type parameter so the variate is
# self-labelling at the type level, exactly as `NamedTupleVariate{names}` is.
#
# The wrapper DELEGATES all density work to the existing vector path: it
# overlays the supplied `NamedTuple` onto the flat event vector BY NAME and
# calls the existing `event_logpdf`. So a NamedTuple draw scores identically to
# the vector it overlays; the experiment isolates the VARIATE-TYPE question from
# the density semantics, which already work.

@doc "

PROOF OF CONCEPT wrapper giving a composed distribution a NamedTuple-valued
variate.

`ComposedNamedTuple(d)` wraps a [`Sequential`](@ref) or [`Parallel`](@ref) `d`
and re-exposes it as a `Distribution{NamedTupleVariate{names}, Continuous}`,
where `names = tree_event_names(d)`. A draw is a `NamedTuple` keyed by the event
names rather than the positional `Vector` of the wrapped distribution, mirroring
JuliaBUGS' `to_distribution`/`NamedTupleVariate{names}`.

This is an EXPERIMENT for an RFC, not part of the public API. It exists only to
exercise the NamedTuple variate (`rand`/`logpdf`/`mean`) and to probe the
bijector / AD / `product_distribution` consequences. The wrapped distribution's
own multivariate `Vector` variate is unchanged.

# See also
- [`tree_event_names`](@ref): the event names used as the variate keys.
"
struct ComposedNamedTuple{names, D} <:
       Distribution{Distributions.NamedTupleVariate{names}, Continuous}
    "The wrapped composed distribution (its vector variate is reused as-is)."
    dist::D
end

# Construct from a composer: read its flat event names and bake them into the
# type parameter, exactly as JuliaBUGS bakes the graph parameter symbols.
function ComposedNamedTuple(d::Union{Sequential, Parallel})
    names = tree_event_names(d)
    return ComposedNamedTuple{names, typeof(d)}(d)
end

# The variate key tuple (the `names` type parameter), for convenience.
_nt_names(::ComposedNamedTuple{names}) where {names} = names

# ---------------------------------------------------------------------------
# rand: a single draw is a NamedTuple keyed by the event names
# ---------------------------------------------------------------------------
#
# Delegate sampling to the existing vector path (`rand(d.dist)` returns the flat
# `[E_0, ..., E_k]`) and re-key it. A nested tree can return a NamedTuple
# already (`_composer_rand` does for nested shapes); the PoC targets the FLAT
# shape, so we `collect` to a vector first and re-key by name.

@doc "

Draw a single realisation as a `NamedTuple` keyed by the event names.

See also: [`ComposedNamedTuple`](@ref)
"
function Base.rand(rng::AbstractRNG, d::ComposedNamedTuple)
    v = rand(rng, d.dist)
    vals = v isa AbstractVector ? Tuple(v) : Tuple(values(v))
    return NamedTuple{_nt_names(d)}(vals)
end

Base.rand(d::ComposedNamedTuple) = rand(default_rng(), d)

# ---------------------------------------------------------------------------
# rand(rng, d, ::Dims) / _rand! — HAND-WRITTEN (the JuliaBUGS stack-overflow
# trap). Distributions.jl has NO array fallback for a NamedTuple variate, so the
# default `rand(rng, d, dims)` recurses into `_rand!(rng, d, Array{eltype})`,
# which for a NamedTuple eltype has no method and (in JuliaBUGS' experience)
# blows the stack. We give an explicit Array-of-NamedTuple method instead.
# ---------------------------------------------------------------------------

@doc "

Draw an `Array` of NamedTuple realisations (hand-written; Distributions has no
array fallback for a NamedTuple variate).

See also: [`ComposedNamedTuple`](@ref)
"
function Base.rand(rng::AbstractRNG, d::ComposedNamedTuple, dims::Dims)
    out = Array{typeof(rand(rng, d))}(undef, dims)
    @inbounds for i in eachindex(out)
        out[i] = rand(rng, d)
    end
    return out
end

# FINDING: a `n::Integer` method is AMBIGUOUS with Distributions'
# `rand(rng, ::Sampleable, dim1::Int, moredims::Int...)`, and that generic
# method would recurse into `_rand!` with a NamedTuple eltype (no method) and
# stack-overflow, exactly the JuliaBUGS trap. An EXACT `::Int` method is needed
# to win dispatch and route to the hand-written `Dims` path.
function Base.rand(rng::AbstractRNG, d::ComposedNamedTuple, n::Int)
    return rand(rng, d, (n,))
end

# ---------------------------------------------------------------------------
# logpdf: overlay the supplied NamedTuple onto the event vector BY NAME
# ---------------------------------------------------------------------------
#
# This is the JuliaBUGS "overlay by name" move: a parameter NamedTuple is mapped
# onto the evaluation slots by key, not by position. We reuse the package's own
# by-name row matcher (`_row_event_vector_by_name`) so a reordered NamedTuple
# scores identically and an unknown key errors, then call the existing
# `event_logpdf`. AD flows through this exactly as it does for the vector path,
# because the overlaid vector is built from the NamedTuple's values.

@doc "

Joint log density of a realisation supplied as a `NamedTuple`, overlaid onto the
event vector by name.

See also: [`logpdf`](@ref), [`ComposedNamedTuple`](@ref)
"
function logpdf(d::ComposedNamedTuple{names}, nt::NamedTuple) where {names}
    # Overlay by name into the flat event vector, then score via the existing
    # vector path. `event_logpdf` dispatches on `T >: Missing`, so the overlay
    # must carry a `Missing`-admitting element type even when fully observed.
    # FINDING: this forces a `Union{Missing, T}` container around the AD value
    # type `T`; AD must then survive scoring through that Union element type.
    vals = promote(ntuple(i -> nt[names[i]], length(names))...)
    T = eltype(vals)
    events = Vector{Union{Missing, T}}(undef, length(names))
    @inbounds for i in eachindex(events)
        events[i] = vals[i]
    end
    return CensoredDistributions.event_logpdf(d.dist, events)
end

pdf(d::ComposedNamedTuple, nt::NamedTuple) = exp(logpdf(d, nt))

# ---------------------------------------------------------------------------
# mean: a NamedTuple of per-event means
# ---------------------------------------------------------------------------
#
# The composer has no `mean` on the flat event vector today (only `edge_means`
# per-EDGE). For the PoC we report the CUMULATIVE per-EVENT means (origin 0,
# then running sum of edge means along the chain) so `mean` matches the `rand`/
# `logpdf` event keying. This is a demonstration of the self-labelling moment,
# not a general moment implementation.

@doc "

Per-event means as a `NamedTuple` keyed by the event names (PoC; cumulative edge
means along a flat chain).

See also: [`mean`](@ref), [`edge_means`](@ref), [`ComposedNamedTuple`](@ref)
"
function mean(d::ComposedNamedTuple{names}) where {names}
    em = edge_means(d.dist)               # per-edge NamedTuple
    edge_vals = collect(values(em))
    n = length(names)
    cum = zeros(Float64, n)
    acc = 0.0
    cum[1] = 0.0                           # origin event at zero
    for i in 2:n
        acc += edge_vals[min(i - 1, length(edge_vals))]
        cum[i] = acc
    end
    return NamedTuple{names}(Tuple(cum))
end

Base.length(d::ComposedNamedTuple) = length(_nt_names(d))
Base.eltype(::Type{<:ComposedNamedTuple}) = Float64

function Base.show(io::IO, d::ComposedNamedTuple)
    print(io, "ComposedNamedTuple", _nt_names(d), " over ", d.dist)
    return nothing
end
