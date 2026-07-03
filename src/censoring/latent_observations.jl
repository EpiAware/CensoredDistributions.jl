# Batched latent-form observation distributions over a vector of per-record
# censored distributions (`dists`), each carrying its own primary event window,
# secondary interval and truncation. A latent fit samples one primary per record
# from `PrimaryEvent(dists)` and scores the observed delays against those
# primaries with `PrimaryConditional(dists, ps)`, both in a single `~`.

@doc "

Prior over the per-record within-window primary event times of a latent fit.

`PrimaryEvent(dists)` is the product of each record's primary event prior, read
from its distribution with [`get_primary_event`](@ref), so the per-record windows
may differ. Sampling `ps ~ PrimaryEvent(dists)` in a Turing model makes the
within-window primaries named latent variables; the product form links natively
under NUTS. Returns a `Distributions.product_distribution`.

# Arguments
- `dists`: a vector of per-record censored distributions (for example
  [`double_interval_censored`](@ref) nodes, optionally [`latent`](@ref)-wrapped).

# See also
- [`PrimaryConditional`](@ref): scores the observed delays given these primaries.
- [`get_primary_event`](@ref): the per-record primary prior read here.
"
function PrimaryEvent(dists::AbstractVector)
    return product_distribution([get_primary_event(d) for d in dists])
end

@doc "

Conditional distribution of observed delays given realised primary events.

Two forms:
- scalar `PrimaryConditional(dist, p::Real)`: the univariate observed-time
  conditional of a single record given its primary `p` (the kernel [`latent`](@ref)
  scores and samples); see the struct docstring below.
- batched `PrimaryConditional(dists, ps)`: a multivariate distribution over a
  vector of records, scoring each observed delay against its own primary in one
  pass, so a Turing model can write `obs ~ PrimaryConditional(dists, ps)` instead
  of a per-record loop. Per-record windows may differ.

Each record's observed delay cannot precede its primary, so the secondary is
truncated below by the primary (see [`get_primary_event`](@ref)); a primary at or
after its whole interval is infeasible and scores `-Inf`.

# Arguments
- `dists`: a vector of per-record censored distributions.
- `ps`: the realised primary event time per record.

# See also
- [`PrimaryEvent`](@ref): the prior these primaries are sampled from.
"
struct BatchedPrimaryConditional{DS <: AbstractVector, PS <: AbstractVector} <:
       Distribution{Multivariate, Continuous}
    dists::DS
    ps::PS
    function BatchedPrimaryConditional(dists::AbstractVector, ps::AbstractVector)
        length(dists) == length(ps) || throw(DimensionMismatch(
            "dists and ps must have equal length; got $(length(dists)) and " *
            "$(length(ps))"))
        return new{typeof(dists), typeof(ps)}(dists, ps)
    end
end

# `PrimaryConditional(dists, ps)` (vector args) builds the batched multivariate
# form; `PrimaryConditional(dist, p::Real)` stays the scalar kernel.
function PrimaryConditional(dists::AbstractVector, ps::AbstractVector)
    BatchedPrimaryConditional(dists, ps)
end

Base.length(d::BatchedPrimaryConditional) = length(d.dists)
Base.eltype(::Type{<:BatchedPrimaryConditional}) = Float64

@doc "

Summed log density of a vector of observed delays under a batched
[`PrimaryConditional`](@ref): each delay scored against its record's primary via
the per-record conditional (secondary truncated below by the primary).
"
function logpdf(d::BatchedPrimaryConditional, ys::AbstractVector)
    length(ys) == length(d.dists) || throw(DimensionMismatch(
        "ys and dists must have equal length; got $(length(ys)) and " *
        "$(length(d.dists))"))
    return sum(
        logpdf(_conditional(d.dists[i], d.ps[i]), ys[i])
        for i in eachindex(ys); init = zero(eltype(d.ps)))
end

@doc "

Draw one observed delay per record from its primary-conditional secondary.
"
function Base.rand(rng::AbstractRNG, d::BatchedPrimaryConditional)
    return [rand(rng, _conditional(d.dists[i], d.ps[i]))
            for i in eachindex(d.dists)]
end
