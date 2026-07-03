# Batched latent-form observation distributions over a vector of per-record
# censored distributions (`dists`), each carrying its own primary event window,
# secondary interval and truncation. A latent fit samples one primary per record
# from `PrimaryEvent(dists)` and scores the observed delays against those
# primaries with `PrimaryConditional(dists, ps)`, both in a single `~`.

@doc "

Prior over the within-window primary event time(s) of a latent fit, extracted
from the censored distribution(s) with [`get_primary_event`](@ref) so the user
feeds the delay distribution and `PrimaryEvent` pulls out the primary.

- `PrimaryEvent(d)` (scalar): the single record's primary prior,
  `get_primary_event(d)`; `ps ~ PrimaryEvent(d)` samples its one primary.
- `PrimaryEvent(dists)` (vector): the product of the per-record primary priors,
  so the per-record windows may differ and `ps ~ PrimaryEvent(dists)` samples one
  primary per record as named latent variables. The product form links natively
  under NUTS (returns a `Distributions.product_distribution`).

# Arguments
- `d` / `dists`: a censored distribution or a vector of them (for example
  [`double_interval_censored`](@ref) nodes, optionally [`latent`](@ref)-wrapped).

# Examples
```@example
using CensoredDistributions, Distributions

dists = [double_interval_censored(LogNormal(1.5, 0.75);
             primary_event = Uniform(0, 1), upper = 8, interval = 1),
    double_interval_censored(LogNormal(1.5, 0.75);
        primary_event = Uniform(0, 3), upper = 12, interval = 3)]
PrimaryEvent(dists)       # product over the per-record primary priors
PrimaryEvent(dists[1])    # scalar: one record's primary prior
```

# See also
- [`PrimaryConditional`](@ref): scores the observed delays given these primaries.
- [`get_primary_event`](@ref): the per-record primary prior read here.
"
PrimaryEvent(d::Distribution) = get_primary_event(d)
function PrimaryEvent(dists::AbstractVector)
    return product_distribution([get_primary_event(d) for d in dists])
end

# Internal multivariate type backing the batched `PrimaryConditional(dists, ps)`;
# users construct it through that public constructor, not directly.
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

@doc "

Batched conditional distribution of observed delays given realised primaries.

`PrimaryConditional(dists, ps)` is a multivariate distribution over a vector of
per-record distributions, scoring each observed delay against its own primary in
one pass, so a Turing model can write `obs ~ PrimaryConditional(dists, ps)`
instead of a per-record loop. Per-record windows may differ. This complements the
scalar `PrimaryConditional(dist, p::Real)` kernel (see the struct docstring).

Each record's observed delay cannot precede its primary, so the secondary is
truncated below by the primary (see [`get_primary_event`](@ref)); a primary at or
after its whole interval is infeasible and scores `-Inf`.

# Arguments
- `dists`: a vector of per-record censored distributions.
- `ps`: the realised primary event time per record.

# See also
- [`PrimaryEvent`](@ref): the prior these primaries are sampled from.
"
function PrimaryConditional(dists::AbstractVector, ps::AbstractVector)
    return BatchedPrimaryConditional(dists, ps)
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
