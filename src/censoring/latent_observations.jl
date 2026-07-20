# Batched latent-form observation distributions: `PrimaryEvent` (the per-record
# primary prior) and `PrimaryConditional` (observed delays given the primaries).

@doc "

Prior over the within-window primary event time(s) of a latent fit, extracted
from the censored distribution(s) with [`get_primary_event`](@ref) so the user
feeds the delay distribution and `PrimaryEvent` pulls out the primary.

- `PrimaryEvent(d)` (scalar): the single record's primary prior,
  `get_primary_event(d)`; `rand(PrimaryEvent(d))` draws its one primary.
- `PrimaryEvent(dists)` (vector): the product of the per-record primary priors,
  so the per-record windows may differ and `rand(PrimaryEvent(dists))` draws one
  primary per record. The product form is a `Distributions.product_distribution`,
  which links natively when used as a prior in a probabilistic program.

# Arguments
- `d` / `dists`: a censored distribution or a vector of them (for example
  [`double_interval_censored`](@ref) distributions, optionally
  [`latent`](@ref)-wrapped).

# Examples
```@example
using CensoredDistributions, Distributions

dists = [double_interval_censored(LogNormal(1.5, 0.75);
             primary_event = Uniform(0, 1), upper = 8, interval = 1),
    double_interval_censored(LogNormal(1.5, 0.75);
        primary_event = Uniform(0, 3), upper = 12, interval = 3)]
prior = PrimaryEvent(dists)   # product over the per-record primary priors
ps = rand(prior)              # one primary per record
logpdf(prior, ps)             # scored as a plain distribution
PrimaryEvent(dists[1])        # scalar: one record's primary prior
```

# See also
- [`PrimaryConditional`](@ref): the distribution of the observed delays given
  these primaries.
- [`get_primary_event`](@ref): the per-record primary prior read here.
"
PrimaryEvent(d::Distribution) = get_primary_event(d)
function PrimaryEvent(dists::AbstractVector)
    return product_distribution([get_primary_event(d) for d in dists])
end

"""
Internal multivariate distribution backing the batched
`PrimaryConditional(dists, ps)`. Holds the per-record distributions `dists` and
their realised primaries `ps` (equal length); users build it through the public
`PrimaryConditional` constructor, not directly.
"""
struct _BatchedPrimaryConditional{DS <: AbstractVector, PS <: AbstractVector} <:
       Distribution{Multivariate, Continuous}
    dists::DS
    ps::PS
    function _BatchedPrimaryConditional(dists::AbstractVector,
            ps::AbstractVector)
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
one pass, so `logpdf(PrimaryConditional(dists, ps), obs)` replaces a per-record
loop. Per-record windows may differ. This complements the scalar
`PrimaryConditional(dist, p::Real)` kernel (see the struct docstring).

Each record's observed delay cannot precede its primary, so the secondary is
truncated below by the primary (see [`get_primary_event`](@ref)); a primary at or
after its whole interval is infeasible and scores `-Inf`.

# Arguments
- `dists`: a vector of per-record censored distributions.
- `ps`: the realised primary event time per record.

# See also
- [`PrimaryEvent`](@ref): the distribution of the primary event times these
  condition on.
"
function PrimaryConditional(dists::AbstractVector, ps::AbstractVector)
    return _BatchedPrimaryConditional(dists, ps)
end

Base.length(d::_BatchedPrimaryConditional) = length(d.dists)
Base.eltype(::Type{<:_BatchedPrimaryConditional}) = Float64

# A vector of observed delays is in support when every record is in the support
# of its own primary-conditional secondary.
function insupport(d::_BatchedPrimaryConditional, ys::AbstractVector)
    length(ys) == length(d.dists) || return false
    return all(insupport(_conditional(d.dists[i], d.ps[i]), ys[i])
    for i in eachindex(ys))
end

@doc "

Summed log density of a vector of observed delays under a batched
[`PrimaryConditional`](@ref): each delay scored against its record's primary via
the per-record conditional (secondary truncated below by the primary).
"
function logpdf(d::_BatchedPrimaryConditional, ys::AbstractVector)
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
function Base.rand(rng::AbstractRNG, d::_BatchedPrimaryConditional)
    return [rand(rng, _conditional(d.dists[i], d.ps[i]))
            for i in eachindex(d.dists)]
end
