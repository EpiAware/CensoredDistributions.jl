@doc "

Latent-variable wrapper turning a primary-censored delay into its latent
representation, sampling the primary event time rather than integrating it out.

A latent primary-censored delay is multivariate over the event times
`[primary, observed]`: the primary event time is a sampled latent variable
rather than integrated out. `rand` produces `[primary, observed]` and
`logpdf([primary, observed])` is the primary prior plus the conditional of the
observed time given the primary.

Marginal versus latent is dispatch on the type: the plain
[`PrimaryCensored`](@ref) is the univariate marginal default, and wrapping it in
`Latent` selects the latent representation. Construct via [`latent`](@ref).

The `dist` field holds the wrapped primary-censored node.

# See also
- [`latent`](@ref): constructor
- [`PrimaryConditional`](@ref): the conditional scored and sampled here
- [`get_primary_event`](@ref): the primary prior sampled here
"
struct Latent{D} <: Distribution{Multivariate, Continuous}
    "The wrapped primary-censored node."
    dist::D
end

@doc "

Turn a primary-censored node into its latent representation.

Returns a [`Latent`](@ref) multivariate distribution over `[primary, observed]`.
A draw is a labelled `NamedTuple` `(primary = ..., observed = ...)`.

# Arguments
- `d`: A primary-censored node (for example from [`primary_censored`](@ref)).

# Examples
```@example
using CensoredDistributions, Distributions

d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
ld = latent(d)
rand(ld)
```

# See also
- [`marginal`](@ref): the inverse, recovering the wrapped marginal node.
"
latent(d) = Latent(d)

@doc "

Recover the marginal node a [`latent`](@ref) wraps (the inverse of `latent`).

`marginal(d)` is the inverse of [`latent`](@ref): it unwraps a [`Latent`](@ref)
back to the marginal node it carries, so `marginal(latent(d)) == d`. It is
idempotent, so a node that is not a `Latent` is returned unchanged, giving
`marginal(d) == d` and `marginal(marginal(x)) == marginal(x)`. Use it to move
from the per-event latent view back to the collapsed marginal observed view.

# Arguments
- `d`: A [`Latent`](@ref) to unwrap, or any node (returned unchanged).

# Examples
```@example
using CensoredDistributions, Distributions

d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
marginal(latent(d)) == d
```

# See also
- [`latent`](@ref): the forward direction this inverts.
"
marginal(d::Latent) = d.dist
marginal(d) = d

Base.length(::Latent) = 2
Base.eltype(::Type{<:Latent{D}}) where {D} = eltype(D)
params(d::Latent) = params(d.dist)

@doc "

Draw a labelled latent event record `(primary = ..., observed = ...)`: a primary
event time from the primary prior, then the observed time from
[`PrimaryConditional`](@ref) given that primary. The draw is a `NamedTuple`
(self-labelling; the underlying scored representation is the vector
`[primary, observed]`).

See also: [`logpdf`](@ref)
"
function Base.rand(rng::AbstractRNG, d::Latent)
    p = rand(rng, get_primary_event(d))
    y = rand(rng, PrimaryConditional(d, p))
    return (primary = p, observed = y)
end

# Batch the count form into `n` independent labelled records. A `Latent` is
# multivariate but `rand(rng, d)` returns a labelled `NamedTuple`, not a numeric
# vector that can fill a matrix column, so the generic `rand(::Multivariate,
# ::Int)` matrix fallback recurses (StackOverflow). This terminating method
# returns one full record per draw, matching the record-aware `rand(d, rows)`
# batch shape. The `n::Integer` count form is disambiguated from
# `rand(d, rows::AbstractVector)`.
function Base.rand(rng::AbstractRNG, d::Latent, n::Integer)
    return [rand(rng, d) for _ in 1:n]
end

# Disambiguate against Distributions' `rand(::Sampleable{Multivariate,
# Continuous}, ::Int)`: `Latent` is more specific in the distribution argument
# but `Int` ties on the count, so spell out the `Int` method explicitly.
function Base.rand(rng::AbstractRNG, d::Latent, n::Int)
    invoke(
        rand, Tuple{AbstractRNG, Latent, Integer}, rng, d, n)
end

Base.rand(d::Latent, n::Integer) = rand(default_rng(), d, n)

# Disambiguate the no-rng count form against `rand(::Sampleable, ::Int...)`.
Base.rand(d::Latent, n::Int) = rand(default_rng(), d, n)

@doc "

Joint log density of the latent event record: the primary prior density plus
the [`PrimaryConditional`](@ref) of the observed time given the primary,
`logpdf(get_primary_event(d), p) + logpdf(PrimaryConditional(d, p), y)`.

Accepts either the scored vector `[primary, observed]` or the labelled
`NamedTuple` `(primary = ..., observed = ...)` (converted internally to the
scored vector). The latent form does not define a scalar observed density, since
the observed marginal is the marginal default's job, recovered by `marginal(d)`
(see [`PrimaryCensored`](@ref)).

See also: [`PrimaryConditional`](@ref), [`rand`](@ref)
"
function logpdf(d::Latent, x::AbstractVector)
    p = x[1]
    y = x[2]
    return logpdf(get_primary_event(d), p) + logpdf(PrimaryConditional(d, p), y)
end

# Accept the labelled NamedTuple draw, converting to the scored `[primary,
# observed]` vector internally. Keying is by name, so field order does not
# matter and an unexpected field errors.
function logpdf(d::Latent, x::NamedTuple)
    return logpdf(d, _latent_record_vector(x))
end

function _latent_record_vector(x::NamedTuple)
    Set(keys(x)) == Set((:primary, :observed)) || throw(ArgumentError(
        "a latent event record needs fields (:primary, :observed); got " *
        "$(collect(keys(x)))"))
    return [x.primary, x.observed]
end

# No scalar observed density (`logpdf(d::Latent, x::Real)`, `pdf`, `cdf`, ...).
# Re-marginalising the primary from a scalar observed value duplicates the
# marginal default and defeats latent mode; the observed marginal is the
# marginal node's job, recovered via `marginal(d)` (see `PrimaryCensored`). The
# latent scalar appears only as a sampled `~` event inside the composer models
# (see the DynamicPPL extension), not as a closed-form density on `Latent`.
