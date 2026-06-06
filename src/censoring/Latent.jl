@doc "

Latent wrapper turning a primary-censored node into its latent representation.

A latent primary-censored node is multivariate over the event times
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

# Arguments
- `d`: A primary-censored node (for example from [`primary_censored`](@ref)).

# Examples
```@example
using CensoredDistributions, Distributions

d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
ld = latent(d)
rand(ld)
```
"
latent(d) = Latent(d)

Base.length(::Latent) = 2
Base.eltype(::Type{<:Latent{D}}) where {D} = eltype(D)
params(d::Latent) = params(d.dist)

@doc "

Draw a latent event-time vector `[primary, observed]`: a primary event time from
the primary prior, then the observed time from [`PrimaryConditional`](@ref)
given that primary.

See also: [`logpdf`](@ref)
"
function Base.rand(rng::AbstractRNG, d::Latent)
    p = rand(rng, get_primary_event(d))
    y = rand(rng, PrimaryConditional(d, p))
    return [p, y]
end

@doc "

Log density of the latent event-time vector `x = [primary, observed]`: the
primary prior density plus the [`PrimaryConditional`](@ref) of the observed time
given the primary,
`logpdf(get_primary_event(d), p) + logpdf(PrimaryConditional(d, p), y)`.

See also: [`PrimaryConditional`](@ref), [`rand`](@ref)
"
function logpdf(d::Latent, x::AbstractVector)
    p = x[1]
    y = x[2]
    return logpdf(get_primary_event(d), p) + logpdf(PrimaryConditional(d, p), y)
end
