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

The `dist` field holds the wrapped primary-censored distribution.

# See also
- [`latent`](@ref): constructor
- [`PrimaryConditional`](@ref): the distribution of the observed time given the
  primary, scored and sampled here
- [`get_primary_event`](@ref): the primary prior sampled here
"
struct Latent{D} <: Distribution{Multivariate, Continuous}
    "The wrapped primary-censored distribution."
    dist::D
end

@doc "

Turn a primary-censored distribution into its latent representation.

Returns a [`Latent`](@ref) multivariate distribution over `[primary, observed]`.
A draw is a labelled `NamedTuple` `(primary = ..., observed = ...)`.

# Arguments
- `d`: A primary-censored distribution (for example from
  [`primary_censored`](@ref)).

# Examples
```@example
using CensoredDistributions, Distributions

d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
ld = latent(d)
rand(ld)
```

# See also
- [`marginal`](@ref): the inverse, recovering the wrapped marginal
  distribution.
"
latent(d) = Latent(d)

@doc "

Recover the marginal distribution a [`latent`](@ref) wraps (the inverse of
`latent`).

`marginal(d)` is the inverse of [`latent`](@ref): it unwraps a [`Latent`](@ref)
back to the marginal distribution it carries, so `marginal(latent(d)) == d`. It
is idempotent, so a distribution that is not a `Latent` is returned unchanged,
giving
`marginal(d) == d` and `marginal(marginal(x)) == marginal(x)`. Use it to move
from the per-event latent view back to the collapsed marginal observed view.

# Arguments
- `d`: A [`Latent`](@ref) to unwrap, or any distribution (returned unchanged).

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

Log density of a [`latent`](@ref) distribution over a vector argument, in two
forms selected by the `primary` keyword.

Without `primary`, the vector `x` is a single scored joint record
`[primary, observed]` and the result is the joint log density, the primary prior
density plus the [`PrimaryConditional`](@ref) of the observed time given the
primary, `logpdf(get_primary_event(d), p) + logpdf(PrimaryConditional(d, p), y)`.
The labelled `NamedTuple` `(primary = ..., observed = ...)` is also accepted and
converted internally.

With a `primary` vector, `x` is a vector of observed times and the result is the
*batched* conditional log density: each observed time is scored against its own
primary for the single distribution and the per-record log densities are summed,
in one vectorised pass, so `logpdf(latent(d), ys; primary = ps)` scores a whole
vector of records in place of a per-record loop. `ys` and `primary` must have
equal length. Each record is
truncated below by its own primary (see [`get_primary_event`](@ref)); a record
whose primary falls at or after its whole interval is infeasible and contributes
`-Inf`. This scales the latent form to many records; the marginal is the separate
[`PrimaryCensored`](@ref) default, recovered by `marginal(d)`.

See also: [`PrimaryConditional`](@ref), [`rand`](@ref)
"
function logpdf(d::Latent, x::AbstractVector; primary = nothing)
    return _latent_vector_logpdf(d, x, primary)
end

# No `primary`: `x` is the scored joint record `[primary, observed]`.
function _latent_vector_logpdf(d::Latent, x::AbstractVector, ::Nothing)
    p = x[1]
    y = x[2]
    return logpdf(get_primary_event(d), p) + logpdf(PrimaryConditional(d, p), y)
end

# A `primary` vector: `x` is a vector of observed times scored per record against
# its own primary for the single distribution, summed (the batched conditional).
# Delegates
# to `_latent_batched_logpdf`, dispatched on the wrapped distribution so the
# truncation pipeline scores in one vectorised pass (see `secondary_conditional`).
function _latent_vector_logpdf(d::Latent, ys::AbstractVector,
        primary::AbstractVector)
    length(ys) == length(primary) || throw(DimensionMismatch(
        "ys and primary must have equal length; got $(length(ys)) and " *
        "$(length(primary))"))
    return _latent_batched_logpdf(d.dist, ys, primary)
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

# Resolve the conditioning primary: the passed value, or a fresh MC draw from the
# primary prior when none is given.
_latent_primary(::AbstractRNG, ::Latent, primary::Real) = primary
function _latent_primary(rng::AbstractRNG, d::Latent, ::Nothing)
    return rand(rng, get_primary_event(d))
end

@doc "

Scalar observed density / tail / quantile of a [`latent`](@ref) distribution,
conditional on a primary event.

`latent(d)` is always conditional, never the integrating marginal. With a primary
passed (`logpdf(d, y; primary = p)`) the call is the deterministic
[`PrimaryConditional`](@ref)`(d, p)` kernel; with no primary one is Monte-Carlo
sampled from [`get_primary_event`](@ref)`(d)` and conditioned on, a stochastic
single-draw estimate. The integrating marginal is the separate
[`PrimaryCensored`](@ref), recovered via [`marginal`](@ref).

See also: [`PrimaryConditional`](@ref), [`logpdf`](@ref)
"
function logpdf(d::Latent, y::Real; primary = nothing,
        rng::AbstractRNG = default_rng())
    return logpdf(PrimaryConditional(d, _latent_primary(rng, d, primary)), y)
end

for f in (:pdf, :cdf, :logcdf, :ccdf, :logccdf)
    @eval function $f(d::Latent, y::Real; primary = nothing,
            rng::AbstractRNG = default_rng())
        return $f(PrimaryConditional(d, _latent_primary(rng, d, primary)), y)
    end
end

# Quantile of the observed time given the primary (sampled when none passed). The
# bare primary-censored distribution has a closed-form conditional quantile; an
# interval/truncation pipeline conditional does not.
function quantile(d::Latent, q::Real; primary = nothing,
        rng::AbstractRNG = default_rng())
    return quantile(PrimaryConditional(d, _latent_primary(rng, d, primary)), q)
end

@doc "

Draw a single observed time from a [`latent`](@ref) distribution, conditional on
a primary event.

`rand_observed` is the conditional dual of the joint `rand(d)` record: with a
primary passed (`rand_observed(d; primary = p)`) the draw is
`rand(`[`PrimaryConditional`](@ref)`(d, p))`; with none, one primary is
Monte-Carlo sampled from [`get_primary_event`](@ref)`(d)` first. It returns the
scalar observed time, where the joint `rand(d)` returns the labelled
`(primary, observed)` record.

# Arguments
- `d`: A [`latent`](@ref) distribution.
- `primary`: The primary event time to condition on; sampled from the primary
  prior when omitted.

# Examples
```@example
using CensoredDistributions, Distributions
using CensoredDistributions: rand_observed

ld = latent(primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1)))
rand_observed(ld; primary = 0.5)
```

# See also
- [`rand`](@ref): the joint `(primary, observed)` draw.
- [`PrimaryConditional`](@ref): the conditional this samples from.
"
function rand_observed(rng::AbstractRNG, d::Latent; primary = nothing)
    return rand(rng, PrimaryConditional(d, _latent_primary(rng, d, primary)))
end
function rand_observed(d::Latent; primary = nothing)
    return rand_observed(default_rng(), d; primary = primary)
end
