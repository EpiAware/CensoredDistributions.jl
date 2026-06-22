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
IDEMPOTENT — a node that is not a `Latent` is returned unchanged, so
`marginal(d) == d` and `marginal(marginal(x)) == marginal(x)`. Use it to move
from the per-event latent view back to the collapsed marginal/observed view.

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

@doc "

Joint log density of the latent event record: the primary prior density plus
the [`PrimaryConditional`](@ref) of the observed time given the primary,
`logpdf(get_primary_event(d), p) + logpdf(PrimaryConditional(d, p), y)`.

Accepts either the scored vector `[primary, observed]` or the labelled
`NamedTuple` `(primary = ..., observed = ...)` (converted internally to the
scored vector). For the single-value observed marginal, call `logpdf(d, x)`
with a scalar `x`, which delegates to `marginal(d)`.

See also: [`PrimaryConditional`](@ref), [`rand`](@ref)
"
function logpdf(d::Latent, x::AbstractVector)
    p = x[1]
    y = x[2]
    return logpdf(get_primary_event(d), p) + logpdf(PrimaryConditional(d, p), y)
end

# Accept the labelled NamedTuple draw, converting to the scored `[primary,
# observed]` vector internally. Keying is BY NAME, so field order does not
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

# Observed-delay marginal interface
# ---------------------------------
# The observed-delay marginal under the latent model is exactly the marginal
# distribution it wraps (the primary-censored node), which integrates the
# primary out analytically. So scoring a single observed value delegates
# straight to `marginal(d)`. These are the analytic counterparts of the joint
# `logpdf(d, [p, y])`: no Monte Carlo, no manual quadrature.

@doc "

Observed-delay marginal density of `x` under the latent model. The primary is
integrated out analytically, so this equals `pdf(marginal(d), x)`, the density
of the wrapped primary-censored node. Distinct from the joint
`pdf(d, [p, y])`.

See also: [`cdf`](@ref)
"
pdf(d::Latent, x::Real) = pdf(marginal(d), x)

@doc "

Observed-delay marginal log density of `x` under the latent model, the
single-value observed marginal. The primary is integrated out analytically, so
this equals `logpdf(marginal(d), x)`. Distinct from the joint
`logpdf(d, [p, y])` (the primary-explicit score over the scored vector).

See also: [`pdf`](@ref), [`cdf`](@ref)
"
logpdf(d::Latent, x::Real) = logpdf(marginal(d), x)

@doc "

Observed-delay cumulative distribution function of the latent model. The
primary is integrated out analytically, so this equals `cdf(marginal(d), x)`.

See also: [`logcdf`](@ref), [`ccdf`](@ref)
"
cdf(d::Latent, x::Real) = cdf(marginal(d), x)

@doc "

Observed-delay log cumulative distribution function of the latent model, equal
to `logcdf(marginal(d), x)`.

See also: [`cdf`](@ref)
"
logcdf(d::Latent, x::Real) = logcdf(marginal(d), x)

@doc "

Observed-delay complementary cumulative distribution function (survival) of the
latent model, equal to `ccdf(marginal(d), x)`.

See also: [`logccdf`](@ref), [`cdf`](@ref)
"
ccdf(d::Latent, x::Real) = ccdf(marginal(d), x)

@doc "

Observed-delay log complementary cumulative distribution function of the latent
model, equal to `logccdf(marginal(d), x)`.

See also: [`ccdf`](@ref)
"
logccdf(d::Latent, x::Real) = logccdf(marginal(d), x)

@doc "

Observed-delay quantile (inverse CDF) of the latent model, equal to
`quantile(marginal(d), q)`.

See also: [`cdf`](@ref)
"
quantile(d::Latent, q::Real) = quantile(marginal(d), q)
