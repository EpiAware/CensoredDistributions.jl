@doc "

Conditional distribution of the observed time given a realised primary `p`.

For a primary-censored node `d` with delay `get_dist(d)`, the observed time is
the primary plus the delay, so conditioning on a realised primary `p` shifts the
delay: `logpdf` scores `logpdf(get_dist(d), y - p)` (support `y > p`) and `rand`
returns `p + rand(get_dist(d))`.

This is the single source of the conditional: [`Latent`](@ref) and
[`primary_conditional_logpdf`](@ref) both go through it. Turing-free, so it can
be used with `~` in a model — `y ~ PrimaryConditional(d, p)` both scores an
observed `y` and generates a missing one, with `p` the sampled latent primary.

The `dist` field holds the primary-censored node; the `p` field holds the
realised primary event time.

# See also
- [`primary_conditional_logpdf`](@ref): the named `logpdf` entry point
- [`Latent`](@ref): the joint that reuses this conditional
- [`get_dist`](@ref): the delay distribution scored here
"
struct PrimaryConditional{D, P <: Real} <: UnivariateDistribution{Continuous}
    "The primary-censored node (or its `Latent` wrapper)."
    dist::D
    "The realised primary event time conditioned on."
    p::P
end

minimum(d::PrimaryConditional) = d.p + minimum(get_dist(d))
maximum(d::PrimaryConditional) = d.p + maximum(get_dist(d))
insupport(d::PrimaryConditional, y::Real) = insupport(get_dist(d), y - d.p)
params(d::PrimaryConditional) = (params(get_dist(d))..., d.p)
Base.eltype(::Type{<:PrimaryConditional{D}}) where {D} = eltype(D)

@doc "

Log density of the observed time `y` given the primary `p`: the delay density at
the implied gap, `logpdf(get_dist(d), y - p)`.

See also: [`pdf`](@ref)
"
logpdf(d::PrimaryConditional, y::Real) = logpdf(get_dist(d), y - d.p)

@doc "

Density of the observed time `y` given the primary `p`.

See also: [`logpdf`](@ref)
"
pdf(d::PrimaryConditional, y::Real) = pdf(get_dist(d), y - d.p)

@doc "

Cumulative distribution function of the observed time given the primary `p`.

See also: [`logcdf`](@ref)
"
cdf(d::PrimaryConditional, y::Real) = cdf(get_dist(d), y - d.p)

@doc "

Log cumulative distribution function.

See also: [`cdf`](@ref)
"
logcdf(d::PrimaryConditional, y::Real) = logcdf(get_dist(d), y - d.p)

@doc "

Draw an observed time given the primary `p`: `p + rand(get_dist(d))`.

See also: [`logpdf`](@ref)
"
Base.rand(rng::AbstractRNG, d::PrimaryConditional) = d.p + rand(rng, get_dist(d))

@doc "

Conditional log density of the observed time `y` given a primary event time `p`.

The named entry point for [`PrimaryConditional`](@ref); equal to
`logpdf(PrimaryConditional(d, p), y) = logpdf(get_dist(d), y - p)`. Turing-free:
`p` is the sampled latent primary, scored against the delay at the implied gap.

# Arguments
- `d`: A primary-censored node, or its [`Latent`](@ref) wrapper.
- `p`: A primary event time (the sampled latent).
- `y`: The observed time.

# Examples
```@example
using CensoredDistributions, Distributions

d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
primary_conditional_logpdf(d, 0.3, 2.7)
```
"
function primary_conditional_logpdf(d, p, y)
    return logpdf(PrimaryConditional(d, p), y)
end
