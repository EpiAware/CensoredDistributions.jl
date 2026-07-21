@doc "

A distribution wrapper that lets a per-record `Bool` flag select, at
`logpdf` evaluation time, between the exact density of a distribution and
the density of its censored (or truncated, or interval-censored) form.

A dataset often mixes exact observations with bounded ones: some records
observed precisely, others only known to fall in an interval, or only
known to not yet have occurred by some time. `CensoringIndicator` lets
such a dataset be scored in one pass without splitting the data by
observation type: the wrapped distribution `dist` supplies the bounded
(censored) contribution directly, and its unwrapped form (via
[`get_dist`](@ref)) supplies the exact contribution.

Only `logpdf` on a joint observation is affected by the indicator - every
other method (pdf, cdf, sampling, etc.) delegates directly to the wrapped
`dist`, exactly as if no indicator had been supplied.

# Examples
```@example
using CensoredDistributions, Distributions

# A LogNormal delay, interval-censored to whole-day bins.
delay = LogNormal(1.5, 0.75)
ic = interval_censored(delay, 1.0)
d = indicate_censoring(ic)

# A record observed exactly scores against the underlying LogNormal ...
logpdf(d, (value = 2.0, exact = true))

# ... and a record only known to fall in its day-bin scores against the
# interval-censored form, identically to a censored leaf with no indicator.
logpdf(d, (value = 2.0, exact = false)) == logpdf(ic, 2.0)
```
"
struct CensoringIndicator{D <: UnivariateDistribution} <:
       UnivariateDistribution{ValueSupport}
    "The wrapped (censored, truncated, or interval-censored) distribution."
    dist::D
end

@doc "

Wrap `dist` so that `logpdf` on a joint observation `(value, exact)` can
select the exact contribution (`logpdf` of `dist` unwrapped one level via
[`get_dist`](@ref)) or the censored contribution (`logpdf` of `dist`
itself) per record.

`dist` is typically already a censored, truncated, or interval-censored
distribution (e.g. built with [`interval_censored`](@ref),
[`primary_censored`](@ref), `Distributions.censored`, or
`Distributions.truncated`); the exact contribution is derived from it via
`get_dist`, so no second distribution needs to be constructed or kept in
sync by the caller.

# Arguments
- `dist`: The (typically censored/truncated) distribution to wrap.

# Examples
```@example
using CensoredDistributions, Distributions

ic = interval_censored(LogNormal(1.5, 0.75), 1.0)
d = indicate_censoring(ic)
logpdf(d, (value = 2.0, exact = true))
```

# See also
- [`CensoringIndicator`](@ref): the wrapper type this constructs.
- [`get_dist`](@ref): how the exact contribution's distribution is derived.
"
function indicate_censoring(dist::UnivariateDistribution)
    return CensoringIndicator(dist)
end

# ============================================================================
# Distributions.jl Interface - delegates to the wrapped `dist`
# ============================================================================

Base.eltype(::Type{<:CensoringIndicator{D}}) where {D} = eltype(D)
minimum(d::CensoringIndicator) = minimum(d.dist)
maximum(d::CensoringIndicator) = maximum(d.dist)
insupport(d::CensoringIndicator, x::Real) = insupport(d.dist, x)
params(d::CensoringIndicator) = params(d.dist)

pdf(d::CensoringIndicator, x::Real) = pdf(d.dist, x)
cdf(d::CensoringIndicator, x::Real) = cdf(d.dist, x)
logcdf(d::CensoringIndicator, x::Real) = logcdf(d.dist, x)
ccdf(d::CensoringIndicator, x::Real) = ccdf(d.dist, x)
logccdf(d::CensoringIndicator, x::Real) = logccdf(d.dist, x)
quantile(d::CensoringIndicator, p::Real) = quantile(d.dist, p)
Base.rand(rng::AbstractRNG, d::CensoringIndicator) = rand(rng, d.dist)
sampler(d::CensoringIndicator) = sampler(d.dist)

@doc "

Score a scalar observation as the censored (bounded) contribution, exactly
as `dist` itself would score it. Used when no per-record indicator is
supplied; see the joint-observation `logpdf` method for the per-record
form.

See also: [`logpdf`](@ref)
"
function logpdf(d::CensoringIndicator, x::Real)
    return logpdf(d.dist, x)
end

@doc "

Score a joint observation `(value, exact)`: the exact density of
`get_dist(dist)` when `exact` is `true`, or `dist`'s own (censored)
density otherwise.

`exact` is data carried alongside the observation, not a model parameter,
so branching on it is safe under every AD backend the package supports -
no differentiated quantity is inspected by the branch.

See also: [`CensoringIndicator`](@ref), [`indicate_censoring`](@ref)
"
function logpdf(d::CensoringIndicator, obs::NamedTuple{(:value, :exact)})
    return obs.exact ? logpdf(get_dist(d.dist), obs.value) : logpdf(d.dist, obs.value)
end

@doc "

Score a table of joint observations `(values, exacts)`, one record per
index: the sum of each record's single-observation log density.

See also: [`logpdf`](@ref)
"
function loglikelihood(
        d::CensoringIndicator, obs::NamedTuple{(:values, :exacts)})
    return sum(logpdf(d, (value = v, exact = e))
    for (v, e) in zip(obs.values, obs.exacts))
end

@doc "

Score a single joint observation `(value, exact)` as a log-likelihood
(identical to [`logpdf`](@ref) for a single record).
"
function loglikelihood(
        d::CensoringIndicator, obs::NamedTuple{(:value, :exact)})
    return logpdf(d, obs)
end
