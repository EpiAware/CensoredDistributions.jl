@doc "

A distribution wrapper that lets a per-record `Bool` flag select, at
scoring time, between the density of a distribution's censored (or
truncated, or interval-censored) form and the exact density of its
unwrapped form.

A dataset often mixes exact observations with bounded ones: some records
observed precisely, others only known to fall in an interval, or only
known to not yet have occurred by some time. `CensoringIndicator` lets
such a dataset be scored in one pass without splitting the data by
observation type: the wrapped distribution `dist` supplies the bounded
(censored) contribution directly, and its unwrapped form (via
[`get_dist`](@ref)) supplies the exact contribution.

`pdf`, `logpdf`, `cdf`, `logcdf`, `ccdf`, and `logccdf` all recognise a
joint observation `(value, censored)` and dispatch on `censored` the same
way; `quantile`, sampling, and the other basic distribution methods take
no observation to branch on, so they delegate directly to the wrapped
`dist`, exactly as if no indicator had been supplied.

# Examples
```@example
using CensoredDistributions, Distributions

# A LogNormal delay, interval-censored to whole-day bins.
delay = LogNormal(1.5, 0.75)
ic = interval_censored(delay, 1.0)
d = indicate_censoring(ic)

# A record only known to fall in its day-bin scores against the
# interval-censored form, identically to a censored leaf with no
# indicator.
logpdf(d, (value = 2.0, censored = true)) == logpdf(ic, 2.0)

# ... while a record observed exactly scores against the underlying
# LogNormal.
logpdf(d, (value = 2.0, censored = false))
```
"
struct CensoringIndicator{D <: UnivariateDistribution} <:
       UnivariateDistribution{ValueSupport}
    "The wrapped (censored, truncated, or interval-censored) distribution."
    dist::D
end

@doc "

Wrap `dist` so that `pdf`, `logpdf`, `cdf`, `logcdf`, `ccdf`, and `logccdf`
on a joint observation `(value, censored)` can select the censored
contribution (`dist` itself) or the exact contribution (`dist` unwrapped
one level via [`get_dist`](@ref)) per record.

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
logpdf(d, (value = 2.0, censored = true))
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

quantile(d::CensoringIndicator, p::Real) = quantile(d.dist, p)
Base.rand(rng::AbstractRNG, d::CensoringIndicator) = rand(rng, d.dist)
sampler(d::CensoringIndicator) = sampler(d.dist)

# ============================================================================
# Scalar scoring - defaults to the censored (bounded) contribution
# ============================================================================

@doc "

Score a scalar observation as the censored (bounded) contribution, exactly
as `dist` itself would score it. Used when no per-record indicator is
supplied; see the joint-observation `pdf` method for the per-record form.

See also: [`logpdf`](@ref)
"
function pdf(d::CensoringIndicator, x::Real)
    return pdf(d.dist, x)
end

@doc "

Score a scalar observation as the censored (bounded) contribution, exactly
as `dist` itself would score it. Used when no per-record indicator is
supplied; see the joint-observation `logpdf` method for the per-record
form.

See also: [`pdf`](@ref)
"
function logpdf(d::CensoringIndicator, x::Real)
    return logpdf(d.dist, x)
end

@doc "

Score a scalar observation as the censored (bounded) contribution, exactly
as `dist` itself would score it.

See also: [`logcdf`](@ref)
"
function cdf(d::CensoringIndicator, x::Real)
    return cdf(d.dist, x)
end

@doc "

Score a scalar observation as the censored (bounded) contribution, exactly
as `dist` itself would score it.

See also: [`cdf`](@ref)
"
function logcdf(d::CensoringIndicator, x::Real)
    return logcdf(d.dist, x)
end

@doc "

Score a scalar observation as the censored (bounded) contribution, exactly
as `dist` itself would score it.

See also: [`logccdf`](@ref)
"
function ccdf(d::CensoringIndicator, x::Real)
    return ccdf(d.dist, x)
end

@doc "

Score a scalar observation as the censored (bounded) contribution, exactly
as `dist` itself would score it.

See also: [`ccdf`](@ref)
"
function logccdf(d::CensoringIndicator, x::Real)
    return logccdf(d.dist, x)
end

# ============================================================================
# Joint-observation scoring - `censored` selects the branch per record
# ============================================================================

@doc "

Select the censored branch (`dist` itself) or the exact branch (`dist`
unwrapped one level via [`get_dist`](@ref)) for a joint observation
`(value, censored)`, then apply scoring function `f` to the chosen
distribution and value.

`censored` is data carried alongside the observation, not a model
parameter, so branching on it is safe under every AD backend the package
supports - no differentiated quantity is inspected by the branch.
"
function _score(f, d::CensoringIndicator, obs::NamedTuple{(:value, :censored)})
    return obs.censored ? f(d.dist, obs.value) : f(get_dist(d.dist), obs.value)
end

@doc "

Score a joint observation `(value, censored)`: the density of `dist`
itself when `censored` is `true`, or the exact density of
`get_dist(dist)` otherwise.

See also: [`logpdf`](@ref)
"
function pdf(d::CensoringIndicator, obs::NamedTuple{(:value, :censored)})
    return _score(pdf, d, obs)
end

@doc "

Score a joint observation `(value, censored)`: the log density of `dist`
itself when `censored` is `true`, or the exact log density of
`get_dist(dist)` otherwise.

`censored` is data carried alongside the observation, not a model
parameter, so branching on it is safe under every AD backend the package
supports - no differentiated quantity is inspected by the branch.

See also: [`CensoringIndicator`](@ref), [`indicate_censoring`](@ref)
"
function logpdf(d::CensoringIndicator, obs::NamedTuple{(:value, :censored)})
    return _score(logpdf, d, obs)
end

@doc "

Score a joint observation `(value, censored)`: the cumulative probability
under `dist` itself when `censored` is `true`, or under `get_dist(dist)`
otherwise.

See also: [`logcdf`](@ref)
"
function cdf(d::CensoringIndicator, obs::NamedTuple{(:value, :censored)})
    return _score(cdf, d, obs)
end

@doc "

Score a joint observation `(value, censored)`: the log cumulative
probability under `dist` itself when `censored` is `true`, or under
`get_dist(dist)` otherwise.

See also: [`cdf`](@ref)
"
function logcdf(d::CensoringIndicator, obs::NamedTuple{(:value, :censored)})
    return _score(logcdf, d, obs)
end

@doc "

Score a joint observation `(value, censored)`: the complementary
cumulative probability under `dist` itself when `censored` is `true`, or
under `get_dist(dist)` otherwise.

See also: [`logccdf`](@ref)
"
function ccdf(d::CensoringIndicator, obs::NamedTuple{(:value, :censored)})
    return _score(ccdf, d, obs)
end

@doc "

Score a joint observation `(value, censored)`: the log complementary
cumulative probability under `dist` itself when `censored` is `true`, or
under `get_dist(dist)` otherwise.

See also: [`ccdf`](@ref)
"
function logccdf(d::CensoringIndicator, obs::NamedTuple{(:value, :censored)})
    return _score(logccdf, d, obs)
end

@doc "

Score a table of joint observations `(values, censored)`, one record per
index: the sum of each record's single-observation log density.

See also: [`logpdf`](@ref)
"
function loglikelihood(
        d::CensoringIndicator, obs::NamedTuple{(:values, :censored)})
    return sum(logpdf(d, (value = v, censored = c))
    for (v, c) in zip(obs.values, obs.censored))
end

@doc "

Score a single joint observation `(value, censored)` as a log-likelihood
(identical to [`logpdf`](@ref) for a single record).
"
function loglikelihood(
        d::CensoringIndicator, obs::NamedTuple{(:value, :censored)})
    return logpdf(d, obs)
end
