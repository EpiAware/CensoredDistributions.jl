@doc "

Conditional distribution of the observed time given a realised primary `p`.

For a primary-censored distribution `d` with delay `get_dist(d)`, the observed
time is the primary plus the delay, so conditioning on a realised primary `p`
shifts the delay: `logpdf` scores `logpdf(get_dist(d), y - p)` (support `y > p`)
and `rand` returns `p + rand(get_dist(d))`.

When the distribution also carries secondary interval censoring and/or
truncation (a [`double_interval_censored`](@ref) pipeline), those modifiers are
kept on the secondary: conditioning on `p` keeps the interval censoring and the
truncation on the total time `p + delay`, rather than stripping to the bare
continuous delay. So `latent` over a `double_interval_censored` distribution
samples the primary `p` and scores the interval-censored, truncated secondary
against it, and the joint integrates over `p` to the analytic marginal.

This is the single source of the conditional that [`Latent`](@ref) scores and
samples: `logpdf(PrimaryConditional(d, p), y)` scores an observed `y` and
`rand(PrimaryConditional(d, p))` draws one, with `p` the realised primary event
time. It depends only on Distributions, so it plugs into a probabilistic program
without pulling in Turing.

The `dist` field holds the primary-censored distribution (or the
interval/truncation pipeline over it); the `p` field holds the realised primary
event time.

# See also
- [`Latent`](@ref): the joint that reuses this conditional
- [`get_dist`](@ref): the delay distribution scored here
"
struct PrimaryConditional{D, P <: Real} <: AbstractPrimaryCensored
    "The primary-censored distribution (or its `Latent` / pipeline wrapper)."
    dist::D
    "The realised primary event time conditioned on."
    p::P
end

# The conditional secondary distribution scored against the observed time `y`
# given the realised primary `p`. The wrapped distribution selects the form:
#   - a `PrimaryCensored` (or a `Latent` over one) gives the bare continuous
#     delay shifted by `p`, so `logpdf` is `logpdf(delay, y - p)`;
#   - an `IntervalCensored` / `Truncated` pipeline over a `PrimaryCensored` keeps
#     the secondary interval and truncation on the total `p + delay` (built in
#     `secondary_conditional.jl`, included after the pipeline types).
# Dispatch on the wrapped distribution type picks the form. The `Latent`-unwrap
# method lives in `secondary_conditional.jl`, included after `Latent` is defined.
_conditional(d::PrimaryConditional) = _conditional(d.dist, d.p)

# Bare primary-censored distribution: the secondary is the continuous delay
# shifted by `p` (the sampled-origin rule the joint already uses).
_conditional(d::PrimaryCensored, p) = _ShiftedDelayCore(d.dist, p)

# Fail loud on an unknown inner distribution: a wrapper that changes the delay
# density (e.g. weighting, reparameterisation) has no defined primary-conditional,
# and silently unwrapping it would score a wrong density. The `Latent`,
# `IntervalCensored`/`Truncated` and `PrimaryCensored` methods are more specific,
# so only genuinely-unsupported distributions reach this.
function _conditional(d, p)
    throw(ArgumentError(
        "conditioning a latent observation on its primary event is not defined " *
        "for $(typeof(d)); supported inner distributions are PrimaryCensored, " *
        "IntervalCensored, Truncated (and their combinations). A wrapper that " *
        "changes the delay density has no defined primary-conditional — use the " *
        "marginal form or unwrap it."))
end

"""
Internal continuous secondary of a bare primary-censored conditional: the delay
shifted by the primary, so `logpdf(y) == logpdf(delay, y - shift)` with support
`y > shift + minimum(delay)`. Fields `delay` and `shift` (the realised primary).
"""
struct _ShiftedDelayCore{D, S} <: UnivariateDistribution{Continuous}
    delay::D
    shift::S
end
minimum(d::_ShiftedDelayCore) = d.shift + minimum(d.delay)
maximum(d::_ShiftedDelayCore) = d.shift + maximum(d.delay)
insupport(d::_ShiftedDelayCore, y::Real) = insupport(d.delay, y - d.shift)
logpdf(d::_ShiftedDelayCore, y::Real) = logpdf(d.delay, y - d.shift)
pdf(d::_ShiftedDelayCore, y::Real) = pdf(d.delay, y - d.shift)
cdf(d::_ShiftedDelayCore, y::Real) = cdf(d.delay, y - d.shift)
logcdf(d::_ShiftedDelayCore, y::Real) = logcdf(d.delay, y - d.shift)
ccdf(d::_ShiftedDelayCore, y::Real) = ccdf(d.delay, y - d.shift)
logccdf(d::_ShiftedDelayCore, y::Real) = logccdf(d.delay, y - d.shift)
quantile(d::_ShiftedDelayCore, q::Real) = d.shift + quantile(d.delay, q)
mean(d::_ShiftedDelayCore) = d.shift + mean(d.delay)
Base.rand(rng::AbstractRNG, d::_ShiftedDelayCore) = d.shift + rand(rng, d.delay)

minimum(d::PrimaryConditional) = minimum(_conditional(d))
maximum(d::PrimaryConditional) = maximum(_conditional(d))
insupport(d::PrimaryConditional, y::Real) = insupport(_conditional(d), y)
mean(d::PrimaryConditional) = mean(_conditional(d))
params(d::PrimaryConditional) = (params(get_dist(d))..., d.p)
Base.eltype(::Type{<:PrimaryConditional{D}}) where {D} = eltype(D)

@doc "

Log density of the observed time `y` given the primary `p`. For a bare
primary-censored distribution this is the delay density at the implied gap,
`logpdf(get_dist(d), y - p)`; for an interval/truncation pipeline it is the
interval-censored, truncated mass of the total `p + delay`.

See also: [`pdf`](@ref)
"
logpdf(d::PrimaryConditional, y::Real) = logpdf(_conditional(d), y)

@doc "

Density of the observed time `y` given the primary `p`.

See also: [`logpdf`](@ref)
"
pdf(d::PrimaryConditional, y::Real) = pdf(_conditional(d), y)

@doc "

Cumulative distribution function of the observed time given the primary `p` (bare
secondary only; the interval/truncation pipeline scores via `logpdf`).

See also: [`logcdf`](@ref)
"
cdf(d::PrimaryConditional, y::Real) = cdf(_conditional(d), y)

@doc "

Log cumulative distribution function.

See also: [`cdf`](@ref)
"
logcdf(d::PrimaryConditional, y::Real) = logcdf(_conditional(d), y)

@doc "

Complementary cumulative distribution function of the observed time given the
primary `p` (bare secondary only; the interval/truncation pipeline scores via
`logpdf`).

See also: [`cdf`](@ref)
"
ccdf(d::PrimaryConditional, y::Real) = ccdf(_conditional(d), y)

@doc "

Log complementary cumulative distribution function.

See also: [`ccdf`](@ref)
"
logccdf(d::PrimaryConditional, y::Real) = logccdf(_conditional(d), y)

@doc "

Quantile (inverse CDF) of the observed time given the primary `p` (bare secondary
only; the interval/truncation pipeline has no closed-form quantile).

See also: [`cdf`](@ref)
"
quantile(d::PrimaryConditional, q::Real) = quantile(_conditional(d), q)

@doc "

Draw an observed time given the primary `p`. For a bare primary-censored
distribution this is `p + rand(get_dist(d))`; for a pipeline the total is
truncated to the bounds and floored to its interval.

See also: [`logpdf`](@ref)
"
Base.rand(rng::AbstractRNG, d::PrimaryConditional) = rand(rng, _conditional(d))
