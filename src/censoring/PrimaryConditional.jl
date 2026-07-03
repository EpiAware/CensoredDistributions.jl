@doc "

Conditional distribution of the observed time given a realised primary `p`.

For a primary-censored node `d` with delay `get_dist(d)`, the observed time is
the primary plus the delay, so conditioning on a realised primary `p` shifts the
delay: `logpdf` scores `logpdf(get_dist(d), y - p)` (support `y > p`) and `rand`
returns `p + rand(get_dist(d))`.

When the node also carries secondary interval censoring and/or truncation (a
[`double_interval_censored`](@ref) pipeline), those modifiers are kept on the
secondary: conditioning on `p` keeps the interval censoring and the truncation
on the total time `p + delay`, rather than stripping to the bare continuous
delay. So `latent(double_interval_censored(leaf))` samples the primary `p` and
scores the interval-censored, truncated secondary against it, and the joint
integrates over `p` to the analytic marginal.

This is the single source of the conditional that [`Latent`](@ref) scores and
samples. Turing-free, so it can be used with `~` in a model —
`y ~ PrimaryConditional(d, p)` both scores an observed `y` and generates a
missing one, with `p` the sampled latent primary.

The `dist` field holds the primary-censored node (or the interval/truncation
pipeline over it); the `p` field holds the realised primary event time.

# See also
- [`Latent`](@ref): the joint that reuses this conditional
- [`get_dist`](@ref): the delay distribution scored here
"
struct PrimaryConditional{D, P <: Real} <: AbstractPrimaryCensored
    "The primary-censored node (or its `Latent` / pipeline wrapper)."
    dist::D
    "The realised primary event time conditioned on."
    p::P
end

# The conditional secondary distribution scored against the observed time `y`
# given the realised primary `p`. The wrapped node selects the form:
#   - a `PrimaryCensored` (or a `Latent` over one) gives the bare continuous
#     delay shifted by `p`, so `logpdf` is `logpdf(delay, y - p)`;
#   - an `IntervalCensored` / `Truncated` pipeline over a `PrimaryCensored` keeps
#     the secondary interval and truncation on the total `p + delay` (built in
#     `secondary_conditional.jl`, included after the pipeline node types).
# Dispatch on the wrapped node type picks the form. The `Latent`-unwrapping
# method lives in `secondary_conditional.jl`, included after `Latent` is defined.
_conditional(d::PrimaryConditional) = _conditional(d.dist, d.p)

# Bare primary-censored leaf: the secondary is the continuous delay shifted by
# `p` (the sampled-origin rule the joint already uses).
_conditional(node::PrimaryCensored, p) = _ShiftedDelayCore(node.dist, p)

# Fail loud on an unknown inner node: a wrapper that changes the delay density
# (e.g. weighting, reparameterisation) has no defined primary-conditional, and
# silently unwrapping it would score a wrong density. The `Latent`,
# `IntervalCensored`/`Truncated` and `PrimaryCensored` methods are more specific,
# so only genuinely-unsupported nodes reach this.
function _conditional(node, p)
    throw(ArgumentError(
        "conditioning a latent observation on its primary event is not defined " *
        "for $(typeof(node)); supported inner distributions are PrimaryCensored, " *
        "IntervalCensored, Truncated (and their combinations). A wrapper that " *
        "changes the delay density has no defined primary-conditional — use the " *
        "marginal form or unwrap it."))
end

# A continuous delay shifted by the primary, the bare conditional secondary.
# `logpdf(delay, y - shift)` with support `y > shift + minimum(delay)`.
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
Base.rand(rng::AbstractRNG, d::_ShiftedDelayCore) = d.shift + rand(rng, d.delay)

minimum(d::PrimaryConditional) = minimum(_conditional(d))
maximum(d::PrimaryConditional) = maximum(_conditional(d))
insupport(d::PrimaryConditional, y::Real) = insupport(_conditional(d), y)
params(d::PrimaryConditional) = (params(get_dist(d))..., d.p)
Base.eltype(::Type{<:PrimaryConditional{D}}) where {D} = eltype(D)

@doc "

Log density of the observed time `y` given the primary `p`. For a bare
primary-censored node this is the delay density at the implied gap,
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

Draw an observed time given the primary `p`. For a bare node this is
`p + rand(get_dist(d))`; for a pipeline the total is truncated to the bounds and
floored to its interval.

See also: [`logpdf`](@ref)
"
Base.rand(rng::AbstractRNG, d::PrimaryConditional) = rand(rng, _conditional(d))
