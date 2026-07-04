# Interval-censored, truncated secondary conditional of a
# `double_interval_censored` pipeline given a sampled primary. Split from
# `PrimaryConditional.jl` to reference pipeline types and helpers included later.

# Unwrap a `Latent` to its wrapped distribution before selecting the conditional.
_conditional(d::Latent, p) = _conditional(d.dist, p)

# Dedicated per-wrapper methods (mirroring `_pipeline_bounds`), so the interval-
# censored and truncated-only pipelines each dispatch to their own handler with
# no `Union`. Both keep the AD-stable below-support handling downstream (the
# interval mass and the truncation constant go through `_delay_cdf`; see there
# for why a stock `truncated(delay, ...)` is not used).
# Interval-censored pipeline: the conditional scores the interval-censored,
# truncated mass of the total `p + delay`.
_conditional(d::IntervalCensored, p) = _secondary_conditional(d, p)
# Truncated-only pipeline: the conditional is the continuous shifted delay inside
# the truncation window (no interval spec).
_conditional(d::Truncated, p) = _secondary_conditional(d, p)
# Shared builder: read the delay, primary prior, truncation bounds and interval
# spec (`nothing` when truncated-only) off the pipeline and carry `p`.
function _secondary_conditional(d, p)
    return _SecondaryConditional(
        get_dist_recursive(d), get_primary_event(d),
        _pipeline_bounds(d)..., _pipeline_interval(d), p)
end

"""
The distribution of the observed time under a `double_interval_censored`
pipeline given a realised primary `p`: the continuous total `p + delay`
truncated to `[lower, upper]` and interval-censored on the absolute grid. Scored
with `logpdf`/`rand` (no closed-form `cdf`/`quantile`) and normalised by the
pipeline truncation constant so the joint over `p` integrates to the analytic
`double_interval_censored` marginal. A `nothing` `interval` is the
truncated-only continuous case.

Fields `delay`, `primary_event`, `lower`, `upper`, `interval`, `p`. Always
carries a truncation or an interval; the bare untruncated case is a
`_ShiftedDelayCore` instead.
"""
struct _SecondaryConditional{D, E, B, I, P <: Real} <:
       UnivariateDistribution{Continuous}
    delay::D
    primary_event::E
    lower::B
    upper::B
    interval::I
    p::P
end
minimum(d::_SecondaryConditional) = d.p + max(d.lower - d.p, minimum(d.delay))
maximum(d::_SecondaryConditional) = min(d.p + maximum(d.delay), d.upper)
# The observed `y` is in support inside the truncation window. With interval
# censoring the floored `y` is the start of its interval `[lo, hi)`, which can
# straddle a bound (e.g. `[0, 1)` with `lower = 0.5`), so the test is on the
# interval's overlap with `[lower, upper]`; the continuous case tests `y` itself.
insupport(d::_SecondaryConditional, y::Real) = _insupport(d.interval, d, y)
function _insupport(interval, d::_SecondaryConditional, y::Real)
    lo, hi = _interval_bounds(interval, y)
    return hi > d.lower && lo < d.upper
end
_insupport(::Nothing, d::_SecondaryConditional, y::Real) = d.lower <= y <= d.upper

# Log of the normalising constant Z = P(lower <= P + delay <= upper) of the
# truncated primary-censored total, shared across `p` (a pipeline-level constant,
# not a per-`p` truncation). An untruncated pipeline (both bounds infinite) has
# Z = 1, short-circuited to skip building the primary-censored cdf; otherwise Z is
# the primary-censored mass between the bounds.
function _secondary_logZ(d::_SecondaryConditional)
    (isinf(d.lower) && isinf(d.upper)) && return zero(float(d.upper))
    pc = primary_censored(d.delay, d.primary_event)
    hi = isinf(d.upper) ? one(float(d.upper)) : cdf(pc, d.upper)
    lo = (isinf(d.lower) || d.lower <= zero(d.lower)) ? zero(float(d.lower)) :
         cdf(pc, d.lower)
    return log(hi - lo)
end

function logpdf(d::_SecondaryConditional, y::Real)
    return _secondary_logpdf(d.interval, d, y)
end
pdf(d::_SecondaryConditional, y::Real) = exp(logpdf(d, y))

# An interval-censored / truncated pipeline secondary has no closed-form CDF,
# quantile or mean; it is scored through `logpdf`/`pdf` and drawn with `rand`.
# Defining these here keeps the `PrimaryConditional` dispatch total -- the bare
# `_ShiftedDelayCore` form carries real versions, this pipeline form raises an
# explanatory error instead of a bare `MethodError`. `logcdf`/`ccdf` fall through
# to the generics over `cdf`.
function cdf(d::_SecondaryConditional, ::Real)
    throw(ArgumentError(
        "cdf is undefined for an interval-censored / truncated " *
        "primary-conditional pipeline; score it with `logpdf` instead"))
end
function quantile(d::_SecondaryConditional, ::Real)
    throw(ArgumentError(
        "quantile is undefined for an interval-censored / truncated " *
        "primary-conditional pipeline; sample it with `rand` instead"))
end
function mean(d::_SecondaryConditional)
    throw(ArgumentError(
        "mean is undefined for an interval-censored / truncated " *
        "primary-conditional pipeline; estimate it from `rand` draws instead"))
end

# Interval-censored secondary: the mass of the total `p + delay` over the
# interval `[lo, hi)` containing `y`, clamped to the truncation bounds and
# normalised by Z.
function _secondary_logpdf(interval, d::_SecondaryConditional, y::Real)
    lo, hi = _interval_bounds(interval, y)
    # Truncate the observed interval below by the primary event time (secondary >=
    # primary), then shift to delay space. A primary at/after the interval leaves
    # no overlap -> zero mass, -Inf.
    lo = max(lo, d.lower, d.p) - d.p
    hi = min(hi, d.upper) - d.p
    hi > lo || return oftype(float(y), -Inf)
    mass = _delay_cdf(d.delay, hi) - _delay_cdf(d.delay, lo)
    return log(max(mass, zero(mass))) - _secondary_logZ(d)
end

# Delay cdf contributing zero below its support (the below-primary truncation can
# put the lower edge at the boundary). Omitting the sub-support term rather than
# calling `cdf` there keeps the value and gives a finite AD gradient (`d/dσ
# cdf(LogNormal, 0)` is `0 * Inf = NaN`). This is why the interval mass is built
# by hand instead of via `truncated(delay, lower, upper)`: for the zero-delay
# records the lower edge lands at `minimum(delay)`, and `truncated`'s
# normalisation evaluates `cdf(delay, lower)` there ungarded, reintroducing the
# NaN gradient (verified: FD and Mooncake give NaN through `truncated`, finite
# here).
_delay_cdf(delay, x) = x <= minimum(delay) ? zero(float(x)) : cdf(delay, x)

# Continuous secondary (truncated but not interval-censored): the shifted delay
# density inside the truncation window, normalised by Z.
function _secondary_logpdf(::Nothing, d::_SecondaryConditional, y::Real)
    (d.lower <= y <= d.upper) || return oftype(float(y), -Inf)
    return logpdf(d.delay, y - d.p) - _secondary_logZ(d)
end

# The half-open interval `[lo, hi)` of the absolute grid containing `y`, from the
# pipeline's secondary interval spec (a regular width or arbitrary boundaries).
function _interval_bounds(width::Real, y::Real)
    lo = floor_to_interval(y, width)
    return lo, lo + width
end
function _interval_bounds(boundaries::AbstractVector, y::Real)
    idx = find_interval_index(y, boundaries)
    (idx == 0 || idx == length(boundaries)) &&
        return (oftype(float(y), NaN), oftype(float(y), NaN))
    return boundaries[idx], boundaries[idx + 1]
end

# Draw an observed time given `p`: the total `p + delay` truncated to the bounds,
# then floored to its interval (the continuous total when there is no interval). A
# bounded resample keeps the draw simple and correct (the latent path uses `rand`
# only for simulation).
function Base.rand(rng::AbstractRNG, d::_SecondaryConditional)
    for _ in 1:10_000
        total = d.p + rand(rng, d.delay)
        d.lower <= total <= d.upper || continue
        return _floor_observed(d.interval, total)
    end
    throw(ErrorException(
        "could not draw an in-bounds secondary delay after 10000 tries; " *
        "the truncation bounds may be inconsistent with the delay support"))
end
_floor_observed(interval, total) = first(_interval_bounds(interval, total))
_floor_observed(::Nothing, total) = total

# The truncation bounds `(lower, upper)` of an interval/truncation pipeline, read
# through the wrappers. A `Truncated` stores an absent bound as `nothing`, mapped
# to the open end. An untruncated pipeline gives `(-Inf, Inf)`.
_pipeline_bounds(d::IntervalCensored) = _pipeline_bounds(d.dist)
function _pipeline_bounds(d::Truncated)
    return promote(_bound_or(d.lower, -Inf), _bound_or(d.upper, Inf))
end
_pipeline_bounds(d::PrimaryCensored) = (-Inf, Inf)
_pipeline_bounds(d) = (-Inf, Inf)
_bound_or(::Nothing, default) = default
_bound_or(x, _) = x

# The secondary interval spec (a regular width or arbitrary boundaries) of an
# interval/truncation pipeline, read through the wrappers, or `nothing`.
_pipeline_interval(d::IntervalCensored) = d.boundaries
_pipeline_interval(d::Truncated) = _pipeline_interval(d.untruncated)
_pipeline_interval(d) = nothing

# --- Batched latent conditional -------------------------------------------
# Score a vector of observed times `ys` against a single latent distribution,
# each with its own primary in `ps`, and sum the per-record log densities in one
# vectorised pass (one AD tape) so a model can add a single
# `logpdf(latent(d), ys; primary = ps)` term instead of a per-record loop.
# Dispatched on the wrapped distribution, mirroring `_conditional`. The public
# entry is `logpdf(d::Latent, ys; primary)` in `Latent.jl`.

# Bare primary-censored distribution: the secondary is the continuous delay
# shifted by the primary, so the summed score is `sum(logpdf(delay, y - p))`.
function _latent_batched_logpdf(d::PrimaryCensored, ys::AbstractVector,
        ps::AbstractVector)
    return sum(logpdf.(d.dist, ys .- ps))
end

# Interval-censored pipeline: the interval-censored, truncated secondary mass per
# record, truncated below by each primary. A regular-width interval takes the
# one-tape array path; arbitrary boundaries fall back to the per-record scalar
# conditional (correctness over the one-tape path for that rarer spec).
function _latent_batched_logpdf(d::IntervalCensored, ys::AbstractVector,
        ps::AbstractVector)
    return _batched_secondary(d, ys, ps)
end
# Truncated-only pipeline: the continuous shifted-delay secondary per record
# (interval `nothing`), mirroring the interval-censored method.
function _latent_batched_logpdf(d::Truncated, ys::AbstractVector,
        ps::AbstractVector)
    return _batched_secondary(d, ys, ps)
end
function _batched_secondary(d, ys::AbstractVector, ps::AbstractVector)
    delay = get_dist_recursive(d)
    pe = get_primary_event(d)
    lower, upper = _pipeline_bounds(d)
    return _batched_pipeline_logpdf(
        _pipeline_interval(d), delay, pe, lower, upper, ys, ps)
end

# Regular-width interval: one vectorised pass. The interval `[lo, hi)` containing
# each `y` is truncated to the bounds and below by that record's primary (the
# `_delay_cdf` guard omits the lower term at/below the support), and the shared
# truncation constant `Z` is subtracted once per record. An infeasible record
# (primary at/after its interval, `hi <= lo`) contributes `-Inf` without
# evaluating a bad log: `max(mass, tiny)` keeps the unselected branch finite and
# the `ifelse` selects `-Inf` with a zero gradient.
function _batched_pipeline_logpdf(width::Real, delay, pe, lower, upper, ys, ps)
    logZ = _secondary_logZ(_SecondaryConditional(
        delay, pe, lower, upper, width, zero(eltype(ps))))
    los = floor_to_interval.(ys, width)
    his = los .+ width
    lo = max.(los, lower, ps) .- ps
    hi = min.(his, upper) .- ps
    mass = _delay_cdf.(Ref(delay), hi) .- _delay_cdf.(Ref(delay), lo)
    tiny = floatmin(float(eltype(ys)))
    logterms = @. ifelse(hi > lo, log(max(mass, tiny)), oftype(mass, -Inf))
    return sum(logterms) - length(ys) * logZ
end

# Arbitrary boundaries or a continuous (no-interval) window: score each record via
# the scalar secondary conditional and sum.
function _batched_pipeline_logpdf(interval, delay, pe, lower, upper, ys, ps)
    return sum(_secondary_logpdf(interval,
                   _SecondaryConditional(delay, pe, lower, upper, interval, ps[i]), ys[i])
    for i in eachindex(ys))
end
