# The interval-censored, truncated secondary conditional of a
# `double_interval_censored` pipeline, given a sampled primary `p`. Split from
# `PrimaryConditional.jl` so it can reference the pipeline node types
# (`IntervalCensored`, `Truncated`) and the helpers `_origin_primary_event`,
# `get_dist_recursive`, `floor_to_interval`, `find_interval_index`, all defined in
# files included later than `PrimaryConditional.jl`.

# Unwrap a `Latent` to its node before selecting the conditional form.
_conditional(node::Latent, p) = _conditional(node.dist, p)

# An interval/truncation pipeline over a primary-censored node keeps its
# modifiers on the secondary, built from the pipeline pieces and `p`.
function _conditional(node::Union{IntervalCensored, Truncated}, p)
    return _SecondaryConditional(
        get_dist_recursive(node), _origin_primary_event(node),
        _pipeline_bounds(node)..., _pipeline_interval(node), p)
end

# The interval-censored, truncated secondary conditional given the primary `p`.
# The continuous total is `p + delay`; the observed time `y` is interval-censored
# on the absolute grid and the total is truncated to `[lower, upper]`. The
# interval mass is `cdf(delay, hi - p) - cdf(delay, lo - p)` over the interval
# `[lo, hi)` containing `y`, clamped to the truncation bounds, normalised by the
# pipeline's truncation constant `Z = cdf(pc, upper) - cdf(pc, lower)` (with
# `pc = primary_censored(delay, primary_event)`), so the joint integrates over
# `p` to the analytic `double_interval_censored` marginal. With no truncation
# `Z = 1`. A `nothing` interval (truncated but not interval-censored) scores the
# continuous shifted delay inside the window instead of the interval mass; the
# bare `PrimaryCensored` method (no interval and no truncation) is handled in
# `PrimaryConditional.jl`, so this type always carries truncation or an interval.
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

# Interval-censored secondary: the mass of the total `p + delay` over the
# interval `[lo, hi)` containing `y`, clamped to the truncation bounds and
# normalised by Z.
function _secondary_logpdf(interval, d::_SecondaryConditional, y::Real)
    lo, hi = _interval_bounds(interval, y)
    # Clamp the interval to the truncation bounds (an interval may straddle a
    # bound), then shift to delay space. No overlap gives zero mass / `-Inf`.
    lo = max(lo, d.lower) - d.p
    hi = min(hi, d.upper) - d.p
    hi > lo || return oftype(float(y), -Inf)
    mass = cdf(d.delay, hi) - cdf(d.delay, lo)
    return log(max(mass, zero(mass))) - _secondary_logZ(d)
end

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
