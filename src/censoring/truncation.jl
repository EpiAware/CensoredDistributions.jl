# Right-truncation across delay chains.
#
# Right-truncation of a delay depends on what is observed. When the
# intermediate event in a delay chain is recorded, each segment is observed
# on its own and the truncation denominator is the single-delay CDF up to
# the remaining observation window. When the intermediate event is not
# recorded, only the total delay across the chain is observed, so the
# denominator must be the CDF of the convolution of the unobserved segments.
# Mixing these up is a known, easy-to-make error (the andv truncation_model
# carries both forms side by side). This file builds the correct
# right-truncated object given the observation horizon and which splitting
# events are observed, dispatching the single-delay denominator against the
# convolved-chain denominator. It is pure Distributions.jl: the returned
# object is an ordinary `truncated` distribution whose log-normaliser is the
# right-truncation term `-logcdf(dist, window)`.

@doc "

Build the right-truncated delay for one observation, dispatching the
single-delay denominator against the convolved-chain denominator.

Right-truncation differs by what is observed. With the intermediate
splitting event **observed**, the recorded delay is a single segment and the
truncation denominator is `cdf(delay, window)` for that segment. With the
intermediate event **unobserved**, only the total delay across the chain is
seen and the denominator must be `cdf(convolution, window)` over the
convolution of the unobserved segments. `window` is the time still available
before the observation horizon, `horizon - anchor`, where `anchor` is the
time of the event the delay is measured from.

The returned object is `truncated(dist; upper = window)`: its log-normaliser
is exactly the andv right-truncation term `-logcdf(dist, window)`, so a
likelihood built from `logpdf` of this object reproduces the per-record
truncation correction without any PPL dependency. Pass a single delay for the
single-delay denominator; pass a [`Convolved`](@ref) (from
[`convolve_distributions`](@ref)) for the convolved-chain denominator. The
[`truncate_chain`](@ref) helper picks between the two from a chain of
segments and an observation mask.

This is the upper-only observation-horizon form of right-truncation; a
δ-bounded variant, truncating to a finite window of width δ, is planned and
tracked separately.

# Arguments
- `delay`: the delay distribution to right-truncate. A single delay gives the
  single-delay denominator; a [`Convolved`](@ref) gives the convolved-chain
  denominator.
- `window`: the remaining observation window `horizon - anchor` the delay is
  right-truncated to. A non-positive window truncates to an empty support.

# Examples
```@example
using CensoredDistributions, Distributions

# Single observed delay: denominator is cdf(inc_dist, window).
inc_dist = LogNormal(1.5, 0.5)
single = truncate_to_horizon(inc_dist, 6.0)
log_norm_single = logcdf(inc_dist, 6.0)

# Unobserved intermediate event: denominator is cdf(convolution, window).
delta_dist = Gamma(2.0, 1.0)
conv = convolve_distributions(inc_dist, delta_dist)
chain = truncate_to_horizon(conv, 6.0)
log_norm_chain = logcdf(conv, 6.0)
```

# See also
- [`truncate_chain`](@ref): pick single vs convolved from a chain mask
- [`convolve_distributions`](@ref): builds the convolved-chain delay
- [`Convolved`](@ref): the convolution distribution
"
function truncate_to_horizon(delay::UnivariateDistribution, window::Real)
    return _truncate_window(delay, window)
end

@doc "

Build the right-truncated delay for one observation from a chain of segments
and a mask of which intermediate splitting events are observed.

The chain is the ordered sequence of delay segments between the anchor event
and the observation. `observed` marks, for each *internal* boundary between
consecutive segments, whether that splitting event was recorded; it has one
fewer entry than `segments`. Consecutive unobserved segments are collapsed
into a single [`Convolved`](@ref) (their splitting event is not seen, so only
their sum is observed), while an observed boundary keeps the segments on
either side separate. The truncation denominator for the observation is then
`cdf(dist, window)` of the delay reaching the observation, with `dist` the
single segment when its boundaries are observed and the convolution of a run
of unobserved segments otherwise.

This reduces to the andv split. Index cases observe a single delay, so their
record is one segment with no internal boundary, giving the single-delay
denominator. Sourced cases observe the total over two segments whose
splitting event is not recorded, `observed = (false,)`, giving the convolved
denominator. A `true` boundary closes the run before it, so the truncation
distribution is the convolution of the trailing unobserved segments reaching
the observation.

# Arguments
- `segments`: ordered tuple or vector of delay distributions for the chain.
- `observed`: tuple or vector of `Bool`, one per internal boundary
  (`length(segments) - 1` entries), `true` where the splitting event is
  recorded.
- `window`: the remaining observation window `horizon - anchor` for the
  delay reaching the observation.

# Examples
```@example
using CensoredDistributions, Distributions

inc_dist = LogNormal(1.5, 0.5)
delta_dist = Gamma(2.0, 1.0)

# Index cases: single observed delay -> single-delay denominator.
index = truncate_chain((inc_dist,), (), 6.0)

# Sourced cases: intermediate event unobserved -> convolved denominator.
sourced = truncate_chain((inc_dist, delta_dist), (false,), 6.0)
```

# See also
- [`truncate_to_horizon`](@ref): the single/convolved dispatch primitive
- [`convolve_distributions`](@ref): builds the convolved-chain delay
"
function truncate_chain(segments, observed, window::Real)
    length(segments) >= 1 ||
        throw(ArgumentError("truncate_chain needs at least one segment"))
    length(observed) == length(segments) - 1 || throw(ArgumentError(
        "observed must have one fewer entry than segments"))
    dist = _collapse_to_observation(Tuple(segments), Tuple(observed))
    return _truncate_window(dist, window)
end

# Right-truncate `dist` at `window`. Upper-only: no lower bound is added, so
# the log-normaliser is exactly the right-truncation term `-logcdf(dist,
# window)` and AD never differentiates `logcdf(dist, minimum(dist))` (which is
# `-Inf` for e.g. a LogNormal index delay, giving a NaN gradient). A
# non-positive window (the horizon has passed the anchor) clamps to the
# distribution's minimum, giving an empty-support truncation that matches the
# andv `floatmin` guard on a zero denominator.
function _truncate_window(dist::UnivariateDistribution, window::Real)
    upper = max(window, minimum(dist))
    return truncated(dist; upper = upper)
end

# Collapse a chain into the single distribution reaching the observation.
# Walk the segments, accumulating a run of consecutive unobserved segments;
# an observed boundary closes the current run. Only the final run reaches the
# observation (later segments are separated by observed events and are their
# own records), so the truncation distribution is the convolution of that
# trailing unobserved run with the last segment.
function _collapse_to_observation(segments::Tuple, observed::Tuple)
    # Index of the last observed boundary; segments after it form the run
    # reaching the observation.
    start = 1
    for i in eachindex(observed)
        observed[i] && (start = i + 1)
    end
    run = segments[start:end]
    return length(run) == 1 ? run[1] : convolve_distributions(run)
end
