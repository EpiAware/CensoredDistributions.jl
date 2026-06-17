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

This is the upper-only observation-horizon form of right-truncation. The
δ-bounded variant [`truncate_to_window`](@ref) adds a finite lower edge, so
the observation window is `[upper - δ, upper]` rather than `(-∞, upper]`; the
upper-only form here is the special case `δ → window` (lower edge 0).

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
- [`truncate_to_window`](@ref): the δ-bounded variant (finite lower edge)
- [`truncate_chain`](@ref): pick single vs convolved from a chain mask
- [`convolve_distributions`](@ref): builds the convolved-chain delay
- [`Convolved`](@ref): the convolution distribution
"
function truncate_to_horizon(delay::UnivariateDistribution, window::Real)
    return _truncate_window(delay, window)
end

@doc "

Build the δ-bounded right-truncated delay for one observation: the finite
observation window `[upper - δ, upper]`.

This is the δ-bounded variant of [`truncate_to_horizon`](@ref). The upper-only
horizon form conditions on the event falling at or before the remaining window
`upper` (normalised by `cdf(delay, upper)`). The δ-bounded form additionally
adds a LOWER edge a width `δ` below the upper edge, so the event is observed
only within the finite window `[upper - δ, upper]`, normalised by
`cdf(delay, upper) - cdf(delay, upper - δ)`.

The lower edge is anchored at the upper edge (`upper - δ`), so the window has
a fixed width `δ` ending at the same remaining-window upper bound the
upper-only form uses. This composes consistently with the upper-only form: it
is the special case `δ === nothing` (or any `δ >= upper`, i.e. `δ → upper`),
which clamps the lower edge to the distribution's minimum and reproduces
[`truncate_to_horizon`](@ref)`(delay, upper)` byte-identically.

# Arguments
- `delay`: the delay distribution to right-truncate. A single delay gives the
  single-delay denominator; a [`Convolved`](@ref) gives the convolved-chain
  denominator.
- `upper`: the remaining observation window `horizon - anchor` (the upper edge,
  as in [`truncate_to_horizon`](@ref)). A non-positive upper edge truncates to
  an empty support.
- `δ`: the observation-window width. `nothing` (or `δ >= upper`) drops the
  lower edge and reproduces the upper-only form. A non-positive `δ` is an empty
  window and errors.

# Examples
```@example
using CensoredDistributions, Distributions

delay = LogNormal(1.5, 0.5)
# Finite window [6 - 4, 6] = [2, 6]: normaliser cdf(6) - cdf(2).
windowed = truncate_to_window(delay, 6.0, 4.0)
log_norm = log(cdf(delay, 6.0) - cdf(delay, 2.0))

# δ = nothing reproduces the upper-only truncate_to_horizon.
upper_only = truncate_to_window(delay, 6.0, nothing)
```

# See also
- [`truncate_to_horizon`](@ref): the upper-only form (the `δ → upper` case)
- [`Convolved`](@ref): the convolution distribution
"
function truncate_to_window(
        delay::UnivariateDistribution, upper::Real, δ::Union{Real, Nothing})
    return _truncate_window(delay, upper, δ)
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

# δ-bounded right-truncation: add a finite lower edge a width `δ` below the
# upper edge, giving the finite observation window `[upper - δ, upper]`
# (normaliser `cdf(upper) - cdf(lower)`). `δ === nothing` (no lower edge) falls
# back to the upper-only `_truncate_window`, so the upper-only path is
# byte-identical. A non-positive `δ` is an empty window and errors. The lower
# edge is clamped UP to the distribution's minimum (so `δ >= upper` collapses to
# the upper-only form, never differentiating `logcdf(dist, minimum)`), and the
# upper edge keeps the upper-only non-positive-window empty-support guard. The
# normaliser `cdf(upper) - cdf(lower)` is what AD differentiates.
function _truncate_window(
        dist::UnivariateDistribution, window::Real, δ::Nothing)
    return _truncate_window(dist, window)
end

function _truncate_window(
        dist::UnivariateDistribution, window::Real, δ::Real)
    δ > 0 || throw(ArgumentError(
        "δ-bounded truncation needs a positive window width δ; got $(δ) " *
        "(a non-positive width is an empty observation window)"))
    upper = max(window, minimum(dist))
    lower = max(window - δ, minimum(dist))
    return truncated(dist; lower = lower, upper = upper)
end

# ---------------------------------------------------------------------------
# Threaded per-record horizon carrier (upper-only vs δ-bounded)
# ---------------------------------------------------------------------------
#
# The per-record observation horizon is threaded through the scorers as a single
# value (`horizon`). The upper-only form is a plain `Real` (or `nothing`); the
# δ-bounded form pairs the horizon with the window width δ in this carrier, so
# the SAME threaded slot carries either form and every existing `horizon ===
# nothing` / pass-through site is untouched. The three helpers below are the only
# places the carrier is unpacked: `_horizon_time` (the scalar horizon for the
# `horizon - anchor` arithmetic), `_horizon_delta` (the δ width, or `nothing`),
# and `_truncate_horizon` (build the truncated delay for a window/anchor),
# keeping the δ off every other signature.
struct WindowedHorizon{H <: Real, D <: Real}
    horizon::H
    δ::D
end

# The scalar horizon time used in the `horizon - anchor` window arithmetic.
_horizon_time(h::Real) = h
_horizon_time(h::WindowedHorizon) = h.horizon

# The δ window width carried by a horizon, or `nothing` for the upper-only form.
_horizon_delta(::Real) = nothing
_horizon_delta(h::WindowedHorizon) = h.δ

# Right-truncate `delay` to the per-record observation `window` (= `horizon -
# anchor`), honouring the threaded horizon's δ. A plain-`Real` horizon (or a δ of
# `nothing`) gives the upper-only `truncate_to_horizon` byte-identically; a
# `WindowedHorizon` δ-bounds the truncation to `[window - δ, window]`. `horizon`
# is the threaded carrier (so the δ rides along) and `window` is the already
# anchor-shifted upper edge the caller computed with `_horizon_time`.
function _truncate_horizon(delay, window::Real, horizon)
    _truncate_window(delay, window, _horizon_delta(horizon))
end

# The log-normaliser (denominator) of the per-record right-truncation, threaded
# the same way `_truncate_horizon` is. Used where the scorer subtracts a single
# right-truncation denominator from a factorised numerator (the flat
# whole-compose chain truncation), rather than building a truncated object. The
# upper-only form is exactly `logcdf(dist, window)`, kept byte-identical so a
# plain horizon reproduces today's `total - logcdf(seg, window)`. The δ-bounded
# form is `log(cdf(dist, upper) - cdf(dist, lower))` with `lower = window - δ`
# clamped up to the distribution's minimum, the log of the finite-window mass.
function _truncation_lognorm(dist, window::Real, horizon)
    _truncation_lognorm_δ(dist, window, _horizon_delta(horizon))
end

_truncation_lognorm_δ(dist, window::Real, ::Nothing) = logcdf(dist, window)

function _truncation_lognorm_δ(dist, window::Real, δ::Real)
    δ > 0 || throw(ArgumentError(
        "δ-bounded truncation needs a positive window width δ; got $(δ) " *
        "(a non-positive width is an empty observation window)"))
    lower = max(window - δ, minimum(dist))
    # The finite-window mass `cdf(upper) - cdf(lower)`. `logsubexp` differentiates
    # cleanly (it is the AD-safe `log(exp(logcdf(upper)) - exp(logcdf(lower)))`).
    return logsubexp(logcdf(dist, window), logcdf(dist, lower))
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
