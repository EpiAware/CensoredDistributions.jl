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
#
# There is ONE truncation verb, `truncated`. The single-delay and
# convolved-chain right-truncations are both `truncated(dist; upper = window)`,
# the dispatch picked by passing the single delay or the `Convolved` chain
# total. The δ-bounded finite observation window is `truncated(dist; upper,
# lower = upper - δ)`. The composed-node directions live in `composers/wrap.jl`
# (`truncated(node; lower, upper)` distributes into the node's leaf cores). The
# helpers below are the internal primitives the per-record scorers thread the
# observation horizon through (`_truncate_window`, `_truncate_horizon`,
# `_truncation_lognorm`); they add the AD-safe empty-support clamp and the
# δ-bounded normaliser the bare `truncated` does not.

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
# `nothing`) gives the upper-only right-truncation; a `WindowedHorizon` δ-bounds
# the truncation to `[window - δ, window]`. `horizon` is the threaded carrier (so
# the δ rides along) and `window` is the already anchor-shifted upper edge the
# caller computed with `_horizon_time`.
function _truncate_horizon(delay, window::Real, horizon)
    _truncate_window(delay, window, _horizon_delta(horizon))
end

# The log-normaliser (denominator) of the per-record right-truncation, threaded
# the same way `_truncate_horizon` is. Used where the scorer subtracts a single
# right-truncation denominator from a factorised numerator (the flat
# whole-compose chain truncation), rather than building a truncated object. The
# upper-only form is exactly `logcdf(dist, window)`. The δ-bounded form is
# `log(cdf(dist, upper) - cdf(dist, lower))` with `lower = window - δ` clamped up
# to the distribution's minimum, the log of the finite-window mass.
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
# trailing unobserved run with the last segment. The caller then right-truncates
# the collapsed distribution with `truncated(...; upper = window)`: a single
# observed delay gives the single-delay denominator, a `Convolved` chain total
# the convolved-chain denominator.
function _collapse_to_observation(segments::Tuple, observed::Tuple)
    length(segments) >= 1 ||
        throw(ArgumentError("a delay chain needs at least one segment"))
    length(observed) == length(segments) - 1 || throw(ArgumentError(
        "the observed mask must have one fewer entry than segments"))
    # Index of the last observed boundary; segments after it form the run
    # reaching the observation.
    start = 1
    for i in eachindex(observed)
        observed[i] && (start = i + 1)
    end
    run = segments[start:end]
    return length(run) == 1 ? run[1] : convolved(run)
end
