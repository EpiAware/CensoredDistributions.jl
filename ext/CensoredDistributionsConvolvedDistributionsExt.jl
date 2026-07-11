module CensoredDistributionsConvolvedDistributionsExt

# Bridge CensoredDistributions' explicit censoring schemes into
# ConvolvedDistributions' timeseries convolution.
#
# ConvolvedDistributions' `convolve_series` is discrete-only: a continuous
# delay must be turned into a probability mass function on a regular grid
# before it can be convolved with a series sampled on that same grid. The
# statistically correct discretisation for interval-binned primary events is
# double interval censoring, which is exactly what
# `double_interval_censored(dist; interval = w)` builds. This extension lets
# such a censoring object be passed straight to `convolve_series`: it reads
# the discretised delay PMF off the censoring scheme and forwards to the
# stable PMF-vector method `convolve_series(pmf, series)`.
#
# The grid width is NOT assumed to be 1. It comes from the censored
# distribution's own regular interval `w`, declared by the caller when they
# built `double_interval_censored(dist; interval = w)` or
# `interval_censored(dist, w)`. This mirrors ConvolvedDistributions' own
# `convolve_series(pmf::DelayPMF, series)`, where the grid width travels with
# the PMF and the series is read on that grid.
#
# The methods dispatch ONLY on CensoredDistributions-owned types
# (`IntervalCensored`, `PrimaryCensored`), never on plain Distributions
# types, so there is no type piracy: `convolve_series` is owned by
# ConvolvedDistributions and the dispatched argument is owned by
# CensoredDistributions, the two packages that trigger this extension.
# See https://github.com/EpiAware/ConvolvedDistributions.jl/issues/31.

import ConvolvedDistributions: convolve_series
import CensoredDistributions: IntervalCensored, PrimaryCensored,
                              is_regular_intervals, interval_width
using Distributions: pdf

# The discretised delay PMF at grid lags `0..(n - 1)` for a regular
# interval-censored distribution of width `w = interval_width(d)`.
#
# For a regular grid, `pdf(d, x)` is the censored mass on the interval that
# floors `x`, i.e. `[floor(x / w) * w, floor(x / w) * w + w)`. Evaluating at
# `x = k * w` (an exact grid boundary) therefore returns the mass on
# `[k * w, (k + 1) * w)` for any width `w`, not just `w == 1`. The vectorised
# `pdf` shares one CDF evaluation per boundary and keeps the AD-safe typing,
# so gradients flow through the delay parameters.
function _grid_pmf(d::IntervalCensored, n::Integer)
    w = interval_width(d)
    return pdf(d, w .* (0:(n - 1)))
end

@doc "

Convolve a timeseries with an interval-censored delay PMF on the delay's
own grid.

`convolve_series(d, series)` reads the discretised delay probability mass
function off the interval-censored distribution `d` and returns the causal
discrete convolution of `series` with that PMF (via
`ConvolvedDistributions.convolve_series(pmf, series)`). It is the
explicit-censoring route into ConvolvedDistributions' discrete-only
timeseries convolution: `d` supplies the discretisation, so no implicit
continuous-to-discrete step is hidden inside `convolve_series`.

The grid width comes from `d` itself. `d` must be a regular
[`interval_censored`](@ref) distribution of some width `w` (including the
return of `double_interval_censored(dist; interval = w)`, the statistically
correct double interval censoring for interval-binned primary events). The
delay mass at grid lag `k` is `pdf(d, k * w)`, the censored mass on
`[k * w, (k + 1) * w)`; the PMF is read at lags `0..(length(series) - 1)`.

The `series` is interpreted on that same grid: entry `i` is the value at
time `(i - 1) * w`, and the output series is on the same grid. The caller's
discretisation choice IS the series grid; there is no separate step argument
to conflict with it. This matches ConvolvedDistributions' own
`convolve_series(pmf::DelayPMF, series)`, where the grid width travels with
the PMF.

Only genuinely incoherent inputs are rejected. Arbitrary-boundary
interval-censored distributions are rejected with an `ArgumentError`,
because they have no single grid step for the causal convolution to shift
by.

# Arguments
- `d`: a regular [`interval_censored`](@ref) delay of width `w` (e.g. the
  return of `double_interval_censored(dist; interval = w)`).
- `series`: the input timeseries (values at times `(i - 1) * w`, sampled at
  steps of `w` from time 0).

# Examples
```@example
using CensoredDistributions, ConvolvedDistributions, Distributions

# Weekly grid: the interval width sets the series grid.
delay = double_interval_censored(LogNormal(1.5, 0.75); interval = 7)
infections = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]
expected_counts = convolve_series(delay, infections)
```

# See also
- [`double_interval_censored`](@ref): build the interval-binned delay
- [`interval_censored`](@ref): the regular-grid discretisation
"
function convolve_series(
        d::IntervalCensored, series::AbstractVector{<:Real})
    is_regular_intervals(d) || throw(ArgumentError(
        "convolve_series needs a regular-grid interval-censored delay, but " *
        "got arbitrary interval boundaries. The causal convolution shifts " *
        "the series by a single fixed grid step, which arbitrary boundaries " *
        "do not define. Use a regular interval instead (e.g. " *
        "interval_censored(dist, w) or double_interval_censored(dist; " *
        "interval = w)); the width w becomes the series grid."))
    pmf = _grid_pmf(d, length(series))
    return convolve_series(pmf, series)
end

@doc "

Reject a bare primary-censored delay in `convolve_series`.

A [`primary_censored`](@ref) distribution is still continuous: primary
event censoring convolves the delay with the primary-event window but does
not bin it onto a grid. ConvolvedDistributions' `convolve_series` is
discrete-only, so this method throws an `ArgumentError` directing the caller
to add an explicit secondary interval censoring step (e.g.
`double_interval_censored(dist; interval = w)`, or wrapping with
[`interval_censored`](@ref)) before convolving.

# See also
- [`double_interval_censored`](@ref): add the interval-binned secondary
  censoring
- [`interval_censored`](@ref): the regular-grid discretisation
"
function convolve_series(
        d::PrimaryCensored, series::AbstractVector{<:Real})
    throw(ArgumentError(
        "convolve_series needs a discretised (regular-grid) delay, but got " *
        "a continuous primary-censored distribution. Primary censoring does " *
        "not bin the delay onto a grid; add a secondary interval censoring " *
        "step first, e.g. double_interval_censored(dist; interval = w) or " *
        "interval_censored(primary_censored(dist, primary_event), w)."))
end

end
