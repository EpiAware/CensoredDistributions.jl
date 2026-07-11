module CensoredDistributionsConvolvedDistributionsExt

# Bridge CensoredDistributions' explicit censoring schemes into
# ConvolvedDistributions' timeseries convolution.
#
# ConvolvedDistributions' `convolve_series` is discrete-only: a continuous
# delay must be turned into a probability mass function on the unit integer
# grid `0, 1, 2, ...` before it can be convolved with a unit-spaced series.
# The statistically correct discretisation for day-binned primary events is
# double interval censoring, which is exactly what
# `double_interval_censored(dist; interval = 1)` builds. This extension lets
# such a censoring object be passed straight to `convolve_series`: it reads
# the discretised delay PMF off the censoring scheme and forwards to the
# stable PMF-vector method `convolve_series(pmf, series)`.
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

# The discretised delay PMF at integer lags `0..(length(series) - 1)`.
# For a regular unit-grid `IntervalCensored`, `pdf(d, k)` is the interval
# mass on `[k, k + 1)` (`F(k + 1) - F(k)` of the inner, possibly
# primary-censored and truncated, distribution), which is precisely the
# delay mass at lag `k` on the same unit grid as `series`. The vectorised
# `pdf` shares one CDF evaluation per boundary and keeps the AD-safe typing,
# so gradients flow through the delay parameters.
function _unit_grid_pmf(d::IntervalCensored, n::Integer)
    return pdf(d, 0:(n - 1))
end

@doc "

Convolve a timeseries with a double-interval-censored delay PMF.

`convolve_series(d, series)` reads the discretised delay probability mass
function off the interval-censored distribution `d` and returns the causal
discrete convolution of `series` with that PMF (via
`ConvolvedDistributions.convolve_series(pmf, series)`). It is the
explicit-censoring route into ConvolvedDistributions' discrete-only
timeseries convolution: `d` supplies the day-binned discretisation, so no
implicit continuous-to-discrete step is hidden inside `convolve_series`.

`d` must discretise onto the unit integer grid, i.e. a regular
[`interval_censored`](@ref) distribution with `interval == 1` (including
the return of `double_interval_censored(dist; interval = 1)`, which is the
statistically correct double interval censoring for day-binned primary
events). The delay mass at integer lag `k` is `pdf(d, k)`, the interval
mass on `[k, k + 1)`; the PMF is read at lags `0..(length(series) - 1)`.

Arbitrary-boundary or non-unit interval-censored distributions are rejected
with an `ArgumentError`, since their masses do not line up with the
unit-spaced series steps the causal convolution shifts by.

# Arguments
- `d`: a unit-grid [`interval_censored`](@ref) delay (e.g. the return of
  `double_interval_censored(dist; interval = 1)`).
- `series`: the input timeseries (expected events at unit-spaced times
  from 0).

# Examples
```@example
using CensoredDistributions, ConvolvedDistributions, Distributions

delay = double_interval_censored(LogNormal(1.5, 0.75); interval = 1)
infections = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]
expected_counts = convolve_series(delay, infections)
```

# See also
- [`double_interval_censored`](@ref): build the day-binned delay
- [`interval_censored`](@ref): the unit-grid discretisation
"
function convolve_series(
        d::IntervalCensored, series::AbstractVector{<:Real})
    is_regular_intervals(d) || throw(ArgumentError(
        "convolve_series needs a unit-grid interval-censored delay, but " *
        "got arbitrary interval boundaries. The causal convolution shifts " *
        "by integer unit-spaced series steps, so the discretisation must " *
        "be a regular interval of width 1 (e.g. interval_censored(dist, " *
        "1) or double_interval_censored(dist; interval = 1))."))
    isone(interval_width(d)) || throw(ArgumentError(
        "convolve_series needs a unit-grid interval-censored delay, but " *
        "the interval width is $(interval_width(d)). The series is " *
        "unit-spaced and the causal convolution shifts by integer series " *
        "steps, so only an interval width of 1 aligns the delay PMF with " *
        "the series grid."))
    pmf = _unit_grid_pmf(d, length(series))
    return convolve_series(pmf, series)
end

@doc "

Reject a bare primary-censored delay in `convolve_series`.

A [`primary_censored`](@ref) distribution is still continuous: primary
event censoring convolves the delay with the primary-event window but does
not bin it onto an integer grid. ConvolvedDistributions' `convolve_series`
is discrete-only, so this method throws an `ArgumentError` directing the
caller to add an explicit secondary interval censoring step (e.g.
`double_interval_censored(dist; interval = 1)`, or wrapping with
[`interval_censored`](@ref)) before convolving.

# See also
- [`double_interval_censored`](@ref): add the day-binned secondary
  censoring
- [`interval_censored`](@ref): the unit-grid discretisation
"
function convolve_series(
        d::PrimaryCensored, series::AbstractVector{<:Real})
    throw(ArgumentError(
        "convolve_series needs a discretised (unit-grid) delay, but got a " *
        "continuous primary-censored distribution. Primary censoring does " *
        "not bin the delay onto an integer grid; add a secondary interval " *
        "censoring step first, e.g. double_interval_censored(dist; " *
        "interval = 1) or interval_censored(primary_censored(dist, " *
        "primary_event), 1)."))
end

end
