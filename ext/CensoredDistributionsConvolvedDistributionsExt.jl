module CensoredDistributionsConvolvedDistributionsExt

# Bridges CensoredDistributions' explicit censoring schemes into
# ConvolvedDistributions' discrete-only `convolve_series`: these methods
# read a discretised delay PMF off `IntervalCensored`/`PrimaryCensored` and
# forward to the stable `convolve_series(pmf, series)`. Dispatch is on
# CensoredDistributions-owned types only, so no type piracy. See
# https://github.com/EpiAware/ConvolvedDistributions.jl/issues/31 and
# https://github.com/EpiAware/ConvolvedDistributions.jl/issues/68 (this
# package is the intended discretisation route for continuous delays).

import ConvolvedDistributions: convolve_series
using CensoredDistributions: IntervalCensored, PrimaryCensored,
                             interval_width, is_regular_intervals
using Distributions: pdf

# Delay PMF at grid lags `0..(n - 1)` for a regular interval-censored delay
# of width `w = interval_width(d)`. `pdf(d, k * w)` is the censored mass on
# `[k * w, (k + 1) * w)` for any `w`.
#
# Scalar `pdf` per grid point rather than the batched `pdf(d, vector)`: the
# batched path (`_compute_boundary_cdfs`) cannot be differentiated by Enzyme
# (fwd + rev) when the inner delay is a stacked `Truncated{PrimaryCensored}`
# (the extra truncation layer trips Enzyme's shadow/type analysis;
# `IllegalTypeAnalysisException`/`EnzymeNoShadowError`, #889). The scalar `pdf`
# path is Enzyme-clean at every layer and returns identical values, so the grid
# builds through it. Every other backend already differentiated the batched
# path; this keeps convolve_series AD-safe across all six.
function _grid_pmf(d::IntervalCensored, n::Integer)
    w = interval_width(d)
    return [pdf(d, w * k) for k in 0:(n - 1)]
end

@doc "

Convolve a timeseries with an interval-censored delay PMF on the delay's
own grid.

Reads the discretised delay PMF off the regular-grid interval-censored
distribution `d` (e.g. the return of `double_interval_censored(dist;
interval = w)`) and returns the causal discrete convolution of `series`
with it. `series` is read on the same grid `d` discretises onto: entry `i`
is the value at time `(i - 1) * w`.

Arbitrary-boundary interval-censored distributions are rejected: the causal
convolution needs a single fixed grid step to shift by.

# Examples
```@example
using CensoredDistributions, ConvolvedDistributions, Distributions

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
        "got arbitrary interval boundaries. Use a regular interval instead " *
        "(e.g. interval_censored(dist, w) or double_interval_censored(dist; " *
        "interval = w)); the width w becomes the series grid."))
    pmf = _grid_pmf(d, length(series))
    return convolve_series(pmf, series)
end

@doc "

Reject a bare primary-censored delay in `convolve_series`.

A [`primary_censored`](@ref) distribution is still continuous.
ConvolvedDistributions' `convolve_series` is discrete-only, so this throws
an `ArgumentError` directing the caller to add an explicit secondary
interval censoring step first, e.g. `double_interval_censored(dist;
interval = w)`.

# See also
- [`double_interval_censored`](@ref): add the interval-binned secondary
  censoring
- [`interval_censored`](@ref): the regular-grid discretisation
"
function convolve_series(
        d::PrimaryCensored, series::AbstractVector{<:Real})
    throw(ArgumentError(
        "convolve_series needs a discretised (regular-grid) delay, but got " *
        "a continuous primary-censored distribution. Add a secondary " *
        "interval censoring step first, e.g. double_interval_censored(dist; " *
        "interval = w) or interval_censored(primary_censored(dist, " *
        "primary_event), w)."))
end

end
