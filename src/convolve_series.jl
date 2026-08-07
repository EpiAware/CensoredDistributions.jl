
# Bridges CensoredDistributions' censoring schemes into
# ConvolvedDistributions' discrete `convolve_series`. A continuous delay is
# discretised by double interval censoring, an interval-censored delay has
# its grid PMF read off directly, and a bare primary-censored delay is
# rejected until a secondary censoring step is added. CensoredDistributions
# is the intended discretisation route for continuous delays, so the
# continuous method deliberately extends `convolve_series` for a foreign
# type.

# Delay PMF at grid lags `0..(n - 1)` for a regular interval-censored delay
# of width `w = interval_width(d)`. `pdf(d, k * w)` is the censored mass on
# `[k * w, (k + 1) * w)` for any `w`. Arbitrary boundaries have no single
# grid step to shift by, so they are rejected here.
function _grid_pmf(d::IntervalCensored, n::Integer)
    is_regular_intervals(d) || throw(ArgumentError(
        "convolve_series needs a regular-grid interval-censored delay, but " *
        "got arbitrary interval boundaries. Use a regular interval instead " *
        "(e.g. interval_censored(dist, w) or double_interval_censored(dist; " *
        "interval = w)); the width w becomes the series grid."))
    w = interval_width(d)
    return pdf(d, w .* (0:(n - 1)))
end

@doc "

Convolve a timeseries with a continuous delay, discretised by double
interval censoring.

`convolve_series(d, series)` for a continuous `d` wraps it with
[`double_interval_censored`](@ref) and convolves `series` with the resulting
grid PMF. `interval` sets the grid step and defaults to 1, so `series` entry
`i` is read at time `(i - 1) * interval`. Any other keyword arguments pass
through to [`double_interval_censored`](@ref), so the primary event,
truncation, and solver can be set.

A continuous delay carries no mass on the lag grid until it is discretised,
and the discretisation is a censoring choice; this method makes the usual
epidemiological choice for you. Pass a pre-built censored delay to control
it exactly.

# Examples
```@example
using CensoredDistributions, ConvolvedDistributions, Distributions

infections = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]
expected_counts = convolve_series(LogNormal(1.5, 0.75), infections)
```

# See also
- [`double_interval_censored`](@ref): the discretisation this dispatches to
- [`interval_censored`](@ref): the regular-grid discretisation
"
function convolve_series(
        d::ContinuousUnivariateDistribution,
        series::AbstractVector{<:Real}; interval::Real = 1, kwargs...)
    return convolve_series(
        double_interval_censored(d; interval = interval, kwargs...), series)
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
    return convolve_series(_grid_pmf(d, length(series)), series)
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

# Fast path: read an interval-censored delay's grid PMF directly, avoiding
# the unit-impulse round-trip of the generic `delay_masses` fallback.
delay_masses(d::IntervalCensored, n::Int) = _grid_pmf(d, n)
