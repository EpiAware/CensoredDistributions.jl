# Timeseries convolution with continuous and time-varying delays

md"""
This tutorial shows what `convolve_series` does by plotting it. It convolves a
numeric `series` with a delay to produce an expected downstream count curve —
the renewal-style observation layer that turns an infection curve into an
expected counts / reports curve.

`convolve_series` lives in ConvolvedDistributions.jl; loading CensoredDistributions
alongside it activates a bridge that understands our censored delays. In
particular it lets you convolve a **raw continuous delay** (discretised for you
via `double_interval_censored`), a pre-built **interval-censored delay**, and —
new — a **time-varying** sequence of delays.
"""

md"""
### What are we going to do

1. Build a delay and a synthetic infection series.
2. Convolve the infection series into an expected count curve:
   - with a raw continuous delay,
   - with a weekly (interval-censored) delay,
   - with a time-varying delay.
"""

md"""
### What might I need to know before starting

`convolve_series(series, delay)` is a causal discrete convolution. The delay is
turned into a probability mass function on a lag grid, and the series entry at
time `i` is smeared forward by that mass. For a continuous delay the grid step
is the `interval` keyword (default `1`), so be sure the series and the delay are
on the same time unit.
"""

md"""
### Packages used
"""

using CensoredDistributions, ConvolvedDistributions, Distributions
using CairoMakie, AlgebraOfGraphics, DataFramesMeta

CairoMakie.activate!(type = "png", px_per_unit = 2)

md"""
### A raw continuous delay

Pass a continuous distribution straight in. It is discretised with
`double_interval_censored` on a unit grid by default, so `series` entry `i` is
read at time `(i - 1)`.
"""

t = 0:40
infections = 100 .* exp.(-((t .- 12.0) .^ 2) ./ 30.0)
expected = convolve_series(LogNormal(1.5, 0.75), infections)

timeseries_df = vcat(
    DataFrame(t = t, count = infections, series = "Infections"),
    DataFrame(t = t, count = expected, series = "Expected reports")
)
draw(
    data(timeseries_df) * mapping(:t, :count, color = :series) *
    visual(Lines, linewidth = 2);
    axis = (xlabel = "Day", ylabel = "Expected count")
)

md"""
The `interval` keyword sets the grid step, and any other keyword passes through
to [`double_interval_censored`](@ref) (e.g. a `primary_event`).
"""

expected_weekly = convolve_series(LogNormal(1.5, 0.75), infections; interval = 7)
expected[1:5]

md"""
### A pre-built interval-censored delay

If you discretise the delay yourself (e.g. to control the binning or interval
width), pass the `IntervalCensored` distribution directly and it is read on its
own grid.
"""

delay = double_interval_censored(LogNormal(1.5, 0.75); interval = 7)
expected_weekly_built = convolve_series(delay, infections)

md"""
### A time-varying delay

`convolve_series` also accepts a **vector** of delays, one per series entry.
Each entry is smeared forward through its own delay, which lets the delay
distribution change over time (e.g. reporting delays that shorten as an
outbreak matures).
"""

delays = [double_interval_censored(LogNormal(m, 0.6); interval = 1)
          for m in range(1.0, 1.8; length = length(t))]
expected_timevarying = convolve_series(delays, infections)

expected_timevarying[1:5]

md"""
### Summary

- `convolve_series(delay, series)` turns an infection series into an expected
  counts curve = a causal convolution with the delay's PMF.
- A raw continuous delay is discretised for you (`interval` keyword, other
  kwargs to `double_interval_censored`); a bare `primary_censored` delay needs
  an explicit secondary interval censoring step.
- A pre-built `interval_censored` delay is read on its own grid.
- A vector of delays gives a time-varying convolution.
"""
