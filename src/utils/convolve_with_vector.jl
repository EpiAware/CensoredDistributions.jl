# ============================================================================
# convolve_distributions(stack, series): push a timeseries through a stack
# ============================================================================
#
# A `convolve_distributions` method whose second argument is a numeric
# timeseries vector convolves that series THROUGH a composed delay stack (the
# delay chain) to event counts at the same times. With `series` the expected
# events at times `0..t` (e.g. infections), the result is the expected event
# counts at the same times: the EpiNow2-style latent / renewal observation layer
# falls out of the composed delay stack automatically.
#
# This reuses the existing distribution-level
# `convolve_distributions(dists...)`: the second positional argument is
# `AbstractVector{<:Real}` (a numeric series), distinct from the
# `AbstractVector{<:UnivariateDistribution}` / two-distribution forms, so the
# renewal method and the distribution-args forms never collide.
#
# Event selectivity. In a fit only SOME events are observed (e.g. only onsets),
# so the `events` keyword names which event(s) to produce. The default is the
# END-POINT event; a [`Sequential`](@ref) chain's named INTERIM events map to
# its PREFIX convolutions. Only the requested events are discretised and
# convolved, so an unobserved prefix costs nothing.
#
# Efficiency. The stack convolution is composed at the DISTRIBUTION level ONCE:
# the total delay is `observed_distribution(stack)` (the convolution of the
# steps) and each interim cumulative delay is a Sequential PREFIX convolution.
# Each requested delay is discretised to a PMF over the integer grid `0..t` ONCE
# via `interval_censored` (raw interval masses, no renormalise). The expensive
# part is then the discrete VECTOR convolution of the series with that PMF, done
# once per requested event.
#
# AD-safety. The vector convolution is linear and the PMF depends differentiably
# on the delay parameters (interval_censored routes through the AD-safe CDF
# helpers), so gradients flow through ForwardDiff / ReverseDiff w.r.t. the delay
# params.

# --- the discrete delay PMF over the grid 0..maxlag ------------------------

# Probability masses of `delay` on the unit-spaced intervals
# `[0, 1), [1, 2), ..., [maxlag, maxlag + 1)`, as a length `maxlag + 1` vector.
# Masses are the raw `interval_censored` interval probabilities (no silent
# renormalise). The SCALAR `IntervalCensored` PDF is mapped over the grid: it
# routes each interval mass through `_cdf_ad_safe`, so `Dual`/tracked CDF values
# survive (the batched PDF would force its cache to `Float64`, breaking AD).
function _delay_pmf(delay::UnivariateDistribution, maxlag::Integer, interval)
    ic = interval_censored(delay, interval)
    grid = (0:maxlag) .* interval
    return map(g -> pdf(ic, g), grid)
end

# --- the causal discrete convolution series ⊛ pmf --------------------------

# Causal discrete convolution of a series with a delay PMF, truncated to the
# series window. `out[i] = Σ_{k≥0} pmf[k + 1] * series[i - k]`, i.e. mass from
# lag `k` carries `series[i - k]` forward to time `i`. The accumulator element
# type is seeded from the product so `Dual`/tracked numbers propagate.
function _causal_convolve(series::AbstractVector, pmf::AbstractVector)
    n = length(series)
    T = promote_type(eltype(series), eltype(pmf))
    out = zeros(T, n)
    @inbounds for i in 1:n
        acc = zero(T)
        kmax = min(length(pmf), i)
        for k in 1:kmax
            acc += pmf[k] * series[i - k + 1]
        end
        out[i] = acc
    end
    return out
end

# --- the cumulative delays of a stack --------------------------------------

# The flat ordered leaves whose running convolution gives the cumulative delays
# of a chain. A bare leaf has a single trivial prefix (itself); a Sequential
# flattens to its observed leaves (`_observed_leaves`).
_stack_leaves(d::UnivariateDistribution) = (d,)
_stack_leaves(d::Sequential) = Tuple(_observed_leaves(d.components))

# The cumulative (prefix) delay distribution to event index `i`: the convolution
# of the first `i` leaves. `i == length(leaves)` is the total delay. Composed at
# the distribution level only; no vector work here.
function _prefix_delay(leaves::Tuple, i::Integer)
    return i == 1 ? leaves[1] : convolve_distributions(leaves[1:i])
end

# --- event-name -> prefix-index map ----------------------------------------
#
# The i-th prefix delay (first `i` leaves) ends at target event `target_i`, so
# the target names index the prefixes one-to-one. A `Sequential` exposes
# `[origin, target_1, ..., target_k]`, so its targets are everything after the
# origin. A bare leaf has a single target event, its endpoint, conventionally
# named `:event_1` (there is no separate origin to strip), so its target list is
# `(:event_1,)` directly — selecting that name must reach the single prefix, not
# an empty target set.
_stack_target_names(d::Sequential) = _flat_event_names(d)[2:end]
_stack_target_names(::UnivariateDistribution) = (:event_1,)

# The prefix index of a requested event name, erroring clearly otherwise.
function _event_prefix_index(targets, name::Symbol)
    idx = findfirst(==(name), targets)
    idx === nothing && throw(ArgumentError(
        "event $(repr(name)) is not produced by this stack; available events " *
        "are $(collect(targets))"))
    return idx
end

# --- public API: a convolve_distributions renewal method -------------------

@doc "

Convolve a timeseries through a composed delay stack to event counts.

`convolve_distributions(stack, series)`, where `series` is a numeric timeseries
vector, discretises the stack's delay to a PMF over the unit grid and returns
the causal discrete convolution of `series` with that PMF, truncated to the
`series` window. With `series` the expected events at times `0, 1, ..., t` (e.g.
infections), the result is the expected event counts at the same times (the
EpiNow2-style latent / observation layer).

The `events` keyword selects which event(s) to produce, because a fit often
observes only some of them:

- `nothing` (default): the stack's END-POINT event; returns a bare `Vector`.
- a single event name (`Symbol`): that event's series; returns a bare `Vector`.
  An INTERIM event of a [`Sequential`](@ref) chain (a step's target event, see
  [`_flat_event_names`](@ref)) uses the cumulative delay to that event.
- a tuple of event names: returns a `NamedTuple` keyed by the requested events.

Only the requested events are discretised and convolved, so an unobserved prefix
costs nothing.

This method does a DIFFERENT operation from the distribution-level
`convolve_distributions(dists...)`. That form convolves DISTRIBUTIONS together
to produce a single `Convolved` distribution (the sum of independent delays).
This form convolves a NUMERIC SERIES through a delay PMF to produce a count
series (or a `NamedTuple` of them). They share the name but never collide: the
numeric-vector second argument (`AbstractVector{<:Real}`) selects this renewal
method, distinct from the `AbstractVector{<:UnivariateDistribution}` / tuple /
two-distribution forms, so the `convolve_distributions(dists...)` forms are
unaffected.

The stack convolution is composed at the distribution level once per requested
event; the cost is the vector convolution. The PMF depends differentiably on the
delay parameters, so gradients flow under ForwardDiff / ReverseDiff.

# Arguments
- `stack`: a composed delay stack (a [`Sequential`](@ref) chain) or a bare
  univariate delay distribution.
- `series`: the input timeseries (expected events at unit-spaced times from 0).

# Keyword Arguments
- `events`: the event(s) to produce — `nothing` (the endpoint, default), a
  single event name, or a tuple of names. For a bare-leaf stack the single
  endpoint event is named `:event_1`.
- `interval`: the discretisation grid width, which is also the series time-step.
  The series is unit-spaced and the causal convolution shifts by integer series
  steps, so this must be `1` (the default); any other value is rejected with an
  `ArgumentError` to avoid conflating the grid width with the series step.

# Examples
```@example
using CensoredDistributions, Distributions

stack = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0]
endpoint = convolve_distributions(stack, series)
```

# See also
- [`convolve_distributions`](@ref): the distribution-level convolution
- [`_flat_event_names`](@ref): the named events a chain can produce
"
function convolve_distributions(stack, series::AbstractVector{<:Real};
        events = nothing, interval = 1.0)
    # The causal convolution shifts the series by integer SERIES steps, so the
    # PMF bin width must equal the series time-step. `series` is unit-spaced
    # (see the docstring), so only `interval == 1` keeps the discretisation grid
    # aligned with the shift; any other width conflates the two and silently
    # mis-aligns the result. Reject it rather than return a wrong answer.
    isone(interval) || throw(ArgumentError(
        "interval must be 1: the series is unit-spaced and the causal " *
        "convolution shifts by integer series steps, so a PMF grid width " *
        "other than 1 conflates the discretisation width with the series " *
        "time-step. Got interval = $(interval)."))
    leaves = _stack_leaves(stack)
    targets = _stack_target_names(stack)
    maxlag = length(series) - 1
    return _convolve_events(stack, leaves, targets, series, events, maxlag,
        interval)
end

# One causal convolution of `series` through the cumulative delay to prefix `i`.
function _convolve_prefix(leaves::Tuple, i::Integer, series, maxlag, interval)
    pmf = _delay_pmf(_prefix_delay(leaves, i), maxlag, interval)
    return _causal_convolve(series, pmf)
end

# `events = nothing`: the endpoint (last prefix), bare vector.
function _convolve_events(stack, leaves, targets, series, ::Nothing, maxlag,
        interval)
    return _convolve_prefix(leaves, length(leaves), series, maxlag, interval)
end

# `events = :name`: a single requested event, bare vector.
function _convolve_events(stack, leaves, targets, series, name::Symbol, maxlag,
        interval)
    i = _event_prefix_index(targets, name)
    return _convolve_prefix(leaves, i, series, maxlag, interval)
end

# `events = (:a, :b, ...)`: a NamedTuple keyed by the requested events.
function _convolve_events(stack, leaves, targets, series, names::Tuple, maxlag,
        interval)
    vals = map(names) do name
        i = _event_prefix_index(targets, name)
        _convolve_prefix(leaves, i, series, maxlag, interval)
    end
    return NamedTuple{names}(vals)
end
