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

# --- the cumulative delay of an ordered leaf tuple -------------------------

# The cumulative delay distribution of an ordered leaf tuple: the convolution of
# all its leaves (their total delay). A single leaf is itself. Composed at the
# distribution level only; no vector work here.
function _cumulative_delay(leaves::Tuple)
    return length(leaves) == 1 ? leaves[1] : convolve_distributions(leaves)
end

# --- event specs: (name, cumulative-delay leaves, forward ops) --------------
#
# Every event a stack produces, in tree order, carrying the ordered delay leaves
# whose convolution is the cumulative delay to the event and the forward ops
# (thin/cumulative, Resolve branch probabilities) applied to its series. A
# Sequential threads the running prefix step to step (its interim events are the
# split target names, e.g. `:admit`, `:death`); a Parallel/Resolve edge fans
# an event out per branch/outcome, keyed by the user's branch/outcome name.

struct _EventSpec{L <: Tuple, O <: Tuple}
    name::Symbol
    leaves::L
    ops::O
end

# A Resolve branch probability is just a thinning factor, read by the convolve
# layer through the same forward-op path as `thin`.

# Collect the specs a (sub)stack produces, given the shared `prefix` leaves and
# `ops` above it. A Sequential threads the prefix; a Parallel hangs each branch
# off it.
function _collect_specs!(specs, d::Sequential, prefix, ops, counter)
    enames = component_names(d)
    cur, curops = prefix, ops
    for i in eachindex(d.components)
        cur,
        curops = _collect_chain_edge!(specs, enames[i], d.components[i],
            cur, curops, counter)
    end
    return nothing
end

function _collect_specs!(specs, d::Parallel, prefix, ops, counter)
    enames = component_names(d)
    for i in eachindex(d.components)
        delay, fops = _peel_forward(d.components[i])
        _collect_branch!(specs, enames[i], delay, prefix, (ops..., fops...),
            counter)
    end
    return nothing
end

# One Sequential chain edge: push its (split) target spec and return the
# (leaves, ops) the next step continues from. Peels forward wrappers first.
function _collect_chain_edge!(specs, edge_name, child, prefix, ops, counter)
    delay, fops = _peel_forward(child)
    return _chain_inner!(specs, edge_name, delay, prefix, (ops..., fops...),
        counter)
end

# A leaf chain edge: one target (split name or positional), prefix extended.
function _chain_inner!(specs, edge_name, child::UnivariateDistribution,
        prefix, ops, counter)
    split = _split_edge_name(edge_name)
    target = split === nothing ? _next_event_name(counter) : split[2]
    leaves = (prefix..., child)
    push!(specs, _EventSpec(target, leaves, ops))
    return leaves, ops
end

# A nested chain edge: recurse and continue from its terminal (last spec).
function _chain_inner!(specs, edge_name, child::Sequential, prefix, ops, counter)
    _collect_specs!(specs, child, prefix, ops, counter)
    last = specs[end]
    return last.leaves, last.ops
end

# A nested Parallel chain edge: fan out the branches; the chain is terminal here
# (continues from the shared prefix, mirroring `_nested_terminal_name`).
function _chain_inner!(specs, edge_name, child::Parallel, prefix, ops, counter)
    _collect_specs!(specs, child, prefix, ops, counter)
    return prefix, ops
end

# A Resolve chain edge: one event per LEAF outcome, each thinned by its branch
# probability; a NON-TERMINAL outcome whose payload is a composer SUBTREE
# fans the subtree's own events out, each carrying the outcome's
# branch-probability thinning in the forward op PREFIX so its sub-stream is the
# outcome's mass times the subtree convolution. Terminal (continues from the
# shared prefix). A no-event outcome produces NO event series and is skipped (its
# mass leaves the observed stream).
function _chain_inner!(specs, edge_name, c::Resolve, prefix, ops, counter)
    for i in eachindex(c.names)
        _is_no_event(c.delays[i]) && continue
        delay, fops = _peel_forward(c.delays[i])
        _collect_branch!(specs, c.names[i], delay, prefix,
            (ops..., fops..., ThinOp(c.branch_probs[i])), counter)
    end
    return prefix, ops
end

# A racing-hazard chain edge: one event per cause, each the cause-resolved
# SUB-density `f_j ∏_{k≠j} S_k` (sub-stochastic, NOT renormalised; its mass is
# the derived winning probability). No thinning op — the winning mass is already
# in the sub-density. Terminal (continues from the shared prefix).
function _chain_inner!(specs, edge_name, c::Compete, prefix, ops, counter)
    for i in eachindex(c.names)
        cause = _HazardCauseDelay(c, i)
        _collect_branch!(specs, c.names[i], cause, prefix, ops, counter)
    end
    return prefix, ops
end

# A Parallel/Resolve branch keyed by the user's branch/outcome name: a leaf
# branch is one event; a nested composer keeps its own sub-event names.
function _collect_branch!(specs, bname, delay::UnivariateDistribution, prefix,
        ops, counter)
    push!(specs, _EventSpec(bname, (prefix..., delay), ops))
    return nothing
end
function _collect_branch!(specs, bname, delay::Union{Sequential, Parallel},
        prefix, ops, counter)
    _collect_specs!(specs, delay, prefix, ops, counter)
    return nothing
end
function _collect_branch!(specs, bname, c::AbstractOneOf, prefix, ops, counter)
    _chain_inner!(specs, bname, c, prefix, ops, counter)
    return nothing
end

# All event specs of a stack, in tree order (names mirror `event_names`).
function _event_specs(stack::Union{Sequential, Parallel})
    specs = _EventSpec[]
    counter = Ref(0)
    _root_origin_name(stack, counter)
    _collect_specs!(specs, stack, (), (), counter)
    return specs
end
function _event_specs(stack::UnivariateDistribution)
    delay, ops = _peel_forward(stack)
    return _EventSpec[_EventSpec(:event_1, (delay,), ops)]
end

# A standalone Resolve / Compete fans an event out per outcome (each
# thinned by its branch probability, resp. the cause-resolved sub-density), the
# renewal layer's per-outcome partition. Dispatches on `AbstractOneOf` so
# both one_of nodes use the same standalone entry; the per-outcome arithmetic
# is selected inside `_chain_inner!`.
function _event_specs(c::AbstractOneOf)
    specs = _EventSpec[]
    _chain_inner!(specs, :one_of, c, (), (), Ref(0))
    return specs
end

# Find a requested event spec by name, erroring clearly otherwise.
function _find_spec(specs, name::Symbol)
    idx = findfirst(s -> s.name == name, specs)
    idx === nothing && throw(ArgumentError(
        "event $(repr(name)) is not produced by this stack; available events " *
        "are $([s.name for s in specs])"))
    return specs[idx]
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
  `_flat_event_names`) uses the cumulative delay to that event.
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
- `_flat_event_names`: the named events a chain can produce
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
    specs = _event_specs(stack)
    maxlag = length(series) - 1
    return _select_specs(specs, events, series, maxlag, interval)
end

# Convolve one event spec: causal convolution of `series` with the cumulative
# delay PMF, then the spec's forward ops (thin/cumulative/branch-prob factor).
function _convolve_spec(spec, series, maxlag, interval)
    pmf = _delay_pmf(_cumulative_delay(spec.leaves), maxlag, interval)
    return _apply_forward_ops(_causal_convolve(series, pmf), spec.ops)
end

# `events = nothing`: the endpoint (last spec), bare vector. For a branched stack
# pass `events` explicitly; the default is the last terminal in tree order.
function _select_specs(specs, ::Nothing, series, maxlag, interval)
    return _convolve_spec(specs[end], series, maxlag, interval)
end

# `events = :name`: a single requested event, bare vector.
function _select_specs(specs, name::Symbol, series, maxlag, interval)
    return _convolve_spec(_find_spec(specs, name), series, maxlag, interval)
end

# `events = (:a, :b, ...)`: a NamedTuple keyed by the requested events.
function _select_specs(specs, names::Tuple, series, maxlag, interval)
    vals = map(
        n -> _convolve_spec(_find_spec(specs, n), series, maxlag, interval),
        names)
    return NamedTuple{names}(vals)
end

# ============================================================================
# Opt-in build-once delay PMF for vector evaluation
# ============================================================================
#
# The renewal method above rebuilds the delay PMF on EVERY call. For the
# nowcasting use case a single delay PMF is applied across a whole vector of
# reference dates / many timeseries (the delay params are FIXED for the build),
# so rebuilding the (relatively expensive) discretised PMF per element is wasted
# work. `DelayPMF` is an EXPLICIT precomputed-PMF value object: the caller builds
# it ONCE with `discretise_pmf` and reuses it across many evaluations.
#
# The object is IMMUTABLE and holds no mutable memo: it is built from whatever
# parameter type the delay carries (a plain `Float64` or an AD `Dual`/tracked
# number), so the build itself differentiates and the masses propagate gradients.
# When the parameters change the caller builds a NEW object; there is no hidden
# param-keyed cache that could go stale under sampling (the Enzyme footgun), and
# no interpolation/approximation — the masses are EXACTLY the `_delay_pmf`
# interval probabilities the rebuild-every-time path computes, so results are
# numerically identical.

@doc raw"
A precomputed discretised delay PMF, built ONCE and reused across many vector
evaluations.

`DelayPMF` holds the raw [`interval_censored`](@ref) interval masses of a delay
distribution on the unit-spaced grid ``[0, 1), [1, 2), \dots, [m, m+1)`` (where
``m`` is `maxlag`), so a single discretisation is shared across a whole vector of
reference dates / timeseries instead of being rebuilt per element. This is the
nowcasting build-once optimisation: discretise the delay once, then convolve or
look it up across the whole reference-date vector.

Build it with [`discretise_pmf`](@ref); apply it with
`convolve_distributions(pmf, series)` (the causal renewal convolution) or look up
masses at integer lags with `pdf(pmf, lags)`. The object is immutable and carries
no mutable cache: the masses keep the delay's parameter type, so the build and
every reuse differentiate cleanly, and a parameter change is handled by building
a fresh object (never a stale memo).

# Fields
- `masses`: the length `maxlag + 1` vector of interval probabilities.
- `interval`: the grid width the masses were discretised on.

# See also
- [`discretise_pmf`](@ref): the build-once constructor.
- [`convolve_distributions`](@ref): apply the PMF across a series.
"
struct DelayPMF{V <: AbstractVector, I <: Real}
    "The discretised interval masses over the grid `0..maxlag`."
    masses::V
    "The grid width the masses were discretised on."
    interval::I

    function DelayPMF(masses::V, interval::I) where {
            V <: AbstractVector, I <: Real}
        length(masses) >= 1 ||
            throw(ArgumentError("DelayPMF needs at least one mass"))
        interval > 0 ||
            throw(ArgumentError("DelayPMF interval must be positive"))
        new{V, I}(masses, interval)
    end
end

# The number of grid points is `maxlag + 1`; `maxlag` is the largest integer lag
# the PMF carries a mass for.
Base.length(pmf::DelayPMF) = length(pmf.masses)
_maxlag(pmf::DelayPMF) = length(pmf.masses) - 1

@doc raw"
Discretise a delay distribution to a [`DelayPMF`](@ref) ONCE for reuse across a
vector of evaluation points.

`discretise_pmf(delay, maxlag; interval = 1.0)` computes the raw
[`interval_censored`](@ref) interval masses of `delay` on the grid
``[0, 1), \dots, [\text{maxlag}, \text{maxlag} + 1)`` (scaled by `interval`),
returning a precomputed [`DelayPMF`](@ref) the caller passes into
`convolve_distributions(pmf, series)` or `pdf(pmf, lags)`. Building it once and
reusing it avoids rediscretising the delay per reference date / per record — the
nowcasting build-once optimisation.

The masses are EXACTLY those the rebuild-every-time
`convolve_distributions(delay, series)` path computes (raw interval
probabilities, no renormalise, no interpolation), so a prebuilt PMF gives
numerically identical results. The masses keep the delay's parameter type, so the
discretisation differentiates w.r.t. the delay parameters; a parameter change is
handled by calling `discretise_pmf` again (there is no stale cache).

# Arguments
- `delay`: the delay distribution to discretise (e.g. a leaf or a
  [`Convolved`](@ref) total delay).
- `maxlag`: the largest integer lag to carry a mass for; the PMF has
  `maxlag + 1` entries.

# Keyword Arguments
- `interval`: the discretisation grid width (default `1.0`).

# Examples
```@example
using CensoredDistributions, Distributions

delay = convolve_distributions(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
# Build the delay PMF ONCE for a 30-day reference window.
pmf = CensoredDistributions.discretise_pmf(delay, 30)

# Reuse it across many reference-date series without rediscretising.
infections = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]
counts = convolve_distributions(pmf, infections)
```

# See also
- [`DelayPMF`](@ref): the precomputed-PMF object.
- [`convolve_distributions`](@ref): apply the PMF across a series.
"
function discretise_pmf(delay::UnivariateDistribution, maxlag::Integer;
        interval::Real = 1.0)
    maxlag >= 0 ||
        throw(ArgumentError("maxlag must be non-negative, got $(maxlag)"))
    return DelayPMF(_delay_pmf(delay, maxlag, interval), interval)
end

@doc raw"
Apply a precomputed [`DelayPMF`](@ref) across a timeseries with the causal
renewal convolution, reusing the build-once PMF.

`convolve_distributions(pmf, series)` is the same causal, window-truncated
convolution as `convolve_distributions(delay, series)` but takes a PMF that was
discretised ONCE (via [`discretise_pmf`](@ref)) instead of rebuilding it. The
result is numerically identical to the rebuild-every-time path when the PMF was
built from the same `delay`. This is the nowcasting build-once path: discretise
the delay once, then push every reference-date series through the same PMF.

# Arguments
- `pmf`: a precomputed [`DelayPMF`](@ref).
- `series`: the input timeseries (expected events at unit-spaced times from 0).

# See also
- [`discretise_pmf`](@ref): build the PMF once.
- [`DelayPMF`](@ref): the precomputed-PMF object.
"
function convolve_distributions(pmf::DelayPMF, series::AbstractVector{<:Real})
    isone(pmf.interval) || throw(ArgumentError(
        "convolve_distributions(pmf, series) needs a unit-spaced PMF: the " *
        "causal convolution shifts by integer series steps, so a PMF grid " *
        "width other than 1 conflates the discretisation width with the " *
        "series time-step. Got interval = $(pmf.interval)."))
    return _causal_convolve(series, pmf.masses)
end

@doc raw"
Look up the precomputed [`DelayPMF`](@ref) masses at integer lags.

`pdf(pmf, lags)` returns the discretised interval mass at each lag in `lags` (an
integer or a vector of integers), reusing the build-once PMF instead of
re-evaluating the censored density per lag. A lag outside `0..maxlag` returns a
zero of the PMF's element type (no mass is carried there).

This is the per-reference-date lookup the nowcasting path uses once the delay PMF
is built: every reference date reads the same precomputed masses at ~O(1).

# Arguments
- `pmf`: a precomputed [`DelayPMF`](@ref).
- `lags`: an integer lag, or a vector of integer lags.

# See also
- [`discretise_pmf`](@ref): build the PMF once.
"
function pdf(pmf::DelayPMF, lag::Integer)
    (0 <= lag <= _maxlag(pmf)) || return zero(eltype(pmf.masses))
    return @inbounds pmf.masses[lag + 1]
end

function pdf(pmf::DelayPMF, lags::AbstractVector{<:Integer})
    return map(l -> pdf(pmf, l), lags)
end
