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
# This reuses the existing distribution-level `convolve_distributions(dists...)`:
# the second positional argument is `AbstractVector{<:Real}` (a numeric series),
# distinct from the `AbstractVector{<:UnivariateDistribution}` / two-distribution
# forms, so the renewal method and the distribution-args forms never collide.
#
# Event selectivity. In a fit only SOME events are observed (e.g. only onsets),
# so the `events` keyword names which event(s) to produce. The default is the
# END-POINT event; a [`Sequential`](@ref) chain's named INTERIM events map to its
# PREFIX convolutions. Only the requested events are discretised and convolved,
# so an unobserved prefix costs nothing.
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
# (thin/cumulative, Competing branch probabilities) applied to its series. A
# Sequential threads the running prefix step to step (its interim events are the
# split target names, e.g. `:admit`, `:death`); a Parallel/Competing edge fans
# an event out per branch/outcome, keyed by the user's branch/outcome name.

struct _EventSpec{L <: Tuple, O <: Tuple}
    name::Symbol
    leaves::L
    ops::O
end

# A bare numeric forward factor (e.g. a Competing branch probability), read by
# the convolve layer the same way a `Scaled` op is.
struct _Factor{T <: Real}
    factor::T
end
_forward_apply(op::_Factor, series) = op.factor .* series

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

# A Competing chain edge: one event per outcome, each thinned by its branch
# probability; terminal (continues from the shared prefix).
function _chain_inner!(specs, edge_name, c::Competing, prefix, ops, counter)
    for i in eachindex(c.names)
        delay, fops = _peel_forward(c.delays[i])
        _collect_branch!(specs, c.names[i], delay, prefix,
            (ops..., fops..., _Factor(c.branch_probs[i])), counter)
    end
    return prefix, ops
end

# A Parallel/Competing branch keyed by the user's branch/outcome name: a leaf
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
function _collect_branch!(specs, bname, c::Competing, prefix, ops, counter)
    _chain_inner!(specs, bname, c, prefix, ops, counter)
    return nothing
end

# All event specs of a stack, in tree order (names mirror `tree_event_names`).
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

# A standalone Competing fans an event out per outcome (each thinned by its
# branch probability), the renewal layer's per-outcome partition.
function _event_specs(c::Competing)
    specs = _EventSpec[]
    _chain_inner!(specs, :competing, c, (), (), Ref(0))
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
  [`tree_event_names`](@ref)) uses the cumulative delay to that event.
- a tuple of event names: returns a `NamedTuple` keyed by the requested events.

Only the requested events are discretised and convolved, so an unobserved prefix
costs nothing. This is the same `convolve_distributions` as the
distribution-level convolution; the numeric-vector second argument selects this
renewal method, leaving the `convolve_distributions(dists...)` forms unaffected.

The stack convolution is composed at the distribution level once per requested
event; the cost is the vector convolution. The PMF depends differentiably on the
delay parameters, so gradients flow under ForwardDiff / ReverseDiff.

# Arguments
- `stack`: a composed delay stack (a [`Sequential`](@ref) chain) or a bare
  univariate delay distribution.
- `series`: the input timeseries (expected events at unit-spaced times from 0).

# Keyword Arguments
- `events`: the event(s) to produce — `nothing` (the endpoint, default), a single
  event name, or a tuple of names.
- `interval`: the discretisation interval width (default: `1.0`).

# Examples
```@example
using CensoredDistributions, Distributions

stack = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0]
endpoint = convolve_distributions(stack, series)
```

# See also
- [`convolve_distributions`](@ref): the distribution-level convolution
- [`tree_event_names`](@ref): the named events a chain can produce
"
function convolve_distributions(stack, series::AbstractVector{<:Real};
        events = nothing, interval = 1.0)
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
