# Internal alias for the convolution constructor. The function was renamed
# from `generic_convolve` to `convolve_distributions`; bind whichever the
# loaded module exports so this file builds on either base. New code below
# calls `_convolve` exclusively.
const _convolve = isdefined(@__MODULE__, :convolve_distributions) ?
                  convolve_distributions : generic_convolve

@doc """

Data-free distribution of a chain of sequential delays evaluated against a
per-event observation vector that may contain `Missing`.

A chain ``E_0 \\to E_1 \\to \\dots \\to E_k`` links events through delays
``D_1, \\dots, D_k`` where ``D_i`` is the continuous delay from ``E_{i-1}``
to ``E_i``. `SequentialDistribution` stores only the delays and the
censoring parameters, never the data. The observation vector is supplied to
[`logpdf`](@ref) at evaluation time, and the marginalise-versus-condition
choice is made per record from the missingness pattern of that vector.

For a given observation vector with entries ``E_0, \\dots, E_k`` (each a
value or `Missing`):

- an **unobserved** intermediate event (`Missing`) is marginalised: the run
  of delays it spans is convolved (via the convolution constructor) and
  evaluated at the observed gap between the two surrounding observed events;
- an **observed** intermediate event cuts the chain there and conditions:
  its adjacent delay becomes an independent factor evaluated at the observed
  gap.

The log-density is the scalar sum of the per-segment log-densities at the
observed gaps:

```math
\\log f(E_0, \\dots, E_k) = \\sum_{j=1}^m \\log f_{S_j}(g_j),
```

where the segments ``S_j`` and their gaps ``g_j`` are determined by the
observation vector's missingness, and ``m`` is one fewer than the number of
observed events.

# Automatic differentiation

`logpdf` is safe to differentiate with the observation vector passed as a
constant. The missingness pattern drives only control flow when grouping the
chain into segments; the differentiated arithmetic sees only the concrete
observed values, so no `Union{Missing}` type ever enters the gradient tape.
This is verified across every supported backend.

# Censoring

Censoring attaches to events, not to delays. The `primary_event`
distribution censors the origin end of the first segment; the `interval`
width and right-truncation to the `horizon` are applied at the observed end
of each segment, via [`double_interval_censored`](@ref) and the
[`truncated`](@ref) method on this type. `force_numeric` forces numeric
primary-censoring integration. The struct's fields are therefore `delays`,
`primary_event`, `interval`, `horizon` and `force_numeric`.

# See also
- [`sequential_distribution`](@ref): Constructor function
- [`logpdf`](@ref): Per-record missingness-dispatched evaluation
- [`double_interval_censored`](@ref): Per-segment censoring
"""
struct SequentialDistribution{
    D <: Tuple, P, I, H} <: Distribution{Multivariate, Continuous}
    "Tuple of the continuous delay distributions ``D_1, \\dots, D_k``."
    delays::D
    "Primary event distribution censoring the origin, or `nothing`."
    primary_event::P
    "Secondary interval-censoring width, or `nothing`."
    interval::I
    "Right-truncation horizon (observation cut-off), or `nothing`."
    horizon::H
    "Whether to force numeric primary-censoring integration."
    force_numeric::Bool

    function SequentialDistribution(
            delays::D, primary_event::P, interval::I, horizon::H,
            force_numeric::Bool) where {D <: Tuple, P, I, H}
        length(delays) >= 1 ||
            throw(ArgumentError("SequentialDistribution needs >= 1 delay"))
        all(d -> d isa UnivariateDistribution, delays) ||
            throw(ArgumentError("all delays must be UnivariateDistributions"))
        new{D, P, I, H}(
            delays, primary_event, interval, horizon, force_numeric)
    end
end

@doc """

Build a [`SequentialDistribution`](@ref) from a chain's continuous delays and
its censoring parameters.

`delays[i]` is the continuous delay distribution ``D_i`` from event
``E_{i-1}`` to ``E_i``. The distribution is data-free: which events are
observed is decided per record from the observation vector passed to
[`logpdf`](@ref), not at construction.

Pass continuous delays only. Censoring is driven by the observation vector
and the keyword parameters below, not by pre-wrapping a component in
`primary_censored` or `double_interval_censored`.

# Arguments
- `delays`: Vector or tuple of the ``k`` continuous delay
  `UnivariateDistribution`s.

# Keyword Arguments
- `primary_event`: Primary event distribution censoring the origin ``E_0``,
  applied to the first segment. Defaults to `nothing` (no primary
  censoring).
- `interval`: Secondary interval-censoring width applied at the observed end
  of each segment. Defaults to `nothing`.
- `horizon`: Right-truncation horizon applied at the observed end of each
  segment. Defaults to `nothing`.
- `force_numeric`: Force numeric primary-censoring integration. Defaults to
  `false`.

# Examples
```@example
using CensoredDistributions, Distributions

delays = [Gamma(2.0, 1.0), Gamma(1.0, 1.0), LogNormal(0.5, 0.4)]
d = sequential_distribution(delays)
# Middle event unobserved: first gap marginalises D1+D2, second is D3.
logpdf(d, [0.0, missing, 3.0, 5.0])
```

# See also
- [`SequentialDistribution`](@ref): The distribution type
- [`logpdf`](@ref): Per-record missingness-dispatched evaluation
"""
function sequential_distribution(
        delays::AbstractVector{<:UnivariateDistribution};
        primary_event = nothing, interval = nothing, horizon = nothing,
        force_numeric::Bool = false)
    return SequentialDistribution(
        Tuple(delays), primary_event, interval, horizon, force_numeric)
end

function sequential_distribution(
        delays::Tuple; primary_event = nothing, interval = nothing,
        horizon = nothing, force_numeric::Bool = false)
    return SequentialDistribution(
        delays, primary_event, interval, horizon, force_numeric)
end

# Number of delays (chain edges).
_nedges(d::SequentialDistribution) = length(d.delays)

Base.length(d::SequentialDistribution) = _nedges(d)

function Base.eltype(::Type{<:SequentialDistribution{D}}) where {D <: Tuple}
    return mapreduce(eltype, promote_type, fieldtypes(D))
end

params(d::SequentialDistribution) = map(params, d.delays)

# ---------------------------------------------------------------------------
# Right-truncation as a `truncated` method on the chain
# ---------------------------------------------------------------------------

@doc """

Right-truncate a [`SequentialDistribution`](@ref) to an observation horizon.

Returns a copy of `d` carrying `horizon` as its right-truncation cut-off.
The truncation denominator is applied per record inside [`logpdf`](@ref):
for a segment whose intermediate events are observed the denominator is the
single-delay CDF up to the remaining window, while for a segment spanning
unobserved events it is the CDF of the convolution of those delays. The
correct denominator is chosen from the observation vector's missingness, so
a single horizon expresses both forms without a separate mask.

# Arguments
- `d`: the chain to right-truncate.
- `horizon`: the observation cut-off time.

# Examples
```@example
using CensoredDistributions, Distributions

d = sequential_distribution([Gamma(2.0, 1.0), Gamma(1.5, 1.0)])
dt = truncated(d, 8.0)
logpdf(dt, [0.0, missing, 4.0])
```

# See also
- [`SequentialDistribution`](@ref): The distribution type
- [`logpdf`](@ref): Applies the per-record truncation denominator
"""
function truncated(d::SequentialDistribution, horizon::Real)
    return SequentialDistribution(
        d.delays, d.primary_event, d.interval, horizon, d.force_numeric)
end

# ---------------------------------------------------------------------------
# logpdf: per-record missingness dispatch returning a scalar
# ---------------------------------------------------------------------------

@doc """

Compute the joint log probability density of a chain observation vector.

`observations` has one entry per event ``E_0, \\dots, E_k`` (length
`length(d) + 1`); each entry is a value (the event is observed) or `Missing`
(the event is unobserved). The chain is grouped into segments from the
missingness pattern, each segment is built and censored, and the result is
the scalar sum of the per-segment log-densities at the observed gaps.

Missingness drives only the control flow that groups segments; the
differentiated arithmetic sees only the concrete observed values, so this is
safe to differentiate with `observations` held constant.

See also: [`SequentialDistribution`](@ref), [`pdf`](@ref)
"""
function logpdf(d::SequentialDistribution, observations::AbstractVector)
    length(observations) == _nedges(d) + 1 || throw(DimensionMismatch(
        "expected $(_nedges(d) + 1) observation entries, got " *
        "$(length(observations))"))

    # Pre-pass (pure control flow on the constant observation vector):
    # collect the observed event indices and their concrete values into
    # plain `Int` / `Float64` vectors. Reading the `Union{Missing}` entries
    # happens only here; the differentiated arithmetic below touches only the
    # resulting concrete `Float64` gaps, so no `Union` type ever reaches the
    # gradient tape (the property that lets every AD backend differentiate
    # `logpdf` with `observations` passed as a constant).
    obs_idx, obs_val = _observed_indices_values(observations)
    length(obs_idx) >= 2 || throw(ArgumentError(
        "need at least two observed events to define a gap"))

    # Sum the per-segment log-densities at the observed gaps. The segment
    # distribution is built from the delays (control flow), then evaluated at
    # a concrete value gap.
    total = zero(promote_type(eltype(obs_val), float(eltype(d))))
    for j in 1:(length(obs_idx) - 1)
        seg = _segment_distribution(d, obs_idx[j], obs_idx[j + 1], j == 1)
        gap = obs_val[j + 1] - obs_val[j]
        total += logpdf(seg, gap)
    end
    return total
end

# Pre-pass: walk the (constant) observation vector and return the observed
# event indices (`Vector{Int}`) and their concrete values (`Vector{Float64}`).
# Kept separate from the arithmetic so the `Union{Missing}` handling is pure
# control flow and the differentiated path sees only concrete values.
function _observed_indices_values(observations)
    idx = Int[]
    val = Float64[]
    for i in eachindex(observations)
        o = observations[i]
        if !(o === missing)
            push!(idx, i)
            push!(val, Float64(o))
        end
    end
    return idx, val
end

# Build the segment distribution spanning observed events at chain indices
# `a` and `b` (1-based over E0..Ek). The delays linking them are
# `delays[a:(b-1)]`. A single delay (adjacent observed events) is one factor;
# a run of two or more (unobserved events between) is one convolution of the
# continuous cores. Per-segment censoring (primary at the origin segment,
# interval and horizon truncation at the observed end) is then applied.
function _segment_distribution(d::SequentialDistribution, a, b, is_first)
    run = d.delays[a:(b - 1)]
    base = length(run) == 1 ? run[1] :
           _convolve(map(_continuous_delay, collect(run)))
    pe = is_first ? d.primary_event : nothing
    return _censor_segment(base, pe, d.interval, d.horizon, d.force_numeric)
end

# Continuous underlying delay of a component, for marginalisation. Strips
# every censoring layer via `get_dist_recursive` so a censored component
# contributes only its continuous delay to the convolution, never a discrete
# object. A `Convolved` run component is already a continuous sum, so it is
# left intact for the convolution to fold.
_continuous_delay(d::UnivariateDistribution) = get_dist_recursive(d)
_continuous_delay(d::Convolved) = d

# Apply the per-segment censoring. With no censoring the base distribution is
# returned untouched. Primary censoring (with interval) is composed via
# `double_interval_censored`; otherwise interval and horizon truncation are
# applied directly so the origin is not censored spuriously. The horizon is
# applied as right-truncation, dispatching the single-delay versus
# convolved-chain denominator from whether `base` is a `Convolved`.
function _censor_segment(base, primary_event, interval, horizon, force_numeric)
    if primary_event === nothing && interval === nothing && horizon === nothing
        return base
    end
    if primary_event === nothing
        return _truncate_interval(base, horizon, interval)
    end
    return double_interval_censored(
        base; primary_event = primary_event, upper = horizon,
        interval = interval, force_numeric = force_numeric)
end

# Right-truncate to the horizon then interval-censor, mirroring the order in
# `double_interval_censored`. The horizon denominator is the segment's own
# CDF, which is the single-delay CDF for a bare delay and the convolution CDF
# for a `Convolved` run (the per-record dispatch required for chain
# truncation).
function _truncate_interval(base, horizon, interval)
    result = horizon === nothing ? base : truncated(base; upper = horizon)
    return interval === nothing ? result : interval_censored(result, interval)
end

@doc """

Compute the joint probability density of a chain observation vector.

See also: [`logpdf`](@ref)
"""
function pdf(d::SequentialDistribution, observations::AbstractVector)
    return exp(logpdf(d, observations))
end

# ---------------------------------------------------------------------------
# rand: full event-time path
# ---------------------------------------------------------------------------

@doc """

Sample a full event-time path ``(E_0, E_1, \\dots, E_k)`` for the chain.

The origin ``E_0`` is at time zero and each subsequent event time is the
previous time plus a draw from the corresponding delay. Returns the length
``k + 1`` vector of event times.

See also: [`SequentialDistribution`](@ref)
"""
function Base.rand(rng::AbstractRNG, d::SequentialDistribution)
    k = _nedges(d)
    times = Vector{float(eltype(d))}(undef, k + 1)
    times[1] = zero(eltype(times))
    for i in 1:k
        times[i + 1] = times[i] + rand(rng, d.delays[i])
    end
    return times
end

Base.rand(d::SequentialDistribution) = rand(default_rng(), d)

sampler(d::SequentialDistribution) = d
