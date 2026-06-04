@doc raw"""

Distribution of the observed gaps along a chain of sequential delays with
mixed observed / unobserved intermediate events.

A chain ``E_0 \to E_1 \to \dots \to E_k`` links events through delays
``D_1, \dots, D_k`` where ``D_i`` is the delay from ``E_{i-1}`` to
``E_i``. Each event carries an observation spec: an *unobserved* event
(`Missing`) is marginalised by convolution over its delay; an *observed*
event cuts the chain there and is conditioned on. `SequentialDistribution` is
the multivariate distribution of the gaps between consecutive observed
events.

The chain is split at construction into independent **segments**, one per
consecutive pair of observed events:

- a maximal run of delays between two observed events (the events between
  them unobserved) becomes one convolution segment
  ([`generic_convolve`](@ref) over the run);
- a single delay between two adjacent observed events becomes one factor
  segment (that delay directly).

Censoring attaches to events, not to delays. Primary event censoring is
applied at the origin end of the *first* segment; truncation (`upper`) and
interval censoring (`interval`) are applied at the observed end of *each*
segment, via [`double_interval_censored`](@ref). Two adjacent observed
intermediate events therefore yield two separate double-censored factors,
never a convolution of two interval-censored quantities.

The segments condition only on their two surrounding observed events, so
they are independent and the joint log-density factorises:

```math
\log f(g_1, \dots, g_m) = \sum_{j=1}^m \log f_{S_j}(g_j),
```

where ``g_j`` is the observed gap for segment ``S_j`` and ``m`` is the
number of segments (one fewer than the number of observed events). This is
the composition layer over the existing convolution and censoring
primitives; it adds no new quadrature.

# See also
- [`sequential_distribution`](@ref): Constructor function
- [`generic_convolve`](@ref): Marginalises an unobserved run
- [`double_interval_censored`](@ref): Per-segment censoring
"""
struct SequentialDistribution{S <: Tuple} <:
       Distribution{Multivariate, Continuous}
    "Tuple of independent univariate gap distributions, one per segment."
    segments::S

    function SequentialDistribution(segments::S) where {S <: Tuple}
        length(segments) >= 1 ||
            throw(ArgumentError("SequentialDistribution needs at least one segment"))
        all(s -> s isa UnivariateDistribution, segments) ||
            throw(ArgumentError(
                "all segments must be UnivariateDistributions"))
        new{S}(segments)
    end
end

@doc raw"""

Build a [`SequentialDistribution`](@ref) from a chain's delays and per-event
observations (event-first interface).

`delays[i]` is the *continuous* delay distribution ``D_i`` from event
``E_{i-1}`` to ``E_i``, so `delays` has length ``k`` and `observations`
has length ``k + 1`` (one entry per event ``E_0, \dots, E_k``). An
observation entry is either `Missing` (the event is unobserved and its
delay marginalised) or a value (the event is observed and conditioned on,
cutting the chain there).

The constructor finds the observed events, then for each consecutive pair
groups the spanning delays into one segment: a single delay between
adjacent observed events becomes one factor, a run of two or more delays
(unobserved events between the pair) becomes one [`generic_convolve`](@ref).
Per-event censoring is then applied as a unit to each segment via
[`double_interval_censored`](@ref): `primary_event` at the origin end of
the first segment, and `upper` / `interval` at the observed end of every
segment. Unobserved events at the ends of the chain (before the first or
after the last observed event) carry no segment and are dropped.

Pass *continuous* delays only: never pre-wrap a component in
`primary_censored` / `double_interval_censored` here, since censoring is
driven by the observation spec. To compose pre-built segment distributions
directly, use the escape-hatch method
[`sequential_distribution(segments)`](@ref).

# Arguments
- `delays`: Vector/tuple of the ``k`` continuous delay
  `UnivariateDistribution`s.
- `observations`: Vector of length ``k + 1`` of observed event times or
  `Missing`.

# Keyword Arguments
- `primary_event`: Primary event distribution censoring the origin ``E_0``;
  applied to the first segment. Defaults to `nothing` (no primary
  censoring).
- `upper`: Truncation horizon applied at the observed end of each segment.
  Defaults to `nothing`.
- `interval`: Secondary interval-censoring width applied at the observed
  end of each segment. Defaults to `nothing`.
- `force_numeric`: Force numeric primary-censoring integration. Defaults to
  `false`.

# Examples
```@example
using CensoredDistributions, Distributions

# Three delays, middle event unobserved: segments are
# (D1 ⊕ D2) for the first gap and D3 for the second.
delays = [Gamma(2.0, 1.0), Gamma(1.0, 1.0), LogNormal(0.5, 0.4)]
obs = [0.0, missing, 3.0, 5.0]
d = sequential_distribution(delays, obs)
logpdf(d, [3.0, 2.0])
```

# See also
- [`SequentialDistribution`](@ref): The distribution type
- [`double_interval_censored`](@ref): Per-segment censoring
"""
function sequential_distribution(
        delays::AbstractVector{<:UnivariateDistribution},
        observations::AbstractVector{<:Union{Missing, Real}};
        kwargs...)
    length(observations) == length(delays) + 1 || throw(ArgumentError(
        "observations must have length(delays) + 1 entries " *
        "(one per event E0..Ek)"))
    # Derive the design (which events are observed) from the value pattern:
    # `Missing` means unobserved-by-design, a value means observed. The
    # values themselves do not enter the gap distributions (gaps are
    # differences), only the observed/unobserved pattern does.
    observed = [!(o === missing) for o in observations]
    return sequential_distribution(delays, observed; kwargs...)
end

@doc raw"""

Build a [`SequentialDistribution`](@ref) from a chain's delays and an
observation *design* (structure-first interface).

`delays[i]` is the continuous delay ``D_i`` from ``E_{i-1}`` to ``E_i``
(length ``k``); `observed[i]` is `true` when event ``E_{i-1}`` is observed
and `false` when it is unobserved-by-design (length ``k + 1``, one per
event ``E_0, \dots, E_k``).

This constructor takes the design only, with no observed values, so the
returned distribution can be sampled (`rand(d)` simulates the observed
gaps) before any data exist, and evaluated (`logpdf(d, gaps)`) once gaps
are available. The value-pattern method
[`sequential_distribution(delays, observations)`](@ref) derives this
boolean design from a `Missing`/value vector for inference.

Segments and per-segment censoring are assembled exactly as for the
value-pattern method (convolve unobserved runs, factorise observed-adjacent
delays, primary at the origin end of the first segment, `upper`/`interval`
at the observed end of each segment).

# Arguments
- `delays`: Vector of the ``k`` continuous delay `UnivariateDistribution`s.
- `observed`: Boolean vector of length ``k + 1`` marking observed events.

# Keyword Arguments
- `primary_event`, `upper`, `interval`, `force_numeric`: as for the
  value-pattern method.

# Examples
```@example
using CensoredDistributions, Distributions

delays = [Gamma(2.0, 1.0), Gamma(1.0, 1.0), LogNormal(0.5, 0.4)]
# E0, E2, E3 observed; E1 unobserved-by-design.
d = sequential_distribution(delays, [true, false, true, true])
sim_gaps = rand(d)
logpdf(d, sim_gaps)
```

# See also
- [`SequentialDistribution`](@ref): The distribution type
"""
function sequential_distribution(
        delays::AbstractVector{<:UnivariateDistribution},
        observed::AbstractVector{Bool};
        primary_event = nothing, upper = nothing, interval = nothing,
        force_numeric::Bool = false)
    length(observed) == length(delays) + 1 || throw(ArgumentError(
        "observed must have length(delays) + 1 entries " *
        "(one per event E0..Ek)"))

    # Indices (1-based over events E0..Ek) of the observed events.
    obs_idx = [i for i in eachindex(observed) if observed[i]]
    length(obs_idx) >= 2 || throw(ArgumentError(
        "need at least two observed events to define a gap"))

    # One segment per consecutive observed pair. The delays spanning the
    # pair (a, b) are D_{a}, ..., D_{b-1} in 1-based delay indexing, since
    # delay i links event i to event i+1.
    segments = map(1:(length(obs_idx) - 1)) do j
        a = obs_idx[j]
        b = obs_idx[j + 1]
        run = delays[a:(b - 1)]
        is_first = j == 1
        return _build_segment(
            run, is_first ? primary_event : nothing, upper, interval,
            force_numeric)
    end

    return SequentialDistribution(Tuple(segments))
end

@doc raw"""

Build a [`SequentialDistribution`](@ref) from pre-built segment distributions
(escape hatch).

Each element of `segments` is the gap distribution for one segment (any
`UnivariateDistribution`, e.g. a [`generic_convolve`](@ref) for an
unobserved run or a [`double_interval_censored`](@ref) factor). The
log-density is the sum of the per-segment log-densities at the observed
gaps. Use this when composing the segment structure manually; the
event-first method [`sequential_distribution(delays, observations)`](@ref) is
preferred when starting from raw delays and an observation spec.

# Arguments
- `segments`: Vector/tuple of the per-segment `UnivariateDistribution`s.

# Examples
```@example
using CensoredDistributions, Distributions

run = generic_convolve(Gamma(2.0, 1.0), Gamma(1.0, 1.0))
factor = double_interval_censored(LogNormal(0.5, 0.4); interval = 1.0)
d = sequential_distribution([run, factor])
logpdf(d, [3.0, 2.0])
```

# See also
- [`SequentialDistribution`](@ref): The distribution type
"""
function sequential_distribution(
        segments::AbstractVector{<:UnivariateDistribution})
    return SequentialDistribution(Tuple(segments))
end

sequential_distribution(segments::Tuple) = SequentialDistribution(segments)

# Build one segment's gap distribution from its run of delays and the
# per-event censoring spec, dispatching on whether the run spans observed
# events (a single delay) or marginalises unobserved ones (a run of >= 2).
#
# - OBSERVED-adjacent single delay: the component is kept as a FACTOR,
#   retaining whatever censoring it already carries (e.g. a passed
#   `double_interval_censored`/`truncated`/`primary_censored`), then any
#   additional per-segment endpoint censoring is composed on top.
# - UNOBSERVED run (>= 2 delays): the intermediate events are marginalised,
#   so the run is convolved over the *continuous* underlying delays
#   (extracted with `get_dist` so a censored component contributes its
#   continuous core, never a discrete interval-censored object), then the
#   endpoint censoring is applied to the convolution as a unit.
function _build_segment(run, primary_event, upper, interval, force_numeric)
    if length(run) == 1
        return _censor_segment(
            run[1], primary_event, upper, interval, force_numeric)
    end
    base = generic_convolve(map(_continuous_delay, collect(run)))
    return _censor_segment(base, primary_event, upper, interval, force_numeric)
end

# Continuous underlying delay of a component, for marginalisation. Strips
# every censoring layer (primary/interval/truncation) via
# `get_dist_recursive` so a `double_interval_censored` component contributes
# only its continuous delay to the convolution, never a discrete object. A
# bare distribution is returned unchanged; a `Convolved` run component is
# already a continuous sum, so it is left intact for the convolution to
# fold.
_continuous_delay(d::UnivariateDistribution) = get_dist_recursive(d)
_continuous_delay(d::Convolved) = d

# Apply the per-segment censoring. When no censoring is requested the base
# distribution is returned untouched (keeping the bare-delay / convolution
# fast paths). Otherwise `double_interval_censored` composes primary
# censoring, truncation and interval censoring in the correct order.
function _censor_segment(base, primary_event, upper, interval, force_numeric)
    if primary_event === nothing && upper === nothing && interval === nothing
        return base
    end
    pe = primary_event === nothing ? Uniform(0, 1) : primary_event
    # `double_interval_censored` always applies primary censoring; when the
    # caller gave no primary_event for this segment, fall back to truncation
    # / interval censoring of the bare delay so the origin is not censored
    # spuriously.
    if primary_event === nothing
        return _truncate_interval(base, upper, interval)
    end
    return double_interval_censored(
        base; primary_event = pe, upper = upper, interval = interval,
        force_numeric = force_numeric)
end

# Truncation then interval censoring of a delay with no primary censoring,
# mirroring the order in `double_interval_censored`.
function _truncate_interval(base, upper, interval)
    result = upper === nothing ? base : truncated(base; upper = upper)
    return interval === nothing ? result : interval_censored(result, interval)
end

# ---------------------------------------------------------------------------
# Distributions.jl interface
# ---------------------------------------------------------------------------

Base.length(d::SequentialDistribution) = length(d.segments)

function Base.eltype(::Type{<:SequentialDistribution{S}}) where {S <: Tuple}
    mapreduce(eltype, promote_type, fieldtypes(S))
end

params(d::SequentialDistribution) = map(params, d.segments)

function insupport(d::SequentialDistribution, x::AbstractVector{<:Real})
    length(x) == length(d) || return false
    return all(((s, xi),) -> insupport(s, xi), zip(d.segments, x))
end

@doc "

Compute the joint log probability density of the segment gaps.

The segments are independent, so this is the sum of the per-segment
log-densities.

See also: [`SequentialDistribution`](@ref)
"
function logpdf(d::SequentialDistribution, x::AbstractVector{<:Real})
    length(x) == length(d) || throw(DimensionMismatch(
        "expected $(length(d)) gaps, got $(length(x))"))
    return sum(((s, xi),) -> logpdf(s, xi), zip(d.segments, x))
end

@doc "

Compute the joint probability density of the segment gaps.

See also: [`logpdf`](@ref)
"
function pdf(d::SequentialDistribution, x::AbstractVector{<:Real})
    return exp(logpdf(d, x))
end

@doc "

Sample a gap for each segment, returning the vector of gaps.

See also: [`SequentialDistribution`](@ref)
"
function Distributions._rand!(
        rng::AbstractRNG, d::SequentialDistribution, x::AbstractVector{<:Real})
    for (i, s) in enumerate(d.segments)
        x[i] = rand(rng, s)
    end
    return x
end

sampler(d::SequentialDistribution) = d
