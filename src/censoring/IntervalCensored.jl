@doc "

Interval-censored distribution where continuous values are observed only within
intervals.

Supports both:
- Regular intervals: fixed-width intervals (e.g., daily reporting)
- Arbitrary intervals: custom interval boundaries

"
struct IntervalCensored{D <: UnivariateDistribution, T} <:
       UnivariateDistribution{ValueSupport}
    "The underlying continuous distribution"
    dist::D
    "Either a scalar (regular intervals) or vector (boundaries for arbitrary intervals)"
    boundaries::T

    function IntervalCensored(dist::D, interval::Real) where {D}
        interval > 0 || throw(ArgumentError("Interval width must be positive"))
        new{D, typeof(interval)}(dist, interval)
    end

    function IntervalCensored(dist::D, boundaries::AbstractVector{<:Real}) where {D}
        length(boundaries) >= 2 ||
            throw(ArgumentError("Must provide at least 2 interval boundaries"))
        issorted(boundaries) ||
            throw(ArgumentError("Interval boundaries must be sorted"))
        all(diff(boundaries) .> 0) ||
            throw(
                ArgumentError("Interval boundaries must be strictly increasing")
            )

        new{D, typeof(boundaries)}(dist, boundaries)
    end
end

@doc "

Construct an interval-censored distribution with regular intervals.

Creates a distribution where observations are censored to regular intervals of
width `interval`, starting from 0. For example, with `interval=1`, observations
fall into [0,1), [1,2), [2,3), etc.

# Arguments
- `dist`: The underlying continuous distribution
- `interval`: Width of regular intervals (must be positive)

# Examples
```@example
using CensoredDistributions, Distributions

# Daily reporting intervals
d = interval_censored(Normal(5, 2), 1.0)

# Evaluate distribution functions
pdf_at_5 = pdf(d, 5.0)      # probability mass at interval containing 5
q25 = quantile(d, 0.25)     # 25th percentile (interval boundary)
```
"
function interval_censored(dist::UnivariateDistribution, interval::Real)
    return IntervalCensored(dist, interval)
end

@doc "

Construct an interval-censored distribution with arbitrary intervals.

Creates a distribution where observations are censored to specified intervals
defined by the boundaries. For example, with `boundaries=[0, 2, 5, 10]`,
observations fall into [0,2), [2,5), or [5,10).

# Arguments
- `dist`: The underlying continuous distribution
- `boundaries`: Vector of interval boundaries (must be sorted and strictly increasing, minimum length 2)

# Examples
```@example
using CensoredDistributions, Distributions

# Age groups: 0-18, 18-65, 65+
age_dist = interval_censored(Normal(40, 20), [0, 18, 65, 100])
q75 = quantile(age_dist, 0.75)     # 75th percentile (boundary value)
```
"
function interval_censored(
        dist::UnivariateDistribution, boundaries::AbstractVector{<:Real})
    return IntervalCensored(dist, boundaries)
end

# Helper function to determine if we have regular or arbitrary intervals
is_regular_intervals(d::IntervalCensored{D, <:Real}) where {D} = true
is_regular_intervals(d::IntervalCensored{D, <:AbstractVector}) where {D} = false

# Show an interval-censored distribution as a one-line summary of its inner
# distribution and its interval spec, summarising an arbitrary-boundary vector by
# its count so a long boundary array is never dumped. Detailed nested inspection
# is available via `inspect`.
function Base.show(io::IO, d::IntervalCensored)
    spec = is_regular_intervals(d) ? "interval=$(d.boundaries)" :
           "boundaries=$(length(d.boundaries))"
    print(io, "IntervalCensored(", d.dist, "; ", spec, ")")
    return nothing
end

# Get interval width for regular intervals
interval_width(d::IntervalCensored{D, <:Real}) where {D} = d.boundaries

# Floor value to interval
function floor_to_interval(x::Real, interval::Real)
    return floor(x / interval) * interval
end

# Find which interval contains x for arbitrary intervals
function find_interval_index(x::Real, intervals::AbstractVector)
    if x < intervals[1]
        return 0  # Before first interval
    elseif x >= intervals[end]
        return length(intervals)  # After last interval
    else
        # Binary search for efficiency
        return searchsortedlast(intervals, x)
    end
end

# Find the appropriate boundary for quantile purposes (left boundary of
# containing interval)
function find_interval_boundary(x::Real, intervals::AbstractVector)
    idx = find_interval_index(x, intervals)
    if idx == 0
        return intervals[1]  # First boundary
    elseif idx >= length(intervals)
        return intervals[end]  # Last boundary
    else
        return intervals[idx]  # Left boundary of containing interval
    end
end

# Get interval bounds for a value
function get_interval_bounds(d::IntervalCensored, x::Real)
    if is_regular_intervals(d)
        lower = floor_to_interval(x, interval_width(d))
        upper = lower + interval_width(d)
        return (lower, upper)
    else
        idx = find_interval_index(x, d.boundaries)
        if idx == 0 || idx == length(d.boundaries)
            # Outside defined intervals
            return (NaN, NaN)
        else
            return (d.boundaries[idx], d.boundaries[idx + 1])
        end
    end
end

# Distribution interface methods
function params(d::IntervalCensored)
    if is_regular_intervals(d)
        return (params(get_dist(d))..., d.boundaries)
    else
        return (params(get_dist(d))..., d.boundaries)
    end
end

Base.eltype(::Type{<:IntervalCensored{D, T}}) where {D, T} = eltype(D)

function minimum(d::IntervalCensored)
    cont_min = minimum(get_dist(d))
    if is_regular_intervals(d)
        return floor_to_interval(cont_min, interval_width(d))
    else
        # Find first interval that could contain values
        idx = find_interval_index(cont_min, d.boundaries)
        return idx > 0 ? d.boundaries[idx] : d.boundaries[1]
    end
end

function maximum(d::IntervalCensored)
    cont_max = maximum(get_dist(d))
    if is_regular_intervals(d)
        return floor_to_interval(cont_max, interval_width(d))
    else
        # Find last interval that could contain values
        idx = find_interval_index(cont_max, d.boundaries)
        return idx < length(d.boundaries) ? d.boundaries[idx] :
               d.boundaries[end - 1]
    end
end

function insupport(d::IntervalCensored, x::Real)
    # For interval-censored distributions, support is continuous within the
    # underlying distribution. The PDF is non-zero for any x where the underlying
    # distribution has support
    return insupport(get_dist(d), x)
end

#### Probability functions

@doc "

Compute the probability mass for the interval containing `x`.

See also: [`logpdf`](@ref), [`cdf`](@ref)
"
function pdf(d::IntervalCensored, x::Real)
    lower, upper = get_interval_bounds(d, x)
    if isnan(lower) || isnan(upper)
        return 0.0
    end

    # Handle boundary cases for distributions with bounded support
    dist_min = minimum(get_dist(d))
    dist_max = maximum(get_dist(d))

    # For lower bound at or below distribution minimum, CDF is 0.
    # `_cdf_ad_safe` routes `Gamma` through `_gamma_cdf` so the boundary
    # CDF stays differentiable; stock `cdf(Gamma, x)` hits `gamma_inc`,
    # which no AD backend can push `Dual`/tracked numbers through.
    cdf_lower = lower <= dist_min ? 0.0 : _cdf_ad_safe(get_dist(d), lower)

    # For upper bound at or above distribution maximum, CDF is 1
    cdf_upper = upper >= dist_max ? 1.0 : _cdf_ad_safe(get_dist(d), upper)

    return max(cdf_upper - cdf_lower, zero(cdf_upper))
end

@doc "

Compute the log probability mass for the interval containing `x`.

See also: [`pdf`](@ref), [`logcdf`](@ref)
"
function logpdf(d::IntervalCensored, x::Real)
    if !insupport(d, x)
        return -Inf
    end
    return log(pdf(d, x))
end

#### Vectorised PDF optimization

"""
    _collect_unique_boundaries(d::IntervalCensored, x::AbstractVector)

Collect all unique interval boundaries needed for vectorised PDF computation.

Returns a sorted vector of unique boundaries with appropriate type promotion.
The boundaries are functions of the (constant) lags and the interval spec, not
the distribution's AD parameters, so AD rules mark this non-differentiable (the
`unique`/sort internals are never traced); see the AD extensions (#699, #701).
"""
function _collect_unique_boundaries(d::IntervalCensored, x::AbstractVector{<:Real})
    T = promote_type(eltype(x), eltype(d.boundaries))

    # Push into a concretely-typed `T[]` buffer (rather than a `vcat`-splat of
    # per-lag tuples, which infers `Vector{Any}` and makes the downstream
    # boundary-CDF `map` a `Union`-typed Generator that Enzyme rejects, #720).
    boundaries = T[]
    if is_regular_intervals(d)
        interval = interval_width(d)
        for xi in x
            lower = floor_to_interval(xi, interval)
            push!(boundaries, T(lower), T(lower + interval))
        end
    else
        for xi in x
            lower, upper = get_interval_bounds(d, xi)
            if !isnan(lower) && !isnan(upper)
                push!(boundaries, T(lower), T(upper))
            end
        end
    end

    return sort(unique(boundaries))
end

"""
    _interval_cdf_eltype(d::IntervalCensored, x::AbstractVector)

Element type for the cached interval CDF values and the `0`/`1` boundary
seeds in the batched PDF path.

This must follow the underlying distribution's parameter type, not just the
evaluation points. When the distribution carries AD `Dual`/tracked
parameters but `x` and the boundaries are plain `Float64`, a type derived
from the eval points alone (e.g. `Float64`) would strip the AD numbers and
break gradients on the batched path. `eltype(d)` reports the support type
(e.g. `Float64` for `Gamma` regardless of its parameter type), so it is not
enough; we promote in the actual CDF result type from `_cdf_ad_safe`.
"""
function _interval_cdf_eltype(d::IntervalCensored, x::AbstractVector{<:Real})
    # The CDF value type follows the distribution's PARAMETER type (carrying any
    # AD `Dual`/tracked number). `partype` reads it directly from the
    # parameters without EVALUATING the CDF, so the type probe never traces a
    # computation onto the AD tape. An earlier probe that evaluated `cdf` at the
    # distribution minimum tripped ReverseDiff: `cdf(LogNormal, 0.0)` has a NaN
    # gradient at the support edge, and tracing it (even just for `typeof`)
    # poisoned the batched gradient with NaN.
    cdf_t = float(Distributions.partype(get_dist(d)))
    return promote_type(eltype(x), eltype(d.boundaries), cdf_t)
end

"""
    _lookup_boundary_cdf(boundaries, cdf_values, b)

Look up the cached CDF for boundary `b` in the sorted, unique
`boundaries` vector via `searchsortedfirst`. `boundaries` and
`cdf_values` are parallel arrays produced by
[`_compute_boundary_cdfs`](@ref); the index found in `boundaries` selects
the matching `cdf_values` entry. Using parallel concretely-typed arrays
plus an index lookup (rather than a `Dict{Any,Any}`) keeps the boundary
CDF cache type-stable so the AD tangent type is tracked through it. The
`Dict{Any,Any}` version forced a `DynamicDerivedRule{Dict{Any,Any}}` and
a bitcast Mooncake reverse-mode refuses to differentiate (#699).
"""
function _lookup_boundary_cdf(boundaries::AbstractVector, cdf_values::AbstractVector, b)
    idx = searchsortedfirst(boundaries, b)
    return cdf_values[idx]
end

"""
    _compute_boundary_cdfs(d::IntervalCensored, boundaries)

Evaluate the boundary CDF once per unique boundary, returning a vector
parallel to the sorted unique `boundaries`. `_cdf_ad_safe` keeps the
`Gamma` path differentiable.

The CDF values are kept in their natural type rather than converted to the
boundary/data element type. The CDF carries the AD tangent w.r.t. the
distribution parameters; converting it to the (constant) data eltype would
strip a `Dual`/tracked number and drop the gradient.

Boundaries at or below `minimum(dist)` / at or above `maximum(dist)` get
the literal CDF (`0` / `1`) instead of an evaluation, mirroring the scalar
[`pdf`](@ref) method's boundary guard. This matters for reverse-mode AD:
evaluating `_cdf_ad_safe` at a degenerate boundary (e.g. `LogNormal` at
`0`) can produce a `-Inf`/`NaN` adjoint that poisons the reverse sweep even
though the value itself is unused (the scalar path guards before the call).
Keeping every boundary in the parallel arrays (with literal `0`/`1` seeds
for the degenerate ones) also avoids the `KeyError` the old skip-and-`Dict`
cache hit when a lag's interval bound landed exactly on `maximum(dist)`.
"""
function _compute_boundary_cdfs(d::IntervalCensored, boundaries::AbstractVector)
    dist = get_dist(d)
    dist_min = minimum(dist)
    dist_max = maximum(dist)
    # `partype(dist)` is the distribution's parameter type, which is the
    # AD-tracked number type (e.g. a `Dual`) under differentiation. A zero of
    # that type types the literal-branch `0`/`1` so the result vector stays
    # type-stable and AD-aware WITHOUT evaluating `_cdf_ad_safe` at a
    # degenerate boundary: doing so (e.g. `LogNormal` at `0`) produces a
    # `-Inf`/`NaN` reverse adjoint that poisons the sweep even though `zero`
    # discards the primal (the derivative is still taped). See #699.
    z = zero(float(Distributions.partype(dist)))
    return map(boundaries) do b
        if b <= dist_min
            z
        elseif b >= dist_max
            z + one(z)
        else
            _cdf_ad_safe(dist, b)
        end
    end
end

"""
    _compute_pdfs_with_cache(d, x, boundaries, cdf_values)

Compute PDFs efficiently using the cached boundary CDFs held in the
parallel arrays `boundaries` (sorted, unique) and `cdf_values`.

Uses the same boundary case handling as the scalar method.
"""
function _compute_pdfs_with_cache(
        d::IntervalCensored, x::AbstractVector{<:Real},
        boundaries::AbstractVector, cdf_values::AbstractVector)
    # Get distribution bounds once for boundary case handling
    dist_min = minimum(get_dist(d))
    dist_max = maximum(get_dist(d))

    # Boundary `0`/`1` seeds must carry the distribution's (possibly AD)
    # parameter type so gradients flow through the batched path.
    T = _interval_cdf_eltype(d, x)

    return map(x) do xi
        lower, upper = get_interval_bounds(d, xi)

        if isnan(lower) || isnan(upper)
            return zero(T)
        end

        # Handle boundary cases for distributions with bounded support
        # For lower bound at or below distribution minimum, CDF is 0
        cdf_lower = lower <= dist_min ? zero(T) :
                    _lookup_boundary_cdf(boundaries, cdf_values, lower)

        # For upper bound at or above distribution maximum, CDF is 1
        cdf_upper = upper >= dist_max ? one(T) :
                    _lookup_boundary_cdf(boundaries, cdf_values, upper)

        return max(cdf_upper - cdf_lower, zero(cdf_upper))
    end
end

@doc "

Compute probability masses for an array of values using optimised vectorisation.

This method collects unique interval boundaries, computes CDFs once, then uses
cached values for efficient PDF computation across the array.

See also: [`pdf`](@ref), [`logpdf`](@ref)
"
function pdf(d::IntervalCensored, x::AbstractVector{<:Real})
    # Collect all unique boundaries needed
    boundaries = _collect_unique_boundaries(d, x)

    # Element type for the CDF VALUES follows the distribution's parameter
    # type so AD `Dual`/tracked numbers flow through the batched path.
    Tval = _interval_cdf_eltype(d, x)
    if isempty(boundaries)
        return fill(zero(Tval), length(x))
    end

    # Compute CDFs once per unique boundary, stored in an array parallel to
    # the sorted unique `boundaries`. A concretely-typed parallel array plus
    # a `searchsortedfirst` index lookup replaces the old `Dict{Any,Any}`
    # cache, which forced a dynamic rule and a bitcast Mooncake reverse-mode
    # refuses to differentiate (#699). `_compute_boundary_cdfs` keeps the
    # `Gamma` path differentiable and guards degenerate boundaries with typed
    # `0`/`1` seeds (so they are never evaluated and never missing).
    cdf_values = _compute_boundary_cdfs(d, boundaries)

    # Use cached values to compute PDFs efficiently
    return _compute_pdfs_with_cache(d, x, boundaries, cdf_values)
end

@doc "

Compute log probability masses for an array of values using optimised PDF computation.

See also: [`pdf`](@ref), [`logpdf`](@ref)
"
function logpdf(d::IntervalCensored, x::AbstractVector{<:Real})
    # Use vectorised PDF computation then handle logs with proper error handling
    pdf_vals = pdf(d, x)

    # Follow the PDF value type (which carries any AD `Dual`/tracked parameter
    # type) rather than the eval points, so `log`/`-Inf` conversions do
    # not strip gradients on the batched path.
    T = promote_type(eltype(x), eltype(pdf_vals))

    return map(zip(x, pdf_vals)) do (xi, pdf_val)
        if !insupport(d, xi)
            T(-Inf)
        else
            T(log(pdf_val))
        end
    end
end

# Internal function for efficient cdf/logcdf computation
function _interval_cdf(d::IntervalCensored, x::Real, f::Function)
    # Handle edge cases first
    if x < minimum(get_dist(d))
        return f === logcdf ? -Inf : 0.0
    elseif x >= maximum(get_dist(d))
        return f === logcdf ? 0.0 : 1.0
    end

    # Route through the AD-safe helpers so the `Gamma` path stays
    # differentiable rather than hitting `gamma_inc` directly.
    f_safe = f === logcdf ? _logcdf_ad_safe : _cdf_ad_safe

    if is_regular_intervals(d)
        # For regular intervals, use floor behavior from Discretised
        discretised_x = floor_to_interval(x, interval_width(d))
        return f_safe(get_dist(d), discretised_x)
    else
        # For arbitrary intervals, use the lower bound of the
        # containing interval
        idx = find_interval_index(x, d.boundaries)
        if idx == 0
            return f === logcdf ? -Inf : 0.0
        elseif idx >= length(d.boundaries)
            return f_safe(get_dist(d), d.boundaries[end])
        else
            return f_safe(get_dist(d), d.boundaries[idx])
        end
    end
end

@doc "

Compute the cumulative distribution function.

See also: [`logcdf`](@ref)
"
function cdf(d::IntervalCensored, x::Real)
    return _interval_cdf(d, x, cdf)
end

@doc "

Compute the log cumulative distribution function.

See also: [`cdf`](@ref)
"
function logcdf(d::IntervalCensored, x::Real)
    return _interval_cdf(d, x, logcdf)
end

function ccdf(d::IntervalCensored, x::Real)
    return 1 - cdf(d, x)
end

function logccdf(d::IntervalCensored, x::Real)
    # Use log1mexp for numerical stability: log(1 - exp(logcdf))
    logcdf_val = logcdf(d, x)

    # Handle edge cases
    if logcdf_val == -Inf
        return 0.0  # log(1) when CDF = 0
    elseif logcdf_val >= 0.0
        return -Inf  # log(0) when CDF = 1
    end

    return log1mexp(logcdf_val)
end

#### Sampling

@doc "

Generate a random sample by discretising a sample from the underlying
distribution.

See also: [`quantile`](@ref)
"
function Base.rand(rng::AbstractRNG, d::IntervalCensored)
    # Sample once from the underlying distribution
    x = rand(rng, get_dist(d))

    if is_regular_intervals(d)
        # Discretise to regular intervals
        return floor_to_interval(x, interval_width(d))
    else
        # Find which arbitrary interval contains x
        idx = find_interval_index(x, d.boundaries)
        if idx == 0 || idx >= length(d.boundaries)
            # Outside intervals - this shouldn't happen if dist is properly
            # bounded
            # Return closest boundary
            return idx == 0 ? d.boundaries[1] : d.boundaries[end]
        else
            return d.boundaries[idx]
        end
    end
end

#### Quantile function

@doc "

Compute the quantile using numerical optimization.

The returned quantile respects the interval structure:
- For regular intervals: quantiles are multiples of the interval width
- For arbitrary intervals: quantiles correspond to interval boundary values

See also: [`cdf`](@ref)
"
function quantile(d::IntervalCensored, p::Real)
    # Post-processing function to snap result to interval boundary
    result_postprocess_fn = function (result)
        return if is_regular_intervals(d)
            floor_to_interval(result, interval_width(d))
        else
            find_interval_boundary(result, d.boundaries)
        end
    end

    return _quantile_optimization(d, p;
        result_postprocess_fn = result_postprocess_fn,
        check_nan = true)
end

# Sampler method for efficient sampling
sampler(d::IntervalCensored) = d
