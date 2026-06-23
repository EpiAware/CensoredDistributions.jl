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
    # which no AD backend can push `Dual`/tracked numbers through (#257).
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
"""
function _collect_unique_boundaries(d::IntervalCensored, x::AbstractVector{<:Real})
    # Determine promoted type for type stability
    T = promote_type(eltype(x), eltype(d.boundaries))

    # Collect all unique boundaries needed using functional approach
    boundary_pairs = if is_regular_intervals(d)
        interval = interval_width(d)
        map(x) do xi
            lower = floor_to_interval(xi, interval)
            upper = lower + interval
            (T(lower), T(upper))
        end
    else
        # For arbitrary intervals, collect all boundaries that could be needed
        boundary_pairs = map(x) do xi
            lower, upper = get_interval_bounds(d, xi)
            if !isnan(lower) && !isnan(upper)
                (T(lower), T(upper))
            else
                ()  # Empty tuple for invalid bounds
            end
        end
        # Filter out empty tuples
        filter(!isempty, boundary_pairs)
    end

    # Flatten pairs to boundaries (single vcat operation)
    boundaries = isempty(boundary_pairs) ? T[] :
                 vcat([collect(pair) for pair in boundary_pairs]...)

    # Return sorted unique boundaries without mutation. Sort first, then drop
    # adjacent duplicates, rather than `unique` (which builds an internal
    # `Dict` seen-set → a `DynamicDerivedRule{Dict{Any,Any}}` and a bitcast
    # Mooncake reverse-mode refuses to differentiate, #699). The boundaries
    # are functions of the (constant) lags, not the AD parameters, so this
    # only affects which CDF evaluations happen, never their tangents.
    return _sorted_unique(boundaries)
end

"""
    _sorted_unique(v)

Return the sorted unique elements of `v`. Equivalent to `sort(unique(v))`
but implemented with a self-contained insertion sort and an adjacent-dedup
pass, avoiding two things that break Mooncake reverse-mode (#699):

  - the `Dict`-backed `unique` (a `DynamicDerivedRule{Dict{Any,Any}}`), and
  - `Base`'s float sort, which reinterprets `Float64` as `UInt64` (the
    IEEE-float radix optimisation, a bitcast Mooncake refuses).

The boundaries are functions of the (constant) lags and carry no
meaningful tangent, so insertion sort is purely to dodge the bitcast. The
boundary count is `O(nlags)` (a few dozen here), so `O(n²)` is fine.
"""
function _sorted_unique(v::AbstractVector{T}) where {T}
    n = length(v)
    n == 0 && return T[]
    # Insertion sort into a pre-sized buffer (no `reinterpret`/bitcast, no
    # `push!`-driven array growth — the latter introduces `Union`-typed
    # branches that break Enzyme's strict type analysis).
    s = collect(T, v)
    @inbounds for i in 2:n
        x = s[i]
        j = i - 1
        while j >= 1 && isless(x, s[j])
            s[j + 1] = s[j]
            j -= 1
        end
        s[j + 1] = x
    end
    # Drop adjacent duplicates into a pre-sized buffer, then trim.
    out = Vector{T}(undef, n)
    out[1] = s[1]
    m = 1
    @inbounds for i in 2:n
        if s[i] != s[i - 1]
            m += 1
            out[m] = s[i]
        end
    end
    return out[1:m]
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
`Gamma` path differentiable (see #257).

The CDF values are kept in their natural type rather than converted to the
boundary/data element type. The CDF carries the AD tangent w.r.t. the
distribution parameters; converting it to the (constant) data eltype would
strip a `Dual`/tracked number and drop the gradient (#699).

Boundaries at or below `minimum(dist)` / at or above `maximum(dist)` get
the literal CDF (`0` / `1`) instead of an evaluation, mirroring the scalar
[`pdf`](@ref) method's boundary guard. This matters for reverse-mode AD:
evaluating `_cdf_ad_safe` at a degenerate boundary (e.g. `LogNormal` at
`0`) can produce a `-Inf`/`NaN` adjoint that poisons the reverse sweep even
though the value itself is unused (the scalar path guards before the call).
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
    z = zero(Distributions.partype(dist))
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

    return map(x) do xi
        lower, upper = get_interval_bounds(d, xi)

        if isnan(lower) || isnan(upper)
            return zero(promote_type(eltype(x), eltype(d)))
        end

        # Handle boundary cases for distributions with bounded support
        # For lower bound at or below distribution minimum, CDF is 0
        cdf_lower = lower <= dist_min ? zero(promote_type(eltype(x), eltype(d))) :
                    _lookup_boundary_cdf(boundaries, cdf_values, lower)

        # For upper bound at or above distribution maximum, CDF is 1
        cdf_upper = upper >= dist_max ? one(promote_type(eltype(x), eltype(d))) :
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

    # Handle empty boundaries case (all x values outside intervals)
    T = promote_type(eltype(x), eltype(d.boundaries))
    if isempty(boundaries)
        return fill(zero(T), length(x))
    end

    # Compute CDFs once per unique boundary, stored in an array parallel to
    # the sorted unique `boundaries`. A concretely-typed parallel array plus
    # a `searchsortedfirst` index lookup replaces the old `Dict{Any,Any}`
    # cache, which forced a dynamic rule and a bitcast Mooncake reverse-mode
    # refuses to differentiate (#699). `_cdf_ad_safe` keeps the `Gamma` path
    # differentiable (see #257).
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

    # Derive the log-pdf type from the computed pdf values, not from
    # `eltype(d)` (the distribution's value type, e.g. `Float64`). Under AD
    # `pdf_val` is a `Dual`/tracked number; converting to the plain value
    # type would strip the tangent and drop the gradient (#699).
    return map(zip(x, pdf_vals)) do (xi, pdf_val)
        if !insupport(d, xi)
            oftype(log(pdf_val), -Inf)
        else
            log(pdf_val)
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
    # differentiable rather than hitting `gamma_inc` directly (#257).
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
