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

    # For lower bound at or below distribution minimum, CDF is 0
    cdf_lower = lower <= dist_min ? 0.0 : cdf(get_dist(d), lower)

    # For upper bound at or above distribution maximum, CDF is 1
    cdf_upper = upper >= dist_max ? 1.0 : cdf(get_dist(d), upper)

    return cdf_upper - cdf_lower
end

@doc "

Compute the log probability mass for the interval containing `x`.

See also: [`pdf`](@ref), [`logcdf`](@ref)
"
function logpdf(d::IntervalCensored, x::Real)
    try
        # Check support first for consistency with Distributions.jl
        if !insupport(d, x)
            return -Inf
        end

        pdf_val = pdf(d, x)
        if pdf_val <= 0.0
            return -Inf
        end
        return log(pdf_val)
    catch e
        if isa(e, DomainError) || isa(e, BoundsError) || isa(e, ArgumentError)
            return -Inf
        else
            rethrow(e)
        end
    end
end

#### Vectorised PDF optimization

"""
    _collect_unique_boundaries(d::IntervalCensored, x::AbstractArray)

Collect all unique interval boundaries needed for vectorised PDF computation.

Returns a sorted vector of unique boundaries with appropriate type promotion.
"""
function _collect_unique_boundaries(d::IntervalCensored, x::AbstractArray{<:Real})
    # Determine promoted type for type stability
    T = promote_type(eltype(x), eltype(d.boundaries))
    boundaries = T[]

    # Collect all unique boundaries needed
    if is_regular_intervals(d)
        interval = interval_width(d)
        for xi in x
            lower = floor_to_interval(xi, interval)
            upper = lower + interval
            push!(boundaries, lower, upper)
        end
    else
        # For arbitrary intervals, collect all boundaries that could be needed
        for xi in x
            lower, upper = get_interval_bounds(d, xi)
            if !isnan(lower) && !isnan(upper)
                push!(boundaries, lower, upper)
            end
        end
    end

    # Return sorted unique boundaries
    return sort!(unique!(boundaries))
end

"""
    _compute_pdfs_with_cache(d::IntervalCensored, x::AbstractArray, cdf_lookup::Dict)

Compute PDFs efficiently using cached CDF values.

Uses the same boundary case handling as the scalar method.
"""
function _compute_pdfs_with_cache(d::IntervalCensored, x::AbstractArray{<:Real}, cdf_lookup::Dict)
    T = promote_type(eltype(x), eltype(d))
    pdfs = Vector{T}(undef, length(x))

    # Get distribution bounds once for boundary case handling
    dist_min = minimum(get_dist(d))
    dist_max = maximum(get_dist(d))

    for (i, xi) in enumerate(x)
        lower, upper = get_interval_bounds(d, xi)

        if isnan(lower) || isnan(upper)
            pdfs[i] = zero(T)
            continue
        end

        # Handle boundary cases for distributions with bounded support
        # For lower bound at or below distribution minimum, CDF is 0
        cdf_lower = lower <= dist_min ? zero(T) : cdf_lookup[lower]

        # For upper bound at or above distribution maximum, CDF is 1
        cdf_upper = upper >= dist_max ? one(T) : cdf_lookup[upper]

        pdfs[i] = cdf_upper - cdf_lower
    end

    return pdfs
end

@doc "

Compute probability masses for an array of values using optimised vectorisation.

This method collects unique interval boundaries, computes CDFs once, then uses
cached values for efficient PDF computation across the array.

See also: [`pdf`](@ref), [`logpdf`](@ref)
"
function pdf(d::IntervalCensored, x::AbstractArray{<:Real})
    # Collect all unique boundaries needed
    boundaries = _collect_unique_boundaries(d, x)

    # Compute CDFs once for all unique boundaries
    T = promote_type(eltype(x), eltype(d.boundaries))
    cdf_lookup = Dict{T, T}()

    for boundary in boundaries
        cdf_lookup[boundary] = cdf(get_dist(d), boundary)
    end

    # Use cached values to compute PDFs efficiently
    return _compute_pdfs_with_cache(d, x, cdf_lookup)
end

@doc "

Compute log probability masses for an array of values using optimised PDF computation.

See also: [`pdf`](@ref), [`logpdf`](@ref)
"
function logpdf(d::IntervalCensored, x::AbstractArray{<:Real})
    # Use vectorised PDF computation then handle logs with proper error handling
    pdf_vals = pdf(d, x)

    T = promote_type(eltype(x), eltype(d))
    logpdfs = Vector{T}(undef, length(x))

    for (i, (xi, pdf_val)) in enumerate(zip(x, pdf_vals))
        try
            # Check support first for consistency with Distributions.jl
            if !insupport(d, xi)
                logpdfs[i] = T(-Inf)
            elseif pdf_val <= 0.0
                logpdfs[i] = T(-Inf)
            else
                logpdfs[i] = log(pdf_val)
            end
        catch e
            if isa(e, DomainError) || isa(e, BoundsError) || isa(e, ArgumentError)
                logpdfs[i] = T(-Inf)
            else
                rethrow(e)
            end
        end
    end

    return logpdfs
end

# Resolve ambiguity with Distributions.jl for 0-dimensional arrays
function pdf(d::IntervalCensored, x::AbstractArray{<:Real, 0})
    return pdf(d, x[])  # Extract scalar and delegate to scalar method
end

function logpdf(d::IntervalCensored, x::AbstractArray{<:Real, 0})
    return logpdf(d, x[])  # Extract scalar and delegate to scalar method
end

# Internal function for efficient cdf/logcdf computation
function _interval_cdf(d::IntervalCensored, x::Real, f::Function)
    try
        # Handle edge cases first
        if x < minimum(get_dist(d))
            return f === logcdf ? -Inf : 0.0
        elseif x >= maximum(get_dist(d))
            return f === logcdf ? 0.0 : 1.0
        end

        if is_regular_intervals(d)
            # For regular intervals, use floor behavior from Discretised
            discretised_x = floor_to_interval(x, interval_width(d))
            return f(get_dist(d), discretised_x)
        else
            # For arbitrary intervals, use the lower bound of the containing
            # interval
            idx = find_interval_index(x, d.boundaries)
            if idx == 0
                return f === logcdf ? -Inf : 0.0
            elseif idx >= length(d.boundaries)
                return f(get_dist(d), d.boundaries[end])
            else
                return f(get_dist(d), d.boundaries[idx])
            end
        end
    catch e
        if isa(e, DomainError) || isa(e, BoundsError) || isa(e, ArgumentError)
            return f === logcdf ? -Inf : 0.0
        else
            rethrow(e)
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
