@doc raw"
Interval-censored distribution where continuous values are observed only within intervals.

Supports both:
- Regular intervals: fixed-width intervals (e.g., daily reporting)
- Arbitrary intervals: custom interval boundaries

# Fields
- `dist`: The underlying continuous distribution
- `intervals`: Either a scalar (regular interval width) or vector (arbitrary boundaries)
"
struct IntervalCensored{D <: UnivariateDistribution, T} <:
       Distributions.UnivariateDistribution{Distributions.ValueSupport}
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
        issorted(boundaries) || throw(ArgumentError("Interval boundaries must be sorted"))
        all(diff(boundaries) .> 0) ||
            throw(ArgumentError("Interval boundaries must be strictly increasing"))

        new{D, typeof(boundaries)}(dist, boundaries)
    end
end

@doc raw"
Construct an interval-censored distribution with regular intervals.

Creates a distribution where observations are censored to regular intervals of width `interval`,
starting from 0. For example, with `interval=1`, observations fall into [0,1), [1,2), [2,3), etc.

# Arguments
- `dist`: The underlying continuous distribution
- `interval`: The width of regular intervals

# Returns
An `IntervalCensored` distribution

# Examples
```julia
using CensoredDistributions, Distributions

# Daily reporting intervals
d = interval_censored(Normal(5, 2), 1.0)
rand(d, 10)  # Returns values like 4.0, 5.0, 6.0, etc.

# Weekly intervals
d_weekly = interval_censored(Exponential(3), 7.0)
```
"
function interval_censored(dist::UnivariateDistribution, interval::Real)
    return IntervalCensored(dist, interval)
end

@doc raw"
Construct an interval-censored distribution with arbitrary intervals.

Creates a distribution where observations are censored to specified intervals defined by
the boundaries. For example, with `boundaries=[0, 2, 5, 10]`, observations fall into
[0,2), [2,5), or [5,10).

# Arguments
- `dist`: The underlying continuous distribution
- `boundaries`: Vector of interval boundaries (must be sorted and strictly increasing)

# Returns
An `IntervalCensored` distribution

# Examples
```julia
using CensoredDistributions, Distributions

# Age groups: 0-18, 18-65, 65+
age_dist = interval_censored(Normal(40, 20), [0, 18, 65, 100])

# Custom measurement bins
measure_dist = interval_censored(Gamma(2, 3), [0.0, 0.5, 1.0, 2.5, 5.0, Inf])
```
"
function interval_censored(dist::UnivariateDistribution, boundaries::AbstractVector{<:Real})
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
function Distributions.params(d::IntervalCensored)
    if is_regular_intervals(d)
        return (params(d.dist)..., d.boundaries)
    else
        return (params(d.dist)..., d.boundaries)
    end
end

Base.eltype(::Type{<:IntervalCensored{D, T}}) where {D, T} = eltype(D)

function Distributions.minimum(d::IntervalCensored)
    cont_min = minimum(d.dist)
    if is_regular_intervals(d)
        return floor_to_interval(cont_min, interval_width(d))
    else
        # Find first interval that could contain values
        idx = find_interval_index(cont_min, d.boundaries)
        return idx > 0 ? d.boundaries[idx] : d.boundaries[1]
    end
end

function Distributions.maximum(d::IntervalCensored)
    cont_max = maximum(d.dist)
    if is_regular_intervals(d)
        return floor_to_interval(cont_max, interval_width(d))
    else
        # Find last interval that could contain values
        idx = find_interval_index(cont_max, d.boundaries)
        return idx < length(d.boundaries) ? d.boundaries[idx] : d.boundaries[end - 1]
    end
end

function Distributions.insupport(d::IntervalCensored, x::Real)
    if is_regular_intervals(d)
        # For regular intervals, x must be a multiple of the interval
        return x % interval_width(d) ≈ 0.0 && insupport(d.dist, x)
    else
        # For arbitrary intervals, x must be an interval boundary
        return x in d.boundaries && insupport(d.dist, x)
    end
end

#### Probability functions

function Distributions.pdf(d::IntervalCensored, x::Real)
    lower, upper = get_interval_bounds(d, x)
    if isnan(lower) || isnan(upper)
        return 0.0
    end
    return cdf(d.dist, upper) - cdf(d.dist, lower)
end

function Distributions.logpdf(d::IntervalCensored, x::Real)
    return log(pdf(d, x))
end

function Distributions.cdf(d::IntervalCensored, x::Real)
    if is_regular_intervals(d)
        # For regular intervals, use floor behavior from Discretised
        discretised_x = floor_to_interval(x, interval_width(d))
        return cdf(d.dist, discretised_x)
    else
        # For arbitrary intervals, use the lower bound of the containing interval
        idx = find_interval_index(x, d.boundaries)
        if idx == 0
            return 0.0
        elseif idx >= length(d.boundaries)
            return cdf(d.dist, d.boundaries[end])
        else
            return cdf(d.dist, d.boundaries[idx])
        end
    end
end

function Distributions.logcdf(d::IntervalCensored, x::Real)
    return log(cdf(d, x))
end

function Distributions.ccdf(d::IntervalCensored, x::Real)
    return 1 - cdf(d, x)
end

function Distributions.logccdf(d::IntervalCensored, x::Real)
    return log(ccdf(d, x))
end

#### Sampling

function Base.rand(rng::AbstractRNG, d::IntervalCensored)
    # Sample once from the underlying distribution
    x = rand(rng, d.dist)

    if is_regular_intervals(d)
        # Discretise to regular intervals
        return floor_to_interval(x, interval_width(d))
    else
        # Find which arbitrary interval contains x
        idx = find_interval_index(x, d.boundaries)
        if idx == 0 || idx >= length(d.boundaries)
            # Outside intervals - this shouldn't happen if dist is properly bounded
            # Return closest boundary
            return idx == 0 ? d.boundaries[1] : d.boundaries[end]
        else
            return d.boundaries[idx]
        end
    end
end

# Sampler method for efficient sampling
Distributions.sampler(d::IntervalCensored) = d
