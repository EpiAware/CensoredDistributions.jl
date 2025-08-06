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
```@example
using CensoredDistributions, Distributions

# Daily reporting intervals
d = interval_censored(Normal(5, 2), 1.0)

# Evaluate distribution functions
pdf_at_5 = pdf(d, 5.0)      # probability mass at interval containing 5
cdf_at_7 = cdf(d, 7.0)      # P(X ≤ 7) accounting for interval censoring
ccdf_at_3 = ccdf(d, 3.0)    # survival function

# Compute quantiles and statistics
q25 = quantile(d, 0.25)     # 25th percentile (interval boundary)
median_val = quantile(d, 0.5)  # median
q90 = quantile(d, 0.9)      # 90th percentile
samples = rand(d, 100)      # random samples (interval boundaries)

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
```@example
using CensoredDistributions, Distributions

# Age groups: 0-18, 18-65, 65+
age_dist = interval_censored(Normal(40, 20), [0, 18, 65, 100])

# Evaluate distribution functions for arbitrary intervals
pdf_at_25 = pdf(age_dist, 25.0)    # probability mass in interval containing 25
cdf_at_50 = cdf(age_dist, 50.0)    # P(X ≤ 50) with interval censoring
q75 = quantile(age_dist, 0.75)     # 75th percentile (boundary value)
median_age = quantile(age_dist, 0.5)  # median age group
samples = rand(age_dist, 50)       # random age group boundaries

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

# Find the appropriate boundary for quantile purposes (left boundary of containing interval)
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
function Distributions.params(d::IntervalCensored)
    if is_regular_intervals(d)
        return (params(get_dist(d))..., d.boundaries)
    else
        return (params(get_dist(d))..., d.boundaries)
    end
end

Base.eltype(::Type{<:IntervalCensored{D, T}}) where {D, T} = eltype(D)

function Distributions.minimum(d::IntervalCensored)
    cont_min = minimum(get_dist(d))
    if is_regular_intervals(d)
        return floor_to_interval(cont_min, interval_width(d))
    else
        # Find first interval that could contain values
        idx = find_interval_index(cont_min, d.boundaries)
        return idx > 0 ? d.boundaries[idx] : d.boundaries[1]
    end
end

function Distributions.maximum(d::IntervalCensored)
    cont_max = maximum(get_dist(d))
    if is_regular_intervals(d)
        return floor_to_interval(cont_max, interval_width(d))
    else
        # Find last interval that could contain values
        idx = find_interval_index(cont_max, d.boundaries)
        return idx < length(d.boundaries) ? d.boundaries[idx] : d.boundaries[end - 1]
    end
end

function Distributions.insupport(d::IntervalCensored, x::Real)
    # For interval-censored distributions, support is continuous within the underlying distribution
    # The PDF is non-zero for any x where the underlying distribution has support
    return insupport(get_dist(d), x)
end

#### Probability functions

function Distributions.pdf(d::IntervalCensored, x::Real)
    lower, upper = get_interval_bounds(d, x)
    if isnan(lower) || isnan(upper)
        return 0.0
    end
    return cdf(get_dist(d), upper) - cdf(get_dist(d), lower)
end

function Distributions.logpdf(d::IntervalCensored, x::Real)
    # Check support first for type stability
    if !insupport(d, x)
        return -Inf
    end

    lower, upper = get_interval_bounds(d, x)
    if isnan(lower) || isnan(upper)
        return -Inf
    end

    # Compute log(P(lower < X <= upper)) = log(F(upper) - F(lower))
    # Use numerical stability approach that's AD-friendly
    cdf_upper = cdf(get_dist(d), upper)
    cdf_lower = cdf(get_dist(d), lower)
    pdf_mass = cdf_upper - cdf_lower

    # Handle edge cases
    if pdf_mass <= 0.0
        return -Inf
    end

    return log(pdf_mass)
end

# Internal function for efficient cdf/logcdf computation
function _interval_cdf(d::IntervalCensored, x::Real, f::Function)
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
        # For arbitrary intervals, use the lower bound of the containing interval
        idx = find_interval_index(x, d.boundaries)
        if idx == 0
            return f === logcdf ? -Inf : 0.0
        elseif idx >= length(d.boundaries)
            return f(get_dist(d), d.boundaries[end])
        else
            return f(get_dist(d), d.boundaries[idx])
        end
    end
end

function Distributions.cdf(d::IntervalCensored, x::Real)
    return _interval_cdf(d, x, cdf)
end

function Distributions.logcdf(d::IntervalCensored, x::Real)
    return _interval_cdf(d, x, logcdf)
end

function Distributions.ccdf(d::IntervalCensored, x::Real)
    return 1 - cdf(d, x)
end

function Distributions.logccdf(d::IntervalCensored, x::Real)
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
            # Outside intervals - this shouldn't happen if dist is properly bounded
            # Return closest boundary
            return idx == 0 ? d.boundaries[1] : d.boundaries[end]
        else
            return d.boundaries[idx]
        end
    end
end

#### Quantile function

@doc raw"
Compute the quantile of an interval-censored distribution.

For interval-censored distributions, the quantile function returns the value `x` such that
`P(X ≤ x) = p`, where `X` follows the interval-censored distribution. Due to the discrete
nature of interval censoring, this is computed numerically by solving the equation
`cdf(d, x) = p` using optimization.

The returned quantile respects the interval structure:
- For regular intervals: quantiles are multiples of the interval width
- For arbitrary intervals: quantiles correspond to interval boundary values

# Arguments
- `d::IntervalCensored`: The interval-censored distribution
- `p::Real`: The probability level (must be in [0, 1])

# Returns
The quantile value `x` such that `P(X ≤ x) = p`

# Throws
- `ArgumentError`: If `p` is not in [0, 1]
- `ErrorException`: If the optimization fails to converge

# Examples
```@example
using CensoredDistributions, Distributions

# Regular intervals with daily censoring
d = interval_censored(Normal(5, 2), 1.0)
q25 = quantile(d, 0.25)  # 25th percentile
q50 = quantile(d, 0.50)  # Median
q75 = quantile(d, 0.75)  # 75th percentile

# Arbitrary intervals (age groups)
age_dist = interval_censored(Normal(40, 20), [0, 18, 65, 100])
median_age = quantile(age_dist, 0.5)  # Median age group boundary
```

# Implementation Notes
Uses numerical optimization with the Nelder-Mead algorithm to solve `cdf(d, x) - p = 0`.
The initial guess is based on the quantile of the underlying continuous distribution.
For efficiency, the search is constrained to the relevant interval boundaries.
"
function Distributions.quantile(d::IntervalCensored, p::Real)
    if p < 0.0 || p > 1.0
        throw(ArgumentError("p must be in [0, 1]"))
    end

    # Handle boundary cases
    if p == 0.0
        return minimum(d)
    elseif p == 1.0
        return maximum(d)
    end

    # Unified approach: use optimization with appropriate interval snapping
    # Objective function: minimize (cdf(d, snapped_x) - p)²
    objective = function (x, _)
        x_val = x[1]
        # Snap to appropriate interval boundary based on interval type
        interval_x = if is_regular_intervals(d)
            floor_to_interval(x_val, interval_width(d))
        else
            # For arbitrary intervals, find the appropriate boundary
            find_interval_boundary(x_val, d.boundaries)
        end
        cdf_val = cdf(d, interval_x)
        return (cdf_val - p)^2
    end

    # Initial guess based on quantile of underlying distribution
    underlying_quantile = quantile(get_dist(d), p)
    x0 = [underlying_quantile]

    # Set up optimization problem
    optfun = OptimizationFunction(objective)
    prob = OptimizationProblem(optfun, x0, nothing)

    # Solve using NelderMead (derivative-free, robust for discrete problems)
    sol = solve(prob, NelderMead(); reltol = 1e-8, abstol = 1e-8, maxiters = 10000)

    # Check convergence and return result
    if sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.Default
        result = sol.u[1]
        # Apply same boundary snapping as in objective function
        return if is_regular_intervals(d)
            floor_to_interval(result, interval_width(d))
        else
            find_interval_boundary(result, d.boundaries)
        end
    else
        error("Quantile optimization failed to converge for p = $p")
    end
end

# Sampler method for efficient sampling
Distributions.sampler(d::IntervalCensored) = d
