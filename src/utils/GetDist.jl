@doc raw"""
    get_dist(d)

Extract the underlying distribution from a wrapped distribution type.

This utility function provides a consistent interface for extracting the core
distribution from various wrapper types in the CensoredDistributions ecosystem.
For unwrapped distributions, it returns the distribution unchanged.

# Arguments
- `d`: A distribution or wrapped distribution

# Returns
The underlying distribution. For base distributions, returns `d` unchanged.

# Examples
```@example
using CensoredDistributions, Distributions

# Base distribution - returns unchanged
d1 = Normal(0, 1)
get_dist(d1) == d1  # true

# Primary censored distribution
delay = LogNormal(1.5, 0.75)
window = Uniform(0, 1)
pc = primary_censored(delay, window)
get_dist(pc) == delay  # true

# Interval censored distribution
continuous = Normal(5, 2)
ic = interval_censored(continuous, 1.0)
get_dist(ic) == continuous  # true

# Weighted distribution
wd = weight(Normal(0, 1), 2.5)
get_dist(wd) isa Normal  # true

# Product distribution (returns vector of component distributions)
pd = product_distribution([Normal(0, 1), Exponential(1)])
components = get_dist(pd)
length(components) == 2  # true
```
"""
function get_dist(d)
    return d
end

@doc raw"""
    get_dist(d::PrimaryCensored)

Extract the delay distribution from a primary censored distribution.

Returns the underlying delay distribution (the distribution of times from
primary event to observation).
"""
function get_dist(d::PrimaryCensored)
    return d.dist
end

@doc raw"""
    get_dist(d::IntervalCensored)

Extract the underlying continuous distribution from an interval censored
distribution.

Returns the continuous distribution that was discretised into intervals.
"""
function get_dist(d::IntervalCensored)
    return d.dist
end

@doc raw"""
    get_dist(d::Weighted)

Extract the underlying distribution from a weighted distribution.

Returns the base distribution before weighting was applied.
"""
function get_dist(d::Weighted)
    return d.dist
end

@doc raw"""
    get_dist(d::Product)

Extract the component distributions from a product distribution.

Returns a vector containing the underlying distributions for each component
of the product distribution. This is useful when working with vectorised
distributions or multiple independent observations.

# Note
This method has different behaviour from other `get_dist` methods as it
returns a vector of distributions rather than a single distribution.
"""
function get_dist(d::Product)
    return d.v
end
