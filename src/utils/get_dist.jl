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

# Truncated distribution
trunc_dist = truncated(Normal(0, 1), -2, 2)
get_dist(trunc_dist) isa Normal  # true

# Censored distribution (from Distributions.jl)
censored_dist = censored(Normal(0, 1), -2, 2)
get_dist(censored_dist) isa Normal  # true

# Double interval censored distributions (all forms)
# Basic case - returns PrimaryCensored
basic_dic = double_interval_censored(LogNormal(1.5, 0.75))
get_dist(basic_dic) isa LogNormal  # true

# With truncation - returns Truncated{PrimaryCensored}
trunc_dic = double_interval_censored(LogNormal(1.5, 0.75); upper=10)
get_dist(trunc_dic) isa CensoredDistributions.PrimaryCensored  # true

# With interval censoring - returns IntervalCensored{PrimaryCensored}
interval_dic = double_interval_censored(LogNormal(1.5, 0.75); interval=1)
get_dist(interval_dic) isa CensoredDistributions.PrimaryCensored  # true

# Full double censoring - returns IntervalCensored{Truncated{PrimaryCensored}}
full_dic = double_interval_censored(LogNormal(1.5, 0.75); upper=10, interval=1)
get_dist(full_dic) isa Truncated  # true

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
    get_dist(d::Truncated)

Extract the untruncated distribution from a truncated distribution.

Returns the underlying continuous distribution before truncation bounds
were applied.

# Examples
```@example
using Distributions

# Extract Normal from truncated Normal
base = Normal(0, 1)
trunc_dist = truncated(base, -2, 2)
get_dist(trunc_dist) === base  # true

# Works with any truncated distribution
gamma_base = Gamma(2, 1)
trunc_gamma = truncated(gamma_base, 0.1, 5.0)
get_dist(trunc_gamma) === gamma_base  # true
```
"""
function get_dist(d::Truncated)
    return d.untruncated
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

@doc raw"""
    get_dist(d::Distributions.Censored)

Extract the underlying distribution from a censored distribution.

Returns the uncensored distribution before censoring bounds were applied.
This method supports the `Censored` type from Distributions.jl created by
the `censored()` function.

# Examples
```@example
using Distributions

# Extract Normal from censored Normal
base = Normal(0, 1)
censored_dist = censored(base, -2, 2)
get_dist(censored_dist) === base  # true

# Works with any censored distribution
gamma_base = Gamma(2, 1)
censored_gamma = censored(gamma_base, 0.1, 5.0)
get_dist(censored_gamma) === gamma_base  # true
```
"""
function get_dist(d::Censored)
    return d.uncensored
end

@doc raw"""
    get_dist_recursive(d)

Recursively extract the underlying distribution from nested wrapper types.

This function keeps applying `get_dist` until it reaches a distribution that
doesn't have a specialised method, meaning no further unwrapping is possible.
This is useful for deeply nested distributions like
`IntervalCensored{Truncated{PrimaryCensored{...}}}`.

# Arguments
- `d`: A distribution or nested wrapped distribution

# Returns
The deeply underlying distribution after all unwrapping is complete.

# Examples
```@example
using CensoredDistributions, Distributions

# Single wrapper - same as get_dist
delay = LogNormal(1.5, 0.75)
window = Uniform(0, 1)
pc = primary_censored(delay, window)
get_dist_recursive(pc) == delay  # true

# Nested wrappers
continuous = Normal(5, 2)
ic = interval_censored(continuous, 1.0)
weighted = weight(ic, 2.0)
get_dist_recursive(weighted) == continuous  # true

# Truncated nested wrappers
base = Normal(0, 1)
trunc_dist = truncated(base, -2, 2)
weighted_trunc = weight(trunc_dist, 2.0)
get_dist_recursive(weighted_trunc) == base  # true

# Censored nested wrappers
censored_dist = censored(Normal(0, 1), -2, 2)
weighted_censored = weight(censored_dist, 2.0)
get_dist_recursive(weighted_censored) == base  # true

# Double interval censored distributions (fully recursive)
delay = LogNormal(1.5, 0.75)
full_double = double_interval_censored(delay; upper=10, interval=1)
get_dist_recursive(full_double) == delay  # true

# Weighted double interval censored
weighted_double = weight(full_double, 2.0)
get_dist_recursive(weighted_double) == delay  # true

# Base distribution - returns unchanged
d = Normal(0, 1)
get_dist_recursive(d) == d  # true
```

# Note
For `Product` distributions, this function applies recursive extraction
to each component, potentially returning mixed types of underlying
distributions.
"""
function get_dist_recursive(d)
    next = get_dist(d)
    # If get_dist returns the same object, we've reached the end
    if next === d
        return d
    end
    # For Product distributions, recursively unwrap components
    if next isa AbstractVector
        return [get_dist_recursive(component) for component in next]
    end
    # Otherwise, recursively unwrap the next level
    return get_dist_recursive(next)
end
