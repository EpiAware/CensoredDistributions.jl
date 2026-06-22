@doc "

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
get_dist(d1)

# Primary censored distribution
delay = LogNormal(1.5, 0.75)
window = Uniform(0, 1)
pc = primary_censored(delay, window)
get_dist(pc)

# Interval censored distribution
continuous = Normal(5, 2)
ic = interval_censored(continuous, 1.0)
get_dist(ic)
```
"
function get_dist(d)
    return d
end

@doc "

Extract the delay distribution from a primary censored distribution.

Returns the underlying delay distribution (the distribution of times from
primary event to observation).
"
function get_dist(d::PrimaryCensored)
    return d.dist
end

@doc "

Extract the primary event time distribution from a primary censored
distribution.

Returns the distribution of primary event times within the censoring window.
The conditional delay given a realised primary `p` is then
`logpdf(get_dist(d), observed - p)`.

# Arguments
- `d`: A primary censored distribution.

# Examples
```@example
using CensoredDistributions, Distributions

d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
get_primary_event(d)
```
"
function get_primary_event(d::PrimaryCensored)
    return d.primary_event
end

@doc "

Extract the primary event time distribution through a secondary censoring layer.

A [`double_interval_censored`](@ref) delay wraps its primary-censored node in an
[`IntervalCensored`](@ref) (and optionally a `Truncated`), so
`latent(double_interval_censored(...))` must reach the primary event THROUGH
those wrappers. This recurses into the wrapped node to surface that primary.
"
get_primary_event(d::IntervalCensored) = get_primary_event(d.dist)

@doc "

Extract the primary event time distribution through a truncation layer.

Recurses into the untruncated node so a truncated primary-censored node still
surfaces its primary event.
"
get_primary_event(d::Truncated) = get_primary_event(d.untruncated)

@doc "

Extract the BARE continuous delay from a latent node.

The latent representation SAMPLES the primary event, so its conditional scores
the bare continuous delay, every censoring layer (primary, truncation, secondary
interval) stripped. Recurses with [`get_dist_recursive`](@ref) so a latent
`double_interval_censored` node surfaces the same continuous core as a latent
`primary_censored` node, and avoids reapplying the secondary interval, which
would double-count the within-window uncertainty already represented by the
sampled primary.
"
get_dist(d::Latent) = get_dist_recursive(d.dist)

@doc "

Extract the primary event time distribution from a latent primary-censored node.

Delegates to the wrapped node.
"
get_primary_event(d::Latent) = get_primary_event(d.dist)

@doc "

Extract the delay distribution from a primary-conditional distribution.

Delegates to the wrapped node, so the conditional reuses the same delay as the
marginal and latent forms.
"
get_dist(d::PrimaryConditional) = get_dist(d.dist)

@doc "

Extract the underlying continuous distribution from an interval censored
distribution.

Returns the continuous distribution that was discretised into intervals.
"
function get_dist(d::IntervalCensored)
    return d.dist
end

@doc "

Extract the underlying distribution from a weighted distribution.

Returns the base distribution before weighting was applied.
"
function get_dist(d::Weighted)
    return d.dist
end

@doc "

Extract the component distributions from a convolved distribution.

Returns a vector of the independent component distributions being summed.
Like the `Product` method, this returns a vector rather than a single
distribution.
"
function get_dist(d::Convolved)
    return collect(d.components)
end

@doc "

Extract the untruncated distribution from a truncated distribution.

Returns the underlying continuous distribution before truncation bounds
were applied.

# Examples
```@example
using Distributions

# Extract Normal from truncated Normal
base = Normal(0, 1)
trunc_dist = truncated(base, -2, 2)
get_dist(trunc_dist)

# Works with any truncated distribution
gamma_base = Gamma(2, 1)
trunc_gamma = truncated(gamma_base, 0.1, 5.0)
get_dist(trunc_gamma)
```
"
function get_dist(d::Truncated)
    return d.untruncated
end

@doc "

Extract the component distributions from a product distribution.

Returns a vector containing the underlying distributions for each component
of the product distribution. This is useful when working with vectorised
distributions or multiple independent observations.

# Note
This method has different behaviour from other `get_dist` methods as it
returns a vector of distributions rather than a single distribution.
"
function get_dist(d::Product)
    return d.v
end

@doc "

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
get_dist(censored_dist)

# Works with any censored distribution
gamma_base = Gamma(2, 1)
censored_gamma = censored(gamma_base, 0.1, 5.0)
get_dist(censored_gamma)
```
"
function get_dist(d::Censored)
    return d.uncensored
end

@doc "

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
get_dist_recursive(pc)

# Nested wrappers
continuous = Normal(5, 2)
ic = interval_censored(continuous, 1.0)
weighted = weight(ic, 2.0)
get_dist_recursive(weighted)

# Double interval censored distributions (fully recursive)
delay = LogNormal(1.5, 0.75)
full_double = double_interval_censored(delay; upper=10, interval=1)
get_dist_recursive(full_double)

# Base distribution - returns unchanged
d = Normal(0, 1)
get_dist_recursive(d)
```

# Note
For `Product` distributions, this function applies recursive extraction
to each component, potentially returning mixed types of underlying
distributions.
"
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
