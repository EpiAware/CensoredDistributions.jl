@doc raw"
Implement a censoring function.

Takes a UnivariateDistribution and returns a WithinIntervalCensored object but depends only on the cdf method.

# Arguments
- `dist` the truncated primarycensored distribution
- `lower` the lower bound of the interval
- `upper` the upper bound of the interval

# Returns
- WithinIntervalCensored object

"
struct WithinIntervalCensored{D <: UnivariateDistribution, L <: Real, U <: Real} <:
       Distributions.UnivariateDistribution{Distributions.ValueSupport}
    dist::D
    lower::L
    upper::U
end

@doc raw"
Generic wrapper for within window censored an object

# Contructors

"
function within_interval_censored(dist::UnivariateDistribution, lower::Real, upper::Real)
    WithinIntervalCensored(dist, lower, upper)
end

@doc raw"
Returning a within interval censored pmf

# Arguments
- d an WithinIntervalCensoredDist object

# Returns
- discretized probability mass function

# Examples

```@example
using CensoredDistributions, Distributions
d = truncated(Normal(5,2), 0, 5)
trunc_d = within_interval_censored(d, 2, 4)
pdf(trunc_d)
```
"

function Distributions.pdf(d::WithinIntervalCensored)
    return cdf(d.dist, d.upper) - cdf(d.dist, d.lower)
end

@doc raw"
Returning a within interval censored logpdf

# Arguments
- d an WithinIntervalCensoredDist object

# Returns
- discretized log probability mass function

# Examples

```@example
using CensoredDistributions, Distributions
d = truncated(Normal(5,2), 0, 5)
trunc_d = within_interval_censored(d, 2, 4)
logpdf(trunc_d)
rand(trunc_d)
```
"
function Distributions.logpdf(d::WithinIntervalCensored)
    return log(pdf(d))
end

@doc raw"
Generate a random sample from a within-interval censored distribution.

# Arguments
- `rng`: Random number generator
- `d`: WithinIntervalCensored distribution object

# Returns
A random sample from the within-interval censored distribution

# Examples
```@example
using CensoredDistributions, Distributions
d = truncated(Normal(5,2), 0, 5)
trunc_d = within_interval_censored(d, 2, 4)
rand(trunc_d)
```
"
function Base.rand(
        rng::Random.AbstractRNG, d::WithinIntervalCensored)
    # Sample from the underlying distribution truncated to the interval
    truncated_dist = truncated(d.dist; lower = d.lower, upper = d.upper)
    return d.lower
end

# Sampler method for efficient sampling
Distributions.sampler(d::WithinIntervalCensored) = d
