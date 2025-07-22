@doc raw"
Implements a discretised distribution that rounds continuous values to intervals.

Takes a continuous distribution and discretises it by rounding values down to 
multiples of the interval width. This is commonly used for secondary censoring 
where observations are rounded to the nearest time interval (e.g., daily reporting).

# Arguments
- `dist`: The underlying continuous distribution
- `interval`: The width of discretisation intervals

# Examples
```@example
using CensoredDistributions, Distributions

# Discretise a normal distribution to daily intervals
d = discretise(Normal(5, 2), 1.0)

# Sample from discretised distribution
samples = rand(d, 1000)

# All samples will be multiples of 1.0
all(s % 1.0 == 0.0 for s in samples)
```
"
struct Discretised{D <: UnivariateDistribution, T <: Real} <:
       Distributions.UnivariateDistribution{Distributions.ValueSupport}
    "The underlying continuous distribution"
    dist::D
    "The discretisation interval width"
    interval::T
    
    function Discretised(dist::D, interval::T) where {D, T}
        interval > 0 || throw(ArgumentError("Interval must be positive"))
        new{D, T}(dist, interval)
    end
end

@doc raw"
Construct a discretised distribution.

# Arguments
- `dist`: The underlying continuous distribution
- `interval`: The width of discretisation intervals

# Returns
A `Discretised` object
"
function discretise(dist::UnivariateDistribution, interval::Real)
    return Discretised(dist, interval)
end

# American spelling alias
discretize(dist::UnivariateDistribution, interval::Real) = discretise(dist, interval)

function Distributions.params(d::Discretised)
    return (params(d.dist)..., d.interval)
end

Base.eltype(::Type{<:Discretised{D, T}}) where {D, T} = promote_type(eltype(D), T)

function Distributions.minimum(d::Discretised)
    cont_min = minimum(d.dist)
    return floor_to_interval(cont_min, d.interval)
end

function Distributions.maximum(d::Discretised)
    cont_max = maximum(d.dist)
    return floor_to_interval(cont_max, d.interval)
end

function Distributions.insupport(d::Discretised, x::Real)
    # Value must be a multiple of the interval and in support of underlying dist
    return x % d.interval ≈ 0.0 && insupport(d.dist, x)
end

#### Probability functions

function Distributions.pdf(d::Discretised, x::Real)
    if !insupport(d, x)
        return 0.0
    end
    # Probability mass for interval [x, x + interval)
    return cdf(d.dist, x + d.interval) - cdf(d.dist, x)
end

function Distributions.logpdf(d::Discretised, x::Real)
    return log(pdf(d, x))
end

function Distributions.cdf(d::Discretised, x::Real)
    # CDF at discretised point x
    discretised_x = floor_to_interval(x, d.interval)
    return cdf(d.dist, discretised_x)
end

function Distributions.logcdf(d::Discretised, x::Real)
    return log(cdf(d, x))
end

function Distributions.ccdf(d::Discretised, x::Real)
    return 1 - cdf(d, x)
end

function Distributions.logccdf(d::Discretised, x::Real)
    return log(ccdf(d, x))
end

#### Sampling

function Base.rand(rng::AbstractRNG, d::Discretised)
    # Sample from continuous distribution and discretise
    x = rand(rng, d.dist)
    return floor_to_interval(x, d.interval)
end

function floor_to_interval(x::Real, interval::Real)
    return floor(x / interval) * interval
end
