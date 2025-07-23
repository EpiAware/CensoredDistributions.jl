@doc raw"
Create a primary event censored distribution.

Models a process where a primary event occurs within a censoring window, followed by a delay.
The primary event time is not observed directly but is known to fall within the censoring
distribution's support. The observed time is the sum of the primary event time and the delay.

This is useful for modeling:
- Infection-to-symptom onset times when infection time is uncertain
- Exposure-to-outcome delays with uncertain exposure timing
- Any process where the initiating event time has uncertainty

# Arguments
- `dist`: Distribution of the delay from primary event to observation
- `primary_event`: Distribution of the primary event time (typically Uniform(0, window))

# Returns
A `PrimaryCensored` distribution representing the convolution of the censoring and delay distributions.

# Examples
```julia
using CensoredDistributions, Distributions

# Incubation period (delay) with uncertain infection time (primary event)
incubation = LogNormal(1.5, 0.75)  # Delay distribution
infection_window = Uniform(0, 1)    # Daily infection window
d = primary_censored(incubation, infection_window)

# Sample observed symptom onset times
onsets = rand(d, 1000)

# Calculate CDFs
x = 0:0.1:10
cdf_original = cdf.(incubation, x)
cdf_censored = cdf.(d, x)
```
"
function primary_censored(
        dist::UnivariateDistribution, primary_event::UnivariateDistribution)
    return PrimaryCensored(dist, primary_event)
end

@doc raw"
Primary event censored distribution.

Represents the distribution of observed delays when the primary event time is subject to censoring.

# Fields
- `dist`: Distribution of the delay from primary event to observation
- `primary_event`: Distribution of the primary event time
"
struct PrimaryCensored{D1 <: UnivariateDistribution, D2 <: UnivariateDistribution} <:
       Distributions.UnivariateDistribution{Distributions.ValueSupport}
    "The delay distribution from primary event to observation."
    dist::D1
    "The primary event time distribution."
    primary_event::D2

    function PrimaryCensored(dist::D1, primary_event::D2) where {D1, D2}
        minimum(dist) == 0 ||
            throw(ArgumentError("Delay distribution must have minimum of zero"))
        new{D1, D2}(dist, primary_event)
    end
end

function Distributions.params(d::PrimaryCensored)
    d0params = params(d.dist)
    d1params = params(d.primary_event)
    return (d0params..., d1params...)
end

Base.eltype(::Type{<:PrimaryCensored{D}}) where {D} = promote_type(eltype(D), eltype(D))
Distributions.minimum(d::PrimaryCensored) = minimum(d.dist)
Distributions.maximum(d::PrimaryCensored) = maximum(d.dist)
Distributions.insupport(d::PrimaryCensored, x::Real) = insupport(d.dist, x)

function Distributions.cdf(d::PrimaryCensored, x::Real)
    function f(u, x)
        return exp(logcdf(d.dist, u) + logpdf(d.primary_event, x - u))
    end

    domain = (
        max(x - maximum(d.primary_event), 0.0), max(x - minimum(d.primary_event), 0.0))

    if domain[2] - domain[1] ≈ 0.0
        return 0.0
    end

    prob = IntegralProblem(f, domain, x)
    result = solve(prob, QuadGKJL())[1]
    return result
end

function Distributions.logcdf(d::PrimaryCensored, x::Real)
    if x == -Inf
        return -Inf
    end
    result = log(cdf(d, x))
    return result
end

function Distributions.ccdf(d::PrimaryCensored, x::Real)
    result = 1 - cdf(d, x)
    return result
end

function Distributions.logccdf(d::PrimaryCensored, x::Real)
    result = log(ccdf(d, x))
    return result
end

#### Sampling

function Base.rand(rng::AbstractRNG, d::PrimaryCensored)
    rand(rng, d.dist) + rand(rng, d.primary_event)
end

function Base.rand(
        rng::Random.AbstractRNG, d::Truncated{<:PrimaryCensored})
    d0 = d.untruncated
    lower = d.lower
    upper = d.upper
    while true
        r = rand(rng, d0)
        if Distributions._in_closed_interval(r, lower, upper)
            return r
        end
    end
end

# Sampler method for efficient sampling
Distributions.sampler(d::PrimaryCensored) = d
