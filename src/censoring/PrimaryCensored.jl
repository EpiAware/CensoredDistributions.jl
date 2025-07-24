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
- `solver=QuadGKJL()`: Numerical integration solver for CDF computation
- `force_numeric=false`: If true, always use numerical integration; if false, use analytical solutions when available

# Returns
A `PrimaryCensored` distribution representing the convolution of the censoring and delay distributions.

# Examples
```@example
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

# Force numerical integration (useful for testing)
d_numeric = primary_censored(incubation, infection_window; force_numeric=true)
```
"
function primary_censored(
        dist::UnivariateDistribution, primary_event::UnivariateDistribution;
        solver = QuadGKJL(), force_numeric = false)
    method = if force_numeric
        NumericSolver(solver)
    else
        AnalyticalSolver(solver)
    end
    return PrimaryCensored(dist, primary_event, method)
end

@doc raw"
Primary event censored distribution.

Represents the distribution of observed delays when the primary event time is subject to censoring.

# Fields
- `dist`: Distribution of the delay from primary event to observation
- `primary_event`: Distribution of the primary event time
- `method`: Solver method for CDF computation (analytical or numeric)
"
struct PrimaryCensored{
    D1 <: UnivariateDistribution, D2 <: UnivariateDistribution, M <: AbstractSolverMethod} <:
       Distributions.UnivariateDistribution{Distributions.ValueSupport}
    "The delay distribution from primary event to observation."
    dist::D1
    "The primary event time distribution."
    primary_event::D2
    "The solver method for CDF computation."
    method::M

    function PrimaryCensored(
            dist::D1, primary_event::D2, method::M) where {
            D1, D2, M <: AbstractSolverMethod}
        minimum(dist) == 0 ||
            throw(ArgumentError("Delay distribution must have minimum of zero"))
        new{D1, D2, M}(dist, primary_event, method)
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
    primarycensored_cdf(d.dist, d.primary_event, x, d.method)
end

function Distributions.logcdf(d::PrimaryCensored, x::Real)
    primarycensored_logcdf(d.dist, d.primary_event, x, d.method)
end

function Distributions.ccdf(d::PrimaryCensored, x::Real)
    result = 1 - cdf(d, x)
    return result
end

function Distributions.logccdf(d::PrimaryCensored, x::Real)
    result = log(ccdf(d, x))
    return result
end

#### PDF using numerical differentiation of CDF
function Distributions.pdf(d::PrimaryCensored, x::Real)
    return exp(logpdf(d, x))
end

function Distributions.logpdf(d::PrimaryCensored, x::Real)
    if !insupport(d, x)
        return -Inf
    end

    # Use central difference for numerical differentiation
    h = 1e-8  # Small step size for differentiation
    x_lower = max(x - h/2, minimum(d))
    x_upper = min(x + h/2, maximum(d))

    # Handle edge cases where we can't center the difference
    if x_lower == minimum(d)
        # Forward difference at minimum
        logcdf_upper = logcdf(d, x + h)
        logcdf_x = logcdf(d, x)
        return log(exp(logcdf_upper) - exp(logcdf_x)) - log(h)
    elseif x_upper == maximum(d)
        # Backward difference at maximum
        logcdf_x = logcdf(d, x)
        logcdf_lower = logcdf(d, x - h)
        return log(exp(logcdf_x) - exp(logcdf_lower)) - log(h)
    else
        # Central difference for interior points
        logcdf_upper = logcdf(d, x_upper)
        logcdf_lower = logcdf(d, x_lower)
        return log(exp(logcdf_upper) - exp(logcdf_lower)) - log(x_upper - x_lower)
    end
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
