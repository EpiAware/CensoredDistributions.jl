@doc "

Create a primary event censored distribution.

Models a process where a primary event occurs within a censoring window,
followed by a delay. The primary event time is not observed directly but is
known to fall within the censoring distribution's support. The observed time is
the sum of the primary event time and the delay.

# Method Selection

The CDF computation is handled by `primarycensored_cdf` which automatically
dispatches between:
- **Analytical methods**: Available for these distribution pairs with Uniform
  primary events when `force_numeric=false`:
  - `Gamma` delay distribution
  - `LogNormal` delay distribution
  - `Weibull` delay distribution
- **Numeric integration**: Falls back to quadrature for all other distribution
  pairs or when `force_numeric=true`

Set `force_numeric=true` to always use numeric integration, which may be
necessary for certain AD backends or when debugging.

# Arguments
- `dist`: The delay distribution from primary event to observation
- `primary_event`: The distribution of primary event times within the window

# Keyword Arguments
- `solver`: Quadrature solver (default: `QuadGKJL()`)
- `force_numeric`: Force numeric integration even when analytical available (default: `false`)

This is useful for modeling:
- Infection-to-symptom onset times when infection time is uncertain
- Exposure-to-outcome delays with uncertain exposure timing
- Any process where the initiating event time has uncertainty

# Examples
```@example
using CensoredDistributions, Distributions

# Incubation period (delay) with uncertain infection time (primary event)
incubation = LogNormal(1.5, 0.75)  # Delay distribution
infection_window = Uniform(0, 1)    # Daily infection window
d = primary_censored(incubation, infection_window)

# Evaluate distribution functions
pdf_at_2 = pdf(d, 2.0)    # probability density at 2 days
cdf_at_5 = cdf(d, 5.0)    # cumulative probability by 5 days
q50 = quantile(d, 0.5)    # median

# Force numeric integration for debugging or AD compatibility
d_numeric = primary_censored(incubation, infection_window; force_numeric=true)
```

# See also
- [`primarycensored_cdf`](@ref): The underlying CDF computation with method dispatch
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

@doc "

Create a primary event censored distribution with keyword arguments.

This is a convenience version of `primary_censored` that uses keyword arguments
for consistency with `double_interval_censored`. The primary event distribution
defaults to `Uniform(0, 1)`.

# Examples
```@example
using CensoredDistributions, Distributions

# Using default Uniform(0, 1) primary event
d1 = primary_censored(LogNormal(1.5, 0.75))

# Custom primary event distribution
d2 = primary_censored(LogNormal(1.5, 0.75); primary_event=Uniform(0, 2))
```
"
function primary_censored(
        dist::UnivariateDistribution;
        primary_event::UnivariateDistribution = Uniform(0, 1),
        solver = QuadGKJL(), force_numeric = false)
    return primary_censored(
        dist, primary_event; solver = solver, force_numeric = force_numeric)
end

@doc "

Represents the distribution of observed delays when the primary event time is
subject to censoring.

The `method` field determines computation strategy:
- `AnalyticalSolver`: Uses closed-form solutions when available (Gamma,
  LogNormal, Weibull with Uniform primary), falls back to numeric otherwise
- `NumericSolver`: Always uses quadrature integration


# See also
- [`primary_censored`](@ref): Constructor function
- [`primarycensored_cdf`](@ref): CDF computation with method dispatch
"
struct PrimaryCensored{
    D1 <: UnivariateDistribution, D2 <: UnivariateDistribution,
    M <: AbstractSolverMethod} <:
       UnivariateDistribution{Continuous}
    "The delay distribution from primary event to observation."
    dist::D1
    "The primary event time distribution."
    primary_event::D2
    "The solver method for CDF computation."
    method::M

    function PrimaryCensored(
            dist::D1, primary_event::D2, method::M) where {
            D1, D2, M <: AbstractSolverMethod}
        new{D1, D2, M}(dist, primary_event, method)
    end
end

function params(d::PrimaryCensored)
    d0params = params(get_dist(d))
    d1params = params(d.primary_event)
    return (d0params..., d1params...)
end

Base.eltype(::Type{<:PrimaryCensored{D}}) where {D} = promote_type(eltype(D), eltype(D))
minimum(d::PrimaryCensored) = minimum(get_dist(d))
maximum(d::PrimaryCensored) = maximum(get_dist(d))
insupport(d::PrimaryCensored, x::Real) = insupport(get_dist(d), x)

@doc "

Compute the cumulative distribution function.

See also: [`logcdf`](@ref)
"
function cdf(d::PrimaryCensored, x::Real)
    primarycensored_cdf(get_dist(d), d.primary_event, x, d.method)
end

@doc "

Compute the log cumulative distribution function.

See also: [`cdf`](@ref)
"
function logcdf(d::PrimaryCensored, x::Real)
    primarycensored_logcdf(get_dist(d), d.primary_event, x, d.method)
end

function ccdf(d::PrimaryCensored, x::Real)
    result = 1 - cdf(d, x)
    return result
end

function logccdf(d::PrimaryCensored, x::Real)
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

#### PDF using numerical differentiation of CDF
@doc "

Compute the probability density function using numerical differentiation.

See also: [`logpdf`](@ref)
"
function pdf(d::PrimaryCensored, x::Real)
    return exp(logpdf(d, x))
end

@doc "

Compute the log probability density function using numerical differentiation
of the log CDF.

See also: [`pdf`](@ref), [`logcdf`](@ref)
"
function logpdf(d::PrimaryCensored, x::Real)
    try
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
            return logsubexp(logcdf_upper, logcdf_x) - log(h)
        elseif x_upper == maximum(d)
            # Backward difference at maximum
            logcdf_x = logcdf(d, x)
            logcdf_lower = logcdf(d, x - h)
            return logsubexp(logcdf_x, logcdf_lower) - log(h)
        else
            # Central difference for interior points
            logcdf_upper = logcdf(d, x_upper)
            logcdf_lower = logcdf(d, x_lower)
            return logsubexp(logcdf_upper, logcdf_lower) -
                   log(x_upper - x_lower)
        end
    catch e
        # If numerical differentiation fails (e.g., domain error in logsubexp),
        # return -Inf
        if isa(e, DomainError) || isa(e, BoundsError) || isa(e, ArgumentError)
            return -Inf
        else
            rethrow(e)
        end
    end
end

#### Quantile function using numerical optimization

@doc "

Compute the quantile (inverse CDF) using numerical optimization.

See also: [`cdf`](@ref)
"
function quantile(d::PrimaryCensored, p::Real)
    # Custom initial guess: underlying quantile + mean of primary event
    initial_guess_fn = function (d, p)
        underlying_quantile = quantile(get_dist(d), p)
        primary_mean = mean(d.primary_event)
        return [underlying_quantile + primary_mean]
    end

    return _quantile_optimization(d, p; initial_guess_fn = initial_guess_fn)
end

#### Sampling

@doc "

Generate a random sample by summing samples from delay and primary event
distributions.

See also: [`quantile`](@ref)
"
function Base.rand(rng::AbstractRNG, d::PrimaryCensored)
    rand(rng, get_dist(d)) + rand(rng, d.primary_event)
end

function Base.rand(
        rng::AbstractRNG, d::Truncated{<:PrimaryCensored})
    d0 = d.untruncated
    lower = d.lower
    upper = d.upper
    while true
        r = rand(rng, d0)
        if _in_closed_interval(r, lower, upper)
            return r
        end
    end
end

# Sampler method for efficient sampling
sampler(d::PrimaryCensored) = d
