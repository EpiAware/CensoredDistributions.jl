@doc "

Create a primary event censored distribution.

Models a process where a primary event occurs within a censoring window,
followed by a delay. The primary event time is not observed directly but is
known to fall within the censoring distribution's support. The observed time is
the sum of the primary event time and the delay.

# Method Selection

The CDF computation is handled by `primarycensored_cdf`, which dispatches on
the `method`:
- [`AnalyticalSolver`](@ref) (the default): closed-form solutions for these
  distribution pairs with Uniform primary events, falling back to numeric
  quadrature otherwise:
  - `Gamma` delay distribution
  - `LogNormal` delay distribution
  - `Weibull` delay distribution
- [`NumericSolver`](@ref): always uses quadrature integration, which may be
  necessary for certain AD backends or when debugging.

Passing the solver method as a concrete object keeps the return type concrete
even when the delay parameters are runtime values (e.g. inside a probabilistic
model), so it is preferred over the deprecated `force_numeric` flag.

# Arguments
- `dist`: The delay distribution from primary event to observation
- `primary_event`: The distribution of primary event times within the window

# Keyword Arguments
- `method`: The solver method, an [`AnalyticalSolver`](@ref) or
  [`NumericSolver`](@ref). Defaults to `AnalyticalSolver()`. Each takes an
  optional quadrature solver, e.g. `NumericSolver(QuadGKJL())`.
- `solver`: Quadrature solver used when `method` is not given (default:
  `GaussLegendre(; n = 64)`, AD-friendly; pass `QuadGKJL()` for adaptive
  accuracy).
- `force_numeric`: Deprecated. Pass `method = NumericSolver()` instead.

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
d_numeric = primary_censored(incubation, infection_window;
    method = NumericSolver())
```

# See also
- [`primarycensored_cdf`](@ref): The underlying CDF computation with method dispatch
"
function primary_censored(
        dist::UnivariateDistribution, primary_event::UnivariateDistribution;
        method::Union{AbstractSolverMethod, Nothing} = nothing,
        solver = GaussLegendre(; n = 64), force_numeric = nothing)
    resolved = _resolve_solver_method(method, solver, force_numeric)
    return PrimaryCensored(dist, primary_event, resolved)
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
        method::Union{AbstractSolverMethod, Nothing} = nothing,
        solver = GaussLegendre(; n = 64), force_numeric = nothing)
    return primary_censored(dist, primary_event; method = method,
        solver = solver, force_numeric = force_numeric)
end

@doc "

Represents the distribution of observed delays when the primary event time is
subject to censoring.

The `dist` field contains the delay distribution from primary event to observation.
The `primary_event` field contains the primary event time distribution.
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
       AbstractPrimaryCensored
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
    if !insupport(d, x)
        return -Inf
    end

    # Use central difference for numerical differentiation
    h = 1e-8  # Small step size for differentiation
    x_lower = max(x - h/2, minimum(d))
    x_upper = min(x + h/2, maximum(d))

    # Handle edge cases where we can't center the difference
    # Guard logsubexp: numerical noise in the CDF can make
    # the upper value smaller than the lower, which would
    # cause DomainError in logsubexp (log of negative)
    if x_lower == minimum(d)
        # Forward difference at minimum
        logcdf_upper = logcdf(d, x + h)
        logcdf_x = logcdf(d, x)
        logcdf_upper <= logcdf_x && return -Inf
        return logsubexp(logcdf_upper, logcdf_x) - log(h)
    elseif x_upper == maximum(d)
        # Backward difference at maximum
        logcdf_x = logcdf(d, x)
        logcdf_lower = logcdf(d, x - h)
        logcdf_x <= logcdf_lower && return -Inf
        return logsubexp(logcdf_x, logcdf_lower) - log(h)
    else
        # Central difference for interior points
        logcdf_upper = logcdf(d, x_upper)
        logcdf_lower = logcdf(d, x_lower)
        logcdf_upper <= logcdf_lower && return -Inf
        return logsubexp(logcdf_upper, logcdf_lower) -
               log(x_upper - x_lower)
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
