@doc "

Create a primary event censored distribution.

Models a process where a primary event occurs within a censoring window,
followed by a delay. The primary event time is not observed directly but is
known to fall within the censoring distribution's support. The observed time is
the sum of the primary event time and the delay.

# Method Selection

The distribution is a thin wrapper over
`ConvolvedDistributions.Convolved((primary_event, dist))`. The default
[`AnalyticalSolver`](@ref) uses ConvolvedDistributions' closed-form solutions
for distribution pairs with `Uniform` primary events (see the
[`convolved_cdf` implementations](https://github.com/EpiAware/ConvolvedDistributions.jl/blob/main/src/uniform_window.jl)
for the supported families), falling back to its numeric quadrature otherwise.

[`NumericSolver`](@ref) (re-exported from ConvolvedDistributions) always uses
quadrature integration, which may be necessary for certain AD backends or
when debugging.

Passing the solver method as a concrete object keeps the return type concrete
even when the delay parameters are runtime values (e.g. inside a probabilistic
model). The former `solver` keyword has been removed: numeric integration now
lives in ConvolvedDistributions and uses its fixed default quadrature (custom
solver payloads are not yet honoured there — tracked as
[ConvolvedDistributions#92](https://github.com/EpiAware/ConvolvedDistributions.jl/issues/92)).

# Arguments
- `dist`: The delay distribution from primary event to observation
- `primary_event`: The distribution of primary event times within the window

# Keyword Arguments
- `method`: The solver method, an [`AnalyticalSolver`](@ref) or
  [`NumericSolver`](@ref), re-exported from `ConvolvedDistributions`.
  Defaults to `AnalyticalSolver()`.

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
- [`ConvolvedDistributions.Convolved`](@extref): The backing convolution
"
function primary_censored(
        dist::UnivariateDistribution, primary_event::UnivariateDistribution;
        method::Union{AbstractSolverMethod, Nothing} = nothing)
    resolved = _resolve_solver_method(method)
    return PrimaryCensored(dist, primary_event; method = resolved)
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
        method::Union{AbstractSolverMethod, Nothing} = nothing)
    return primary_censored(dist, primary_event; method = method)
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

All evaluation delegates to the wrapped
[`ConvolvedDistributions.Convolved`](@extref) in the `convolved` field, built
with the `method` supplied at construction.

# See also
- [`primary_censored`](@ref): Constructor function
"
struct PrimaryCensored{
    D1 <: UnivariateDistribution, D2 <: UnivariateDistribution,
    C, M} <: UnivariateDistribution{Continuous}
    "The delay distribution from primary event to observation."
    dist::D1
    "The primary event time distribution."
    primary_event::D2
    "The wrapped ConvolvedDistributions.Convolved((primary_event, dist))."
    convolved::C
    "The solver method (ConvolvedDistributions.AbstractSolverMethod) for evaluation."
    method::M

    function PrimaryCensored(
            dist::D1, primary_event::D2;
            method::AbstractSolverMethod = AnalyticalSolver()) where {
            D1 <: UnivariateDistribution, D2 <: UnivariateDistribution}
        c = Convolved((primary_event, dist); method = method)
        new{D1, D2, typeof(c), typeof(c.method)}(dist, primary_event, c, c.method)
    end
end

function params(d::PrimaryCensored)
    d0params = params(get_dist(d))
    d1params = params(d.primary_event)
    return (d0params..., d1params...)
end

function Base.eltype(::Type{<:PrimaryCensored{D1, D2}}) where {D1, D2}
    promote_type(eltype(D1), eltype(D2))
end

minimum(d::PrimaryCensored) = minimum(d.convolved)
maximum(d::PrimaryCensored) = maximum(d.convolved)
insupport(d::PrimaryCensored, x::Real) = insupport(d.convolved, x)

@doc "

Compute the cumulative distribution function.

See also: [`logcdf`](@ref)
"
function cdf(d::PrimaryCensored, x::Real)
    cdf(d.convolved, x)
end

@doc "

Compute the log cumulative distribution function.

See also: [`cdf`](@ref)
"
function logcdf(d::PrimaryCensored, x::Real)
    logcdf(d.convolved, x)
end

function ccdf(d::PrimaryCensored, x::Real)
    ccdf(d.convolved, x)
end

function logccdf(d::PrimaryCensored, x::Real)
    logccdf(d.convolved, x)
end

@doc "

Compute the probability density function.

See also: [`logpdf`](@ref)
"
function pdf(d::PrimaryCensored, x::Real)
    pdf(d.convolved, x)
end

@doc "

Compute the log probability density function.

See also: [`pdf`](@ref), [`logcdf`](@ref)
"
function logpdf(d::PrimaryCensored, x::Real)
    logpdf(d.convolved, x)
end

#### Quantile function using numerical optimization

@doc "

Compute the quantile (inverse CDF) using numerical optimization.

Inverts `cdf(d, ·)` via `quantile_by_optimization`, seeded with a custom
initial guess.

See also: [`cdf`](@ref)
"
function quantile(d::PrimaryCensored, p::Real)
    initial_guess = [0 <= p <= 1 ?
                     quantile(d.dist, p) + mean(d.primary_event) : p]
    return quantile_by_optimization(d, p, initial_guess)
end

#### Sampling

@doc "

Generate a random sample by summing samples from delay and primary event
distributions.

See also: [`quantile`](@ref)
"
function Base.rand(rng::AbstractRNG, d::PrimaryCensored)
    rand(rng, d.convolved)
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

_resolve_solver_method(method::AbstractSolverMethod) = method
_resolve_solver_method(::Nothing) = AnalyticalSolver()
