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
- `solver`: Quadrature solver (default: `GaussLegendre(; n = 64)`, AD-friendly;
  pass `QuadGKJL()` for adaptive accuracy)
- `force_numeric`: Force numeric integration even when analytical available (default: `false`)
- `formulation`: [`Marginal`](@ref)`()` (default) integrates the primary event
  out; [`Latent`](@ref)`(p)` conditions on a sampled primary event time `p`, so
  the cdf becomes `cdf(delay, x - p)` and the logpdf `logpdf(delay, x - p)`
  exactly. Sample `p` from [`primary_prior`](@ref).

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
        solver = GaussLegendre(; n = 64), force_numeric = false,
        formulation::AbstractFormulation = Marginal())
    method = if force_numeric
        NumericSolver(solver)
    else
        AnalyticalSolver(solver)
    end
    return PrimaryCensored(dist, primary_event, method, formulation)
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
        solver = GaussLegendre(; n = 64), force_numeric = false,
        formulation::AbstractFormulation = Marginal())
    return primary_censored(
        dist, primary_event; solver = solver, force_numeric = force_numeric,
        formulation = formulation)
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

The `formulation` field selects the representation:
- [`Marginal`](@ref) (default): integrate the primary event out via the
  `method` above.
- [`Latent`](@ref): condition on a sampled primary event time `p`, so the cdf
  collapses to the shifted delay cdf `F_delay(x - p)` and the logpdf to
  `logpdf(delay, x - p)` exactly.

# See also
- [`primary_censored`](@ref): Constructor function
- [`primarycensored_cdf`](@ref): CDF computation with method dispatch
- [`Marginal`](@ref), [`Latent`](@ref): formulation method types
- [`primary_prior`](@ref): prior over the latent primary `p`
"
struct PrimaryCensored{
    D1 <: UnivariateDistribution, D2 <: UnivariateDistribution,
    M <: AbstractSolverMethod, F <: AbstractFormulation} <:
       UnivariateDistribution{Continuous}
    "The delay distribution from primary event to observation."
    dist::D1
    "The primary event time distribution."
    primary_event::D2
    "The solver method for CDF computation."
    method::M
    "The formulation (Marginal or Latent) selecting the cdf/logpdf path."
    formulation::F

    function PrimaryCensored(
            dist::D1, primary_event::D2, method::M,
            formulation::F = Marginal()) where {
            D1, D2, M <: AbstractSolverMethod, F <: AbstractFormulation}
        new{D1, D2, M, F}(dist, primary_event, method, formulation)
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

Dispatches on the [`formulation`](@ref AbstractFormulation): [`Marginal`](@ref)
integrates the primary event out via [`primarycensored_cdf`](@ref); [`Latent`](@ref)
conditions on the sampled primary `p` and returns the shifted delay cdf
`cdf(delay, x - p)`.

See also: [`logcdf`](@ref)
"
function cdf(d::PrimaryCensored, x::Real)
    return _primarycensored_cdf(d, d.formulation, x)
end

# Marginal: integrate the primary event out (existing behaviour).
function _primarycensored_cdf(d::PrimaryCensored, ::Marginal, x::Real)
    primarycensored_cdf(get_dist(d), d.primary_event, x, d.method)
end

# Latent: condition on the sampled primary p -> shifted delay cdf.
function _primarycensored_cdf(d::PrimaryCensored, f::Latent, x::Real)
    return cdf(get_dist(d), x - f.p)
end

@doc "

Compute the log cumulative distribution function.

Dispatches on the [`formulation`](@ref AbstractFormulation): [`Marginal`](@ref)
uses [`primarycensored_logcdf`](@ref); [`Latent`](@ref) returns
`logcdf(delay, x - p)`.

See also: [`cdf`](@ref)
"
function logcdf(d::PrimaryCensored, x::Real)
    return _primarycensored_logcdf(d, d.formulation, x)
end

function _primarycensored_logcdf(d::PrimaryCensored, ::Marginal, x::Real)
    primarycensored_logcdf(get_dist(d), d.primary_event, x, d.method)
end

function _primarycensored_logcdf(d::PrimaryCensored, f::Latent, x::Real)
    return logcdf(get_dist(d), x - f.p)
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

Compute the log probability density function.

Dispatches on the [`formulation`](@ref AbstractFormulation): [`Latent`](@ref)
returns the exact shifted delay logpdf `logpdf(delay, x - p)` (no quadrature,
no finite differencing); [`Marginal`](@ref) numerically differentiates the log
CDF.

See also: [`pdf`](@ref), [`logcdf`](@ref)
"
function logpdf(d::PrimaryCensored, x::Real)
    return _primarycensored_logpdf(d, d.formulation, x)
end

# Latent: exact shifted delay density conditioned on the sampled primary p.
function _primarycensored_logpdf(d::PrimaryCensored, f::Latent, x::Real)
    return logpdf(get_dist(d), x - f.p)
end

# Marginal: numerically differentiate the log CDF (existing behaviour).
function _primarycensored_logpdf(d::PrimaryCensored, ::Marginal, x::Real)
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

@doc "

Return the prior distribution over the latent primary event time `p`.

This is the distribution a user samples to drive the [`Latent`](@ref)
formulation, e.g. in their own PPL

```julia
p ~ primary_prior(d)
y ~ <Latent-formulation delay conditioned on p>
```

# Arguments
- `d`: A primary-censored distribution.
- `secondary`: (optional) An already-sampled secondary time bounding the primary
  window from above, for the coupled case.

# Uncoupled case
`primary_prior(d)` returns the distribution's own `primary_event`. Sampling it
and conditioning the delay on `p` recovers the [`Marginal`](@ref) formulation in
expectation (the marginal integrates this same prior out).

# Coupled case
`primary_prior(d, secondary)` returns a bounded prior whose window is truncated
above by an already-sampled `secondary` time (so the implied delay is
non-negative), with the `log(upper - lower)` Jacobian that restores the
implicit uniform-over-window prior. This is the folded former
`WithinWindowPrimary` logic; it requires `d.primary_event` to be a `Uniform`
window.

# Examples
```@example
using CensoredDistributions, Distributions

d = primary_censored(LogNormal(1.5, 0.75), Uniform(0.0, 1.0))

# Uncoupled prior over p
prior = primary_prior(d)            # === Uniform(0.0, 1.0)
p = rand(prior)

# Coupled prior bounded by a sampled secondary time
coupled = primary_prior(d, 0.6)     # bounded to [0.0, 0.6]
```

# See also
- [`Latent`](@ref): the formulation that consumes this prior
"
primary_prior(d::PrimaryCensored) = d.primary_event

function primary_prior(d::PrimaryCensored, secondary::Real)
    pe = d.primary_event
    pe isa Uniform ||
        throw(ArgumentError(
            "Coupled primary_prior requires a Uniform primary_event window; " *
            "got $(typeof(pe))"))
    lower = minimum(pe)
    width = maximum(pe) - lower
    return _bounded_primary(lower, width, secondary)
end
