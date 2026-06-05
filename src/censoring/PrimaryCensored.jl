@doc "

Create a primary event censored distribution.

Models a process where a primary event occurs within a censoring window,
followed by a delay. The primary event time is not observed directly but is
known to fall within the censoring distribution's support. The observed time is
the sum of the primary event time and the delay.

# Formulation method

The `method` keyword selects the formulation (a type parameter, so each variant
is type-stable):
- [`Marginal`](@ref) (default): integrate the primary event out. The result is
  a **univariate** distribution over the scalar observed delay, with behaviour
  identical to the classic `PrimaryCensored`.
- [`Latent`](@ref): keep the primary event explicit. The result is a
  **multivariate** distribution over the event times `[primary, observed]`;
  `rand` returns both, and `logpdf([p, y])` is the combined joint density. This
  is the data-augmentation form a sampler uses directly.

# CDF computation (Marginal)

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
- `method`: Formulation method, [`Marginal`](@ref)`()` (default) or
  [`Latent`](@ref)`()`
- `latent`: Convenience flag; `latent=true` is equivalent to `method=Latent()`

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

# Latent (multivariate) formulation over [primary, observed] event times
d_latent = primary_censored(incubation, infection_window; latent=true)
sample = rand(d_latent)               # [primary, observed]
lp = logpdf(d_latent, sample)         # joint density
```

# See also
- [`primarycensored_cdf`](@ref): The underlying CDF computation with method dispatch
- [`Marginal`](@ref), [`Latent`](@ref): formulation methods
- [`primary_prior`](@ref): the prior over the latent primary event time
"
function primary_censored(
        dist::UnivariateDistribution, primary_event::UnivariateDistribution;
        solver = GaussLegendre(; n = 64), force_numeric = false,
        method::AbstractPCMethod = Marginal(), latent::Bool = false)
    solver_method = if force_numeric
        NumericSolver(solver)
    else
        AnalyticalSolver(solver)
    end
    pcmethod = latent ? Latent() : method
    return PrimaryCensored(dist, primary_event, solver_method, pcmethod)
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
        method::AbstractPCMethod = Marginal(), latent::Bool = false)
    return primary_censored(
        dist, primary_event; solver = solver, force_numeric = force_numeric,
        method = method, latent = latent)
end

@doc "

Represents the distribution of observed delays when the primary event time is
subject to censoring.

The `dist` field contains the delay distribution from primary event to observation.
The `primary_event` field contains the primary event time distribution.
The `solver` field determines the CDF computation strategy:
- `AnalyticalSolver`: Uses closed-form solutions when available (Gamma,
  LogNormal, Weibull with Uniform primary), falls back to numeric otherwise
- `NumericSolver`: Always uses quadrature integration

The `method` field (a type parameter) selects the formulation:
- [`Marginal`](@ref): univariate over the scalar observed delay (default).
- [`Latent`](@ref): multivariate over `[primary, observed]` event times.

The variate form `V` is `Univariate` for `Marginal` and `Multivariate` for
`Latent`, so the right `Distributions.jl` interface is dispatched automatically.

# See also
- [`primary_censored`](@ref): Constructor function
- [`primarycensored_cdf`](@ref): CDF computation with method dispatch
"
struct PrimaryCensored{
    V <: VariateForm, D1 <: UnivariateDistribution,
    D2 <: UnivariateDistribution, S <: AbstractSolverMethod,
    M <: AbstractPCMethod} <: Distribution{V, Continuous}
    "The delay distribution from primary event to observation."
    dist::D1
    "The primary event time distribution."
    primary_event::D2
    "The solver method for CDF computation."
    solver::S
    "The formulation method (Marginal or Latent)."
    method::M

    function PrimaryCensored(
            dist::D1, primary_event::D2, solver::S,
            method::M = Marginal()) where {
            D1, D2, S <: AbstractSolverMethod, M <: AbstractPCMethod}
        V = _variate_form(method)
        new{V, D1, D2, S, M}(dist, primary_event, solver, method)
    end
end

# Variate form implied by the formulation method.
_variate_form(::Marginal) = Univariate
_variate_form(::Latent) = Multivariate

# Convenience aliases for dispatch on the formulation.
const MarginalPrimaryCensored{D1, D2, S} = PrimaryCensored{
    Univariate, D1, D2, S, Marginal}
const LatentPrimaryCensored{D1, D2, S} = PrimaryCensored{
    Multivariate, D1, D2, S, Latent}

function params(d::PrimaryCensored)
    d0params = params(get_dist(d))
    d1params = params(d.primary_event)
    return (d0params..., d1params...)
end

# ============================================================================
# Marginal (univariate) interface — unchanged classic behaviour
# ============================================================================

function Base.eltype(::Type{<:MarginalPrimaryCensored{D}}) where {D}
    promote_type(
        eltype(D), eltype(D))
end
minimum(d::MarginalPrimaryCensored) = minimum(get_dist(d))
maximum(d::MarginalPrimaryCensored) = maximum(get_dist(d))
insupport(d::MarginalPrimaryCensored, x::Real) = insupport(get_dist(d), x)

@doc "

Compute the cumulative distribution function.

See also: [`logcdf`](@ref)
"
function cdf(d::MarginalPrimaryCensored, x::Real)
    primarycensored_cdf(get_dist(d), d.primary_event, x, d.solver)
end

@doc "

Compute the log cumulative distribution function.

See also: [`cdf`](@ref)
"
function logcdf(d::MarginalPrimaryCensored, x::Real)
    primarycensored_logcdf(get_dist(d), d.primary_event, x, d.solver)
end

function ccdf(d::MarginalPrimaryCensored, x::Real)
    result = 1 - cdf(d, x)
    return result
end

function logccdf(d::MarginalPrimaryCensored, x::Real)
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
function pdf(d::MarginalPrimaryCensored, x::Real)
    return exp(logpdf(d, x))
end

@doc "

Compute the log probability density function using numerical differentiation
of the log CDF.

See also: [`pdf`](@ref), [`logcdf`](@ref)
"
function logpdf(d::MarginalPrimaryCensored, x::Real)
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
function quantile(d::MarginalPrimaryCensored, p::Real)
    # Custom initial guess: underlying quantile + mean of primary event
    initial_guess_fn = function (d, p)
        underlying_quantile = quantile(get_dist(d), p)
        primary_mean = mean(d.primary_event)
        return [underlying_quantile + primary_mean]
    end

    return _quantile_optimization(d, p; initial_guess_fn = initial_guess_fn)
end

#### Sampling (Marginal)

@doc "

Generate a random sample by summing samples from delay and primary event
distributions.

See also: [`quantile`](@ref)
"
function Base.rand(rng::AbstractRNG, d::MarginalPrimaryCensored)
    rand(rng, get_dist(d)) + rand(rng, d.primary_event)
end

function Base.rand(
        rng::AbstractRNG, d::Truncated{<:MarginalPrimaryCensored})
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
sampler(d::MarginalPrimaryCensored) = d

# ============================================================================
# Latent (multivariate) interface — joint over [primary, observed]
# ============================================================================

@doc "

Length of the latent event-time vector, `[primary, observed]` (always 2).
"
Base.length(::LatentPrimaryCensored) = 2
Base.eltype(::Type{<:LatentPrimaryCensored{D}}) where {D} = eltype(D)

@doc "

Test whether `x = [primary, observed]` is in the joint support: the primary in
the primary-event support and the implied delay `observed - primary` in the
delay support.

See also: [`logpdf`](@ref)
"
function insupport(d::LatentPrimaryCensored, x::AbstractVector)
    length(x) == 2 || return false
    p, y = x[1], x[2]
    return insupport(d.primary_event, p) && insupport(get_dist(d), y - p)
end

@doc """

Combined joint log density over the latent event times `x = [primary, observed]`,

```math
\\log f(p, y) = \\log f_\\mathrm{primary}(p) + \\log f_\\mathrm{delay}(y - p),
```

i.e. the sum of the primary event prior density and the conditional delay
density at the implied delay `y - p`. Deterministic given `x` (no hidden
randomness); the sampler owns the primary draw.

See also: [`rand`](@ref), [`primary_prior`](@ref)
"""
function logpdf(d::LatentPrimaryCensored, x::AbstractVector)
    p = x[1]
    y = x[2]
    return logpdf(d.primary_event, p) + logpdf(get_dist(d), y - p)
end

@doc "

Joint density over the latent event times `[primary, observed]`.

See also: [`logpdf`](@ref)
"
pdf(d::LatentPrimaryCensored, x::AbstractVector) = exp(logpdf(d, x))

@doc "

Draw a latent event-time vector `[primary, observed]`: sample a fresh primary
event time, then the observed time as primary plus an independent delay draw.

See also: [`logpdf`](@ref)
"
function Base.rand(rng::AbstractRNG, d::LatentPrimaryCensored)
    p = rand(rng, d.primary_event)
    y = p + rand(rng, get_dist(d))
    return [p, y]
end

# In-place sampling for the multivariate interface.
function Distributions._rand!(
        rng::AbstractRNG, d::LatentPrimaryCensored, x::AbstractVector)
    p = rand(rng, d.primary_event)
    x[1] = p
    x[2] = p + rand(rng, get_dist(d))
    return x
end

# ============================================================================
# Primary-event prior accessor (both formulations)
# ============================================================================

@doc "

Return the prior distribution over the latent primary event time.

The distribution owns the primary *specification*; the sampler owns the *draw*.
A user samples this prior to drive the latent / data-augmentation workflow, e.g.

```julia
p ~ primary_prior(d)
obs ~ <conditional delay at p>
```

# Arguments
- `d`: A primary-censored distribution.
- `secondary`: (optional) An already-sampled secondary time bounding the primary
  window from above, for the coupled case.

# Uncoupled case
`primary_prior(d)` returns the distribution's own `primary_event`.

# Coupled case
`primary_prior(d, secondary)` returns a [`BoundedPrimary`](@ref): a bounded
prior whose window is truncated above by `secondary` (so the implied delay is
non-negative), with the `log(upper - lower)` Jacobian that restores the implicit
uniform-over-window prior. This requires a `Uniform` primary event (the Jacobian
is Uniform-only) and errors otherwise.

# Examples
```@example
using CensoredDistributions, Distributions

d = primary_censored(LogNormal(1.5, 0.75), Uniform(0.0, 1.0))

# Uncoupled prior over the primary event time
prior = primary_prior(d)            # === Uniform(0.0, 1.0)
p = rand(prior)

# Coupled prior bounded by a sampled secondary time
coupled = primary_prior(d, 0.6)     # bounded to [0.0, 0.6]
```

# See also
- [`Latent`](@ref): the formulation that consumes this prior
- [`BoundedPrimary`](@ref): the coupled bounded prior
"
primary_prior(d::PrimaryCensored) = d.primary_event

function primary_prior(d::PrimaryCensored, secondary::Real)
    return _coupled_primary_prior(d.primary_event, secondary)
end
