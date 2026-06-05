@doc "

Create a primary event censored distribution.

Models a process where a primary event occurs within a censoring window,
followed by a delay. The primary event time is not observed directly but is
known to fall within the censoring distribution's support. The observed time is
the sum of the primary event time and the delay.

# Formulation

The `mode` keyword selects the formulation:
- [`Marginal`](@ref)`()` (the default): **marginalise** the primary event time,
  giving the classic univariate distribution over the scalar observed delay with
  the full scalar interface (`cdf`, `quantile`, `rand`, `mean`, truncation, ...).
  `Marginal` **auto-falls-back to [`Latent`](@ref)** at construction for any
  component that genuinely cannot be marginalised (no analytical or quadrature
  route, event times coupled across records, or the internal times are needed) —
  this is real runtime behaviour. For a single delay the marginal route always
  exists, so it stays univariate.
- [`Latent`](@ref)`()`: force the latent formulation always, keeping the primary
  event time as a **sampler-owned latent variable**. The result is a
  **multivariate** distribution over `[primary, observed]`; [`rand`](@ref)
  *produces* the internal event times, and `logpdf([primary, observed])` scores
  the full self-contained joint
  (`logpdf(primary_event, primary) + logpdf(delay, observed - primary)`, prior
  included), so it drops into the same weighted likelihood loop as the marginal
  form. The user never passes the primary in as data. Opt into `Latent()`
  whenever you want this on purpose — for the internal times via `rand` or for
  coupled models.

# CDF computation (marginal default)

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
- `mode`: Formulation mode, [`Marginal`](@ref)`()` (default; marginalise with
  auto-fallback to latent) or [`Latent`](@ref)`()` (force latent always).

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

# Default: the primary is marginalised (the classic univariate distribution)
pdf_at_2 = pdf(d, 2.0)    # marginal density at 2 days
cdf_at_5 = cdf(d, 5.0)    # marginal cumulative probability by 5 days
q50 = quantile(d, 0.5)    # median

# Force numeric integration for debugging or AD compatibility
d_numeric = primary_censored(incubation, infection_window; force_numeric=true)

# Latent: rand produces the internal event times [primary, observed]; logpdf
# scores the joint of the sampler-owned values (the primary is not passed in)
d_latent = primary_censored(incubation, infection_window; mode=Latent())
sample = rand(d_latent)          # [primary, observed], primary produced by rand
lp = logpdf(d_latent, sample)    # joint density
```

# See also
- [`primarycensored_cdf`](@ref): The underlying CDF computation with method dispatch
- [`Marginal`](@ref), [`Latent`](@ref): the formulation modes
- [`get_primary_event`](@ref): the primary event distribution
"
function primary_censored(
        dist::UnivariateDistribution, primary_event::UnivariateDistribution;
        solver = GaussLegendre(; n = 64), force_numeric = false,
        mode::AbstractPCMethod = Marginal())
    solver_method = if force_numeric
        NumericSolver(solver)
    else
        AnalyticalSolver(solver)
    end
    return PrimaryCensored(dist, primary_event, solver_method, mode)
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
        mode::AbstractPCMethod = Marginal())
    return primary_censored(
        dist, primary_event; solver = solver, force_numeric = force_numeric,
        mode = mode)
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

The `method` field (a type parameter) holds the effective formulation mode:
- [`Marginal`](@ref): univariate over the scalar observed delay, integrating the
  primary event time out (the default; auto-falls-back to `Latent` for a
  component that cannot be marginalised).
- [`Latent`](@ref): multivariate over `[primary, observed]`, keeping the primary
  as a sampler-owned latent variable.

The variate form `V` is `Univariate` for [`Marginal`](@ref) and `Multivariate`
for [`Latent`](@ref), so the right `Distributions.jl` interface is dispatched
automatically.

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
    "The formulation mode ([`Marginal`](@ref) default, or [`Latent`](@ref))."
    method::M

    function PrimaryCensored(
            dist::D1, primary_event::D2, solver::S,
            mode::AbstractPCMethod = Marginal()) where {
            D1, D2, S <: AbstractSolverMethod}
        # `Marginal` auto-falls-back to `Latent` for any component that cannot be
        # marginalised; `Latent` is always latent. `eff` is the effective mode.
        eff = _resolve_mode(dist, primary_event, solver, mode)
        M = typeof(eff)
        V = _variate_form(eff)
        new{V, D1, D2, S, M}(dist, primary_event, solver, eff)
    end
end

# Resolve the requested mode into the effective mode actually used. `Latent`
# stays latent. `Marginal` stays marginal when the component can be marginalised
# and otherwise falls back to `Latent` (real runtime behaviour, not just docs).
_resolve_mode(dist, primary_event, solver, mode::Latent) = mode
function _resolve_mode(dist, primary_event, solver, mode::Marginal)
    return _can_marginalise(dist, primary_event, solver) ? mode : Latent()
end

# Whether the primary event can be marginalised out of the delay. For a single
# delay the marginal CDF always has a quadrature route (see `primarycensored_cdf`
# / `NumericSolver`), so this is `true`. The hook exists so composed components
# that genuinely cannot be marginalised can return `false` and trigger the
# `Marginal -> Latent` fallback above.
_can_marginalise(dist, primary_event, solver) = true

# Variate form implied by the (effective) formulation mode.
#
# `Marginal` is Univariate: it integrates the primary event out and exposes the
# classic scalar interface (cdf, quantile, rand, mean, truncation, ...), so
# `primary_censored(dist, pe)` is the familiar univariate distribution. `Latent`
# is Multivariate: the joint over the sampler-owned event times.
_variate_form(::Marginal) = Univariate
_variate_form(::Latent) = Multivariate

# Convenience aliases for dispatch on the formulation.
const MarginalPrimaryCensored{D1, D2, S} = PrimaryCensored{
    Univariate, D1, D2, S, Marginal}
const LatentPrimaryCensored{D1, D2, S} = PrimaryCensored{
    Multivariate, D1, D2, S, Latent}

# Scalar-interface alias: the default marginal formulation exposes the classic
# univariate scalar interface (cdf, logcdf, logpdf, pdf, quantile, rand,
# truncation). Kept generic on the method parameter for forward compatibility.
const ScalarPrimaryCensored{D1, D2, S, M} = PrimaryCensored{
    Univariate, D1, D2, S, M}

function params(d::PrimaryCensored)
    d0params = params(get_dist(d))
    d1params = params(d.primary_event)
    return (d0params..., d1params...)
end

# ============================================================================
# Scalar (univariate) interface — shared by the Marginal force path and the
# Auto default. The Latent (multivariate, sampler-owned) interface and the Auto
# vector method live in `primary_censored_latent.jl`.
# ============================================================================

function Base.eltype(::Type{<:ScalarPrimaryCensored{D, P}}) where {D, P}
    # Observed delay = delay draw + primary draw, so promote both element types.
    promote_type(eltype(D), eltype(P))
end
minimum(d::ScalarPrimaryCensored) = minimum(get_dist(d))
maximum(d::ScalarPrimaryCensored) = maximum(get_dist(d))
insupport(d::ScalarPrimaryCensored, x::Real) = insupport(get_dist(d), x)

# ----------------------------------------------------------------------------
# Shared marginal computations (used by the Marginal force path and the Auto
# marginal/missing path). They operate on the distribution fields so they are
# independent of the wrapper's variate form.
# ----------------------------------------------------------------------------

function _marginal_cdf(d::PrimaryCensored, x::Real)
    primarycensored_cdf(get_dist(d), d.primary_event, x, d.solver)
end

function _marginal_logcdf(d::PrimaryCensored, x::Real)
    primarycensored_logcdf(get_dist(d), d.primary_event, x, d.solver)
end

# Marginal log density by central-difference of the log CDF (the AD-safe scalar
# path). `xmin`/`xmax` are the delay support bounds.
function _marginal_logpdf(d::PrimaryCensored, x::Real)
    xmin = minimum(get_dist(d))
    xmax = maximum(get_dist(d))
    if !insupport(get_dist(d), x)
        return -Inf
    end

    # Use central difference for numerical differentiation
    h = 1e-8  # Small step size for differentiation
    x_lower = max(x - h/2, xmin)
    x_upper = min(x + h/2, xmax)

    # Handle edge cases where we can't center the difference
    # Guard logsubexp: numerical noise in the CDF can make
    # the upper value smaller than the lower, which would
    # cause DomainError in logsubexp (log of negative)
    if x_lower == xmin
        # Forward difference at minimum
        logcdf_upper = _marginal_logcdf(d, x + h)
        logcdf_x = _marginal_logcdf(d, x)
        logcdf_upper <= logcdf_x && return -Inf
        return logsubexp(logcdf_upper, logcdf_x) - log(h)
    elseif x_upper == xmax
        # Backward difference at maximum
        logcdf_x = _marginal_logcdf(d, x)
        logcdf_lower = _marginal_logcdf(d, x - h)
        logcdf_x <= logcdf_lower && return -Inf
        return logsubexp(logcdf_x, logcdf_lower) - log(h)
    else
        # Central difference for interior points
        logcdf_upper = _marginal_logcdf(d, x_upper)
        logcdf_lower = _marginal_logcdf(d, x_lower)
        logcdf_upper <= logcdf_lower && return -Inf
        return logsubexp(logcdf_upper, logcdf_lower) -
               log(x_upper - x_lower)
    end
end

@doc "

Compute the cumulative distribution function.

See also: [`logcdf`](@ref)
"
cdf(d::ScalarPrimaryCensored, x::Real) = _marginal_cdf(d, x)

@doc "

Compute the log cumulative distribution function.

See also: [`cdf`](@ref)
"
logcdf(d::ScalarPrimaryCensored, x::Real) = _marginal_logcdf(d, x)

function ccdf(d::ScalarPrimaryCensored, x::Real)
    result = 1 - cdf(d, x)
    return result
end

function logccdf(d::ScalarPrimaryCensored, x::Real)
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
function pdf(d::ScalarPrimaryCensored, x::Real)
    return exp(logpdf(d, x))
end

@doc "

Compute the log probability density function using numerical differentiation
of the log CDF.

See also: [`pdf`](@ref), [`logcdf`](@ref)
"
logpdf(d::ScalarPrimaryCensored, x::Real) = _marginal_logpdf(d, x)

#### Quantile function using numerical optimization

@doc "

Compute the quantile (inverse CDF) using numerical optimization.

See also: [`cdf`](@ref)
"
function quantile(d::ScalarPrimaryCensored, p::Real)
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
function Base.rand(rng::AbstractRNG, d::ScalarPrimaryCensored)
    rand(rng, get_dist(d)) + rand(rng, d.primary_event)
end

function Base.rand(
        rng::AbstractRNG, d::Truncated{<:ScalarPrimaryCensored})
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
sampler(d::ScalarPrimaryCensored) = d

# ============================================================================
# Primary-event prior accessor (both formulations)
# ============================================================================

@doc "

Return the primary event time distribution.

Named to match the `get_dist` extraction tooling: where [`get_dist`](@ref)
returns the delay distribution, `get_primary_event` returns the primary event
distribution. This is the prior over the latent primary time; under the
[`Latent`](@ref) formulation the sampler draws the primary from it (the user
does not pass the primary in).

# Arguments
- `d`: A primary-censored distribution.
- `secondary`: (optional) One secondary time, or a tuple/vector of several,
  bounding the primary window from above, for the coupled case.

# Uncoupled case
`get_primary_event(d)` returns the distribution's own `primary_event`.

# Coupled case
`get_primary_event(d, secondary)` returns a [`BoundedPrimary`](@ref): a bounded
primary distribution whose window upper is `min(lower + width, secondary)` (so
the implied delay stays non-negative), with the `log(upper - lower)` Jacobian
that restores the implicit uniform-over-window prior. This is the coupled
[`Latent`](@ref) case where the sampler draws the primary within a window bound
by other sampler-owned event times. When several secondaries are passed the
upper is `min(lower + width, secondaries...)` (the earliest binds), which is the
multi-secondary case in the bdbv joint fit where, for example, admission is
bounded by both death and discharge. This requires a `Uniform` primary event
(the Jacobian is Uniform-only) and errors otherwise.

# Examples
```@example
using CensoredDistributions, Distributions

d = primary_censored(LogNormal(1.5, 0.75), Uniform(0.0, 1.0))

# Uncoupled prior over the primary event time
prior = get_primary_event(d)        # === Uniform(0.0, 1.0)
p = rand(prior)

# Coupled prior bounded by a sampled secondary time
coupled = get_primary_event(d, 0.6) # bounded to [0.0, 0.6]

# Coupled prior bounded by the earliest of several secondaries
coupled2 = get_primary_event(d, (0.8, 0.5, 0.9))  # bounded to [0.0, 0.5]
```

# See also
- [`get_dist`](@ref): the matching delay-distribution accessor
- [`Latent`](@ref): the formulation that consumes this prior
- [`BoundedPrimary`](@ref): the coupled bounded prior
"
get_primary_event(d::PrimaryCensored) = d.primary_event

function get_primary_event(
        d::PrimaryCensored,
        secondary::Union{Real, Tuple, AbstractVector})
    return _coupled_primary_prior(d.primary_event, secondary)
end

@doc "

Deprecated alias for [`get_primary_event`](@ref), retained for backward
compatibility.

# Arguments
- `d`: A primary-censored distribution.
- `secondary`: (optional) One secondary time, or a tuple/vector of several,
  bounding the primary window from above.

# Examples
```@example
using CensoredDistributions, Distributions

d = primary_censored(LogNormal(1.5, 0.75), Uniform(0.0, 1.0))
prior = primary_prior(d)   # same as get_primary_event(d)
```

See also: [`get_primary_event`](@ref)
"
primary_prior(d::PrimaryCensored) = get_primary_event(d)
function primary_prior(
        d::PrimaryCensored,
        secondary::Union{Real, Tuple, AbstractVector})
    return get_primary_event(d, secondary)
end
