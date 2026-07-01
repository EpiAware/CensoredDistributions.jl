@doc "

Distribution of a difference of two independent random variables.

`Difference` represents ``Z = X - Y`` where `X` and `Y` are independent
univariate distributions. It is the dual of [`Convolved`](@ref) (the sum
``X + Y``): a convolution of `X` with the reflection of `Y`. Where a
convolution gathers two delays into one longer gap, a difference is the
signed gap between two events, so it arises as a derived observation
(for example the offset between two independently timed measurements)
rather than as a delay leaf.

Because the subtraction reflects `Y`, the support of `Z` is in general
two-sided and can be negative: it runs from ``\\min(X) - \\max(Y)`` to
``\\max(X) - \\min(Y)``, taking the value ``\\pm\\infty`` where a component
is unbounded. `Z` is therefore not a non-negative delay distribution, and
the delay-oriented censoring wrappers ([`primary_censored`](@ref),
[`interval_censored`](@ref)) — which assume a non-negative support — do
not apply to it the way they do to a delay leaf. Treat a `Difference` as
an observation or derived quantity, not as a leaf to censor.

# Independence

The construction assumes `X` and `Y` are independent. The density, CDF,
mean and variance below all rely on this; they are not correct for
dependent components.

# Density and CDF computation

The density is the cross-correlation of the two component densities:

```math
f_Z(z) = \\int f_X(z + y)\\, f_Y(y)\\, \\mathrm{d}y
```

and the CDF integrates `X`'s CDF against `Y`'s density:

```math
F_Z(z) = \\int F_X(z + y)\\, f_Y(y)\\, \\mathrm{d}y .
```

For a `Normal`-`Normal` pair the closed form
``\\mathrm{Normal}(\\mu_X - \\mu_Y, \\sqrt{\\sigma_X^2 + \\sigma_Y^2})`` is
used directly unless a [`NumericSolver`](@ref) method is set. All other
cases use AD-safe fixed-node Gauss-Legendre quadrature (`gl_integrate`),
the same construction [`Convolved`](@ref) uses: the integral is mapped
from the fixed reference domain ``(-1, 1)`` onto the integration bounds
inside the integrand and reduced as a bare weighted dot product, so every
AD backend specialises on the integrand's own type and component `Dual`s
and tangents propagate.

The `method` field selects the backend: an [`AnalyticalSolver`](@ref) (the
default) uses the analytic difference where one exists and falls back to
quadrature otherwise, while a [`NumericSolver`](@ref) forces the numeric
path even for a `Normal`-`Normal` pair (useful for validation).

# See also
- [`difference`](@ref): Constructor function
- [`Convolved`](@ref): The dual sum ``X + Y``
"
struct Difference{X <: UnivariateDistribution, Y <: UnivariateDistribution,
    M <: AbstractSolverMethod} <:
       AbstractCombinedDistribution{Univariate, Continuous}
    "The minuend component (the `X` in `Z = X - Y`)."
    x::X
    "The subtrahend component (the `Y` in `Z = X - Y`)."
    y::Y
    "Solver method choose the analytic vs numeric quadrature backend."
    method::M

    function Difference(x::X, y::Y;
            method::AbstractSolverMethod = AnalyticalSolver()) where {
            X <: UnivariateDistribution, Y <: UnivariateDistribution}
        new{X, Y, typeof(method)}(x, y, method)
    end
end

@doc "

Create the distribution of a difference of two independent variables.

Returns a [`Difference`](@ref) representing ``Z = X - Y``. This is the dual
of [`convolved`](@ref): a convolution forms the sum of two
delays, a difference the signed gap between two events. The support of `Z`
can be negative, so `Z` is an observation or derived quantity rather than a
non-negative delay leaf (see [`Difference`](@ref) for why the delay
censoring wrappers do not apply).

`X` and `Y` are assumed independent.

# Arguments
- `x`: The minuend distribution (the `X` in `Z = X - Y`), a
  `UnivariateDistribution`.
- `y`: The subtrahend distribution (the `Y` in `Z = X - Y`), a
  `UnivariateDistribution`.

# Keyword Arguments
- `method`: The solver method, an [`AnalyticalSolver`](@ref) (the default)
  or [`NumericSolver`](@ref). `NumericSolver` forces numeric quadrature
  even for a `Normal`-`Normal` pair, mirroring `convolved`.

# Examples
```@example
using CensoredDistributions, Distributions

# Difference of two delays; mean is the difference of the means (≈ 3)
d = difference(Normal(5.0, 1.0), Normal(2.0, 1.0))
mean(d)
```

# See also
- [`Difference`](@ref): The distribution type
- [`convolved`](@ref): The dual sum ``X + Y``
"
function difference(x::UnivariateDistribution, y::UnivariateDistribution;
        method::AbstractSolverMethod = AnalyticalSolver())
    return Difference(x, y; method = method)
end

# ---------------------------------------------------------------------------
# Interface: params / support / sampling
# ---------------------------------------------------------------------------

params(d::Difference) = (params(d.x), params(d.y))

function Base.eltype(::Type{<:Difference{X, Y}}) where {X, Y}
    return promote_type(eltype(X), eltype(Y))
end

# Reflecting Y is what gives the two-sided support: the largest difference
# pairs the largest X with the smallest Y, the smallest the reverse.
minimum(d::Difference) = minimum(d.x) - maximum(d.y)
maximum(d::Difference) = maximum(d.x) - minimum(d.y)

function insupport(d::Difference, z::Real)
    return minimum(d) <= z <= maximum(d)
end

function Base.rand(rng::AbstractRNG, d::Difference)
    return rand(rng, d.x) - rand(rng, d.y)
end

sampler(d::Difference) = d

# ---------------------------------------------------------------------------
# Moments: exact for independent components
# ---------------------------------------------------------------------------
#
# Z = X - Y with X ⟂ Y, so the mean is the difference of the means and the
# variance the sum of the variances (subtraction flips the sign of the mean
# contribution but not of the variance). Both flow through the component
# parameters via the components' own analytic `mean`/`var`, so the path is
# AD-safe; a component without an analytic moment errors from its own call.

@doc "

Mean of the difference: the difference of the component means,
``\\mathbb{E}[X] - \\mathbb{E}[Y]``.

See also: [`var`](@ref), [`std`](@ref)
"
mean(d::Difference) = mean(d.x) - mean(d.y)

@doc "

Variance of the difference: the sum of the component variances,
``\\mathrm{Var}[X] + \\mathrm{Var}[Y]`` (independence makes the variance
additive even though the means subtract).

See also: [`mean`](@ref), [`std`](@ref)
"
var(d::Difference) = var(d.x) + var(d.y)

@doc "

Standard deviation of the difference, ``\\sqrt{\\mathrm{Var}[Z]}``.

See also: [`var`](@ref), [`mean`](@ref)
"
std(d::Difference) = sqrt(var(d))

# ---------------------------------------------------------------------------
# Analytical fast path for Normal - Normal
# ---------------------------------------------------------------------------

# `_try_difference` returns the analytic difference distribution when a
# closed form exists, otherwise `nothing`. Dispatch (rather than
# `try`/`catch`) selects the analytic pair so the path stays
# differentiable under every AD backend. Only Normal - Normal is enabled:
# the difference of two independent normals is normal with the means
# subtracted and the variances summed.
_try_difference(x::UnivariateDistribution, y::UnivariateDistribution) = nothing

function _try_difference(x::Normal, y::Normal)
    return Normal(mean(x) - mean(y), sqrt(var(x) + var(y)))
end

# The analytic difference to use for `d`, or `nothing` when none exists or
# when `d.method` is a `NumericSolver` requesting the numeric path.
function _maybe_analytic(d::Difference)
    d.method isa NumericSolver && return nothing
    return _try_difference(d.x, d.y)
end

# ---------------------------------------------------------------------------
# Numeric cross-correlation (AD-safe Gauss-Legendre dot product)
# ---------------------------------------------------------------------------

# Integration window over Y. Both the density and CDF integrands carry the
# factor f_Y(y), negligible outside Y's effective support, so an infinite
# endpoint is clamped to an extreme quantile of Y on AD-stripped params
# (`_window_quantile`, shared with Convolved) so the window stays a
# non-differentiated constant across every AD backend.
function _difference_window(d::Difference)
    ymin = minimum(d.y)
    ymax = maximum(d.y)
    lo = isfinite(ymin) ? ymin : _window_quantile(d.y, _CONVOLVED_TAIL)
    hi = isfinite(ymax) ? ymax : _window_quantile(d.y, 1 - _CONVOLVED_TAIL)
    return lo, hi
end

# Numeric difference density:
#   f_Z(z) = ∫ f_X(z + y) f_Y(y) dy   over y ∈ support(Y).
function _difference_numeric_pdf(d::Difference, z::Real)
    isnan(z) && return convert(float(typeof(z)), NaN)
    (z <= minimum(d) || z >= maximum(d)) && return zero(float(typeof(z)))

    lower, upper = _difference_window(d)
    upper <= lower && return zero(float(typeof(z)))

    result = gl_integrate(
        y -> pdf(d.x, z + y) * pdf(d.y, y), lower, upper)
    return max(result, zero(result))
end

# Numeric difference CDF:
#   F_Z(z) = P(X - Y ≤ z) = ∫ F_X(z + y) f_Y(y) dy   over y ∈ support(Y).
function _difference_numeric_cdf(d::Difference, z::Real)
    isnan(z) && return convert(float(typeof(z)), NaN)
    z <= minimum(d) && return zero(float(typeof(z)))
    z >= maximum(d) && return one(float(typeof(z)))

    lower, upper = _difference_window(d)
    upper <= lower && return zero(float(typeof(z)))

    result = gl_integrate(
        y -> _cdf_ad_safe(d.x, z + y) * pdf(d.y, y), lower, upper)
    return clamp(result, zero(result), one(result))
end

# ---------------------------------------------------------------------------
# CDF / logcdf / pdf / logpdf
# ---------------------------------------------------------------------------

@doc "

Compute the cumulative distribution function.

Uses the analytic `Normal`-`Normal` difference where it applies, otherwise
AD-safe numeric quadrature of ``\\int F_X(z + y) f_Y(y)\\,\\mathrm{d}y``.

See also: [`logcdf`](@ref)
"
function cdf(d::Difference, z::Real)
    analytic = _maybe_analytic(d)
    if analytic !== nothing
        return cdf(analytic, z)
    end
    return _difference_numeric_cdf(d, z)
end

@doc "

Compute the log cumulative distribution function.

See also: [`cdf`](@ref)
"
function logcdf(d::Difference, z::Real)
    analytic = _maybe_analytic(d)
    if analytic !== nothing
        return logcdf(analytic, z)
    end
    c = _difference_numeric_cdf(d, z)
    return c <= 0 ? oftype(float(c), -Inf) : log(c)
end

function ccdf(d::Difference, z::Real)
    return 1 - cdf(d, z)
end

function logccdf(d::Difference, z::Real)
    logcdf_val = logcdf(d, z)
    if logcdf_val == -Inf
        return zero(logcdf_val)
    elseif logcdf_val >= 0
        return oftype(logcdf_val, -Inf)
    end
    return log1mexp(logcdf_val)
end

@doc "

Compute the probability density function.

Uses the exact analytic `Normal`-`Normal` density where it applies,
otherwise the AD-safe numeric cross-correlation
``f_Z(z) = \\int f_X(z + y) f_Y(y)\\,\\mathrm{d}y``.

See also: [`logpdf`](@ref)
"
function pdf(d::Difference, z::Real)
    analytic = _maybe_analytic(d)
    if analytic !== nothing
        return pdf(analytic, z)
    end
    return _difference_numeric_pdf(d, z)
end

@doc "

Compute the log probability density function.

See also: [`pdf`](@ref), [`logcdf`](@ref)
"
function logpdf(d::Difference, z::Real)
    analytic = _maybe_analytic(d)
    if analytic !== nothing
        return logpdf(analytic, z)
    end
    if !insupport(d, z)
        return oftype(float(z), -Inf)
    end
    p = _difference_numeric_pdf(d, z)
    return p <= 0 ? oftype(float(z), -Inf) : log(p)
end

@doc "

Compute the quantile (inverse CDF) of the difference.

No closed form exists for a generic difference, so the quantile is found by
numerically inverting [`cdf`](@ref). The initial guess is the difference of
the component quantile and median, a good starting point for the
optimiser. Providing this method lets a `Difference` compose under
`truncated`.

See also: [`cdf`](@ref)
"
function quantile(d::Difference, p::Real)
    return _quantile_optimization(
        d, p; initial_guess_fn = _difference_quantile_guess)
end

# Difference of the X quantile and the Y median as the inversion start.
function _difference_quantile_guess(d::Difference, p::Real)
    guess = float(quantile(d.x, p)) - float(quantile(d.y, 0.5))
    return [guess]
end
