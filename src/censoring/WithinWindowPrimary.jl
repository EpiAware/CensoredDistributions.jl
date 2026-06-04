@doc raw"""

Bounded within-window prior on a latent primary event time.

This is the building block consumed by the `Latent` formulation of
`primary_censored` / `double_interval_censored` (tracked in #299): the
`Latent` formulation samples the primary event time `p` and scores the delay as
the deterministic difference, and `WithinWindowPrimary` supplies the bounded
prior on `p` (with the Jacobian) that makes that sampled prior match the
`Marginal` formulation's implicit prior. It is not a standalone user-facing
mode on its own; users select the representation through the formulation method
rather than constructing this directly.

A primary-censored delay built with a uniform primary window of width ``w``
implies that the latent primary event time is uniform over its window,
independent of the secondary (observed) time. When the latent times are sampled
explicitly the ordering constraint (the primary event precedes the secondary
time, so the delay is non-negative) is awkward for samplers at the wedge corner.
Following the reparametrisation used in [park2024estimating](@cite) (and the
`bdbv_model` joint fit) the primary time is instead drawn from a window bounded
above by the secondary time,

```math
T_\mathrm{secondary} \sim \mathrm{Uniform}(L_s, L_s + w), \qquad
T_\mathrm{primary}   \sim \mathrm{Uniform}(L_p, \min(L_p + w, T_\mathrm{secondary})).
```

Sampling on this bounded support changes the implied prior, so a Jacobian
correction ``+\log(\mathrm{upper} - L_p)`` (with
``\mathrm{upper} = \min(L_p + w, T_\mathrm{secondary})``) is added to `logpdf`.
This restores the implicit independent-uniform-over-window prior of the
equivalent marginalised model: `logpdf` of this distribution equals
`logpdf(Uniform(lower, lower + width), t)` for any admissible `t`.

# See also
- [`within_window_primary`](@ref): Constructor function
- [`primary_censored`](@ref): The marginalised counterpart
"""
struct WithinWindowPrimary{T <: Real} <: UnivariateDistribution{Continuous}
    "Lower edge of the primary event window."
    lower::T
    "Width of the primary event window."
    width::T
    "The secondary (observed) time bounding the primary window from above."
    secondary::T

    function WithinWindowPrimary(lower::T, width::T, secondary::T) where {T <: Real}
        width > 0 || throw(ArgumentError("Window width must be positive"))
        secondary >= lower ||
            throw(ArgumentError("secondary must be >= lower (non-negative delay)"))
        new{T}(lower, width, secondary)
    end
end

@doc raw"""

Create a bounded within-window prior on a latent primary event time.

This is the building block used by the `Latent` formulation of
`primary_censored` / `double_interval_censored` (#299), which supplies the
sampled primary time `p`; it is not intended as a standalone user mode.

The primary event time is drawn from `Uniform(lower, min(lower + width,
secondary))`. The `logpdf` includes the ``\log(\mathrm{upper} - L_p)`` Jacobian
correction so that it reproduces the implicit `Uniform(lower, lower + width)`
prior of the marginalised primary-censored model (see
[`WithinWindowPrimary`](@ref) for the mathematical background).

# Arguments
- `lower`: Lower edge of the primary event window (often the integer day).
- `width`: Width of the primary event window (e.g. `1.0` for a daily window).
- `secondary`: The secondary (observed/sampled) time, which bounds the primary
  window from above so the implied delay is non-negative.

# Examples
```@example
using CensoredDistributions, Distributions

# Daily primary window, secondary observed at 0.7 within the same day
d = within_window_primary(0.0, 1.0, 0.7)

# Sample a latent primary event time within its bounded window
t = rand(d)

# logpdf reproduces the implicit independent-uniform prior of the
# marginalised model (Jacobian included)
logpdf(d, 0.3) ≈ logpdf(Uniform(0.0, 1.0), 0.3)
```

# See also
- [`WithinWindowPrimary`](@ref): The underlying distribution type
- [`primary_censored`](@ref): The marginalised counterpart
"""
function within_window_primary(lower::Real, width::Real, secondary::Real)
    l, w, s = promote(lower, width, secondary)
    return WithinWindowPrimary(l, w, s)
end

# Upper edge of the bounded primary window.
_upper(d::WithinWindowPrimary) = min(d.lower + d.width, d.secondary)

params(d::WithinWindowPrimary) = (d.lower, d.width, d.secondary)
Base.eltype(::Type{<:WithinWindowPrimary{T}}) where {T} = T
minimum(d::WithinWindowPrimary) = d.lower
maximum(d::WithinWindowPrimary) = _upper(d)
insupport(d::WithinWindowPrimary, x::Real) = d.lower <= x <= _upper(d)

@doc raw"""

Compute the log density of the latent primary event time.

Returns the bounded-Uniform log density plus the ``\log(\mathrm{upper} - L_p)``
Jacobian correction, which equals
`logpdf(Uniform(lower, lower + width), x)` for admissible `x`. This makes the
sampled latent prior match the marginalised primary-censored model's implicit
prior.

See also: [`pdf`](@ref)
"""
function logpdf(d::WithinWindowPrimary, x::Real)
    if !insupport(d, x)
        return oftype(float(x), -Inf)
    end
    # Bounded-uniform logpdf is -log(upper - lower); the Jacobian adds
    # +log(upper - lower), leaving the flat implicit prior -log(width).
    return -log(d.width)
end

@doc raw"""

Compute the density of the latent primary event time.

See also: [`logpdf`](@ref)
"""
pdf(d::WithinWindowPrimary, x::Real) = exp(logpdf(d, x))

@doc raw"""

Compute the cumulative distribution function of the bounded primary window.

The CDF is that of the bounded sampling support `Uniform(lower, upper)`; the
Jacobian correction in [`logpdf`](@ref) affects only the scored density, not
the support over which the latent time is drawn.

See also: [`logcdf`](@ref)
"""
function cdf(d::WithinWindowPrimary, x::Real)
    u = _upper(d)
    x <= d.lower && return zero(float(x))
    x >= u && return one(float(x))
    return (x - d.lower) / (u - d.lower)
end

@doc raw"""

Compute the log cumulative distribution function.

See also: [`cdf`](@ref)
"""
logcdf(d::WithinWindowPrimary, x::Real) = log(cdf(d, x))

@doc raw"""

Sample a latent primary event time uniformly from the bounded window
`[lower, min(lower + width, secondary)]`.

See also: [`logpdf`](@ref)
"""
function Base.rand(rng::AbstractRNG, d::WithinWindowPrimary{T}) where {T}
    u = _upper(d)
    return d.lower + (u - d.lower) * rand(rng, T)
end

sampler(d::WithinWindowPrimary) = d
