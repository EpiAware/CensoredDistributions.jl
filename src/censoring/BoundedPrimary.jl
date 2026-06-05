@doc """

Bounded within-window prior on a latent primary event time.

Internal distribution used by the [`Latent`](@ref) formulation for the coupled
case, where the primary event window is bounded above by an already-sampled
secondary time (so the implied delay stays non-negative). It is a genuine
sampleable distribution (`rand` + `logpdf`) so a user can write
`p ~ primary_prior(d, secondary)` in their own probabilistic programme. It is
internal rather than a loudly-marketed public type (see #299, which folded the
former `WithinWindowPrimary` here).

A primary-censored delay built with a uniform primary window of width ``w``
implies a uniform-over-window prior on the latent primary time, independent of
the secondary time. When the latent times are sampled explicitly the ordering
constraint (the primary precedes the secondary, so the delay is non-negative) is
awkward for samplers at the wedge corner. Following the reparametrisation used
in [park2024estimating](@cite) (and the bdbv joint fit) the primary time is
drawn from a window bounded above by the secondary time,

```math
T_\\mathrm{secondary} \\sim \\mathrm{Uniform}(L_s, L_s + w), \\qquad
T_\\mathrm{primary} \\sim
    \\mathrm{Uniform}(L_p, \\min(L_p + w, T_\\mathrm{secondary})).
```

Sampling on this bounded support changes the implied prior, so a Jacobian
correction ``+\\log(\\mathrm{upper} - L_p)`` (with
``\\mathrm{upper} = \\min(L_p + w, T_\\mathrm{secondary})``) is added to
`logpdf`. This restores the implicit independent-uniform-over-window prior of
the equivalent marginalised model: `logpdf` of this distribution equals
`logpdf(Uniform(lower, lower + width), t)` for any admissible `t`. The Jacobian
is uniform-only, so the constructor accepts only a `Uniform` primary event.

When the primary event is bounded by several secondaries (for example, in the
bdbv joint fit admission precedes both death and discharge), the binding
secondary is their minimum and ``\\mathrm{upper} = \\min(L_p + w,
T_\\mathrm{secondary}^{(1)}, T_\\mathrm{secondary}^{(2)}, \\dots)``.

# See also
- [`Latent`](@ref): the formulation that consumes this prior
- [`primary_prior`](@ref): the accessor that returns it
"""
struct BoundedPrimary{T <: Real} <: UnivariateDistribution{Continuous}
    "Lower edge of the primary event window."
    lower::T
    "Width of the primary event window."
    width::T
    "The secondary (observed) time bounding the primary window from above."
    secondary::T

    function BoundedPrimary(lower::T, width::T, secondary::T) where {T <: Real}
        width > 0 || throw(ArgumentError("Window width must be positive"))
        secondary >= lower ||
            throw(ArgumentError(
                "secondary must be >= lower (non-negative delay)"))
        new{T}(lower, width, secondary)
    end
end

# Build the coupled bounded prior from a primary-event distribution. The
# log(upper - lower) Jacobian only restores a uniform-over-window prior, so a
# non-Uniform primary event is rejected. A single secondary or several
# secondaries are accepted; the binding secondary is their minimum, so the
# window upper becomes min(lower + width, secondaries...).
function _coupled_primary_prior(primary_event::UnivariateDistribution, secondary)
    throw(ArgumentError(
        "Coupled primary_prior requires a Uniform primary event " *
        "(the bounded-primary Jacobian is uniform-only); got " *
        "$(typeof(primary_event))"))
end

function _coupled_primary_prior(primary_event::Uniform, secondary::Real)
    lower = minimum(primary_event)
    width = maximum(primary_event) - lower
    l, w, s = promote(lower, width, secondary)
    return BoundedPrimary(l, w, s)
end

# Multiple secondaries: the upper is bounded by the earliest (minimum) of them.
function _coupled_primary_prior(
        primary_event::Uniform, secondaries::Union{Tuple, AbstractVector})
    isempty(secondaries) &&
        throw(ArgumentError("At least one secondary time is required"))
    return _coupled_primary_prior(primary_event, minimum(secondaries))
end

# Upper edge of the bounded primary window.
_upper(d::BoundedPrimary) = min(d.lower + d.width, d.secondary)

params(d::BoundedPrimary) = (d.lower, d.width, d.secondary)
Base.eltype(::Type{<:BoundedPrimary{T}}) where {T} = T
minimum(d::BoundedPrimary) = d.lower
maximum(d::BoundedPrimary) = _upper(d)
insupport(d::BoundedPrimary, x::Real) = d.lower <= x <= _upper(d)

@doc """

Log density of the latent primary event time.

Returns the bounded-uniform log density plus the
``\\log(\\mathrm{upper} - L_p)`` Jacobian correction, which equals
`logpdf(Uniform(lower, lower + width), x)` for admissible `x`, so the sampled
latent prior matches the marginalised model's implicit prior.

See also: [`pdf`](@ref)
"""
function logpdf(d::BoundedPrimary, x::Real)
    if !insupport(d, x)
        return oftype(float(x), -Inf)
    end
    # Bounded-uniform logpdf is -log(upper - lower); the Jacobian adds
    # +log(upper - lower), leaving the flat implicit prior -log(width).
    return -log(d.width)
end

@doc "

Density of the latent primary event time.

See also: [`logpdf`](@ref)
"
pdf(d::BoundedPrimary, x::Real) = exp(logpdf(d, x))

@doc "

Cumulative distribution function over the bounded sampling support
`Uniform(lower, upper)`. The Jacobian in [`logpdf`](@ref) affects only the
scored density, not the support over which the latent time is drawn.

See also: [`logcdf`](@ref)
"
function cdf(d::BoundedPrimary, x::Real)
    u = _upper(d)
    x <= d.lower && return zero(float(x))
    x >= u && return one(float(x))
    return (x - d.lower) / (u - d.lower)
end

@doc "

Log cumulative distribution function.

See also: [`cdf`](@ref)
"
logcdf(d::BoundedPrimary, x::Real) = log(cdf(d, x))

@doc "

Sample a latent primary event time uniformly from the bounded window
`[lower, min(lower + width, secondary)]`.

See also: [`logpdf`](@ref)
"
function Base.rand(rng::AbstractRNG, d::BoundedPrimary{T}) where {T}
    u = _upper(d)
    return d.lower + (u - d.lower) * rand(rng, T)
end

sampler(d::BoundedPrimary) = d
