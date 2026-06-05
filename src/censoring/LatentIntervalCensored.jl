@doc """

Interval censoring applied to a [`Latent`](@ref) (multivariate)
[`PrimaryCensored`](@ref) distribution.

A latent primary-censored distribution is multivariate over the event times
`[primary, observed]`. Interval censoring (for example daily reporting) applies
only to the **observed** coordinate (the last dimension); the latent primary
event time stays continuous. This wrapper holds the underlying latent
distribution and the censoring interval, and scores

```math
\\log f(p) + \\log\\big[F_\\mathrm{delay}(\\mathrm{ceil} - p)
    - F_\\mathrm{delay}(\\mathrm{floor} - p)\\big],
```

where `[floor, ceil]` is the interval containing the observed coordinate. This
is the data-augmentation analogue of the marginal interval-censored density,
keeping the primary explicit.

# See also
- [`interval_censored`](@ref): constructor (dispatches here for a latent input)
- [`Latent`](@ref): the formulation this censors
"""
struct LatentIntervalCensored{
    D <: LatentPrimaryCensored, T} <: Distribution{Multivariate, Continuous}
    "The underlying latent (multivariate) primary-censored distribution."
    dist::D
    "The censoring interval applied to the observed coordinate (scalar width)."
    interval::T

    function LatentIntervalCensored(dist::D, interval::T) where {
            D <: LatentPrimaryCensored, T <: Real}
        interval > 0 || throw(ArgumentError("Interval width must be positive"))
        new{D, T}(dist, interval)
    end
end

@doc """

Apply interval censoring to a forced [`Latent`](@ref) (multivariate)
primary-censored distribution, censoring the observed coordinate while keeping
the primary event time continuous. Interval censoring of the default
[`Auto`](@ref) distribution uses the univariate marginal path instead.

See also: [`LatentIntervalCensored`](@ref)
"""
function interval_censored(dist::LatentPrimaryCensored, interval::Real)
    return LatentIntervalCensored(dist, interval)
end

Base.length(::LatentIntervalCensored) = 2
Base.eltype(::Type{<:LatentIntervalCensored{D}}) where {D} = eltype(D)
get_dist(d::LatentIntervalCensored) = d.dist

function insupport(d::LatentIntervalCensored, x::AbstractVector)
    return insupport(d.dist, x)
end

@doc """

Joint log density of the interval-censored latent event times
`x = [primary, observed]`: the continuous primary prior density plus the log
probability that the delay falls in the interval containing `observed - primary`.

See also: [`pdf`](@ref)
"""
function logpdf(d::LatentIntervalCensored, x::AbstractVector)
    p = x[1]
    y = x[2]
    delay = get_dist(d.dist)
    lp_primary = logpdf(d.dist.primary_event, p)
    # Interval [floor, ceil) containing the observed coordinate, shifted to the
    # delay scale by subtracting the primary event time.
    lower = floor_to_interval(y, d.interval)
    upper = lower + d.interval
    mass = cdf(delay, upper - p) - cdf(delay, lower - p)
    return lp_primary + log(max(mass, zero(mass)))
end

@doc "

Joint density of the interval-censored latent event times.

See also: [`logpdf`](@ref)
"
pdf(d::LatentIntervalCensored, x::AbstractVector) = exp(logpdf(d, x))

@doc "

Draw a latent event-time vector `[primary, observed]` with the observed
coordinate censored to its interval (the primary stays continuous).

See also: [`logpdf`](@ref)
"
function Base.rand(rng::AbstractRNG, d::LatentIntervalCensored)
    py = rand(rng, d.dist)
    py[2] = floor_to_interval(py[2], d.interval)
    return py
end

function Distributions._rand!(
        rng::AbstractRNG, d::LatentIntervalCensored, x::AbstractVector)
    py = rand(rng, d)
    x[1] = py[1]
    x[2] = py[2]
    return x
end
