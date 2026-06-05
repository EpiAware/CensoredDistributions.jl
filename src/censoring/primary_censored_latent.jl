# Latent (multivariate, sampler-owned) interface for `PrimaryCensored`. The
# default (marginal) univariate scalar interface lives in `PrimaryCensored.jl`.

# Joint log density of the sampler-owned latent event times [p, y]: the primary
# event prior plus the shifted delay density at the implied delay y - p. Used by
# the Latent formulation, where `p` is produced by `rand` and owned by the
# sampler (never passed in as data).
function _latent_logpdf(d::PrimaryCensored, p::Real, y::Real)
    logpdf(d.primary_event, p) + logpdf(get_dist(d), y - p)
end

# Test whether `x = [primary, observed]` is in the joint support: the primary in
# the primary-event support and the implied delay `observed - primary` in the
# delay support.
function _event_insupport(d::PrimaryCensored, x::AbstractVector)
    length(x) == 2 || return false
    p = x[1]
    y = x[2]
    return insupport(d.primary_event, p) && insupport(get_dist(d), y - p)
end

# ----------------------------------------------------------------------------
# Latent (force sample/score-always) — multivariate over [primary, observed]
# ----------------------------------------------------------------------------

@doc "

Length of the latent event-time vector, `[primary, observed]` (always 2).
"
Base.length(::LatentPrimaryCensored) = 2
# rand returns [primary, observed], so promote both component element types.
function Base.eltype(::Type{<:LatentPrimaryCensored{D, P}}) where {D, P}
    promote_type(eltype(D), eltype(P))
end

@doc "

Test whether `x = [primary, observed]` is in the latent joint support.

See also: [`logpdf`](@ref)
"
insupport(d::LatentPrimaryCensored, x::AbstractVector) = _event_insupport(d, x)

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

# In-place sampling for the multivariate Latent interface.
function Distributions._rand!(
        rng::AbstractRNG, d::LatentPrimaryCensored, x::AbstractVector)
    p = rand(rng, d.primary_event)
    x[1] = p
    x[2] = p + rand(rng, get_dist(d))
    return x
end

@doc """

Joint log density over the latent event times `x = [primary, observed]`,

```math
\\log f(p, y) = \\log f_\\mathrm{primary}(p) + \\log f_\\mathrm{delay}(y - p),
```

the sum of the primary event prior density and the delay density at the implied
delay `y - p`.

The primary event time `p` is a **latent variable owned by the sampler**, not
data: [`rand`](@ref) produces the whole `[primary, observed]` vector, and a
probabilistic-programming sampler that proposes `p` evaluates this joint on its
own proposed values. The user never passes `p` in. In a model the whole vector
is scored in one statement,

```julia
[primary, observed] ~ primary_censored(delay, primary_event; mode = Latent())
```

Use the default (marginal) formulation when the primary should instead be
integrated out.

See also: [`rand`](@ref), [`primary_censored`](@ref)
"""
function logpdf(d::LatentPrimaryCensored, x::AbstractVector)
    p = x[1]
    p === missing && throw(ArgumentError(
        "Latent() keeps the primary event as a sampler-owned latent variable " *
        "and scores a concrete primary; use the default (marginal) " *
        "formulation to integrate the primary out instead"))
    return _latent_logpdf(d, p, x[2])
end

@doc "

Joint density over the latent event times `[primary, observed]`.

See also: [`logpdf`](@ref)
"
pdf(d::LatentPrimaryCensored, x::AbstractVector) = exp(logpdf(d, x))
