@doc "

Abstract type for the formulation of a primary-censored delay.

A primary-censored delay can be expressed in two equivalent ways. The
[`Marginal`](@ref) formulation integrates the latent primary event time out
analytically (or by quadrature). The [`Latent`](@ref) formulation conditions on
a concrete sampled primary event time `p`, turning the delay into the
deterministic difference `x - p` (data augmentation).

The two formulations share `rand`, support, `params` and the like; they differ
only in the cdf/logcdf path, which `pdf`/`logpdf` then cascade from.

# See also
- [`Marginal`](@ref): integrate the primary event (default)
- [`Latent`](@ref): condition on a sampled primary event
- [`primary_prior`](@ref): the prior over the latent primary `p`
"
abstract type AbstractFormulation end

@doc "

Marginal formulation: integrate the latent primary event time out.

This is the default. The cdf is the convolution of the delay distribution with
the primary event distribution over the primary window, computed by
[`primarycensored_cdf`](@ref) (analytical where available, quadrature
otherwise).

# See also
- [`Latent`](@ref): the data-augmentation counterpart
"
struct Marginal <: AbstractFormulation end

@doc """

Latent formulation: condition on a concrete sampled primary event time `p`.

Given a sampled primary event time `p`, the observed delay is the deterministic
difference, so the conditional cdf collapses to ``F_\\mathrm{delay}(x - p)`` and
the logpdf to ``\\log f_\\mathrm{delay}(x - p)`` exactly, with no quadrature and
no finite differencing.

`p` is supplied per draw by sampling the prior returned by
[`primary_prior`](@ref). The user writes, in their own PPL,

```julia
p ~ primary_prior(d)
y ~ <Latent-formulation delay conditioned on p>
```

so the package never depends on any particular PPL.

# See also
- [`Marginal`](@ref): the integrate-out counterpart
- [`primary_prior`](@ref): the prior over `p`
"""
struct Latent{P <: Real} <: AbstractFormulation
    "The sampled primary event time the delay is conditioned on."
    p::P
end
