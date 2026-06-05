@doc "

Abstract type for the formulation method of a [`PrimaryCensored`](@ref)
distribution.

The method is a type parameter of `PrimaryCensored`, so each formulation is
type-stable and dispatches without runtime branching. Two concrete methods are
provided:

- [`Marginal`](@ref) (default): integrate the primary event out, giving a
  univariate distribution over the scalar observed delay.
- [`Latent`](@ref): keep the primary event explicit, giving a multivariate
  distribution over the event times `[primary, observed]`.

This mirrors the `AnalyticalSolver`/`NumericSolver` solver-method pattern.

# See also
- [`Marginal`](@ref), [`Latent`](@ref): the concrete methods
- [`primary_censored`](@ref): constructor selecting the method
"
abstract type AbstractPCMethod end

@doc "

Marginal formulation method (default).

Integrates the latent primary event time out, so a [`PrimaryCensored`](@ref)
built with `Marginal` is a univariate distribution over the scalar observed
delay. The cdf is the convolution of the delay with the primary event
distribution, computed by [`primarycensored_cdf`](@ref).

# See also
- [`Latent`](@ref): the multivariate, data-augmentation counterpart
"
struct Marginal <: AbstractPCMethod end

@doc "

Latent formulation method (data augmentation).

Keeps the primary event time explicit, so a [`PrimaryCensored`](@ref) built with
`Latent` is a multivariate distribution over the event times
`[primary, observed]`. `rand` returns both event times (a fresh primary draw per
sample); `logpdf([p, y])` is the combined joint density
`logpdf(primary_event, p) + logpdf(delay, y - p)`. The sampler owns the primary
draw, so the density stays deterministic.

# See also
- [`Marginal`](@ref): the univariate, integrate-out counterpart
- [`primary_prior`](@ref): the prior over the primary event time
"
struct Latent <: AbstractPCMethod end
