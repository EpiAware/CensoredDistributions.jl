@doc "

Abstract type for the formulation method of a [`PrimaryCensored`](@ref)
distribution.

The method is a type parameter of `PrimaryCensored`, so each formulation is
type-stable. Three concrete methods are provided:

- [`Auto`](@ref) (default): a multivariate distribution over the event times
  `[primary, observed]` that dispatches on the observation. A missing primary
  marginalises (the quadrature path); a concrete primary conditions on it. A
  scalar observed time also takes the marginal path.
- [`Marginal`](@ref): force the marginal (integrate-always) formulation, a
  univariate distribution over the scalar observed delay.
- [`Latent`](@ref): force the latent (condition/sample-always) formulation, a
  multivariate distribution over `[primary, observed]`.

`Marginal` and `Latent` are the explicit force overrides; `Auto` is the
missingness-dispatch default. This mirrors the `force_numeric` override of the
solver-method selection.

# See also
- [`Auto`](@ref), [`Marginal`](@ref), [`Latent`](@ref): the concrete methods
- [`primary_censored`](@ref): constructor selecting the method
"
abstract type AbstractPCMethod end

@doc "

Auto formulation method (default).

Gives a [`PrimaryCensored`](@ref) that is multivariate over the event times
`[primary, observed]` and dispatches on the observation:
- `logpdf([missing, y])` marginalises the primary (the quadrature path),
- `logpdf([p, y])` conditions on the concrete primary `p`,
- `logpdf(d, y)` with a scalar observed time also takes the marginal path.

The missingness is inspected through control flow only; concrete values alone
enter the differentiated arithmetic, so the log density differentiates on every
supported automatic-differentiation backend.

# See also
- [`Marginal`](@ref), [`Latent`](@ref): the explicit force overrides
"
struct Auto <: AbstractPCMethod end

@doc "

Marginal formulation method (force integrate-always).

Forces the marginal formulation, so a [`PrimaryCensored`](@ref) built with
`Marginal` is a univariate distribution over the scalar observed delay. The cdf
is the convolution of the delay with the primary event distribution, computed by
[`primarycensored_cdf`](@ref).

# See also
- [`Auto`](@ref): the missingness-dispatch default
- [`Latent`](@ref): the multivariate, condition-always counterpart
"
struct Marginal <: AbstractPCMethod end

@doc "

Latent formulation method (force condition/sample-always).

Forces the latent formulation, so a [`PrimaryCensored`](@ref) built with `Latent`
is a multivariate distribution over the event times `[primary, observed]`. `rand`
returns both event times (a fresh primary draw per sample); `logpdf([p, y])` is
the combined joint density `logpdf(primary_event, p) + logpdf(delay, y - p)`. The
sampler owns the primary draw, so the density stays deterministic.

# See also
- [`Auto`](@ref): the missingness-dispatch default
- [`Marginal`](@ref): the univariate, integrate-always counterpart
- [`primary_prior`](@ref): the prior over the primary event time
"
struct Latent <: AbstractPCMethod end
