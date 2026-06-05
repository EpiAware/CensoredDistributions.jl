@doc "

Abstract type for the formulation method of a [`PrimaryCensored`](@ref)
distribution.

The method is a type parameter of `PrimaryCensored`, so each formulation is
type-stable. There are two formulations:

- the default (no `method` given): **marginalise** the primary event time,
  giving a univariate distribution over the scalar observed delay. This is the
  classic behaviour. It is documented to fall back to the latent formulation
  where marginalisation is not possible (no analytical solution, event times
  coupled across records, or the internal times are needed).
- [`Latent`](@ref): keep the primary event time as a sampler-owned latent
  variable, giving a multivariate distribution over `[primary, observed]`.

`Latent` is the single explicit opt-in override; marginal is simply the default.

# See also
- [`Latent`](@ref): the opt-in latent formulation
- [`primary_censored`](@ref): constructor selecting the formulation
"
abstract type AbstractPCMethod end

# Internal default method tag: marginalise the primary event time. Not a
# user-facing override (marginal is simply the default), so it is not exported;
# `_Marginal()` is the default value of the `method` keyword.
struct _Marginal <: AbstractPCMethod end

@doc "

Latent formulation method: keep the primary event time as a sampler-owned latent
variable.

Pass `method = Latent()` to [`primary_censored`](@ref) to opt into the latent
formulation. The result is a multivariate distribution over the event times
`[primary, observed]`:
- [`rand`](@ref) **produces** the internal censored times `[primary, observed]`
  (a fresh primary draw per sample);
- `logpdf([primary, observed])` is the **full self-contained joint**
  `logpdf(primary_event, primary) + logpdf(delay, observed - primary)` — the
  primary event prior lives inside the distribution's `logpdf`, so the primary
  is a free latent the sampler explores and the distribution drops into the same
  weighted likelihood loop as the marginal form.

Choose `Latent()` whenever you want the latent formulation on purpose, for
example to recover the internal event times via `rand` or to plug into a coupled
model, regardless of whether marginalisation would also have been possible.

# See also
- [`primary_censored`](@ref): constructor
- [`get_primary_event`](@ref): the primary event distribution the sampler draws
"
struct Latent <: AbstractPCMethod end
