@doc "

Abstract type for the formulation mode of a [`PrimaryCensored`](@ref)
distribution.

The mode is a type parameter of `PrimaryCensored`, so each formulation is
type-stable. Two concrete modes are provided and passed via the `mode` keyword
of [`primary_censored`](@ref):

- [`Marginal`](@ref) (the default): marginalise the primary event time **inside
  `logpdf`** (analytically or by deterministic quadrature, AD-safe, with no
  exposed `p`), giving a univariate distribution over the scalar observed delay,
  and automatically fall back to the latent formulation for any component that
  genuinely cannot be marginalised.
- [`Latent`](@ref): force the latent formulation always, keeping the primary
  event time as a sampler-owned latent variable (the only mode that exposes the
  primary to the sampler).

# See also
- [`Marginal`](@ref), [`Latent`](@ref): the concrete modes
- [`primary_censored`](@ref): constructor selecting the mode
"
abstract type AbstractPCMethod end

@doc "

Marginal formulation mode (the default).

Pass `mode = Marginal()` (or omit it) to [`primary_censored`](@ref). The primary
event time is **marginalised** (integrated out), giving the classic univariate
distribution over the scalar observed delay with the full scalar interface
(`cdf`, `quantile`, `rand`, `mean`, truncation, ...).

# Contract: the primary is integrated out *inside* `logpdf`

`logpdf(d, observed)` under `Marginal` integrates the primary event time (and any
unobserved internal event) out **internally**, analytically where possible and
otherwise by **deterministic quadrature** (a fixed Gauss-Legendre rule, never
fresh random draws), so the density is exact and automatic-differentiation safe.
The user never has to expose the primary: no `Flat()` prior, no `p ~ ...`
statement, and no `if`/`else` branching on missingness. The marginal path is the
zero-ceremony default.

[`Latent`](@ref) is the only mode that exposes the primary to the sampler, used
deliberately when the internal times are wanted or marginalisation is not
possible.

# Auto-fallback

`Marginal` **auto-falls-back to [`Latent`](@ref)** at construction for any
component that genuinely cannot be marginalised (no analytical or quadrature
route, event times coupled across records, or the internal times are needed).
This fallback is real runtime behaviour, not just documentation: such a
component is built as the multivariate latent form instead. For a single delay
the marginal route always exists (quadrature), so `Marginal` stays univariate.

# See also
- [`Latent`](@ref): force the latent formulation always (exposes the primary)
- [`primary_censored`](@ref): constructor
"
struct Marginal <: AbstractPCMethod end

@doc "

Latent formulation mode (force latent always).

Pass `mode = Latent()` to [`primary_censored`](@ref) to force the latent
formulation regardless of whether marginalisation would also have been possible.
The result is a multivariate distribution over the event times
`[primary, observed]`:
- [`rand`](@ref) **produces** the internal censored times `[primary, observed]`
  (a fresh primary draw per sample);
- `logpdf([primary, observed])` is the **full self-contained joint**
  `logpdf(primary_event, primary) + logpdf(delay, observed - primary)` — the
  primary event prior lives inside the distribution's `logpdf`, so the primary
  is a free latent the sampler proposes and explores, and the distribution drops
  into the same weighted likelihood loop as the marginal form.

Choose `Latent()` whenever you want the latent formulation on purpose, for
example to recover the internal event times via `rand` or to plug into a coupled
model.

# See also
- [`Marginal`](@ref): the marginalise-with-fallback default
- [`primary_censored`](@ref): constructor
- [`get_primary_event`](@ref): the primary event distribution the sampler draws
"
struct Latent <: AbstractPCMethod end
