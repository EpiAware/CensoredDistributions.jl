# DynamicPPL submodel constructors.
#
# These are declared (but given no methods) in the core package so they can be
# exported and referenced, and so the docstrings live with the rest of the API.
# The actual `@model` methods are added by the DynamicPPL weak-dependency
# extension (`ext/CensoredDistributionsDynamicPPLExt.jl`), which loads only when
# the user brings in `DynamicPPL` (or `Turing`, which re-exports it). The base
# package therefore stays Turing-free.
#
# The MARGINAL path needs no extension to be usable for inference: `logpdf(d, y)`
# on a marginal object is already deterministic, AD-safe, and sampler-agnostic,
# so it composes with any log-density consumer directly. These submodels exist
# for the LATENT ergonomics — declaring the latent primary event with its own
# `~` inside the submodel so it never appears in the user's model — and to give
# the marginal path a matching submodel form for uniform composition.
#
# See issue #88 for the locked design.

@doc "

Construct a [`DynamicPPL`](https://github.com/TuringLang/DynamicPPL.jl) submodel
for a [`primary_censored`](@ref) distribution.

Available only when `DynamicPPL` (or `Turing`) is loaded; provided by the
`CensoredDistributionsDynamicPPLExt` extension.

The submodel reads the distribution's already-resolved formulation `mode` (owned
by the struct) and dispatches on it: a [`Marginal`](@ref) `d` scores the pure
`logpdf(d, y)`, while a [`Latent`](@ref) `d` declares the latent primary event
inside the submodel and scores the conditional delay. Switch a model between the
two paths by flipping only the `mode` on `d` when building it.

See the extension method for full documentation and examples.

# See also
- [`primary_censored`](@ref), [`interval_censored_model`](@ref),
  [`double_interval_censored_model`](@ref).
"
function primary_censored_model end

@doc "

Construct a [`DynamicPPL`](https://github.com/TuringLang/DynamicPPL.jl) submodel
for an [`interval_censored`](@ref) distribution.

Available only when `DynamicPPL` (or `Turing`) is loaded; provided by the
`CensoredDistributionsDynamicPPLExt` extension.

Interval censoring is a univariate marginal operation, so the submodel scores
`logpdf(d, y)` (weighted via the [`weight`](@ref) wrapper when a multiplicity
weight is supplied).

# See also
- [`interval_censored`](@ref), [`primary_censored_model`](@ref),
  [`double_interval_censored_model`](@ref).
"
function interval_censored_model end

@doc "

Construct a [`DynamicPPL`](https://github.com/TuringLang/DynamicPPL.jl) submodel
for a [`double_interval_censored`](@ref) distribution.

Available only when `DynamicPPL` (or `Turing`) is loaded; provided by the
`CensoredDistributionsDynamicPPLExt` extension.

Under the default [`Marginal`](@ref) formulation the composed pipeline is
univariate, so the submodel scores `logpdf(d, y)` (weighted via the
[`weight`](@ref) wrapper when a multiplicity weight is supplied).

# See also
- [`double_interval_censored`](@ref), [`primary_censored_model`](@ref),
  [`interval_censored_model`](@ref).
"
function double_interval_censored_model end
