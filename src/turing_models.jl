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

Available only when `DynamicPPL` (or `Turing`) is loaded; the `@model` methods
are provided by the `CensoredDistributionsDynamicPPLExt` extension.

The submodel reads the distribution's already-resolved formulation `mode` (owned
by the struct) and dispatches on it; it does not decide the mode. A
[`Marginal`](@ref) `d` scores the pure `logpdf(d, y)` through a `~` statement,
with the primary integrated out inside `logpdf`. A [`Latent`](@ref) `d` declares
the latent primary event INSIDE the submodel (`p ~ get_primary_event(d)`), scores
the conditional delay given `p`, and returns the internal event times. Switch a
model between the two paths by flipping only the `mode` on `d` when building it.

# Arguments
- `d`: A primary-censored distribution carrying the resolved formulation `mode`.
- `y`: The observed delay.

# Keyword Arguments
- `weight`: Optional multiplicity weight for the marginal path, applied via the
  [`weight`](@ref) distribution wrapper (`weight * logpdf`). The latent path
  ignores it and vectorises over records instead.

# Examples
```@example
using CensoredDistributions, Distributions, DynamicPPL
using DynamicPPL: to_submodel

d = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))

@model function fit(d, y)
    inner ~ to_submodel(primary_censored_model(d, y), false)
end

logjoint(fit(d, 2.0), (;))   # equals logpdf(d, 2.0) for the marginal default
```

# See also
- [`primary_censored`](@ref): builds `d` and resolves its `mode`.
- [`Marginal`](@ref), [`Latent`](@ref): the formulation modes.
- [`interval_censored_model`](@ref), [`double_interval_censored_model`](@ref).
"
function primary_censored_model end

@doc "

Construct a [`DynamicPPL`](https://github.com/TuringLang/DynamicPPL.jl) submodel
for an [`interval_censored`](@ref) distribution.

Available only when `DynamicPPL` (or `Turing`) is loaded; the `@model` method is
provided by the `CensoredDistributionsDynamicPPLExt` extension.

Interval censoring is a univariate marginal operation (the continuous value is
integrated over the containing interval inside `logpdf`), so the submodel scores
`logpdf(d, y)` through a `~` statement.

# Arguments
- `d`: An interval-censored distribution.
- `y`: The observed (interval-censored) value.

# Keyword Arguments
- `weight`: Optional multiplicity weight, applied via the [`weight`](@ref)
  distribution wrapper (`weight * logpdf`).

# Examples
```@example
using CensoredDistributions, Distributions, DynamicPPL
using DynamicPPL: to_submodel

d = interval_censored(Normal(5, 2), 1.0)

@model function fit(d, y)
    inner ~ to_submodel(interval_censored_model(d, y), false)
end

logjoint(fit(d, 5.0), (;))   # equals logpdf(d, 5.0)
```

# See also
- [`interval_censored`](@ref): builds `d`.
- [`primary_censored_model`](@ref), [`double_interval_censored_model`](@ref).
"
function interval_censored_model end

@doc "

Construct a [`DynamicPPL`](https://github.com/TuringLang/DynamicPPL.jl) submodel
for a [`double_interval_censored`](@ref) distribution.

Available only when `DynamicPPL` (or `Turing`) is loaded; the `@model` method is
provided by the `CensoredDistributionsDynamicPPLExt` extension.

Under the default [`Marginal`](@ref) formulation the composed pipeline (primary
censoring, optional truncation, optional secondary interval censoring) is
univariate and AD-safe, so the submodel scores `logpdf(d, y)` through a `~`
statement.

# Arguments
- `d`: A double-interval-censored distribution (the composed object).
- `y`: The observed value.

# Keyword Arguments
- `weight`: Optional multiplicity weight, applied via the [`weight`](@ref)
  distribution wrapper (`weight * logpdf`).

# Examples
```@example
using CensoredDistributions, Distributions, DynamicPPL
using DynamicPPL: to_submodel

d = double_interval_censored(LogNormal(1.5, 0.75); upper = 10, interval = 1)

@model function fit(d, y)
    inner ~ to_submodel(double_interval_censored_model(d, y), false)
end

logjoint(fit(d, 3.0), (;))   # equals logpdf(d, 3.0)
```

# See also
- [`double_interval_censored`](@ref): builds `d`.
- [`primary_censored_model`](@ref), [`interval_censored_model`](@ref).
"
function double_interval_censored_model end
