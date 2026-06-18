# ============================================================================
# Batched record-aware `rand`: the forward-simulation dual to scoring
# ============================================================================
#
# The scoring path has a batched record entry: `record_distributions(d, rows)`
# builds per-record composed distributions (routing each record to its Choose
# alternative by the `:kind` selector, carrying per-record covariates and
# horizons), and `batched_event_logpdf` / `composed_distribution_model` score
# them. This is the GENERATIVE dual: `rand(d, rows)` draws one full event path
# per record, reusing the SAME per-record machinery. Each record's draw is
# labelled by that record's event names, so the result is a self-describing,
# Tables-friendly vector of NamedTuples (one per row).

# A single record's draw, labelled by that record's event names. An
# `EventRecord` and a composer `_GenericRecord` are keyed by the resolved tree's
# flat event names (`_flat_event_names`); a univariate-leaf record is one event,
# keyed `:value`. The path comes from the record's own `rand` (the SAME draw the
# scoring-path record samples), so a batched draw of a row equals
# `rand(record_distributions(d, rows)[i])` under the same rng.
function _record_named_draw(rng::AbstractRNG, r)
    path = rand(rng, r)
    return _as_named(_record_event_names(r), path)
end

# The event names labelling one record's drawn path. An `EventRecord` carries
# the shared composed distribution; a `_GenericRecord` carries the record's
# resolved tree (a composer, or a routed univariate-leaf alternative). A
# composer is named by its flat event names; a univariate leaf is one event
# named `:value`.
_record_event_names(r::EventRecord) = _flat_event_names(r.dist)
_record_event_names(r::_GenericRecord) = _generic_record_event_names(r.dist)
function _generic_record_event_names(d::Union{Sequential, Parallel})
    return _flat_event_names(d)
end
_generic_record_event_names(::UnivariateDistribution) = (:value,)

@doc "

Draw one labelled event path per record, the generative dual to scoring.

`rand(d, rows)` is the batched record-aware sampler: it builds the per-record
composed distributions with [`record_distributions`](@ref)`(d, rows)` (so each
record routes to its [`Choose`](@ref) alternative by the row's selector and
carries that row's covariates / horizon) and draws ONE full event path from
each. The result is a vector with one entry per row, each a `NamedTuple` keyed
by that record's event names (a univariate-leaf record is keyed `:value`), so
the draws are self-describing and Tables-friendly. This mirrors the scoring
path: where [`batched_event_logpdf`](@ref)`(d, rows)` scores a table,
`rand(d, rows)` generates one. The batched draw of a row equals the single
`rand` of that row's `record_distributions` entry under the same `rng`.

For `n` independent iid paths from the shared distribution (no per-record
covariates) use the standard multivariate `rand(d, n)`, which returns the
event-path matrix.

# Arguments
- `rng`: an optional random number generator (defaults to `default_rng()`).
- `d`: the shared composed distribution (a [`Sequential`](@ref), a
  [`Parallel`](@ref), or a [`Choose`](@ref)).
- `rows`: a Tables.jl row source of records (only the selector / covariate
  fields are read; event values are sampled).

# Examples
```@example
using CensoredDistributions, Distributions, Random

seq = Sequential(
    primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
    primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
rows = [(event_1 = missing, event_2 = missing, event_3 = missing)
        for _ in 1:3]
rand(Random.Xoshiro(1), seq, rows)
```

# See also
- [`record_distributions`](@ref): the per-record assembly this draws from.
- [`batched_event_logpdf`](@ref): the scoring dual over the same records.
"
# Restricted to the composer types this package OWNS (a method over a bare
# `UnivariateDistribution` from Distributions would be type piracy); a
# single-delay model wraps its leaf in a one-edge `Sequential` or uses the
# standard `rand(leaf, n)`.
function Base.rand(rng::AbstractRNG, d::Union{Sequential, Parallel, Choose},
        rows::AbstractVector)
    recs = record_distributions(d, rows)
    return [_record_named_draw(rng, r) for r in recs]
end

function Base.rand(d::Union{Sequential, Parallel, Choose}, rows::AbstractVector)
    return rand(default_rng(), d, rows)
end
