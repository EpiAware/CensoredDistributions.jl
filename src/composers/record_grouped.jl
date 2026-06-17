# ---------------------------------------------------------------------------
# Grouped per-stratum assembly: a per-stratum composed distribution per record
# ---------------------------------------------------------------------------
#
# The single-`d` `record_distributions(d, rows)` above assumes ONE shared
# composed distribution with GLOBAL params: only the reserved metadata
# (`obs_time`/`weight`) and the missingness pattern vary per record. The GROUPED
# entry lifts that restriction so each record's edge can use DIFFERENT (sampled)
# params: the caller supplies `ds`, a VECTOR of composed distributions (one per
# STRATUM), and `group`, an INTEGER stratum id per record (a 1-based index into
# `ds`). Record `i` is built from `ds[group[i]]`.
#
# AD-SAFETY (the #321 Enzyme footgun): the group key is the INTEGER stratum id
# from an AD-free data pass; the params arrive as `Dual`s INSIDE the `ds`
# distributions and are built once per stratum, never keyed by a float. Records
# are bucketed by their integer stratum, each stratum's records are built ONCE
# through the single-`d` machinery (sharing that stratum's segment construction),
# then scattered back to row order. One stratum (`length(ds) == 1`, all groups
# `1`) reduces to `record_distributions(ds[1], rows)` exactly (regression-safe).

@doc "

Assemble per-record composed distributions from a PER-STRATUM distribution set.

`record_distributions(ds, rows; group)` is the varying-parameter primitive: each
record's edge may use DIFFERENT (e.g. sampled, partially-pooled) parameters,
selected by an integer STRATUM id. `ds` is a vector of composed distributions,
one per stratum; `group` is a vector of 1-based stratum ids, one per record
(`group[i]` indexes `ds`). Record `i`'s distribution is built from
`ds[group[i]]`, baking the record's reserved metadata (`obs_time`/`weight`) and
missingness pattern in exactly as the single-`d`
[`record_distributions`](@ref)`(d, rows)` does.

The build-once segment construction is shared WITHIN each stratum: records are
bucketed by their integer stratum, each stratum's records are assembled once,
then scattered back to row order. The stratum id is an integer from an AD-free
pass, so the sampled params (carried inside `ds`) never key a lookup - AD-safe.
A single stratum (`length(ds) == 1`, every `group[i] == 1`) is bit-identical to
`record_distributions(ds[1], rows)`.

A stratum's distribution may be a composer (a [`Sequential`](@ref) /
[`Parallel`](@ref) / [`Select`](@ref)) OR a BARE leaf (a univariate / censored
leaf): a single-delay model can pass a vector of bare leaves and each record
scores its leaf directly, with no one-edge `Sequential` wrapper. The bare-leaf
record is density-equal to the one-edge-`Sequential`-wrapped form observed from a
zero origin.

# Arguments
- `ds`: a vector of composed distributions OR bare leaves, one per stratum.
- `rows`: a Tables.jl row source of records keyed by event name.

# Keyword Arguments
- `group`: a vector of 1-based stratum ids (one per record) indexing `ds`.

# Examples
```@example
using CensoredDistributions, Distributions

mk(scale) = Sequential(
    primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
    primary_censored(Gamma(2.0, scale), Uniform(0, 1)))
ds = [mk(1.0), mk(2.0)]
rows = [(onset = 0.0, admit = 2.0, death = 5.0),
    (onset = 1.0, admit = 3.0, death = 9.0)]
recs = CensoredDistributions.record_distributions(ds, rows; group = [1, 2])
logpdf(recs[1], [0.0, 2.0, 5.0])
```

# See also
- [`record_distributions`](@ref)`(d, rows)`: the single shared-`d` entry.
"
function record_distributions(ds::AbstractVector, rows; group)
    rowvec = collect(Tables.rows(rows))
    n = length(rowvec)
    n == 0 && throw(ArgumentError(
        "record_distributions needs at least one record; got an empty table"))
    length(group) == n || throw(ArgumentError(
        "`group` must have one stratum id per record; got $(length(group)) " *
        "ids for $n records"))
    _check_group_ids(group, length(ds))
    # Fast path: one stratum used by every record is the shared-`d` path exactly.
    if length(ds) == 1
        return record_distributions(ds[1], rowvec)
    end
    # Bucket the row INDICES by integer stratum (the AD-free data pass), build
    # each stratum's records ONCE (sharing that stratum's segment construction),
    # then scatter back to row order. The output element type is the promotion of
    # the per-stratum record types, so a heterogeneous `ds` still yields a typed
    # vector for `product_distribution`.
    buckets = [Int[] for _ in eachindex(ds)]
    @inbounds for i in 1:n
        push!(buckets[group[i]], i)
    end
    out = Vector{Any}(undef, n)
    @inbounds for k in eachindex(ds)
        idxs = buckets[k]
        isempty(idxs) && continue
        sub = record_distributions(ds[k], [rowvec[i] for i in idxs])
        for (j, i) in enumerate(idxs)
            out[i] = sub[j]
        end
    end
    return _narrow(out)
end

# Validate the integer stratum ids index `ds` (1..nstrata). A float/non-integer
# id or an out-of-range id is the user's data-pass error; reject it clearly
# rather than silently mis-indexing or (worse) keying on a float.
function _check_group_ids(group, nstrata::Int)
    @inbounds for g in group
        g isa Integer || throw(ArgumentError(
            "`group` stratum ids must be integers (an AD-free data-pass id); " *
            "got $(typeof(g))"))
        (1 <= g <= nstrata) || throw(ArgumentError(
            "`group` stratum id $g is out of range; expected 1..$nstrata " *
            "(one entry of `ds` per stratum)"))
    end
    return nothing
end

@doc "

The grouped (or shared-`d`) per-record log density as a direct value.

`batched_event_logpdf(ds, rows; group)` scores a whole table of records under a
PER-STRATUM distribution set: each record is built from `ds[group[i]]` (via
[`record_distributions`](@ref)`(ds, rows; group)`) and the result is the sum of
the per-record log densities, equal to
`sum(logpdf(record_distributions(ds, rows; group)[i], obs_i))` over the observed
records, with a fully-missing record contributing zero. It is the
varying-parameter grouped invariant as a plain number (bypassing
`product_distribution`).

This is the Turing-friendly grouped primitive: it is a plain `logpdf`-style
scalar, so it drops straight into a `@model` with `@addlogprob!` (no submodel, no
`product_distribution`, no `to_submodel`), and it differentiates under ForwardDiff
and Mooncake because the `group` ids are integers from an AD-free data pass and
the sampled params ride INSIDE `ds`. Use it when the data is fully observed and
you only need to SCORE (the common partial-pooling likelihood):

```julia
@model function pooled(ds_template, rows, group)
    mu ~ Normal(0, 1)
    tau ~ truncated(Normal(0, 1); lower = 0)
    scales ~ filldist(LogNormal(mu, tau), nstrata)
    ds = [rebuild(ds_template, scales[k]) for k in 1:nstrata]
    @addlogprob! CensoredDistributions.batched_event_logpdf(ds, rows; group)
end
```

Prefer the submodel entry [`composed_distribution_model`](@ref)`(ds, table;
group)` (the dual-purpose `obs ~ product_distribution(...)` form) when you also
need to SAMPLE missing / fully-missing records; this scalar form scores only.

`ds` may be a vector of composed distributions OR a vector of BARE leaves (a
single-delay model scores each leaf directly, no one-edge `Sequential` wrapper);
the bare and the wrapped forms give the same log density.

`batched_event_logpdf(d, rows)` is the single shared-`d` form, mirroring
[`record_distributions`](@ref)`(d, rows)`.

# Arguments
- `ds`: a vector of composed distributions OR bare leaves, one per stratum (or a
  single composed distribution / bare leaf `d` for the shared form).
- `rows`: a Tables.jl row source of records keyed by event name.

# Keyword Arguments
- `group`: a vector of 1-based stratum ids (one per record) indexing `ds`.

# Examples
```@example
using CensoredDistributions, Distributions

mk(scale) = Sequential(
    primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
    primary_censored(Gamma(2.0, scale), Uniform(0, 1)))
ds = [mk(1.0), mk(2.0)]
rows = [(onset = 0.0, admit = 2.0, death = 5.0),
    (onset = 1.0, admit = 3.0, death = 9.0)]
CensoredDistributions.batched_event_logpdf(ds, rows; group = [1, 2])
```

# See also
- [`record_distributions`](@ref): the per-record / per-stratum assembly entry.
- [`composed_distribution_model`](@ref): the dual-purpose (fit + generate)
  submodel form of the grouped entry.
"
function batched_event_logpdf(ds::AbstractVector, rows; group)
    recs = record_distributions(ds, rows; group)
    return _batched_records_logpdf(recs)
end

# The single shared-`d` form, mirroring `record_distributions(d, rows)`.
function batched_event_logpdf(d, rows)
    recs = record_distributions(d, rows)
    return _batched_records_logpdf(recs)
end

# Sum each record's `logpdf` at its OWN observed event vector (missing slots
# zeroed, ignored by the marginalising `logpdf`), the per-record-loop value the
# vectorised `product_distribution` path reproduces.
function _batched_records_logpdf(recs)
    total = 0.0
    @inbounds for r in recs
        x = [e === missing ? 0.0 : Float64(e) for e in r.events]
        total += logpdf(r, x)
    end
    return total
end
