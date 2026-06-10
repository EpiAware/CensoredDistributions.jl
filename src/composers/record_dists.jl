# ============================================================================
# Per-record composed distributions for vectorised scoring + sampling
# ============================================================================
#
# Scoring many records that share one composed distribution `d` is the standard
# vectorised-distribution pattern: assemble a VECTOR of per-record distributions
# (each record's reserved metadata - the `obs_time` horizon, the weight, and the
# missingness pattern - baked into that record's distribution at ASSEMBLY) and
# score/sample with `product_distribution`. This is dual-purpose: `logpdf` scores
# present observations and `rand` samples missing ones, so ONE entry point fits
# AND generates.
#
# The expensive convolution / collapsed-segment construction is shared: each
# distinct segment run is built ONCE in an AD-free data pre-pass (keyed by the
# integer observed-pattern run, never by a parameter value) and reused across the
# records whose pattern needs it.

# ---------------------------------------------------------------------------
# EventRecord: one record's composed distribution
# ---------------------------------------------------------------------------

# A single record's composed distribution over the flat event vector
# `[E_0, ..., E_k]`. The observed slots score against the prebuilt segments; the
# missing slots marginalise on `logpdf` and are sampled on `rand`. The per-record
# `weight` and `horizon` are baked in, so `logpdf` equals
# `event_logpdf(d, events; horizon) * weight`.
@doc "

One record's composed distribution over the flat event vector `[E_0, ..., E_k]`.

Built by [`record_distributions`](@ref) from a table row. The observed event
slots score against the prebuilt shared segments (`segs`) and the missing slots
marginalise on `logpdf`; `rand` samples the full event path. The record's
reserved metadata (the observation horizon, the weight) is baked in at assembly,
so `logpdf` equals [`event_logpdf`](@ref)`(dist, events; horizon)` scaled by the
weight.

# Examples
```@example
using CensoredDistributions, Distributions

seq = Sequential(
    primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
    primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
recs = CensoredDistributions.record_distributions(
    seq, [(onset = 0.0, admit = 2.0, death = 5.0)])
logpdf(recs[1], [0.0, 2.0, 5.0])
```

# See also
- [`record_distributions`](@ref): the assembly entry that builds these.
"
struct EventRecord{D, E, S, W, H} <: Distribution{Multivariate, Continuous}
    "The shared composed distribution."
    dist::D
    "The record's flat event vector (values or `missing`)."
    events::Vector{Union{Missing, E}}
    "Prebuilt shared segments for this record's observed pattern."
    segs::S
    "The per-record multiplicity weight, or `nothing`."
    weight::W
    "The per-record observation horizon, or `nothing`."
    horizon::H
end

Base.length(r::EventRecord) = length(r.events)
Base.eltype(::Type{<:EventRecord}) = Float64

# ---------------------------------------------------------------------------
# Sequential record: prebuilt per-run segments
# ---------------------------------------------------------------------------

# A built segment plus the integer run `(a, b)` that produced it. The run is the
# pre-pass key; `dist` is the segment built once for that run.
struct _SeqSeg{D}
    a::Int
    b::Int
    dist::D
end

# A Sequential record's segment bundle: the prebuilt run segments shared across
# records, plus the latent origin primary for sampling the full path.
struct _SeqSegs{S, P}
    segs::S
    primary::P
end

# Score a Sequential record's observed events against its prebuilt segments,
# weighted, mirroring `event_logpdf(d, events; horizon)`. Equals the per-record
# loop value.
function _record_logpdf(r::EventRecord{<:Sequential}, x::AbstractVector)
    bundle = r.segs
    obs = _observed_event_indices(r.events)
    vals = [x[i] for i in obs]
    nseg = length(obs) - 1
    if r.horizon !== nothing
        nseg == 1 || throw(ArgumentError(
            "per-record horizon truncation of a Sequential is defined for the " *
            "endpoint-observed case (origin + terminal observed, " *
            "intermediates unobserved)"))
        seg = _lookup_seg(bundle.segs, obs[1], obs[2])
        window = r.horizon - vals[1]
        lp = logpdf(truncate_to_horizon(seg, window), vals[2] - vals[1])
        return _weight_lp(lp, r.weight)
    end
    # The accumulator is seeded with a `Float64` zero and left UNANNOTATED so each
    # segment's (possibly AD-tracked) log density widens it naturally, mirroring
    # the flat `_seq_event_logpdf_untrunc` pattern (the data is constant; the
    # `Dual` comes from the leaf params via the segments).
    lp = zero(float(eltype(vals)))
    @inbounds for j in 1:nseg
        seg = _lookup_seg(bundle.segs, obs[j], obs[j + 1])
        lp += logpdf(seg, vals[j + 1] - vals[j])
    end
    return _weight_lp(lp, r.weight)
end

# Sample the full Sequential event path: the latent origin draw then each event
# as the previous plus a continuous-core delay draw. Returns the whole path so a
# missing record generates every internal event time.
function _record_rand(rng::AbstractRNG, r::EventRecord{<:Sequential})
    return _composer_rand(rng, r.dist)
end

# ---------------------------------------------------------------------------
# Parallel record: prebuilt shared primary + branch cores
# ---------------------------------------------------------------------------

# A Parallel record's segment bundle: the shared origin primary and the branch
# cores, built once for the batch and reused by every record.
struct _ParSegs{P, C}
    primary::P
    cores::C
end

# Score a Parallel record's observed events against the shared primary + cores,
# weighted, mirroring the flat-Parallel `event_logpdf`.
function _record_logpdf(r::EventRecord{<:Parallel}, x::AbstractVector)
    bundle = r.segs
    primary = bundle.primary
    cores = bundle.cores
    events = r.events
    # Promote over the leaf params (carrying any AD `Dual`) and the data, off the
    # CONCRETE primary + cores types (a Tuple), so this is inferred.
    T = float(promote_type(eltype(x), _param_eltype(primary),
        map(_param_eltype, cores)...))
    if events[1] === missing
        r.horizon === nothing || throw(ArgumentError(
            "per-record horizon truncation needs an observed shared origin"))
        lp = _parallel_marginal_logpdf(
            primary, cores, _merge_events(events, x), T)
        return _weight_lp(lp, r.weight)
    end
    o = x[1]
    insupport(primary, o) || return _weight_lp(convert(T, -Inf), r.weight)
    lp = convert(T, logpdf(primary, o))
    window = r.horizon === nothing ? nothing : r.horizon - o
    @inbounds for i in eachindex(cores)
        events[i + 1] === missing && continue
        seg = window === nothing ? cores[i] :
              truncate_to_horizon(cores[i], window)
        lp += convert(T, logpdf(seg, x[i + 1] - o))
    end
    return _weight_lp(lp, r.weight)
end

# Sample the full Parallel event path: one shared origin draw then each branch as
# the origin plus a branch delay draw.
function _record_rand(rng::AbstractRNG, r::EventRecord{<:Parallel})
    return _composer_rand(rng, r.dist)
end

# Build an event view that keeps the record's missingness pattern but takes the
# observed values from `x` (so a marginalised origin still drops the missing
# branches). `missing` slots stay missing; observed slots take `x`.
function _merge_events(events, x)
    out = Vector{Union{Missing, Float64}}(undef, length(events))
    @inbounds for i in eachindex(events)
        out[i] = events[i] === missing ? missing : Float64(x[i])
    end
    return out
end

# ---------------------------------------------------------------------------
# Distributions.jl interface for EventRecord
# ---------------------------------------------------------------------------

function Distributions._logpdf(r::EventRecord, x::AbstractVector)
    return _record_logpdf(r, x)
end
logpdf(r::EventRecord, x::AbstractVector) = _record_logpdf(r, x)

function Distributions._rand!(rng::AbstractRNG, r::EventRecord, x::AbstractVector)
    path = _record_rand(rng, r)
    @inbounds for i in eachindex(x)
        x[i] = path[i]
    end
    return x
end

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Apply a per-record weight to a log density (left unannotated so an AD `Dual`
# from the weight or the log density flows through).
_weight_lp(lp, ::Nothing) = lp
_weight_lp(lp, w) = w * lp

# Look up the prebuilt segment for run `(a, b)`; linear scan over the few runs.
function _lookup_seg(segs, a::Int, b::Int)
    @inbounds for s in segs
        s.a == a && s.b == b && return s.dist
    end
    throw(KeyError((a, b)))
end

# ---------------------------------------------------------------------------
# Assembly: rows -> vector of per-record distributions (shared construction)
# ---------------------------------------------------------------------------

@doc "

Assemble a vector of per-record composed distributions from a table of records.

`record_distributions(d, rows)` turns each record in `rows` into its own
[`EventRecord`](@ref) distribution over the flat event vector, baking the
record's reserved metadata (the `obs_time` horizon, the `weight`/`count`, and the
missingness pattern) into that record's distribution. The expensive convolution /
collapsed-segment construction is SHARED: each distinct segment run is built once
and reused across the records that need it.

The returned vector is the standard vectorised entry: score and sample with
`obs ~ product_distribution(record_distributions(d, rows))`, which is
dual-purpose - Turing scores present observations and samples missing ones.

# Arguments
- `d`: the shared composed distribution (a [`Sequential`](@ref) or
  [`Parallel`](@ref)).
- `rows`: a Tables.jl row source of records keyed by event name.

# Examples
```@example
using CensoredDistributions, Distributions

seq = Sequential(
    primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
    primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
rows = [(onset = 0.0, admit = missing, death = 5.0),
    (onset = 1.0, admit = missing, death = 7.0)]
recs = CensoredDistributions.record_distributions(seq, rows)
logpdf(product_distribution(recs), [[0.0, 0.0, 5.0], [1.0, 0.0, 7.0]])
```

# See also
- [`event_logpdf`](@ref): the per-record log density this reproduces.
"
function record_distributions(d::Sequential, rows)
    parsed = _parse_rows(d, rows)
    # A nested tree (a nested composer or a Competing step) has no flat
    # collapsed-segment layout; build each record generically (correctness first).
    _nested_trait(d.components) isa _Nested && return _generic_records(d, parsed)
    bundle = _build_seq_bundle(d, parsed)
    return [_seq_record(d, p, bundle) for p in parsed]
end

function record_distributions(d::Parallel, rows)
    parsed = _parse_rows(d, rows)
    _nested_trait(d.components) isa _Nested && return _generic_records(d, parsed)
    primary = _shared_primary_event(d.components)
    primary === nothing && throw(ArgumentError(
        "vectorised `record_distributions` needs censored branches with a " *
        "shared primary event; plain-branch Parallel is handled by the " *
        "direct `logpdf` path (`_par_plain_logpdf`), not record assembly"))
    cores = map(_marginal_core, d.components)
    bundle = _ParSegs(primary, cores)
    return [_par_record(d, p, bundle) for p in parsed]
end

# A parsed record: the event vector plus the reserved weight / horizon and the
# raw per-record `branch_probs` override (or `nothing`). Data only.
struct _ParsedRow{E, W, H, B}
    events::Vector{Union{Missing, E}}
    weight::W
    horizon::H
    branch_probs::B
end

# Parse a table into `_ParsedRow`s once (the AD-free data pre-pass): each row is
# matched to the tree's flat event vector by name and its reserved weight /
# horizon / branch_probs read, reusing the shared core row helpers.
function _parse_rows(d::Union{Sequential, Parallel}, rows)
    rowvec = collect(Tables.rows(rows))
    isempty(rowvec) && throw(ArgumentError(
        "record_distributions needs at least one record; got an empty table"))
    return [_parse_row(d, row) for row in rowvec]
end

function _parse_row(d::Union{Sequential, Parallel}, row)
    nt = _row_namedtuple(row)
    ev = _row_event_vector(d, nt)
    w = _row_weight_field(nt, nothing)
    h = _row_horizon_field(nt)
    bp = _row_branch_probs_field(nt)
    E = _data_value_type(eltype(ev))
    return _ParsedRow{E, typeof(w), typeof(h), typeof(bp)}(ev, w, h, bp)
end

# The raw per-record `branch_probs` override carried by a row (a NamedTuple of
# outcome -> prob or a scalar), or `nothing` when the row carries none. Read but
# NOT coerced here; coercion is deferred to the per-record build, which has the
# Competing node to validate against.
function _row_branch_probs_field(row::NamedTuple)
    haskey(row, :branch_probs) || return nothing
    return row.branch_probs
end

# A Tables.jl row as a NamedTuple (a NamedTuple passes through; any other row is
# converted via its column names) so the by-name helpers apply uniformly.
_row_namedtuple(row::NamedTuple) = row
function _row_namedtuple(row)
    names = Tuple(Tables.columnnames(row))
    return NamedTuple{names}(map(n -> Tables.getcolumn(row, n), names))
end

# Build the shared Sequential segment bundle: the distinct observed-pattern runs
# across all records, each built once, plus the origin primary for sampling.
function _build_seq_bundle(d::Sequential, parsed)
    primary = _origin_primary_event(d.components[1])
    runs = _distinct_runs(parsed)
    segs = map(runs) do run
        a, b = run
        seg = _sequential_segment(
            d.components, a, b, a == 1 ? primary : nothing)
        _SeqSeg(a, b, seg)
    end
    return _SeqSegs(segs, primary)
end

# The distinct segment runs `(a, b)` across all records (the run-id is the
# position). Pure integer/data: reads only the observed indices. A record with
# fewer than two observed events contributes no run (a fully-missing record is
# scored as zero and SAMPLED through `rand`, the generative path).
function _distinct_runs(parsed)
    seen = Set{Tuple{Int, Int}}()
    runs = Tuple{Int, Int}[]
    for p in parsed
        obs = _observed_event_indices(p.events)
        length(obs) >= 2 || continue
        for j in 1:(length(obs) - 1)
            run = (obs[j], obs[j + 1])
            run in seen && continue
            push!(seen, run)
            push!(runs, run)
        end
    end
    return runs
end

# The 1-based observed event indices (non-missing slots), in order. Pure data.
function _observed_event_indices(events)
    idx = Int[]
    @inbounds for i in eachindex(events)
        events[i] === missing || push!(idx, i)
    end
    return idx
end

function _seq_record(d::Sequential, p::_ParsedRow, bundle::_SeqSegs)
    return EventRecord(d, p.events, bundle, p.weight, p.horizon)
end

function _par_record(d::Parallel, p::_ParsedRow, bundle::_ParSegs)
    return EventRecord(d, p.events, bundle, p.weight, p.horizon)
end

# ---------------------------------------------------------------------------
# Generic per-record build: nested trees, per-record branch_probs, Select
# ---------------------------------------------------------------------------
#
# A nested-Competing tree (bdbv) and a Select top need per-record handling the
# flat shared-segment build cannot express: a nested tree scores through the
# recursive `_tree_score`, a covariate-CFR `branch_probs` override is baked PER
# RECORD into the record's Competing node, and a Select picks a different branch
# (and obs_time) per row. Each such record holds its OWN (possibly overridden /
# selected) distribution and scores by `event_logpdf(dist, events; horizon)`,
# exactly the per-record loop the vectorised path must equal. The shared-segment
# dedup is skipped here (correctness first); the leaf cores are still cheap to
# rebuild per record.

# One record's distribution that scores through the full `event_logpdf` path.
# `dist` is the record's resolved tree (with any per-record `branch_probs`
# override already applied); `events` carries the missingness pattern; `weight`
# and `horizon` are baked in. `logpdf` equals `event_logpdf(dist, merged;
# horizon) * weight`, matching the per-record loop.
struct _GenericRecord{D, E, W, H} <: Distribution{Multivariate, Continuous}
    dist::D
    events::Vector{Union{Missing, E}}
    weight::W
    horizon::H
end

Base.length(r::_GenericRecord) = length(r.events)
Base.eltype(::Type{<:_GenericRecord}) = Float64

# Score a generic record: merge the supplied values into the record's missingness
# pattern, then route through the shared `event_logpdf` (the SAME function the
# per-record loop calls), scaled by the weight. A composer scores the event
# vector; a univariate-leaf alternative scores its single observed value.
function _record_logpdf(r::_GenericRecord, x::AbstractVector)
    merged = _merge_events(r.events, x)
    lp = _generic_event_logpdf(r.dist, merged, r.horizon)
    return _weight_lp(lp, r.weight)
end

function _generic_event_logpdf(d::Union{Sequential, Parallel}, merged, horizon)
    event_logpdf(d, merged; horizon = horizon)
end
# A univariate-leaf alternative scores its single observed value; a missing slot
# (a fully-missing record that `rand` generates) contributes no factor.
function _generic_event_logpdf(d::UnivariateDistribution, merged, horizon)
    merged[1] === missing && return 0.0
    return event_logpdf(d, merged[1]; horizon = horizon)
end

# Sample a generic record's full event path. A nested tree returns its flat event
# vector (one shared origin draw, each Competing outcome sampled); a flat tree
# uses the flat-path simulation. Both yield a `[E_0, ...]` vector to fill `x`.
function _record_rand(rng::AbstractRNG, r::_GenericRecord)
    return _generic_record_rand(rng, r.dist)
end

function _generic_record_rand(rng::AbstractRNG, d::Union{Sequential, Parallel})
    _nested_trait(d.components) isa _Nested ?
    collect(_tree_event_vector(rng, d)) : _composer_rand(rng, d)
end
# A univariate-leaf alternative samples its single observed value.
_generic_record_rand(rng::AbstractRNG, d::UnivariateDistribution) = [rand(rng, d)]

function Distributions._logpdf(r::_GenericRecord, x::AbstractVector)
    return _record_logpdf(r, x)
end
logpdf(r::_GenericRecord, x::AbstractVector) = _record_logpdf(r, x)

function Distributions._rand!(
        rng::AbstractRNG, r::_GenericRecord, x::AbstractVector)
    path = _record_rand(rng, r)
    # A nested tree's sampled path leaves the UNRESOLVED Competing outcome slots
    # `missing` (one outcome resolves per record); the dual-purpose generative
    # matrix is `Float64`, so an unresolved slot is filled with `NaN` (a sentinel
    # the scored slots never take).
    @inbounds for i in eachindex(x)
        v = path[i]
        x[i] = v === missing ? NaN : v
    end
    return x
end

# Build a generic per-record distribution vector: each record bakes in its own
# `branch_probs` override (coerced + validated against the tree's single Competing
# node) so a covariate CFR flows in per record, plus its weight / horizon.
function _generic_records(d::Union{Sequential, Parallel}, parsed)
    return [_generic_record(d, p) for p in parsed]
end

function _generic_record(d, p::_ParsedRow)
    scored = _bake_branch_probs(d, p.branch_probs)
    E = _data_value_type(eltype(p.events))
    return _GenericRecord{typeof(scored), E, typeof(p.weight),
        typeof(p.horizon)}(scored, p.events, p.weight, p.horizon)
end

# Apply a per-record `branch_probs` override to the tree's single Competing node,
# or return the tree unchanged when the row carries none. Coercion + validation
# reuse the shared core helpers, so the override semantics match the per-record
# `composed_distribution_model` path exactly.
_bake_branch_probs(d, ::Nothing) = d
function _bake_branch_probs(d, raw)
    node = _the_competing_node(d)
    node === nothing && throw(ArgumentError(
        "a `branch_probs` row field needs a Competing node in the tree; none " *
        "found"))
    probs = _coerce_branch_probs(node, raw)
    return _override_competing_branch_probs(d, probs)
end

# The single Competing node of a tree (for coercing a per-record override against
# its outcome names), or `nothing` when there is none; errors if more than one.
_the_competing_node(c::Competing) = c
_the_competing_node(::UnivariateDistribution) = nothing
function _the_competing_node(d::Union{Sequential, Parallel})
    found = nothing
    for c in d.components
        f = _the_competing_node(c)
        f === nothing && continue
        found === nothing || throw(ArgumentError(
            "a per-record `branch_probs` override needs exactly one Competing " *
            "node in the tree; found more than one"))
        found = f
    end
    return found
end

# ---------------------------------------------------------------------------
# Select top: per-record branch selection
# ---------------------------------------------------------------------------
#
# A `Select` top routes each record to ONE of its independent alternatives, named
# by the row's selector field (`row[d.selector]`, default `:kind`). The chosen
# alternative is itself a leaf or a composer, so each record builds that
# alternative's record distribution (with the record's own obs_time / weight),
# selected per row. Different rows may pick different alternatives and carry
# different obs_time horizons, exactly as the per-record
# `composed_distribution_model(d::Select, row)` does.

function record_distributions(d::Select, rows)
    rowvec = collect(Tables.rows(rows))
    isempty(rowvec) && throw(ArgumentError(
        "record_distributions needs at least one record; got an empty table"))
    recs = [_select_record(d, _row_namedtuple(row)) for row in rowvec]
    # The records are scored together via `product_distribution`, which requires a
    # rectangular event matrix: every record must have the SAME number of event
    # slots. Different alternatives may have different event-slot counts (a leaf is
    # one, a composer several), so a table whose rows select differing-length
    # alternatives has no rectangular layout. Distributions.jl would otherwise
    # throw an opaque "all distributions must be of the same size"; raise a clear
    # error naming the cause instead.
    n1 = length(first(recs))
    all(r -> length(r) == n1, recs) || throw(ArgumentError(
        "vectorised record_distributions over a Select needs every selected " *
        "alternative to have the same number of event slots; the rows select " *
        "alternatives of differing length (e.g. a leaf vs a multi-event " *
        "composer). Score each fixed-length subset of rows separately. Full " *
        "Select-in-composer nesting is deferred; see " *
        "https://github.com/EpiAware/CensoredDistributions.jl/issues/413."))
    return recs
end

# Build one Select record: read the selector, pick the alternative, and build
# that alternative's record distribution from the row (selector field stripped).
function _select_record(d::Select, row::NamedTuple)
    kind = row[d.selector]
    kind isa Symbol || throw(ArgumentError(
        "the Select selector field $(repr(d.selector)) must hold a Symbol " *
        "naming the alternative; got $(typeof(kind))"))
    chosen = _pick(d, kind)
    inner = _drop_named_field(row, d.selector)
    return _alternative_record(chosen, inner)
end

# Build the chosen alternative's record distribution. A composer alternative
# reuses the per-record build (so nested trees / branch_probs work under a
# Select); a leaf alternative builds a univariate-leaf record scored through
# `event_logpdf`.
function _alternative_record(d::Union{Sequential, Parallel}, row::NamedTuple)
    return only(record_distributions(d, [row]))
end
function _alternative_record(d::UnivariateDistribution, row::NamedTuple)
    return _leaf_record(d, row)
end

# A univariate-leaf Select alternative as a one-event record: the single observed
# value, the reserved weight / horizon baked in, scored through the shared
# `event_logpdf(::UnivariateDistribution, x; horizon)` so the value equals the
# per-record leaf model.
function _leaf_record(d::UnivariateDistribution, row::NamedTuple)
    ev = _row_event_vector(row)
    length(ev) == 1 || throw(ArgumentError(
        "a leaf Select alternative takes one event value; got $(length(ev))"))
    w = _row_weight_field(row, nothing)
    h = _row_horizon_field(row)
    E = _data_value_type(eltype(ev))
    return _GenericRecord{typeof(d), E, typeof(w), typeof(h)}(d, ev, w, h)
end

# Drop a single named field from a NamedTuple, preserving the order of the rest.
function _drop_named_field(row::NamedTuple, field::Symbol)
    ks = filter(!=(field), keys(row))
    return NamedTuple{ks}(map(k -> row[k], ks))
end
