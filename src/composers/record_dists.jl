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
    # Numerator: the untruncated factorised per-segment density. The accumulator
    # is seeded with a `Float64` zero and left UNANNOTATED so each segment's
    # (possibly AD-tracked) log density widens it naturally, mirroring the flat
    # `_seq_event_logpdf_untrunc` pattern (the data is constant; the `Dual` comes
    # from the leaf params via the segments).
    lp = zero(float(eltype(vals)))
    @inbounds for j in 1:nseg
        seg = _lookup_seg(bundle.segs, obs[j], obs[j + 1])
        lp += logpdf(seg, vals[j + 1] - vals[j])
    end
    r.horizon === nothing && return _weight_lp(lp, r.weight)
    # Denominator: whole-compose TOTAL truncation (#366). A single
    # conv-to-last-observed right-truncation term `-logcdf(C, window)`, where `C`
    # is the prebuilt origin->last-observed segment and `window = horizon -
    # origin`. With one observed segment `C` IS that segment, reducing to the
    # endpoint-observed term. A non-positive window is an empty-support truncation
    # (`-Inf`), matching `event_logpdf`'s guard.
    last_seg = _lookup_seg(bundle.segs, obs[1], obs[end])
    window = r.horizon - vals[1]
    lp = window <= minimum(last_seg) ? convert(typeof(lp), -Inf) :
         lp - logcdf(last_seg, window)
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
    # A tree with a nested Select routes per record by the row's selector, so each
    # record resolves its own (Select-free) tree before parsing (the selector
    # field is data, not an event).
    _count_selects(d) > 0 && return _select_resolved_records(d, rows)
    parsed = _parse_rows(d, rows)
    # A nested tree (a nested composer or a Competing step) has no flat
    # collapsed-segment layout; build each record generically (correctness first).
    _nested_trait(d.components) isa _Nested && return _generic_records(d, parsed)
    bundle = _build_seq_bundle(d, parsed)
    return [_seq_record(d, p, bundle) for p in parsed]
end

function record_distributions(d::Parallel, rows)
    _count_selects(d) > 0 && return _select_resolved_records(d, rows)
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

# A BARE leaf (a univariate / censored leaf, no composer wrapper) as a vector of
# one-event records: a single-delay model scores `d` directly without wrapping it
# in a one-edge `Sequential`. Each row carries one observed event value (plus the
# optional reserved `weight`/`count`/`obs_time`); the record scores through
# `event_logpdf(::UnivariateDistribution, value; horizon) * weight`, the SAME
# `_GenericRecord` path a leaf `Select` alternative uses. This is exactly the
# density of the one-edge `Sequential(d)` wrapper observed from a zero origin (the
# wrapper's single segment IS the leaf core), so the wrapped and bare forms give
# the same logpdf; the bare form just drops the synthetic origin slot. A
# fully-missing row contributes zero on `logpdf` and is sampled on `rand`.
function record_distributions(d::UnivariateDistribution, rows)
    rowvec = collect(Tables.rows(rows))
    isempty(rowvec) && throw(ArgumentError(
        "record_distributions needs at least one record; got an empty table"))
    return [_leaf_record(d, _row_namedtuple(row)) for row in rowvec]
end

# Build per-record distributions for a tree that nests a `Select`: each record
# resolves its OWN (Select-free) tree from the row's selector(s), strips the
# selector field(s) so they are not matched as events, then parses + builds a
# generic record over the resolved tree. A record missing a needed selector field
# errors in `_resolve_selects`, so a data record never silently routes to the
# first alternative. The resolved tree may itself be flat or nested; either way
# the generic record scores it through `event_logpdf`, equal to the per-record
# loop over the resolved trees.
function _select_resolved_records(d::Union{Sequential, Parallel}, rows)
    rowvec = collect(Tables.rows(rows))
    isempty(rowvec) && throw(ArgumentError(
        "record_distributions needs at least one record; got an empty table"))
    fields = _select_fields(d)
    out = map(rowvec) do row
        nt = _row_namedtuple(row)
        resolved = _resolve_selects(d, nt)
        inner = _drop_named_fields(nt, fields)
        p = _parse_row(resolved, inner)
        _generic_record(resolved, p)
    end
    return out
end

# The selector field names of every nested `Select` in a tree (each Select's
# `selector`), so they are stripped from a row before event matching. RECURSES
# through the same nesting as `_count_selects`/`_resolve_selects` (composer
# components, an `AbstractCompeting`'s outcome `delays`, a Select's alternatives, a
# Latent's inner dist), so a Select nested inside a competing-outcome subtree has
# its selector field stripped too (else the field would be matched as a spurious
# event).
_select_fields(::UnivariateDistribution) = Symbol[]
function _select_fields(d::Select)
    vcat([d.selector],
        reduce(vcat, map(_select_fields, d.alternatives); init = Symbol[]))
end
function _select_fields(d::Union{Sequential, Parallel})
    return reduce(vcat, map(_select_fields, d.components); init = Symbol[])
end
function _select_fields(c::AbstractCompeting)
    return reduce(vcat, map(_select_fields, c.delays); init = Symbol[])
end
_select_fields(d::Latent) = _select_fields(d.dist)

# Drop several named fields from a NamedTuple, preserving the order of the rest.
function _drop_named_fields(row::NamedTuple, fields)
    ks = filter(k -> !(k in fields), keys(row))
    return NamedTuple{ks}(map(k -> row[k], ks))
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
#
# A record carrying a per-record horizon also needs the conv-to-last-observed run
# `(obs[1], obs[end])` for its whole-compose TOTAL truncation denominator (#366),
# so that origin->last-observed segment is registered (and built once, shared)
# alongside the per-segment numerator runs. For an endpoint-observed record this
# is the same run as its single segment.
function _distinct_runs(parsed)
    seen = Set{Tuple{Int, Int}}()
    runs = Tuple{Int, Int}[]
    push_run!(run) = run in seen || (push!(seen, run); push!(runs, run))
    for p in parsed
        obs = _observed_event_indices(p.events)
        length(obs) >= 2 || continue
        for j in 1:(length(obs) - 1)
            push_run!((obs[j], obs[j + 1]))
        end
        p.horizon === nothing || push_run!((obs[1], obs[end]))
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
    return _override_competing_outcome_probs(d, probs)
end

# The single Competing node of a tree (for coercing a per-record override against
# its outcome names), or `nothing` when there is none; errors if more than one.
# RECURSES through the same nesting as `_count_competing`/`_replace_competing`
# (composer components, an `AbstractCompeting`'s outcome `delays`, a `Select`'s
# alternatives, a `Latent`'s inner dist), so a Competing nested inside a
# competing-outcome subtree or a Select alternative is found and coerced against.
# (Previously a nested `AbstractCompeting` hit the `::UnivariateDistribution`
# fallback — they share that supertype — so a nested Competing was missed, and
# `Select`/`Latent` hit no method.) A `HazardCompeting` is NOT branch-prob-
# overridable, so it is not itself returned, but its delays are still searched.
_the_competing_node(c::Competing) = _merge_competing(c,
    _the_competing_node_in(c.delays))
_the_competing_node(c::HazardCompeting) = _the_competing_node_in(c.delays)
_the_competing_node(::UnivariateDistribution) = nothing
function _the_competing_node(d::Union{Sequential, Parallel})
    return _the_competing_node_in(d.components)
end
_the_competing_node(d::Select) = _the_competing_node_in(d.alternatives)
_the_competing_node(d::Latent) = _the_competing_node(d.dist)

_the_competing_node_in(::Tuple{}) = nothing
function _the_competing_node_in(xs::Tuple)
    return _merge_competing(_the_competing_node(first(xs)),
        _the_competing_node_in(Base.tail(xs)))
end

_merge_competing(a, ::Nothing) = a
_merge_competing(::Nothing, b) = b
_merge_competing(::Nothing, ::Nothing) = nothing
function _merge_competing(a, b)
    throw(ArgumentError(
        "a per-record `branch_probs` override needs exactly one Competing " *
        "node in the tree; found more than one"))
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
# A `latent`-wrapped alternative in the VECTORISED `product_distribution` path has
# no place to sample its latent, so it scores through its MARGINAL equivalent (the
# wrapped node), which is density-equal to the latent form. A latent leaf scores
# its single observed value; a latent composer reuses the marginal record build.
function _alternative_record(d::Latent, row::NamedTuple)
    return _alternative_record(d.dist, row)
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

# ---------------------------------------------------------------------------
# Vectorised LATENT scoring (stacked primary priors + vectorised conditional)
# ---------------------------------------------------------------------------
#
# A `latent`-wrapped LEAF carries ONE latent primary per record; a latent CHAIN
# (`latent(Sequential(...))` with `k` edges) carries `k` latent values per record
# (the origin draw plus one independent intermediate GAP per non-terminal edge),
# with the terminal event conditioned on the reconstructed chain. A single `~`
# cannot half-sample (the latents) and half-condition (the observed events), so
# the vectorised latent flow is a two-statement pair driven by these helpers:
#
#   primaries ~ product_distribution(latent_primary_priors(d, rows))
#   @addlogprob! latent_observed_logpdf(d, rows, primaries)
#
# `latent_primary_priors` returns the STACKED priors of every latent row's latent
# values, FLATTENED in row order: a leaf row contributes one prior (its origin
# primary), a `k`-edge chain row contributes `k` priors (the origin primary then
# the first `k - 1` DECLARED edges). `primaries` carries the matching draws in the
# same flat order, so a chain row reads a CONTIGUOUS `k`-slot block.
# `latent_observed_logpdf` scores the WHOLE table given those sampled latents:
# a leaf row conditions its observed event on its matched primary through the
# delay at the implied gap; a chain row reconstructs its event times (E_0 = the
# origin draw, E_i = E_{i-1} + gap_i) and conditions the terminal event on the
# last DECLARED edge at the final gap; a MARGINAL row (an index alternative in a
# mixed Select table) scores through its marginal record `logpdf`, so one
# `@addlogprob!` covers the mixed table. This is a VECTORISED form (a broadcast
# over the rows), not a per-record submodel loop, so it differentiates under
# ForwardDiff and Mooncake reverse. The chain's INTERMEDIATE gaps are sampled
# INDEPENDENTLY (each off its own DECLARED edge) so they ride one
# `product_distribution`; the chaining (the running sum) is pure arithmetic in
# the conditional, and the shift Jacobian is 1, so the joint equals the
# per-record latent chain model exactly.

@doc "

The stacked priors of a latent table's latent values, flattened in row order.

For a [`latent`](@ref)-wrapped leaf each latent record carries one latent primary
event; for a latent CHAIN (`latent(Sequential(...))` with `k` edges) it carries
`k` latent values (the origin draw plus one intermediate gap per non-terminal
edge). `latent_primary_priors(d, rows)` returns the vector of those priors,
FLATTENED in row order and restricted to the LATENT rows: a leaf row contributes
one prior, a `k`-edge chain row contributes `k` priors (the origin primary then
the first `k - 1` declared edges), and a marginal row (an `index` alternative in a
mixed [`Select`](@ref) table) contributes none. The result is the input to a
single `primaries ~ product_distribution(latent_primary_priors(d, rows))`,
sampling every latent value at once.

An all-marginal table (no latent rows, e.g. every record an `index` alternative)
returns an EMPTY prior vector. `product_distribution` of an empty vector is a
degenerate product that throws on `rand`, so guard the empty case at the call
site (skip the `primaries ~ ...` statement when `latent_primary_priors(d, rows)`
is empty: there is nothing latent to sample).

# Arguments
- `d`: a latent leaf or latent chain, or a [`Select`](@ref) with latent
  alternative(s).
- `rows`: a Tables.jl row source of records keyed by event name.

# Examples
```@example
using CensoredDistributions, Distributions
using CensoredDistributions: latent, latent_primary_priors

d = latent(primary_censored(Gamma(4.0, 1.5), Uniform(0, 1)))
priors = latent_primary_priors(d, [(delay = 3.0,), (delay = 5.0,)])
length(priors)
```

# See also
- [`latent_observed_logpdf`](@ref): the matching vectorised observed
  conditional.
"
function latent_primary_priors(d, rows)
    rowvec = collect(Tables.rows(rows))
    priors = Any[]
    for row in rowvec
        nt = _row_namedtuple(row)
        alt = _latent_alternative(d, nt)
        alt === nothing && continue
        append!(priors, _latent_row_priors(alt))
    end
    return _narrow(priors)
end

# The latent priors of one latent row's alternative, flattened. A latent LEAF
# contributes its single origin primary; a latent CHAIN contributes the origin
# primary then the first `k - 1` edges as the intermediate gap priors, so a
# `k`-edge chain row stacks `k` priors. The endpoint-observed chain SAMPLES the
# origin and every intermediate, so each gap `E_i - E_{i-1}` is distributed as the
# BARE edge core (#453): when both endpoints of an edge are sampled latents the
# marginal convolves the bare cores, so the latent gap prior and the terminal
# conditional must be bare too for marginal == latent (no spurious primary smear).
_latent_row_priors(alt::Latent) = _latent_row_priors(alt.dist)
_latent_row_priors(alt::UnivariateDistribution) = (get_primary_event(alt),)
function _latent_row_priors(chain::Sequential)
    origin = _origin_primary_event(_first_origin_node(chain))
    edges = map(_bare_latent_edge, Base.front(chain.components))
    return (origin, edges...)
end

# The number of latent values one latent row's alternative carries (the size of
# its contiguous block in `primaries`): one for a leaf, `k` for a `k`-edge chain.
_latent_row_width(alt::Latent) = _latent_row_width(alt.dist)
_latent_row_width(::UnivariateDistribution) = 1
_latent_row_width(chain::Sequential) = length(chain.components)

@doc "

The vectorised observed conditional of a latent table given sampled primaries.

`latent_observed_logpdf(d, rows, primaries)` scores the whole table in one
contribution: a latent LEAF row conditions its observed event on the matched
sampled primary through the delay at the implied gap (`logpdf(get_dist(alt),
y - p)`); a latent CHAIN row reconstructs its event times from its contiguous
block of `primaries` (`E_0` = the origin draw, `E_i = E_{i-1} + gap_i`) and
conditions the terminal event on the last declared edge at the final gap; and a
MARGINAL row (an `index` alternative in a mixed [`Select`](@ref) table) scores
through its marginal record `logpdf`. The `primaries` are the draws from
`product_distribution(`[`latent_primary_priors`](@ref)`(d, rows))`, flattened in
latent-row order (a leaf row reads one value, a `k`-edge chain row reads a `k`-
slot block); a per-record `weight`/`count` scales each row's contribution. This
is the second statement of the vectorised latent pair, added with `@addlogprob!`.

# Arguments
- `d`: a latent leaf or latent chain, or a [`Select`](@ref) with latent
  alternative(s).
- `rows`: the same Tables.jl row source passed to
  [`latent_primary_priors`](@ref).
- `primaries`: the sampled latent values, flattened in latent-row order (one per
  leaf row, `k` per `k`-edge chain row).

# Examples
```@example
using CensoredDistributions, Distributions
using CensoredDistributions: latent, latent_observed_logpdf

d = latent(primary_censored(Gamma(4.0, 1.5), Uniform(0, 1)))
rows = [(delay = 3.0,), (delay = 5.0,)]
latent_observed_logpdf(d, rows, [0.3, 0.6])
```

# See also
- [`latent_primary_priors`](@ref): the matching stacked primary priors.
"
function latent_observed_logpdf(d, rows, primaries)
    rowvec = collect(Tables.rows(rows))
    total = zero(_latent_acc_type(primaries))
    k = 0
    for row in rowvec
        nt = _row_namedtuple(row)
        alt = _latent_alternative(d, nt)
        w = _row_weight_field(nt, nothing)
        if alt === nothing
            # A marginal row scores through its marginal record logpdf, so one
            # contribution covers a mixed Select table.
            total += _marginal_row_logpdf(d, nt)
        else
            # The row's latent values are a contiguous block of `primaries`; its
            # width is one for a leaf, `k` for a `k`-edge chain.
            width = _latent_row_width(alt)
            block = view(primaries, (k + 1):(k + width))
            k += width
            lp = _latent_row_observed_logpdf(alt, _latent_row_events(d, nt),
                block)
            total += _weight_lp(lp, w)
        end
    end
    return total
end

# The observed conditional of one latent row given its block of sampled latents.
# A latent LEAF conditions its single observed value `y` on its primary `p`
# through the delay gap `y - p`. A latent CHAIN reconstructs the event times from
# the origin draw and the intermediate gaps, then conditions the terminal
# observed event on the last declared edge at the final gap.
function _latent_row_observed_logpdf(alt::Latent, events, block)
    _latent_row_observed_logpdf(alt.dist, events, block)
end
function _latent_row_observed_logpdf(alt::UnivariateDistribution, events, block)
    y = only(events)
    return logpdf(get_dist(alt), y - block[1])
end
function _latent_row_observed_logpdf(chain::Sequential, events, block)
    # `events` is the chain's flat event vector `[E_0, ..., E_k]`; only the
    # terminal is observed for the endpoint-observed chain (origin and EVERY
    # intermediate sampled). Reconstruct the latent event times from the origin
    # draw and the intermediate gaps, then condition the observed terminal on the
    # last edge. #453 rule, matching the per-record `_latent_edge`: when there is a
    # sampled INTERMEDIATE before the terminal (k >= 2) the whole observed->observed
    # segment is a marginalised run, so the terminal edge scores the BARE core; a
    # SINGLE-edge chain (k == 1) is the origin->terminal segment and keeps its
    # DECLARED censoring with the floored sampled origin (#419/#423).
    edges = chain.components
    k = length(edges)
    prev = block[1]
    @inbounds for i in 2:k
        prev += block[i]
    end
    terminal = _the_terminal_observed(events)
    if k >= 2
        return logpdf(_bare_latent_edge(edges[k]), terminal - prev)
    end
    iv = _leaf_interval(edges[k])
    shift = iv === nothing ? prev : _apply_leaf_interval(prev, iv)
    return logpdf(edges[k], terminal - shift)
end

# The terminal (last) observed value of a chain's flat event vector. The
# endpoint-observed chain row observes only its terminal event; an intermediate
# observed event would need the per-segment conditioning of the per-record model,
# which the vectorised chain path does not cover, so that is rejected.
function _the_terminal_observed(events)
    obs = filter(!ismissing, events)
    length(obs) == 1 || throw(ArgumentError(
        "the vectorised latent chain path scores the endpoint-observed chain " *
        "(only the terminal event observed); got $(length(obs)) observed " *
        "events"))
    return only(obs)
end

# The latent alternative scoring a row, or `nothing` when the row is marginal. A
# top-level Latent leaf is latent for every row; a Select reads the row's
# selector and returns the selected alternative only when it is a Latent.
_latent_alternative(d::Latent, ::NamedTuple) = d
_latent_alternative(::UnivariateDistribution, ::NamedTuple) = nothing
function _latent_alternative(d::Select, row::NamedTuple)
    chosen = _pick(d, _select_kind(d, row))
    return chosen isa Latent ? chosen : nothing
end

# The selector value of a Select row, validated to be a Symbol naming an
# alternative (mirroring the per-record Select path).
function _select_kind(d::Select, row::NamedTuple)
    kind = row[d.selector]
    kind isa Symbol || throw(ArgumentError(
        "the Select selector field $(repr(d.selector)) must hold a Symbol " *
        "naming the alternative; got $(typeof(kind))"))
    return kind
end

# The observed event value(s) of a latent row, matched to the selected
# alternative. A latent LEAF carries one observed event (a single value); a
# latent CHAIN carries its flat event vector `[E_0, ..., E_k]` (the terminal
# observed, intermediates missing). The selector field is stripped first under a
# Select so the alternative sees only its own events.
_latent_row_events(d::Latent, row::NamedTuple) = _latent_alt_events(d.dist, row)
function _latent_row_events(d::Select, row::NamedTuple)
    inner = _drop_named_field(row, d.selector)
    alt = _pick(d, _select_kind(d, row))
    return _latent_alt_events(_unwrap_latent(alt), inner)
end

# The wrapped node of a (possibly latent) alternative, so the event-vector
# extraction dispatches on the leaf-vs-chain structure.
_unwrap_latent(alt::Latent) = alt.dist
_unwrap_latent(alt) = alt

# A latent leaf alternative's single observed value; a latent chain
# alternative's flat event vector matched by name.
_latent_alt_events(::UnivariateDistribution, row::NamedTuple) = (_the_observed_value(row),)
_latent_alt_events(chain::Sequential, row::NamedTuple) = _row_event_vector(chain, row)

# The single observed event value of a latent leaf row: the lone non-reserved,
# non-selector field.
_latent_observed_value(d::Latent, row::NamedTuple) = _the_observed_value(row)
function _latent_observed_value(d::Select, row::NamedTuple)
    inner = _drop_named_field(row, d.selector)
    return _the_observed_value(inner)
end

function _the_observed_value(row::NamedTuple)
    ev = _row_event_vector(row)
    length(ev) == 1 || throw(ArgumentError(
        "a latent leaf record takes one observed event value; got " *
        "$(length(ev))"))
    return ev[1]
end

# The marginal log-density of a non-latent row in a mixed Select table: the
# selected (marginal) alternative scored at its single observed value, weighted.
function _marginal_row_logpdf(d::Select, row::NamedTuple)
    chosen = _pick(d, _select_kind(d, row))
    inner = _drop_named_field(row, d.selector)
    rec = _alternative_record(chosen, inner)
    return logpdf(rec, _record_obs_value(rec))
end

# The observed value(s) a record scores at, with missing slots zeroed (the
# marginalising logpdf ignores them), as the `~`-supplied value would be.
function _record_obs_value(rec)
    return [e === missing ? 0.0 : Float64(e) for e in rec.events]
end

# Accumulator element type for the vectorised latent sum: the primaries' element
# type (carrying any AD `Dual`/tracked type), widened to float.
_latent_acc_type(primaries) = float(eltype(primaries))
_latent_acc_type(primaries::AbstractVector{Any}) = Float64

# Narrow an `Any[]` vector of priors to its concrete element type so
# `product_distribution` builds a typed product (and the draws are concretely
# typed). An empty list (no latent rows) returns an empty typed vector.
function _narrow(xs::Vector)
    isempty(xs) && return Union{}[]
    T = mapreduce(typeof, promote_type, xs)
    return collect(T, xs)
end

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
