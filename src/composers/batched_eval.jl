# ============================================================================
# Automatic batched / shared evaluation of a composed distribution (#364)
# ============================================================================
#
# Scoring MANY records that share the SAME composed distribution `d` repeats the
# same convolution work once per record: the per-record `event_logpdf`
# factorises a `Sequential`/`Parallel` into segments and builds each
# unobserved-intermediate run's convolution (`convolve_distributions`) and the
# origin's `primary_censored` core EVERY call, even though those constructions
# depend only on `d` and the record's OBSERVED-PATTERN (which event slots are
# present), not on the per-record observed VALUES.
#
# `batched_event_logpdf(d, rows)` shares that work AUTOMATICALLY (approach 1 of
# #364), with NO user-facing batching API: the modeller composes one `d` and
# scores a whole table; the framework dedupes the construction. The result EQUALS
# the per-record loop `sum(event_logpdf(d, ev_r; horizon = h_r) * w_r)` exactly.
#
# AD safety (the #321 footgun: a param-keyed mutable memo SIGSEGV'd Enzyme):
#   1. AD-FREE PRE-PASS. Read only the row DATA (which slots are `missing`), never
#      a parameter value. Group rows by observed-pattern signature and enumerate
#      the DISTINCT segment RUNS `(a, b)` (1-based observed event-index pairs)
#      needed across all patterns. This is pure integer/structure bookkeeping.
#   2. AD-TRACED BUILD ONCE. For each distinct run build its segment from `d`'s
#      components a single time into a `Vector` indexed by the pre-pass run-id.
#      Straight-line construction; the index is an `Int` run-id, NOT a Dict keyed
#      by parameter values, so there is no hidden mutable memo in the
#      differentiated path. The buffer eltype is seeded from the params so AD
#      `Dual`s propagate (mirrors the nested-scoring eltype handling).
#   3. AD-TRACED EVALUATE. For each row look up its prebuilt segment(s) by the
#      pre-pass mapping, evaluate `logpdf(seg, gap)` plus the per-record
#      truncation correction (the `-logcdf` at the D-anchor when `obs_time` is
#      present, via `truncate_to_horizon`) times the row weight, and accumulate.
#
# SCOPE (first PR): ONE shared `d`, many rows, dedup convolutions across
# observed-patterns for flat `Sequential` / `Parallel`. A nested tree, a
# `Competing` / `Select` node, and a leaf fall back to the exact per-record
# `event_logpdf` (still correct, no sharing); per-edge / strata grouping
# (#364 approach 5) and the opt-in grid/PMF mode (#364 approach 3) are explicit
# FOLLOW-UPS noted in the issue, not built here.

# ---------------------------------------------------------------------------
# Row normalisation (DATA only — the AD-free pre-pass reads these)
# ---------------------------------------------------------------------------

# A normalised record extracted from a table row: the flat event vector (values
# or `missing`, matched by name), the per-record weight (or `nothing`), and the
# per-record observation horizon (or `nothing`). Built once per row up front so
# the pre-pass and the evaluate pass share the same parsed data. The event
# vector's `Missing`-admitting element type matches the per-record path.
struct _BatchRecord{E, W, H}
    events::Vector{Union{Missing, E}}
    weight::W
    horizon::H
end

# The DATA value type for a batch of records: the non-missing event element type,
# stripped at COMPILE time via a `@generated` body (mirrors `_data_value_type` in
# the per-record path). A plain `::Type{Union{Missing, E}} where E` method leaves
# `E` unbound (Aqua flags it) and is not type-stable, so the bare type is spliced
# in as a constant instead. Falls back to `Float64` for an all-missing column.
@generated function _batch_value_type(::Type{T}) where {T}
    S = nonmissingtype(T)
    bare = (S === Union{} || S === Missing) ? Float64 : S
    return :($bare)
end

# The observed-pattern SIGNATURE of a record: the tuple of 1-based event indices
# whose slot is observed (non-missing). Pure data; identical signatures share the
# same segment runs. Sorted ascending by construction (the layout is in order).
function _observed_signature(rec::_BatchRecord)
    idx = Int[]
    @inbounds for i in eachindex(rec.events)
        rec.events[i] === missing || push!(idx, i)
    end
    return Tuple(idx)
end

# ---------------------------------------------------------------------------
# Sequential: dedup the per-segment convolutions across observed-patterns
# ---------------------------------------------------------------------------
#
# A flat `Sequential` chain's per-record contribution is, over its observed event
# indices `o_1 < o_2 < ... < o_m`, the sum of segment log-densities
# `logpdf(seg(o_j, o_{j+1}), val_{j+1} - val_j)`, the origin segment carrying the
# (latent) primary (`_sequential_segment`, the exact per-record factorisation).
# A segment is fully determined by its run `(a, b)` (the observed-index endpoints,
# `a == 1` flagging the origin), NOT by the observed values, so two records with
# the same observed-pattern share every segment. Dedup builds each distinct run
# once.

# A built segment plus the run that produced it. The run `(a, b)` is the pre-pass
# key; `dist` is the AD-traced segment distribution built once.
struct _SegEntry{D}
    a::Int
    b::Int
    dist::D
end

# The flat-Sequential batched log density: dedup the segments, build each once,
# then evaluate every record reusing its prebuilt segments. Equals the per-record
# `sum(event_logpdf(d, ev; horizon) * w)` for the endpoint-observed and
# all-observed flat chain.
function _batched_seq_logpdf(d::Sequential, recs::Vector{<:_BatchRecord},
        ::Type{V}) where {V}
    primary = _origin_primary_event(d.components[1])
    # (1) AD-FREE PRE-PASS: enumerate the distinct runs across all records.
    runs = _distinct_seq_runs(recs)
    # (2) AD-TRACED BUILD ONCE: one segment per distinct run, indexed by run-id.
    segs = _build_seq_segments(d.components, runs, primary)
    # The accumulator type carries any AD `Dual` from the built segments' params
    # plus the data value type, so a leaf-param gradient is not narrowed.
    T = promote_type(V, _segments_param_eltype(segs, V))
    total = zero(T)
    # (3) AD-TRACED EVALUATE: each record reuses its prebuilt segments.
    @inbounds for rec in recs
        total += _batched_seq_record(rec, segs, T)
    end
    return total
end

# The distinct segment runs `(a, b)` needed across all records, as a `Vector` of
# `(a, b)` tuples (the run-id is the position). Pure integer/data: reads only the
# observed indices, never a parameter value.
function _distinct_seq_runs(recs::Vector{<:_BatchRecord})
    seen = Set{Tuple{Int, Int}}()
    runs = Tuple{Int, Int}[]
    for rec in recs
        obs = _observed_event_indices(rec)
        length(obs) >= 2 || throw(ArgumentError(
            "a Sequential record needs at least two observed events; got " *
            "$(length(obs))"))
        for j in 1:(length(obs) - 1)
            run = (obs[j], obs[j + 1])
            run in seen && continue
            push!(seen, run)
            push!(runs, run)
        end
    end
    return runs
end

# The 1-based observed event indices of a record (the slots whose value is not
# `missing`), in order. Pure data.
function _observed_event_indices(rec::_BatchRecord)
    idx = Int[]
    @inbounds for i in eachindex(rec.events)
        rec.events[i] === missing || push!(idx, i)
    end
    return idx
end

# Build one segment distribution per distinct run, ONCE, into a `Vector` indexed
# by run-id (the run's position in `runs`). Straight-line construction reading
# only `components` and the integer run endpoints — no Dict keyed by parameter
# values. The origin run (`a == 1`) reapplies the latent primary.
function _build_seq_segments(components, runs::Vector{Tuple{Int, Int}}, primary)
    return map(runs) do run
        a, b = run
        seg = _sequential_segment(
            components, a, b, a == 1 ? primary : nothing)
        _SegEntry(a, b, seg)
    end
end

# The contribution of one record reusing the prebuilt segments. Looks up each of
# the record's consecutive observed-index runs in `segs` and scores the gap,
# applying the per-record horizon truncation to the (single) endpoint segment
# when present (the endpoint-observed hanta shape). Equals
# `event_logpdf(d, rec.events; horizon = rec.horizon) * rec.weight`.
function _batched_seq_record(rec::_BatchRecord, segs, ::Type{T}) where {T}
    obs = _observed_event_indices(rec)
    vals = _observed_event_values(rec, obs)
    lp = zero(T)
    nseg = length(obs) - 1
    if rec.horizon !== nothing
        # Per-record horizon: defined for the endpoint-observed case (origin +
        # terminal observed). Truncate the single collapsed total at the window
        # from the observed origin, matching `_seq_event_logpdf_h`.
        nseg == 1 || throw(ArgumentError(
            "per-record horizon truncation of a Sequential is defined for the " *
            "endpoint-observed case (origin + terminal observed, " *
            "intermediates unobserved); a batched record with observed " *
            "intermediates needs the whole-compose-vs-per-segment decision"))
        seg = _lookup_segment(segs, obs[1], obs[2])
        window = rec.horizon - vals[1]
        lp += logpdf(truncate_to_horizon(seg, window), vals[2] - vals[1])
        return _apply_weight(lp, rec.weight, T)
    end
    @inbounds for j in 1:nseg
        seg = _lookup_segment(segs, obs[j], obs[j + 1])
        lp += logpdf(seg, vals[j + 1] - vals[j])
    end
    return _apply_weight(lp, rec.weight, T)
end

# The observed event VALUES of a record at the given observed indices, as `T`.
# Converting the data to `T` (not the log densities) keeps the AD type intact.
function _observed_event_values(rec::_BatchRecord, obs::Vector{Int})
    return [Float64(rec.events[i]) for i in obs]
end

# Look up the prebuilt segment for run `(a, b)`. Linear scan over the (small)
# distinct-run vector; the runs are few (one per observed-pattern transition), so
# this stays cheap and AD-transparent (no Dict keyed by params).
function _lookup_segment(segs, a::Int, b::Int)
    @inbounds for s in segs
        s.a == a && s.b == b && return s.dist
    end
    throw(KeyError((a, b)))
end

# ---------------------------------------------------------------------------
# Parallel: every branch shares the prebuilt origin primary + branch cores
# ---------------------------------------------------------------------------
#
# A flat `Parallel` of censored branches shares one origin; each present branch
# contributes `logpdf(core_i, y_i - o)` off the observed origin `o`, plus
# `logpdf(primary, o)` once. The cores and primary are built once for the whole
# batch (they depend only on `d`), then every record reuses them. The missing-
# origin (marginalised) case has no shared collapsible structure beyond the cores
# themselves, so it reuses the prebuilt cores through the per-record marginal.

function _batched_par_logpdf(d::Parallel, recs::Vector{<:_BatchRecord},
        ::Type{V}) where {V}
    primary = _shared_primary_event(d.components)
    primary === nothing && throw(ArgumentError(
        "Parallel shared-origin scoring needs censored branches with a " *
        "primary event; got plain branches"))
    cores = map(_marginal_core, d.components)        # built once for the batch
    T = promote_type(V, _param_eltype(primary), map(_param_eltype, cores)...)
    total = zero(T)
    @inbounds for rec in recs
        total += _batched_par_record(rec, primary, cores, T)
    end
    return total
end

# One Parallel record reusing the prebuilt primary + cores. Mirrors
# `_par_event_logpdf` / `_par_event_logpdf_h`: an observed origin conditions each
# present branch (optionally truncated at the per-record window), a missing origin
# marginalises by the shared-origin integral.
function _batched_par_record(rec::_BatchRecord, primary, cores, ::Type{T}
) where {T}
    events = rec.events
    origin = events[1]
    if origin === missing
        rec.horizon === nothing || throw(ArgumentError(
            "per-record horizon truncation needs an observed shared origin; a " *
            "missing (marginalised) origin has no anchor to truncate from"))
        lp = _parallel_marginal_logpdf(primary, cores, events, T)
        return _apply_weight(lp, rec.weight, T)
    end
    o = Float64(origin)
    insupport(primary, o) || return _apply_weight(convert(T, -Inf), rec.weight, T)
    lp = convert(T, logpdf(primary, o))
    window = rec.horizon === nothing ? nothing : rec.horizon - o
    @inbounds for i in eachindex(cores)
        y = events[i + 1]
        y === missing && continue
        seg = window === nothing ? cores[i] :
              truncate_to_horizon(cores[i], window)
        u = Float64(y) - o
        lp += convert(T, logpdf(seg, u))
    end
    return _apply_weight(lp, rec.weight, T)
end

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Apply a per-record weight to a log density, keeping the accumulator type `T`.
_apply_weight(lp, ::Nothing, ::Type{T}) where {T} = convert(T, lp)
_apply_weight(lp, w, ::Type{T}) where {T} = convert(T, w * lp)

# The parameter element type carried by the prebuilt segments, so the accumulator
# widens to any AD `Dual` from the leaf params (a distribution's `eltype` is its
# variate type and would drop the `Dual`, so the parameters are used instead).
function _segments_param_eltype(segs, ::Type{V}) where {V}
    T = float(V)
    for s in segs
        T = promote_type(T, _param_eltype(s.dist))
    end
    return T
end

# ---------------------------------------------------------------------------
# Public Turing-free entry: dispatch on the composed type
# ---------------------------------------------------------------------------

@doc raw"

Batched log density of a composed distribution over MANY records (#364).

`batched_event_logpdf(d, rows)` scores every record in `rows` against the SAME
composed distribution `d`, sharing the repeated convolution work AUTOMATICALLY:
the result EQUALS the per-record loop
``\sum_r w_r\,\text{event\_logpdf}(d, \text{events}_r;\ \text{horizon} = h_r)``
but each distinct constituent segment (an unobserved-intermediate convolution or
the origin's primary-censored core) is built ONCE and reused across the records
that share its observed-pattern.

`rows` is any Tables.jl row source (a vector of `NamedTuple`s, a column table).
Each row is a record keyed BY EVENT NAME (the same by-name matching the
per-record [`composed_distribution_model`](@ref) uses), with the reserved
`weight`/`count` and `obs_time` fields honoured identically to the per-record
path.

This is the Turing-free core of the batched entry; under DynamicPPL the
`composed_distribution_model(d, table)` submodel adds this contribution with
`@addlogprob!`, so a whole table scores in one `~`.

For a flat [`Sequential`](@ref) or [`Parallel`](@ref) the segments are deduped
across observed-patterns; a nested tree, a [`Competing`](@ref) / [`Select`](@ref)
node, and a leaf fall back to the exact per-record `event_logpdf` (correct, no
sharing yet — per-edge grouping and the grid/PMF mode are #364 follow-ups).

# Arguments
- `d`: the shared composed distribution (a [`Sequential`](@ref),
  [`Parallel`](@ref), [`Competing`](@ref), [`Select`](@ref), or leaf).
- `rows`: a Tables.jl row source of records keyed by event name.

# Examples
```@example
using CensoredDistributions, Distributions

seq = Sequential(
    primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
    primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
rows = [(onset = 0.0, admit = missing, death = 5.0),
    (onset = 1.0, admit = missing, death = 7.0)]
CensoredDistributions.batched_event_logpdf(seq, rows)
```

# See also
- [`event_logpdf`](@ref): the per-record log density this batches.
- [`composed_distribution_model`](@ref): the submodel entry over a table.
"
function batched_event_logpdf(d, rows)
    recs = _collect_records(d, rows)
    return _batched_logpdf_dispatch(d, recs)
end

# Collect the table rows into normalised `_BatchRecord`s once (the AD-free parse).
# Each row is matched to the tree's flat event vector by name, and its reserved
# weight / horizon read, exactly as the per-record `composed_distribution_model`
# does. The event element type is unified across rows so the buffer is concrete.
function _collect_records(d, rows)
    rowvec = collect(Tables.rows(rows))
    isempty(rowvec) && throw(ArgumentError(
        "batched_event_logpdf needs at least one record; got an empty table"))
    return [_record_of(d, row) for row in rowvec]
end

# Normalise a composer row to a `_BatchRecord`: the by-name event vector plus the
# reserved weight / horizon, reusing the shared core row-parsing helpers so the
# batched matching is identical to the per-record path.
function _record_of(d::Union{Sequential, Parallel}, row)
    nt = _as_namedtuple(row)
    ev = _row_event_vector(d, nt)
    w = _row_weight_field(nt, nothing)
    h = _row_horizon_field(nt)
    E = _batch_value_type(eltype(ev))
    return _BatchRecord{E, typeof(w), typeof(h)}(ev, w, h)
end

# A Tables.jl row materialised as a NamedTuple (a `NamedTuple` row passes through;
# any other row type is converted via its column names), so the by-name helpers
# apply uniformly.
_as_namedtuple(row::NamedTuple) = row
function _as_namedtuple(row)
    names = Tuple(Tables.columnnames(row))
    return NamedTuple{names}(map(n -> Tables.getcolumn(row, n), names))
end

# Dispatch the collected records: a flat Sequential/Parallel batches with
# segment sharing; a nested tree or any other composed type falls back to the
# per-record loop (still exact). Resolved on the nested/flat trait at compile
# time, mirroring `logpdf(::Sequential, events)`.
function _batched_logpdf_dispatch(d::Sequential, recs)
    return _batched_seq_dispatch(_nested_trait(d.components), d, recs)
end
function _batched_logpdf_dispatch(d::Parallel, recs)
    return _batched_par_dispatch(_nested_trait(d.components), d, recs)
end

function _batched_seq_dispatch(::_Flat, d::Sequential, recs)
    return _batched_seq_logpdf(d, recs, _batch_records_value_type(recs))
end
_batched_seq_dispatch(::_Nested, d::Sequential, recs) = _fallback_loop(d, recs)
function _batched_par_dispatch(::_Flat, d::Parallel, recs)
    return _batched_par_logpdf(d, recs, _batch_records_value_type(recs))
end
_batched_par_dispatch(::_Nested, d::Parallel, recs) = _fallback_loop(d, recs)

# The shared data value type across a batch of records (their unified event
# element type), seeding the accumulator.
function _batch_records_value_type(recs)
    isempty(recs) ? Float64 :
    _batch_value_type(eltype(first(recs).events))
end

# The per-record fallback: sum the exact per-record `event_logpdf` over the
# records, applying the weight. Used for the cases the dedup does not yet cover
# (nested trees); preserves correctness with no sharing.
function _fallback_loop(d, recs)
    V = _batch_records_value_type(recs)
    T = float(V)
    total = zero(T)
    for rec in recs
        lp = event_logpdf(d, rec.events; horizon = rec.horizon)
        total += rec.weight === nothing ? lp : rec.weight * lp
        T = promote_type(T, typeof(total))
    end
    return total
end
