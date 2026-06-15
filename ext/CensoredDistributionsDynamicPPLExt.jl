module CensoredDistributionsDynamicPPLExt

# DynamicPPL methods for the submodel constructors declared (and documented) in
# `src/turing_models.jl`. Loaded only when DynamicPPL is available, keeping the
# core Turing-free.

using CensoredDistributions: CensoredDistributions, PrimaryCensored, Latent,
                             IntervalCensored, PrimaryConditional, Sequential,
                             Parallel, Competing, Select,
                             get_primary_event, component_names
import CensoredDistributions: primary_censored_model, interval_censored_model,
                              double_interval_censored_model,
                              composed_distribution_model,
                              composed_parameters_model
using DynamicPPL: DynamicPPL, @model, to_submodel, VarName
using Distributions: Distributions, UnivariateDistribution, logpdf,
                     product_distribution
using Random: AbstractRNG
import Tables

# `CensoredDistributions.weight(d, w)` is called with the module qualifier
# because the `weight` keyword argument shadows the function name inside the
# `@model` bodies. `weight(d, nothing)` returns `d` unweighted, so one `~`
# statement covers both the weighted and unweighted cases.
const _weight = CensoredDistributions.weight

# ===========================================================================
# Leaf submodels
# ===========================================================================

# Marginal: the primary event is integrated out inside `logpdf`, so score the
# marginal log-density via `~`.
@model function primary_censored_model(
        d::PrimaryCensored, y; weight = nothing)
    y ~ _weight(d, weight)
    return y
end

# Latent: the primary event is a sampled latent. Declare it inside the `@model`
# (`p ~ get_primary_event(d)`) so the user never passes it, then score the
# observed time against the conditional distribution given that sampled `p`. Both
# statements are `~`, so the model does inference and generation (`rand`/missing
# `y` draws `[p, y]`) and the `weight` keyword still applies through the `~`.
@model function primary_censored_model(
        d::Latent, y; weight = nothing)
    p ~ get_primary_event(d)
    y ~ _weight(PrimaryConditional(d, p), weight)
    return y
end

@model function interval_censored_model(
        d::IntervalCensored, y; weight = nothing)
    y ~ _weight(d, weight)
    return y
end

# `double_interval_censored` returns a composed univariate distribution (primary
# censoring, optional truncation, optional interval censoring), all marginal, so
# one `~` scores the whole pipeline.
@model function double_interval_censored_model(
        d::UnivariateDistribution, y; weight = nothing)
    y ~ _weight(d, weight)
    return y
end

# ===========================================================================
# Generic record entry: composed_distribution_model
# ===========================================================================
#
# `composed_distribution_model(d, row)` is the SINGLE generic entry for a whole
# record. It dispatches on `typeof(d)`: a leaf/univariate distribution delegates
# to the matching LEAF model (`primary_censored_model` / `interval_censored_model`
# / `double_interval_censored_model`), and a composed distribution recurses
# through the composer structure. This keeps the leaf models correctly named while
# never misnaming a composed record (a `Sequential` of double-censored edges is
# composed, not 'primary censored').
#
# The composed methods take the observations as a `NamedTuple` ROW keyed by EVENT
# NAME (`(onset = 2.0, admit = missing, death = 9.0)`); the field ORDER is the
# event order `E_0, E_1, ..., E_k` the composer's components span (a Tables.jl
# linelist row IS such a NamedTuple, so a row passes straight in). `missing`
# fields are the per-record signal that drives the marginalise-vs-condition
# dispatch. A reserved `weight`/`count` field (or the `weight =` keyword)
# scales the likelihood and is EXCLUDED from the event dispatch.
#
# Marginal vs latent is DISPATCH on the STRUCT TYPE (not a runtime predicate),
# mirroring the leaf (`PrimaryCensored` marginal, `Latent` latent):
#   - a bare composer (`Sequential` / `Parallel` / `Competing`) is MARGINAL: it
#     scores the row vector via `~ d`, so the censored composer `logpdf`
#     auto-marginalises unobserved intermediates / the origin and conditions on
#     observed events. The submodel log-density equals the direct
#     `logpdf(d, event_vector)`, and missing fields drive the per-record path.
#   - a `latent`-wrapped composer turns the shared/origin primary ON: the origin
#     `o ~ ...` is declared INSIDE the model (so the sampled origin lives in the
#     VarInfo and the model fits AND generates the full event path), each
#     observed downstream event conditions on `o` through its own edge,
#     and unobserved intermediates marginalise by convolving the adjacent cores.
# Per-record/branch latents are namespaced by `prefix` at the call site (see the
# composer tests), so chains stay readable and groupable.

# --- Leaf delegation -------------------------------------------------------
#
# A leaf/univariate record routes to its correctly named leaf model. The record
# may arrive as a bare observed value or as a one-event `NamedTuple` row (with an
# optional reserved weight); `_leaf_value`/`_leaf_weight` normalise both. The
# `weight =` keyword still applies and overrides a row weight.

# A `PrimaryCensored` (marginal) -> the primary-censored leaf model. A reserved
# `obs_time` row field right-truncates the marginal leaf at that horizon;
# the leaf is observed from the origin, so its window is the horizon itself. The
# truncated leaf is a plain univariate, so it routes through the univariate
# marginal model; without a horizon the correctly named primary-censored model is
# kept.
function composed_distribution_model(
        d::PrimaryCensored, row; weight = nothing)
    _row_horizon_of(row) === nothing && return primary_censored_model(
        d, _leaf_value(row); weight = _leaf_weight(row, weight))
    return double_interval_censored_model(
        _leaf_horizon(d, row), _leaf_value(row);
        weight = _leaf_weight(row, weight))
end

# A `Latent{<:PrimaryCensored}` -> the latent primary-censored leaf model. The
# latent path samples the origin internally, so a per-record `obs_time`
# horizon is not supported here (truncating a latent twin is out of scope);
# reject it rather than silently drop it.
function composed_distribution_model(
        d::Latent{<:PrimaryCensored}, row; weight = nothing)
    _reject_latent_horizon(row)
    return primary_censored_model(
        d, _leaf_value(row); weight = _leaf_weight(row, weight))
end

# A per-record `obs_time` horizon is not supported under the latent form (the
# origin is sampled internally, so right-truncating the latent twin is out of
# scope). Reject a row that carries one rather than silently dropping it; the
# marginal form applies the horizon.
_reject_latent_horizon(row) = nothing
function _reject_latent_horizon(row::NamedTuple)
    _row_horizon(row) === nothing || throw(ArgumentError(
        "per-record obs_time horizon is not supported under `latent`; " *
        "use the marginal form"))
    return nothing
end

# An `IntervalCensored` -> the interval-censored leaf model. An `obs_time` row
# field right-truncates the marginal leaf (routed through the univariate marginal
# model since `truncated(::IntervalCensored)` is a plain univariate).
function composed_distribution_model(
        d::IntervalCensored, row; weight = nothing)
    h = _row_horizon_of(row)
    h === nothing && return interval_censored_model(
        d, _leaf_value(row); weight = _leaf_weight(row, weight))
    return double_interval_censored_model(
        _leaf_horizon(d, row), _leaf_value(row);
        weight = _leaf_weight(row, weight))
end

# Any other univariate leaf (a `double_interval_censored` pipeline, a
# `Truncated`, a `convolve_distributions` result, ...) -> the marginal univariate
# leaf model. A bare `Latent{<:PrimaryCensored}` is handled above; the composer
# `Latent{<:Sequential}` / `Latent{<:Parallel}` methods below are multivariate
# and dispatch ahead of this univariate fallback. A reserved `obs_time` row field
# right-truncates the leaf at that horizon.
function composed_distribution_model(
        d::UnivariateDistribution, row; weight = nothing)
    return double_interval_censored_model(
        _leaf_horizon(d, row), _leaf_value(row);
        weight = _leaf_weight(row, weight))
end

# Apply a per-record `obs_time` horizon to a leaf distribution by right-truncating
# it (the leaf is observed from the origin, so the window is the horizon itself);
# no `obs_time` field returns the leaf unchanged.
function _leaf_horizon(d, row)
    h = _row_horizon_of(row)
    return h === nothing ? d : CensoredDistributions.truncate_to_horizon(d, h)
end

# The horizon of a leaf record: a bare value carries none; a row reads `obs_time`.
_row_horizon_of(row) = nothing
_row_horizon_of(row::NamedTuple) = _row_horizon(row)

# The single observed value of a leaf record: a bare value passes through; a
# one-event `NamedTuple` row yields its single event (dropping any reserved
# weight/count).
_leaf_value(row) = row
function _leaf_value(row::NamedTuple)
    ev = _event_vector(row)
    length(ev) == 1 || throw(ArgumentError(
        "a leaf record takes one event value; got $(length(ev))"))
    return ev[1]
end

# The weight of a leaf record: the keyword wins, else a row's reserved
# weight/count, else `nothing`.
_leaf_weight(row, kw_weight) = kw_weight
_leaf_weight(row::NamedTuple, kw_weight) = _row_weight(row, kw_weight)

# --- Reserved-field handling -----------------------------------------------
#
# The pure row -> event-vector / reserved-field parsing now lives in the CORE
# (`src/composers/tree_events.jl`, Turing-free) so the per-record and batched
# paths share one source of truth. These thin aliases keep the ext's local
# names while delegating to the core helpers.

# Reserved row fields that are NOT events come from the CORE (`tree_events.jl`),
# shared by the per-record and batched paths. They include the per-record
# Competing branch-probability override `branch_probs`, so it is excluded
# from by-name event matching for a nested Competing tree too.
const _RESERVED_ROW_FIELDS = CensoredDistributions._RESERVED_ROW_FIELDS

_event_vector(row::NamedTuple) = CensoredDistributions._row_event_vector(row)
function _event_vector(d::Union{Sequential, Parallel}, row::NamedTuple)
    return CensoredDistributions._row_event_vector(d, row)
end

function _row_weight(row::NamedTuple, kw_weight)
    return CensoredDistributions._row_weight_field(row, kw_weight)
end

_row_horizon(row::NamedTuple) = CensoredDistributions._row_horizon_field(row)

# --- Marginal composer models ----------------------------------------------

# Marginal `Sequential` / `Parallel`: score the row's event vector directly
# through the censored composer `logpdf`. The whole row is DATA (carried in the
# `row` argument), so the contribution is added with `@addlogprob!` rather than a
# sampled `~`, leaving no spurious VarInfo variable (matching the leaf marginal,
# which has no latent). The composer dispatches on the event vector's
# `Missing`-admitting element type and per-record missingness pattern:
# unobserved intermediates / a missing origin marginalise, observed events
# condition on their declared censoring. The contribution equals
# `logpdf(d, event_vector)`, scaled by the row weight, so the submodel
# log-density equals the direct `logpdf`.
#
# A NESTED `Competing` node is scored by the same path: its outcome
# columns occupy one event slot each (`_flat_event_names`), so the observed
# outcome is identified positionally and `_tree_step(::Competing)` conditions on
# that branch. A per-record `branch_probs` field OVERRIDES the (single) nested
# Competing's stored probabilities by rebuilding the tree for the record, so a
# covariate CFR `logistic(Xβ)` flows in exactly as for the top-level node.
@model function composed_distribution_model(
        d::Union{Sequential, Parallel}, row::NamedTuple; weight = nothing)
    # A NamedTuple of equal-length columns IS a Tables.jl table (a `columntable`),
    # not one record; route it to the vectorised table path. A scalar-valued row
    # NamedTuple is not a table and scores as one record below.
    if Tables.istable(row)
        weight === nothing || throw(ArgumentError(
            "the vectorised composed model takes per-row weights via a reserved " *
            "`weight`/`count` row field, not the `weight` keyword"))
        recs = CensoredDistributions.record_distributions(d, row)
        obs ~ to_submodel(_vectorised_records_model(
            recs, _record_obs_matrix(recs)))
        return obs
    end
    scored = _apply_branch_probs_override(d, row)
    events = _event_vector(scored, row)
    w = _row_weight(row, weight)
    horizon = _row_horizon(row)
    DynamicPPL.@addlogprob! _marginal_logprob_h(scored, events, w, horizon)
    return events
end

# Rebuild a composed tree with the per-record `branch_probs` override applied to
# its single nested `Competing` node, or return it unchanged when the row
# carries no override. The override is coerced + validated against that node's
# outcomes via the SHARED core helper, then the tree is rebuilt with the new
# probabilities (their element type preserved so a covariate `Dual` flows). The
# tree must contain exactly one Competing node for a scalar/NamedTuple override.
function _apply_branch_probs_override(d, row::NamedTuple)
    haskey(row, :branch_probs) || return d
    node = _the_competing(d)
    node === nothing && throw(ArgumentError(
        "a `branch_probs` row field needs a Competing node in the tree; none " *
        "found"))
    probs = CensoredDistributions._coerce_branch_probs(node, row.branch_probs)
    return CensoredDistributions._override_competing_outcome_probs(d, probs)
end

# The single Competing node of a tree (for coercing a per-record override against
# its outcome names), or `nothing` when there is none. Errors if more than one.
function _the_competing(d)
    found = _find_competing(d)
    return found
end
_find_competing(c::Competing) = c
_find_competing(::UnivariateDistribution) = nothing
function _find_competing(d::Union{Sequential, Parallel})
    found = nothing
    for c in d.components
        f = _find_competing(c)
        f === nothing && continue
        found === nothing || throw(ArgumentError(
            "a per-record `branch_probs` override needs exactly one Competing " *
            "node in the tree; found more than one"))
        found = f
    end
    return found
end

# VECTORISED entry: score a WHOLE TABLE of records sharing the same composed `d`
# in one `~`. `rows` is any Tables.jl row source that is NOT a single
# `NamedTuple` row (a `Vector` of rows, a column table); the single-`NamedTuple`
# method above stays the per-record entry. Each record becomes its own
# distribution via `record_distributions` (baking the reserved
# `weight`/`count`/`obs_time` row fields and the missingness pattern, sharing the
# convolution construction across records). The table scores AND samples with the
# standard `obs ~ product_distribution(...)` tilde: the observed event matrix
# (one column per record) is read from the rows and supplied as the `~` value, so
# present observations score; a fully-missing table supplies `missing`, so Turing
# samples the full event paths. This is dual-purpose - the same model fits and
# generates - with no `@addlogprob!`. Usable as `y ~ to_submodel(...)`.
function composed_distribution_model(
        d::Union{Sequential, Parallel}, rows::AbstractVector; weight = nothing)
    weight === nothing || throw(ArgumentError(
        "the vectorised composed model takes per-row weights via a reserved " *
        "`weight`/`count` row field, not the `weight` keyword"))
    recs = CensoredDistributions.record_distributions(d, rows)
    return _vectorised_records_model(recs, _record_obs_matrix(recs))
end

# A whole Tables.jl table (e.g. a `DataFrame`) scores in one `~` like the vector
# entry: a DataFrame IS a Tables.jl source, so the doc passes it straight in
# instead of iterating rows. `record_distributions` consumes any Tables.jl source
# directly, so the table flows through the same vectorised record path; a single
# `NamedTuple` row and a `Vector` of rows keep their own (more specific) methods,
# so only a column table (a DataFrame) lands here.
function composed_distribution_model(
        d::Union{Sequential, Parallel, Select}, table; weight = nothing)
    Tables.istable(table) || throw(ArgumentError(
        "composed_distribution_model(d, table) takes a NamedTuple row, a vector " *
        "of rows, or a Tables.jl table; got $(typeof(table))"))
    weight === nothing || throw(ArgumentError(
        "the vectorised composed model takes per-row weights via a reserved " *
        "`weight`/`count` row field, not the `weight` keyword"))
    recs = CensoredDistributions.record_distributions(d, table)
    return _vectorised_records_model(recs, _record_obs_matrix(recs))
end

# The inner vectorised submodel: `obs` is a MODEL ARGUMENT, so a supplied matrix
# is observed (scores) and `missing` is sampled (generates the full event paths).
@model function _vectorised_records_model(recs, obs)
    obs ~ product_distribution(recs)
    return obs
end

# The observed event matrix for a vector of records: one column per record, each
# the record's flat event vector with `missing` slots replaced by a placeholder
# (ignored by the marginalising `logpdf`). A column whose record is fully missing
# stays `missing`, so a fully-missing table yields a `missing` matrix that Turing
# samples instead of scores.
function _record_obs_matrix(recs)
    any(_record_has_observed, recs) || return missing
    n = length(first(recs))
    M = Matrix{Float64}(undef, n, length(recs))
    @inbounds for (j, r) in enumerate(recs)
        for i in 1:n
            v = r.events[i]
            M[i, j] = v === missing ? 0.0 : Float64(v)
        end
    end
    return M
end

_record_has_observed(r) = any(v -> v !== missing, r.events)

# A `Competing` node SELF-DISPATCHES on the row's outcome missingness
# (decision 2), mirroring the observed-intermediate dispatch of a chain:
#
#   - EXACTLY ONE outcome's event time observed in the row -> CONDITION on that
#     branch: `log(p[obs]) + logpdf(delay[obs], gap)`, the observed branch's own
#     (censored) logpdf at its gap (its observed time, the delay from the origin);
#   - the outcome unknown (a `resolved` resolution time but no per-outcome time,
#     OR a single bare non-outcome event field) -> MARGINALISE: the mixture
#     `logpdf` at the resolution time;
#   - ALL outcome columns missing and no resolution time -> the record is fully
#     missing and contributes no factor (zero).
#
# BRANCH PROBABILITIES (three regimes via the SAME node):
#   (a) fixed/sampled scalar  -> the node's stored `branch_probs`;
#   (b) COVARIATE-DRIVEN      -> a reserved `branch_probs` ROW field (a NamedTuple
#       of outcome -> prob, or a scalar for a two-outcome node giving the FIRST
#       outcome's probability) OVERRIDES the stored probabilities per record. This
#       is how a covariate CFR `logistic(Xβ)` (computed in PLAIN TURING,
#       decision 2) flows in per record; the regression stays OUT of the node, the
#       node only CONSUMES the probability. Validated to lie in `[0, 1]` and (for
#       a NamedTuple) to sum to one across outcomes.
#   (c) unknown outcome       -> the probabilities enter only as the mixture
#       weights of the marginalise path.
#
# The observed OUTCOME (which outcome column is non-missing, Feature 1's by-name
# missingness) and the per-record PROBABILITY (the reserved `branch_probs` field)
# are DISTINCT row inputs. The whole record is DATA, so the
# contribution is added with `@addlogprob!` (no spurious VarInfo variable).
@model function composed_distribution_model(
        d::Competing, row::NamedTuple; weight = nothing)
    w = _row_weight(row, weight)
    probs = _competing_outcome_probs(d, row)
    horizon = _row_horizon(row)
    DynamicPPL.@addlogprob! _competing_logprob(d, row, probs, w, horizon)
    return nothing
end

# Reserved row fields specific to a `Competing` node: the resolution time when the
# outcome is unknown (`resolved`); the branch-probability override
# (`branch_probs`) is already a shared reserved field.
const _COMPETING_RESERVED = (_RESERVED_ROW_FIELDS..., :resolved)

# The branch probabilities to USE for this record: a reserved `branch_probs` row
# field overrides the node's stored probabilities (regime b), else the stored ones
# (regime a). The coercion + validation is the SHARED core helper
# (`_coerce_branch_probs`), so the top-level and nested paths agree on the
# NamedTuple/scalar override semantics.
function _competing_outcome_probs(d::Competing, row::NamedTuple)
    haskey(row, :branch_probs) || return d.branch_probs
    return CensoredDistributions._coerce_branch_probs(d, row.branch_probs)
end

# The competing log-density for one record under branch probabilities `probs`,
# scaled by the weight `w`, optionally right-truncated at the per-record horizon
# (hanta). Selects condition / marginalise / fully-missing from the row's
# outcome missingness. A `horizon` right-truncates the conditioned branch delay
# (resp. the mixture) at the horizon (the competing time is measured from the
# origin, so the window is the horizon itself).
function _competing_logprob(d::Competing, row::NamedTuple, probs, w, horizon)
    obs = _observed_outcomes(d, row)
    if length(obs) == 1
        # Observed outcome: condition on that branch through the SHARED core
        # arithmetic (the per-record horizon right-truncates the branch delay).
        i, gap = obs[1]
        branch = _maybe_truncate(d.delays[i], horizon)
        lp = CensoredDistributions._competing_condition_logpdf(
            probs, branch, gap, i)
        return _scale(lp, w)
    end
    isempty(obs) || throw(ArgumentError(
        "a Competing record may observe at most one outcome time; got " *
        "$(length(obs)) ($(collect(first.(obs))))"))
    # No outcome observed: marginalise at the resolution time if known, else the
    # record is fully missing and contributes nothing.
    t = _competing_resolution_time(d, row)
    t === nothing && return _scale(zero(eltype(probs)), w)
    delays = map(g -> _maybe_truncate(g, horizon), d.delays)
    lp = CensoredDistributions._competing_marginal_logpdf(probs, delays, t)
    return _scale(lp, w)
end

# Right-truncate `dist` at the per-record horizon when one is supplied (the
# competing time is measured from the origin), else return it unchanged.
_maybe_truncate(dist, ::Nothing) = dist
function _maybe_truncate(dist, horizon)
    return CensoredDistributions.truncate_to_horizon(dist, horizon)
end

# The observed outcomes of a row as `(outcome_index, gap)` pairs: a per-outcome
# field (keyed by the outcome name) holding a non-missing time. The gap is the
# observed delay from the origin (the value carried in the column).
function _observed_outcomes(d::Competing, row::NamedTuple)
    out = Tuple{Int, Float64}[]
    for (i, name) in enumerate(d.names)
        haskey(row, name) || continue
        v = row[name]
        v === missing && continue
        push!(out, (i, Float64(v)))
    end
    return out
end

# The resolution time for the marginalise path: a reserved `resolved` field if
# present and non-missing, else a single bare event field (a non-reserved,
# non-outcome field) for backward compatibility with a plain `(resolve = y,)`
# row; `nothing` when no resolution time is supplied (fully missing).
function _competing_resolution_time(d::Competing, row::NamedTuple)
    if haskey(row, :resolved) && row.resolved !== missing
        return Float64(row.resolved)
    end
    bare = filter(
        k -> !(k in _COMPETING_RESERVED) && !(k in d.names), keys(row))
    isempty(bare) && return nothing
    length(bare) == 1 || throw(ArgumentError(
        "a Competing record with an unknown outcome takes one resolution time " *
        "(a `resolved` field or a single event value); got fields " *
        "$(collect(bare))"))
    v = row[only(bare)]
    return v === missing ? nothing : Float64(v)
end

_scale(lp, ::Nothing) = lp
_scale(lp, w) = w * lp

# Marginal log-density of a constant observation `x` under `d`, scaled by an
# optional weight `w`. Shared by the marginal composer and `Competing` models.
_marginal_logprob(d, x, ::Nothing) = logpdf(d, x)
_marginal_logprob(d, x, w) = w * logpdf(d, x)

# Marginal event-vector log-density of a censored composer, optionally right-
# truncated at the per-record observation horizon (hanta), scaled by the
# weight `w`. With `horizon === nothing` this is the untruncated composer logpdf
# (back-compat); a horizon routes through `event_logpdf`, which wraps each
# already-factorised observed segment in `truncate_to_horizon`.
function _marginal_logprob_h(d, events, w, horizon)
    lp = CensoredDistributions.event_logpdf(d, events; horizon = horizon)
    return w === nothing ? lp : w * lp
end

# --- Select (data-selected disjunction) ------------------------------------
#
# A `Select` routes a record to ONE of its independent alternatives, chosen by
# the row's selector field (`row[d.selector]`, default `:kind`). The selector
# VALUE is the alternative's name (a `Symbol`). `composed_distribution_model`
# reads that value, picks the alternative through the type-stable
# `CensoredDistributions._pick`, and delegates to the SELECTED alternative's own
# `composed_distribution_model` as a submodel. Because the alternative is itself
# any leaf or composer, the selected branch's full handling (marginal / latent /
# condition, weight, missingness) is exactly its own; `Select` adds only the
# data-driven routing. The selector field is stripped from the row before
# delegating so the alternative sees only its events (plus any reserved weight).
@model function composed_distribution_model(
        d::Select, row::NamedTuple; weight = nothing)
    # A column table (a `columntable` NamedTuple) is the whole table, not one
    # record; route it to the vectorised path. A scalar-valued row is not a table.
    if Tables.istable(row)
        weight === nothing || throw(ArgumentError(
            "the vectorised composed model takes per-row weights via a reserved " *
            "`weight`/`count` row field, not the `weight` keyword"))
        recs = CensoredDistributions.record_distributions(d, row)
        obs ~ to_submodel(_vectorised_records_model(
            recs, _record_obs_matrix(recs)))
        return obs
    end
    kind = row[d.selector]
    kind isa Symbol || throw(ArgumentError(
        "the Select selector field $(repr(d.selector)) must hold a Symbol " *
        "naming the alternative; got $(typeof(kind))"))
    chosen = CensoredDistributions._pick(d, kind)
    inner_row = _drop_field(row, d.selector)
    obs ~ DynamicPPL.to_submodel(
        composed_distribution_model(chosen, inner_row; weight = weight))
    return obs
end

# Drop a single named field from a NamedTuple, preserving the order of the rest.
# Used to remove the `Select` selector field before delegating to the chosen
# alternative, so the alternative's event dispatch sees only its own fields.
function _drop_field(row::NamedTuple, field::Symbol)
    ks = filter(!=(field), keys(row))
    return NamedTuple{ks}(map(k -> row[k], ks))
end

# VECTORISED Select entry: score a WHOLE TABLE of records whose top node is a
# `Select` in one `~`. Each record selects its alternative (and carries its own
# obs_time) per row via `record_distributions`, then the table scores AND samples
# with the standard `obs ~ product_distribution(...)` tilde, dual-purpose like the
# Sequential/Parallel vectorised entry.
function composed_distribution_model(
        d::Select, rows::AbstractVector; weight = nothing)
    weight === nothing || throw(ArgumentError(
        "the vectorised composed model takes per-row weights via a reserved " *
        "`weight`/`count` row field, not the `weight` keyword"))
    recs = CensoredDistributions.record_distributions(d, rows)
    return _vectorised_records_model(recs, _record_obs_matrix(recs))
end

# --- Latent composer models ------------------------------------------------
#
# A `latent`-wrapped composer turns its origin primary ON and declares every
# internal event time as a `~` latent INSIDE the model, mirroring the leaf latent
# (`p ~ get_primary_event(d); y ~ PrimaryConditional(d, p)`). Because each event
# is a `~`, the SAME model both fits (observed events score) and generates
# (missing events sample): a fully-missing row draws the complete event path, and
# a partially-observed row samples only the missing events while scoring the
# observed ones. Each sampled event lives in the VarInfo (named after the event
# field), so the chain is readable and groupable, and `prefix` at the call site
# namespaces per-record latents.

# A latent composer declares every internal event time as an indexed `~` over the
# SAME flat event layout the marginal scoring uses: entry 1 is the
# root origin `E_0`, then one slot per LEAF event in depth-first order. A FLAT
# chain/branch set is handled directly by the loop; a NESTED composer step/branch
# is unrolled to its leaf events by `_latent_leaf_plan`, so the latent twin
# recurses through an irregular tree exactly as the marginal `_tree_score` does.
# The plan is a PURE-INTEGER description of the tree (which leaf hangs off
# which already-sampled event, with which edge delay), built from the same
# `_child_nleaves` / `_terminal_offset` helpers; the model body then runs one
# indexed `~` per leaf, so a missing event samples and an observed event scores
# its gap through the edge delay. Keeping every `~` in the model body (the
# plan is just data) avoids nested-submodel prefixing and keeps the AD backends on
# the same shape they already differentiate for the flat latent loop.

# One leaf event in the latent sampling plan: the event slot `event_idx`, the
# event slot `shift_idx` it hangs off (its predecessor terminal in a chain, or the
# shared origin in a parallel set), the DECLARED `edge` delay distribution from
# that predecessor to this event, and `always_bare`. The bare-vs-declared density
# choice is made at SCORING time (see `_latent_edge`):
#   - `always_bare = false` (a `Sequential` step at any depth, or a nested
#     `Parallel` branch): the edge keeps DECLARED censoring only when BOTH its
#     endpoints are observed (#419); a sampled endpoint makes it bare.
#   - `always_bare = true` (the ROOT flat shared-origin `Parallel` branches): the
#     branch ALWAYS scores the bare core, matching the marginal
#     `_parallel_conditional_logpdf` (which conditions each branch on the
#     continuous core whether or not the shared origin is observed).
struct _LeafPlan{C}
    event_idx::Int
    shift_idx::Int
    edge::C
    always_bare::Bool
end

# Build the per-leaf sampling plan for a composer rooted with origin at event slot
# `origin_idx` and its first leaf event at `event_start`, appending to `plan`.
# Mirrors the marginal `_tree_score` layout (origin shared with the parent, leaf
# events contiguous), recursing into nested composer steps/branches. Pure integer
# + structure bookkeeping (no `~`, no sampled values), so it runs once up front.
# `root_parallel` marks the ROOT flat `Parallel` branches as `always_bare` (they
# couple through the shared origin and the marginal scores them bare); every other
# level passes `false`.
function _latent_plan!(plan, d::Sequential, origin_idx::Int, event_start::Int,
        root_parallel::Bool = false)
    comps = d.components
    o_idx = origin_idx
    ev_idx = event_start
    for step in comps
        # A Sequential step is never an always-bare root-Parallel branch.
        _latent_plan_step!(plan, step, o_idx, ev_idx, false)
        o_idx = ev_idx + CensoredDistributions._terminal_offset(step)
        # Advance by EVENT slots (a Competing step would span one slot per
        # outcome), matching the marginal scorer's layout.
        ev_idx += CensoredDistributions._event_child_nleaves(step)
    end
    return plan
end

function _latent_plan!(plan, d::Parallel, origin_idx::Int, event_start::Int,
        root_parallel::Bool = false)
    ev_idx = event_start
    for branch in d.components
        _latent_plan_step!(plan, branch, origin_idx, ev_idx, root_parallel)
        ev_idx += CensoredDistributions._event_child_nleaves(branch)
    end
    return plan
end

# A nested `Competing` in a LATENT-wrapped composer is out of scope: the
# latent turn-on must SAMPLE which outcome occurs and its time, a distinct
# multivariate construction not built here. Reject it clearly rather than
# mis-index the event slots. The MARGINAL composer model handles a nested
# Competing (conditioning / per-row branch_probs); turn the latent flow on for
# the rest of the tree, not for a Competing node.
function _latent_plan_step!(plan, ::Competing, o_idx::Int, ev_idx::Int,
        always_bare::Bool)
    throw(ArgumentError(
        "a latent-wrapped composer with a nested Competing node is not " *
        "supported; score it through the marginal composed model " *
        "(condition on the observed outcome / pass per-record branch_probs)"))
end

# A nested composer step/branch recurses on the same shared event vector: its
# origin is the parent's event slot `o_idx` and its own leaf events begin at
# `ev_idx`. A nested level is below the root, so its leaf edges follow the
# both-observed rule (`root_parallel = false`).
function _latent_plan_step!(plan, step::Union{Sequential, Parallel},
        o_idx::Int, ev_idx::Int, always_bare::Bool)
    return _latent_plan!(plan, step, o_idx, ev_idx, false)
end

# A leaf edge: one event at `ev_idx` hanging off the predecessor/origin `o_idx`
# through the DECLARED edge delay `step`. `always_bare` flags a root flat
# `Parallel` branch (scored bare regardless of observation); otherwise the
# bare-vs-declared choice is made at scoring time from the endpoints' missingness.
function _latent_plan_step!(plan, step::UnivariateDistribution,
        o_idx::Int, ev_idx::Int, always_bare::Bool)
    push!(plan, _LeafPlan(ev_idx, o_idx, step, always_bare))
    return plan
end

# A `latent`-wrapped `Sequential` chain spans events `E_0, ..., E_k` over the flat
# event layout (origin then one slot per leaf event). The split is DRIVEN BY THE
# ROW: an OBSERVED event (a non-missing row slot) CONDITIONS on its edge through
# `@addlogprob!` (its value is data, not a sampled latent), while a genuinely
# UNOBSERVED event (a missing slot) is SAMPLED with an indexed `~` so it lives in
# the VarInfo. This mirrors the leaf latent, whose observed `y` (a model argument)
# conditions while the primary `p` is sampled; declaring a free `~` on an observed
# slot would instead re-sample it as a latent and silently drop its likelihood.
# The origin `E_0` is the latent primary; each leaf event time is its predecessor
# plus the edge delay (a `_ShiftedDelay` over the edge as DECLARED so its censoring
# is kept, matching the marginal, with the predecessor shifted via `_latent_shift`
# so an interval-censored edge discretises the shift the same way the observed
# events are). The predecessor is read off `e` (whether sampled or observed). A
# nested composer step recurses through `_latent_plan!`, so an irregular tree
# splits its full sub-path the same way. The likelihood is scaled by the row weight
# via the conditional contribution.
@model function composed_distribution_model(
        d::Latent{<:Sequential}, row::NamedTuple; weight = nothing)
    # A NamedTuple of equal-length columns IS a Tables.jl table (a `columntable`),
    # not one record; route it to the batch latent path. A scalar-valued row
    # NamedTuple is not a table and scores as one record below.
    if Tables.istable(row)
        obs ~ to_submodel(_latent_batch_model(
            d, _latent_batch_rows(row); weight = weight))
        return obs
    end
    _reject_latent_horizon(row)
    chain = d.dist
    obs = _event_vector(chain, row)
    w = _row_weight(row, weight)

    origin = CensoredDistributions._origin_primary_event(
        CensoredDistributions._first_origin_node(chain))

    e = Vector{Union{Missing, Float64}}(obs)
    # The ORIGINAL missingness pattern: an edge whose TARGET slot is SAMPLED
    # (missing in the row) scores the BARE core (the marginal convolves bare cores
    # across the sampled run); an edge whose target is OBSERVED keeps its declared
    # censoring (see `_latent_edge`). Captured before any `~` fills a slot.
    sampled = [v === missing for v in e]
    plan = _latent_plan!(_LeafPlan[], chain, 1, 2, false)
    # Origin E_0. A MISSING origin must be SAMPLED, so the chain needs a primary
    # prior to sample it from (a bare-edge chain offers none -> reject). An OBSERVED
    # origin is a fixed continuous reference: it conditions through its primary prior
    # when the chain declares one, and contributes no prior term when it does not (a
    # bare-edge chain with an observed origin reference, the target's continuous
    # source-onset anchor). NOT weighted (the weight scales the LIKELIHOOD).
    if e[1] === missing
        origin === nothing && throw(ArgumentError(
            "latent Sequential model needs a censored origin to SAMPLE a missing " *
            "origin event (its first step must carry a primary event); supply the " *
            "origin as an observed reference, or declare a primary on the first " *
            "edge"))
        e[1] ~ origin
    elseif origin !== nothing
        DynamicPPL.@addlogprob! logpdf(origin, e[1])
    end
    for p in plan
        edge = _latent_edge(p.edge, e[p.shift_idx],
            sampled[p.shift_idx], p.shift_idx == 1,
            sampled[p.event_idx], p.always_bare)
        if e[p.event_idx] === missing
            e[p.event_idx] ~ edge
        else
            DynamicPPL.@addlogprob! _scale(logpdf(edge, e[p.event_idx]), w)
        end
    end
    return e
end

# A `latent`-wrapped `Parallel` shares one latent origin across its branches over
# the flat event layout `[O, leaf events...]`. The split is DRIVEN BY THE ROW like
# the latent Sequential: an observed branch (or origin) CONDITIONS on its edge
# through `@addlogprob!`, a missing one is SAMPLED with an indexed `~`. The root
# flat shared-origin Parallel conditions each branch on the continuous core
# (`strip = true` in the plan), matching the marginal `_parallel_conditional_logpdf`;
# a nested composer branch keeps its declared censoring. The shared origin couples
# the branches; each leaf event is the origin plus the branch delay. A nested
# composer branch recurses through `_latent_plan!`. The likelihood is scaled by the
# row weight via the conditional contribution.
@model function composed_distribution_model(
        d::Latent{<:Parallel}, row::NamedTuple; weight = nothing)
    # A column table is the whole table, not one record; route it to the batch
    # latent path (a scalar-valued row NamedTuple scores as one record below).
    if Tables.istable(row)
        obs ~ to_submodel(_latent_batch_model(
            d, _latent_batch_rows(row); weight = weight))
        return obs
    end
    _reject_latent_horizon(row)
    tree = d.dist
    obs = _event_vector(tree, row)
    w = _row_weight(row, weight)

    shared = CensoredDistributions._shared_primary_event(tree.components)

    e = Vector{Union{Missing, Float64}}(obs)
    # The ORIGINAL missingness pattern (captured before any `~` fills a slot)
    # drives the target-sampled bare-vs-declared choice (see `_latent_edge`); the
    # root flat Parallel additionally forces every branch bare (`always_bare`),
    # matching the marginal `_parallel_conditional_logpdf` / `_parallel_marginal_logpdf`.
    sampled = [v === missing for v in e]
    # Root flat Parallel branches are always bare (the marginal conditional scores
    # each branch on the continuous core).
    plan = _latent_plan!(_LeafPlan[], tree, 1, 2, true)
    # Shared origin. A MISSING origin must be SAMPLED, so the branches need a shared
    # primary prior; bare branches with an observed origin reference need none. NOT
    # weighted (weight scales the observed conditionals, not the prior).
    if e[1] === missing
        shared === nothing && throw(ArgumentError(
            "latent Parallel model needs censored branches sharing one primary " *
            "event to SAMPLE a missing shared origin; supply the origin as an " *
            "observed reference, or declare a shared primary"))
        e[1] ~ shared
    elseif shared !== nothing
        DynamicPPL.@addlogprob! logpdf(shared, e[1])
    end
    for p in plan
        edge = _latent_edge(p.edge, e[p.shift_idx],
            sampled[p.shift_idx], p.shift_idx == 1,
            sampled[p.event_idx], p.always_bare)
        if e[p.event_idx] === missing
            e[p.event_idx] ~ edge
        else
            DynamicPPL.@addlogprob! _scale(logpdf(edge, e[p.event_idx]), w)
        end
    end
    return e
end

# --- Batch latent entry: a whole table of records in one `~` -----------------
#
# The MARGINAL form scores a whole table in one `~` through
# `product_distribution(record_distributions(d, rows))`, because each marginal
# record is an independent distribution with no per-record latent. A LATENT
# record instead SAMPLES its own per-record latents (the origin and any
# unobserved intermediate event), so it cannot collapse to a single
# `product_distribution`; the per-record loop must stay. This batch entry moves
# that loop INTO the package: a looping `@model` submodel that delegates each
# record to the existing per-record latent model
# (`composed_distribution_model(d, row)`), prefixing record `i` with `:recN`
# exactly as the manual hand-written loop did. The user model then collapses to
# one tilde:
#
#     delays ~ to_submodel(composed_parameters_model(template, priors))
#     obs    ~ to_submodel(composed_distribution_model(latent(delays), rows))
#
# Mirrors the marginal batch method's signature (a vector of rows AND any
# Tables.jl table), so the marginal and latent batch entries are symmetric.

# A vector of rows: collect to NamedTuple rows up front (so the per-record
# delegation is a pure loop in the model body) and build the looping submodel.
function composed_distribution_model(
        d::Latent, rows::AbstractVector; weight = nothing)
    return _latent_batch_model(d, _latent_batch_rows(rows); weight = weight)
end

# Any Tables.jl table (e.g. a `DataFrame`): a column table is a Tables.jl source,
# so it passes straight in, mirroring the marginal table entry. A single
# `NamedTuple` row and a vector of rows keep their own (more specific) methods,
# so only a non-vector Tables.jl source lands here.
function composed_distribution_model(d::Latent, table; weight = nothing)
    Tables.istable(table) || throw(ArgumentError(
        "composed_distribution_model(latent(d), table) takes a vector of rows " *
        "or a Tables.jl table; got $(typeof(table))"))
    return _latent_batch_model(d, _latent_batch_rows(table); weight = weight)
end

# Normalise any Tables.jl row source to a `Vector` of `NamedTuple` rows, reusing
# the core row->NamedTuple helper so the batch path matches the per-record path.
function _latent_batch_rows(rows)
    rowvec = collect(Tables.rows(rows))
    isempty(rowvec) && throw(ArgumentError(
        "the batch latent model needs at least one record; got an empty table"))
    return [CensoredDistributions._row_namedtuple(r) for r in rowvec]
end

# The looping latent batch submodel: each record is scored by the existing
# per-record latent model, prefixed `:recN` so its sampled latents stay readable
# and groupable, exactly matching the manual loop. The per-record `weight`
# keyword is forwarded to every record (a per-record weight still rides on the
# row's reserved `weight`/`count` field).
@model function _latent_batch_model(
        d::Latent, rows::AbstractVector; weight = nothing)
    for i in eachindex(rows)
        obs ~ to_submodel(
            DynamicPPL.prefix(
                composed_distribution_model(d, rows[i]; weight = weight),
                Symbol(:rec, i)),
            false)
    end
    return nothing
end

# A location-shifted view of a delay distribution: `logpdf(shifted, y) =
# logpdf(delay, y - shift)` and `rand = shift + rand(delay)`, generalising
# [`PrimaryConditional`](@ref) to any edge delay (a continuous core OR a censored
# edge as declared, per `_latent_edge`). Used to declare each downstream event time
# as a `~` of its predecessor plus the edge delay, so the latent composer model
# both scores observed events and samples missing ones. Turing-free arithmetic;
# `shift` carries any sampled/AD type.
struct _ShiftedDelay{D, S} <: UnivariateDistribution{Distributions.Continuous}
    delay::D
    shift::S
end
Distributions.minimum(d::_ShiftedDelay) = d.shift + minimum(d.delay)
Distributions.maximum(d::_ShiftedDelay) = d.shift + maximum(d.delay)
Distributions.insupport(d::_ShiftedDelay, y::Real) = insupport(d.delay, y - d.shift)
Distributions.logpdf(d::_ShiftedDelay, y::Real) = logpdf(d.delay, y - d.shift)
Distributions.pdf(d::_ShiftedDelay, y::Real) = exp(logpdf(d, y))
Base.rand(rng::AbstractRNG, d::_ShiftedDelay) = d.shift + rand(rng, d.delay)

# The shifted edge distribution one leaf event is scored against, given its
# DECLARED `edge`, the predecessor value `shift`, whether the predecessor / target
# were SAMPLED, whether the predecessor is the chain ORIGIN, and `always_bare`
# (a root flat `Parallel` branch). The bare-vs-declared choice MATCHES the marginal
# scorer's segment factorisation edge-for-edge: the marginal splits the chain into
# SEGMENTS between consecutive OBSERVED events, scoring a single observed-bounded
# edge on its DECLARED censoring and convolving the BARE cores across a segment
# that spans a sampled (unobserved) intermediate.
#
#   - ROOT flat `Parallel` branch (`always_bare`): bare core, matching the marginal
#     `_parallel_conditional_logpdf` / `_parallel_marginal_logpdf` (each
#     shared-origin branch conditions / integrates on `_marginal_core`).
#   - TARGET SAMPLED (a mid-run latent intermediate): bare core — the target is
#     integrated out, so this edge is the head of a marginalised run the marginal
#     scores as a bare convolution.
#   - TARGET OBSERVED, PREDECESSOR a SAMPLED INTERMEDIATE (not the origin): bare
#     core — this edge is the TAIL of a run whose intermediate predecessor was
#     sampled, so the whole observed->observed segment is a bare convolution in the
#     marginal; scoring it bare makes latent-integrated == marginal.
#   - TARGET OBSERVED, PREDECESSOR OBSERVED, or the SAMPLED ORIGIN: DECLARED edge,
#     shifted by the FLOORED predecessor (`_latent_shift`). A single
#     observed-bounded edge conditions on its declared interval/double censoring
#     (#419); the origin->first-observed edge keeps the declared edge and floors
#     the sampled continuous origin so the floored gap stays in-support (#423).
function _latent_edge(edge, shift, pred_sampled::Bool, pred_is_origin::Bool,
        target_sampled::Bool, always_bare::Bool)
    bare = always_bare || target_sampled ||
           (pred_sampled && !pred_is_origin)
    if bare
        return _ShiftedDelay(_bare_edge(edge), shift)
    end
    return _ShiftedDelay(edge, _latent_shift(edge, shift))
end

# The BARE continuous core of an edge for a sampled-endpoint latent edge
# (delegates to the shared core helper): every censoring layer stripped (primary
# AND secondary interval), matching the `_marginal_core` the marginal scorer
# conditions a sampled-endpoint edge on, so latent-integrated == marginal.
_bare_edge(edge) = CensoredDistributions._bare_latent_edge(edge)

# The predecessor value an interval-censored latent edge is shifted by when the
# predecessor is OBSERVED. The observed downstream events of a chain are
# DISCRETISED (floored to the secondary interval), but a continuous predecessor is
# not. Scoring `logpdf(edge, target - shift)` with a continuous shift compares a
# floored target against a continuous predecessor, so a target that floors into the
# SAME interval as its predecessor gives a NEGATIVE gap (out of support, `-Inf`)
# for any shift above the floored target -- the `double_interval_censored`
# latent-init failure of #423. Flooring the shift to the edge's interval
# discretises the predecessor the SAME way the observed events are, so the scored
# gap is `floor(target) - floor(shift)` (matching the marginal, which scores the
# floored origin's observed gap) and stays in-support. An edge without secondary
# interval censoring keeps the continuous shift unchanged.
function _latent_shift(edge, shift)
    iv = CensoredDistributions._leaf_interval(edge)
    iv === nothing && return shift
    return CensoredDistributions._apply_leaf_interval(shift, iv)
end

# ===========================================================================
# composed_parameters_model: priors -> sampled, reconstructed composed dist
# ===========================================================================
#
# `composed_parameters_model(template, priors)` is a submodel that SAMPLES a
# composed distribution's free parameters from user priors and RETURNS the
# reconstructed distribution, ready for the matching record submodel to score.
# It completes the prior workflow: `compose` -> `params_table` (the
# inventory) -> the user keys priors against it -> this helper materialises
# the sampling submodel.
#
# `priors` is a nested `NamedTuple` mirroring `params(template)`: a leaf is keyed
# by its parameter names (`_leaf_param_names`), a `Sequential`/`Parallel` by its
# edge/event names, a `Competing` by its outcome names plus an optional
# `branch_probs` entry. The traversal REUSES the composers' `component_names` and
# the leaf `params`/`_leaf_param_names` introspection for both structure
# and names, never re-walking the tree by hand, and rebuilds the SAME structure
# and names so the result matches the record submodel's by-name expectations
# (Option A).
#
# Names are namespaced by edge path through nested submodel prefixing
# (`DynamicPPL.prefix(child_model, Val(name))`), so a multi-edge chain produces
# readable, groupable Turing chain names like `onset_admit.shape` and
# `resolution.death.scale`.

# --- prior-key validation --------------------------------------------------

# Validate that `priors` (a NamedTuple) covers EXACTLY `expected` (a tuple of
# names) at the current node, with a clear error on a missing or extra key. The
# `what` label names the node in the message (e.g. `"edge :onset_admit"`).
function _check_prior_keys(priors::NamedTuple, expected::Tuple, what::AbstractString)
    have = keys(priors)
    missing_keys = filter(k -> !(k in have), expected)
    extra_keys = filter(k -> !(k in expected), have)
    isempty(missing_keys) || throw(ArgumentError(
        "$what is missing priors for $(collect(missing_keys)); " *
        "expected $(collect(expected))"))
    isempty(extra_keys) || throw(ArgumentError(
        "$what has unexpected prior keys $(collect(extra_keys)); " *
        "expected $(collect(expected))"))
    return nothing
end

function _check_prior_keys(priors, ::Tuple, what::AbstractString)
    throw(ArgumentError(
        "$what expects a NamedTuple of priors; got $(typeof(priors))"))
end

# --- leaf reconstruction ---------------------------------------------------

# The base (un-parameterised) constructor of a leaf distribution, so a leaf
# reconstructs from sampled parameters carrying any (AD) element type rather than
# the template's concrete one (e.g. `Gamma` from a `Gamma{Float64}` template).
# Resolves the INNER free delay of a (possibly censored) leaf so a censored leaf
# rebuilds its delay family, not the censoring wrapper.
_base_ctor(leaf) = Base.typename(typeof(CensoredDistributions.free_leaf(leaf))).wrapper

# Reconstruct a leaf from sampled parameters. For a censored leaf the inner free
# delay is rebuilt from the params and the FIXED censoring is re-applied via
# `rewrap_leaf`, so a `double_interval_censored(Gamma)` round-trips to the same
# censored distribution. Argument checks are skipped (`check_args = false`) so a
# sampler probing an out-of-support point yields `-Inf` rather than throwing
# mid-gradient; families whose constructor lacks the keyword fall back to the
# plain (checked) constructor.
function _reconstruct_leaf(leaf, vals::Tuple)
    ctor = _base_ctor(leaf)
    inner = _construct_unchecked(ctor, vals)
    return CensoredDistributions.rewrap_leaf(leaf, inner)
end

function _construct_unchecked(ctor, vals::Tuple)
    try
        return ctor(vals...; check_args = false)
    catch err
        err isa MethodError || rethrow()
        return ctor(vals...)
    end
end

# --- recursive submodel builder --------------------------------------------
#
# `_params_submodel(template, priors)` returns a DynamicPPL submodel sampling the
# node's parameters and returning the reconstructed node. Each composer level
# nests a child submodel per named child, prefixed by that child's name, so the
# sampled parameter names carry the full edge path. Leaf parameters are sampled
# directly under their parameter names via `tilde_assume!!`, so a leaf's chain
# names are exactly `<path>.<param>` (no synthetic inner variable).

# A leaf: sample each parameter (in `params` order) from its named prior and
# rebuild the leaf via its base constructor. `tilde_assume!!` with `VarName{p}()`
# gives each sampled parameter the bare name `p`; the enclosing submodel prefixes
# add the edge path, so the chain name is `<edge>.<p>`. A SHARED-tagged leaf is
# NOT sampled here: its value tuple is already in `shared` (sampled once up
# front), so the occurrence reconstructs from that tracked tuple, re-applying its
# own censoring.
@model function _leaf_params_model(leaf, priors::NamedTuple, shared)
    tag = CensoredDistributions._shared_tag(leaf)
    if tag !== nothing
        return _reconstruct_leaf(leaf, shared[tag])
    end
    pnames = CensoredDistributions._leaf_param_names(leaf)
    _check_prior_keys(priors, pnames, "leaf $(nameof(typeof(leaf)))")
    ctx = __model__.context
    vals = ntuple(length(pnames)) do i
        p = pnames[i]
        v,
        __varinfo__ = DynamicPPL.tilde_assume!!(
            ctx, priors[p], VarName{p}(), nothing, __varinfo__)
        v
    end
    return _reconstruct_leaf(leaf, vals)
end

# A child's per-occurrence priors: a shared-tagged child carries no per-occurrence
# prior (its prior lives at the top level under the tag), so an absent key yields
# an empty NamedTuple and the leaf reads `shared`.
function _child_priors(priors::NamedTuple, name::Symbol)
    haskey(priors, name) ? priors[name] : NamedTuple()
end

# A `Sequential` / `Parallel`: sample each named child through a prefixed child
# submodel, then rebuild the SAME composer type with the SAME names. A child whose
# only params are shared has no own prior key and samples nothing locally.
@model function _composer_params_model(
        d::Union{Sequential, Parallel}, priors::NamedTuple, shared)
    names = component_names(d)
    _check_composer_prior_keys(priors, names, "$(nameof(typeof(d)))", shared)
    parts = Vector{Any}(undef, length(names))
    for i in 1:length(names)
        name = names[i]
        child = d.components[i]
        sub = DynamicPPL.prefix(
            _params_submodel(child, _child_priors(priors, name), shared),
            Val(name))
        parts[i] ~ to_submodel(sub, false)
    end
    return _rebuild(d, Tuple(parts))
end

# A `Select`: sample each named alternative through a prefixed child submodel; a
# tag shared across alternatives is sampled once (up front) and reused, so the
# alternative that only carries the shared parameter samples nothing locally.
@model function _select_params_model(d::Select, priors::NamedTuple, shared)
    _check_composer_prior_keys(priors, d.names, "Select", shared)
    alts = Vector{Any}(undef, length(d.names))
    for i in 1:length(d.names)
        name = d.names[i]
        sub = DynamicPPL.prefix(
            _params_submodel(d.alternatives[i], _child_priors(priors, name),
                shared), Val(name))
        alts[i] ~ to_submodel(sub, false)
    end
    return Select(d.names, Tuple(alts), d.selector)
end

# A `Competing`: sample each outcome delay through a prefixed child submodel; the
# branch probabilities are kept fixed from the template unless a `branch_probs`
# entry of priors is supplied (then each is sampled, prefixed under
# `branch_probs`). Rebuild the `Competing` with the SAME outcome names.
@model function _competing_params_model(c::Competing, priors::NamedTuple, shared)
    expected = (c.names..., :branch_probs)
    have = keys(priors)
    # `branch_probs` is optional; every other expected key is required.
    required = c.names
    _check_competing_keys(priors, required, expected, shared)
    delays = Vector{Any}(undef, length(c.names))
    for i in 1:length(c.names)
        name = c.names[i]
        sub = DynamicPPL.prefix(
            _params_submodel(c.delays[i], _child_priors(priors, name), shared),
            Val(name))
        delays[i] ~ to_submodel(sub, false)
    end
    if :branch_probs in have
        bp_priors = priors.branch_probs
        _check_prior_keys(bp_priors, c.names, "Competing branch_probs")
        sub = DynamicPPL.prefix(
            _branch_probs_model(c, bp_priors), Val(:branch_probs))
        probs ~ to_submodel(sub, false)
    else
        probs = c.branch_probs
    end
    return Competing(c.names, Tuple(delays), Tuple(probs))
end

# A racing-hazard `HazardCompeting`: sample each racing outcome delay through a
# prefixed child submodel and rebuild with the SAME outcome names. There is NO
# `branch_probs` block (the winning probability is derived from the hazards).
@model function _hazard_competing_params_model(
        c::CensoredDistributions.HazardCompeting, priors::NamedTuple, shared)
    _check_composer_prior_keys(priors, c.names, :HazardCompeting, shared)
    delays = Vector{Any}(undef, length(c.names))
    for i in 1:length(c.names)
        name = c.names[i]
        sub = DynamicPPL.prefix(
            _params_submodel(c.delays[i], _child_priors(priors, name), shared),
            Val(name))
        delays[i] ~ to_submodel(sub, false)
    end
    return CensoredDistributions.HazardCompeting(c.names, Tuple(delays))
end

# Sample the competing branch probabilities from their named priors. Returns the
# tuple in outcome order; `tilde_assume!!` names each by its outcome name so the
# chain names are `branch_probs.<outcome>`.
@model function _branch_probs_model(c::Competing, priors::NamedTuple)
    ctx = __model__.context
    probs = ntuple(length(c.names)) do i
        name = c.names[i]
        v,
        __varinfo__ = DynamicPPL.tilde_assume!!(
            ctx, priors[name], VarName{name}(), nothing, __varinfo__)
        v
    end
    return probs
end

# `branch_probs` is optional, the outcome names are required UNLESS an outcome's
# only params are shared (its prior is top-level, no per-outcome key): validate the
# required outcome priors are present and no key outside `expected`/shared appears.
function _check_competing_keys(priors::NamedTuple, required::Tuple,
        expected::Tuple, shared)
    have = keys(priors)
    missing_keys = filter(k -> !(k in have), required)
    extra_keys = filter(k -> !(k in expected) && !(k in keys(shared)), have)
    isempty(missing_keys) || throw(ArgumentError(
        "Competing is missing priors for $(collect(missing_keys)); " *
        "expected outcomes $(collect(required))"))
    isempty(extra_keys) || throw(ArgumentError(
        "Competing has unexpected prior keys $(collect(extra_keys)); " *
        "expected $(collect(expected))"))
    return nothing
end

# A composer node's child prior keys. A child key may be ABSENT (its only params
# are shared; the prior is top-level under the tag), so missing names are
# tolerated; an UNEXPECTED key (not a child name or a shared tag) errors. With no
# shared tags this is the exact-cover check of `_check_prior_keys`.
function _check_composer_prior_keys(priors::NamedTuple, names::Tuple, what, shared)
    no_shared = isempty(keys(shared))
    if no_shared
        return _check_prior_keys(priors, names, what)
    end
    allowed = (names..., keys(shared)...)
    extra_keys = filter(k -> !(k in allowed), keys(priors))
    isempty(extra_keys) || throw(ArgumentError(
        "$what has unexpected prior keys $(collect(extra_keys)); " *
        "expected $(collect(names))"))
    return nothing
end

# Dispatch a node to its parameter submodel: a composer to its composer model, a
# `Competing` to its competing model, any other (leaf) distribution to the leaf
# model. Mirrors the `params`/`params_table` traversal dispatch. The
# `shared` NamedTuple (tag -> sampled value tuple) is threaded so a shared-tagged
# leaf reuses the one sampled group rather than sampling again.
function _params_submodel(d::Union{Sequential, Parallel}, priors, shared)
    _composer_params_model(d, priors, shared)
end
_params_submodel(d::Select, priors, shared) = _select_params_model(d, priors, shared)
_params_submodel(c::Competing, priors, shared) = _competing_params_model(c, priors, shared)
function _params_submodel(
        c::CensoredDistributions.HazardCompeting, priors, shared)
    return _hazard_competing_params_model(c, priors, shared)
end
_params_submodel(leaf, priors, shared) = _leaf_params_model(leaf, priors, shared)

# `_rebuild` (preserve composer type + names) is shared with the core `update`
# helper; reuse it rather than redefining.
const _rebuild = CensoredDistributions._rebuild

# ---------------------------------------------------------------------------
# Shared-parameter tracking (tie a leaf across branches by name)
# ---------------------------------------------------------------------------
#
# A shared-tagged leaf (`shared(:inc, dist)`) is ONE free parameter even when it
# occurs in several branches. Its prior lives at the TOP level of `priors` under
# the tag (matching `params_table`'s tag edge). The public model samples each
# shared group ONCE up front (named `tag.param`, prefixed by the tag), then
# threads the sampled value tuples down as `shared`; every occurrence of the tag
# reconstructs from the one sampled tuple, re-applying its OWN censoring, so the
# same tracked value flows to all occurrences (AD-safe).

# Sample one shared group from its top-level prior, returning its sampled value
# tuple. Names each parameter `tag.param` via a prefixed leaf-sampling submodel.
@model function _shared_group_model(leaf, priors::NamedTuple)
    pnames = CensoredDistributions._leaf_param_names(leaf)
    _check_prior_keys(priors, pnames, "shared $(repr(CensoredDistributions._shared_tag(leaf)))")
    ctx = __model__.context
    vals = ntuple(length(pnames)) do i
        p = pnames[i]
        v,
        __varinfo__ = DynamicPPL.tilde_assume!!(
            ctx, priors[p], VarName{p}(), nothing, __varinfo__)
        v
    end
    return vals
end

# The public entry. Samples each shared group once (prefixed by its tag), then
# builds the tree, threading the sampled shared value tuples to every occurrence.
# With no shared tags this is exactly the per-node recursive builder.
@model function composed_parameters_model(template, priors)
    groups = CensoredDistributions._collect_shared(template)
    shared = NamedTuple()
    for (tag, leaf) in groups
        haskey(priors, tag) || throw(ArgumentError(
            "missing prior for shared parameter $(repr(tag)); expected a " *
            "top-level `$tag` entry in the priors"))
        sub = DynamicPPL.prefix(_shared_group_model(leaf, priors[tag]), Val(tag))
        vals ~ to_submodel(sub, false)
        shared = merge(shared, NamedTuple{(tag,)}((vals,)))
    end
    d ~ to_submodel(_params_submodel(template, priors, shared), false)
    return d
end

end
