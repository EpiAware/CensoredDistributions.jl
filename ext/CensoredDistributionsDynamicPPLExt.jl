module CensoredDistributionsDynamicPPLExt

# DynamicPPL methods for the submodel constructors declared (and documented) in
# `src/turing_models.jl`. Loaded only when DynamicPPL is available, keeping the
# core Turing-free.

using CensoredDistributions: CensoredDistributions, PrimaryCensored, Latent,
                             IntervalCensored, PrimaryConditional, Sequential,
                             Parallel, Competing, Select, as_mixture,
                             get_primary_event, get_dist_recursive,
                             convolve_distributions, component_names,
                             tree_event_names, primary_censored
import CensoredDistributions: primary_censored_model, interval_censored_model,
                              double_interval_censored_model,
                              composed_distribution_model,
                              composed_parameters_model, predict_events
using DynamicPPL: DynamicPPL, @model, to_submodel, VarName
using Distributions: Distributions, UnivariateDistribution, params, logpdf,
                     MixtureModel
using Random: AbstractRNG, default_rng

# `CensoredDistributions.weight(d, w)` is called with the module qualifier
# because the `weight` keyword argument shadows the function name inside the
# `@model` bodies. `weight(d, nothing)` returns `d` unweighted, so one `~`
# statement covers both the weighted and unweighted cases.
const _weight = CensoredDistributions.weight

# ===========================================================================
# Leaf submodels (#88, PR2)
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
# Generic record entry: composed_distribution_model (#335, PR3d; #350)
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
# dispatch (#329). A reserved `weight`/`count` field (or the `weight =` keyword)
# scales the likelihood and is EXCLUDED from the event dispatch.
#
# Marginal vs latent is DISPATCH on the STRUCT TYPE (not a runtime predicate),
# mirroring the leaf (`PrimaryCensored` marginal, `Latent` latent):
#   - a bare composer (`Sequential` / `Parallel` / `Competing`) is MARGINAL: it
#     scores the row vector via `~ d`, so the censored composer `logpdf`
#     auto-marginalises unobserved intermediates / the origin and conditions on
#     observed events (#329). The submodel log-density equals the direct
#     `logpdf(d, event_vector)`, and missing fields drive the per-record path.
#   - a `latent`-wrapped composer turns the shared/origin primary ON: the origin
#     `o ~ ...` is declared INSIDE the model (so the sampled origin lives in the
#     VarInfo and the model fits AND generates the full event path), each
#     observed downstream event conditions on `o` through its own edge (#329),
#     and unobserved intermediates marginalise by convolving the adjacent cores.
# Per-record/branch latents are namespaced by `prefix` at the call site (see the
# composer tests), so chains stay readable and groupable.

# --- Leaf delegation -------------------------------------------------------
#
# A leaf/univariate record routes to its correctly named leaf model. The record
# may arrive as a bare observed value or as a one-event `NamedTuple` row (with an
# optional reserved weight); `_leaf_value`/`_leaf_weight` normalise both. The
# `weight =` keyword still applies and overrides a row weight.

# A `PrimaryCensored` (marginal) or its `Latent` wrapper -> the primary-censored
# leaf model.
function composed_distribution_model(
        d::Union{PrimaryCensored, Latent{<:PrimaryCensored}}, row;
        weight = nothing)
    return primary_censored_model(
        d, _leaf_value(row); weight = _leaf_weight(row, weight))
end

# An `IntervalCensored` -> the interval-censored leaf model.
function composed_distribution_model(
        d::IntervalCensored, row; weight = nothing)
    return interval_censored_model(
        d, _leaf_value(row); weight = _leaf_weight(row, weight))
end

# Any other univariate leaf (a `double_interval_censored` pipeline, a
# `Truncated`, a `convolve_distributions` result, ...) -> the marginal univariate
# leaf model. A bare `Latent{<:PrimaryCensored}` is handled above; the composer
# `Latent{<:Sequential}` / `Latent{<:Parallel}` methods below are multivariate
# and dispatch ahead of this univariate fallback.
function composed_distribution_model(
        d::UnivariateDistribution, row; weight = nothing)
    return double_interval_censored_model(
        d, _leaf_value(row); weight = _leaf_weight(row, weight))
end

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

# Reserved row fields that are NOT events: a multiplicity weight may ride in the
# row as `weight` or `count`.
const _RESERVED_ROW_FIELDS = (:weight, :count)

# The event values of a row in field order, dropping the reserved weight/count
# fields, returned as a `Vector{Union{Missing, Float64}}` (one entry per event,
# `missing` admitted). The `Missing`-admitting element type keeps the censored
# composer `logpdf` specialisation (#329) selected even for an all-observed row.
# The remaining field ORDER is the event order the composer spans. This is the
# POSITIONAL fallback, used only when a composer carries no derivable event names
# (its edges are positional defaults), per #362.
function _event_vector(row::NamedTuple)
    ks = filter(k -> !(k in _RESERVED_ROW_FIELDS), keys(row))
    out = Vector{Union{Missing, Float64}}(undef, length(ks))
    for (i, k) in enumerate(ks)
        v = row[k]
        out[i] = v === missing ? missing : Float64(v)
    end
    return out
end

# The event vector for a composer `d` from a `row`, matched to the tree's flat
# EVENT names BY NAME (#362): `row.onset, row.admit, row.death` land in their slots
# regardless of field order, `missing` fields drive the marginalise/condition
# dispatch, and a reserved `weight`/`count` is excluded. A row field that is not a
# tree event, or a missing required event, raises a clear `ArgumentError`. When
# the tree's event names are all positional defaults (`:event_i`), the row is
# matched POSITIONALLY through `_event_vector(row)` (the documented fallback).
function _event_vector(d::Union{Sequential, Parallel}, row::NamedTuple)
    enames = tree_event_names(d)
    CensoredDistributions._all_positional_event_names(enames) &&
        return _event_vector(row)
    return _event_vector_by_name(enames, row)
end

# Build the by-name event vector: validate every non-reserved row field is a known
# event, then place each event by name (a missing required event errors). Pure,
# Turing-free.
function _event_vector_by_name(enames::Tuple, row::NamedTuple)
    for k in keys(row)
        k in _RESERVED_ROW_FIELDS && continue
        k in enames || throw(ArgumentError(
            "row field $(repr(k)) is not an event of this tree; expected " *
            "events $(collect(enames)) (reordering is allowed; names are not)"))
    end
    out = Vector{Union{Missing, Float64}}(undef, length(enames))
    for (i, name) in enumerate(enames)
        haskey(row, name) || throw(ArgumentError(
            "row is missing required event $(repr(name)); expected events " *
            "$(collect(enames))"))
        v = row[name]
        out[i] = v === missing ? missing : Float64(v)
    end
    return out
end

# The multiplicity weight carried by a row: an explicit `weight =` keyword wins,
# otherwise a reserved `weight`/`count` field, otherwise `nothing` (unweighted).
function _row_weight(row::NamedTuple, kw_weight)
    kw_weight === nothing || return kw_weight
    haskey(row, :weight) && return row.weight
    haskey(row, :count) && return row.count
    return nothing
end

# --- Marginal composer models ----------------------------------------------

# Marginal `Sequential` / `Parallel`: score the row's event vector directly
# through the censored composer `logpdf`. The whole row is DATA (carried in the
# `row` argument), so the contribution is added with `@addlogprob!` rather than a
# sampled `~`, leaving no spurious VarInfo variable (matching the leaf marginal,
# which has no latent). The composer dispatches on the event vector's
# `Missing`-admitting element type and per-record missingness pattern (#329):
# unobserved intermediates / a missing origin marginalise, observed events
# condition on their declared censoring. The contribution equals
# `logpdf(d, event_vector)`, scaled by the row weight, so the submodel
# log-density equals the direct `logpdf`.
@model function composed_distribution_model(
        d::Union{Sequential, Parallel}, row::NamedTuple; weight = nothing)
    events = _event_vector(d, row)
    w = _row_weight(row, weight)
    DynamicPPL.@addlogprob! _marginal_logprob(d, events, w)
    return events
end

# A `Competing` node SELF-DISPATCHES on the row's outcome missingness (#329
# decision 2), mirroring the observed-intermediate dispatch of a chain:
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
#       is how a covariate CFR `logistic(Xβ)` (computed in PLAIN TURING, #329
#       decision 2) flows in per record; the regression stays OUT of the node, the
#       node only CONSUMES the probability. Validated to lie in `[0, 1]` and (for
#       a NamedTuple) to sum to one across outcomes.
#   (c) unknown outcome       -> the probabilities enter only as the mixture
#       weights of the marginalise path.
#
# The observed OUTCOME (which outcome column is non-missing, Feature 1's by-name
# missingness) and the per-record PROBABILITY (the reserved `branch_probs` field)
# are DISTINCT row inputs (#362 interaction). The whole record is DATA, so the
# contribution is added with `@addlogprob!` (no spurious VarInfo variable).
@model function composed_distribution_model(
        d::Competing, row::NamedTuple; weight = nothing)
    w = _row_weight(row, weight)
    probs = _competing_branch_probs(d, row)
    DynamicPPL.@addlogprob! _competing_logprob(d, row, probs, w)
    return nothing
end

# Reserved row fields specific to a `Competing` node: the resolution time when the
# outcome is unknown, and the per-record branch-probability override.
const _COMPETING_RESERVED = (_RESERVED_ROW_FIELDS..., :resolved, :branch_probs)

# The branch probabilities to USE for this record: a reserved `branch_probs` row
# field overrides the node's stored probabilities (regime b), else the stored ones
# (regime a). A NamedTuple override is reordered to outcome order and validated; a
# scalar override is taken as the FIRST outcome's probability of a two-outcome
# node (`(p, 1 - p)`).
function _competing_branch_probs(d::Competing, row::NamedTuple)
    haskey(row, :branch_probs) || return d.branch_probs
    return _coerce_branch_probs(d, row.branch_probs)
end

# The probabilities keep their (possibly AD `Dual`) element type so a covariate
# CFR `logistic(Xβ)` carrying a `Dual` differentiates through the node.
function _coerce_branch_probs(d::Competing, bp::NamedTuple)
    Set(keys(bp)) == Set(d.names) || throw(ArgumentError(
        "per-record branch_probs must name exactly the outcomes " *
        "$(collect(d.names)); got $(collect(keys(bp)))"))
    probs = map(n -> bp[n], d.names)
    _validate_record_probs(probs)
    return probs
end

function _coerce_branch_probs(d::Competing, p::Real)
    length(d.names) == 2 || throw(ArgumentError(
        "a scalar per-record branch_probs is only defined for a two-outcome " *
        "Competing (the first outcome's probability); node has " *
        "$(length(d.names)) outcomes, pass a NamedTuple instead"))
    probs = (p, one(p) - p)
    _validate_record_probs(probs)
    return probs
end

# Per-record branch probabilities must each lie in `[0, 1]` and sum to one.
function _validate_record_probs(probs)
    all(p -> p >= 0 && p <= 1, probs) || throw(ArgumentError(
        "each per-record branch probability must lie in [0, 1]; got " *
        "$(collect(probs))"))
    isapprox(sum(probs), 1; atol = 1e-8) || throw(ArgumentError(
        "per-record branch probabilities sum to $(sum(probs)), not one"))
    return nothing
end

# The competing log-density for one record under branch probabilities `probs`,
# scaled by the weight `w`. Selects condition / marginalise / fully-missing from
# the row's outcome missingness.
function _competing_logprob(d::Competing, row::NamedTuple, probs, w)
    obs = _observed_outcomes(d, row)
    if length(obs) == 1
        # Observed outcome: condition on that branch.
        i, gap = obs[1]
        lp = log(probs[i]) + logpdf(d.delays[i], gap)
        return _scale(lp, w)
    end
    isempty(obs) || throw(ArgumentError(
        "a Competing record may observe at most one outcome time; got " *
        "$(length(obs)) ($(collect(first.(obs))))"))
    # No outcome observed: marginalise at the resolution time if known, else the
    # record is fully missing and contributes nothing.
    t = _competing_resolution_time(d, row)
    t === nothing && return _scale(zero(eltype(probs)), w)
    mix = MixtureModel(collect(d.delays), collect(float.(probs)))
    return _scale(logpdf(mix, t), w)
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
# SAME flat event layout the marginal scoring uses (#329, #361): entry 1 is the
# root origin `E_0`, then one slot per LEAF event in depth-first order. A FLAT
# chain/branch set is handled directly by the loop; a NESTED composer step/branch
# is unrolled to its leaf events by `_latent_leaf_plan`, so the latent twin
# recurses through an irregular tree exactly as the marginal `_tree_score` does
# (#361). The plan is a PURE-INTEGER description of the tree (which leaf hangs off
# which already-sampled event, with which continuous core), built from the same
# `_child_nleaves` / `_terminal_offset` helpers; the model body then runs one
# indexed `~` per leaf, so a missing event samples and an observed event scores
# its gap through the edge core (#329). Keeping every `~` in the model body (the
# plan is just data) avoids nested-submodel prefixing and keeps the AD backends on
# the same shape they already differentiate for the flat latent loop.

# One leaf event in the latent sampling plan: the event slot `event_idx`, the
# already-sampled event slot `shift_idx` it hangs off (its predecessor terminal in
# a chain, or the shared origin in a parallel set), and the continuous delay
# `core` from that predecessor to this event. Built by a pure walk over the tree.
struct _LeafPlan{C}
    event_idx::Int
    shift_idx::Int
    core::C
end

# Build the per-leaf sampling plan for a composer rooted with origin at event slot
# `origin_idx` and its first leaf event at `event_start`, appending to `plan`.
# Mirrors the marginal `_tree_score` layout (origin shared with the parent, leaf
# events contiguous), recursing into nested composer steps/branches. Pure integer
# + structure bookkeeping (no `~`, no sampled values), so it runs once up front.
function _latent_plan!(plan, d::Sequential, origin_idx::Int, event_start::Int)
    comps = d.components
    o_idx = origin_idx
    ev_idx = event_start
    for step in comps
        _latent_plan_step!(plan, step, o_idx, ev_idx)
        o_idx = ev_idx + CensoredDistributions._terminal_offset(step)
        ev_idx += CensoredDistributions._child_nleaves(step)
    end
    return plan
end

function _latent_plan!(plan, d::Parallel, origin_idx::Int, event_start::Int)
    ev_idx = event_start
    for branch in d.components
        _latent_plan_step!(plan, branch, origin_idx, ev_idx)
        ev_idx += CensoredDistributions._child_nleaves(branch)
    end
    return plan
end

# A nested composer step/branch recurses on the same shared event vector: its
# origin is the parent's event slot `o_idx` and its own leaf events begin at
# `ev_idx`.
function _latent_plan_step!(plan, step::Union{Sequential, Parallel},
        o_idx::Int, ev_idx::Int)
    return _latent_plan!(plan, step, o_idx, ev_idx)
end

# A leaf edge: one event at `ev_idx` hanging off the predecessor/origin `o_idx`
# through the edge's continuous core.
function _latent_plan_step!(plan, step::UnivariateDistribution,
        o_idx::Int, ev_idx::Int)
    core = get_dist_recursive(step)
    push!(plan, _LeafPlan(ev_idx, o_idx, core))
    return plan
end

# A `latent`-wrapped `Sequential` chain spans events `E_0, ..., E_k` over the flat
# event layout (origin then one slot per leaf event). The origin `E_0` is the
# latent primary (`e[1] ~ origin`); each leaf event time is its predecessor plus
# the edge delay, declared as a shifted `~` (`e[j] ~ _ShiftedDelay(core, e[s])`),
# so an observed event scores its gap through the edge's declared censoring (#329)
# and a missing event samples it. A nested composer step recurses through
# `_latent_plan!`, so an irregular tree samples its full sub-path. Indexed
# VarNames `e[i]` give each event a distinct name; the likelihood is scaled by the
# row weight via the `~`.
@model function composed_distribution_model(
        d::Latent{<:Sequential}, row::NamedTuple; weight = nothing)
    chain = d.dist
    obs = _event_vector(chain, row)
    w = _row_weight(row, weight)

    origin = CensoredDistributions._origin_primary_event(
        CensoredDistributions._first_origin_node(chain))
    origin === nothing && throw(ArgumentError(
        "latent Sequential model needs a censored origin (its first step " *
        "must carry a primary event)"))

    e = Vector{Union{Missing, Float64}}(obs)
    plan = _latent_plan!(_LeafPlan[], chain, 1, 2)
    # Origin E_0: the latent primary prior, declared but NOT weighted (the weight
    # scales the LIKELIHOOD, the observed conditionals, not the prior). Indexed
    # `~` so a missing origin samples it.
    e[1] ~ origin
    for p in plan
        edge = _ShiftedDelay(p.core, e[p.shift_idx])
        e[p.event_idx] ~ _weight(edge, w)
    end
    return e
end

# A `latent`-wrapped `Parallel` shares one latent origin across its branches over
# the flat event layout `[O, leaf events...]`. The shared origin `e[1] ~ shared`
# is declared once; each leaf event is its predecessor/origin plus the branch
# delay, declared as a shifted `~`, so an observed branch scores `logpdf(core,
# y - o)` and a missing branch samples its observation. A nested composer branch
# recurses through `_latent_plan!` so its whole sub-path is sampled off the shared
# origin. The shared origin couples the branches; likelihood scaled by the weight.
@model function composed_distribution_model(
        d::Latent{<:Parallel}, row::NamedTuple; weight = nothing)
    tree = d.dist
    obs = _event_vector(tree, row)
    w = _row_weight(row, weight)

    shared = CensoredDistributions._shared_primary_event(tree.components)
    shared === nothing && throw(ArgumentError(
        "latent Parallel model needs censored branches sharing one primary " *
        "event"))

    e = Vector{Union{Missing, Float64}}(obs)
    plan = _latent_plan!(_LeafPlan[], tree, 1, 2)
    # Shared origin prior, declared but NOT weighted (weight scales the observed
    # conditionals, not the prior).
    e[1] ~ shared
    for p in plan
        edge = _ShiftedDelay(p.core, e[p.shift_idx])
        e[p.event_idx] ~ _weight(edge, w)
    end
    return e
end

# A location-shifted view of a delay distribution: `logpdf(shifted, y) =
# logpdf(delay, y - shift)` and `rand = shift + rand(delay)`, generalising
# [`PrimaryConditional`](@ref) to any (continuous-core) edge delay. Used to
# declare each downstream event time as a `~` of its predecessor plus the edge
# delay, so the latent composer model both scores observed events and samples
# missing ones. Turing-free arithmetic; `shift` carries any sampled/AD type.
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

# ===========================================================================
# predict_events (#350): recover observed records' latent event times
# ===========================================================================

@doc raw"

Recover the observed records' integrated-out latent event times from a
marginal-fit posterior.

Fit a model in its efficient MARGINAL form (the primary event integrated out, no
extra latent dimensions), then call `predict_events(chain, model)` to recover the
internal event times of the records you fit, by running the LATENT form of the
same model over the fitted posterior. This works because the marginal and latent
forms are one family sharing the same parameter names together with the
marginal-equals-latent equivalence (#301), so the marginal-fit posterior drops
straight into the latent form. It delegates to `DynamicPPL.predict`: the latent
`model` is executed conditioned on each parameter draw in `chain`, re-sampling the
event variables the marginal chain does not carry. The observed events are
supplied in `model` (the censored observations are fixed), so this re-samples only
the integrated-out latents — the primary event time and any unobserved
intermediate events — conditioned on the data and the posterior parameters.

For forward-simulating fresh event paths from parameters (no `@model`, no
conditioning on data), use the Turing-free raw-distribution method
[`predict_events`](@ref)`(d, ...)` instead.

`DynamicPPL.predict` is provided by DynamicPPL's MCMCChains extension, available
whenever `chain` is an `MCMCChains.Chains`. Calling `DynamicPPL.predict` rather
than `Turing.predict` keeps the extension Turing-free (DynamicPPL weak-dep only).

# Arguments
- `chain`: An `MCMCChains.Chains` from fitting the MARGINAL model.
- `model`: The LATENT form of the same model, carrying the observed event times
  (built with a [`latent`](@ref)-wrapped node via [`primary_censored_model`](@ref),
  with the same parameter names as the marginal model that produced `chain`).

# Keyword Arguments
- `rng`: Random number generator for the predictive sampling.
- `include_all`: Passed through to `DynamicPPL.predict`. `false` (the default)
  returns only the recovered event variables; `true` also keeps the parameters
  from `chain`.

# See also
- [`predict_events`](@ref): the Turing-free `(d, ...)` forward-simulation
  method.
- [`latent`](@ref), [`composed_distribution_model`](@ref).
"
function predict_events(
        chain, model::DynamicPPL.Model;
        rng = default_rng(), include_all = false)
    return DynamicPPL.predict(rng, model, chain; include_all = include_all)
end

# ===========================================================================
# composed_parameters_model: priors -> sampled, reconstructed composed dist
# ===========================================================================
#
# `composed_parameters_model(template, priors)` is a submodel that SAMPLES a
# composed distribution's free parameters from user priors and RETURNS the
# reconstructed distribution, ready for the matching record submodel to score
# (#353). It completes the prior workflow: `compose` -> `params_table` (the
# inventory, #351) -> the user keys priors against it -> this helper materialises
# the sampling submodel.
#
# `priors` is a nested `NamedTuple` mirroring `params(template)`: a leaf is keyed
# by its parameter names (`_leaf_param_names`), a `Sequential`/`Parallel` by its
# edge/event names, a `Competing` by its outcome names plus an optional
# `branch_probs` entry. The traversal REUSES the composers' `component_names` and
# the leaf `params`/`_leaf_param_names` introspection (#351) for both structure
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
_base_ctor(leaf) = Base.typename(typeof(leaf)).wrapper

# Reconstruct a leaf of the same family from sampled parameters. Argument checks
# are skipped (`check_args = false`) so a sampler probing an out-of-support point
# yields a `-Inf` log-density rather than throwing mid-gradient; families whose
# constructor lacks the keyword fall back to the plain (checked) constructor.
function _reconstruct_leaf(leaf, vals::Tuple)
    ctor = _base_ctor(leaf)
    return _construct_unchecked(ctor, vals)
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
# add the edge path, so the chain name is `<edge>.<p>`.
@model function _leaf_params_model(leaf, priors::NamedTuple)
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

# A `Sequential` / `Parallel`: sample each named child through a prefixed child
# submodel, then rebuild the SAME composer type with the SAME names.
@model function _composer_params_model(
        d::Union{Sequential, Parallel}, priors::NamedTuple)
    names = component_names(d)
    _check_prior_keys(priors, names, "$(nameof(typeof(d)))")
    parts = Vector{Any}(undef, length(names))
    for i in 1:length(names)
        name = names[i]
        child = d.components[i]
        sub = DynamicPPL.prefix(
            _params_submodel(child, priors[name]), Val(name))
        parts[i] ~ to_submodel(sub, false)
    end
    return _rebuild(d, Tuple(parts))
end

# A `Competing`: sample each outcome delay through a prefixed child submodel; the
# branch probabilities are kept fixed from the template unless a `branch_probs`
# entry of priors is supplied (then each is sampled, prefixed under
# `branch_probs`). Rebuild the `Competing` with the SAME outcome names.
@model function _competing_params_model(c::Competing, priors::NamedTuple)
    expected = (c.names..., :branch_probs)
    have = keys(priors)
    # `branch_probs` is optional; every other expected key is required.
    required = c.names
    _check_competing_keys(priors, required, expected)
    delays = Vector{Any}(undef, length(c.names))
    for i in 1:length(c.names)
        name = c.names[i]
        sub = DynamicPPL.prefix(
            _params_submodel(c.delays[i], priors[name]), Val(name))
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

# `branch_probs` is optional, the outcome names are required: validate the
# required outcome priors are present and no key outside `expected` appears.
function _check_competing_keys(priors::NamedTuple, required::Tuple, expected::Tuple)
    have = keys(priors)
    missing_keys = filter(k -> !(k in have), required)
    extra_keys = filter(k -> !(k in expected), have)
    isempty(missing_keys) || throw(ArgumentError(
        "Competing is missing priors for $(collect(missing_keys)); " *
        "expected outcomes $(collect(required))"))
    isempty(extra_keys) || throw(ArgumentError(
        "Competing has unexpected prior keys $(collect(extra_keys)); " *
        "expected $(collect(expected))"))
    return nothing
end

# Dispatch a node to its parameter submodel: a composer to its composer model, a
# `Competing` to its competing model, any other (leaf) distribution to the leaf
# model. Mirrors the `params`/`params_table` traversal dispatch (#351).
_params_submodel(d::Union{Sequential, Parallel}, priors) = _composer_params_model(d, priors)
_params_submodel(c::Competing, priors) = _competing_params_model(c, priors)
_params_submodel(leaf, priors) = _leaf_params_model(leaf, priors)

# Rebuild a composer of the same type with new components, preserving its names.
_rebuild(d::Sequential, components::Tuple) = Sequential(components, d.names)
_rebuild(d::Parallel, components::Tuple) = Parallel(components, d.names)

# The public entry: dispatch to the recursive submodel builder. A composed
# `template` rebuilds its structure; a bare leaf template rebuilds the leaf.
composed_parameters_model(template, priors) = _params_submodel(template, priors)

end
