module CensoredDistributionsDynamicPPLExt

# DynamicPPL methods for the submodel constructors declared (and documented) in
# `src/turing_models.jl`. Loaded only when DynamicPPL is available, keeping the
# core Turing-free.

using CensoredDistributions: CensoredDistributions, PrimaryCensored, Latent,
                             IntervalCensored, PrimaryConditional, Sequential,
                             Parallel, Competing, as_mixture, get_primary_event,
                             get_dist_recursive, convolve_distributions
import CensoredDistributions: primary_censored_model, interval_censored_model,
                              double_interval_censored_model,
                              composed_distribution_model, predict_events
using DynamicPPL: DynamicPPL, @model
using Distributions: Distributions, UnivariateDistribution, logpdf
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
# The remaining field ORDER is the event order the composer spans.
function _event_vector(row::NamedTuple)
    ks = filter(k -> !(k in _RESERVED_ROW_FIELDS), keys(row))
    out = Vector{Union{Missing, Float64}}(undef, length(ks))
    for (i, k) in enumerate(ks)
        v = row[k]
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
    events = _event_vector(row)
    w = _row_weight(row, weight)
    DynamicPPL.@addlogprob! _marginal_logprob(d, events, w)
    return events
end

# A `Competing` node is univariate (the marginal time-to-resolution), so it
# scores a single observed time through its `MixtureModel` lowering. The row
# carries one event value (plus any reserved weight); the time conditions on the
# mixture marginal. No latent origin is declared: the competing outcome and its
# delay are marginalised inside the mixture (#329).
@model function composed_distribution_model(
        d::Competing, row::NamedTuple; weight = nothing)
    events = _event_vector(row)
    length(events) == 1 || throw(ArgumentError(
        "composed_distribution_model(::Competing, row) takes one event " *
        "value; got $(length(events))"))
    w = _row_weight(row, weight)
    DynamicPPL.@addlogprob! _marginal_logprob(as_mixture(d), events[1], w)
    return events[1]
end

# Marginal log-density of a constant observation `x` under `d`, scaled by an
# optional weight `w`. Shared by the marginal composer and `Competing` models.
_marginal_logprob(d, x, ::Nothing) = logpdf(d, x)
_marginal_logprob(d, x, w) = w * logpdf(d, x)

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

# A `latent`-wrapped `Sequential` chain spans events `E_0, ..., E_k` (one per row
# field). The origin `E_0` is the latent primary (`e[1] ~ origin`); each
# subsequent event time is its predecessor plus the edge delay, declared as a
# shifted `~` (`e[i+1] ~ _ShiftedDelay(edge_i, e[i])`), so an observed event
# scores its gap through the edge's declared censoring (#329) and a missing event
# samples the next event time. Indexed VarNames `e[i]` give each event a distinct
# name in the VarInfo. The likelihood is scaled by the row weight via the `~`.
@model function composed_distribution_model(
        d::Latent{<:Sequential}, row::NamedTuple; weight = nothing)
    chain = d.dist
    obs = _event_vector(row)
    w = _row_weight(row, weight)

    origin = CensoredDistributions._origin_primary_event(chain.components[1])
    origin === nothing && throw(ArgumentError(
        "latent Sequential model needs a censored origin (its first step " *
        "must carry a primary event)"))

    e = Vector{Union{Missing, Float64}}(obs)
    # Origin E_0: the latent primary prior, declared but NOT weighted (the weight
    # scales the LIKELIHOOD, the observed conditionals, not the prior). Indexed
    # `~` so a missing origin samples it.
    e[1] ~ origin
    for i in 1:length(chain.components)
        edge = _ShiftedDelay(get_dist_recursive(chain.components[i]), e[i])
        e[i + 1] ~ _weight(edge, w)
    end
    return e
end

# A `latent`-wrapped `Parallel` shares one latent origin across its branches. The
# row is `(origin, branch_1, ..., branch_n)`. The shared origin `e[1] ~ shared`
# is declared once; each branch event is the origin plus the branch delay,
# declared as a shifted `~`, so an observed branch scores `logpdf(core_i,
# y_i - o)` and a missing branch samples its observation. The shared origin
# couples the branches (one common latent). Likelihood scaled by the weight.
@model function composed_distribution_model(
        d::Latent{<:Parallel}, row::NamedTuple; weight = nothing)
    tree = d.dist
    obs = _event_vector(row)
    w = _row_weight(row, weight)

    shared = CensoredDistributions._shared_primary_event(tree.components)
    shared === nothing && throw(ArgumentError(
        "latent Parallel model needs censored branches sharing one primary " *
        "event"))

    e = Vector{Union{Missing, Float64}}(obs)
    # Shared origin prior, declared but NOT weighted (weight scales the observed
    # conditionals, not the prior).
    e[1] ~ shared
    for i in 1:length(tree.components)
        branch = _ShiftedDelay(get_dist_recursive(tree.components[i]), e[1])
        e[i + 1] ~ _weight(branch, w)
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

end
