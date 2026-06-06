# ============================================================================
# Censored specialisations of the generic composers (#329, PR3b)
# ============================================================================
#
# The generic `Sequential` / `Parallel` / `Competing` composers (PR3a) score a
# value vector with one entry per step/branch. When their internal nodes are our
# censored distributions (`primary_censored` / `interval_censored` /
# `double_interval_censored`), per-record marginalisation is AUTOMATIC and
# data-driven, selected by MULTIPLE DISPATCH on the value vector's element type
# (a `Missing`-admitting event vector) and on the node types (#329). There is no
# runtime predicate, no `mode` keyword, and no new node-type hierarchy.
#
# Per #329, evaluating a censored chain against an EVENT vector
# `[E_0, E_1, ..., E_k]` (one entry per event, each a value or `missing`):
#   - the origin's primary censoring is ALWAYS applied (first segment);
#   - an OBSERVED intermediate -> CONDITION on its (censored) value: the
#     adjacent delay is an independent factor at the observed gap;
#   - an UNOBSERVED intermediate -> MARGINALISE by CONVOLVING the adjacent
#     delays and DROPPING that intermediate's censoring (the latent is a
#     continuous time, not a windowed observation), via `convolve_distributions`
#     over the continuous cores recovered with `get_dist_recursive`.
# For `Parallel` with censored branches the shared origin couples the branches:
# a missing origin entry is marginalised by one 1-D origin integral, a present
# origin conditions. `Competing` already lowers to a `MixtureModel`
# (`as_mixture`, PR3a), so it needs no event-vector specialisation here.

# ---------------------------------------------------------------------------
# Origin primary-event recovery
# ---------------------------------------------------------------------------

# The primary event distribution censoring a node's origin, or `nothing` when
# the node carries no primary censoring. Recurses through the censoring wrappers
# (`Truncated`, `IntervalCensored`, `Weighted`) so a `double_interval_censored`
# origin still surfaces its primary event. Dispatch on the node type is the
# whole selection: a plain delay returns `nothing` and so keeps the generic
# (uncensored) treatment.
_origin_primary_event(d::PrimaryCensored) = d.primary_event
_origin_primary_event(d::Truncated) = _origin_primary_event(d.untruncated)
_origin_primary_event(d::IntervalCensored) = _origin_primary_event(d.dist)
_origin_primary_event(d::Weighted) = _origin_primary_event(d.dist)
_origin_primary_event(::UnivariateDistribution) = nothing

# The continuous delay core of a (possibly censored) node, for marginalisation:
# strip every censoring layer so a marginalised run convolves only continuous
# delays, never a discrete/windowed object. A `Convolved` node is already a
# continuous sum and is left intact for the fold.
_marginal_core(d::UnivariateDistribution) = get_dist_recursive(d)
_marginal_core(d::Convolved) = d

# ---------------------------------------------------------------------------
# Sequential: per-record missingness dispatch over a censored chain
# ---------------------------------------------------------------------------

# A `Sequential` chain of `k` steps spans `k + 1` events `E_0, ..., E_k`. Scored
# against an EVENT vector (one entry per event, `missing` admitted) the chain
# marginalises unobserved intermediates and conditions on observed ones. The
# element-type dispatch (`>: Missing`) keeps the all-concrete one-value-per-step
# generic path (PR3a) untouched: a `Vector{Float64}` of step gaps still hits the
# generic `logpdf`, while a `Vector{Union{Missing, Float64}}` event vector hits
# this specialisation.

@doc raw"

Log density of a [`Sequential`](@ref) chain scored against an EVENT vector.

`events` has one entry per event ``E_0, \dots, E_k`` (length `length(d) + 1`);
each entry is a value (observed) or `missing` (unobserved). The chain is grouped
into segments by the missingness pattern and the log density is the sum of the
per-segment log-densities at the observed gaps:

- the origin's primary censoring (from the first step) is always applied to the
  first segment;
- an unobserved intermediate is marginalised by convolving the continuous cores
  of the delays it spans and dropping their censoring;
- an observed intermediate conditions: its adjacent delay is an independent
  factor at the observed gap.

Missingness drives only the control flow grouping the segments; the
differentiated arithmetic sees only concrete observed values, so this is safe to
differentiate with `events` held constant.

See also: [`Sequential`](@ref), [`Parallel`](@ref)
"
function logpdf(d::Sequential, events::AbstractVector{T}) where {T >: Missing}
    length(events) == length(d.components) + 1 || throw(DimensionMismatch(
        "a Sequential event vector needs $(length(d.components) + 1) entries " *
        "(one per event), got $(length(events))"))

    # Pre-pass on the constant event vector: collect the observed event indices
    # and their concrete values. The `Union{Missing}` entries are read only
    # here, so the differentiated arithmetic below touches only concrete gaps.
    obs_idx, obs_val = _observed_indices_values(events)
    length(obs_idx) >= 2 || throw(ArgumentError(
        "a Sequential event vector needs at least two observed events"))

    primary = _origin_primary_event(d.components[1])
    total = zero(promote_type(eltype(obs_val), float(eltype(d))))
    for j in 1:(length(obs_idx) - 1)
        seg = _sequential_segment(
            d.components, obs_idx[j], obs_idx[j + 1], j == 1 ? primary : nothing)
        gap = obs_val[j + 1] - obs_val[j]
        total += logpdf(seg, gap)
    end
    return total
end

@doc "

Density of a [`Sequential`](@ref) chain scored against an EVENT vector.

See also: [`logpdf`](@ref)
"
function pdf(d::Sequential, events::AbstractVector{T}) where {T >: Missing}
    return exp(logpdf(d, events))
end

# Walk the (constant) event vector and return the observed event indices and
# their concrete values. Kept separate from the arithmetic so the
# `Union{Missing}` handling is pure control flow.
function _observed_indices_values(events)
    idx = Int[]
    val = Float64[]
    for i in eachindex(events)
        o = events[i]
        o === missing && continue
        push!(idx, i)
        push!(val, Float64(o))
    end
    return idx, val
end

# Segment distribution spanning observed events at 1-based event indices `a` and
# `b`. The steps linking them are `components[a:(b - 1)]`. Both endpoints are
# observed, so the segment conditions on the continuous delay between them: a
# single step contributes its continuous core, a run of two or more (unobserved
# events between) is the convolution of the spanned cores, in both cases
# dropping the steps' own primary censoring (its windowed primary event is no
# longer latent for an observed-bounded edge). When a primary event is supplied
# (the origin segment, whose origin E_0 is the latent primary) it is reapplied
# via `primary_censored`.
function _sequential_segment(components, a, b, primary)
    run = components[a:(b - 1)]
    core = length(run) == 1 ? _marginal_core(run[1]) :
           convolve_distributions(map(_marginal_core, collect(run)))
    primary === nothing && return core
    return primary_censored(core, primary)
end

# ---------------------------------------------------------------------------
# Parallel: shared-origin marginalisation over censored branches
# ---------------------------------------------------------------------------

# When the branches of a `Parallel` are censored they share one latent origin
# (the common primary event), so the branch observations are coupled through
# that origin. Scored against an event vector `[O, Y_1, ..., Y_n]` (origin slot
# first), a missing origin marginalises by one 1-D integral over the shared
# origin and a present origin conditions. A missing branch drops from the joint.
# The branch primaries must agree (one shared origin); they are checked once.

@doc raw"

Log density of a [`Parallel`](@ref) of censored branches scored against the
shared-origin event vector ``[O, Y_1, \dots, Y_n]``.

The branches share one latent origin ``O`` (their common primary event), so the
observations are coupled: ``Y_i = O + D_i`` with independent branch delays
``D_i``. The first entry is the shared origin and the rest are the branch
observations (each a value or `missing`):

- a missing origin is marginalised by a single one-dimensional integral over the
  shared origin against the present-branch delay densities,
  ``\int f_O(o) \prod_{i \text{ present}} f_{D_i}(y_i - o)\, do``;
- a present origin conditions: ``\log f_O(o) + \sum_{i} \log f_{D_i}(y_i - o)``;
- a missing branch drops from the joint.

The branch primaries must agree (one shared origin). Missingness drives only
control flow, so this differentiates with the event vector held constant.

See also: [`Parallel`](@ref), [`Sequential`](@ref)
"
function logpdf(d::Parallel, events::AbstractVector{T}) where {T >: Missing}
    length(events) == length(d.components) + 1 || throw(DimensionMismatch(
        "a Parallel event vector needs $(length(d.components) + 1) entries " *
        "(shared origin then one per branch), got $(length(events))"))

    primary = _shared_primary_event(d.components)
    primary === nothing && throw(ArgumentError(
        "Parallel shared-origin scoring needs censored branches with a " *
        "primary event; got plain branches"))

    cores = map(_marginal_core, d.components)
    T2 = promote_type(_event_eltype(events), float(eltype(d)))
    origin = events[1]
    if origin === missing
        return _parallel_marginal_logpdf(primary, cores, events, T2)
    end
    return _parallel_conditional_logpdf(primary, cores, events, T2)
end

@doc "

Density of a [`Parallel`](@ref) of censored branches scored against the
shared-origin event vector.

See also: [`logpdf`](@ref)
"
function pdf(d::Parallel, events::AbstractVector{T}) where {T >: Missing}
    return exp(logpdf(d, events))
end

# Element type of an event vector, treating the `missing`-only / mixed slots as
# Float64 so the arithmetic stays concrete.
function _event_eltype(events)
    (eltype(events) === Missing) ? Float64 :
    promote_type(Float64, nonmissingtype(eltype(events)))
end

# The single shared primary event of a `Parallel`'s branches, or `nothing` when
# no branch carries primary censoring. Errors if the branches disagree on the
# primary event (a shared origin must be unique).
function _shared_primary_event(components::Tuple)
    primary = nothing
    @inbounds for c in components
        p = _origin_primary_event(c)
        p === nothing && continue
        if primary === nothing
            primary = p
        else
            primary == p || throw(ArgumentError(
                "Parallel shared-origin branches must share one primary " *
                "event; got $(primary) and $(p)"))
        end
    end
    return primary
end

# Present (non-missing) branch indices and their concrete observed values from
# the event vector `[O, Y_1, ...]`. Returns `nothing` values when a present
# observation is NaN so the caller propagates it.
function _parallel_present(events, ::Type{T}) where {T}
    present = Int[]
    vals = T[]
    @inbounds for i in 2:length(events)
        y = events[i]
        y === missing && continue
        isnan(y) && return present, nothing
        push!(present, i - 1)
        push!(vals, convert(T, y))
    end
    return present, vals
end

# Concrete origin: log f_O(o) + Σ_present log f_{D_i}(y_i - o). Missing branches
# drop out.
function _parallel_conditional_logpdf(primary, cores, events, ::Type{T}) where {T}
    o = events[1]
    o isa Real || return convert(T, -Inf)
    isnan(o) && return convert(T, NaN)
    insupport(primary, o) || return convert(T, -Inf)

    lp = convert(T, logpdf(primary, o))
    @inbounds for i in eachindex(cores)
        y = events[i + 1]
        y === missing && continue
        isnan(y) && return convert(T, NaN)
        u = y - o
        insupport(cores[i], u) || return convert(T, -Inf)
        lp += convert(T, logpdf(cores[i], u))
    end
    return lp
end

# Missing origin: marginalise the shared origin over the present branches by a
# single 1-D Gauss-Legendre integral. Missing branches contribute no factor and
# do not narrow the origin window.
function _parallel_marginal_logpdf(primary, cores, events, ::Type{T}) where {T}
    present, yvals = _parallel_present(events, T)
    yvals === nothing && return convert(T, NaN)
    isempty(present) && return zero(T)

    lo = float(minimum(primary))
    hi = float(maximum(primary))
    @inbounds for k in eachindex(present)
        i = present[k]
        a = float(minimum(cores[i]))
        b = float(maximum(cores[i]))
        y = yvals[k]
        lo = max(lo, y - b)
        hi = min(hi, y - a)
    end
    hi > lo || return convert(T, -Inf)

    val = gl_integrate(lo, hi, _PRIMARY_GL) do o
        p = pdf(primary, o)
        @inbounds for k in eachindex(present)
            p *= pdf(cores[present[k]], yvals[k] - o)
        end
        p
    end
    val <= 0 && return convert(T, -Inf)
    return convert(T, log(val))
end
