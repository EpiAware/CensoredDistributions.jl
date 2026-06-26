# ============================================================================
# Censored composers: flat (one-level) Sequential / Parallel scoring
# ============================================================================
#
# Split out of `censored_specialisations.jl`. Per-record missingness dispatch
# over a flat censored `Sequential` chain / `Parallel` shared-origin
# marginalisation, the per-record observation-horizon right-truncation (hanta),
# and the `event_logpdf` / parameter-eltype helpers.

# ---------------------------------------------------------------------------
# Sequential: per-record missingness dispatch over a censored chain
# ---------------------------------------------------------------------------

# A `Sequential` chain of `k` steps spans `k + 1` events `E_0, ..., E_k`. Scored
# against an EVENT vector (one entry per event, `missing` admitted) the chain
# marginalises unobserved intermediates and conditions on observed ones. The
# element-type dispatch (`>: Missing`) keeps the all-concrete one-value-per-step
# generic path untouched: a `Vector{Float64}` of step gaps still hits the
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
    # The flat event vector carries one entry per EVENT slot plus the root
    # origin. A nested composer contributes its whole subtree; a `Resolve`
    # contributes one slot per OUTCOME, so the count is
    # `_event_nleaves(d.components) + 1`, not `length(d) + 1` (the value layout).
    n = _event_nleaves(d.components) + 1
    length(events) == n || throw(DimensionMismatch(
        "a Sequential event vector needs $n entries " *
        "(one per event), got $(length(events))"))
    # Dispatch on the nested/flat trait so the flat and nested scoring are
    # separate compiled methods (see `_nested_trait`): a chain with a nested
    # composer step recurses through the tree, a flat chain keeps the one-level
    # segment-grouped scoring, and neither method body carries the other's code.
    return _seq_event_logpdf(_nested_trait(d.components), d, events)
end

function _seq_event_logpdf(::_Nested, d::Sequential, events)
    return _nested_tree_logpdf(d, events,
        _origin_primary_event(_first_origin_node(d)),
        _tree_acc_type(d, events))
end

function _seq_event_logpdf(::_Flat, d::Sequential, events)
    return _seq_event_logpdf_h(d, events, nothing)
end

# Flat `Sequential` scoring with an OPTIONAL per-record observation horizon
# `horizon` (real-time right-truncation). The WHOLE composed distribution is
# truncated per record at the horizon (the same combine-then-censor semantics as
# wrapping a chain in `double_interval_censored`): the record is included only if
# its LAST OBSERVED event occurred by the horizon `D`. Resolved semantics
# (maintainer decision): whole-compose TOTAL truncation.
#
#   numerator   = the untruncated factorised per-segment density (unchanged): an
#                 observed intermediate conditions on its own edge at the observed
#                 gap; an unobserved-intermediate run convolves its cores.
#   denominator = a SINGLE term `-logcdf(C, window)` where `C` is the convolution
#                 of all components from the origin to the LAST OBSERVED event
#                 (the origin primary reapplied) and `window = D - origin`.
#
# This is the standard real-time right-truncation (a record is in the sample iff
# its last observed event has occurred by `D`) and is well-defined for ALL
# observed patterns, so it covers BOTH the endpoint-observed hanta index/sourced
# shape (a single collapsed total) and the observed-intermediate case (the
# factorised numerator over the single conv-to-last-observed denominator). With a
# single observed segment the denominator's `C` IS that segment, so the result
# reduces to `truncated(seg; upper = window)`. `horizon === nothing` leaves the
# scoring untruncated.
function _seq_event_logpdf_h(d::Sequential, events, horizon)
    horizon === nothing && return _seq_event_logpdf_untrunc(d, events)
    obs_idx, obs_val = _observed_indices_values(events)
    length(obs_idx) >= 2 || throw(ArgumentError(
        "a Sequential event vector needs at least two observed events"))
    primary = _origin_primary_event(d.components[1])
    # Numerator: the untruncated factorised per-segment density. The origin
    # primary is reapplied only to the segment that actually starts at the latent
    # origin E_0 (`obs_idx[j] == 1`), matching `_sequential_segment`'s `a == 1`
    # contract and the vectorised `_build_seq_bundle` run.
    total = zero(promote_type(eltype(obs_val), float(eltype(d))))
    for j in 1:(length(obs_idx) - 1)
        seg = _sequential_segment(
            d.components, obs_idx[j], obs_idx[j + 1],
            obs_idx[j] == 1 ? primary : nothing)
        gap = obs_val[j + 1] - obs_val[j]
        total += logpdf(seg, gap)
    end
    # Denominator: a single conv-to-last-observed right-truncation term. `C` is
    # the convolution of every component from the origin to the LAST observed
    # event, truncated at the remaining window from the observed origin. With one
    # observed segment `C` is that segment, so this matches the endpoint-observed
    # term `-logcdf(seg, window)`. The origin primary is reapplied only when the
    # first observed event IS the latent origin E_0 (`obs_idx[1] == 1`), matching
    # `_sequential_segment`'s `a == 1` contract and the vectorised
    # `_build_seq_bundle` run; a denominator anchored at an observed intermediate
    # (origin unobserved) is an exact-time anchor, not the latent primary.
    last_seg = _sequential_segment(
        d.components, obs_idx[1], obs_idx[end],
        obs_idx[1] == 1 ? primary : nothing)
    window = _horizon_time(horizon) - obs_val[1]
    # A non-positive window (the horizon already passed the observed origin) is an
    # empty-support truncation: the record cannot have been observed, so the whole
    # contribution is `-Inf`, matching the `_truncate_window` empty-support
    # guard rather than the `+Inf` a bare `total - logcdf(., minimum)` would give.
    window <= minimum(last_seg) &&
        return convert(typeof(total), -Inf)
    # The denominator is the δ-aware right-truncation log-normaliser: a plain
    # horizon gives the upper-only `-logcdf(C, window)`, a δ-bounded horizon the
    # finite `-log(cdf(C, window) - cdf(C, window - δ))` term.
    return total - _truncation_lognorm(last_seg, window, horizon)
end

# The untruncated flat-chain scoring (the original per-segment factorisation).
function _seq_event_logpdf_untrunc(d::Sequential, events)
    # Pre-pass on the constant event vector: collect the observed event indices
    # and their concrete values. The `Union{Missing}` entries are read only
    # here, so the differentiated arithmetic below touches only concrete gaps.
    obs_idx, obs_val = _observed_indices_values(events)
    length(obs_idx) >= 2 || throw(ArgumentError(
        "a Sequential event vector needs at least two observed events"))

    primary = _origin_primary_event(d.components[1])
    total = zero(promote_type(eltype(obs_val), float(eltype(d))))
    for j in 1:(length(obs_idx) - 1)
        # Reapply the origin primary only to the segment starting at the latent
        # origin E_0 (`obs_idx[j] == 1`); a leading run anchored at an observed
        # event is an exact-time anchor, matching the vectorised build.
        seg = _sequential_segment(
            d.components, obs_idx[j], obs_idx[j + 1],
            obs_idx[j] == 1 ? primary : nothing)
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
# `b`. The steps linking them are `components[a:(b - 1)]`.
#
# A SINGLE observed-bounded edge conditions on its OWN declared censoring: the
# edge's gap is scored through the edge distribution AS DECLARED ("condition
# on its (censored) value"). A `primary_censored` / `double_interval_censored` /
# `interval_censored` edge therefore keeps that censoring (real linelist data is
# day-resolution, so a day-observed edge delay is interval-censored, not an exact
# continuous time); a plain continuous edge is conditioned on the continuous
# value, which is correct for that edge.
#
# A RUN of two or more steps (one or more UNOBSERVED intermediate events between
# the two observed endpoints) is marginalised: the spanned steps' continuous
# cores are convolved and the intermediates' own censoring is dropped (the latent
# intermediate is a continuous time, not a windowed observation). When a primary
# event is supplied (the origin segment, whose origin E_0 is the latent primary)
# it is reapplied to the convolved core via `primary_censored`.
function _sequential_segment(components, a, b, primary)
    run = components[a:(b - 1)]
    # Single observed-bounded edge: respect the edge's declared censoring.
    length(run) == 1 && return run[1]
    # Unobserved-intermediate run: convolve the continuous cores; the origin
    # segment reapplies the (latent) primary event. Map over the component
    # tuple (not a collected `Vector{UnivariateDistribution}`) so the
    # `Convolved` keeps a concrete component tuple; an abstract-eltype vector
    # defeats Enzyme's activity analysis on the reverse pass (the same defect
    # fixed for `observed_distribution(::Sequential)`).
    core = convolved(map(_marginal_core, run))
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
    # One entry per EVENT slot plus the shared origin; a tree branch contributes
    # its whole subtree's events and a `Resolve` one slot per OUTCOME,
    # so the length is `_event_nleaves(d.components) + 1`.
    n = _event_nleaves(d.components) + 1
    length(events) == n || throw(DimensionMismatch(
        "a Parallel event vector needs $n entries " *
        "(shared origin then one per branch), got $(length(events))"))
    # Dispatch on the nested/flat trait so the nested tree recursion and the flat
    # shared-origin (coupled-integral) scoring are separate compiled methods (see
    # `_nested_trait`); the flat coupled `gl_integrate` path stays out of the
    # nested method that the compiled AD backends differentiate.
    return _par_event_logpdf(_nested_trait(d.components), d, events)
end

function _par_event_logpdf(::_Nested, d::Parallel, events)
    return _nested_tree_logpdf(d, events,
        _shared_primary_event(d.components), _tree_acc_type(d, events))
end

function _par_event_logpdf(::_Flat, d::Parallel, events)
    primary = _shared_primary_event(d.components)
    # Plain branches: no shared primary to integrate, so the origin must be
    # observed; condition each branch on its gap and drop missing branches.
    primary === nothing && return _par_plain_logpdf(d, events)

    cores = map(_marginal_core, d.components)
    # Promote over the PARAMETER types of the cores and primary (which carry any
    # AD Dual from the leaf params; a distribution's `eltype` is its variate type,
    # not its parameter type, so it would drop the Dual) plus the event vector,
    # so the marginal/condition arithmetic stays on the differentiated type.
    T2 = promote_type(_event_eltype(events), _param_eltype(primary),
        map(_param_eltype, cores)...)
    origin = events[1]
    if origin === missing
        return _parallel_marginal_logpdf(primary, cores, events, T2)
    end
    return _parallel_conditional_logpdf(primary, cores, events, T2)
end

# Plain-branch Parallel sharing an exactly-observed continuous origin: condition
# each present branch on its gap `y_i - o` (no primary prior, no integral); drop
# missing branches. A missing origin cannot be marginalised without a
# distribution for it, so it is rejected (promote it to a primary or go latent).
function _par_plain_logpdf(d::Parallel, events)
    origin = events[1]
    origin === missing && throw(ArgumentError(
        "a plain-branch Parallel needs an observed shared origin to condition " *
        "on; a missing continuous origin cannot be marginalised without a " *
        "distribution for it (declare a primary event or pass a latent origin)"))
    T = promote_type(_event_eltype(events),
        map(_param_eltype, d.components)...)
    o = events[1]
    o isa Real || return convert(T, -Inf)
    isnan(o) && return convert(T, NaN)
    lp = zero(T)
    @inbounds for i in 2:length(events)
        y = events[i]
        y === missing && continue
        isnan(y) && return convert(T, NaN)
        u = y - o
        branch = d.components[i - 1]
        insupport(branch, u) || return convert(T, -Inf)
        lp += convert(T, logpdf(branch, u))
    end
    return lp
end

@doc "

Density of a [`Parallel`](@ref) of censored branches scored against the
shared-origin event vector.

See also: [`logpdf`](@ref)
"
function pdf(d::Parallel, events::AbstractVector{T}) where {T >: Missing}
    return exp(logpdf(d, events))
end

# ---------------------------------------------------------------------------
# Per-record observation-horizon right-truncation (hanta)
# ---------------------------------------------------------------------------
#
# `event_logpdf(d, events; horizon)` is the horizon-aware event-vector log
# density. With `horizon === nothing` it is exactly the censored composer
# `logpdf(d, events)` (back-compat). With a per-record `horizon` the WHOLE
# composed distribution is right-truncated at that observation time for the
# record, the same combine-then-censor direction as wrapping the compose in
# `double_interval_censored` (`wrap.jl`) but with the upper bound supplied PER
# RECORD rather than baked in. For a `Sequential` the record's observed total
# (origin -> terminal) is truncated at `horizon - origin` (the endpoint-observed
# hanta index/sourced shape); for a `Parallel` each branch endpoint is truncated
# at `horizon - origin` off the shared origin. `_truncate_window` is the
# implementation primitive (upper-only, AD-safe, with the non-positive-window
# empty-support guard); it is NOT a user-facing per-segment interface here.

@doc raw"

Horizon-aware event-vector log density of a censored composer.

`event_logpdf(d, events; horizon)` scores `events` exactly as
`logpdf(d, events)` when `horizon === nothing`. With a per-record `horizon` the
WHOLE composed distribution is right-truncated at that observation time for the
record, the same combine-then-censor direction as wrapping the compose in
[`double_interval_censored`](@ref) but with the upper bound supplied per record.

For a [`Sequential`](@ref) the truncation is whole-compose TOTAL truncation:
the factorised per-segment numerator is divided by a single
``F(\text{window})`` denominator, where the denominator delay is the convolution
of every component from the origin to the LAST OBSERVED event and
``\text{window} = \text{horizon} - \text{origin}``. A record is then included
only if its last observed event occurred by the horizon. This is well-defined for
all observed patterns, so observed intermediates are scored (their numerator
factorises, the single conv-to-last-observed denominator truncates), not rejected.
For a [`Parallel`](@ref) each branch endpoint is truncated at
``\text{horizon} - \text{origin}`` off the shared origin. The truncation
contributes the ``-\log F(\text{window})`` correction, upper-only and AD-safe.

For a NESTED tree (a chain/set whose step or branch is itself a composer) the
per-record `horizon` threads down to the nested scorer: each nested
[`Resolve`](@ref)/`Compete` node is right-truncated at the remaining
window from its anchor (`horizon - anchor`), the same right-truncation the
top-level node applies, while plain leaf/chain edges ignore the horizon. The
nested marginal path is then density-identical to the latent-conditioned model.

# Arguments
- `d`: a censored [`Sequential`](@ref) or [`Parallel`](@ref) composer.
- `events`: the flat event vector `[E_0, ..., E_k]` (value or `missing` each).

# Keyword Arguments
- `horizon`: the per-record observation horizon, or `nothing` for no truncation.

# Examples
```@example
using CensoredDistributions, Distributions

seq = Sequential(
    primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
    primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
# Endpoint-observed record (intermediate unobserved): the total is truncated.
ev = Vector{Union{Missing, Float64}}([0.0, missing, 5.0])
CensoredDistributions.event_logpdf(seq, ev; horizon = 8.0)
```

# See also
- [`double_interval_censored`](@ref): the whole-compose wrap with a baked-in
  bound
- `truncated`: the truncation verb (`truncated(node; upper)`)
"
function event_logpdf(
        d::Sequential, events::AbstractVector{T}; horizon = nothing
) where {T >: Missing}
    horizon === nothing && return logpdf(d, events)
    # Dispatch on the nested/flat trait. A FLAT chain takes the whole-compose
    # TOTAL truncation (the collapsed conv-to-last-observed denominator). A NESTED
    # tree threads the per-record horizon down to the now horizon-capable nested
    # scorer, which right-truncates each nested `Resolve`/`Compete`
    # node at the remaining window from its anchor, exactly as the top-level
    # `Resolve` truncation does; plain leaf/chain edges ignore the horizon.
    return _seq_event_logpdf_horizon(_nested_trait(d.components), d, events,
        horizon)
end

function _seq_event_logpdf_horizon(::_Flat, d::Sequential, events, horizon)
    return _seq_event_logpdf_h(d, events, horizon)
end

function _seq_event_logpdf_horizon(::_Nested, d::Sequential, events, horizon)
    return _nested_tree_logpdf(d, events,
        _origin_primary_event(_first_origin_node(d)),
        _tree_acc_type(d, events), horizon)
end

function event_logpdf(
        d::Parallel, events::AbstractVector{T}; horizon = nothing
) where {T >: Missing}
    horizon === nothing && return logpdf(d, events)
    _nested_trait(d.components) isa _Flat || throw(ArgumentError(
        "per-record horizon truncation is defined for a flat Parallel set; " *
        "a nested tree has no single observed total to truncate"))
    return _par_event_logpdf_h(d, events, horizon)
end

# A univariate leaf record's horizon truncation: the leaf is observed from the
# (zero) origin, so the window is the horizon itself.
function event_logpdf(d::UnivariateDistribution, x::Real; horizon = nothing)
    horizon === nothing && return logpdf(d, x)
    # The leaf is observed from the (zero) origin, so its window is the horizon
    # itself; `_truncate_horizon` honours a δ-bounded horizon and is the upper-only
    # `truncated(d; upper = window)` for a plain horizon.
    return logpdf(_truncate_horizon(d, _horizon_time(horizon), horizon), x)
end

# Flat `Parallel` with a horizon: every branch endpoint shares the observed
# origin, so each is right-truncated at `horizon - origin` (each branch is its own
# endpoint, no intermediate, so the whole-compose meaning is unambiguous). A
# missing (marginalised) origin has no observed anchor, so truncation would
# require integrating the denominator over the origin; that is out of scope and
# rejected with a clear error.
function _par_event_logpdf_h(d::Parallel, events, horizon)
    origin = events[1]
    origin === missing && throw(ArgumentError(
        "per-record horizon truncation needs an observed shared origin; a " *
        "missing (marginalised) origin has no anchor to truncate from"))
    primary = _shared_primary_event(d.components)
    primary === nothing && throw(ArgumentError(
        "Parallel shared-origin scoring needs censored branches with a " *
        "primary event; got plain branches"))
    o = Float64(origin)
    cores = map(_marginal_core, d.components)
    T2 = promote_type(_event_eltype(events), _param_eltype(primary),
        map(_param_eltype, cores)...)
    insupport(primary, o) || return convert(T2, -Inf)
    lp = convert(T2, logpdf(primary, o))
    window = _horizon_time(horizon) - o
    @inbounds for i in eachindex(cores)
        y = events[i + 1]
        y === missing && continue
        seg = _truncate_horizon(cores[i], window, horizon)
        u = Float64(y) - o
        lp += convert(T2, logpdf(seg, u))
    end
    return lp
end

# Element type of an event vector, treating the `missing`-only / mixed slots as
# Float64 so the arithmetic stays concrete.
function _event_eltype(events)
    (eltype(events) === Missing) ? Float64 :
    promote_type(Float64, nonmissingtype(eltype(events)))
end

# Numeric (parameter) type of a distribution, recovered from its flattened
# `params`. A distribution's `eltype` is its variate type (`Float64`) and would
# drop an AD `Dual` carried by the parameters, so the arithmetic type is taken
# from the parameters instead. `Uniform`'s bounds are plain `Float64`; a censored
# delay's shape/scale carry the differentiated `Dual`.
function _param_eltype(d)
    return mapreduce(typeof, promote_type, _flatten_params(params(d));
        init = Float64)
end

# Flatten the (possibly nested) `params` of a distribution to its scalar leaves,
# so `_param_eltype` promotes over every parameter. The composed `params` is now
# a name-keyed `NamedTuple`, so a NamedTuple is flattened over its values
# just like a tuple; a bare scalar is its own leaf.
_flatten_params(t::Tuple) = mapreduce(_flatten_params, (a, b) -> (a..., b...), t;
    init = ())
function _flatten_params(nt::NamedTuple)
    return mapreduce(_flatten_params, (a, b) -> (a..., b...), values(nt);
        init = ())
end
_flatten_params(x) = (x,)

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
