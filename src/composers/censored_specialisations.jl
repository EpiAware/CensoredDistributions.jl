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
# A nested multivariate composer is not a primary-censored origin leaf, so it
# surfaces no origin primary event (the censored treatment is resolved within
# the child). `Competing` is univariate and already hits the fallback above.
_origin_primary_event(::Union{Sequential, Parallel}) = nothing

# The continuous delay core of a (possibly censored) node, for marginalisation:
# strip every censoring layer so a marginalised run convolves only continuous
# delays, never a discrete/windowed object. A `Convolved` node is already a
# continuous sum and is left intact for the fold.
_marginal_core(d::UnivariateDistribution) = get_dist_recursive(d)
_marginal_core(d::Convolved) = d

# Whether a component is itself a nested composer (a branch/step that recurses)
# rather than a leaf edge. Dispatch on the type, so the recursion is selected at
# compile time with no runtime type lookup. A `Competing` is univariate (a single
# marginal time-to-resolution leaf), so it is NOT a nested composer here.
_is_nested_composer(::Union{Sequential, Parallel}) = true
_is_nested_composer(::UnivariateDistribution) = false

# Whether any of a composer's components is itself a nested composer. Selects the
# recursive tree walk over the flat one-level scoring. Resolved on the component
# TYPES (the `Tuple` element types), so it is a compile-time constant and the
# branch on it is eliminated, keeping the scoring type-stable.
_any_nested_composer(components::Tuple) = any(_is_nested_composer, components)

# A SINGLETON trait carrying the nested/flat choice in its TYPE, so the nested and
# flat scoring become separate dispatched methods. A plain `if _any_nested_...`
# branch leaves BOTH branches in one method body, and the compiled AD backends
# (Mooncake) build a rule for every reachable branch -- including the flat
# `convolve_distributions` marginalisation path that hard-crashes them
# uncatchably (#319). Splitting by dispatch keeps the flat path out of the nested
# method entirely, so a nested tree differentiates without dragging in the
# crashing convolution code.
#
# The trait is selected by DISPATCH on the component `Tuple` TYPE, returning a
# concrete singleton -- NOT a runtime `cond ? _Nested() : _Flat()` ternary, whose
# result is a `Union{_Nested, _Flat}` value that forces a dynamic dispatch on the
# trait. Mooncake's reverse-mode codegen crashes (uncatchable `signal 4`) on that
# union-typed dispatch, so the trait must resolve to a single concrete type at
# compile time.
struct _Nested end
struct _Flat end
_nested_trait(components::C) where {C <: Tuple} = _nested_trait(C)
@generated function _nested_trait(::Type{C}) where {C <: Tuple}
    has_nested = any(t -> t <: Sequential || t <: Parallel, fieldtypes(C))
    return has_nested ? :(_Nested()) : :(_Flat())
end

# Whether a composer's components carry primary censoring (an origin primary
# event), checked on the component types. `true` selects the censored
# full-event-path `rand`; `false` keeps the generic per-leaf-value `rand` (PR3a).
function _is_censored_composer(components::Tuple)
    any(c -> _origin_primary_event(c) !== nothing, components)
end

# ---------------------------------------------------------------------------
# Recursive nested-composer scoring (#329, #333, #345)
# ---------------------------------------------------------------------------
#
# A composer whose branch/step is ITSELF a `Sequential` / `Parallel` is an
# arbitrary, irregular event tree (e.g. bdbv `onset -> {admit -> {death,
# discharge}, notif}`). The flat one-level scoring above only handles leaf
# steps/branches; the recursion here scores the tree at every depth, each node
# consuming its contiguous slice of the flat event vector.
#
# The flat event vector lays out one entry per LEAF event in depth-first order,
# preceded by the root origin `E_0`. A nested composer's ORIGIN is the parent's
# event at its position (the origin is shared between parent and child, so a
# nested composer does NOT re-introduce its own origin slot); it then consumes
# one slot per leaf event in its subtree (`_nleaves`). `_subtree_logpdf` walks
# the tree returning `(logprob, next_idx)`: the subtree log density and the index
# one past the last event it consumed. Dispatch on the node type drives the
# recursion (no runtime type lookup), and the components are processed by
# head/tail tuple recursion so the compiler specialises each child type, keeping
# the walk type-stable and AD-safe.

# The DATA value type for a tree walk: the event vector's non-missing element
# type. Used only to convert the observed event VALUES (data) and the `zero`
# seed; the per-edge log densities keep their own (possibly AD-tracked) type and
# flow into the UNANNOTATED accumulator, so an AD `Dual`/tracked value from the
# leaf params propagates through the sum without being narrowed.
#
# Extracted by a `@generated` function that strips `Missing` at COMPILE time and
# splices the bare type in as a constant. A plain runtime `nonmissingtype(eltype(
# events))` crashes Mooncake's reverse-mode codegen (uncatchable `signal 4`) when
# applied to the tracked event vector; folding it into a generated constant keeps
# `nonmissingtype` off the runtime (and so differentiated) path entirely. The
# `@generated` body has bound parameters, so it is also Aqua-clean.
_tree_acc_type(d, events) = _data_value_type(eltype(events))
@generated function _data_value_type(::Type{E}) where {E}
    S = nonmissingtype(E)
    bare = (S === Union{} || S === Missing) ? Float64 : S
    return :($bare)
end

# The node whose origin primary event seeds the root of a `Sequential` tree: the
# first step (its origin E_0 is the latent primary). Kept as a tiny helper so the
# dispatch site reads clearly.
_first_origin_node(d::Sequential) = d.components[1]

# --- Recursive tree scoring via per-node runtime loops ------------------------
#
# `_tree_score` walks the tree and sums each edge's contribution, recursing into
# a nested composer step/branch through a dispatched function call. The walk over
# a node's children is a RUNTIME `for` loop over the component tuple (mirroring
# the generic `_composite_logpdf` in `nesting.jl`), NOT a head/tail type
# recursion: the compiled AD backends (Mooncake) differentiate a per-node runtime
# loop plus dispatched recursion, but choke on deep tuple-splatting head/tail
# recursion that carries the tracked leaf distributions. The mutual recursion
# (Sequential -> Parallel -> Sequential) goes through dispatch, the same shape
# the generic composer logpdf already differentiates.
#
# The flat event vector is laid out depth-first: entry 1 is the root origin E_0,
# then one entry per LEAF event in traversal order. A node's ORIGIN is the
# parent's event (a SHARED slot), while its own leaf events occupy a CONTIGUOUS
# slice; the walk carries both `origin_idx` (the origin event) and `event_start`
# (where this node's own leaf events begin) so an irregular tree scores at every
# depth. Each node returns `(logprob, next_event_idx, terminal_idx)`.

# The event-index layout is computed by PURE-INTEGER helpers (`_child_nleaves`
# from `nesting.jl`, plus `_terminal_offset` below) that touch only the tree
# structure, never the leaf params, so the scoring recursion can return a plain
# SCALAR log density. Returning a scalar (not an index-carrying tuple) is what
# keeps the compiled AD backends (Mooncake) happy: a recursion returning a tuple
# that mixes a tracked log density with inactive `Int` indices hits Mooncake's
# mixed-activity codegen, while a scalar-returning recursion over the same
# unrolled-tuple loop differentiates exactly as the generic `_composite_logpdf`
# does (#319 notwithstanding for Enzyme on deeply mixed trees).

# Offset (relative to a node's `event_start`) of the TERMINAL event a following
# chain step hangs off. A leaf edge terminates at its own (single) event; a
# Sequential at its last leaf event; a Parallel at its shared origin, which is the
# PARENT's event (offset -1 relative to the branch's own first event). Pure Int.
_terminal_offset(::UnivariateDistribution) = 0
_terminal_offset(d::Sequential) = _seq_terminal_offset(d.components)
_terminal_offset(::Parallel) = -1
function _seq_terminal_offset(components::Tuple)
    # The last step's terminal, measured from the chain's own first event.
    off = 0
    @inbounds for i in 1:(length(components) - 1)
        off += _child_nleaves(components[i])
    end
    last = components[end]
    return off + _terminal_offset(last)
end

# Top entry: origin at 1, first event at 2; returns the scalar log density. The
# return type is NOT `T` (the event-data type): the per-edge log densities carry
# any AD `Dual`/tracked type from the leaf params, so the result widens past the
# data type and must not be asserted to `T`.
function _nested_tree_logpdf(d::Union{Sequential, Parallel}, events,
        primary, ::Type{T}) where {T}
    return _tree_score(d, events, 1, 2, primary, T)
end

# Sequential: the first step hangs off `origin_idx` with the (latent) primary,
# each later step off the previous step's terminal event. The first step is
# PEELED off so the primary lives only on it (a runtime `i == 1 ? primary :
# nothing` would make the primary a `Union{P, Nothing}` value and defeat
# inference); the rest are walked with `for c in tail` (Julia unrolls a `for`
# over a tuple, so the heterogeneous steps stay type-stable). The running event
# cursor and origin index are PURE-INTEGER bookkeeping computed from
# `_child_nleaves` / `_terminal_offset`, so each `_tree_step` returns a scalar.
function _tree_score(d::Sequential, events, origin_idx::Int, event_start::Int,
        primary, ::Type{T}) where {T}
    comps = d.components
    first_step = first(comps)
    acc = _tree_step(first_step, events, origin_idx, event_start, primary, T)
    o_idx = event_start + _terminal_offset(first_step)
    ev_idx = event_start + _child_nleaves(first_step)
    @inbounds for step in Base.tail(comps)
        acc += _tree_step(step, events, o_idx, ev_idx, nothing, T)
        o_idx = ev_idx + _terminal_offset(step)
        ev_idx += _child_nleaves(step)
    end
    return acc
end

# Parallel: every branch hangs off the SHARED origin `origin_idx`; each branch's
# own events continue from the running cursor. The branch walk is
# `for b in components` (unrolled, type-stable); the cursor is pure-Int
# bookkeeping, so each `_tree_step` returns a scalar.
#
# Within a NESTED tree the per-branch scoring is independent given the shared
# origin: an observed origin conditions each branch on the declared edge, a
# missing origin marginalises each LEAF branch against the latent primary via the
# analytic `primary_censored` (handled in `_tree_step`). The coupled 1-D origin
# QUADRATURE (the flat-`Parallel` shared-origin marginal) is deliberately NOT
# reachable from here: its `gl_integrate` path hard-crashes the compiled AD
# backends (Mooncake/Enzyme) uncatchably even when not taken at runtime (they
# build a rule for every reachable branch), so pulling it into the nested path
# would break AD for the whole tree. A flat (non-nested) `Parallel` with a missing
# shared origin still gets the coupled integral through its own `logpdf` method.
function _tree_score(d::Parallel, events, origin_idx::Int, event_start::Int,
        primary, ::Type{T}) where {T}
    acc = zero(T)
    ev_idx = event_start
    @inbounds for branch in d.components
        acc += _tree_step(branch, events, origin_idx, ev_idx, primary, T)
        ev_idx += _child_nleaves(branch)
    end
    return acc
end

# A nested-composer step/branch recurses on a FRESH sub-event view
# `[origin, its_leaf_events...]` (origin at 1, first event at 2), rather than the
# full event vector with offset indices. Passing a shrinking sub-view per level
# mirrors the generic `_composite_logpdf` recursion (which the compiled AD
# backends differentiate); recursing on the full shared array with offsets
# instead trips Mooncake's reverse-mode codegen on the repeated aliasing. The
# sub-view is built from the constant event vector (index control flow only), so
# the differentiated arithmetic still sees only the leaf params. The nested
# (declared-edge) semantics are preserved -- this is `_tree_score`, not the flat
# `logpdf` -- so an observed origin conditions each child on its declared edge.
function _tree_step(step::Union{Sequential, Parallel}, events, o_idx::Int,
        ev_idx::Int, primary, ::Type{T}) where {T}
    sub = _subevent_slice(events, o_idx, ev_idx, length(step))
    return _tree_score(step, sub, 1, 2, primary, T)
end

# The `[origin, leaf_events...]` sub-event vector for a nested node: its origin
# (the parent's event) followed by the node's own `n` leaf-event slice. Always a
# freshly gathered `Vector{eltype(events)}` (one concrete return type, so the
# recursive `_tree_score` call stays type-stable); the gather reads only the
# constant event values, so the differentiated arithmetic is unaffected.
function _subevent_slice(events, o_idx::Int, ev_idx::Int, n::Int)
    out = Vector{eltype(events)}(undef, n + 1)
    out[1] = events[o_idx]
    @inbounds for k in 1:n
        out[k + 1] = events[ev_idx + k - 1]
    end
    return out
end
function _tree_step(step::UnivariateDistribution, events, o_idx::Int,
        ev_idx::Int, primary, ::Type{T}) where {T}
    o = events[o_idx]
    y = events[ev_idx]
    if o !== missing && y !== missing
        # Condition on the edge's declared censoring at the observed gap. Only the
        # DATA gap is converted to `T`; the log density keeps its own (AD-tracked)
        # type so a leaf-param gradient is not narrowed.
        return logpdf(step, convert(T, y) - convert(T, o))
    end
    if o === missing && y !== missing && primary !== nothing
        # Root origin latent: marginalise E_0 against this edge's core.
        return logpdf(primary_censored(_marginal_core(step), primary),
            convert(T, y))
    end
    # A missing endpoint contributes no factor.
    return zero(T)
end

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
    # The flat event vector carries one entry per LEAF event plus the root
    # origin, so a tree with nested composers needs `_nleaves(d) + 1` entries,
    # not `length(d.components) + 1` (which only counts the top-level steps).
    length(events) == length(d) + 1 || throw(DimensionMismatch(
        "a Sequential event vector needs $(length(d) + 1) entries " *
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
# `b`. The steps linking them are `components[a:(b - 1)]`.
#
# A SINGLE observed-bounded edge conditions on its OWN declared censoring: the
# edge's gap is scored through the edge distribution AS DECLARED (#329 "condition
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
    # segment reapplies the (latent) primary event.
    core = convolve_distributions(map(_marginal_core, collect(run)))
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
    # One entry per LEAF event plus the shared origin; a tree branch contributes
    # its whole subtree's leaf events, so the length is `_nleaves(d) + 1`.
    length(events) == length(d) + 1 || throw(DimensionMismatch(
        "a Parallel event vector needs $(length(d) + 1) entries " *
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
    primary === nothing && throw(ArgumentError(
        "Parallel shared-origin scoring needs censored branches with a " *
        "primary event; got plain branches"))

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
# a name-keyed `NamedTuple` (#351), so a NamedTuple is flattened over its values
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

# ---------------------------------------------------------------------------
# Full-path simulation: `rand` returns every event time / internal time
# ---------------------------------------------------------------------------
#
# For simulate-and-recover (the case studies) `rand` must produce the COMPLETE
# set of event times, not a summary. The composer-level `rand` delegates to
# `_composer_rand`, which branches on whether the composer is censored (its
# components carry an origin primary event). A plain composer keeps the generic
# per-leaf-value realisation (PR3a); a censored composer simulates the full
# event-time path including the latent origin draw.

# Sequential: a plain chain returns the per-step value vector; a censored chain
# returns the full event-time path `[E_0, E_1, ..., E_k]` with `E_0` the latent
# origin draw and each subsequent time the previous plus a continuous-core delay
# draw, so every internal event time is in the result.
function _composer_rand(rng::AbstractRNG, d::Sequential)
    _is_censored_composer(d.components) || return _composite_rand(
        rng, d.components, float(eltype(d)))
    primary = _origin_primary_event(d.components[1])
    cores = map(_marginal_core, d.components)
    T = float(promote_type(_param_eltype(primary),
        map(_param_eltype, cores)...))
    times = Vector{T}(undef, length(cores) + 1)
    times[1] = convert(T, rand(rng, primary))
    @inbounds for i in eachindex(cores)
        times[i + 1] = times[i] + rand(rng, cores[i])
    end
    return times
end

# Parallel: plain branches return one value per branch; censored branches
# sharing one latent origin return the full event-time vector
# `[O, Y_1, ..., Y_n]` with `O` the single shared origin draw and
# `Y_i = O + D_i`, so the shared origin and every branch observation are in the
# result.
function _composer_rand(rng::AbstractRNG, d::Parallel)
    primary = _is_censored_composer(d.components) ?
              _shared_primary_event(d.components) : nothing
    primary === nothing && return _composite_rand(
        rng, d.components, float(eltype(d)))
    cores = map(_marginal_core, d.components)
    T = float(promote_type(_param_eltype(primary),
        map(_param_eltype, cores)...))
    out = Vector{T}(undef, length(cores) + 1)
    o = convert(T, rand(rng, primary))
    out[1] = o
    @inbounds for i in eachindex(cores)
        out[i + 1] = o + rand(rng, cores[i])
    end
    return out
end
