# ============================================================================
# Censored specialisations of the generic composers
# ============================================================================
#
# The generic `Sequential` / `Parallel` / `Resolve` composers score a
# value vector with one entry per step/branch. When their internal nodes are our
# censored distributions (`primary_censored` / `interval_censored` /
# `double_interval_censored`), per-record marginalisation is AUTOMATIC and
# data-driven, selected by MULTIPLE DISPATCH on the value vector's element type
# (a `Missing`-admitting event vector) and on the node types. There is no
# runtime predicate, no `mode` keyword, and no new node-type hierarchy.
#
# Evaluating a censored chain against an EVENT vector
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
# origin conditions. `Resolve` already lowers to a `MixtureModel`
# (`as_mixture`), so it needs no event-vector specialisation here.

#
# Shared recovery helpers used by the censored-composer scoring (flat in
# `censored_scoring_flat.jl`, nested-tree in `censored_scoring_tree.jl`,
# one_of-slice in `censored_one_of.jl`) and the full-path simulation in
# `censored_rand.jl`. Those files are included AFTER this one (see the include
# list in `src/CensoredDistributions.jl`) so the helpers and the `_Nested` /
# `_Flat` traits are defined first.
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
# A `shared(:tag, ...)` leaf is TRANSPARENT to scoring (#394), so a shared
# censored leaf at a tree origin must surface its wrapped leaf's primary event
# like the other censoring wrappers; without this the origin's latent primary is
# lost and a `shared(:inc, primary_censored(...))` first step scores as an
# UNCENSORED origin (#395 follow-up: the shared wrapper was stripped by
# `_marginal_core` / `cdf` but not by the origin-primary traversal, diverging
# from the untagged leaf).
_origin_primary_event(d::Shared) = _origin_primary_event(d.dist)
_origin_primary_event(::UnivariateDistribution) = nothing
# A nested multivariate composer is not a primary-censored origin leaf, so it
# surfaces no origin primary event (the censored treatment is resolved within
# the child). `Resolve` is univariate and already hits the fallback above.
_origin_primary_event(::Union{Sequential, Parallel}) = nothing
# A nested `Choose` resolves censoring within the committed alternative (the
# first on the flat path); a `Latent` resolves it within its wrapped node.
# Neither surfaces a flat origin primary event.
_origin_primary_event(::Choose) = nothing
_origin_primary_event(d::Latent) = _origin_primary_event(d.dist)

# The continuous delay core of a (possibly censored) node, for marginalisation:
# strip every censoring layer so a marginalised run convolves only continuous
# delays, never a discrete/windowed object. A `Convolved` node is already a
# continuous sum and is left intact for the fold — including when it is nested
# inside a censoring wrapper (e.g. `primary_censored(Sequential(...))`, whose
# collapsed observed total is a `Convolved`). Stripping with `get_dist_recursive`
# alone would over-unwrap that nested `Convolved` into its component VECTOR (via
# `get_dist(::Convolved)`), and a vector is not a distribution: the downstream
# `rand`/`logpdf`/`_param_eltype` machinery would then draw a random COMPONENT
# rather than the convolution sum and `params(::Vector)` would error (#363). So
# strip one wrapper layer at a time and stop at the first `Convolved`.
_marginal_core(d::UnivariateDistribution) = _strip_to_core(d)
_marginal_core(d::Convolved) = d

# Strip one censoring/wrapper layer at a time (`get_dist`), stopping at a
# `Convolved` (the continuous sum is the core) or at a node `get_dist` no longer
# unwraps. This keeps a `Convolved` nested inside a wrapper intact instead of
# unwrapping it into its component vector the way `get_dist_recursive` would.
function _strip_to_core(d)
    d isa Convolved && return d
    next = get_dist(d)
    next === d && return d
    next isa AbstractVector && return get_dist_recursive(d)
    return _strip_to_core(next)
end

# The secondary interval censoring a leaf carries (its `IntervalCensored` layer),
# recovered THROUGH the `Truncated` / `Weighted` wrappers, or `nothing` when the
# leaf has none (uncensored or primary-only). The sim walk draws a continuous
# delay from `_marginal_core` and then applies this interval to the RECORDED
# event times, matching the scorer that floors each gap to its day. A
# primary-only leaf returns `nothing`, so it stays continuous (no spurious
# flooring). Dispatch on the node type is the whole selection.
_leaf_interval(d::IntervalCensored) = d
_leaf_interval(d::Truncated) = _leaf_interval(d.untruncated)
_leaf_interval(d::Weighted) = _leaf_interval(d.dist)
# A `shared(:tag, ...)` leaf is transparent to scoring (#394), so recover the
# secondary interval THROUGH the shared wrapper like the other wrappers;
# otherwise a `shared(:inc, double_interval_censored(...))` leaf drops its
# interval discretisation and scores its gaps continuously, diverging from the
# untagged leaf.
_leaf_interval(d::Shared) = _leaf_interval(d.dist)
_leaf_interval(::UnivariateDistribution) = nothing

# The BARE continuous core of an edge for a SAMPLED-endpoint latent edge (#453):
# every censoring layer stripped (the same `_marginal_core` the marginal scorer
# uses), so NO primary AND no secondary interval. When an edge's predecessor or
# target is a sampled continuous latent, the marginal form scores the edge on this
# bare core — `_parallel_conditional_logpdf` conditions each flat branch on
# `_marginal_core`, and `_sequential_segment` convolves the `_marginal_core`s
# across a sampled-intermediate run. Re-applying ANY censoring (primary or the
# secondary interval) on a sampled-endpoint edge would diverge from the marginal
# and double-count the within-window uncertainty already represented by the
# sampled continuous time. Shared by the per-record latent scoring (the DynamicPPL
# ext) and the vectorised endpoint-observed chain path so the two latent forms and
# the marginal all agree. An OBSERVED-to-OBSERVED edge keeps its declared
# censoring (#419) and never reaches here.
_bare_latent_edge(edge) = _marginal_core(edge)

# Apply a leaf's secondary interval to a recorded (continuous) event time: floor
# to the regular width or snap to the arbitrary boundary, mirroring
# `rand(::IntervalCensored)`. A `nothing` interval leaves the value continuous.
_apply_leaf_interval(value, ::Nothing) = value
function _apply_leaf_interval(value, d::IntervalCensored)
    is_regular_intervals(d) &&
        return floor_to_interval(value, interval_width(d))
    return find_interval_boundary(value, d.boundaries)
end

# The secondary interval of the edge that ANCHORS a composer's origin: the first
# step of a `Sequential` (recursing into a nested first step), the shared-origin
# edge of a `Parallel`. The recorded origin slot is discretised to this interval
# so a first-level gap (its event minus the origin) reflects `floor(target) -
# floor(origin)` and is unbiased; a primary-only origin edge returns `nothing`,
# leaving the origin continuous.
_origin_interval(d::Sequential) = _origin_interval(d.components[1])
_origin_interval(d::Parallel) = _shared_origin_interval(d.components)
_origin_interval(d::AbstractOneOf) = _shared_origin_interval(d.delays)
# A nested `Choose` / `Latent` anchors its origin within the routed alternative /
# wrapped node; the alternatives share one origin interval, so the first is
# representative. Mirrors the `_tree_primary_event` recursion so a Parallel whose
# origin-anchor branch is itself a `Choose`/`Latent` still discretises its origin
# slot (reached only when that branch is the first censored one, e.g. a
# single-branch `compose((x = select(...),))`, issue #436).
_origin_interval(d::Choose) = _shared_origin_interval(d.alternatives)
_origin_interval(d::Latent) = _origin_interval(d.dist)
_origin_interval(d::UnivariateDistribution) = _leaf_interval(d)

# The shared origin interval of a `Parallel`/`Resolve`: the branches hang off
# one origin, so they must agree on its interval; the first non-`nothing` is
# returned (a mismatch is a malformed shared origin, but the scorer/data already
# assume one shared origin so this stays a simple first-found).
function _shared_origin_interval(components)
    @inbounds for c in components
        iv = _origin_interval(c)
        iv === nothing || return iv
    end
    return nothing
end

# Whether a component is itself a nested composer (a branch/step that recurses)
# rather than a leaf edge. Dispatch on the type, so the recursion is selected at
# compile time with no runtime type lookup. A `Resolve` is univariate (a single
# marginal time-to-resolution leaf), so it is NOT a nested composer here.
_is_nested_composer(::Union{Sequential, Parallel}) = true
_is_nested_composer(::UnivariateDistribution) = false

# Whether any of a composer's components is itself a nested composer. Picks the
# recursive tree walk over the flat one-level scoring. Resolved on the component
# TYPES (the `Tuple` element types), so it is a compile-time constant and the
# branch on it is eliminated, keeping the scoring type-stable.
_any_nested_composer(components::Tuple) = any(_is_nested_composer, components)

# A SINGLETON trait carrying the nested/flat choice in its TYPE, so the nested and
# flat scoring become separate dispatched methods. A plain `if _any_nested_...`
# branch leaves BOTH branches in one method body, and the compiled AD backends
# (Mooncake) build a rule for every reachable branch -- including the flat
# `convolve_distributions` marginalisation path that hard-crashes them
# uncatchably. Splitting by dispatch keeps the flat path out of the nested
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
    # A `Resolve` component also forces the tree path: its multi-slot outcome
    # layout is scored by `_tree_step(::Resolve)`, not the flat segment-grouped
    # scorer. A `Choose` component likewise forces the tree path: a nested Choose
    # routes per row to one of its alternatives (scored by `_tree_step(::Choose)`),
    # not by the flat one-level segment scoring that would silently commit to a
    # single alternative.
    has_nested = any(
        t -> t <: Sequential || t <: Parallel || t <: AbstractOneOf ||
             t <: Choose,
        fieldtypes(C))
    return has_nested ? :(_Nested()) : :(_Flat())
end

# Whether a composer's components carry primary censoring (an origin primary
# event), checked on the component types. `true` selects the censored
# full-event-path `rand`; `false` keeps the generic per-leaf-value `rand`.
function _is_censored_composer(components::Tuple)
    any(c -> _origin_primary_event(c) !== nothing, components)
end
