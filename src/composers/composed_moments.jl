# ============================================================================
# Moments of a composed distribution: overall vs per-event (latent) views
# ============================================================================
#
# A composed tree exposes its moments at two levels:
#
#   1. The OVERALL observed-level moment, via the standard `Distributions.mean`/
#      `var`/`std` on the composer itself. `mean(d)` behaves like a normal delay
#      distribution's mean:
#        - a univariate-collapsible composer (a `Sequential` chain, a `Convolved`,
#          a `Resolve`, a censored leaf) returns the SCALAR moment of its
#          overall observed delay — the moment of `observed_distribution(d)`
#          (the convolved total for a chain, the marginal time-to-resolution for
#          a `Resolve`);
#        - a genuinely multivariate composer (a `Parallel`, several independent
#          observed endpoints) returns the per-ENDPOINT `Vector`, one overall
#          moment per branch endpoint (NOT the latent origin / intermediates).
#
#   2. The FULL per-event moment, via `mean(latent(d))`/`var(latent(d))`/
#      `std(latent(d))`. This is the "get the events" view: a `Vector` in the
#      SAME flat layout as `rand(latent(d))`/[`event_names`](@ref) — for a
#      censored tree the origin event then one moment per leaf edge, for a plain
#      (uncensored) tree the per-step value moments.
#
# Each event slot's moment is that of its underlying FREE delay: `free_leaf`
# peels the fixed censoring (double_interval_censored / Truncated / Weighted) off
# to the inner delay, so a `double_interval_censored(Gamma(2, 3.5))` edge reports
# the Gamma's mean (7.0), not the censored mean. The origin slot of a censored
# tree reports the primary (origin) event's moment.

# --- per-leaf moment (free-delay transparent) ------------------------------

# See through a `Weighted` wrapper to its inner delay too (the introspection
# `free_leaf` already peels PrimaryCensored / IntervalCensored / Truncated); the
# weight is fixed structure, not part of the delay moment.
free_leaf(d::Weighted) = free_leaf(d.dist)

# Mean / variance of a (possibly censored) leaf: that of its inner free delay.
# A plain leaf is its own free leaf; a `Convolved` free-leafs to itself and so
# reuses its additive `mean`/`var`.
_leaf_mean(leaf) = mean(free_leaf(leaf))
_leaf_var(leaf) = var(free_leaf(leaf))

# ============================================================================
# 1. Overall observed-level moments on the composer itself
# ============================================================================

@doc "

Overall mean of a composed distribution (the simple \"mean delay\").

`mean(d)` behaves like a normal delay distribution's mean. For a
univariate-collapsible composer (a [`Sequential`](@ref) chain, a
[`Convolved`](@ref), a [`Resolve`](@ref)) it returns the SCALAR mean of the
overall observed delay — the mean of [`observed_distribution`](@ref)`(d)` (the
convolved total for a chain, the marginal time-to-resolution for a `Resolve`).
For a genuinely multivariate [`Parallel`](@ref) (several independent observed
endpoints) it returns the per-ENDPOINT `Vector`, one overall mean per branch
endpoint, NOT the latent origin / intermediate events. Censoring is seen through
to the free delay.

For the FULL per-event breakdown (the origin and every event), take the moment of
the [`latent`](@ref) form: `mean(latent(d))` returns the per-event `Vector`
matching `rand(latent(d))`/[`event_names`](@ref).

# Examples
```@example
using CensoredDistributions, Distributions

seq = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
mean(seq)                 # overall mean delay (a scalar)
mean(latent(seq))         # the per-event mean Vector
```

# See also
- [`var`](@ref), [`std`](@ref): the matching overall variance / std
- [`latent`](@ref): the per-event view
- [`event_names`](@ref): the flat per-event labels
- [`endpoint`](@ref): collapse a chain to its terminal scalar
"
mean(d::Sequential) = _overall_moment(d, _leaf_mean)

@doc "

Overall variance of a composed distribution.

`var(d)` mirrors [`mean`](@ref): the scalar variance of the overall observed
delay for a univariate-collapsible composer (the variance of
[`observed_distribution`](@ref)`(d)`), or the per-ENDPOINT `Vector` for a
[`Parallel`](@ref). Take `var(latent(d))` for the FULL per-event variance Vector.

# See also
- [`mean`](@ref), [`std`](@ref), [`latent`](@ref)
"
var(d::Sequential) = _overall_moment(d, _leaf_var)

@doc "

Overall standard deviation of a composed distribution.

`std(d)` is `sqrt(var(d))` (or its elementwise form for a [`Parallel`](@ref)).
Take `std(latent(d))` for the FULL per-event std Vector.

# See also
- [`mean`](@ref), [`var`](@ref), [`latent`](@ref)
"
std(d::Sequential) = sqrt(var(d))

# A `Parallel` is genuinely multivariate: its overall moment is the per-ENDPOINT
# NamedTuple, one overall moment per branch endpoint keyed by `_endpoint_names`
# (a nested `Parallel` flattens its own endpoints in). The origin / intermediate
# events are NOT included; take `latent(d)` for the full per-event NamedTuple.
function mean(d::Parallel)
    return _as_named(_endpoint_names(d), _endpoint_moment_vector(d, _leaf_mean))
end
function var(d::Parallel)
    return _as_named(_endpoint_names(d), _endpoint_moment_vector(d, _leaf_var))
end
std(d::Parallel) = map(sqrt, var(d))

# `Resolve` already defines the scalar univariate moment (the marginal
# time-to-resolution, `mean(as_mixture(c))`) in `Resolve.jl`, matching the
# overall semantics; no override needed here.

# A `Choose` has no single layout (the active alternative is data-selected), so a
# whole-tree moment is ill-defined; direct the caller to the chosen alternative.
function mean(::Choose)
    throw(ArgumentError(
        "mean(::Choose) needs a selection; take the moment of the chosen " *
        "alternative, e.g. `mean(event(d, :index))`"))
end
function var(::Choose)
    throw(ArgumentError(
        "var(::Choose) needs a selection; take the moment of the chosen " *
        "alternative, e.g. `var(event(d, :index))`"))
end
function std(::Choose)
    throw(ArgumentError(
        "std(::Choose) needs a selection; take the moment of the chosen " *
        "alternative, e.g. `std(event(d, :index))`"))
end

# --- the overall scalar moment of a univariate-collapsible node -------------
#
# `_overall_moment(d, f)` is the scalar overall moment of a node that collapses
# to a single observed univariate quantity. For independent steps the mean of the
# sum is the sum of means and the variance of the sum is the sum of variances, so
# a `Sequential` is the additive total over its free-delay leaf steps (the moment
# of `observed_distribution(d)`, computed free-delay-transparently). `f` is
# `_leaf_mean` or `_leaf_var`.
function _overall_moment(d::Sequential, f::F) where {F}
    sum(_overall_moment(c, f) for c in d.components)
end
_overall_moment(c::Resolve, ::typeof(_leaf_mean)) = _one_of_mix_mean(c)
_overall_moment(c::Resolve, ::typeof(_leaf_var)) = _one_of_mix_var(c)
# A racing-hazard node collapses to its marginal any-event (min) moment.
_overall_moment(c::Compete, ::typeof(_leaf_mean)) = mean(c)
_overall_moment(c::Compete, ::typeof(_leaf_var)) = var(c)
# A `Parallel` step inside a chain has several independent endpoints, so the chain
# has no single observed scalar to collapse to (mirroring `observed_distribution`,
# which rejects a `Sequential` whose step is a `Parallel`).
function _overall_moment(::Parallel, ::F) where {F}
    throw(ArgumentError(
        "cannot collapse a composer with a Parallel branch to a single overall " *
        "moment; take `mean(latent(d))` for the per-event vector, or the moment " *
        "of each `event(d, name)` branch"))
end
_overall_moment(leaf, f::F) where {F} = float(f(leaf))

# The per-ENDPOINT moment Vector of a `Parallel`: one overall scalar moment per
# branch endpoint, in branch order. A nested `Parallel` branch contributes each
# of its own endpoints (flattened), so the vector length matches the number of
# independent observed endpoints. A `Sequential`/`Resolve`/leaf branch collapses
# to its single overall scalar via `_overall_moment`.
function _endpoint_moment_vector(d::Parallel, f::F) where {F}
    out = Float64[]
    for branch in d.components
        _append_endpoint_moments!(out, branch, f)
    end
    return out
end

function _append_endpoint_moments!(out, branch::Parallel, f::F) where {F}
    (for b in branch.components
            _append_endpoint_moments!(out, b, f)
        end; out)
end
function _append_endpoint_moments!(out, branch, f::F) where {F}
    push!(out, _overall_moment(branch, f))
end

# ============================================================================
# 2. Per-event (latent) moments via the `latent` wrapper
# ============================================================================
#
# `latent(d)` over a composed tree is the per-event view: `rand(latent(d))` and
# its moments are the FULL flat per-event Vector. The wrapper delegates the
# realisation / scoring to the wrapped composer (whose own `rand`/`logpdf` ARE the
# per-event layout) and exposes the per-event moment Vector via `mean`/`var`/`std`.

# `latent(d)` over a composer wraps it in the `Latent` per-event view. (The
# leaf-level `latent(primary_censored(...))` keeps its own `[primary, observed]`
# methods in `Latent.jl`.)
const _ComposerLatent = Latent{<:Union{Sequential, Parallel}}

Base.length(d::_ComposerLatent) = length(rand(d))
Base.eltype(::Type{<:_ComposerLatent}) = Float64

# The per-event realisation / score IS the wrapped composer's own multivariate
# `rand`/`logpdf`.
Base.rand(rng::AbstractRNG, d::_ComposerLatent) = rand(rng, d.dist)
logpdf(d::_ComposerLatent, x::AbstractVector) = logpdf(d.dist, x)

# Flat per-event names of a latent composer view: those of the wrapped composer.
event_names(d::_ComposerLatent) = event_names(d.dist)

@doc "

Per-event means of a composed distribution, as a labelled `NamedTuple`.

`mean(latent(d))` returns the per-event means of a composed tree as a
`NamedTuple` keyed by [`event_names`](@ref), in the SAME flat layout as
`rand(latent(d))`: for a censored tree the origin event's mean followed by one
mean per leaf edge; for a plain (uncensored) tree the per-step value means. Each
edge's mean is that of its underlying FREE delay, so a censored leaf (e.g.
`double_interval_censored(Gamma(2, 3.5))`) reports the inner delay mean (`7.0`).
For the overall (scalar) mean delay use [`mean`](@ref)`(d)` on the bare composer
instead.

# Examples
```@example
using CensoredDistributions, Distributions

tree = compose((onset_admit = double_interval_censored(Gamma(2.0, 3.5);
        primary_event = Uniform(0, 1), interval = 1.0),
    onset_notif = Gamma(0.7, 20.0)))
mean(latent(tree))
```

# See also
- [`mean`](@ref): the overall (scalar / per-endpoint) moment of the bare composer
- [`var`](@ref), [`std`](@ref): the matching per-event variance / std
- [`event_names`](@ref): the flat per-event labels
"
function mean(d::_ComposerLatent)
    return _as_named(_output_names(d.dist),
        _event_moment_vector(d.dist, _leaf_mean))
end

@doc "

Per-event variances of a composed distribution, as a labelled `NamedTuple`.

`var(latent(d))` mirrors `mean(latent(d))`, returning the variance of each
event's underlying FREE delay as a `NamedTuple` keyed by [`event_names`](@ref),
in the same flat per-event layout as `rand(latent(d))`.

# See also
- `mean(latent(d))`, `std(latent(d))`, [`var`](@ref)
"
function var(d::_ComposerLatent)
    return _as_named(_output_names(d.dist),
        _event_moment_vector(d.dist, _leaf_var))
end

@doc "

Per-event standard deviations of a composed distribution, as a labelled
`NamedTuple`.

`std(latent(d))` is the elementwise square root of `var(latent(d))`, a
`NamedTuple` keyed by [`event_names`](@ref) in the same flat per-event layout as
`rand(latent(d))`.

# See also
- `mean(latent(d))`, `var(latent(d))`, [`std`](@ref)
"
std(d::_ComposerLatent) = map(sqrt, var(d))

# --- flat per-event moment vector -------------------------------------------
#
# `_event_moment_vector(d, f)` builds the per-event moment `Vector` matching the
# layout of `rand(d)` for the wrapped composer. A CENSORED tree's `rand` is the
# flat event path `[origin, target_1, ...]` keyed by `_flat_event_names`, so the
# moment vector is `[f(primary), f(edge_1), ...]` (the origin slot the primary
# event's moment, each later slot the free-delay moment of its leaf edge,
# `Resolve` outcomes each their own slot). A PLAIN tree's `rand` is the per-step
# value vector, so the moment vector is the per-value free-delay moments. `f` is
# `_leaf_mean` or `_leaf_var`.

function _event_moment_vector(d::Union{Sequential, Parallel}, f::F) where {F}
    primary = _tree_primary_event(d)
    primary === nothing && return _value_moment_vector(d, f)
    out = Vector{Float64}(undef, _event_nleaves(d.components) + 1)
    out[1] = float(f(primary))
    _event_moment_targets!(out, d, f, 2)
    return out
end

# Fill the target-event moment slots of composer `d` from flat index `start`.
# Returns the next free index. Mirrors the `_flat_event_names` / `_tree_rand!`
# depth-first walk: a `Sequential` threads step to step, a `Parallel` hangs each
# branch off the shared origin, both filling one slot per leaf edge.
function _event_moment_targets!(out, d::Union{Sequential, Parallel}, f::F,
        start::Int) where {F}
    idx = start
    for child in d.components
        idx = _event_moment_step!(out, child, f, idx)
    end
    return idx
end

function _event_moment_step!(out, child::Union{Sequential, Parallel}, f::F,
        idx::Int) where {F}
    return _event_moment_targets!(out, child, f, idx)
end

# A `Resolve` step: one slot per outcome, each the outcome's free-delay moment
# (a nested-composer outcome uses its scalar marginal moment so the slot stays a
# scalar, mirroring the per-outcome event layout).
function _event_moment_step!(out, child::Resolve, f::F, idx::Int) where {F}
    for i in eachindex(child.delays)
        out[idx + i - 1] = float(_outcome_scalar_moment(child.delays[i], f))
    end
    return idx + _n_branches(child)
end

# A racing-hazard step: one slot per outcome. Cause and timing are COUPLED, so a
# per-slot scalar uses the marginal any-event (min) moment for every outcome slot
# (the racing time is shared across causes); the per-cause split is a probability,
# not a separate time scale.
function _event_moment_step!(out, child::Compete, f::F, idx::Int) where {F}
    m = float(f(child))
    for i in 1:_n_branches(child)
        out[idx + i - 1] = m
    end
    return idx + _n_branches(child)
end

function _event_moment_step!(out, child, f::F, idx::Int) where {F}
    out[idx] = float(f(child))
    return idx + 1
end

# The per-VALUE moment vector for a plain (uncensored) tree: one slot per leaf
# value in `_child_nleaves` order, each a free-delay moment, matching the generic
# `_composite_rand` value layout.
function _value_moment_vector(d::Union{Sequential, Parallel}, f::F) where {F}
    out = Vector{Float64}(undef, _nleaves(d.components))
    _value_moment_fill!(out, d.components, f, 1)
    return out
end

function _value_moment_fill!(out, components::Tuple, f::F, start::Int) where {F}
    idx = start
    for c in components
        idx = _value_moment_child!(out, c, f, idx)
    end
    return idx
end

function _value_moment_child!(out, c::Union{Sequential, Parallel}, f::F,
        idx::Int) where {F}
    return _value_moment_fill!(out, c.components, f, idx)
end

function _value_moment_child!(out, c, f::F, idx::Int) where {F}
    out[idx] = float(_outcome_scalar_moment(c, f))
    return idx + 1
end

# The SCALAR marginal moment of an outcome / value child: a leaf's free-delay
# moment; a `Resolve`'s branch-prob-weighted mixture moment (over its free
# per-outcome moments, seeing through censored leaves, NOT the censored
# `mean`/`var(Resolve)`); a `Latent` delegates to its wrapped distribution.
_outcome_scalar_moment(leaf, ::typeof(_leaf_mean)) = _leaf_mean(leaf)
_outcome_scalar_moment(leaf, ::typeof(_leaf_var)) = _leaf_var(leaf)
function _outcome_scalar_moment(c::Resolve, ::typeof(_leaf_mean))
    return _one_of_mix_mean(c)
end
function _outcome_scalar_moment(c::Resolve, ::typeof(_leaf_var))
    return _one_of_mix_var(c)
end
# A racing-hazard outcome's scalar moment is the node's marginal any-event moment.
_outcome_scalar_moment(c::Compete, ::typeof(_leaf_mean)) = mean(c)
_outcome_scalar_moment(c::Compete, ::typeof(_leaf_var)) = var(c)
function _outcome_scalar_moment(d::Latent, f::F) where {F}
    return _outcome_scalar_moment(d.dist, f)
end
# Disambiguate a `Latent` against the typed-`f` leaf methods above: a `Latent`
# delegates to its wrapped distribution for both the mean and the variance.
function _outcome_scalar_moment(d::Latent, ::typeof(_leaf_mean))
    _outcome_scalar_moment(d.dist, _leaf_mean)
end
function _outcome_scalar_moment(d::Latent, ::typeof(_leaf_var))
    _outcome_scalar_moment(d.dist, _leaf_var)
end

# A `Resolve`'s branch-prob-weighted mixture mean / variance, built from the
# FREE per-outcome moments so it sees through censored leaves (NOT the censored
# `mean(Resolve)`/`var(Resolve)`, which lower through `as_mixture` and have
# no analytic moment for a censored leaf).
function _one_of_mix_mean(c::Resolve)
    scalar_means = map(d -> _outcome_scalar_moment(d, _leaf_mean), c.delays)
    return sum(c.branch_probs .* scalar_means)
end

function _one_of_mix_var(c::Resolve)
    scalar_means = map(d -> _outcome_scalar_moment(d, _leaf_mean), c.delays)
    scalar_vars = map(d -> _outcome_scalar_moment(d, _leaf_var), c.delays)
    mix_mean = sum(c.branch_probs .* scalar_means)
    second = sum(c.branch_probs .* (scalar_vars .+ scalar_means .^ 2))
    return second - mix_mean^2
end

# --- endpoint: the terminal scalar of a chain -------------------------------

@doc "

Collapse a composed chain to its terminal scalar distribution.

`endpoint(d)` is an alias for [`observed_distribution`](@ref): it lowers a
composed distribution to the single univariate quantity a censoring wrapper would
observe (a [`Sequential`](@ref) chain's total elapsed time, a univariate node
itself). `mean(endpoint(seq))` gives the endpoint (total-delay) mean, the same
value [`mean`](@ref)`(seq)` returns; use [`latent`](@ref) for the per-event
breakdown.

# Examples
```@example
using CensoredDistributions, Distributions

seq = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
mean(endpoint(seq))
```

# See also
- [`observed_distribution`](@ref): the underlying lowering
- [`mean`](@ref): the overall (scalar / per-endpoint) moment
- [`latent`](@ref): the per-event moment Vector
"
endpoint(d) = observed_distribution(d)
