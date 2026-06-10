# ============================================================================
# Per-event moments of a composed distribution via the standard interface
# ============================================================================
#
# A composed tree is a `Multivariate` distribution: `rand(d)` returns the full
# per-event realisation. The standard `Distributions.mean`/`var`/`std` are
# defined here to return the per-event moments in the SAME flat layout, a
# `Vector` matching `rand(d)`/[`event_names`](@ref). There is NO separate wrapper
# type: the multivariate `mean`/`var`/`std` IS the per-event view (pair it with
# `event_names(d)` for a labelled NamedTuple).
#
# Each event slot's moment is that of its underlying FREE delay: `free_leaf`
# peels the fixed censoring (double_interval_censored / Truncated / Weighted) off
# to the inner delay, so a `double_interval_censored(Gamma(2, 3.5))` edge reports
# the Gamma's mean (7.0), not the censored mean. The origin slot of a censored
# tree reports the primary (origin) event's moment.
#
# For a univariate node (a `Competing` mixture, a bare leaf) the standard scalar
# moment is kept (defined on `Competing` in `Competing.jl`, on leaves by
# Distributions.jl).

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

# --- standard moments on the composed tree (per-event Vector) ---------------

@doc "

Per-event means of a composed distribution, as a `Vector`.

`mean(d)` on a composed tree (a `Multivariate` distribution) returns the
per-event means in the SAME flat layout as `rand(d)` and [`event_names`](@ref):
for a censored tree the origin event's mean followed by one mean per leaf edge;
for a plain (uncensored) tree the per-step value means. Each edge's mean is that
of its underlying FREE delay, so a censored leaf (e.g.
`double_interval_censored(Gamma(2, 3.5))`) reports the inner delay mean (`7.0`).
Pair with [`event_names`](@ref) for a labelled NamedTuple.

# Examples
```@example
using CensoredDistributions, Distributions

tree = compose((onset_admit = double_interval_censored(Gamma(2.0, 3.5);
        primary_event = Uniform(0, 1), interval = 1.0),
    onset_notif = Gamma(0.7, 20.0)))
NamedTuple{event_names(tree)}(Tuple(mean(tree)))
```

# See also
- [`var`](@ref), [`std`](@ref): the matching per-event variance / std
- [`event_names`](@ref): the flat per-event labels
- [`endpoint`](@ref): collapse a chain to its terminal scalar
"
mean(d::Union{Sequential, Parallel}) = _event_moment_vector(d, _leaf_mean)

@doc "

Per-event variances of a composed distribution, as a `Vector`.

`var(d)` mirrors [`mean`](@ref), returning the variance of each event's
underlying FREE delay in the same flat per-event layout as `rand(d)` /
[`event_names`](@ref).

# Examples
```@example
using CensoredDistributions, Distributions

tree = compose((onset_admit = double_interval_censored(Gamma(2.0, 3.5);
        primary_event = Uniform(0, 1), interval = 1.0),
    onset_notif = Gamma(0.7, 20.0)))
NamedTuple{event_names(tree)}(Tuple(var(tree)))
```

# See also
- [`mean`](@ref), [`std`](@ref)
"
var(d::Union{Sequential, Parallel}) = _event_moment_vector(d, _leaf_var)

@doc "

Per-event standard deviations of a composed distribution, as a `Vector`.

`std(d)` is the elementwise square root of [`var`](@ref), in the same flat
per-event layout as `rand(d)` / [`event_names`](@ref).

# See also
- [`mean`](@ref), [`var`](@ref)
"
std(d::Union{Sequential, Parallel}) = sqrt.(var(d))

# A `Select` has no single layout (the active alternative is data-selected), so a
# whole-tree moment is ill-defined; direct the caller to the chosen alternative.
function mean(::Select)
    throw(ArgumentError(
        "mean(::Select) needs a selection; take the moment of the chosen " *
        "alternative, e.g. `mean(event(d, :index))`"))
end
function var(::Select)
    throw(ArgumentError(
        "var(::Select) needs a selection; take the moment of the chosen " *
        "alternative, e.g. `var(event(d, :index))`"))
end
function std(::Select)
    throw(ArgumentError(
        "std(::Select) needs a selection; take the moment of the chosen " *
        "alternative, e.g. `std(event(d, :index))`"))
end

# --- flat per-event moment vector -------------------------------------------
#
# `_event_moment_vector(d, f)` builds the per-event moment `Vector` matching the
# layout of `rand(d)`. A CENSORED tree's `rand` is the flat event path
# `[origin, target_1, ...]` keyed by `_flat_event_names`, so the moment vector is
# `[f(primary), f(edge_1), ...]` (the origin slot the primary event's moment,
# each later slot the free-delay moment of its leaf edge, `Competing` outcomes
# each their own slot). A PLAIN tree's `rand` is the per-step value vector, so
# the moment vector is the per-value free-delay moments. `f` is `_leaf_mean` or
# `_leaf_var`.

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

# A `Competing` step: one slot per outcome, each the outcome's free-delay moment
# (a nested-composer outcome uses its scalar marginal moment so the slot stays a
# scalar, mirroring the per-outcome event layout).
function _event_moment_step!(out, child::Competing, f::F, idx::Int) where {F}
    for i in eachindex(child.delays)
        out[idx + i - 1] = float(_outcome_scalar_moment(child.delays[i], f))
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
# moment; a `Competing`'s branch-prob-weighted mixture moment (over its free
# per-outcome moments, seeing through censored leaves, NOT the censored
# `mean`/`var(Competing)`); a `Latent` delegates to its wrapped distribution.
_outcome_scalar_moment(leaf, ::typeof(_leaf_mean)) = _leaf_mean(leaf)
_outcome_scalar_moment(leaf, ::typeof(_leaf_var)) = _leaf_var(leaf)
function _outcome_scalar_moment(c::Competing, ::typeof(_leaf_mean))
    return _competing_mix_mean(c)
end
function _outcome_scalar_moment(c::Competing, ::typeof(_leaf_var))
    return _competing_mix_var(c)
end
function _outcome_scalar_moment(d::Latent, f::F) where {F}
    return _outcome_scalar_moment(d.dist, f)
end

# A `Competing`'s branch-prob-weighted mixture mean / variance, built from the
# FREE per-outcome moments so it sees through censored leaves (NOT the censored
# `mean(Competing)`/`var(Competing)`, which lower through `as_mixture` and have
# no analytic moment for a censored leaf).
function _competing_mix_mean(c::Competing)
    scalar_means = map(d -> _outcome_scalar_moment(d, _leaf_mean), c.delays)
    return sum(c.branch_probs .* scalar_means)
end

function _competing_mix_var(c::Competing)
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
itself). `mean(endpoint(seq))` then gives the endpoint (total-delay) mean,
distinct from the per-event [`mean`](@ref) Vector.

# Examples
```@example
using CensoredDistributions, Distributions

seq = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
mean(endpoint(seq))
```

# See also
- [`observed_distribution`](@ref): the underlying lowering
- [`mean`](@ref): the per-event moment Vector
"
endpoint(d) = observed_distribution(d)
