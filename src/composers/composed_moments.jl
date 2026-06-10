# ============================================================================
# Per-edge delay moments of a composed distribution
# ============================================================================
#
# `edge_means(d)` / `edge_vars(d)` walk a composed distribution and return ALL
# per-edge delay moments at once, keyed by edge name, mirroring the tree. Each
# edge's moment is that of its underlying FREE delay: `free_leaf` peels the fixed
# censoring (double_interval_censored / Truncated / Weighted) off to the inner
# delay, so a `double_interval_censored(Gamma(2, 3.5))` edge reports the Gamma's
# mean (7.0), not the censored mean.
#
# Recursion mirrors `params`: a `Sequential`/`Parallel` returns a NamedTuple of
# per-edge moments keyed by component names; a `Competing` returns per-outcome
# moments plus the mixture moment; a `Select` returns per-branch moments keyed by
# branch name; a `Latent` of a composer delegates to the wrapped composer. A
# `Convolved` already has a `mean`/`var` (sum over components), reused directly.
# Hand-rolled type-stable recursion over the component tuples, like the rest of
# the introspection layer.

# --- per-leaf moment (free-delay transparent) ------------------------------

# See through a `Weighted` wrapper to its inner delay too (the introspection
# `free_leaf` already peels PrimaryCensored / IntervalCensored / Truncated); the
# weight is fixed structure, not part of the delay moment.
free_leaf(d::Weighted) = free_leaf(d.dist)

# Mean / variance of a (possibly censored) leaf: that of its inner free delay.
# A plain leaf is its own free leaf; a `Convolved` free-leafs to itself and so
# reuses its additive `mean`/`var`.
_edge_mean(leaf) = mean(free_leaf(leaf))
_edge_var(leaf) = var(free_leaf(leaf))

# --- recursion over the composer tree --------------------------------------

@doc "

All per-edge delay means of a composed distribution, keyed by edge name.

`edge_means(d)` walks the composed distribution and returns every edge's delay
mean at once, mirroring the tree. A [`Sequential`](@ref)/[`Parallel`](@ref)
returns a `NamedTuple` keyed by its component names; a [`Competing`](@ref)
returns its per-outcome means plus a `mixture` entry (the marginal
time-to-resolution mean); a [`Select`](@ref) returns its per-branch means keyed
by branch name; a `Latent` of a composer returns the wrapped composer's edge
means. Each edge's mean is that of its underlying FREE delay, so a censored leaf
(e.g. `double_interval_censored(Gamma(2, 3.5))`) reports the inner delay mean
(`7.0`); a [`Convolved`](@ref) edge reuses its additive mean.

Pair with [`update`](@ref) and [`chain_to_params`](@ref) to read all posterior
delay means off a fitted composed object in one call,
`edge_means(update(template, means))`, rather than extracting each by hand.

# Arguments
- `d`: the composed distribution (or bare leaf) to read edge means from.

# Examples
```@example
using CensoredDistributions, Distributions

tree = compose((onset_admit = double_interval_censored(Gamma(2.0, 3.5);
        primary_event = Uniform(0, 1), interval = 1.0),
    onset_notif = Gamma(0.7, 20.0)))
edge_means(tree)
```

# See also
- [`edge_vars`](@ref): the matching per-edge variances
- [`event_names`](@ref), [`get_event`](@ref): name introspection
- [`update`](@ref): set posterior parameters before reading means
"
edge_means(d::Union{Sequential, Parallel}) = _edge_moments(d, _edge_mean)

@doc "

All per-edge delay variances of a composed distribution, keyed by edge name.

`edge_vars(d)` mirrors [`edge_means`](@ref), returning the variance of each
edge's underlying FREE delay in the same nested shape.

# Arguments
- `d`: the composed distribution (or bare leaf) to read edge variances from.

# Examples
```@example
using CensoredDistributions, Distributions

tree = compose((onset_admit = double_interval_censored(Gamma(2.0, 3.5);
        primary_event = Uniform(0, 1), interval = 1.0),
    onset_notif = Gamma(0.7, 20.0)))
edge_vars(tree)
```

# See also
- [`edge_means`](@ref): the matching per-edge means
"
edge_vars(d::Union{Sequential, Parallel}) = _edge_moments(d, _edge_var)

edge_means(c::Competing) = _competing_means(c)
edge_vars(c::Competing) = _competing_vars(c)

edge_means(d::Select) = _select_moments(d, _edge_mean)
edge_vars(d::Select) = _select_moments(d, _edge_var)

# A Latent of a composer reports the wrapped composer's edge moments.
edge_means(d::Latent) = edge_means(d.dist)
edge_vars(d::Latent) = edge_vars(d.dist)

# A bare (possibly censored) leaf reports its single free-delay moment.
edge_means(leaf) = _edge_mean(leaf)
edge_vars(leaf) = _edge_var(leaf)

# Per-component NamedTuple keyed by edge names; a composer child recurses, a leaf
# reports its free-delay moment. `f` is `_edge_mean` or `_edge_var`.
function _edge_moments(d::Union{Sequential, Parallel}, f::F) where {F}
    names = component_names(d)
    vals = map(c -> _child_moment(c, f), d.components)
    return NamedTuple{names}(vals)
end

_child_moment(c::Union{Sequential, Parallel}, f::F) where {F} = _edge_moments(c, f)
_child_moment(c::Competing, ::typeof(_edge_mean)) = _competing_means(c)
_child_moment(c::Competing, ::typeof(_edge_var)) = _competing_vars(c)
_child_moment(c::Select, f::F) where {F} = _select_moments(c, f)
_child_moment(c::Latent, f::F) where {F} = _child_moment(c.dist, f)
_child_moment(leaf, f::F) where {F} = f(leaf)

# SCALAR marginal moment of a single outcome, used ONLY for the `mixture`
# aggregate. A leaf's marginal mean/var is its free-delay moment; a nested
# composer outcome (a univariate, legal `Competing`) reports a NamedTuple from
# `_child_moment`, so the aggregate instead takes its scalar marginal
# mean/var (the marginal time-to-resolution), not the NamedTuple. This keeps
# the per-outcome entries (which DO recurse to NamedTuples) intact while the
# branch-prob-weighted aggregate stays a scalar.
#
# A nested `Competing` reuses its OWN free `mixture` moment (`_competing_means`/
# `_competing_vars`), which is built from the free per-outcome moments, NOT the
# censored `mean(c)`/`var(c)` (which lower through `as_mixture` and so report the
# censored marginal — and have no analytic moment for a censored leaf). This
# keeps the free-leaf transparency through arbitrarily nested Competing nodes.
_outcome_scalar_mean(leaf) = _edge_mean(leaf)
_outcome_scalar_mean(c::Competing) = _competing_means(c).mixture
_outcome_scalar_mean(d::Latent) = _outcome_scalar_mean(d.dist)
_outcome_scalar_var(leaf) = _edge_var(leaf)
_outcome_scalar_var(c::Competing) = _competing_vars(c).mixture
_outcome_scalar_var(d::Latent) = _outcome_scalar_var(d.dist)

# Competing: per-outcome free-delay means keyed by outcome name plus a `mixture`
# entry, the branch-prob-weighted mean of the outcomes. The per-outcome entries
# recurse (a nested composer outcome reports its own NamedTuple); the mixture
# uses each outcome's SCALAR marginal mean so it stays a scalar even when an
# outcome is itself a composer. Built from the free per-outcome means (not the
# censored `mean(Competing)`) so it sees through censored leaves.
function _competing_means(c::Competing)
    outcome_vals = map(d -> _child_moment(d, _edge_mean), c.delays)
    outcomes = NamedTuple{c.names}(outcome_vals)
    scalar_means = map(_outcome_scalar_mean, c.delays)
    mixture = sum(c.branch_probs .* scalar_means)
    return merge(outcomes, (; mixture = mixture))
end

# Competing: per-outcome free-delay variances plus the mixture variance,
# `Σ p_i (σ_i² + μ_i²) − (Σ p_i μ_i)²`, from the per-outcome SCALAR marginal
# moments so the aggregate is a scalar even for a nested-composer outcome.
function _competing_vars(c::Competing)
    outcome_vars = map(d -> _child_moment(d, _edge_var), c.delays)
    outcomes = NamedTuple{c.names}(outcome_vars)
    scalar_means = map(_outcome_scalar_mean, c.delays)
    scalar_vars = map(_outcome_scalar_var, c.delays)
    mix_mean = sum(c.branch_probs .* scalar_means)
    second = sum(c.branch_probs .* (scalar_vars .+ scalar_means .^ 2))
    return merge(outcomes, (; mixture = second - mix_mean^2))
end

# Select: per-branch moments keyed by branch name (each branch may itself nest).
function _select_moments(d::Select, f::F) where {F}
    vals = map(a -> _child_moment(a, f), d.alternatives)
    return NamedTuple{d.names}(vals)
end
