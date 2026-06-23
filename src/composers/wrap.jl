# ============================================================================
# Wrapping a composer with an external censoring wrapper
# ============================================================================
#
# This layer defines what it MEANS to apply a censoring or truncation wrapper
# (`primary_censored` / `interval_censored` / `double_interval_censored` /
# `truncate_to_horizon`) ON TOP of a composer. It is the node-level direction: a
# censoring/truncation MODIFIER applies at ANY level — to a leaf OR to a whole
# composed node alike (tenet 7) — and the modifier keeps the TREE SHAPE
# unchanged, so the wrapped node is still record-scoreable.
#
# A modifier over a composer DISTRIBUTES the SHARED observation resolution DOWN
# into the node's LEAF CORES, returning a node of the SAME shape whose leaves all
# carry that one resolution:
#   - `Sequential` -> a `Sequential` of the same step names whose every leaf core
#     is wrapped (the origin step's primary anchors the latent origin once and
#     each step is interval-/truncation-resolved), so `logpdf(wrapped, rows)`
#     scores the per-event record exactly as the canonical per-leaf construction
#     `Sequential(modifier(d_1), modifier(d_2), ...)` does;
#   - `Parallel` -> a `Parallel` of the same branch names, each branch endpoint
#     wrapped on its own (its branches hang off one shared origin);
#   - `Choose` -> a `Choose` of the same alternatives, each wrapped;
#   - `Resolve` / `Compete` (the one_of nodes) -> the SAME node with each
#     OUTCOME's delay core wrapped, keeping the per-outcome slots so the result
#     scores a record per outcome (the loser's slot is `missing`). A `NoEvent`
#     no-event branch carries no delay and passes through unchanged.
# Each leaf is reduced to its continuous CORE (`_marginal_core`) BEFORE the
# modifier is reapplied, so wrapping an ALREADY-censored node re-resolves it at
# the new resolution instead of stacking `PrimaryCensored{PrimaryCensored{...}}`.
#
# `Convolved` is univariate (the observed sum), so the existing
# `UnivariateDistribution` wrapper methods accept it directly, and the SCALAR
# combine-then-censor lowering of a chain total or a one_of marginal
# time-to-resolution stays available EXPLICITLY through
# `modifier(observed_distribution(seq))` / `modifier(convolve_distributions(...))`
# — that is the dual, scalar direction, distinct from the node-level wrap here.
# Dispatch on the composer type — no runtime predicate, no new hierarchy.

@doc "

The univariate scalar a censoring wrapper observes for a composer.

A censoring wrapper observes one quantity, so wrapping a composer first lowers
it to that quantity:

- a [`Convolved`](@ref) is already univariate (the observed sum) and is returned
  unchanged;
- a [`Resolve`](@ref) / [`Compete`](@ref) one_of node's observed quantity is the
  marginal time-to-resolution, returned as the `MixtureModel` over its outcome
  delays (via [`as_mixture`](@ref)); this is the SCALAR lowering used by
  `modifier(observed_distribution(node))`, distinct from the node-level wrap that
  distributes the modifier into the outcome slots;
- a [`Sequential`](@ref) chain's observed quantity is the total elapsed time
  from origin to the terminal event, the convolution of its steps, returned as
  a [`Convolved`](@ref).

A [`Parallel`](@ref) has several independent endpoints and so no single observed
scalar; it is not lowered here (censoring a `Parallel` distributes the wrapper
into each branch instead).

# Examples
```@example
using CensoredDistributions, Distributions

seq = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
observed_distribution(seq)
```

# See also
- [`convolve_distributions`](@ref): the chain-step convolution
- [`as_mixture`](@ref): the one_of marginal time-to-resolution
"
observed_distribution(d::UnivariateDistribution) = d

# A one_of node's observed scalar is its marginal time-to-resolution. Returned
# here as a univariate marginal leaf (NOT the node itself) so
# `modifier(observed_distribution(node))` stays the SCALAR combine-then-censor
# lowering even though a bare-node modifier now distributes into the outcome slots
# (#655). The marginal forwards to the node's OWN scalar density (a `Resolve`'s
# mixture, a `Compete`'s racing-hazard `min_k D_k`); a non-terminal one_of (a
# composer-valued outcome) has no scalar marginal and errors there as before.
observed_distribution(d::AbstractOneOf) = _one_of_marginal(d)

function observed_distribution(d::Sequential)
    leaves = _observed_leaves(d.components)
    return length(leaves) == 1 ? only(leaves) :
           convolve_distributions(leaves)
end

# Flatten a composer's components to the univariate leaves whose sum is the
# chain's terminal time. A nested `Sequential` contributes its own steps; a
# nested `Parallel` has no single terminal time, so a chain step that is itself
# a `Parallel` cannot be collapsed and is rejected with a clear message.
function _observed_leaves(components::Tuple)
    leaves = UnivariateDistribution[]
    for c in components
        _append_observed_leaves!(leaves, c)
    end
    return leaves
end

_append_observed_leaves!(leaves, c::UnivariateDistribution) = push!(leaves, c)
function _append_observed_leaves!(leaves, c::Sequential)
    for child in c.components
        _append_observed_leaves!(leaves, child)
    end
    return leaves
end
function _append_observed_leaves!(::Any, ::Parallel)
    throw(ArgumentError(
        "cannot collapse a Sequential chain whose step is a Parallel to a " *
        "single observed time; censor the Parallel's branches instead"))
end

# ---------------------------------------------------------------------------
# Sequential: distribute the shared resolution into the leaf cores.
# ---------------------------------------------------------------------------
#
# A modifier over a `Sequential` keeps the tree shape (tenet 7): it rebuilds the
# SAME chain (same step names) with every leaf reduced to its continuous CORE and
# re-wrapped at the shared resolution. The record scorer then applies the origin
# step's primary once (at the latent origin E_0) and resolves each step's edge,
# so `logpdf(wrapped, rows)` equals the canonical per-leaf construction
# `Sequential(modifier(d_1), ..., modifier(d_k))`. A nested `Sequential` step
# recurses; a nested `Parallel`/`Choose` step distributes through its own method,
# so a chain whose step is a branching node stays scoreable (no collapse error).

# Rebuild a composer node by applying `wrap_leaf` to each LEAF core, preserving
# the tree shape and the step/branch/alternative names. A leaf is first reduced
# to its continuous core (`_marginal_core`) so re-wrapping an already-censored
# leaf re-resolves it rather than stacking censoring layers; a nested composer
# recurses, keeping the structure unchanged.
_distribute_into_leaves(d::UnivariateDistribution, wrap_leaf) = wrap_leaf(_marginal_core(d))
function _distribute_into_leaves(d::Sequential, wrap_leaf)
    return Sequential(
        map(c -> _distribute_into_leaves(c, wrap_leaf), d.components), d.names)
end
function _distribute_into_leaves(d::Parallel, wrap_leaf)
    return Parallel(
        map(c -> _distribute_into_leaves(c, wrap_leaf), d.components), d.names)
end
function _distribute_into_leaves(d::Choose, wrap_leaf)
    return Choose(d.names,
        map(a -> _distribute_into_leaves(a, wrap_leaf), d.alternatives),
        d.selector)
end

# A one_of node (`Resolve` / `Compete`) is a UnivariateDistribution (its scalar
# marginal is the time-to-resolution), but it is a DISJUNCTION over outcome
# delays, the dual of a `Choose`'s alternatives. So a modifier distributes into
# each OUTCOME's delay core, keeping the per-outcome slot layout (the wrapped node
# scores a record per outcome, the loser's slot `missing`), rather than collapsing
# to the scalar marginal via the `UnivariateDistribution` leaf method. This more-
# specific `AbstractOneOf` method shadows that leaf method. A `NoEvent` no-event
# branch carries no delay (its mass is a survival/residual term) and is left
# untouched. `_rebuild_one_of` keeps the node type: a `Resolve` retains its
# `branch_probs`, a `Compete` its derived-hazard form.
function _distribute_into_leaves(d::AbstractOneOf, wrap_leaf)
    wrapped = map(d.delays) do delay
        _is_no_event(delay) ? delay : _distribute_into_leaves(delay, wrap_leaf)
    end
    return _rebuild_one_of(d, wrapped)
end

# Rebuild a one_of node from its wrapped outcome delays, preserving the node type
# and its probability data: a `Resolve` keeps its `branch_probs`, a `Compete`
# derives the winning probabilities from the (wrapped) hazards.
_rebuild_one_of(d::Resolve, delays) = Resolve(d.names, delays, d.branch_probs)
_rebuild_one_of(d::Compete, delays) = Compete(d.names, delays)

@doc "

Primary-event-censor every leaf core of a [`Sequential`](@ref) chain.

The modifier distributes into the chain's leaves, returning a `Sequential` of the
same step names whose every step delay is primary-censored. Tree shape is
unchanged, so `logpdf(wrapped, rows)` scores the per-event record (the canonical
per-leaf construction). The scalar combine-then-censor total stays available
explicitly via `primary_censored(observed_distribution(d), primary_event)`.

See also: [`primary_censored`](@ref), [`observed_distribution`](@ref)
"
function primary_censored(d::Sequential, primary_event::UnivariateDistribution;
        kwargs...)
    return _distribute_into_leaves(
        d, leaf -> primary_censored(leaf, primary_event; kwargs...))
end

function primary_censored(d::Sequential; kwargs...)
    return _distribute_into_leaves(d, leaf -> primary_censored(leaf; kwargs...))
end

@doc "

Interval-censor (day-discretise) every leaf core of a [`Sequential`](@ref) chain.

The interval boundaries distribute into the chain's leaves, returning a
`Sequential` of the same step names whose every step delay is interval-censored;
tree shape is unchanged, so `logpdf(wrapped, rows)` scores the record. The scalar
combine-then-censor total stays available via
`interval_censored(observed_distribution(d), interval)`.

See also: [`interval_censored`](@ref)
"
function interval_censored(d::Sequential, interval)
    return _distribute_into_leaves(d, leaf -> interval_censored(leaf, interval))
end

@doc "

Apply the full primary/truncation/interval pipeline to every leaf core of a
[`Sequential`](@ref) chain.

The pipeline distributes into the chain's leaves, returning a `Sequential` of the
same step names whose every step delay carries the shared resolution; tree shape
is unchanged, so `logpdf(wrapped, rows)` scores the per-event record exactly as
the canonical per-leaf construction
`Sequential(double_interval_censored(d_1; ...), ...)`. The scalar
combine-then-censor chain total stays available explicitly via
`double_interval_censored(observed_distribution(d); ...)`.

See also: [`double_interval_censored`](@ref), [`observed_distribution`](@ref)
"
function double_interval_censored(
        d::Sequential;
        primary_event::UnivariateDistribution = Uniform(0, 1),
        lower::Union{Real, Nothing} = nothing,
        upper::Union{Real, Nothing} = nothing,
        interval::Union{Real, Nothing} = nothing,
        method::Union{AbstractSolverMethod, Nothing} = nothing,
        force_numeric = nothing)
    return _distribute_into_leaves(d,
        leaf -> double_interval_censored(leaf;
            primary_event = primary_event, lower = lower, upper = upper,
            interval = interval, method = method, force_numeric = force_numeric))
end

@doc "

Right-truncate every leaf core of a [`Sequential`](@ref) chain to an observation
`window`.

The right-truncation distributes into the chain's leaves, returning a
`Sequential` of the same step names whose every step delay is right-truncated;
tree shape is unchanged. The scalar combine-then-truncate chain total stays
available via `truncate_to_horizon(observed_distribution(d), window)`.

See also: [`truncate_to_horizon`](@ref), [`observed_distribution`](@ref)
"
function truncate_to_horizon(d::Sequential, window::Real)
    return _distribute_into_leaves(
        d, leaf -> truncate_to_horizon(leaf, window))
end

# ---------------------------------------------------------------------------
# Parallel: distribute the wrapper into every branch's leaf core.
# ---------------------------------------------------------------------------

@doc "

Primary-event-censor every branch core of a [`Parallel`](@ref) independently.

A `Parallel` has several independent endpoints sharing one origin, so censoring
distributes into each branch core, returning a `Parallel` of primary-censored
branches (an already-censored branch is re-resolved, not double-wrapped).

See also: [`primary_censored`](@ref)
"
function primary_censored(d::Parallel, primary_event::UnivariateDistribution;
        kwargs...)
    return _distribute_into_leaves(
        d, leaf -> primary_censored(leaf, primary_event; kwargs...))
end

function primary_censored(d::Parallel; kwargs...)
    return _distribute_into_leaves(d, leaf -> primary_censored(leaf; kwargs...))
end

@doc "

Interval-censor every branch core of a [`Parallel`](@ref) independently.

Distributes the interval boundaries into each branch core, returning a `Parallel`
of interval-censored branches.

See also: [`interval_censored`](@ref)
"
function interval_censored(d::Parallel, interval)
    return _distribute_into_leaves(d, leaf -> interval_censored(leaf, interval))
end

@doc "

Apply the full primary/truncation/interval pipeline to every branch core of a
[`Parallel`](@ref) independently.

Distributes the pipeline into each branch core, returning a `Parallel` of
censored branches (an already-censored branch is re-resolved, not double-wrapped).

See also: [`double_interval_censored`](@ref)
"
function double_interval_censored(d::Parallel; kwargs...)
    return _distribute_into_leaves(
        d, leaf -> double_interval_censored(leaf; kwargs...))
end

@doc "

Right-truncate every branch core of a [`Parallel`](@ref) to an observation
`window` independently.

A `Parallel` has several independent endpoints, so right-truncation distributes
into each branch core, returning a `Parallel` of right-truncated branches (the
same distribute idiom as [`interval_censored`](@ref)`(::Parallel)`).

See also: [`truncate_to_horizon`](@ref)
"
function truncate_to_horizon(d::Parallel, window::Real)
    return _distribute_into_leaves(
        d, leaf -> truncate_to_horizon(leaf, window))
end

# ---------------------------------------------------------------------------
# Choose: distribute the wrapper into every alternative.
# ---------------------------------------------------------------------------
#
# A `Choose` is a data-selected disjunction: each alternative is a full,
# independent sub-distribution with its own endpoint, and a record's `kind`
# selects which alternative scores / `rand`s. A censoring or truncation wrapper
# observes the SELECTED alternative's scalar, so wrapping a `Choose` distributes
# the wrapper into every alternative and returns a `Choose` of wrapped
# alternatives. Scoring stays coherent: `logpdf(wrapped, x; kind)` routes (via
# `_pick`) to the wrapped alternative's own `logpdf`, i.e. the censored /
# truncated score of the selected sub-model, exactly the per-`kind` observation
# model. This mirrors the `Parallel` distribute idiom; the selector and names
# are preserved so the disjunction is unchanged apart from each alternative now
# being observed through the wrapper.

@doc "

Primary-event-censor every alternative of a [`Choose`](@ref) independently.

Each alternative is an independent sub-distribution selected by a record's
`kind`, so censoring distributes into every alternative, returning a `Choose` of
primary-censored alternatives (selector and names preserved). Scoring a record
then routes to its selected alternative's censored score.

See also: [`primary_censored`](@ref), [`Choose`](@ref)
"
function primary_censored(d::Choose, primary_event::UnivariateDistribution;
        kwargs...)
    return _distribute_into_leaves(
        d, leaf -> primary_censored(leaf, primary_event; kwargs...))
end

function primary_censored(d::Choose; kwargs...)
    return _distribute_into_leaves(d, leaf -> primary_censored(leaf; kwargs...))
end

@doc "

Interval-censor every alternative of a [`Choose`](@ref) independently.

Distributes the interval boundaries into each alternative, returning a `Choose`
of interval-censored alternatives (selector and names preserved).

See also: [`interval_censored`](@ref), [`Choose`](@ref)
"
function interval_censored(d::Choose, interval)
    return _distribute_into_leaves(d, leaf -> interval_censored(leaf, interval))
end

@doc "

Apply the full primary/truncation/interval pipeline to every alternative of a
[`Choose`](@ref) independently.

Distributes the pipeline into each alternative, returning a `Choose` of censored
alternatives (selector and names preserved).

See also: [`double_interval_censored`](@ref), [`Choose`](@ref)
"
function double_interval_censored(d::Choose; kwargs...)
    return _distribute_into_leaves(
        d, leaf -> double_interval_censored(leaf; kwargs...))
end

@doc "

Right-truncate every alternative of a [`Choose`](@ref) to an observation
`window` independently.

Distributes the right-truncation into each alternative, returning a `Choose` of
right-truncated alternatives (selector and names preserved).

See also: [`truncate_to_horizon`](@ref), [`Choose`](@ref)
"
function truncate_to_horizon(d::Choose, window::Real)
    return _distribute_into_leaves(
        d, leaf -> truncate_to_horizon(leaf, window))
end

# ---------------------------------------------------------------------------
# Resolve / Compete (one_of): distribute the wrapper into every outcome's delay.
# ---------------------------------------------------------------------------
#
# A one_of node is a DISJUNCTION over outcome delays (the dual of a `Choose`'s
# alternatives): exactly one outcome resolves, the record observes the winner's
# time and leaves the losers `missing`. A censoring/truncation modifier therefore
# distributes into each OUTCOME's delay core and returns the SAME one_of node
# (type and probabilities preserved) with every outcome wrapped, keeping the per-
# outcome slot layout so `logpdf(wrapped, rows)` scores a record per outcome. This
# mirrors the `Choose` distribute idiom; a `NoEvent` no-event branch is left
# untouched (it carries no delay). The SCALAR combine-then-censor marginal time-
# to-resolution stays available explicitly via
# `modifier(observed_distribution(node))`.
#
# A bare-`AbstractOneOf` wrapper method is needed (rather than relying on the
# `UnivariateDistribution` fallback, which a one_of subtypes): without it a
# modifier over a one_of node would hit that leaf method and collapse to the
# scalar marginal, dropping the outcome slots.

@doc "

Primary-event-censor every outcome's delay core of a [`Resolve`](@ref) /
[`Compete`](@ref) one_of node independently.

A one_of node is a disjunction over outcome delays, so censoring distributes into
each outcome, returning the same one_of node (type and probabilities preserved)
with every outcome's delay primary-censored. The per-outcome slot layout is kept,
so the result scores a record per outcome. A `NoEvent` no-event branch is left
untouched. The scalar marginal-time-to-resolution stays available via
`primary_censored(observed_distribution(d), primary_event)`.

See also: [`primary_censored`](@ref), [`Resolve`](@ref), [`Compete`](@ref)
"
function primary_censored(d::AbstractOneOf, primary_event::UnivariateDistribution;
        kwargs...)
    return _distribute_into_leaves(
        d, leaf -> primary_censored(leaf, primary_event; kwargs...))
end

function primary_censored(d::AbstractOneOf; kwargs...)
    return _distribute_into_leaves(d, leaf -> primary_censored(leaf; kwargs...))
end

@doc "

Interval-censor every outcome's delay core of a [`Resolve`](@ref) /
[`Compete`](@ref) one_of node independently.

Distributes the interval boundaries into each outcome, returning the same one_of
node with every outcome's delay interval-censored (per-outcome slots preserved).

See also: [`interval_censored`](@ref), [`Resolve`](@ref), [`Compete`](@ref)
"
# `interval`/`window` are typed `::Real` here (unlike the Sequential/Parallel/
# Choose methods, which need no annotation): a one_of node IS a
# `UnivariateDistribution`, so without the `::Real` match these would be ambiguous
# with the base `interval_censored(::UnivariateDistribution, ::Real)` /
# `truncate_to_horizon(::UnivariateDistribution, ::Real)` leaf methods.
function interval_censored(d::AbstractOneOf, interval::Real)
    return _distribute_into_leaves(d, leaf -> interval_censored(leaf, interval))
end

@doc "

Apply the full primary/truncation/interval pipeline to every outcome's delay core
of a [`Resolve`](@ref) / [`Compete`](@ref) one_of node independently.

Distributes the pipeline into each outcome, returning the same one_of node with
every outcome's delay carrying the shared resolution; the per-outcome slot layout
is unchanged, so `logpdf(wrapped, rows)` scores a record per outcome exactly as
the canonical per-outcome construction
`resolve(:a => (double_interval_censored(d_a; ...), p), ...)`. The scalar marginal
time-to-resolution stays available via
`double_interval_censored(observed_distribution(d); ...)`.

See also: [`double_interval_censored`](@ref), [`Resolve`](@ref), [`Compete`](@ref)
"
function double_interval_censored(
        d::AbstractOneOf;
        primary_event::UnivariateDistribution = Uniform(0, 1),
        lower::Union{Real, Nothing} = nothing,
        upper::Union{Real, Nothing} = nothing,
        interval::Union{Real, Nothing} = nothing,
        method::Union{AbstractSolverMethod, Nothing} = nothing,
        force_numeric = nothing)
    return _distribute_into_leaves(d,
        leaf -> double_interval_censored(leaf;
            primary_event = primary_event, lower = lower, upper = upper,
            interval = interval, method = method, force_numeric = force_numeric))
end

@doc "

Right-truncate every outcome's delay core of a [`Resolve`](@ref) /
[`Compete`](@ref) one_of node to an observation `window` independently.

Distributes the right-truncation into each outcome, returning the same one_of node
with every outcome's delay right-truncated (per-outcome slots preserved).

See also: [`truncate_to_horizon`](@ref), [`Resolve`](@ref), [`Compete`](@ref)
"
function truncate_to_horizon(d::AbstractOneOf, window::Real)
    return _distribute_into_leaves(
        d, leaf -> truncate_to_horizon(leaf, window))
end
