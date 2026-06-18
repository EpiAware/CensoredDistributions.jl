# Shared nesting machinery for the composers, defined once both composer types
# exist so the `Union{Sequential, Parallel}` methods resolve. A realisation of
# any composer is a FLAT vector of leaf values; a nested child contributes its
# own flat sub-vector, so nesting is pure concatenation and that nesting is the
# tree. These helpers do the flat-slice recursion shared by `Sequential` and
# `Parallel`. This layer adds NO censored-internal behaviour.

# A composable child is any univariate distribution (a leaf or a `Resolve`), a
# nested `Sequential` / `Parallel` / `Choose`, or a `latent`-wrapped node. Used to
# validate composer components and `Choose` alternatives. A `Latent` is a
# Multivariate node over `[primary, observed]`, so it is admitted explicitly here
# rather than through the univariate clause; this lets a `Choose` carry a latent
# alternative branch (the index-vs-sourced split's sourced chain).
_is_composable(::UnivariateDistribution) = true
_is_composable(::Union{Sequential, Parallel}) = true
_is_composable(::Choose) = true
_is_composable(::Latent) = true
_is_composable(::Any) = false

# Whether a value is admissible as a one_of OUTCOME delay: a univariate leaf
# (a plain delay, the `NoEvent` marker, or a nested `Resolve`) OR a composer
# SUBTREE (`Sequential` / `Parallel` / `Choose`, the non-terminal branch of #466
# Feature 3). Used by the `one_of` / `Resolve` / `Compete`
# constructors to validate a branch payload without referencing the later-loaded
# composer types in their method signatures.
_is_one_of_branch(::UnivariateDistribution) = true
_is_one_of_branch(::Union{Sequential, Parallel, Choose}) = true
_is_one_of_branch(::Any) = false

# Whether an outcome's payload is itself a composer SUBTREE (a non-terminal
# one_of branch, #466 Feature 3) rather than a leaf delay. A nested `Resolve`
# (univariate but multi-slot) also counts: its event layout spans more than one
# slot. A leaf delay (including the `NoEvent` marker) is terminal. Defined here
# (not in `Resolve.jl`) so `Sequential` / `Parallel` / `Choose` are all loaded.
_is_composer_outcome(::Union{Sequential, Parallel, Choose, AbstractOneOf}) = true
_is_composer_outcome(::UnivariateDistribution) = false

# Whether a one_of node is NON-TERMINAL: any outcome's payload is a composer
# subtree. A non-terminal one_of node is MULTIVARIATE (its outcomes span their
# subtrees' event slots), so its scalar `logpdf` / `mean` / `as_mixture` error and
# its outputs are NamedTuples (#466 Feature 3); an all-leaf node is the unchanged
# univariate (collapsible) terminal node.
_is_nonterminal(c::AbstractOneOf) = any(_is_composer_outcome, c.delays)

# Default positional names for a composer node, used when the front-end (or a
# positional constructor) supplies none. `_default_names(:step, 3)` is
# `(:step_1, :step_2, :step_3)`; the prefix is `:step` for `Sequential` and
# `:branch` for `Parallel`. Built as a typed tuple so the names field stays
# concretely typed.
function _default_names(prefix::Symbol, n::Int)
    return ntuple(i -> Symbol(prefix, :_, i), n)
end

# Coerce a user-supplied names collection (a tuple/vector of Symbols, or
# `nothing` for "use defaults") to a Symbol tuple of the right length. Used by
# the `compose` front-ends so every input format threads names through.
_coerce_names(::Nothing, prefix::Symbol, n::Int) = _default_names(prefix, n)
function _coerce_names(names, ::Symbol, n::Int)
    length(names) == n || throw(ArgumentError(
        "supplied $(length(names)) names for $n components"))
    return Tuple(Symbol(x) for x in names)
end

# --- the public composer-node extension contract ---------------------------
#
# A composer node combines branches into ONE flat event vector. The three
# methods below are the contract a new node implements; they are public (see
# `public.jl`) and documented in `docs/src/developer/extending.md`. They are
# reached by the qualified name (`CensoredDistributions.child_nleaves` etc.), as
# the leaf hooks `free_leaf` / `rewrap_leaf` are. The underscored aliases
# (`_child_nleaves` / `_child_logpdf` / `_child_rand!`) defined alongside each
# are retained for the package's existing internal callers, so dropping the
# underscore is source-compatible.
#
#   - `child_nleaves(node)`: how many flat slots the node occupies.
#   - `child_logpdf(node, x, offset, n)`: the node's contribution to the joint
#     log density, scoring the `n`-wide slice `x[offset + 1 : offset + n]`.
#   - `child_rand!(out, offset, rng, node)`: draw the node into the same slice.
#
# A node walks the flat vector by the SAME offset arithmetic the composer uses:
# `child_nleaves` gives the slice width, `child_logpdf` / `child_rand!` read and
# write `x[offset + 1 : offset + n]`. Composing two nodes is concatenation, so a
# nested node recurses by passing the same `(x, offset, n)` inward.

"""
    child_nleaves(node)

Number of flat event-vector slots a composer node occupies (one per leaf below
it). Part of the public composer-node extension contract, alongside
[`child_logpdf`](@ref) and [`child_rand!`](@ref); see
[Writing a new composer node](@ref new-composer-node). A univariate leaf occupies
one slot; a nested node occupies the sum of its children's widths.

# Arguments
- `node`: the composer node or leaf distribution whose flat slot width is read.

# Examples
```@example
using CensoredDistributions, Distributions

node = compose((onset = Gamma(2.0, 1.0), report = Gamma(1.5, 1.0)))
CensoredDistributions.child_nleaves(node)
```

# See also
- [`child_logpdf`](@ref): score a node's slice of the flat vector.
- [`child_rand!`](@ref): draw a node into its slice of the flat vector.
"""
function child_nleaves end

child_nleaves(::UnivariateDistribution) = 1
child_nleaves(c::Union{Sequential, Parallel}) = length(c)
# A nested `Choose` swaps in ONE alternative of fixed width, so it occupies a
# fixed flat slot only when every alternative has the same leaf count. The
# common width is the nested Choose's leaf count; disagreeing widths cannot
# share one flat slot and error (a `length(::Choose)` has no single answer).
function child_nleaves(c::Choose)
    n = child_nleaves(_flat_select_alternative(c))
    widths = map(child_nleaves, c.alternatives)
    all(==(n), widths) || throw(ArgumentError(
        "a nested Choose needs every alternative to have the same leaf count " *
        "to occupy a fixed flat slot; got $(widths)"))
    return n
end
# A latent alternative scores `[primary, observed]` (two slots); the flat
# value-vector layout collapses it to its marginal leaf count, since a nested
# Choose's flat slot carries observed values, not the latent primary.
child_nleaves(c::Latent) = child_nleaves(c.dist)

# Backward-compatible internal alias: the package's existing callers reach the
# node contract by the underscored name.
const _child_nleaves = child_nleaves

# Total leaf count over a tuple of children. A HEAD/TAIL recursion, NOT
# `sum(_child_nleaves, components)`: `sum(f, ::Tuple)` over a heterogeneous tuple
# is inferred `Any` on the CI compilers (`lts`/`1`) -- it lowers to a generic
# `mapreduce` whose accumulator type the older inference cannot resolve -- which
# poisons every downstream `Vector{...}(undef, _nleaves(...) + 1)` constructor
# (its length argument becomes `Any`, so the constructed array type widens to
# `Any` and the whole sampling/scoring path infers `Any`). Julia 1.12 happens to
# constant-fold the `sum` and so masks the regression locally. The recursion
# below resolves to a concrete `Int` per step on every supported version.
_nleaves(::Tuple{}) = 0
function _nleaves(components::Tuple)
    _child_nleaves(first(components)) + _nleaves(Base.tail(components))
end

# Number of EVENT slots a child contributes to the flat EVENT vector.
# Distinct from `_child_nleaves` (the generic VALUE-vector layout): a `Resolve`
# node contributes ONE value (its marginal time-to-resolution) to the value
# vector but exposes one EVENT slot PER OUTCOME so a record's death/discharge
# columns each land in their own slot and the observed outcome is identified
# positionally (self-dispatch). Every other child contributes the same count
# as `_child_nleaves`, so the value and event layouts coincide for Resolve-free
# trees and `length`/the generic value path are untouched.
_event_child_nleaves(c) = _child_nleaves(c)
# Both one_of nodes (the mixture `Resolve` and the racing-hazard
# `Compete`) expose event slots PER OUTCOME. A LEAF outcome (a plain
# delay) occupies ONE slot; a NON-TERMINAL outcome whose payload is itself a
# composer subtree (`Sequential`/`Parallel`/`Choose`/nested `Resolve`) occupies
# its WHOLE subtree's event-slot width (#466 Feature 3), anchored at the outcome's
# resolution event (shared like a nested-composer origin). The all-leaf fast path
# is exactly `_n_branches(c)` (every outcome contributes one slot), preserving the
# #474 terminal-Resolve layout; a composer outcome instead recurses through
# `_event_child_nleaves`, so its sub-event slots are summed in. Dispatch on the
# shared supertype so the mixture and racing nodes share the layout.
function _event_child_nleaves(c::AbstractOneOf)
    return _one_of_outcome_nleaves(c.delays)
end

# Sum the EVENT-slot width of each one_of outcome: a leaf outcome is one slot,
# a composer outcome is its own `_event_child_nleaves` (its subtree's slots).
# HEAD/TAIL recursion for the same `Any`-inference reason as `_event_nleaves`
# (`sum`/`mapreduce` over a heterogeneous outcome tuple widens to `Any` on the CI
# compilers and poisons the downstream event-vector length).
_one_of_outcome_nleaves(::Tuple{}) = 0
function _one_of_outcome_nleaves(delays::Tuple)
    return _one_of_outcome_slots(first(delays)) +
           _one_of_outcome_nleaves(Base.tail(delays))
end

# Event-slot width of ONE one_of outcome's payload: a leaf delay (including the
# no-event marker) is one slot; a composer payload recurses to its subtree width.
_one_of_outcome_slots(::UnivariateDistribution) = 1
function _one_of_outcome_slots(d::Union{Sequential, Parallel, Choose,
        AbstractOneOf})
    return _event_child_nleaves(d)
end
_event_child_nleaves(c::Union{Sequential, Parallel}) = _event_nleaves(c.components)
# A nested `Choose` occupies its (common) alternative's EVENT-slot width: every
# alternative must expose the same number of event slots to share one flat slot,
# so the chosen alternative for a row lands in the same slice whichever it is.
function _event_child_nleaves(c::Choose)
    n = _event_child_nleaves(_flat_select_alternative(c))
    widths = map(_event_child_nleaves, c.alternatives)
    all(==(n), widths) || throw(ArgumentError(
        "a nested Choose needs every alternative to expose the same number of " *
        "event slots to occupy a fixed flat slot; got $(widths)"))
    return n
end

# Total EVENT-slot count over a tuple of children (the flat event vector minus
# its shared origin). HEAD/TAIL recursion for the same reason as `_nleaves`:
# `sum(_event_child_nleaves, ::Tuple)` infers `Any` on the CI compilers and
# widens the `Vector{Union{Missing, T}}(missing, _event_nleaves(...) + 1)`
# constructor in `_tree_event_vector` to `Any`, breaking `@inferred` on the
# sampling walk on every version except the one that constant-folds it (1.12).
_event_nleaves(::Tuple{}) = 0
function _event_nleaves(components::Tuple)
    _event_child_nleaves(first(components)) +
    _event_nleaves(Base.tail(components))
end

# Sum the per-child log-densities over the matching flat slices of `x`. A leaf
# consumes one scalar; a nested composer consumes a `_child_nleaves`-long slice
# and recurses. The offset walk is pure control flow over the constant index, so
# the differentiated arithmetic sees only concrete values (AD-safe).
function _composite_logpdf(components::Tuple, x::AbstractVector)
    total = zero(eltype(x))
    offset = 0
    @inbounds for c in components
        n = child_nleaves(c)
        total += child_logpdf(c, x, offset, n)
        offset += n
    end
    return total
end

"""
    child_logpdf(node, x, offset, n)

A composer node's contribution to the joint log density, scoring its `n`-wide
slice `x[offset + 1 : offset + n]` of the flat event vector. Part of the public
composer-node extension contract, alongside [`child_nleaves`](@ref) and
[`child_rand!`](@ref); see
[Writing a new composer node](@ref new-composer-node). A univariate leaf scores
the one scalar at its slot; a nested node recurses into its children, passing
each its own offset.

# Arguments
- `node`: the composer node or leaf distribution to score.
- `x`: the flat event vector being scored.
- `offset`: the zero-based start index of this node's slice in `x`.
- `n`: the slice width, `child_nleaves(node)`.

# Examples
```@example
using CensoredDistributions, Distributions

node = compose((onset = Gamma(2.0, 1.0), report = Gamma(1.5, 1.0)))
n = CensoredDistributions.child_nleaves(node)
x = collect(values(rand(node)))
CensoredDistributions.child_logpdf(node, x, 0, n)
```

# See also
- [`child_nleaves`](@ref): the slice width `n` to pass.
- [`child_rand!`](@ref): draw a node into its slice of the flat vector.
"""
function child_logpdf end

child_logpdf(c::UnivariateDistribution, x, offset, ::Int) = logpdf(c, x[offset + 1])
# A nested child scores its own contiguous slice of the value vector; a `@view`
# avoids a copy and differentiates on every supported backend.
function child_logpdf(c::Union{Sequential, Parallel}, x, offset, n::Int)
    logpdf(c, @view x[(offset + 1):(offset + n)])
end
# A nested `Choose` in the data-free flat value-vector path commits to its FIRST
# alternative (a deterministic default so flat `logpdf`/`rand` round-trip); the
# selector-driven choice lives in the row/record path, not the flat path.
function child_logpdf(c::Choose, x, offset, n::Int)
    return child_logpdf(_flat_select_alternative(c), x, offset, n)
end
# A latent alternative on the flat path scores through its marginal node (the
# flat slot carries observed values; the latent primary is integrated out).
function child_logpdf(c::Latent, x, offset, n::Int)
    return child_logpdf(c.dist, x, offset, n)
end

# Backward-compatible internal alias (see `child_nleaves`).
const _child_logpdf = child_logpdf

# The alternative a nested Choose commits to on the data-free path: the FIRST.
# The row/record path overrides this by the row's selector value (`_pick` /
# `_resolve_selects`). This is the SINGLE source of the "Choose routes to its
# first alternative" rule shared by every tree walk -- the flat value path here,
# the event-name walk (`tree_events.jl`), the per-event moment / discretisation /
# sampling walks (`composed_moments.jl` / `censored_rand.jl`), and the AD'd
# scorer (`censored_scoring_tree.jl` / `censored_one_of.jl`). It is a pure
# structural accessor (no leaf values, no closures), so the scorer routing
# through it stays AD-safe (it inlines to the bare `first(c.alternatives)`).
_flat_select_alternative(c::Choose) = first(c.alternatives)

# Concatenate the per-child draws into one flat vector of element type `T`.
function _composite_rand(rng::AbstractRNG, components::Tuple, ::Type{T}) where {T}
    out = Vector{T}(undef, _nleaves(components))
    offset = 0
    @inbounds for c in components
        n = child_nleaves(c)
        child_rand!(out, offset, rng, c)
        offset += n
    end
    return out
end

"""
    child_rand!(out, offset, rng, node)

Draw a composer node in place into its slice `out[offset + 1 : offset + n]` of the
flat output vector, where `n` is [`child_nleaves`](@ref)`(node)`. Returns
`nothing`. Part of the public composer-node extension contract, alongside
[`child_nleaves`](@ref) and [`child_logpdf`](@ref); see
[Writing a new composer node](@ref new-composer-node). A univariate leaf writes
its one slot; a nested node fills its slice by recursing into its children.

# Arguments
- `out`: the flat output vector to write into.
- `offset`: the zero-based start index of this node's slice in `out`.
- `rng`: the random number generator to draw from.
- `node`: the composer node or leaf distribution to draw.

# Examples
```@example
using CensoredDistributions, Distributions, Random

node = compose((onset = Gamma(2.0, 1.0), report = Gamma(1.5, 1.0)))
out = zeros(CensoredDistributions.child_nleaves(node))
CensoredDistributions.child_rand!(out, 0, Random.default_rng(), node)
out
```

# See also
- [`child_nleaves`](@ref): the slice width written.
- [`child_logpdf`](@ref): score a node's slice of the flat vector.
"""
function child_rand! end

function child_rand!(out, offset, rng::AbstractRNG, c::UnivariateDistribution)
    out[offset + 1] = rand(rng, c)
    return nothing
end
function child_rand!(
        out, offset, rng::AbstractRNG, c::Union{Sequential, Parallel})
    # Use the INTERNAL vector-valued realisation (`_composer_rand`), not the
    # public `rand`: the public `rand` labels a top-level multivariate draw as a
    # NamedTuple, but a nested child here is concatenated into the flat value
    # vector by position, so it must stay vector-valued.
    sub = _composer_rand(rng, c)
    @inbounds for k in eachindex(sub)
        out[offset + k] = sub[k]
    end
    return nothing
end
# A nested `Choose` samples its FIRST alternative on the flat path, matching the
# committed alternative the flat `child_logpdf` scores.
function child_rand!(out, offset, rng::AbstractRNG, c::Choose)
    return child_rand!(out, offset, rng, _flat_select_alternative(c))
end
# A latent alternative samples its observed value through its marginal node on
# the flat path (the latent primary is not part of the flat slot).
function child_rand!(out, offset, rng::AbstractRNG, c::Latent)
    return child_rand!(out, offset, rng, c.dist)
end

# Backward-compatible internal alias (see `child_nleaves`).
const _child_rand! = child_rand!

# The recursive indented-tree printing and the `params`/`params_table` traversal
# share the hand-rolled, type-stable helpers defined in `introspection.jl`
# (`_named_children`, `_show_children`, `_node_header`).
