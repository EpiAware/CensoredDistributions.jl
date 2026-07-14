# ============================================================================
# Shared: a name-tagged sub-distribution tied across composed branches
# ============================================================================
#
# `shared(:inc, dist)` tags a leaf with a name so occurrences carrying the same
# tag are one free parameter, not two: the prior/params interface dedups by name
# (inventory once, sample once, place the one value in every occurrence) rather
# than duplicating a parameter that appears in several branches of a tree.
# The wrapper is transparent in the hot path: every `Distributions` method
# delegates to the wrapped leaf, so the hot path is unchanged and AD flows
# through. Only introspection (`params_table`), reconstruction (`update`) and
# the Turing parameter submodel read the tag, deduping occurrences by it.

@doc "

A name-tagged leaf tied across the branches of a composed distribution.

`Shared` wraps a leaf distribution with a `tag` (a `Symbol`) marking it as a
shared parameter group. Two `Shared` leaves carrying the same tag are treated as
the same free parameter by the prior/params interface: [`params_table`](@ref)
lists the group's parameters once (deduped by tag),
[`composed_parameters_model`](@ref) samples the group once and places the sampled
values in every occurrence, and [`update`](@ref) updates all occurrences from one
entry. The wrapper is transparent to scoring and sampling (every distribution
method delegates to the wrapped leaf), so it only changes how parameters are
inventoried, sampled and reconstructed.

# Fields
- `tag`: the shared-parameter group name (`Symbol`).
- `dist`: the wrapped leaf distribution.

# See also
- [`shared`](@ref): constructor over a name and a distribution.
- [`params_table`](@ref), [`update`](@ref): dedup occurrences by tag.
"
struct Shared{D <: UnivariateDistribution} <:
       AbstractModifiedDistribution{Univariate, ValueSupport}
    "The shared-parameter group name (`Symbol`)."
    tag::Symbol
    "The wrapped leaf distribution."
    dist::D
end

@doc "

Tag a leaf distribution as a shared parameter group named `name`.

`shared(name, dist)` marks `dist` as a tied parameter so multiple occurrences of
the same `name` in a composed distribution are handled once by the prior/params
interface (inventoried, sampled and updated as a single free parameter), with the
shared value placed in every occurrence. The result is transparent to scoring and
sampling.

`shared(name, dist)` is the leaf-local spelling of the tie, applied where the leaf
is built. [`tie`](@ref)`(d, paths...; name)` is the tree-level spelling of the
same tie: it walks a composed `d` to the named leaves and wraps each in the exact
`shared(name, leaf)` artefact this produces. Use whichever is convenient; the
tagged occurrences are one free parameter either way.

# Arguments
- `name`: the shared-parameter group name (`Symbol`).
- `dist`: the leaf distribution to tag.

# Examples
```@example
using CensoredDistributions, Distributions

# The same incubation `inc` tied across two branches of a `choose`.
inc = shared(:inc, Gamma(2.0, 1.0))
d = choose(:index => inc,
    :sourced => compose((src = LogNormal(0.5, 0.4), inc = inc)))
event_names(d)
```

# See also
- [`Shared`](@ref): the tagged-leaf type.
- [`tie`](@ref): the tree-level, path-based spelling of the same tie.
"
shared(name::Symbol, dist::UnivariateDistribution) = Shared(name, dist)

# The shared tag of a (possibly censored) leaf, or `nothing` when untagged. The
# tag survives censoring wrappers so `shared(:inc, double_interval_censored(...))`
# and a bare `shared(:inc, Gamma(...))` both report `:inc`.
_shared_tag(leaf) = nothing
_shared_tag(d::Shared) = d.tag
_shared_tag(d::PrimaryCensored) = _shared_tag(d.dist)
_shared_tag(d::IntervalCensored) = _shared_tag(d.dist)
_shared_tag(d::Truncated) = _shared_tag(d.untruncated)

# `Shared` is transparent: every distribution method delegates to the wrapped
# leaf, so the hot path (logpdf/rand/cdf/quantile/...) is unchanged and AD flows
# straight through. Only the introspection/reconstruction layers read the tag.
get_dist(d::Shared) = d.dist
free_leaf(d::Shared) = free_leaf(d.dist)
rewrap_leaf(d::Shared, inner) = Shared(d.tag, rewrap_leaf(d.dist, inner))

# A thin weight under a shared tag (`shared(:inc, thin(d, p))`) surfaces and
# round-trips like any leaf weight: peel the tag to find/set the `ThinOp` factor.
_thin_factor(d::Shared) = _thin_factor(d.dist)
_set_thin_factor(d::Shared, p) = Shared(d.tag, _set_thin_factor(d.dist, p))

# The tag does not change the realisation type, so the element type is the
# wrapped leaf's (keeps a composed tree's `eltype`/`rand` element type correct).
Base.eltype(::Type{<:Shared{D}}) where {D} = eltype(D)
minimum(d::Shared) = minimum(d.dist)
maximum(d::Shared) = maximum(d.dist)
insupport(d::Shared, x::Real) = insupport(d.dist, x)
params(d::Shared) = params(d.dist)

logpdf(d::Shared, x::Real) = logpdf(d.dist, x)
pdf(d::Shared, x::Real) = pdf(d.dist, x)
cdf(d::Shared, x::Real) = cdf(d.dist, x)
logcdf(d::Shared, x::Real) = logcdf(d.dist, x)
ccdf(d::Shared, x::Real) = ccdf(d.dist, x)
logccdf(d::Shared, x::Real) = logccdf(d.dist, x)
quantile(d::Shared, p::Real) = quantile(d.dist, p)
Base.rand(rng::AbstractRNG, d::Shared) = rand(rng, d.dist)

@doc "

Print a [`Shared`](@ref) tagged leaf as its tag and wrapped distribution.

See also: [`shared`](@ref)
"
function Base.show(io::IO, d::Shared)
    print(io, "shared(", repr(d.tag), ", ", d.dist, ")")
    return nothing
end

# --- shared-tag collection (for dedup in params/sampling) -------------------

# Collect the first-occurrence leaf per shared tag in pre-order, as a
# `tag => leaf` ordered pairs vector. The first occurrence defines the tag's free
# parameters (its inner family) for the prior table and the sampling submodel;
# later occurrences reuse the one sampled value. Used by `composed_parameters_model`
# to sample each shared group once.
function _collect_shared(d)
    acc = Pair{Symbol, Any}[]
    seen = Set{Symbol}()
    _collect_shared!(acc, seen, d)
    return acc
end

function _collect_shared!(acc, seen, d::AbstractMultiChild)
    for c in d.components
        _collect_shared!(acc, seen, c)
    end
    return nothing
end
function _collect_shared!(acc, seen, d::Choose)
    for a in d.alternatives
        _collect_shared!(acc, seen, a)
    end
    return nothing
end
function _collect_shared!(acc, seen, c::AbstractOneOf)
    for g in c.delays
        _collect_shared!(acc, seen, g)
    end
    return nothing
end
function _collect_shared!(acc, seen, leaf)
    tag = _shared_tag(leaf)
    (tag === nothing || tag in seen) && return nothing
    push!(seen, tag)
    push!(acc, tag => leaf)
    return nothing
end

# --- tie: tree-level, path-based shared-leaf grouping -----------------------
#
# `shared(:tag, dist)` tags a leaf locally, where the leaf is built. `tie` is the
# same tie done at the tree level: given a composed `d` and the paths of two or
# more leaves, it walks to each named leaf and wraps it in `Shared(name, leaf)`,
# producing the exact artefact a hand-written `shared(name, dist)` would. Every
# tag consumer (`params_table`, `build_priors`, `update`,
# `composed_parameters_model`, the compute-reuse) reads the tag, not how it was
# placed, so a `tie`d tree and a hand-`shared`d tree are identical. The walk
# reuses `_edit_at` (the `update` path machinery); paths take the
# same forms `event`/`update` accept (a bare `Symbol`, a dotted-path `Symbol`
# like `:"sourced.inc"`, or a tuple of edge names).

# Normalise an `event`/`update` path form to a tuple of edge-name steps:
# a tuple stays as-is, a dotted `Symbol` (`:a.b`) splits, a bare `Symbol` is one
# step. Mirrors how `event(d, name::Symbol)` accepts a path.
_tie_path(p::Tuple) = p
_tie_path(p::Symbol) = _split_edge(p)

# True for the composer (non-leaf) nodes a path can run through; a path that
# resolves to one of these is pointing at a subtree, not a tieable leaf.
_is_composer_node(::AbstractComposedDistribution) = true
_is_composer_node(::Any) = false

# The (family, param-names) signature a tie groups by: tied leaves become one
# free parameter, so they must share an inner free-delay family and parameter
# structure. Uses the same `free_leaf`/`_leaf_param_names` the params interface
# inventories with, so "compatible" means "the params table would treat them
# alike".
function _tie_signature(leaf)
    inner = free_leaf(leaf)
    return (Base.typename(typeof(inner)).wrapper, _leaf_param_names(leaf))
end

@doc "

Tie leaves at named paths of a composed distribution into one shared group.

`tie(d, paths...; name)` walks the composed distribution `d` to each leaf named
by `paths` and wraps it in a [`Shared`](@ref) group tagged `name`, returning the
rebuilt composed distribution. This is the tree-level, path-based spelling of
[`shared`](@ref): `tie(d, p1, p2; name = :inc)` produces the exact same artefact
as building `d` with `shared(:inc, leaf)` at each of those leaves, so every tag
consumer ([`params_table`](@ref), [`build_priors`](@ref), [`update`](@ref),
[`composed_parameters_model`](@ref)) inventories, samples and updates the tied
leaves as a single free parameter.

Each `path` takes the same forms [`event`](@ref) and [`update`](@ref) accept: a
bare `Symbol` direct child, a dotted-path `Symbol` (`:\"sourced.inc\"`, as in
[`params_table`](@ref)'s `edge` column), or a tuple of edge names from the root.
Every path must resolve to a leaf (not a composer subtree), and the tied leaves
must be parameter-compatible (same inner family and parameter structure), since
they become one group.

# Arguments
- `d`: the composed distribution to tie leaves in.
- `paths`: one or more leaf paths to tie together.

# Keyword Arguments
- `name`: the shared-parameter group name (`Symbol`, required).

# Examples
```@example
using CensoredDistributions, Distributions

d = choose(:index => compose((inc = Gamma(2.0, 1.0),)),
    :sourced => compose((src = LogNormal(0.5, 0.4), inc = Gamma(2.0, 1.0))))
tied = tie(d, (:index, :inc), (:sourced, :inc); name = :inc)
params_table(tied)
```

# See also
- [`shared`](@ref): the leaf-local spelling of the same tie.
- [`event`](@ref), [`update`](@ref): share the path forms `tie` accepts.
"
function tie(d::AbstractComposedDistribution,
        paths...; name::Symbol)
    isempty(paths) && throw(ArgumentError(
        "tie needs at least one path to a leaf"))
    norm = map(_tie_path, paths)
    sig = nothing
    out = d
    for (raw, path) in zip(paths, norm)
        isempty(path) && throw(ArgumentError(
            "tie path $(repr(raw)) is empty; a path must name a leaf"))
        # Resolve the leaf first so a bad path or a subtree errors loudly,
        # before any rebuild. `event` descends the same name path forms.
        leaf = event(d, path...)
        _is_composer_node(leaf) && throw(ArgumentError(
            "tie path $(repr(raw)) points at a composer subtree " *
            "($(nameof(typeof(leaf)))), not a leaf; tie groups leaves"))
        leafsig = _tie_signature(leaf)
        if sig === nothing
            sig = leafsig
        elseif leafsig != sig
            throw(ArgumentError(
                "tie(:$name): leaf at $(repr(raw)) is not parameter-" *
                "compatible with the others (family/params $(leafsig) " *
                "vs $(sig)); tied leaves become one parameter group"))
        end
        out = _edit_at(out, path, leaf -> shared(name, leaf))
    end
    return out
end
