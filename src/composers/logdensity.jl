# PPL-neutral flat-vector <-> nested-NamedTuple codec and the assembled
# `ComposedLogDensity` spec. The codec is the one genuinely-new core piece for
# the LogDensityProblems layer (EpiAware/CensoredDistributions.jl#734): three of
# the four LDP ingredients (`dimension`, distribution reconstruction, prior and
# data log-density) already exist Turing-free; only the flat <-> nested bridge
# is new. The library-specific glue (LogDensityProblems / DensityInterface /
# Bijectors) stays in weakdep extensions; the codec and spec are core.

# --- flat <-> nested codec --------------------------------------------------
#
# Ordering is fixed by `params_table`'s pre-order row walk: row `i` of the table
# is flat index `i`. The nested NamedTuple is keyed exactly as `build_priors` /
# `update` expect (a `(:edge, :param)` row nests at `_edge_path(edge)` then
# `param`), so a flat draw round-trips to a named, `update`-able NamedTuple and
# back without re-deriving any structure.

# The flat layout: a vector of `(path, param)` keys, one per table row, in row
# order. `path` is the `_edge_path` tuple of the row's edge; `param` the leaf
# key. This list IS the bijection between flat index and named parameter.
function _flat_layout(table)
    edges = Tables.getcolumn(table, :edge)
    params_col = Tables.getcolumn(table, :param)
    return [(_edge_path(edges[i]), params_col[i]) for i in eachindex(edges)]
end

@doc "

The flat parameter dimension of a composed distribution.

`flat_dimension(d)` is the number of scalar free parameters, i.e. the row count
of [`params_table`](@ref)`(d)`. It is the length of the flat vector
[`flatten`](@ref) produces and [`unflatten`](@ref) consumes, and the
`LogDensityProblems.dimension` of the assembled problem.

# Arguments
- `d`: a composed distribution (or bare leaf).
"
flat_dimension(d) = length(Tables.getcolumn(params_table(d), :edge))

# Read the value at `(path..., param)` of a nested NamedTuple.
function _read_path(nt::NamedTuple, path::Tuple, param::Symbol)
    node = nt
    for k in path
        node = getproperty(node, k)
    end
    return getproperty(node, param)
end

@doc "

Flatten a nested parameter `NamedTuple` to a flat vector in table-row order.

`flatten(d, nt)` reads `nt` (keyed like [`build_priors`](@ref)`(d)` /
[`params`](@ref)`(d)`, the shape [`update`](@ref) consumes) at each
[`params_table`](@ref) row and returns the values as a `Vector`, ordered by the
table's pre-order row walk. It is the inverse of [`unflatten`](@ref):
`flatten(d, unflatten(d, x)) == x` and `unflatten(d, flatten(d, nt))` rebuilds
`nt` for the parameters the table inventories.

# Arguments
- `d`: the composed distribution (or bare leaf) whose table fixes the order.
- `nt`: a nested parameter `NamedTuple` keyed like `params(d)`.

# See also
- [`unflatten`](@ref): the inverse, flat vector -> nested NamedTuple.
- [`flat_dimension`](@ref): the flat length.
"
function flatten(d, nt::NamedTuple)
    layout = _flat_layout(params_table(d))
    return [_read_path(nt, path, param) for (path, param) in layout]
end

@doc "

Rebuild a nested parameter `NamedTuple` from a flat vector in table-row order.

`unflatten(d, x)` maps the flat vector `x` (laid out by [`params_table`](@ref)'s
row walk, e.g. a draw from a sampler) back to the nested `NamedTuple`
[`update`](@ref) consumes, keyed like [`build_priors`](@ref)`(d)`. It is the
inverse of [`flatten`](@ref). A shared-tagged leaf nests once under its tag and
a [`thin`](@ref) weight under `<edge>.thin`, exactly as the table inventories
them, so `update(d, unflatten(d, x))` reconstructs the distribution.

# Arguments
- `d`: the composed distribution (or bare leaf) whose table fixes the layout.
- `x`: a flat vector of length [`flat_dimension`](@ref)`(d)`.

# See also
- [`flatten`](@ref): the inverse, nested NamedTuple -> flat vector.
- [`update`](@ref): rebuild the distribution from the result.
"
function unflatten(d, x::AbstractVector)
    layout = _flat_layout(params_table(d))
    length(x) == length(layout) || throw(DimensionMismatch(
        "flat vector has length $(length(x)) but $d has " *
        "$(length(layout)) free parameters"))
    tree = Dict{Symbol, Any}()
    for (i, (path, param)) in enumerate(layout)
        _nest_insert!(tree, path, param, x[i])
    end
    return _freeze_tree(tree)
end

# --- assembled log-density spec ---------------------------------------------

@doc "

A PPL-neutral log-density over a composed distribution's flat parameters.

`ComposedLogDensity` carries everything needed to evaluate the (unnormalised)
log-posterior of a composed distribution over a flat parameter vector, with no
DynamicPPL/Turing dependency: the template `dist`, the per-parameter `priors`
(a nested `NamedTuple` from [`build_priors`](@ref)), the observed `data`, and a
`loglik` reducer scoring `data` against the reconstructed distribution. Build it
with [`as_logdensity`](@ref); evaluate it on a flat vector with
[`logdensity`](@ref).

It is the spec the weakdep extensions wrap: `LogDensityProblemsExt` makes it a
`LogDensityProblems` problem (sampleable by AdvancedHMC / DynamicHMC /
Pathfinder), `BijectorsExt` derives the constrained<->unconstrained transform
from `priors`, and `DensityInterfaceExt` marks it a density. The flat layout is
[`params_table`](@ref)`(dist)`'s row order throughout, so a posterior draw maps
back to named, constrained parameters interchangeable with the Turing path.

# Fields
- `dist`: the template composed distribution (the structure to reconstruct).
- `priors`: nested prior `NamedTuple` keyed like [`build_priors`](@ref)`(dist)`.
- `data`: the observed records scored by `loglik`.
- `loglik`: a reducer `(d, data) -> Real` (default sums `logpdf(d, record)`).

# See also
- [`as_logdensity`](@ref): the assembler.
- [`logdensity`](@ref): evaluate on a flat (constrained) vector.
- [`flatten`](@ref), [`unflatten`](@ref): the flat <-> nested codec.
"
struct ComposedLogDensity{D, P, T, L}
    dist::D
    priors::P
    data::T
    loglik::L
end

# Default likelihood: sum `logpdf(d, record)` over the observed records, the
# same contribution as the DynamicPPL path's per-record `@addlogprob! logpdf`.
_default_loglik(d, data) = sum(record -> logpdf(d, record), data)

@doc "

Assemble a [`ComposedLogDensity`](@ref) from a composed distribution and data.

`as_logdensity(dist, priors, data; loglik)` packages the template `dist`, the
per-parameter `priors` (a nested `NamedTuple`, usually from
[`build_priors`](@ref)`(dist)`) and the observed `data` into the PPL-neutral
log-density spec. The result evaluates the (unnormalised) log-posterior over the
flat parameter vector via [`logdensity`](@ref), and is the object the weakdep
extensions turn into a `LogDensityProblems` problem (sampleable without Turing),
a `DensityInterface` density, and a Bijectors transform.

`priors` defaults to [`build_priors`](@ref)`(dist)` (support-derived defaults),
so the one-argument-data form needs only `dist` and `data`. `loglik` defaults to
summing `logpdf(dist, record)` over `data`; pass a custom reducer (e.g. one over
[`event_logpdf`](@ref) / [`batched_event_logpdf`](@ref) with a per-record
horizon) for record-aware scoring.

# Arguments
- `dist`: the template composed distribution (or bare leaf).
- `priors`: nested prior `NamedTuple` keyed like [`build_priors`](@ref)`(dist)`
  (default: [`build_priors`](@ref)`(dist)`).
- `data`: the observed records.

# Keyword Arguments
- `loglik`: a reducer `(d, data) -> Real` scoring `data` against the
  reconstructed distribution (default: sum of `logpdf(d, record)`).

# Examples
```@example
using CensoredDistributions, Distributions

using Tables

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
data = [[0.5, 2.0], [1.0, 3.0]]
prob = as_logdensity(tree, build_priors(tree), data)
# The table's `value` column is the flat layout; score at those values.
x = collect(Tables.getcolumn(params_table(tree), :value))
CensoredDistributions.logdensity(prob, x)
```

# See also
- [`logdensity`](@ref): evaluate the assembled spec on a flat vector.
- [`flatten`](@ref), [`unflatten`](@ref): the flat <-> nested codec.
- [`build_priors`](@ref): assemble `priors` from the tree.
"
function as_logdensity(dist, priors, data; loglik = _default_loglik)
    return ComposedLogDensity(dist, priors, data, loglik)
end

function as_logdensity(dist, data; loglik = _default_loglik)
    return as_logdensity(dist, build_priors(dist), data; loglik = loglik)
end

@doc "

Evaluate a [`ComposedLogDensity`](@ref) on a flat (constrained) vector.

`logdensity(prob, x)` is the (unnormalised) log-posterior at the flat vector `x`
(laid out by [`params_table`](@ref)`(prob.dist)`, the constrained parameter
scale): the sum of the prior log-densities plus the data log-likelihood of the
distribution reconstructed at those parameters,
`sum(logpdf, priors) + loglik(update(dist, unflatten(dist, x)), data)`.

This is the constrained-scale density. The unconstrained log-density an HMC
sampler needs (transform + log-Jacobian) is added by `BijectorsExt` /
`LogDensityProblemsExt`; this core method is PPL- and transform-free.

# Arguments
- `prob`: the assembled [`ComposedLogDensity`](@ref).
- `x`: a flat constrained parameter vector of length [`flat_dimension`](@ref).

# See also
- [`as_logdensity`](@ref): assemble `prob`.
- [`flatten`](@ref), [`unflatten`](@ref): the flat <-> nested codec.
"
function logdensity(prob::ComposedLogDensity, x::AbstractVector)
    nt = unflatten(prob.dist, x)
    lp = _prior_logpdf(prob.priors, prob.dist, x)
    d = update(prob.dist, nt)
    return lp + prob.loglik(d, prob.data)
end

# Prior log-density of a flat vector: sum each row's prior `logpdf` at the row's
# value. The priors are flattened in the SAME table-row order as `x`, so the two
# align index-for-index without re-walking the tree per row.
function _prior_logpdf(priors, dist, x::AbstractVector)
    flat_priors = flatten(dist, priors)
    return sum(i -> logpdf(flat_priors[i], x[i]), eachindex(x))
end

# The flat priors in table-row order: the per-row constraint registry the
# transform layer reads (`bijector(prior)` per row). A row-aligned vector, so
# `BijectorsExt` builds the whole flat transform from it.
flat_priors(prob::ComposedLogDensity) = flatten(prob.dist, prob.priors)

@doc "

Map an unconstrained vector to the constrained scale and its log-Jacobian.

`to_constrained(prob, z)` returns `(x, logjac)`, the constrained flat parameters
`x` of the unconstrained vector `z` and the log-determinant Jacobian of that
inverse transform, derived from the per-row priors of `prob` (each row's
`bijector(prior)`, **not** the table's `support` column). The unconstrained
log-density a sampler needs is `logdensity(prob, x) + logjac`.

This has no method until `Bijectors` is loaded; the prior-driven transform lives
in the `BijectorsExt` extension so the core stays free of `Bijectors`.

# Arguments
- `prob`: the assembled [`ComposedLogDensity`](@ref).
- `z`: an unconstrained flat vector of length [`flat_dimension`](@ref).
"
function to_constrained end
