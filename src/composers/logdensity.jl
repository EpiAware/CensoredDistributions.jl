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

# Examples
```@example
using CensoredDistributions, Distributions

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
# Public but not exported; reach it by the qualified name.
CensoredDistributions.flat_dimension(tree)
```

# See also
- [`flatten`](@ref), [`unflatten`](@ref): the flat <-> nested codec.
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

# Examples
```@example
using CensoredDistributions, Distributions, Tables

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
x = collect(Tables.getcolumn(params_table(tree), :value))
# Public but not exported; reach the codec by the qualified name.
nt = CensoredDistributions.unflatten(tree, x)
CensoredDistributions.flatten(tree, nt) == x
```

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

# Examples
```@example
using CensoredDistributions, Distributions, Tables

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
x = collect(Tables.getcolumn(params_table(tree), :value))
# Public but not exported; reach it by the qualified name.
update(tree, CensoredDistributions.unflatten(tree, x)) == tree
```

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

# --- fix / estimate plan: exclude fixed params, build fixed subtrees once ----
#
# A parameter slot in `priors` is ESTIMATED when it holds a `Distribution` and
# FIXED when it holds a plain constant (the `_is_sampled_prior` classification
# the Turing path already uses, so the two backends agree on free-vs-fixed). The
# assembled problem's FREE vector covers only the estimated slots: a fixed param
# is EXCLUDED from `flatten`/`unflatten`/`dimension`/the transform and baked
# into the reconstruction. A subtree whose every slot is fixed has a CONSTANT
# concrete distribution, so it is built ONCE here (`fixed_template`) and reused
# on every `logdensity` evaluation rather than rebuilt (the performance point).

# Whether every parameter slot under a prior (sub)tree is fixed: a leaf slot is
# fixed when its entry is not a sampled distribution; a nested prior NamedTuple
# is fully fixed when all of its entries are.
_all_fixed(p) = !_is_sampled_prior(p)
_all_fixed(nt::NamedTuple) = all(_all_fixed, nt)

# The precomputed fix/estimate plan for a `ComposedLogDensity`, built ONCE at
# construction. `fixed_template` is the template with the fixed constants
# substituted (its fully-fixed subtrees are the concrete distributions reused
# every evaluation); `layout`/`est`/`fixed_vals` are the row-aligned codec
# (which table rows are estimated, the constant for each fixed row);
# `free_priors` are the estimated rows' priors in order (the transform list).
struct _FixPlan{FT, FV}
    fixed_template::FT
    layout::Vector{Tuple{Tuple, Symbol}}
    est::Vector{Bool}
    fixed_vals::Vector{Any}
    free_priors::FV
    has_fixed::Bool
    has_shared::Bool
end

# Build the plan from a template and its priors: classify each table row as
# estimated/fixed, collect the estimated priors, and build `fixed_template` ONCE
# (the fixed constants substituted, estimated slots left at the table default
# and overwritten per evaluation).
function _fix_plan(dist, priors)
    table = params_table(dist)
    layout = Vector{Tuple{Tuple, Symbol}}(_flat_layout(table))
    flat_p = flatten(dist, priors)
    est = Bool[_is_sampled_prior(p) for p in flat_p]
    free_priors = [flat_p[i] for i in eachindex(flat_p) if est[i]]
    values = collect(Tables.getcolumn(table, :value))
    fixed_vals = Vector{Any}(undef, length(layout))
    filled = Dict{Symbol, Any}()
    for i in eachindex(layout)
        (path, param) = layout[i]
        v = est[i] ? values[i] : flat_p[i]
        fixed_vals[i] = v
        _nest_insert!(filled, path, param, v)
    end
    has_fixed = any(!, est)
    has_shared = !isempty(_collect_shared(dist))
    fixed_template = update(dist, _freeze_tree(filled))
    return _FixPlan(fixed_template, layout, est, fixed_vals,
        free_priors, has_fixed, has_shared)
end

@doc "

The number of FREE (estimated) parameters of an assembled log-density.

`free_dimension(prob)` is the length of the flat vector [`logdensity`](@ref) and
the prior-driven transform consume: the count of [`params_table`](@ref) rows
whose prior is a distribution to estimate, EXCLUDING any row pinned to a
constant (a fixed parameter). It is the `LogDensityProblems.dimension` of the
problem and equals [`flat_dimension`](@ref)`(prob.dist)` only when nothing is
fixed. Fix a parameter by placing a plain value (rather than a distribution) in
its prior slot, e.g. via [`build_priors`](@ref)'s `fix` keyword.

# Arguments
- `prob`: an assembled [`ComposedLogDensity`](@ref).

# Examples
```@example
using CensoredDistributions, Distributions

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
prob = CensoredDistributions.as_logdensity(
    tree, build_priors(tree), [[0.5, 2.0]])
# Every parameter is estimated here, so this equals `flat_dimension(tree)`.
# Public but not exported; reach it by the qualified name.
CensoredDistributions.free_dimension(prob)
```

# See also
- [`flat_dimension`](@ref): the full row count (fixed included).
- [`as_logdensity`](@ref): assemble `prob`; [`build_priors`](@ref): fix params.
"
free_dimension(prob) = count(prob.free.est)

# The full nested parameter NamedTuple from the FREE vector `x`: estimated slots
# take the next free value, fixed slots take their pinned constant, so the
# result covers every table row (the shape `update` consumes).
function _full_nt(prob, x::AbstractVector)
    plan = prob.free
    n_free = count(plan.est)
    length(x) == n_free || throw(DimensionMismatch(
        "free vector has length $(length(x)) but $(prob.dist) has " *
        "$n_free estimated parameters"))
    tree = Dict{Symbol, Any}()
    j = 0
    for i in eachindex(plan.layout)
        (path, param) = plan.layout[i]
        v = plan.est[i] ? (j += 1; x[j]) : plan.fixed_vals[i]
        _nest_insert!(tree, path, param, v)
    end
    return _freeze_tree(tree)
end

# Reconstruct `node` from the full values `nt`, REUSING the prebuilt `fixed`
# subtree wherever the prior subtree is fully fixed (built once, never rebuilt).
# A partially-fixed `Sequential`/`Parallel` recurses so a deeper fully-fixed
# subtree is still reused; any other partially-fixed node is rebuilt whole from
# its full values (the fixed constants are already filled into `nt`).
function _graft(node, priors, fixed, nt)
    _all_fixed(priors) ? fixed : _graft_node(node, priors, fixed, nt)
end

function _graft_node(node::AbstractMultiChild, priors, fixed, nt)
    names = component_names(node)
    parts = ntuple(length(names)) do i
        nm = names[i]
        _graft(node.components[i], getproperty(priors, nm),
            fixed.components[i], getproperty(nt, nm))
    end
    return _rebuild(node, parts)
end

_graft_node(node, priors, fixed, nt) = _update(node, nt, nt)

# Reconstruct the scored distribution from the FREE vector `x`. With no fixed
# params (or any shared-tagged leaf, whose tie is handled by the whole-tree
# `update`) this is the plain `update`; otherwise fully-fixed subtrees are
# reused from the prebuilt `fixed_template` via `_graft`.
function _reconstruct(prob, x::AbstractVector)
    nt = _full_nt(prob, x)
    plan = prob.free
    (plan.has_fixed && !plan.has_shared) || return update(prob.dist, nt)
    return _graft(prob.dist, prob.priors, plan.fixed_template, nt)
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

This layer is an ORTHOGONAL alternative to the DynamicPPL
[`composed_parameters_model`](@ref) route, not a wrapper or replacement: the two
never interact at runtime, but evaluate the same log-posterior over the same
shared core ([`build_priors`](@ref), [`update`](@ref), [`logpdf`](@ref)). Pick
this route to sample a composed model with a non-Turing sampler.

# Fields
- `dist`: the template composed distribution (the structure to reconstruct).
- `priors`: nested prior `NamedTuple` keyed like [`build_priors`](@ref)`(dist)`.
- `data`: the observed records scored by `loglik`.
- `loglik`: a reducer `(d, data) -> Real` (default sums `logpdf(d, record)`).
- `free`: the precomputed fix/estimate plan, built once at construction. It
  records which [`params_table`](@ref) rows are estimated versus fixed, the
  estimated rows' priors in order, and the prebuilt fully-fixed subtrees reused
  on every evaluation. Its estimated-row count is [`free_dimension`](@ref)`(d)`.

# See also
- [`as_logdensity`](@ref): the assembler.
- [`logdensity`](@ref): evaluate on a flat (constrained) vector.
- [`flatten`](@ref), [`unflatten`](@ref): the flat <-> nested codec.
- [`composed_parameters_model`](@ref): the orthogonal DynamicPPL/Turing route.
"
struct ComposedLogDensity{D, P, T, L, F}
    dist::D
    priors::P
    data::T
    loglik::L
    free::F
end

# Outer constructor: precompute the fix/estimate plan ONCE here (classifying
# free-vs-fixed slots and building every fully-fixed subtree's concrete
# distribution), so each `logdensity` evaluation reuses it, not rebuilds it.
function ComposedLogDensity(dist, priors, data, loglik)
    return ComposedLogDensity(dist, priors, data, loglik,
        _fix_plan(dist, priors))
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
# Public but not exported; reach the layer by the qualified name.
prob = CensoredDistributions.as_logdensity(tree, build_priors(tree), data)
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

`logdensity(prob, x)` is the (unnormalised) log-posterior at the flat FREE
vector `x` (the estimated parameters in [`params_table`](@ref)`(prob.dist)` row
order, EXCLUDING any fixed parameter, on the constrained scale): the sum of the
estimated priors' log-densities plus the data log-likelihood of the
distribution reconstructed there. `x` is [`free_dimension`](@ref)`(prob)` long
(the full row count only when nothing is fixed). Any parameter pinned to a
constant in `priors` (see [`build_priors`](@ref)'s `fix` keyword) is held at its
value, contributes no prior term, and is omitted from `x`. A subtree whose
parameters are ALL fixed is built once when `prob` is assembled and reused on
every evaluation rather than rebuilt.

This is the constrained-scale density. The unconstrained log-density an HMC
sampler needs (transform + log-Jacobian) is added by `BijectorsExt` /
`LogDensityProblemsExt`; this core method is PPL- and transform-free.

# Arguments
- `prob`: the assembled [`ComposedLogDensity`](@ref).
- `x`: a flat constrained parameter vector of length [`flat_dimension`](@ref).

# Examples
```@example
using CensoredDistributions, Distributions, Tables

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
prob = CensoredDistributions.as_logdensity(
    tree, build_priors(tree), [[0.5, 2.0], [1.0, 3.0]])
x = collect(Tables.getcolumn(params_table(tree), :value))
CensoredDistributions.logdensity(prob, x)
```

# See also
- [`as_logdensity`](@ref): assemble `prob`.
- [`flatten`](@ref), [`unflatten`](@ref): the flat <-> nested codec.
"
function logdensity(prob::ComposedLogDensity, x::AbstractVector)
    lp = _prior_logpdf_free(prob, x)
    d = _reconstruct(prob, x)
    return lp + prob.loglik(d, prob.data)
end

# Prior log-density of the FREE vector: sum each ESTIMATED row's prior `logpdf`
# at its value. Fixed params carry no prior term (they are not sampled), so the
# estimated priors align index-for-index with `x` (both fixed-excluded). An
# all-fixed model has no free term, returning a typed zero.
function _prior_logpdf_free(prob::ComposedLogDensity, x::AbstractVector)
    fp = prob.free.free_priors
    isempty(x) && return 0.0
    return sum(i -> logpdf(fp[i], x[i]), eachindex(x))
end

# The estimated priors in row order: the per-row constraint registry the
# transform layer reads (`bijector(prior)` per FREE parameter). Fixed params are
# excluded, so `BijectorsExt` builds the transform over the free vector only.
flat_priors(prob::ComposedLogDensity) = prob.free.free_priors

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

# Examples
```@example
using CensoredDistributions, Distributions, Tables, Bijectors

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
prob = CensoredDistributions.as_logdensity(
    tree, build_priors(tree), [[0.5, 2.0]])
# An unconstrained vector maps back to constrained parameters + log-Jacobian.
z = zeros(CensoredDistributions.flat_dimension(tree))
x, logjac = CensoredDistributions.to_constrained(prob, z)
x
```

# See also
- [`as_logdensity`](@ref): assemble `prob`.
- [`logdensity`](@ref): the constrained-scale density this transform feeds.
"
function to_constrained end
