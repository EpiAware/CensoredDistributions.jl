module CensoredDistributionsFlexiChainsExt

# `chain_to_params` (declared in `src/turing_models.jl`): read a fitted
# FlexiChains chain into the nested NamedTuple `update` consumes. Loaded only
# when both DynamicPPL and FlexiChains are available.

using CensoredDistributions: CensoredDistributions, Sequential, Parallel,
                             Competing, Select, component_names
import CensoredDistributions: chain_to_params, update
using FlexiChains: FlexiChains
using Statistics: mean

# Read a chain's free parameters into a `String`-keyed lookup, so the tree-walk
# matches each template parameter by its dotted name instead of rebuilding a
# `VarName` optic per parameter. The chain keys ARE the submodel-prefixed
# `VarName`s (e.g. `d.onset_admit.shape`); their `string` is the dotted path the
# walk forms.
#
# The reduction is configurable: `summary` is any `AbstractVector -> scalar`
# (default `mean`, so `median`, `mode`, `x -> quantile(x, 0.9)` all work), applied
# to each parameter's draws. `draw` keeps the single-iteration shortcut (an exact
# index). `draws` selects a SUBSET of iterations to reduce over: a range / index
# vector (positional), or a predicate `i -> Bool` over the iteration index (e.g.
# warmup drop / thinning); `nothing` uses every draw.
#
# Only scalar parameters belong in the lookup. A latent fit also carries
# vector-valued per-record event VarNames (e.g. `rec1.e`) whose draws are
# `Vector`s; the template walk never requests them. Filtering on the column
# element type keeps the scalars and skips the vector params, so the reduction is
# safe to read.
function _value_lookup(chain, draw, draws, summary)
    vns = [vn for vn in FlexiChains.parameters(chain)
           if eltype(chain[vn]) <: Real]
    if draw !== nothing
        return Dict(string(vn) => vec(chain[vn])[draw] for vn in vns)
    end
    sel = _draw_indices(chain, draws)
    return Dict(string(vn) => summary(_select_draws(chain[vn], sel))
    for vn in vns)
end

# The iteration indices a `draws` selector picks out. `nothing` is every
# iteration; a predicate filters the index range; anything else (a range / index
# vector) is taken as the indices directly.
function _draw_indices(chain, draws::Nothing)
    return Colon()
end
function _draw_indices(chain, draws)
    if draws isa Function
        n = length(vec(chain[first(FlexiChains.parameters(chain))]))
        return [i for i in 1:n if draws(i)]
    end
    return collect(draws)
end

_select_draws(col, ::Colon) = vec(col)
_select_draws(col, sel) = vec(col)[sel]

# Look up one parameter by its dotted name (`prefix.path...`); `nothing` if
# absent (e.g. a `Competing`'s branch probabilities kept fixed in the template).
_read_value(lookup, key) = get(lookup, key, nothing)

# Form the dotted name a submodel-sampled parameter carries: the `~`-bound
# `prefix` then the edge path and parameter name (e.g. prefix `:d`, path
# `(:onset_admit, :shape)` -> `"d.onset_admit.shape"`), matching `string(vn)`.
_dotted(prefix::Symbol, path::Tuple) = join(string.((prefix, path...)), ".")

# Build the nested NamedTuple by walking the template, so its key order matches
# `params(template)` (deterministic, not Dict-ordered). Each node reads its
# children/parameters from `lookup` at its running dotted name.

function _node_params(d::Union{Sequential, Parallel}, lookup, prefix, path)
    names = component_names(d)
    vals = map(zip(names, d.components)) do (name, child)
        _node_params(child, lookup, prefix, (path..., name))
    end
    return NamedTuple{names}(Tuple(vals))
end

function _node_params(c::Competing, lookup, prefix, path)
    delays = map(zip(c.names, c.delays)) do (name, delay)
        _node_params(delay, lookup, prefix, (path..., name))
    end
    base = NamedTuple{c.names}(Tuple(delays))
    # Branch probabilities only appear in the chain when sampled (a prior was
    # supplied); otherwise omit so `update` keeps the template's fixed values.
    bp = map(c.names) do name
        _read_value(lookup, _dotted(prefix, (path..., :branch_probs, name)))
    end
    any(x -> x === nothing, bp) && return base
    return merge(base, (; branch_probs = NamedTuple{c.names}(Tuple(bp))))
end

# A `Select` walks its alternatives by name (mirroring the core
# `params`/`update` traversal), so a posterior reads back onto a Select
# template. The nested
# NamedTuple is keyed by the alternative names; a shared-tagged leaf inside an
# alternative reads its values from the top level under its tag (see the leaf
# method), so a tie across alternatives maps to the one chain entry.
function _node_params(d::Select, lookup, prefix, path)
    vals = map(zip(d.names, d.alternatives)) do (name, alt)
        _node_params(alt, lookup, prefix, (path..., name))
    end
    return NamedTuple{d.names}(Tuple(vals))
end

# Leaf: read each free parameter (in `_leaf_param_names` order) from `lookup`.
# A leaf parameter is always sampled, so a missing one signals a chain that does
# not match the template (wrong prefix, or not the chain that produced it);
# error rather than build a NamedTuple with a `nothing` value.
#
# A SHARED-tagged leaf (`shared(:inc, ...)`) is deduped in the chain: it is
# sampled ONCE under its tag (`<prefix>.<tag>.<param>`, e.g. `d.inc.shape`),
# not per occurrence, matching `params_table`'s tag edge and `_collect_shared`.
# So an occurrence reads its values at the bare TAG (ignoring its branch path);
# every occurrence then maps to the one chain entry, and the nested NamedTuple
# keys the group ONCE at the top level under its tag (read back by `update`).
function _node_params(leaf, lookup, prefix, path)
    tag = CensoredDistributions._shared_tag(leaf)
    keypath = tag === nothing ? path : (tag,)
    pnames = CensoredDistributions._leaf_param_names(leaf)
    vals = map(pnames) do p
        key = _dotted(prefix, (keypath..., p))
        v = _read_value(lookup, key)
        v === nothing && throw(ArgumentError(
            "leaf parameter $key not found in chain"))
        v
    end
    return NamedTuple{pnames}(Tuple(vals))
end

# Read every shared group ONCE from the chain into a top-level `tag => values`
# NamedTuple, mirroring how `composed_parameters_model` samples each group under
# its tag (`<prefix>.<tag>.<param>`) and how the core `update` reads a shared
# leaf from the top-level tag entry. `_collect_shared` gives the
# first-occurrence leaf per tag (its inner family fixes the parameter names),
# deduped in pre-order.
function _shared_params(template, lookup, prefix)
    groups = CensoredDistributions._collect_shared(template)
    isempty(groups) && return NamedTuple()
    entries = map(groups) do (tag, leaf)
        tag => _node_params(leaf, lookup, prefix, ())
    end
    return NamedTuple(entries)
end

function chain_to_params(template, chain; prefix::Symbol = :d, draw = nothing,
        draws = nothing, summary = mean)
    lookup = _value_lookup(chain, draw, draws, summary)
    tree = _node_params(template, lookup, prefix, ())
    # A shared-tagged leaf is sampled once under its tag, so add a top-level
    # `tag` entry for each shared group; the core `update` reads each
    # occurrence from it (per-occurrence entries in `tree` are tolerated).
    return merge(tree, _shared_params(template, lookup, prefix))
end

# Update a composed distribution directly from a fitted chain, so docs call
# `update(template, chain)` rather than threading `chain_to_params` by hand.
# Reads the chain into the nested NamedTuple (posterior means, or a single
# `draw`) at the submodel `prefix`, then reconstructs through the core `update`.
@doc "

Update a composed distribution's parameters straight from a fitted chain.

`update(template, chain)` reads `chain` (sampled through
[`composed_parameters_model`](@ref)) into the nested NamedTuple and rebuilds
`template` with those values, so the workflow is one call instead of
`update(template, chain_to_params(template, chain))`. By default it reduces each
parameter's draws with `mean`; pass any `summary` reduction, restrict to a
subset of draws with `draws` (a range / index vector, or a predicate over the
iteration index), or pass `draw=i` for a single iteration. The `prefix` keyword
names the submodel variable the parameters were sampled under (default `:d`).

This method is available only when both `DynamicPPL` and `FlexiChains` are
loaded.

# Arguments
- `template`: the composed distribution that was the `composed_parameters_model`
  template.
- `chain`: the fitted `FlexiChains` chain to read parameter values from.

# Keyword Arguments
- `prefix`: the submodel variable name the parameters were sampled under
  (default `:d`).
- `summary`: the reduction `AbstractVector -> scalar` applied to each
  parameter's draws (default `mean`).
- `draws`: a subset of iterations to reduce over (a range / index vector, or a
  predicate over the iteration index); `nothing` uses every draw.
- `draw`: a single iteration index to read (overrides `summary`/`draws`).

# See also
- [`chain_to_params`](@ref): the nested NamedTuple this reads.
- [`update`](@ref): the NamedTuple-keyed reconstruction this delegates to.
"
function update(template, chain::FlexiChains.FlexiChain;
        prefix::Symbol = :d, draw = nothing, draws = nothing, summary = mean)
    params = chain_to_params(template, chain; prefix = prefix, draw = draw,
        draws = draws, summary = summary)
    return update(template, params)
end

end
