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
# walk forms. For posterior means we read `mean(chain)` once (every parameter
# mean, keyed by `VarName`); for a single draw we index that parameter's column.
#
# Only scalar parameters belong in the lookup. A latent fit also carries
# vector-valued per-record event VarNames (e.g. `rec1.e`) whose draws are
# `Vector`s and whose mean the summary keys element-wise, not under the bare
# name; the template walk never requests them. Filtering on the column element
# type keeps the scalars and skips the vector params, so the mean is safe to
# read.
function _value_lookup(chain, draw)
    vns = [vn for vn in FlexiChains.parameters(chain)
           if eltype(chain[vn]) <: Real]
    if draw === nothing
        means = mean(chain)
        return Dict(string(vn) => means[vn] for vn in vns)
    end
    return Dict(string(vn) => vec(chain[vn])[draw] for vn in vns)
end

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

function chain_to_params(template, chain; prefix::Symbol = :d, draw = nothing)
    lookup = _value_lookup(chain, draw)
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
`update(template, chain_to_params(template, chain))`. By default it applies
posterior means; pass `draw=i` for a single iteration. The `prefix` keyword
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
- `draw`: a single iteration index to read, or `nothing` for posterior means
  (default `nothing`).

# See also
- [`chain_to_params`](@ref): the nested NamedTuple this reads.
- [`update`](@ref): the NamedTuple-keyed reconstruction this delegates to.
"
function update(template, chain::FlexiChains.FlexiChain;
        prefix::Symbol = :d, draw = nothing)
    params = chain_to_params(template, chain; prefix = prefix, draw = draw)
    return update(template, params)
end

end
