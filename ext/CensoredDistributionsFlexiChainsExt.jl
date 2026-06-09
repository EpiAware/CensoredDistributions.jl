module CensoredDistributionsFlexiChainsExt

# `chain_to_params` (declared in `src/turing_models.jl`): read a fitted
# FlexiChains chain into the nested NamedTuple `update` consumes. Loaded only
# when both DynamicPPL and FlexiChains are available.

using CensoredDistributions: CensoredDistributions, Sequential, Parallel,
                             Competing, Select, component_names
import CensoredDistributions: chain_to_params, update
using DynamicPPL: DynamicPPL
using FlexiChains: FlexiChains, VarName
import Statistics

# Build the full chain VarName for a free parameter: the submodel `~`-bound
# `prefix` is the root and the edge path + parameter name become nested
# `Property` optics over the `Iden` base, matching the optic a submodel-sampled
# parameter carries (e.g. prefix `:d`, path `(:onset_admit, :shape)` ->
# `d.onset_admit.shape`).
function _full_varname(prefix::Symbol, path::Tuple)
    APPL = DynamicPPL.AbstractPPL
    optic = APPL.Iden()
    for name in reverse(path)
        optic = APPL.Property{name}(optic)
    end
    return VarName{prefix}(optic)
end

# Read one parameter's value from the chain: posterior mean over all draws, or a
# single draw when `draw` is set. Returns `nothing` if the parameter is absent
# (a `KeyError`, e.g. a `Competing`'s branch probabilities kept fixed in the
# template); any other error propagates.
function _read_value(chain, vn, draw)
    samples = try
        chain[vn]
    catch err
        err isa KeyError && return nothing
        rethrow()
    end
    return draw === nothing ? Statistics.mean(samples) : samples[draw]
end

# Build the nested NamedTuple by walking the template, so its key order matches
# `params(template)` (deterministic, not Dict-ordered). Each node reads its
# children/parameters from the chain at the running name `path`.

function _node_params(d::Union{Sequential, Parallel}, chain, prefix, path, draw)
    names = component_names(d)
    vals = map(zip(names, d.components)) do (name, child)
        _node_params(child, chain, prefix, (path..., name), draw)
    end
    return NamedTuple{names}(Tuple(vals))
end

function _node_params(c::Competing, chain, prefix, path, draw)
    delays = map(zip(c.names, c.delays)) do (name, delay)
        _node_params(delay, chain, prefix, (path..., name), draw)
    end
    base = NamedTuple{c.names}(Tuple(delays))
    # Branch probabilities only appear in the chain when sampled (a prior was
    # supplied); otherwise omit so `update` keeps the template's fixed values.
    bp = map(c.names) do name
        _read_value(chain,
            _full_varname(prefix, (path..., :branch_probs, name)), draw)
    end
    any(x -> x === nothing, bp) && return base
    return merge(base, (; branch_probs = NamedTuple{c.names}(Tuple(bp))))
end

# Leaf: read each free parameter (in `_leaf_param_names` order) from the chain.
# A leaf parameter is always sampled, so a missing one signals a chain that does
# not match the template (wrong prefix, or not the chain that produced it);
# error rather than build a NamedTuple with a `nothing` value.
function _node_params(leaf, chain, prefix, path, draw)
    pnames = CensoredDistributions._leaf_param_names(leaf)
    vals = map(pnames) do p
        v = _read_value(chain, _full_varname(prefix, (path..., p)), draw)
        v === nothing && throw(ArgumentError(
            "leaf parameter $(_full_varname(prefix, (path..., p))) " *
            "not found in chain"))
        v
    end
    return NamedTuple{pnames}(Tuple(vals))
end

function chain_to_params(template, chain; prefix::Symbol = :d, draw = nothing)
    return _node_params(template, chain, prefix, (), draw)
end

# Update a composed distribution directly from a fitted chain, so docs call
# `update(template, chain)` rather than threading `chain_to_params` by hand.
# Reads the chain into the nested NamedTuple (posterior means, or a single `draw`)
# at the submodel `prefix`, then reconstructs through the core `update`.
@doc "

Update a composed distribution's parameters straight from a fitted chain.

`update(template, chain)` reads `chain` (sampled through
[`composed_parameters_model`](@ref)) into the nested NamedTuple and rebuilds
`template` with those values, so the workflow is one call instead of
`update(template, chain_to_params(template, chain))`. By default it applies
posterior means; pass `draw=i` for a single iteration. The `prefix` keyword names
the submodel variable the parameters were sampled under (default `:d`).

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
