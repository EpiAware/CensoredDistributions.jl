# ============================================================================
# Turing glue stub for the recurrent / cyclic multi-state models
# ============================================================================
#
# `recurrent_states_model(template, priors)` is the recurrent analogue of
# `composed_parameters_model`: a DynamicPPL submodel that samples each state's
# transition-node parameters from per-state priors and returns the rebuilt
# `RecurrentStates` (or `CTMCStates`) model. The user scores observed paths with
# `@addlogprob! logpdf(model, path)`, so a cyclic model fits like the rest of
# the stack. This function has no methods until DynamicPPL (or Turing) is
# loaded; the method lives in the package extension, keeping the core
# Turing-free.

@doc """

Build a DynamicPPL submodel that samples a recurrent multi-state model's
per-state transition parameters from priors and returns the rebuilt model.

`recurrent_states_model(template, priors)` is the cyclic analogue of
[`composed_parameters_model`](@ref). The `template` is a
[`RecurrentStates`](@ref) model defining the state graph and the parameter
inventory; `priors` is a
`NamedTuple` keyed by state name, each value the per-state node's priors in the
same nested form [`composed_parameters_model`](@ref) takes (a leaf's parameter
priors, or a [`Compete`](@ref) / [`Resolve`](@ref) node's per-edge priors). The
returned submodel samples each prior, rebuilds the SAME state graph, and returns
the reconstructed model, which the user scores against observed paths with
`@addlogprob! logpdf(model, path)`.

Sampled parameters are namespaced by their state and edge through nested
submodel prefixing, so a chain reads `infected.recovered.shape`,
`infected.dead.scale`, and so on.

This function has no methods until `DynamicPPL` (or `Turing`) is loaded; the
method lives in the package extension so the core stays free of `DynamicPPL`.

# Arguments
- `template`: a [`RecurrentStates`](@ref) model defining the state graph and the
  parameter inventory to rebuild.
- `priors`: a `NamedTuple` keyed by state name; each value is that state's
  transition-node priors (the nested form [`composed_parameters_model`](@ref)
  uses for the node).

# Examples
```@example
using CensoredDistributions, Distributions, DynamicPPL

template = recur(
    :well => (:ill => Gamma(2.0, 5.0)),
    :ill => (:well => Gamma(2.0, 3.0), :dead => Gamma(2.0, 10.0)))
priors = (
    well = (ill = (shape = truncated(Normal(2, 0.5); lower = 0),
        scale = truncated(Normal(5, 1); lower = 0)),),
    ill = (well = (shape = truncated(Normal(2, 0.5); lower = 0),
            scale = truncated(Normal(3, 1); lower = 0)),
        dead = (shape = truncated(Normal(2, 0.5); lower = 0),
            scale = truncated(Normal(10, 2); lower = 0))))

@model function fit(t, p, paths)
    m ~ to_submodel(recurrent_states_model(t, p))
    for path in paths
        DynamicPPL.@addlogprob! logpdf(m, path)
    end
end
nothing # hide
```

# See also
- [`RecurrentStates`](@ref): the model the submodel rebuilds.
- [`composed_parameters_model`](@ref): the acyclic analogue this mirrors.
"""
function recurrent_states_model end
