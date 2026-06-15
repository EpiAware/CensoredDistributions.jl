# ============================================================================
# Catalyst reaction-network bridge (stubs; methods in the Catalyst extension)
# ============================================================================
#
# The linear chain trick lowers an Exp/Erlang composed delay to its
# `(rate, stages)` compartment structure with `linear_chain_stages` (Catalyst-
# free, in core). Turning those stages into an actual ODE/compartment model
# needs a reaction-network framework. We use Catalyst.jl, kept optional behind
# a package extension so the core stays free of the heavy SciML stack.
#
# These are the user-facing entry points. They have NO methods until
# Catalyst.jl is loaded; the methods live in `CensoredDistributionsCatalystExt`.
# Calling one without Catalyst loaded raises an informative error telling the
# user to `using Catalyst`.
# See https://github.com/EpiAware/CensoredDistributions.jl/issues/400,
# /177 and /125.

@doc raw"
Build the Catalyst reactions for a composed delay between two compartments.

`linear_chain_reactions(delay, from, to)` lowers a composed Exp/Erlang `delay`
to its [`linear_chain_stages`](@ref) Erlang sub-compartments and builds the
[Catalyst](https://docs.sciml.ai/Catalyst/stable/) `Reaction`s that thread an
individual from the `from` species, through one sub-compartment per Erlang
stage, to the `to` species. This is the modular primitive: it *slots a composed
delay distribution onto a single transition* of a reaction network, so a delay
written with the composers becomes an exact set of compartments on an edge of an
SEIR/SIR-type model.

The per-stage exit rate is `1/\theta` and the stage count is the Erlang shape
`k`, so the sub-compartment chain has dwell-time distribution equal to the
composed delay (the linear chain trick is exact for Exp/Erlang leaves).

This method is only defined when Catalyst.jl is loaded (`using Catalyst`); it
lives in the package extension so the core stays free of the SciML stack.

# Arguments
- `delay`: a composed Exp/Erlang delay (a leaf or a flat [`Sequential`](@ref)
  chain) to lower onto the transition.
- `from`: the upstream Catalyst species individuals enter the chain from.
- `to`: the downstream Catalyst species the chain feeds into.

# Keyword Arguments
- `prefix`: a `Symbol` prefixing the generated sub-compartment species names
  (defaults to `:stage`), e.g. `:E` gives `E1_1, E1_2, ...`.

# Returns
A `NamedTuple` `(species, reactions)`: the generated sub-compartment `species`
(in chain order) and the Catalyst `reactions` threading `from` through them to
`to`. Pass the reactions to a `ReactionSystem`, alongside `from`/`to` and the
returned `species`.

# Examples
```@example
using CensoredDistributions, Distributions, Catalyst

t = Catalyst.default_t()
@species From(t) To(t)
# An Erlang(3, 1.5) infectious period -> 3 sub-compartments on the From -> To edge.
chain = linear_chain_reactions(Gamma(3.0, 1.5), From, To; prefix = :I)
length(chain.species)
```

# See also
- [`linear_chain_stages`](@ref): the Catalyst-free `(rate, stages)` lowering
- [`seir_reaction_network`](@ref): an SEIR built from two composed delays
"
function linear_chain_reactions end

# Fallback when the Catalyst extension is not loaded: the typed method in
# `CensoredDistributionsCatalystExt` is more specific and shadows this, so this
# only fires without Catalyst, giving an actionable error instead of a bare
# `MethodError`.
function linear_chain_reactions(args...; kwargs...)
    throw(ArgumentError("`linear_chain_reactions` needs Catalyst.jl; run " *
                        "`using Catalyst` to load the reaction-network extension."))
end

@doc raw"
Build an SEIR Catalyst reaction network from two composed delays.

`seir_reaction_network(latent, infectious)` assembles a complete
[Catalyst](https://docs.sciml.ai/Catalyst/stable/) `ReactionSystem` for an
SEIR-type model whose latent (E) and infectious (I) periods are the Erlang
sub-compartment chains lowered from the composed `latent` and `infectious`
delays via [`linear_chain_reactions`](@ref). Transmission is frequency-dependent
on the total infectious count (the sum over the I sub-compartments), so giving
the infectious period an Erlang shape changes the dynamics, not just the mean.

This demonstrates the modular goal: a user composes delays with the
distribution composers, then slots each onto a compartment transition of an
SEIR. The returned system is `complete`, ready to pass to `ODEProblem`.

This method is only defined when Catalyst.jl is loaded (`using Catalyst`); it
lives in the package extension so the core stays free of the SciML stack.

# Arguments
- `latent`: the composed Exp/Erlang delay for the exposed (E) period.
- `infectious`: the composed Exp/Erlang delay for the infectious (I) period.

# Keyword Arguments
- `name`: the `ReactionSystem` name (defaults to `:seir`).

# Returns
A `NamedTuple` `(system, exposed, infectious)`: a complete Catalyst
`ReactionSystem` (`system`) with a transmission-rate parameter `β`, the boundary
species `S`/`R`, and the E and I sub-compartment chains; plus the `exposed` and
`infectious` sub-compartment species, returned so the I total can be read back
and the initial state seeded (see the linear-chain tutorial).

# Examples
```@example
using CensoredDistributions, Distributions, Catalyst

# Erlang(2, 2) latent period, Erlang(3, 1.5) infectious period.
seir = seir_reaction_network(Gamma(2.0, 2.0), Gamma(3.0, 1.5))
(length(seir.exposed), length(seir.infectious))
```

# See also
- [`linear_chain_reactions`](@ref): the per-transition primitive this composes
- [`linear_chain_stages`](@ref): the Catalyst-free `(rate, stages)` lowering
"
function seir_reaction_network end

# Fallback when the Catalyst extension is not loaded (see above).
function seir_reaction_network(args...; kwargs...)
    throw(ArgumentError("`seir_reaction_network` needs Catalyst.jl; run " *
                        "`using Catalyst` to load the reaction-network extension."))
end
