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
written with the composers becomes an exact set of compartments on an edge of a
compartmental model. Assembling whole models (e.g. an SEIR or SIR) from these
edges is application territory; the linear-chain tutorial works one through.

The per-stage exit rate is `1/\theta` and the stage count is the Erlang shape
`k`, so the sub-compartment chain has dwell-time distribution equal to the
composed delay (the linear chain trick is exact for Exp/Erlang leaves).

This method is only defined when Catalyst.jl is loaded (`using Catalyst`); it
lives in the package extension so the core stays free of the SciML stack.
The bridge is a CONSUMER of the composers, not part of the composition engine:
it reads a finished composed delay and lowers it onto a reaction network, and it
stays an intentional optional weak-dependency extension for that reason.

# Arguments
- `delay`: a composed Exp/Erlang delay (a leaf or a flat [`Sequential`](@ref)
  chain) to lower onto the transition.
- `from`: the upstream Catalyst species individuals enter the chain from.
- `to`: the downstream Catalyst species the chain feeds into.

# Keyword Arguments
- `prefix`: a `Symbol` prefixing the generated sub-compartment species names
  (defaults to `:stage`), e.g. `:E` gives `E1_1, E1_2, ...`.
- `moment_match`: lower a non-Erlang delay to the nearest Erlang chain by
  matching its first two moments, instead of throwing (see
  [`linear_chain_stages`](@ref)). Defaults to `false`.

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
