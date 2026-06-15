module CensoredDistributionsCatalystExt

# Catalyst.jl bridge for the linear chain trick.
#
# `linear_chain_stages` (in core) lowers an Exp/Erlang composed delay to its
# `(rate, stages)` Erlang sub-compartment structure without touching Catalyst.
# This extension turns that structure into actual Catalyst `Reaction`s, so a
# composed delay distribution can be slotted onto a transition of a reaction
# network (`linear_chain_reactions`) or assembled into a whole SEIR
# (`seir_reaction_network`). Without Catalyst loaded the core stays free of the
# SciML stack and these functions have no methods.
# See https://github.com/EpiAware/CensoredDistributions.jl/issues/400,
# /177 and /125.

import CensoredDistributions: linear_chain_reactions, seir_reaction_network,
                              linear_chain_stages
using Distributions: Distribution
using Catalyst: Catalyst, Reaction, ReactionSystem, complete, @species,
                @parameters, default_t

# Lower a composed delay to one Catalyst sub-compartment species per Erlang
# stage, with the per-stage exit rate aligned to each. The dwell time across the
# whole chain matches the composed delay exactly (the linear chain trick).
# Returns `(species, rates)` in chain order.
function _chain_species(delay::Distribution, prefix::Symbol)
    stages = linear_chain_stages(delay)
    t = default_t()
    species = Any[]
    rates = Float64[]
    for (gi, stage) in enumerate(stages)
        for j in 1:(stage.stages)
            nm = Symbol(prefix, gi, :_, j)
            push!(species, (@species $nm(t))[1])
            push!(rates, stage.rate)
        end
    end
    isempty(species) && throw(ArgumentError(
        "the composed delay lowered to zero stages; nothing to attach"))
    return species, rates
end

# Build the reactions for a chain of sub-compartments running into `to`. The
# reaction `A -> B` fires at A's exit rate, so sub-compartment `species[i]`
# leaves to the next at its own `rates[i]`, and the last leaves to `to` at
# `rates[end]`. The upstream `from` flows into `species[1]` at `from_rate` (the
# rate individuals enter the first sub-compartment); pass `from_rate = rates[1]`
# for a plain entry, or a custom rate (e.g. a force of infection) to wire the
# chain onto a model transition.
function _chain_reactions(from, from_rate, species, rates, to)
    rxs = Reaction[Reaction(from_rate, [from], [species[1]])]
    for idx in 1:(length(species) - 1)
        push!(rxs, Reaction(rates[idx], [species[idx]], [species[idx + 1]]))
    end
    push!(rxs, Reaction(rates[end], [species[end]], [to]))
    return rxs
end

function linear_chain_reactions(
        delay::Distribution, from, to; prefix::Symbol = :stage)
    species, rates = _chain_species(delay, prefix)
    rxs = _chain_reactions(from, rates[1], species, rates, to)
    return (species = species, reactions = rxs)
end

function seir_reaction_network(
        latent::Distribution, infectious::Distribution; name::Symbol = :seir)
    t = default_t()
    @species S(t) R(t)
    @parameters β
    # The latent (E) and infectious (I) periods are the sub-compartment chains
    # lowered from the composed delays. Building the I chain's species up front
    # lets the transmission reaction reference its sub-compartments:
    # frequency-dependent transmission depends on the TOTAL infectious count
    # (the sum over the I sub-compartments), so the Erlang shape of the
    # infectious period shapes the dynamics, not just its mean.
    e_species, e_rates = _chain_species(latent, :E)
    i_species, i_rates = _chain_species(infectious, :I)

    # Infection wires S into the E chain at rate β * (total infectious), the
    # force of infection; the E chain runs into the first I sub-compartment.
    # Each sub-compartment leaves at its own per-stage rate (so each block's
    # dwell time matches its composed delay).
    rxs = _chain_reactions(S, β * sum(i_species), e_species, e_rates,
        i_species[1])
    # The I chain's first sub-compartment is already fed by the E chain, so we
    # add only its internal transitions and its exit into R.
    for idx in 1:(length(i_species) - 1)
        push!(rxs,
            Reaction(i_rates[idx], [i_species[idx]], [i_species[idx + 1]]))
    end
    push!(rxs, Reaction(i_rates[end], [i_species[end]], [R]))

    species = Any[S, e_species..., i_species..., R]
    rn = complete(ReactionSystem(rxs, t, species, [β]; name = name))
    return (system = rn, exposed = e_species, infectious = i_species)
end

end
