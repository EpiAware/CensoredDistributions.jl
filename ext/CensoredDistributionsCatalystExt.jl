module CensoredDistributionsCatalystExt

# Catalyst.jl bridge for the linear chain trick.
#
# `compartment_stages` (in core) lowers an Exp/Erlang composed delay to its
# `(rate, stages)` Erlang sub-compartment structure without touching Catalyst.
# This extension turns that structure into actual Catalyst `Reaction`s, so a
# composed delay distribution can be slotted onto a transition of a reaction
# network (`linear_chain_reactions`). Whole-model assembly (e.g. an SEIR or SIR
# built from these edges) is application territory and lives in the linear-chain
# tutorial. Without Catalyst loaded the core stays free of the SciML stack and
# `linear_chain_reactions` has no methods.
# See https://github.com/EpiAware/CensoredDistributions.jl/issues/400,
# /177 and /125.

import CensoredDistributions: linear_chain_reactions, compartment_stages
using Distributions: Distribution
using Catalyst: Catalyst, Reaction, @species, default_t

# Lower a composed delay to one Catalyst sub-compartment species per Erlang
# stage, with the per-stage exit rate aligned to each. The dwell time across the
# whole chain matches the composed delay exactly (the linear chain trick).
# Returns `(species, rates)` in chain order.
function _chain_species(delay::Distribution, prefix::Symbol, moment_match::Bool)
    stages = compartment_stages(delay; moment_match)
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
        delay::Distribution, from, to;
        prefix::Symbol = :stage, moment_match::Bool = false)
    species, rates = _chain_species(delay, prefix, moment_match)
    rxs = _chain_reactions(from, rates[1], species, rates, to)
    # Split the entry (`from -> species[1]`) from the interior hops + exit so a
    # model builder can swap the entry (e.g. for a force of infection) and reuse
    # `internal` directly, instead of slicing `reactions[2:end]` by hand.
    return (species = species, reactions = rxs,
        entry = rxs[1], internal = rxs[2:end])
end

end
