# Tests for the Catalyst extension: the linear chain
# trick lowered onto Catalyst reactions. `linear_chain_reactions` slots a
# composed delay onto one transition; we solve and check the recovered mean
# dwell time and the per-transition compartment structure. Whole-model assembly
# (e.g. an SEIR or SIR) is application territory and is exercised in the
# linear-chain tutorial. The core `linear_chain_stages` lowering is Catalyst-free
# and tested in `linear_chain.jl`.

@testitem "Catalyst bridge is an exported stub; core lowering is Catalyst-free" begin
    using CensoredDistributions, Distributions

    # The reaction-network entry point is an exported generic-function stub; the
    # Catalyst-requiring method lives in the package extension, so the core stays
    # free of the SciML stack (Catalyst is a weakdep, not a dep).
    @test linear_chain_reactions isa Function
    # The (rate, stages) lowering it builds on works with no Catalyst loaded.
    s = linear_chain_stages(Gamma(3.0, 1.5))
    @test s[1].stages == 3
    @test s[1].rate ≈ 1 / 1.5
end

@testitem "linear_chain_reactions recovers the composed delay mean dwell" begin
    using CensoredDistributions, Distributions
    using Catalyst
    using Catalyst: default_t
    using OrdinaryDiffEq

    # An Erlang(3, 1.5) delay (mean 4.5) slotted onto a from -> to transition
    # gives 3 sub-compartments. A unit pulse entering the first sub-compartment
    # leaves with the Erlang waiting-time distribution, so the mean exit time
    # equals the composed delay's mean.
    delay = Gamma(3.0, 1.5)
    t = default_t()
    @species From(t) To(t)
    chain = linear_chain_reactions(delay, From, To; prefix = :X)
    species, rxs = chain.species, chain.reactions
    @test length(species) == 3

    rn = complete(ReactionSystem(rxs, t, [From, species..., To], []; name = :d))
    # Seed the first sub-compartment with a unit mass; measure the exit flux.
    u0 = [From => 0.0; species[1] => 1.0;
          [species[i] => 0.0 for i in 2:length(species)]; To => 0.0]
    prob = ODEProblem(rn, u0, (0.0, 300.0), [])
    sol = solve(prob, Tsit5(); saveat = 0.05)

    # Exit flux is the last sub-compartment's rate times its occupancy. The mean
    # exit time is the flux-weighted mean of t.
    last_rate = 1 / 1.5
    ts = sol.t
    flux = [last_rate * sol[species[end]][j] for j in eachindex(ts)]
    mass = sum((flux[i] + flux[i + 1]) / 2 * (ts[i + 1] - ts[i])
    for i in 1:(length(ts) - 1))
    meant = sum(((ts[i] * flux[i] + ts[i + 1] * flux[i + 1]) / 2) *
                (ts[i + 1] - ts[i]) for i in 1:(length(ts) - 1)) / mass
    @test mass ≈ 1.0 atol=1e-3
    @test meant ≈ mean(delay) atol=1e-2
end

@testitem "linear_chain_reactions moment-matches a non-Erlang delay" begin
    using CensoredDistributions, Distributions
    using Catalyst
    using Catalyst: default_t

    # A non-Erlang LogNormal delay drops onto a transition under moment matching,
    # giving one sub-compartment per matched Erlang stage.
    delay = LogNormal(1.0, 0.5)
    t = default_t()
    @species From(t) To(t)
    chain = linear_chain_reactions(delay, From, To; moment_match = true)
    stages = linear_chain_stages(delay; moment_match = true)
    @test length(chain.species) == stages[1].stages
    # Without moment matching the same delay is rejected.
    @test_throws ArgumentError linear_chain_reactions(delay, From, To)
end

@testitem "linear_chain_reactions threads a Sequential chain in order" begin
    using CensoredDistributions, Distributions
    using Catalyst
    using Catalyst: default_t

    # A two-step chain lowers to 2 + 1 = 3 sub-compartments on one transition.
    chain = Sequential((Gamma(2.0, 1.0), Exponential(0.5)),
        (:exposed, :infectious))
    t = default_t()
    @species From(t) To(t)
    species, rxs = linear_chain_reactions(chain, From, To)
    @test length(species) == 3
    # from -> s1, two interior hops, s3 -> to => 4 reactions.
    @test length(rxs) == 4
end
