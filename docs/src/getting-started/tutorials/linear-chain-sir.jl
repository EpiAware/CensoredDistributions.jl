# # [Composed delays as SEIR compartments: the linear chain trick](@id linear-chain-sir)
#
# ## Introduction
#
# CensoredDistributions.jl describes delays as composed distributions: a latent
# (exposed) period, an infectious period, and so on, chained with
# [`Sequential`](@ref) and [`compose`](@ref).
# A mechanistic compartmental model describes the same delays as flows between
# compartments of an ODE system.
# This tutorial bridges the two views with the *linear chain trick*, then slots
# composed delays onto the transitions of an SEIR reaction network.
#
# The linear chain trick rests on one fact.
# An Erlang(``k``, ``\theta``) waiting time (an integer-shape Gamma) is the sum
# of ``k`` independent Exponential(``\theta``) waits.
# So a compartment that an individual leaves at constant rate ``1/\theta`` gives
# an Exponential dwell time, and ``k`` such compartments in series give an
# Erlang(``k``, ``\theta``) dwell time.
# A delay we wrote as a distribution becomes a chain of ODE compartments with a
# single per-stage rate, exactly.
#
# ### What are we going to do in this exercise
#
# We do three things:
#
# 1. Take a composed Exp/Erlang delay and read off its linear-chain
#    `(rate, stages)` structure with [`linear_chain_stages`](@ref).
# 2. Slot a composed delay onto a single transition of a reaction network with
#    [`linear_chain_reactions`](@ref).
# 3. Build a whole SEIR as a Catalyst reaction network from two composed delays
#    with [`seir_reaction_network`](@ref), then solve and plot it.
#
# ### What might I need to know before starting
#
# This tutorial builds on [Getting Started with
# CensoredDistributions.jl](@ref getting-started) and the composer reference,
# [Composing censored distributions](@ref composer-toolkit).
#
# #### Which framework, and why
#
# The bridge target is a compartmental-modelling framework that can hold the
# delay compartments and let us slot a composed delay onto a model transition.
# We target [Catalyst.jl](https://docs.sciml.ai/Catalyst/stable/).
# Catalyst builds a model as a `ReactionSystem` of species and reactions, the
# natural home for a chain of sub-compartments threaded between two species, and
# turns straight into an `ODEProblem` for the SciML solvers.
# The Catalyst bridge ships as a **package extension**: the
# [`linear_chain_stages`](@ref) lowering is Catalyst-free and lives in the core,
# while [`linear_chain_reactions`](@ref) and [`seir_reaction_network`](@ref) only
# have methods once Catalyst is loaded.
# This tutorial therefore needs `using Catalyst` to load the extension.
#
# We considered the alternatives.
# [AlgebraicPetri.jl](https://algebraicjulia.github.io/AlgebraicPetri.jl/dev/)
# builds compartmental models as open Petri nets with category-theoretic
# composition and stratification; it is more powerful for large structured
# models but heavier (a Catlab dependency) than this bridge needs.
# A hand-written `ODEProblem` with one flat right-hand side is the lightest
# option but does not let us *slot* a composed delay onto a named transition,
# which is the modular goal here.
# Catalyst sits between the two: it composes named species and reactions without
# the categorical overhead, and lowers straight to the SciML solvers.
#
# ## Packages used
#
# We use Distributions for the delay distributions, CensoredDistributions for
# the composers, the [`linear_chain_stages`](@ref) lowering and the Catalyst
# bridge (`using Catalyst` loads the extension), OrdinaryDiffEq to solve, and
# CairoMakie for the plot.

using CensoredDistributions
using Distributions
using Catalyst
using OrdinaryDiffEq
using CairoMakie

# ## A composed delay
#
# We describe a disease with an *exposed* (latent, not yet infectious) period
# and an *infectious* period.
# We give the latent period an Erlang(2) shape and the infectious period an
# Erlang(3) shape: both are integer-shape Gammas, so both lower exactly.

latent_period = Gamma(2.0, 2.0)        # Erlang(2, 2): mean 4 days
infectious_period = Gamma(3.0, 1.5)    # Erlang(3, 1.5): mean 4.5 days

# ## Reading the linear-chain structure
#
# [`linear_chain_stages`](@ref) reads the `(rate, stages)` structure off a delay,
# one [`ChainStage`](@ref CensoredDistributions.ChainStage) per step.
# The exposed period becomes 2 compartments and the infectious period 3, each
# with its own per-stage rate.
# This lowering is pure and Catalyst-free; it is the small, exact extraction the
# reaction-network bridge consumes.

e_stages = linear_chain_stages(latent_period; name = :exposed)
i_stages = linear_chain_stages(infectious_period; name = :infectious)

# Each stage's `stages / rate` is the mean dwell time, matching the delay it
# came from. This is the linear chain trick's exactness: the chain of
# compartments has the same Erlang waiting-time distribution as the delay.

for s in (e_stages[1], i_stages[1])
    println(s.name, ": ", s.stages, " compartments, mean dwell ",
        round(s.stages / s.rate; digits = 2), " days")
end

# ## Slotting a composed delay onto one transition
#
# [`linear_chain_reactions`](@ref) is the modular primitive: it lowers a composed
# delay and builds the Catalyst reactions that thread an individual from a
# `from` species, through one sub-compartment per Erlang stage, to a `to`
# species.
# This is what it means to "attach a composed delay distribution to a compartment
# transition": the `from -> to` edge becomes the delay's exact compartment chain.

t = Catalyst.default_t()
@species From(t) To(t)
infectious_chain = linear_chain_reactions(
    infectious_period, From, To; prefix = :I)
## the Erlang(3) infectious period -> 3 sub-compartment species
length(infectious_chain.species)

# ## Building an SEIR from two composed delays
#
# [`seir_reaction_network`](@ref) composes that primitive into a whole SEIR.
# It slots the `latent` delay onto the S -> E -> I edge and the `infectious`
# delay onto the I -> R edge, so the exposed and infectious periods are the
# Erlang sub-compartment chains lowered from the composed delays.
# Transmission is frequency-dependent on the *total* infectious count (the sum
# over the I sub-compartments), so the Erlang shape of the infectious period
# shapes the dynamics, not just its mean.
# The returned `system` is a complete Catalyst `ReactionSystem`; `exposed` and
# `infectious` are the sub-compartment species, returned so we can seed and read
# them back.

seir = seir_reaction_network(latent_period, infectious_period)
rn = seir.system
E = seir.exposed
I = seir.infectious

# ## Solving and plotting
#
# We seed a small infectious fraction and solve over 120 days.
# `β` is chosen to give a basic reproduction number of about 2 (``R_0 = \beta``
# times the mean infectious period of 4.5 days).

R0 = 2.0
mean_infectious = mean(infectious_period)
β_val = R0 / mean_infectious

# Catalyst names the transmission parameter `β`, the susceptible species `S` and
# the removed species `R`. We read them off the system to set the initial state.

β = Catalyst.parameters(rn)[1]
S = Catalyst.species(rn)[1]
R = Catalyst.species(rn)[end]

u0 = [S => 0.999; [e => 0.0 for e in E];
      I[1] => 0.001; [I[k] => 0.0 for k in 2:length(I)]; R => 0.0]

prob = ODEProblem(rn, u0, (0.0, 120.0), [β => β_val])
sol = solve(prob, Tsit5(); saveat = 0.5)

# We read the compartment groups back out: total exposed is the sum over the E
# sub-compartments, total infectious over the I sub-compartments.
# The exposed and infectious curves are the delay distributions' two periods
# made explicit as populations over time.

ts = sol.t
S_t = sol[S]
R_t = sol[R]
E_t = [sum(sol[E[i]][j] for i in eachindex(E)) for j in eachindex(ts)]
I_t = [sum(sol[I[i]][j] for i in eachindex(I)) for j in eachindex(ts)]

fig = Figure(; size = (700, 420))
ax = Axis(fig[1, 1]; xlabel = "time (days)", ylabel = "population fraction",
    title = "SEIR with Erlang E and I periods (linear chain trick)")
lines!(ax, ts, S_t; label = "S (susceptible)")
lines!(ax, ts, E_t; label = "E (exposed, 2 compartments)")
lines!(ax, ts, I_t; label = "I (infectious, 3 compartments)")
lines!(ax, ts, R_t; label = "R (removed)")
axislegend(ax; position = :rc)
fig

# The exposed and infectious peaks are the same Erlang delays we wrote as
# `Gamma(2, 2)` and `Gamma(3, 1.5)`, now expressed as compartment populations
# inside the transmission dynamics.
# The single per-stage rate in each chain reproduces the delay's mean exactly,
# and the integer shape sets the number of compartments.

# ## Summary
#
# - The linear chain trick represents an Erlang(``k``, ``\theta``) delay as
#   ``k`` Exponential compartments in series, each leaving at rate ``1/\theta``;
#   an Exponential delay is the ``k = 1`` case.
# - [`linear_chain_stages`](@ref) reads that `(rate, stages)` structure off a
#   composed delay, peeling any censoring to the free delay. It is Catalyst-free,
#   exact only for Exp/Erlang leaves, and throws for other families.
# - [`linear_chain_reactions`](@ref) slots a composed delay onto a single
#   `from -> to` transition of a Catalyst reaction network, and
#   [`seir_reaction_network`](@ref) composes that into a whole SEIR whose E and I
#   periods are the lowered Erlang chains.
# - The Catalyst bridge ships as a package extension, so the core stays free of
#   the SciML stack; load it with `using Catalyst`. AlgebraicPetri is the heavier
#   alternative for stratified or open-system composition.
