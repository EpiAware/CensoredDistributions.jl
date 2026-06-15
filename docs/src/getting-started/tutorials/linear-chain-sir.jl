# # [A composed delay as ODE compartments: the linear chain trick](@id linear-chain-sir)
#
# ## Introduction
#
# CensoredDistributions.jl describes delays as composed distributions: an
# incubation period, an onset-to-recovery delay, and so on, chained with
# [`Sequential`](@ref) and [`compose`](@ref).
# A mechanistic compartmental model describes the same delays as flows between
# compartments of an ODE system.
# This tutorial bridges the two views with the *linear chain trick*.
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
# 1. Take a composed Exp/Erlang delay stack and read off its linear-chain
#    `(rate, stages)` structure with [`linear_chain_stages`](@ref).
# 2. Turn that structure into a set of ODE compartments.
# 3. Compose those delay compartments with an SIR-type transmission model so the
#    latent (E) and infectious (I) periods are explicit sub-systems of one
#    larger ODE, then solve and plot.
#
# ### What might I need to know before starting
#
# This tutorial builds on [Getting Started with
# CensoredDistributions.jl](@ref getting-started) and the composer reference,
# [Composing censored distributions](@ref composer-toolkit).
#
# #### Which framework, and why
#
# The bridge target is a modelling framework that can hold the delay
# compartments as a *sub-system* and compose it with a transmission model.
# We target [ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/)
# (MTK).
# MTK builds an ODE symbolically as a `System` and composes named sub-systems
# with `compose`, namespacing each sub-system's variables.
# That is the cleanest match for the goal here: the delay chain is one named
# module, the transmission model another, and they join into a single system,
# mirroring how [`compose`](@ref) joins delay distributions on the
# distributions side.
#
# We considered two alternatives.
# [AlgebraicPetri.jl](https://algebraicjulia.github.io/AlgebraicPetri.jl/dev/)
# builds compartmental models as open Petri nets with category-theoretic
# composition and stratification.
# It is more powerful for large structured models but heavier (a Catlab
# dependency and the structured-cospan machinery) than this distribution ->
# compartments bridge needs.
# A hand-written `ODEProblem` with one flat right-hand side is the lightest
# option but does not *compose* modules: the delay and transmission parts would
# be tangled in one vector field, which is the opposite of what this issue asks
# to demonstrate.
# MTK sits between the two: it composes named sub-systems without the
# categorical overhead.
# The maintainer may revisit AlgebraicPetri if stratified or open-system
# composition becomes the priority.
#
# ## Packages used
#
# We use Distributions for the delay distributions, CensoredDistributions for
# the composers and the [`linear_chain_stages`](@ref) lowering, ModelingToolkit
# to build and compose the ODE systems, OrdinaryDiffEq to solve them, and
# CairoMakie for the plot.

using CensoredDistributions
using Distributions
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using CairoMakie

# ## A composed delay stack
#
# We describe a disease with an *exposed* (latent, not yet infectious) period
# and an *infectious* period.
# We give the latent period an Erlang(2) shape and the infectious period an
# Erlang(3) shape: both are integer-shape Gammas, so both lower exactly.
# We chain them with [`Sequential`](@ref), naming the steps, so the object reads
# as a delay from exposure through the end of infectiousness.

latent_period = Gamma(2.0, 2.0)        # Erlang(2, 2): mean 4 days
infectious_period = Gamma(3.0, 1.5)    # Erlang(3, 1.5): mean 4.5 days

delay = Sequential((latent_period, infectious_period),
    (:exposed, :infectious))

# ## Lowering the delay to compartments
#
# [`linear_chain_stages`](@ref) reads the `(rate, stages)` structure off the
# chain, one [`ChainStage`](@ref CensoredDistributions.ChainStage) per step.
# The exposed step becomes 2 compartments and the infectious step 3, each with
# its own per-stage rate.

stages = linear_chain_stages(delay)

# Each stage's `stages / rate` is the mean dwell time, matching the delay it
# came from. This is the linear chain trick's exactness: the chain of
# compartments has the same Erlang waiting-time distribution as the delay.

for s in stages
    println(s.name, ": ", s.stages, " compartments, mean dwell ",
        round(s.stages / s.rate; digits = 2), " days")
end

# ## Building a delay sub-system in ModelingToolkit
#
# We turn one [`ChainStage`](@ref CensoredDistributions.ChainStage) group into
# an MTK `System` of compartments in series.
# Each compartment leaves at the stage rate to the next; the last leaves the
# sub-system entirely.
# We expose two interface variables the larger model connects to: `inflow` (the
# rate individuals enter the first compartment) and `outflow` (the rate they
# leave the last).
# Everything else is internal to the module.

function chain_subsystem(stage::CensoredDistributions.ChainStage; name)
    k = stage.stages
    r = stage.rate
    @variables x(t)[1:k] inflow(t) outflow(t)
    eqs = Equation[]
    ## First compartment: in from `inflow`, out at rate r to the next.
    push!(eqs, D(x[1]) ~ inflow - r * x[1])
    for i in 2:k
        push!(eqs, D(x[i]) ~ r * x[i - 1] - r * x[i])
    end
    ## The module's outflow is the rate leaving its last compartment.
    push!(eqs, outflow ~ r * x[k])
    return System(eqs, t; name = name)
end

# We build one sub-system per step. These are the delay *modules*: an
# `:exposed` chain (the E compartments) and an `:infectious` chain (the I
# compartments).

exposed = chain_subsystem(stages[1]; name = :exposed)
infectious = chain_subsystem(stages[2]; name = :infectious)

# ## An SIR-type transmission model
#
# The transmission model carries the susceptible pool `S` and the cumulative
# removed pool `R`, plus the *force of infection* and the total exposed and
# infectious counts as interface variables the delay modules plug into.
# New infections move S into the exposed chain; the infectious chain's outflow
# moves into R.
# Transmission depends on the total infectious count, summed across the
# infectious chain's compartments.

@parameters β
@variables S(t) R(t) foi(t) E_in(t) I_out(t) I_total(t)

# `foi` is the force of infection, `β` times the infectious fraction.
# `E_in` is the new-infection rate (S leaving), which feeds the exposed chain.
# `I_out` is the recovery rate, which the infectious chain supplies.
# We compose the three systems, then connect them: the exposed inflow is the new
# infections, the infectious inflow is the exposed outflow, and removals are the
# infectious outflow.

N = 1.0  # work in population fractions

connections = [
    ## Total infectious is the sum over the infectious chain compartments.
    I_total ~ sum(infectious.x),
    foi ~ β * I_total / N,
    ## New infections: susceptibles leave at the force of infection.
    E_in ~ foi * S,
    D(S) ~ -E_in,
    ## Wire the new infections into the exposed chain.
    exposed.inflow ~ E_in,
    ## The exposed chain's outflow becomes the infectious chain's inflow.
    infectious.inflow ~ exposed.outflow,
    ## Removals accumulate the infectious chain's outflow.
    I_out ~ infectious.outflow,
    D(R) ~ I_out
]

@named epidemic = System(connections, t; systems = [exposed, infectious])

# `mtkcompile` flattens the composed system, resolves the namespaced
# sub-system variables, and simplifies it ready to solve.

sys = mtkcompile(epidemic)

# ## Solving and plotting
#
# We seed a small exposed fraction in the first exposed compartment and solve
# over 120 days.
# `β` is chosen to give a basic reproduction number of about 2 (``R_0 = \beta``
# times the mean infectious period of 4.5 days).

R0 = 2.0
mean_infectious = stages[2].stages / stages[2].rate
β_val = R0 / mean_infectious

u0 = [
    S => 0.999,
    R => 0.0,
    exposed.x[1] => 0.001, exposed.x[2] => 0.0,
    infectious.x[1] => 0.0, infectious.x[2] => 0.0, infectious.x[3] => 0.0
]

prob = ODEProblem(sys, [u0; β => β_val], (0.0, 120.0))
sol = solve(prob, Tsit5(); saveat = 0.5)

# We read the compartment groups back out: total exposed is the sum over the E
# compartments, total infectious over the I compartments.
# The exposed and infectious curves are the delay distribution's two stages made
# explicit as populations over time.

ts = sol.t
S_t = sol[S]
R_t = sol[R]
E_t = [sum(sol[exposed.x[i]][j] for i in 1:stages[1].stages)
       for j in eachindex(ts)]
I_t = [sum(sol[infectious.x[i]][j] for i in 1:stages[2].stages)
       for j in eachindex(ts)]

fig = Figure(; size = (700, 420))
ax = Axis(fig[1, 1]; xlabel = "time (days)", ylabel = "population fraction",
    title = "SIR with Erlang E and I periods (linear chain trick)")
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
#   composed [`Sequential`](@ref) chain of Exp/Erlang leaves, peeling any
#   censoring to the free delay. It is exact only for Exp/Erlang leaves and
#   throws for other families.
# - We built each delay step as a ModelingToolkit `System` of compartments and
#   composed those modules with an SIR-type transmission system, so the latent
#   and infectious periods are explicit compartments of one larger ODE.
# - We targeted ModelingToolkit because its named-sub-system composition mirrors
#   the composer's [`compose`](@ref) on the distributions side; AlgebraicPetri
#   is the heavier alternative for stratified or open-system composition.
