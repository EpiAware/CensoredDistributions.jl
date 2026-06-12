# ============================================================================
# linear_chain_stages: the distributions -> compartments bridge
# ============================================================================
#
# The LINEAR CHAIN TRICK represents an Erlang(k, θ) (integer-shape Gamma) delay
# as a chain of `k` identical Exponential(θ) sub-stages, each leaving at rate
# `1/θ`. An Exponential(θ) delay is the `k = 1` case. A waiting time built from
# such a chain is exactly Erlang-distributed, so a composed Exp/Erlang delay
# stack lowers to a set of linear ODE compartments: every step becomes `k`
# compartments in series with a single per-stage rate.
#
# `linear_chain_stages` reads that `(rate, stages)` structure off a leaf or a
# `Sequential` chain of such leaves. It is the small, exact extraction the
# downstream ODE/compartment view (e.g. a ModelingToolkit `System`) consumes;
# it does NOT build the ODE itself (that lives in the integrating framework).
# The lowering is EXACT only for Exponential / Erlang leaves; any other family
# (general Gamma, LogNormal, ...) throws, since no exact finite linear chain
# represents it.

@doc raw"
A single Erlang stage of a linear-chain delay representation.

`ChainStage` records the `(rate, stages)` pair the [linear chain trick](https://doi.org/10.1007/s00285-019-01412-w)
assigns to one delay: an Erlang(``k``, ``\theta``) delay is `stages = k`
Exponential sub-compartments in series, each leaving at `rate = 1/\theta`. An
Exponential(``\theta``) delay is the `stages = 1` case. The mean dwell time is
`stages / rate = k\theta`, matching the delay's mean.

# Fields
- `name`: the step name this stage came from (`:step_i` for a positional chain,
  or the user's key); a single leaf reports `:delay`.
- `rate`: the per-stage exit rate ``1/\theta`` (units of 1/time).
- `stages`: the integer number of sub-compartments ``k`` (the Erlang shape).

# See also
- [`linear_chain_stages`](@ref): build these from a delay or chain
"
struct ChainStage
    "Step name this stage was extracted from."
    name::Symbol
    "Per-stage exit rate ``1/\\theta``."
    rate::Float64
    "Number of Exponential sub-compartments ``k`` (the Erlang shape)."
    stages::Int
end

function Base.show(io::IO, s::ChainStage)
    print(io, "ChainStage(", s.name, ": ", s.stages,
        s.stages == 1 ? " stage @ rate " : " stages @ rate ",
        round(s.rate; digits = 4), ")")
    return nothing
end

# Round an integer-valued float (within tolerance) to an Int, else error: the
# Erlang shape must be a whole number for an exact linear chain.
function _erlang_shape(shape::Real)
    k = round(Int, shape)
    isapprox(shape, k; atol = 1e-8) || throw(ArgumentError(
        "the linear chain trick is exact only for Exponential or Erlang " *
        "(integer-shape Gamma) delays; got shape = $shape (non-integer). " *
        "Use a phase-type / method-of-stages approximation for general " *
        "delays."))
    k >= 1 || throw(ArgumentError("Erlang shape must be ≥ 1, got $shape"))
    return k
end

# Map a single leaf distribution to its `(rate, stages)`. An Exponential is one
# stage at rate `1/scale`; an Erlang (integer-shape Gamma) is `shape` stages at
# rate `1/scale`. Any other family is not exactly representable.
function _leaf_stage(d::Exponential)
    return (rate = inv(scale(d)), stages = 1)
end

function _leaf_stage(d::Gamma)
    k = _erlang_shape(shape(d))
    return (rate = inv(scale(d)), stages = k)
end

function _leaf_stage(d)
    throw(ArgumentError(
        "the linear chain trick needs an Exponential or Erlang " *
        "(integer-shape Gamma) leaf; got $(typeof(d)). Other delay families " *
        "have no exact finite linear-chain representation."))
end

@doc raw"
Extract the linear-chain (Erlang-stage) representation of an Exp/Erlang delay.

`linear_chain_stages(d)` returns the [`ChainStage`](@ref) vector that the linear
chain trick assigns to `d`, the exact `(rate, stages)` compartment structure an
ODE/compartment model consumes:

- an `Exponential(θ)` leaf gives one stage at rate ``1/\theta``;
- an `Erlang(k, θ)` leaf (an integer-shape `Gamma`) gives `k` stages at rate
  ``1/\theta``;
- a [`Sequential`](@ref) chain of such leaves gives one [`ChainStage`](@ref) per
  step, in chain order, named by the step names.

Censoring wrappers are peeled to the inner free delay first (via
[`free_leaf`](@ref)), so a chain of censored Exp/Erlang delays still lowers
cleanly. The total number of compartments is `sum(s.stages for s in stages)`.

The lowering is EXACT only for Exponential / Erlang leaves; any other family
(general Gamma, LogNormal, ...) throws.

# Examples
```jldoctest
using CensoredDistributions, Distributions

# An Erlang(3, 1.5) incubation -> 3 compartments leaving at rate 1/1.5.
linear_chain_stages(Gamma(3.0, 1.5))

# output

1-element Vector{CensoredDistributions.ChainStage}:
 ChainStage(delay: 3 stages @ rate 0.6667)
```

```jldoctest
using CensoredDistributions, Distributions

# A two-step E -> I -> R chain lowers step by step.
chain = Sequential(Gamma(2.0, 1.0), Exponential(0.5))
linear_chain_stages(chain)

# output

2-element Vector{CensoredDistributions.ChainStage}:
 ChainStage(step_1: 2 stages @ rate 1.0)
 ChainStage(step_2: 1 stage @ rate 2.0)
```

# See also
- [`ChainStage`](@ref): the per-step record
- [`Sequential`](@ref): the chain composer this reads
"
function linear_chain_stages(d::Distribution; name::Symbol = :delay)
    inner = free_leaf(d)
    s = _leaf_stage(inner)
    return [ChainStage(name, Float64(s.rate), s.stages)]
end

function linear_chain_stages(chain::Sequential)
    stages = ChainStage[]
    for (component, nm) in zip(chain.components, chain.names)
        component isa Sequential && throw(ArgumentError(
            "linear_chain_stages handles a flat Sequential of Exp/Erlang " *
            "leaves; flatten nested chains before lowering (nested step " *
            "`$(nm)` is itself a Sequential)."))
        append!(stages, linear_chain_stages(component; name = nm))
    end
    return stages
end
