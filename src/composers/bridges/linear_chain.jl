# ============================================================================
# compartment_stages: the distributions -> compartments bridge
# ============================================================================
#
# The LINEAR CHAIN TRICK represents an Erlang(k, θ) (integer-shape Gamma) delay
# as a chain of `k` identical Exponential(θ) sub-stages, each leaving at rate
# `1/θ`. An Exponential(θ) delay is the `k = 1` case. A waiting time built from
# such a chain is exactly Erlang-distributed, so a composed Exp/Erlang delay
# stack lowers to a set of linear ODE compartments: every step becomes `k`
# compartments in series with a single per-stage rate.
#
# `compartment_stages` reads that `(rate, stages)` structure off a leaf or a
# `Sequential` chain of such leaves. It is the small, exact extraction the
# downstream ODE/compartment view consumes; it does NOT build the ODE itself.
# The Catalyst reaction-network bridge (`linear_chain_reactions`) lives in the
# package extension; this lowering stays Catalyst-free.
# The lowering is exact only for Exponential / Erlang leaves; any other family
# (general Gamma, LogNormal, ...) throws, since no exact finite linear chain
# represents it. Passing `moment_match = true` instead lowers a non-Erlang leaf
# to the nearest Erlang chain by matching its mean and squared coefficient of
# variation (the method of stages); over-dispersed delays still have no Erlang
# representation, and a fuller phase-type fit is future work.

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
- [`compartment_stages`](@ref): build these from a delay or chain
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
        "have no exact finite linear-chain representation. Pass " *
        "`moment_match = true` to lower it to the nearest Erlang chain."))
end

# Lower any leaf to its nearest Erlang `(rate, stages)` by matching the first
# two moments. The squared coefficient of variation `c² = var / mean²` of an
# Erlang(k) chain is `1/k`, so the stage count is `k = round(1 / c²)` and the
# rate `k / mean` fixes the mean exactly. An Erlang chain cannot represent an
# over-dispersed delay (`c² > 1`, i.e. `k < 1`), so that case errors. This is
# the dependency-light method of stages; a fuller phase-type / Coxian fit for
# over-dispersed delays is future work.
function _moment_stage(d)
    m = mean(d)
    v = var(d)
    (isfinite(m) && isfinite(v) && m > 0 && v > 0) || throw(ArgumentError(
        "moment matching needs a finite positive mean and variance; got " *
        "mean = $m, var = $v for $(typeof(d))."))
    scv = v / m^2
    scv <= 1 || throw(ArgumentError(
        "moment matching to an Erlang chain needs squared coefficient of " *
        "variation ≤ 1 (under-dispersed); got $(round(scv; digits = 3)) for " *
        "$(typeof(d)). An over-dispersed delay needs a phase-type / Coxian " *
        "representation, which is future work."))
    k = max(round(Int, inv(scv)), 1)
    return (rate = k / m, stages = k)
end

_is_erlang_shape(::Exponential) = true
_is_erlang_shape(d::Gamma) = isapprox(shape(d), round(shape(d)); atol = 1e-8)
_is_erlang_shape(_) = false

# Pick the `(rate, stages)` for one free leaf. The exact Exp/Erlang fast path is
# always tried first, so `moment_match` only changes behaviour for delays the
# exact lowering would reject; an Erlang lowers identically either way.
function _stage(d, moment_match::Bool)
    moment_match || return _leaf_stage(d)
    (d isa Exponential || d isa Gamma) && _is_erlang_shape(d) &&
        return _leaf_stage(d)
    return _moment_stage(d)
end

@doc raw"
Lower a composed delay to the compartment-stage structure an ODE consumes.

`compartment_stages(d)` returns the [`ChainStage`](@ref) vector that the linear
chain trick assigns to `d`, the exact `(rate, stages)` compartment structure an
ODE/compartment model consumes. The name is deliberately representation-agnostic
(`compartment_stages`, not `linear_chain_stages`): the linear chain trick is the
only lowering today, but a future phase-type / Coxian representation can extend
this same entry point rather than adding a parallel one.

- an `Exponential(θ)` leaf gives one stage at rate ``1/\theta``;
- an `Erlang(k, θ)` leaf (an integer-shape `Gamma`) gives `k` stages at rate
  ``1/\theta``;
- a [`Sequential`](@ref) chain of such leaves gives one [`ChainStage`](@ref) per
  step, in chain order, named by the step names.

Censoring wrappers are peeled to the inner free delay first (via
`free_leaf`), so a chain of censored Exp/Erlang delays still lowers
cleanly. The total number of compartments is `sum(s.stages for s in stages)`.

This is the distributions-to-compartments bridge: a CONSUMER of the composers,
not part of the composition engine. It reads a finished composed delay and hands
its `(rate, stages)` structure to a downstream ODE/compartment view (the
Catalyst reaction-network assembly is the optional weak-dependency extension on
top); it does not build the ODE itself.

The lowering is EXACT only for Exponential / Erlang leaves; any other family
(general Gamma, LogNormal, ...) throws unless `moment_match = true`, which lowers
each leaf to the nearest Erlang chain by matching its mean and squared
coefficient of variation `c² = var / mean²`. The Erlang stage count is
`round(1 / c²)` and the rate `stages / mean` fixes the mean exactly. An Erlang
chain cannot represent an over-dispersed delay (`c² > 1`), so that case still
throws; a fuller phase-type representation is future work.

# Arguments
- `d`: an Exp/Erlang delay leaf, or a flat [`Sequential`](@ref) chain of such
  leaves, to lower to its compartment stages.

# Keyword Arguments
- `name`: the [`ChainStage`](@ref) name for a single-leaf lowering (a chain names
  each stage by its step name instead). Defaults to `:delay`.
- `moment_match`: lower a non-Erlang leaf to the nearest Erlang chain by matching
  the first two moments, instead of throwing. Exp/Erlang leaves stay exact.
  Defaults to `false`.

# Examples
```@example
using CensoredDistributions, Distributions

# An Erlang(3, 1.5) incubation -> 3 compartments leaving at rate 1/1.5.
compartment_stages(Gamma(3.0, 1.5))
```

```@example
using CensoredDistributions, Distributions

# A two-step E -> I -> R chain lowers step by step.
chain = Sequential(Gamma(2.0, 1.0), Exponential(0.5))
compartment_stages(chain)
```

```@example
using CensoredDistributions, Distributions

# A non-Erlang LogNormal delay lowers to its nearest Erlang chain.
compartment_stages(LogNormal(1.0, 0.5); moment_match = true)
```

# See also
- [`ChainStage`](@ref): the per-step record
- [`Sequential`](@ref): the chain composer this reads
"
function compartment_stages(
        d::Distribution; name::Symbol = :delay, moment_match::Bool = false)
    inner = free_leaf(d)
    s = _stage(inner, moment_match)
    return [ChainStage(name, Float64(s.rate), s.stages)]
end

function compartment_stages(chain::Sequential; moment_match::Bool = false)
    stages = ChainStage[]
    for (component, nm) in zip(chain.components, chain.names)
        component isa Sequential && throw(ArgumentError(
            "compartment_stages handles a flat Sequential of Exp/Erlang " *
            "leaves; flatten nested chains before lowering (nested step " *
            "`$(nm)` is itself a Sequential)."))
        append!(stages,
            compartment_stages(component; name = nm, moment_match))
    end
    return stages
end
