# ============================================================================
# renewal(Rt, gi, I0; modulator): the renewal recurrence as a composable scan
# ============================================================================
#
# `convolve_distributions(stack, series)` pushes a FIXED series through a delay
# stack. The renewal RECURRENCE feeds its own output back as input:
#
#   I[t] = R_t * m(t) * Σ_s g_s I[t-s]
#
# so it cannot be a single convolution call. This is a small forward scan that
# reuses the same causal-convolution arithmetic ONE OUTPUT STEP AT A TIME: the
# generation-weighted sum `Σ_s g_s I[t-s]` is `_renewal_force`, the per-step
# analogue of `_causal_convolve`, evaluated against the infections built so far.
#
# The force of infection is then scaled by a composable MODULATOR. A modulator
# is any callable `m(state, t, force) -> (factor, state')`: it returns the
# multiplicative factor at time `t` and an updated carry-state for the next
# step. The default carries nothing and returns `1` (a bare renewal). The
# susceptibility-depletion modulator carries the remaining susceptible pool and
# returns `S[t]/N`, giving the SIR-style renewal `I[t] = R_t (S[t]/N) Σ g_s I`.
# Modulators COMPOSE: `combine_modulators(a, b)` multiplies their factors and
# threads both carry-states, so transmissibility / susceptibility / immunity
# terms stack rather than living in one monolithic recurrence.
#
# AD-safety. The scan is linear in the infection history and the modulator
# factors, and `gi` enters as a plain PMF vector (built by `interval_censored`
# / `discretise_pmf`, the AD-safe discretisation), so gradients flow w.r.t. Rt,
# the generation-interval parameters and the modulator parameters (N, etc.)
# under ForwardDiff / ReverseDiff / Mooncake. The accumulator element type is
# promoted across Rt, the PMF, the seed and the modulator state so Dual /
# tracked numbers propagate.

# --- the per-step force of infection Σ_s g_s I[t-s] ------------------------

# Generation-weighted sum of the infections built so far, `I[1..t-1]`, with the
# generation-interval mass `g[s]` carried back `s` steps. `g[s]` is the weight
# at lag `s` (lag-1-indexed, matching `pdf(gen_dist, 1:gi_max)`); lag 0 carries
# no mass (an infection generates from the next step on). The per-step analogue
# of `_causal_convolve`'s inner loop.
function _renewal_force(I, t, g)
    T = promote_type(eltype(I), eltype(g))
    acc = zero(T)
    smax = min(length(g), t - 1)
    @inbounds for s in 1:smax
        acc += g[s] * I[t - s]
    end
    return acc
end

# A `DelayPMF` carries its masses lag-0-indexed (`masses[1]` is lag 0). A
# generation interval has no lag-0 mass, so the lag-`s` weight is `masses[s+1]`.
_gi_weights(g::AbstractVector) = g
_gi_weights(g::DelayPMF) = @view g.masses[2:end]

# --- modulators: composable per-step force multipliers ----------------------

@doc raw"
A renewal modulator that scales the force of infection by a constant factor.

`renewal` modulators are callables `m(state, t, force) -> (factor, state')`
returning the multiplicative factor at time `t` and a carry-state for the next
step. `NoModulation()` is the identity: factor `1`, no carry. It is the default,
giving the bare renewal `I[t] = R_t Σ_s g_s I[t-s]`.

# See also
- [`renewal`](@ref): the renewal scan.
- [`susceptibility_depletion`](@ref): the SIR-style depleting-susceptible
  modulator.
"
struct NoModulation end

# The identity modulator: factor 1, carries nothing.
(::NoModulation)(state, t, force) = (true, state)
_modulator_init(::NoModulation) = nothing

@doc raw"
A renewal modulator that depletes the susceptible pool (the SIR-style renewal).

Scales the force of infection by the remaining susceptible fraction `S[t-1]/N`
and decrements the pool by each step's new infections, giving

```math
I[t] = R_t \, \frac{S[t-1]}{N} \, \sum_s g_s I[t-s], \qquad
S[t] = S[t-1] - I[t].
```

The factor uses `S` BEFORE the current step's infections are removed (the pool
available to infect at `t`); the carry-state is the post-step `S[t]`. Pass it to
[`renewal`](@ref) as the `modulator` keyword.

# Arguments
- `N`: the population size; the initial susceptible pool is `N - I0` unless an
  explicit `S0` is given.

# Keyword Arguments
- `S0`: the initial susceptible pool (default `N`); the seed infections are
  removed from it by `renewal` before the recurrence starts.

# Examples
```@example
using CensoredDistributions, Distributions
using CensoredDistributions: renewal, susceptibility_depletion

gi = pdf(interval_censored(truncated(Gamma(2.5, 1.3); lower = 1.0,
    upper = 12.0), 1.0), 1:12)
Rt = fill(1.5, 60)
infections = renewal(Rt, gi, 10.0;
    modulator = susceptibility_depletion(1.0e5))
```

# See also
- [`renewal`](@ref): the renewal scan.
- [`NoModulation`](@ref): the identity modulator.
"
struct SusceptibilityDepletion{T <: Real}
    N::T
    S0::T
end

function susceptibility_depletion(N::Real; S0::Real = N)
    T = promote_type(typeof(N), typeof(S0))
    return SusceptibilityDepletion{T}(T(N), T(S0))
end

# The carry-state is the running susceptible pool. The factor is the fraction
# susceptible BEFORE this step infects; the pool then loses this step's
# infections. `force` is the generation-weighted sum already scaled by R_t, i.e.
# this step's new infections.
function (m::SusceptibilityDepletion)(S, t, force)
    factor = S / m.N
    return factor, S - factor * force
end

_modulator_init(m::SusceptibilityDepletion) = m.S0

# --- composing modulators ---------------------------------------------------

# Two modulators threaded together: factors multiply, carry-states pair up. The
# result is itself a modulator, so composition nests.
struct ComposedModulator{A, B}
    a::A
    b::B
end

@doc raw"
Compose two renewal modulators into one.

`combine_modulators(a, b)` returns a modulator whose factor is the product of
the two factors and whose carry-state threads both, so transmissibility /
susceptibility / immunity terms STACK rather than living in one monolithic
recurrence. Composition nests, so any number of modulators combine.

# Arguments
- `a`: the first modulator.
- `b`: the second modulator; its factor multiplies `a`'s.

# Examples
```@example
using CensoredDistributions, Distributions
using CensoredDistributions: renewal, susceptibility_depletion,
                             combine_modulators, NoModulation

gi = pdf(interval_censored(truncated(Gamma(2.5, 1.3); lower = 1.0,
    upper = 12.0), 1.0), 1:12)
Rt = fill(1.3, 40)
# Susceptibility depletion stacked with the identity equals depletion alone.
m = combine_modulators(susceptibility_depletion(1.0e5), NoModulation())
infections = renewal(Rt, gi, 10.0; modulator = m)
```

# See also
- [`renewal`](@ref): the renewal scan.
- [`susceptibility_depletion`](@ref): a depleting-susceptible modulator.
"
combine_modulators(a, b) = ComposedModulator(a, b)

function (m::ComposedModulator)(state, t, force)
    fa, sa = m.a(state[1], t, force)
    fb, sb = m.b(state[2], t, force)
    return fa * fb, (sa, sb)
end

function _modulator_init(m::ComposedModulator)
    return (_modulator_init(m.a), _modulator_init(m.b))
end

# --- the renewal scan -------------------------------------------------------

@doc raw"
Run the renewal recurrence as a forward scan, returning the infection series.

`renewal(Rt, gi, I0; modulator, seed_days)` evaluates

```math
I[t] = R_t \, m(t) \, \sum_s g_s \, I[t-s]
```

step by step over `t = 1, \dots, \text{length}(Rt)`, reusing the same causal
generation-weighted sum the convolution kernel uses, applied one output step at
a time so the output can feed back as input. The first `seed_days` steps are
fixed to the seed `I0` to give the recurrence history to build on; the
recurrence then runs from `seed_days + 1`.

The `modulator` is a composable per-step force multiplier (see
[`NoModulation`](@ref), the default). [`susceptibility_depletion`](@ref) gives
the SIR-style renewal; [`combine_modulators`](@ref) stacks several. This is the
recurrence the rt-renewal tutorial hand-rolls; with the default modulator the
two agree exactly.

# Arguments
- `Rt`: the per-step reproduction number (length sets the horizon).
- `gi`: the generation-interval weights — a PMF vector (`g[s]` the lag-`s` mass,
  lag-1-indexed) or a [`DelayPMF`](@ref) (lag-0-indexed masses, lag 0 dropped).
- `I0`: the seed infection level applied to the first `seed_days` steps.

# Keyword Arguments
- `modulator`: a per-step force modulator (default [`NoModulation`](@ref)).
- `seed_days`: the number of seeded steps (default `length(gi_weights)`).

# Examples
```@example
using CensoredDistributions, Distributions
using CensoredDistributions: renewal, susceptibility_depletion

gi = pdf(interval_censored(truncated(Gamma(2.5, 1.3); lower = 1.0,
    upper = 12.0), 1.0), 1:12)
Rt = vcat(fill(1.6, 20), fill(0.8, 20))
infections = renewal(Rt, gi, 10.0)

# SIR-style with a depleting susceptible pool.
sir = renewal(Rt, gi, 10.0; modulator = susceptibility_depletion(1.0e5))
```

# See also
- [`susceptibility_depletion`](@ref): the depleting-susceptible modulator.
- [`combine_modulators`](@ref): stack several modulators.
- [`convolve_distributions`](@ref): the fixed-series convolution layer.
"
function renewal(Rt::AbstractVector, gi, I0::Real;
        modulator = NoModulation(),
        seed_days::Integer = length(_gi_weights(gi)))
    g = _gi_weights(gi)
    n = length(Rt)
    state0 = _modulator_init(modulator)
    T = _renewal_eltype(Rt, g, I0, state0)
    I = zeros(T, n)
    seed = min(seed_days, n)
    @inbounds for t in 1:seed
        I[t] = I0
    end
    state = _seed_modulator(modulator, state0, I, seed)
    @inbounds for t in (seed + 1):n
        force = Rt[t] * _renewal_force(I, t, g)
        factor, state = modulator(state, t, force)
        I[t] = factor * force
    end
    return I
end

# Promote across every input that carries a parameter so Dual / tracked numbers
# survive. The modulator state may be `nothing` (no carry) or a nested tuple.
function _renewal_eltype(Rt, g, I0, state)
    promote_type(
        eltype(Rt), eltype(g), typeof(I0), _state_eltype(state))
end

_state_eltype(::Nothing) = Bool
_state_eltype(x::Real) = typeof(x)
_state_eltype(x::Tuple) = promote_type(map(_state_eltype, x)...)

# Remove the seed infections from a depleting modulator's pool before the
# recurrence starts, so `S` enters step `seed + 1` already net of the seed. A
# modulator with no depletion (no carry) is unchanged.
_seed_modulator(::Any, ::Nothing, I, seed) = nothing
function _seed_modulator(m::SusceptibilityDepletion, S0, I, seed)
    s = S0
    @inbounds for t in 1:seed
        s -= I[t]
    end
    return s
end
function _seed_modulator(m::ComposedModulator, state, I, seed)
    return (_seed_modulator(m.a, state[1], I, seed),
        _seed_modulator(m.b, state[2], I, seed))
end
