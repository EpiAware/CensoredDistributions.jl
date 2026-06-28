# ============================================================================
# renewal(Rt, gi, I0; modulator): the renewal recurrence as a composable scan
# ============================================================================
#
# `convolved(stack, series)` pushes a fixed series through a delay
# stack. The renewal recurrence feeds its own output back as input:
#
#   I[t] = R_t * m(t) * Σ_s g_s I[t-s]
#
# so it cannot be a single convolution call. This is a small forward scan that
# reuses the same causal-convolution arithmetic one output step at a time: the
# generation-weighted sum `Σ_s g_s I[t-s]` is `_renewal_force`, the per-step
# analogue of `_causal_convolve`, evaluated against the infections built so far.
#
# The force of infection is then scaled by a composable modulator. A modulator
# is any callable `m(state, t, force) -> (factor, state')`: it returns the
# multiplicative factor at time `t` and an updated carry-state for the next
# step. The default carries nothing and returns `1` (a bare renewal). The
# shipped modulators are susceptibility depletion (a depleting susceptible
# pool), transmissibility (a per-step deterministic factor) and immunity waning
# (an immune pool that grows with infections and decays). Modulators compose:
# `combine_modulators(a, b)` multiplies their factors and threads both
# carry-states, so transmissibility, susceptibility and immunity terms stack
# rather than living in one monolithic recurrence.
#
# AD-safety. The scan is linear in the infection history and the modulator
# factors, and `gi` enters as a plain PMF vector (built by `interval_censored`
# / `discretise_pmf`, the AD-safe discretisation), so gradients flow w.r.t. Rt,
# the generation-interval parameters and the modulator parameters under
# ForwardDiff / ReverseDiff / Mooncake. The accumulator element type is promoted
# across Rt, the PMF, the seed and the modulator state so Dual / tracked numbers
# propagate.

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
The identity renewal modulator: leave the force of infection unscaled.

`renewal` modulators are callables `m(state, t, force) -> (factor, state')`
returning the multiplicative factor at step `t` and a carry-state for the next
step. `NoModulation()` returns factor `1` and carries nothing, giving the bare
renewal `I[t] = R_t \sum_s g_s I[t-s]`. It is the default modulator.

# Examples
```@example
using CensoredDistributions, Distributions

gi = pdf(interval_censored(truncated(Gamma(2.5, 1.3); lower = 1.0,
    upper = 12.0), 1.0), 1:12)
infections = renewal(fill(1.3, 40), gi, 10.0; modulator = NoModulation())
```

# See also
- [`renewal`](@ref): the renewal scan.
- [`susceptibility_depletion`](@ref): a depleting-susceptible modulator.
- [`combine_modulators`](@ref): stack several modulators.
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

The factor uses `S` before the current step's infections are removed (the pool
available to infect at `t`); the carry-state is the post-step `S[t]`. Pass it to
[`renewal`](@ref) as the `modulator` keyword.

# Arguments
- `N`: the population size; the susceptible fraction is `S/N`.

# Keyword Arguments
- `S0`: the initial susceptible pool (default `N`); `renewal` removes the seed
  infections from it before the recurrence starts.

# Examples
```@example
using CensoredDistributions, Distributions

gi = pdf(interval_censored(truncated(Gamma(2.5, 1.3); lower = 1.0,
    upper = 12.0), 1.0), 1:12)
infections = renewal(fill(1.5, 60), gi, 10.0;
    modulator = susceptibility_depletion(1.0e5))
```

# See also
- [`renewal`](@ref): the renewal scan.
- [`immunity_waning`](@ref): a depleting pool that recovers over time.
- [`combine_modulators`](@ref): stack several modulators.
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
# susceptible before this step infects; the pool then loses this step's
# infections. `force` is the generation-weighted sum already scaled by R_t, i.e.
# this step's new infections. The fraction is floored at zero so a depleted pool
# never drives the force negative (the pool cannot go below zero), keeping the
# expected counts non-negative under sampling.
function (m::SusceptibilityDepletion)(S, t, force)
    factor = max(S / m.N, zero(S))
    return factor, S - factor * force
end

_modulator_init(m::SusceptibilityDepletion) = m.S0

@doc raw"
A renewal modulator that scales the force of infection by a per-step factor.

Multiplies the force at step `t` by a deterministic transmissibility `β[t]`,
the channel for seasonal forcing, interventions or a contact-rate path. The
factor carries no state, so it composes cleanly with the depleting / waning
modulators.

# Arguments
- `beta`: the transmissibility factor — a scalar applied at every step, or a
  per-step vector read at index `t` (which must cover the horizon).

# Examples
```@example
using CensoredDistributions, Distributions

gi = pdf(interval_censored(truncated(Gamma(2.5, 1.3); lower = 1.0,
    upper = 12.0), 1.0), 1:12)
season = 1.0 .+ 0.3 .* sinpi.((1:60) ./ 30)
infections = renewal(fill(1.4, 60), gi, 10.0;
    modulator = transmissibility(season))
```

# See also
- [`renewal`](@ref): the renewal scan.
- [`combine_modulators`](@ref): stack with susceptibility / immunity.
"
struct Transmissibility{B}
    beta::B
end

transmissibility(beta) = Transmissibility(beta)

# A scalar factor applies at every step; a vector is read at step `t`.
_beta_at(beta::Real, t) = beta
_beta_at(beta::AbstractVector, t) = @inbounds beta[t]

(m::Transmissibility)(state, t, force) = (_beta_at(m.beta, t), state)
_modulator_init(::Transmissibility) = nothing

@doc raw"
A renewal modulator that builds an immune pool which grows and wanes (SIRS).

Scales the force of infection by the susceptible fraction `1 - Z[t-1]/N`, where
`Z` is an immune pool that grows by each step's infections and decays
geometrically at rate `omega` (waning immunity):

```math
I[t] = R_t \, \Bigl(1 - \frac{Z[t-1]}{N}\Bigr) \sum_s g_s I[t-s], \qquad
Z[t] = (1 - \omega)\,Z[t-1] + I[t].
```

With `omega = 0` no immunity is lost and this matches permanent depletion
([`susceptibility_depletion`](@ref) with `S0 = N`); with `omega > 0` recovered
individuals return to the susceptible pool over time.

# Arguments
- `N`: the population size; the susceptible fraction is `1 - Z/N`.
- `omega`: the per-step waning rate in `[0, 1]` (`0` no waning, `1` immediate
  loss of immunity).

# Keyword Arguments
- `Z0`: the initial immune pool (default `0`).

# Examples
```@example
using CensoredDistributions, Distributions

gi = pdf(interval_censored(truncated(Gamma(2.5, 1.3); lower = 1.0,
    upper = 12.0), 1.0), 1:12)
infections = renewal(fill(1.6, 120), gi, 10.0;
    modulator = immunity_waning(1.0e5, 0.02))
```

# See also
- [`renewal`](@ref): the renewal scan.
- [`susceptibility_depletion`](@ref): the no-waning (permanent) limit.
- [`combine_modulators`](@ref): stack several modulators.
"
struct ImmunityWaning{T <: Real}
    N::T
    omega::T
    Z0::T
end

function immunity_waning(N::Real, omega::Real; Z0::Real = zero(N))
    T = promote_type(typeof(N), typeof(omega), typeof(Z0))
    return ImmunityWaning{T}(T(N), T(omega), T(Z0))
end

# The carry-state is the immune pool. The factor is the susceptible fraction
# before this step infects; the pool then wanes and gains this step's
# infections. The fraction is floored at zero so a saturated immune pool never
# drives the force negative.
function (m::ImmunityWaning)(Z, t, force)
    factor = max(1 - Z / m.N, zero(Z))
    infections = factor * force
    return factor, (1 - m.omega) * Z + infections
end

_modulator_init(m::ImmunityWaning) = m.Z0

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
the two factors and whose carry-state threads both, so transmissibility,
susceptibility and immunity terms stack rather than living in one monolithic
recurrence. Composition nests, so any number of modulators combine.

The factors multiply and each modulator keeps its own carry-state, so the result
does not depend on the pairing order (the product is commutative and the states
are independent).

# Arguments
- `a`: the first modulator.
- `b`: the second modulator; its factor multiplies `a`'s.

# Examples
```@example
using CensoredDistributions, Distributions

gi = pdf(interval_censored(truncated(Gamma(2.5, 1.3); lower = 1.0,
    upper = 12.0), 1.0), 1:12)
season = 1.0 .+ 0.2 .* sinpi.((1:80) ./ 40)
# Transmissibility forcing stacked with a depleting susceptible pool.
m = combine_modulators(transmissibility(season),
    susceptibility_depletion(1.0e5))
infections = renewal(fill(1.5, 80), gi, 10.0; modulator = m)
```

# See also
- [`renewal`](@ref): the renewal scan.
- [`susceptibility_depletion`](@ref), [`transmissibility`](@ref),
  [`immunity_waning`](@ref): the shipped modulators.
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
[`NoModulation`](@ref), the default). [`susceptibility_depletion`](@ref),
[`transmissibility`](@ref) and [`immunity_waning`](@ref) are the shipped
modulators; [`combine_modulators`](@ref) stacks several. Feed the infection
series through an observation delay with [`observe_renewal`](@ref) to score
reported counts.

# Arguments
- `Rt`: the per-step reproduction number (length sets the horizon).
- `gi`: the generation-interval weights — a PMF vector (`g[s]` the lag-`s` mass,
  lag-1-indexed) or a [`DelayPMF`](@ref) (lag-0-indexed masses, lag 0 dropped).
- `I0`: the seed infection level applied to the first `seed_days` steps.

# Keyword Arguments
- `modulator`: a per-step force modulator (default [`NoModulation`](@ref)).
- `seed_days`: the number of seeded steps (default the generation-interval
  length).

# Examples
```@example
using CensoredDistributions, Distributions

gi = pdf(interval_censored(truncated(Gamma(2.5, 1.3); lower = 1.0,
    upper = 12.0), 1.0), 1:12)
Rt = vcat(fill(1.6, 20), fill(0.8, 20))
infections = renewal(Rt, gi, 10.0)

# SIR-style with a depleting susceptible pool.
sir = renewal(Rt, gi, 10.0; modulator = susceptibility_depletion(1.0e5))
```

# See also
- [`susceptibility_depletion`](@ref), [`transmissibility`](@ref),
  [`immunity_waning`](@ref): the shipped modulators.
- [`combine_modulators`](@ref): stack several modulators.
- [`observe_renewal`](@ref): push infections through an observation delay.
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

# Update a depleting / waning modulator's pool for the seed infections before
# the recurrence starts, so the carry-state enters step `seed + 1` already net
# of the seed. A stateless modulator (no carry) is unchanged.
#
# A `nothing` carry-state means there is nothing to seed regardless of the
# modulator, so it short-circuits to `nothing`. This is dispatched on the
# stateless `::Nothing` SECOND argument, which overlaps the typed-modulator
# methods below (they accept any state, including `nothing`); the explicit
# `::Nothing` methods for the three stateful modulators resolve that overlap
# so dispatch stays unambiguous (otherwise a `(SusceptibilityDepletion,
# nothing)` call matches both this catch-all and the typed method, and
# `detect_ambiguities` flags every pair — see #775).
_seed_modulator(::Any, ::Nothing, I, seed) = nothing
_seed_modulator(::SusceptibilityDepletion, ::Nothing, I, seed) = nothing
_seed_modulator(::ImmunityWaning, ::Nothing, I, seed) = nothing
_seed_modulator(::ComposedModulator, ::Nothing, I, seed) = nothing
function _seed_modulator(m::SusceptibilityDepletion, S0, I, seed)
    s = S0
    @inbounds for t in 1:seed
        s -= I[t]
    end
    return s
end
function _seed_modulator(m::ImmunityWaning, Z0, I, seed)
    z = Z0
    @inbounds for t in 1:seed
        z = (1 - m.omega) * z + I[t]
    end
    return z
end
function _seed_modulator(m::ComposedModulator, state, I, seed)
    return (_seed_modulator(m.a, state[1], I, seed),
        _seed_modulator(m.b, state[2], I, seed))
end

# --- the renewal -> observation bridge --------------------------------------

@doc raw"
Push a renewal infection series through an observation delay to reported counts.

`observe_renewal(infections, delay; events)` convolves the renewal output
through a delay (a leaf, a composed [`Sequential`](@ref) stack or a precomputed
[`DelayPMF`](@ref)) with the causal renewal convolution, returning the expected
reported series. It is `convolved(delay, infections)` named for the
renewal pipeline: [`renewal`](@ref) produces infections, this reports them, so
the susceptibility-modulated incidence flows through to observed cases in one
step.

# Arguments
- `infections`: the renewal infection series (the [`renewal`](@ref) output).
- `delay`: the observation delay — a univariate leaf, a composed delay stack, or
  a precomputed [`DelayPMF`](@ref).

# Keyword Arguments
- `events`: which event series to return for a branched stack (passed to
  [`convolved`](@ref)); ignored for a leaf or a `DelayPMF`.

# Examples
```@example
using CensoredDistributions, Distributions

gi = pdf(interval_censored(truncated(Gamma(2.5, 1.3); lower = 1.0,
    upper = 12.0), 1.0), 1:12)
infections = renewal(fill(1.4, 60), gi, 10.0;
    modulator = susceptibility_depletion(1.0e5))

# Report through an incubation-to-onset delay, thinned by ascertainment.
delay = thin(double_interval_censored(Gamma(1.8, 1.4); upper = 20.0,
    interval = 1.0), 0.4)
cases = observe_renewal(infections, delay)
```

# See also
- [`renewal`](@ref): the infection series this reports.
- [`convolved`](@ref): the underlying convolution.
"
function observe_renewal(infections::AbstractVector, delay::DelayPMF;
        events = nothing)
    return convolved(delay, infections)
end

function observe_renewal(infections::AbstractVector, delay; events = nothing)
    return convolved(delay, infections; events = events)
end
