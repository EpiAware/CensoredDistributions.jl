# ============================================================================
# Hazards from a composed tree
# ============================================================================
#
# Hazard, log-hazard, cumulative hazard and survival accessors for a composed
# delay, as thin functions of its `pdf`/`ccdf`/`logpdf`/`logccdf`.
#
# Definitions (the standard survival-analysis identities, matching
# `SurvivalDistributions.hazard` / `cumhazard` / `loghazard`):
#
#   survival   S(t) = ccdf(d, t)
#   cumhazard  H(t) = -log S(t) = -logccdf(d, t)
#   hazard     h(t) = f(t) / S(t) = pdf(d, t) / ccdf(d, t)
#   loghazard  log h(t) = logpdf(d, t) - logccdf(d, t)
#
# `loghazard`/`cumhazard` are computed in log space to stay finite and
# AD-stable into the tail where `S(t) → 0`. For a univariate composed delay
# the accessors read its own hazard surface; for the multivariate composers
# they read the marginal univariate time-to-event (a `Sequential` chain's
# total-time convolution; a `Parallel` has no single time-to-event, so it
# errors).

@doc raw"""

Survival function ``S(t) = P(T > t)`` of a composed delay.

`survival(d, t)` is the probability the delay exceeds `t`, i.e. `ccdf(d, t)`. It
reads the survival of any composed delay through the verbs: a leaf, a censoring
wrapper, a [`Modified`](@ref) hazard modification, a [`Convolved`](@ref) sum, or
a univariate composer. For a [`Sequential`](@ref) chain it is the survival of
the marginal total-time convolution.

This matches `SurvivalDistributions.ccdf` and is the ``S`` in the hazard
identities used by [`hazard`](@ref), [`loghazard`](@ref) and
[`cumhazard`](@ref).

# Arguments
- `d`: the composed delay.
- `t`: the time at which to evaluate the survival.

# Examples
```@example
using CensoredDistributions, Distributions

d = sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
CensoredDistributions.survival(d, 3.0)
```

# See also
- [`hazard`](@ref), [`cumhazard`](@ref), [`loghazard`](@ref)
"""
survival(d, t::Real) = ccdf(_hazard_marginal(d), t)

@doc raw"""

Cumulative hazard ``H(t) = -\log S(t)`` of a composed delay.

`cumhazard(d, t)` integrates the hazard up to `t`. It is computed in log space
as `-logccdf(d, t)` (not `-log(survival(d, t))`), so it stays finite and
AD-stable into the tail where `S(t) → 0`. For a [`Sequential`](@ref) chain it is
the cumulative hazard of the marginal total-time convolution.

This matches `SurvivalDistributions.cumhazard`.

# Arguments
- `d`: the composed delay.
- `t`: the time at which to evaluate the cumulative hazard.

# Examples
```@example
using CensoredDistributions, Distributions

d = modify(LogNormal(1.5, 0.5), -log(2.0); link = log)
CensoredDistributions.cumhazard(d, 4.0)
```

# See also
- [`hazard`](@ref), [`loghazard`](@ref), [`survival`](@ref)
"""
cumhazard(d, t::Real) = -logccdf(_hazard_marginal(d), t)

@doc raw"""

Hazard ``h(t) = f(t) / S(t)`` of a composed delay.

`hazard(d, t)` is the instantaneous event rate at `t` given survival to `t`,
`pdf(d, t) / ccdf(d, t)`. It reads the hazard of any composed delay through the
verbs and so is the tree-level entry to a hazard-based downstream layer. For a
[`Sequential`](@ref) chain it is the hazard of the marginal total-time
convolution; for a [`Modified`](@ref) it returns the modified hazard
``g^{-1}(g(h) + \text{effect})`` that `modify` constructed.

This matches `SurvivalDistributions.hazard`, so a composed tree and a
SurvivalDistributions leaf expose the same hazard accessor.

# Arguments
- `d`: the composed delay.
- `t`: the time at which to evaluate the hazard.

# Examples
```@example
using CensoredDistributions, Distributions

# Hazard of a proportional-hazards modification matches g⁻¹(g(h) + β).
base = LogNormal(1.5, 0.5)
m = modify(base, log(2.0); link = log)
CensoredDistributions.hazard(m, 3.0)  # ≈ 2 * hazard(base, 3.0)
```

# See also
- [`loghazard`](@ref), [`cumhazard`](@ref), [`survival`](@ref)
- [`modify`](@ref): the hazard-modification verb this is consistent with
"""
hazard(d, t::Real) = exp(loghazard(d, t))

@doc raw"""

Log hazard ``\log h(t) = \log f(t) - \log S(t)`` of a composed delay.

`loghazard(d, t)` is computed in log space as `logpdf(d, t) - logccdf(d, t)`
(not `log(hazard(d, t))`) so it stays finite and AD-stable into the tail where
`S(t) → 0`. For a [`Sequential`](@ref) chain it is the log hazard of the
marginal total-time convolution.

This matches `SurvivalDistributions.loghazard`.

# Arguments
- `d`: the composed delay.
- `t`: the time at which to evaluate the log hazard.

# Examples
```@example
using CensoredDistributions, Distributions

d = sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
CensoredDistributions.loghazard(d, 3.0)
```

# See also
- [`hazard`](@ref), [`cumhazard`](@ref), [`survival`](@ref)
"""
function loghazard(d, t::Real)
    m = _hazard_marginal(d)
    return logpdf(m, t) - logccdf(m, t)
end

# ---------------------------------------------------------------------------
# Marginal univariate time-to-event of a composed object
# ---------------------------------------------------------------------------
#
# The accessors read the hazard surface of a univariate delay. A univariate
# composed object (a leaf, a censoring wrapper, a `Modified`, a `Convolved`, an
# `AbstractOneOf`) is its own marginal, so it flows straight through. The
# multivariate verb composers reduce to their marginal time-to-event here so the
# hazard of the tree is the hazard of that marginal.

# A univariate delay is its own marginal time-to-event.
_hazard_marginal(d::UnivariateDistribution) = d

# A `Sequential` chain's total time E_k - E_0 is the sum of its step delays, so
# the marginal is the `Convolved` of the step delays as given. Each step's own
# marginal is convolved (recursing `_hazard_marginal` so a nested composer step
# contributes its marginal time-to-event), not its stripped `_marginal_core`:
# that helper drops a `Modified` step's hazard modification (and any other
# wrapper) down to the bare base, which is right for the latent-edge marginal
# scorer but would silently ignore the modification in the chain's hazard.
# `convolved` consumes a `Modified` / censored step directly, so
# the step law flows through intact. A single-step chain is just that step's
# marginal.
function _hazard_marginal(d::Sequential)
    marginals = map(_hazard_marginal, d.components)
    length(marginals) == 1 && return marginals[1]
    return convolved(collect(marginals))
end

# A `Parallel` set has no single canonical univariate time-to-event: its
# branches share an origin but resolve at separate, separately-observed times.
# The hazard needs the user to name the marginal they mean (the min is a
# `compete`, a single branch is that branch), so refuse rather than pick one.
function _hazard_marginal(d::Parallel)
    throw(ArgumentError(
        "hazard of a Parallel is ambiguous: its branches have no single " *
        "time-to-event. Take the hazard of the branch you mean " *
        "(`hazard(d.components[i], t)`), or race the branches with " *
        "`compete(...)` for the time-to-first-event hazard."))
end
