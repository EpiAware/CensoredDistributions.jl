# ============================================================================
# Discrete-time reporting hazard with reference + report effects
# ============================================================================
#
# The epinowcast nowcasting model treats the reporting delay as a DISCRETE-TIME
# HAZARD, not a fixed PMF. The hazard `h_d` is the conditional probability that a
# case is reported at delay `d` GIVEN it has not yet been reported by delay
# `d - 1`. Writing the hazard on the logit scale lets covariate effects enter
# additively and lets the delay distribution VARY by reference date (a slow
# drift in reporting speed) and by report date (e.g. a weekday reporting pattern)
# while staying a proper, normalised PMF per reference date.
#
# This file supplies the small, AD-safe vector pieces that the renewal /
# convolution layer (`convolve_with_vector.jl`) and the censoring pipeline
# (`double_interval_censored`) do not: PMF -> hazard, a logit-scale modification
# of the hazard by additive effects, and hazard -> PMF. Composed, they map a
# baseline discretised delay PMF and per-(reference, report) logit effects to the
# modified per-reference-date delay PMF the convolution then uses.
#
# Relationship to the delay machinery. The BASELINE PMF is whatever the package
# already produces: `_delay_pmf(delay, maxlag, interval)` discretises a
# (primary-/double-censored) delay to interval masses on `0:maxlag`. This file
# only reshapes that PMF through the hazard, so a primary_censored /
# double_interval_censored baseline flows straight in.
#
# Why not a forward-transform op. The `Transformed`/`thin`/`cumulative` protocol
# (`forward_transform.jl`) applies a deterministic map to the 1-D count series
# AFTER `convolve_distributions(stack, series)` has collapsed the delay into a
# single, time-INVARIANT PMF convolved across all times. The nowcasting hazard is
# time-VARYING by construction: a reference-date effect gives each reference date
# its OWN delay PMF, so the result is a reference-by-report MATRIX, not a series a
# forward op can produce. The hazard layer therefore reuses `_delay_pmf` for the
# baseline PMF (the part the convolution layer already builds once) but forms the
# per-reference-date matrix itself.
#
# AD-safety. Every operation here is a plain arithmetic reduction over vectors
# (cumulative sums, `logistic`, products), seeded from the input element type, so
# `Dual`/tracked numbers propagate and the whole hazard layer differentiates
# inside a Turing `@model`.
#
# Relationship to ModifiedDistributions.jl. This is a hazard-MODIFICATION of a
# delay (a sibling concern to that package's remit). For now it lives here,
# next to the delay-discretisation and convolution it feeds; the
# hazard-modification leaves may move into ModifiedDistributions in future.

@doc raw"

Discrete-time reporting hazard of a delay PMF.

`delay_hazard(pmf)` converts a delay probability-mass vector `pmf` (the
probability of report at each delay ``d = 0, 1, \dots, D``) to the discrete-time
hazard vector ``h``,

```math
h_d = \frac{p_d}{1 - \sum_{d' < d} p_{d'}},
```

the conditional probability of report at delay ``d`` given no report by delay
``d - 1``. This is the epinowcast baseline hazard ``\gamma`` before any
reference- or report-date effects. The final hazard is clamped to one
(``h_D = 1``) so a PMF that does not sum to one over the truncated grid is
treated as fully reported by the maximum delay, matching the epinowcast
maximum-delay constraint.

The reduction is a cumulative sum and a divide, seeded from the input element
type, so `Dual`/tracked numbers propagate and the hazard differentiates under
AD.

# Arguments
- `pmf`: the delay PMF over the grid `0:D` (e.g. from `_delay_pmf` or a
  discretised [`double_interval_censored`](@ref) delay).

# Examples
```@example
using CensoredDistributions, Distributions

pmf = CensoredDistributions._delay_pmf(LogNormal(1.5, 0.5), 10, 1.0)
h = CensoredDistributions.delay_hazard(pmf)
```

# See also
- [`hazard_to_pmf`](@ref): the inverse map (hazard -> PMF)
- [`apply_hazard_effects`](@ref): add logit-scale reference / report effects
"
function delay_hazard(pmf::AbstractVector)
    n = length(pmf)
    T = eltype(pmf)
    h = zeros(T, n)
    surv = one(T)             # survival: 1 - Σ_{d' < d} p_{d'}
    @inbounds for d in 1:n
        denom = surv
        # A non-positive survival (numerically exhausted PMF) yields a saturated
        # hazard of one rather than a divide-by-zero NaN.
        h[d] = denom > 0 ? pmf[d] / denom : one(T)
        surv -= pmf[d]
    end
    # Enforce the maximum-delay constraint h_D = 1: everything not yet reported
    # is reported in the final bin, so the reconstructed PMF sums to one.
    h[n] = one(T)
    return h
end

@doc raw"

Delay PMF reconstructed from a discrete-time hazard.

`hazard_to_pmf(h)` is the inverse of [`delay_hazard`](@ref): given a hazard
vector ``h`` over delays ``0, 1, \dots, D`` it returns the report-probability
PMF,

```math
p_0 = h_0, \qquad p_d = \left(1 - \sum_{d' < d} p_{d'}\right) h_d,
```

the epinowcast reconstruction of the reporting probabilities from the hazard.
With ``h_D = 1`` the returned PMF sums to one.

The reduction carries a running survival term, seeded from the input element
type, so `Dual`/tracked hazards propagate under AD.

# Arguments
- `h`: the discrete-time hazard over the grid `0:D`, each entry in ``[0, 1]``.

# Examples
```@example
using CensoredDistributions, Distributions

pmf = CensoredDistributions._delay_pmf(LogNormal(1.5, 0.5), 10, 1.0)
h = CensoredDistributions.delay_hazard(pmf)
back = CensoredDistributions.hazard_to_pmf(h)  # ≈ pmf (renormalised)
```

# See also
- [`delay_hazard`](@ref): the forward map (PMF -> hazard)
- [`apply_hazard_effects`](@ref): add logit-scale reference / report effects
"
function hazard_to_pmf(h::AbstractVector)
    n = length(h)
    T = eltype(h)
    p = zeros(T, n)
    surv = one(T)
    @inbounds for d in 1:n
        p[d] = surv * h[d]
        surv -= p[d]
    end
    return p
end

@doc raw"

Modify a delay PMF through its discrete-time hazard with additive logit effects.

`apply_hazard_effects(pmf, effects)` reshapes a baseline delay `pmf` by adding
`effects` to its hazard on the logit scale and reconstructing the PMF,

```math
\operatorname{logit}(h^{*}_d) = \operatorname{logit}(h_d) + \eta_d, \qquad
h_d = \frac{p_d}{1 - \sum_{d' < d} p_{d'}},
```

then `p^{*} = ` [`hazard_to_pmf`](@ref)`(h^{*})`. This is the epinowcast
logit-hazard model: `effects` is the per-delay total ``\eta_d = \delta_d +
\epsilon_d`` of the reference-date effect ``\delta`` and the report-date effect
``\epsilon`` (e.g. a day-of-week term), so a positive effect speeds reporting at
that delay and the returned PMF is the modified per-reference-date delay
distribution. The maximum-delay hazard stays one, so ``p^{*}`` sums to one.

The map is `logit -> add -> logistic -> reconstruct`, all AD-safe arithmetic, so
`Dual`/tracked `effects` (e.g. sampled random-walk reference effects or
day-of-week coefficients) differentiate through.

# Arguments
- `pmf`: the baseline delay PMF over the grid `0:D`.
- `effects`: the additive logit-hazard effects ``\eta_d``, one per delay, same
  length as `pmf`. Use zeros for no modification.

# Examples
```@example
using CensoredDistributions, Distributions

pmf = CensoredDistributions._delay_pmf(LogNormal(1.5, 0.5), 10, 1.0)
# Slow reporting at short delays, faster at long delays.
effects = range(-0.5, 0.5; length = length(pmf))
modified = CensoredDistributions.apply_hazard_effects(pmf, effects)
```

# See also
- [`delay_hazard`](@ref), [`hazard_to_pmf`](@ref): the hazard <-> PMF maps
- [`reference_report_matrix`](@ref): the per-(reference, report) count matrix
"
function apply_hazard_effects(pmf::AbstractVector, effects::AbstractVector)
    return _apply_hazard_link(pmf, effects, LogitLink)
end

# Generic link version of `apply_hazard_effects`: reshape a baseline PMF by
# adding per-bin `effects` to its discrete-time hazard on the `link`'s scale,
# `h*_d = g⁻¹(g(h_d) + effect_d)`, then reconstruct. `apply_hazard_effects` is
# the logit-link case; `modify(::IntervalCensored; link)` reuses this for any
# link, including a user callable. The final-bin hazard is pinned to one (the
# maximum-delay constraint) so the reconstructed PMF stays normalised.
function _apply_hazard_link(
        pmf::AbstractVector, effects::AbstractVector, link)
    length(pmf) == length(effects) || throw(DimensionMismatch(
        "effects must have one entry per delay; got $(length(effects)) for a " *
        "PMF of length $(length(pmf))"))
    h = delay_hazard(pmf)
    n = length(h)
    T = promote_type(eltype(h), eltype(effects))
    hstar = Vector{T}(undef, n)
    @inbounds for d in 1:n
        if d == n
            hstar[d] = one(T)
        else
            hstar[d] = link.invlink(link.g(h[d]) + effects[d])
        end
    end
    return hazard_to_pmf(hstar)
end

# AD-safe logit / logistic. `_logit` clamps its argument away from the open-
# interval endpoints so a hazard that has saturated to exactly 0 or 1 (e.g. a
# numerically exhausted survival bin) maps to a large finite logit rather than
# ±Inf, keeping the downstream `+ effect` and gradient finite.
function _logit(p::Real)
    T = float(typeof(p))
    eps = T(1e-12)
    q = clamp(p, eps, one(T) - eps)
    return log(q) - log(one(T) - q)
end

_logistic(x::Real) = inv(one(x) + exp(-x))

@doc raw"

Expected counts by reference date and report delay, right-truncated at `now`.

`reference_report_matrix(expected, pmf; reference_effects, report_effect, now)`
forms the epinowcast expected-count matrix ``\mathbb{E}[n_{t,d}] = \lambda_t \,
p_{t,d}``, where ``\lambda_t`` is `expected[t]` (the expected final count for
reference date ``t``) and ``p_{t,d}`` is the delay PMF for reference date ``t``,
obtained from the baseline `pmf` modified through the discrete-time hazard by a
per-reference-date effect and a per-report-date effect
([`apply_hazard_effects`](@ref)).

The two effect families are the epinowcast reference- and report-date hazard
terms:

- `reference_effects[t]` is a SCALAR logit-hazard shift applied at every delay
  for reference date ``t`` (e.g. a random-walk drift in reporting speed). Pass a
  length-`length(expected)` vector, or `nothing` for no reference effect.
- `report_effect(s)` returns the logit-hazard shift for REPORT date ``s = t +
  d`` (e.g. a day-of-week term `dow -> dow_effect[mod1(s, 7)]`). Pass a callable,
  or `nothing` for no report effect. The report effect is evaluated at the report
  date, so it varies along each reference date's delay profile.

Right-truncation enters as `now`: only cells with report date ``t + d \le``
`now` are observed, so cells beyond `now` are set to zero (a record is included
iff its report date has occurred). Pass `now = nothing` to keep the full
untruncated matrix.

The result is a `length(expected) × length(pmf)` matrix `M[t, d]` of expected
counts for reference date `t` (row) and delay `d - 1` (column). Summing a row
over the observed delays gives the expected count SEEN so far for that reference
date; the row total over all delays is ``\lambda_t``.

Every step is AD-safe arithmetic, so sampled `expected`, `reference_effects` and
`report_effect` coefficients differentiate through.

# Arguments
- `expected`: expected final counts ``\lambda_t`` per reference date.
- `pmf`: the baseline delay PMF over the grid `0:D` (the maximum delay is
  `length(pmf) - 1`).

# Keyword Arguments
- `reference_effects`: per-reference-date scalar logit-hazard shifts, or
  `nothing`.
- `report_effect`: a callable `report_date -> logit-hazard shift`, or `nothing`.
- `now`: the real-time horizon; cells with `t + d > now` are zeroed. `nothing`
  keeps the full matrix.

# Examples
```@example
using CensoredDistributions, Distributions

expected = fill(100.0, 14)
pmf = CensoredDistributions._delay_pmf(LogNormal(1.2, 0.5), 7, 1.0)
M = CensoredDistributions.reference_report_matrix(expected, pmf;
    now = length(expected))
```

# See also
- [`apply_hazard_effects`](@ref): the per-reference-date hazard modification
- [`convolve_distributions`](@ref): the renewal observation layer
"
function reference_report_matrix(expected::AbstractVector,
        pmf::AbstractVector;
        reference_effects = nothing,
        report_effect = nothing,
        now = nothing)
    nt = length(expected)
    nd = length(pmf)
    T = promote_type(eltype(expected), eltype(pmf))
    T = reference_effects === nothing ? T :
        promote_type(T, eltype(reference_effects))
    M = zeros(T, nt, nd)
    @inbounds for t in 1:nt
        # Per-reference-date logit effect vector: a scalar reference shift at
        # every delay plus the per-report-date effect evaluated at report date
        # `t + (d - 1)`. Built per row so the report effect tracks the calendar
        # report date, not the delay.
        eta = _row_effects(T, nt, nd, t, reference_effects, report_effect)
        prow = apply_hazard_effects(pmf, eta)
        lambda = expected[t]
        for d in 1:nd
            report_date = t + (d - 1)
            if now === nothing || report_date <= now
                M[t, d] = lambda * prow[d]
            end
        end
    end
    return M
end

# Per-row logit-hazard effect vector for reference date `t`: the scalar
# reference effect (broadcast over delays) plus the report-date effect at each
# report date `t + (d - 1)`. Returns a zero vector when neither effect is set.
function _row_effects(::Type{T}, nt, nd, t, reference_effects,
        report_effect) where {T}
    eta = zeros(T, nd)
    ref = reference_effects === nothing ? zero(T) :
          T(reference_effects[t])
    @inbounds for d in 1:nd
        rep = report_effect === nothing ? zero(T) :
              T(report_effect(t + (d - 1)))
        eta[d] = ref + rep
    end
    return eta
end

@doc raw"""

Expected counts by reference date and report delay from a delay DISTRIBUTION.

This method takes the baseline delay as a `UnivariateDistribution` (typically a
composed [`compose`](@ref) / [`convolve_distributions`](@ref) stack) rather than a
pre-extracted PMF vector, so the reporting hazard modifies the composed
distribution itself. For each reference date the delay is discretised to a daily
PMF over `0:maxlag` and reshaped through a [`modify`](@ref)`(...; link = :logit)`
hazard modification, exactly the per-reference-date [`Modified`](@ref) leaf;
[`reference_report_matrix(expected, pmf; ...)`](@ref) is the equivalent
vector-input form.

# Arguments
- `expected`: expected final counts ``\lambda_t`` per reference date.
- `delay`: the baseline delay distribution (e.g. a composed stack).

# Keyword Arguments
- `maxlag`: the maximum delay; the PMF runs over `0:maxlag`.
- `interval`: the discretisation interval width (default `1`).
- `reference_effects`: per-reference-date scalar logit-hazard shifts, or
  `nothing`.
- `report_effect`: a callable `report_date -> logit-hazard shift`, or `nothing`.
- `now`: the real-time horizon; cells with `t + d > now` are zeroed.

# Examples
```@example
using CensoredDistributions, Distributions

expected = fill(100.0, 14)
delay = convolve_distributions(
    double_interval_censored(Gamma(1.8, 1.4); upper = 20.0, interval = 1.0),
    double_interval_censored(Gamma(1.5, 1.2); upper = 20.0, interval = 1.0))
M = CensoredDistributions.reference_report_matrix(expected, delay;
    maxlag = 7, now = length(expected))
```

# See also
- [`modify`](@ref): the per-reference-date hazard modification this calls.
- [`reference_report_matrix`](@ref): the PMF-vector form.
"""
function reference_report_matrix(expected::AbstractVector,
        delay::UnivariateDistribution;
        maxlag::Integer,
        interval = 1,
        reference_effects = nothing,
        report_effect = nothing,
        now = nothing)
    nt = length(expected)
    nd = maxlag + 1
    # Discretise the composed delay once to the daily grid; the per-reference
    # hazard modification then acts on this interval-censored distribution.
    ic = interval_censored(delay, interval)
    T = promote_type(eltype(expected), eltype(delay))
    T = reference_effects === nothing ? T :
        promote_type(T, eltype(reference_effects))
    M = zeros(T, nt, nd)
    @inbounds for t in 1:nt
        eta = _row_effects(T, nt, nd, t, reference_effects, report_effect)
        # `modify` on the discretised composed delay: the hazard modifies the
        # composed distribution itself, not a hand-extracted PMF vector. Read
        # the modified per-reference-date PMF straight off the `Modified` leaf.
        m = modify(ic, eta; link = :logit)
        lambda = expected[t]
        for d in 1:nd
            report_date = t + (d - 1)
            if now === nothing || report_date <= now
                M[t, d] = lambda * pdf(m, T((d - 1) * interval))
            end
        end
    end
    return M
end
