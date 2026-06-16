# ============================================================================
# Competing: a generic disjunctive composer over plain distributions
# ============================================================================
#
# `Competing(:a => (d, p), :b => (d, p), ...)` composes any
# `UnivariateDistribution`s into competing outcomes: exactly ONE outcome occurs,
# governed by branch probabilities that sum to one. It lowers to a
# `Distributions.MixtureModel` over the outcome delays weighted by those
# probabilities, so the realisation is a single time (the marginal
# time-to-resolution). Because it stays univariate it nests inside
# [`Sequential`](@ref) / [`Parallel`](@ref) as an ordinary child. This layer adds
# NO censored-internal behaviour: the generic composition only.

# ----------------------------------------------------------------------------
# AbstractCompeting: the shared supertype for the competing-outcome composers
# ----------------------------------------------------------------------------
#
# Two competing-outcome composers share one event-tree behaviour and so one
# supertype: the fixed-probability MIXTURE [`Competing`](@ref) (cause and timing
# independent) and the racing-hazard [`HazardCompeting`](@ref) (the branch
# probability DERIVED from the hazards, timing coupled). The shared `competing(
# ...)` constructor builds the right one: a `HazardCompeting` when NO branch
# probabilities are given, a `Competing` when they are. The tree walkers
# (`tree_events.jl` name layout, `nesting.jl` `_event_child_nleaves`,
# `censored_specialisations.jl` scoring / `rand`, the introspection) dispatch on
# `AbstractCompeting` wherever the behaviour is shared (one event slot per
# outcome, the shared origin, the per-outcome `rand`), and on the concrete type
# only where the SCORING arithmetic differs (mixture weight vs hazard survival).
abstract type AbstractCompeting <: UnivariateDistribution{Continuous} end

# Outcome names, one per competing outcome. Both concrete types store `names`.
component_names(c::AbstractCompeting) = c.names
_n_branches(c::AbstractCompeting) = length(c.names)

@doc "

Marker distribution for a NO-EVENT (absorbing) outcome of a [`competing`](@ref)
node: the outcome where *nothing happens* and no event time is written.

A `none => (NoEvent(), q)` branch carries no delay; its mass `q` is the
probability that no event occurs. On `rand` a no-event win yields `missing` (no
time recorded). On `logpdf` an OBSERVED non-occurrence (an explicit `no event by
the horizon` record) scores the survival term `log q` (mixture) or the
racing-hazard survival `∏ S_k`; a latent non-occurrence (a record whose no-event
slot is simply missing) contributes no competing term.

`NoEvent` is a degenerate placeholder, not a sampling distribution: it has no
support and errors if asked for a density or a draw. It exists only to MARK the
absorbing branch so the competing node carries its mass `q`.

# See also
- [`competing`](@ref): the shared constructor.
- [`Competing`](@ref): the mixture competing node.
"
struct NoEvent <: UnivariateDistribution{Continuous} end

# `NoEvent` is a marker, not a sampleable density: every density / draw / support
# query errors with a clear message so a stray use surfaces immediately rather
# than silently scoring a degenerate term. The competing scorers special-case the
# no-event branch (its mass is a survival term, never `logpdf(NoEvent(), .)`).
function _no_event_error()
    throw(ArgumentError(
        "NoEvent is a marker for a competing no-event branch and has no density or " *
        "support; its mass is scored as a survival term by the competing node"))
end
logpdf(::NoEvent, ::Real) = _no_event_error()
pdf(::NoEvent, ::Real) = _no_event_error()
cdf(::NoEvent, ::Real) = _no_event_error()
Base.minimum(::NoEvent) = _no_event_error()
Base.maximum(::NoEvent) = _no_event_error()
Base.rand(::AbstractRNG, ::NoEvent) = _no_event_error()
# An EMPTY param tuple (no free parameters): the marker carries no delay, so the
# tree's `_param_eltype` / `_tree_core_eltype` promotions skip it cleanly.
params(::NoEvent) = ()

# Whether an outcome's delay payload is the no-event marker. Used by the scorers
# and the tree walkers to skip a no-event slot's density and treat its mass as a
# survival term.
_is_no_event(::NoEvent) = true
_is_no_event(::Any) = false

# Whether a competing node carries a no-event branch (one of its delays is the
# marker). Such a node is a DEFECTIVE marginal (the observed-time mass is `< 1`),
# so its scalar `logpdf` / `mean` / `as_mixture` error: it is multivariate /
# sub-stochastic, scored only through the event-vector path.
_has_no_event(c::AbstractCompeting) = any(_is_no_event, c.delays)

# `_is_composer_outcome` / `_is_nonterminal` (the non-terminal competing predicate,
# #466 Feature 3) reference `Sequential` / `Parallel` / `Select`, which are loaded
# AFTER this file, so they are defined in `nesting.jl` (loaded once all composer
# types exist) rather than here.

@doc "

Competing outcomes composed from any univariate distributions: exactly one of
several outcomes occurs, governed by branch probabilities summing to one.

`Competing` names each competing outcome, its delay distribution, and the branch
probability of that outcome. It lowers to a `Distributions.MixtureModel`
(see [`as_mixture`](@ref)) over the outcome delays weighted by the branch
probabilities, so the realisation is a single time and the type is univariate.
A death-versus-recovery competition makes the death branch probability the
case-fatality ratio.

Being univariate, a `Competing` nests as a child of [`Sequential`](@ref) or
[`Parallel`](@ref). This is the plain generic composition; per-record outcome
selection and censoring are not part of this type.

# Fields
- `names`: tuple of the competing outcome names (`Symbol`s).
- `delays`: tuple of the competing outcome delay distributions.
- `branch_probs`: tuple of the branch probabilities, summing to one.

# See also
- [`as_mixture`](@ref): the `MixtureModel` lowering
- [`Sequential`](@ref): a chain of additive steps
- [`Parallel`](@ref): independent branches
"
struct Competing{C <: Tuple, D <: Tuple, P <: Tuple} <: AbstractCompeting
    "Tuple of the competing outcome names (`Symbol`s)."
    names::C
    "Tuple of the competing outcome delay distributions."
    delays::D
    "Tuple of the branch probabilities, summing to one."
    branch_probs::P

    # Validate the structural invariants in the INNER constructor so EVERY
    # construction path (the `Pair...` outer constructor, equality round-trips,
    # `update` value- and node-edits, and direct struct calls) is checked,
    # rather than silently building a malformed node whose failure only surfaces
    # later as a confusing `DomainError` from `Categorical` inside `as_mixture`.
    #
    # The bounds (each prob in `[0, 1]`) and structure (at least two outcomes;
    # names, delays and branch_probs of equal length) hold on EVERY path,
    # including the DynamicPPL extension that rebuilds a `Competing` from branch
    # probabilities sampled INDEPENDENTLY from priors. Those sampled probs are
    # in `[0, 1]` but need NOT sum to one (the AD-safe `_competing_logmix`
    # scorer handles an unnormalised weight set), so the sum-to-one requirement
    # is enforced at the user-facing `Pair...` constructor and at `as_mixture`
    # (which DOES need a normalised `Categorical`), not here.
    function Competing(names::C, delays::D, branch_probs::P) where {
            C <: Tuple, D <: Tuple, P <: Tuple}
        length(names) >= 2 ||
            throw(ArgumentError("Competing needs at least two outcomes"))
        (length(names) == length(delays) == length(branch_probs)) ||
            throw(ArgumentError(
                "Competing names, delays and branch_probs must have equal " *
                "length; got $(length(names)), $(length(delays)), " *
                "$(length(branch_probs))"))
        _validate_branch_prob_bounds(branch_probs)
        return new{C, D, P}(names, delays, branch_probs)
    end
end

@doc "

Build a [`Competing`](@ref) node from `name => (delay, branch_prob)` outcomes.

Each outcome is `name => (delay, branch_prob)`: the outcome name (a `Symbol`),
its delay distribution, and the probability that this outcome occurs. The branch
probabilities must each lie in ``[0, 1]`` and sum to one, and at least two
outcomes are required.

# Examples
```@example
using CensoredDistributions, Distributions

cfr = 0.3
node = Competing(:death => (Gamma(1.5, 1.0), cfr),
    :disch => (Gamma(2.0, 1.5), 1 - cfr))
mean(node)
```

# See also
- [`Competing`](@ref): the composer type
- [`as_mixture`](@ref): the `MixtureModel` lowering
"
function Competing(outcomes::Pair...)
    length(outcomes) >= 2 ||
        throw(ArgumentError("Competing needs at least two outcomes"))
    names = Tuple(o.first for o in outcomes)
    payloads = Tuple(o.second for o in outcomes)
    all(n -> n isa Symbol, names) ||
        throw(ArgumentError("each competing outcome name must be a Symbol"))
    delays = Tuple(_competing_delay(p) for p in payloads)
    branch_probs = Tuple(_competing_prob(p) for p in payloads)
    # The inner constructor validates the bounds and structure; the user-facing
    # constructor additionally requires the probabilities to sum to one.
    _validate_branch_probs_sum(branch_probs)
    return Competing(names, delays, branch_probs)
end

@doc "

Build a [`Competing`](@ref) node from `name => (delay, branch_prob)` outcomes.

Lowercase sugar mirroring [`primary_censored`](@ref) / [`interval_censored`](@ref):
a thin wrapper over the [`Competing`](@ref) struct constructor. Each outcome is
`name => (delay, branch_prob)`; the branch probabilities must each lie in
``[0, 1]`` and sum to one, and at least two outcomes are required.

The LAST outcome's probability may be OMITTED (a bare `name => delay`): it then
takes the residual `1 - sum(of the others)`, so a probability that is fully
determined by the rest need not be written out (and cannot disagree with them).
The leading probabilities must sum to at most one. Omitting any outcome but the
last, or more than one, is rejected. Omitting EVERY probability instead builds a
racing-hazard [`HazardCompeting`](@ref).

# Arguments
- `outcomes`: two or more `name => (delay, branch_prob)` pairs, each giving the
  outcome name (a `Symbol`), its delay distribution, and the probability that
  the outcome occurs. The last pair's probability may be omitted (a bare
  `name => delay`), taking the residual `1 - sum(of the others)`.

# Examples
```@example
using CensoredDistributions, Distributions

cfr = 0.3
node = competing(:death => (Gamma(1.5, 1.0), cfr),
    :disch => (Gamma(2.0, 1.5), 1 - cfr))
mean(node)
```

```@example
using CensoredDistributions, Distributions

# The discharge probability is the residual `1 - cfr`, so it is omitted.
cfr = 0.3
node = competing(:death => (Gamma(1.5, 1.0), cfr),
    :disch => Gamma(2.0, 1.5))
mean(node)
```

# See also
- [`Competing`](@ref): the composer type
- [`as_mixture`](@ref): the `MixtureModel` lowering
- [`compose`](@ref): the front-end that nests a `Competing` as a branch
- [`Sequential`](@ref), [`Parallel`](@ref): the sibling composers
"
function competing(outcomes::Pair...)
    length(outcomes) >= 2 ||
        throw(ArgumentError("competing needs at least two outcomes"))
    payloads = Tuple(o.second for o in outcomes)
    # The SHARED constructor builds the right type from the payload SHAPE: a
    # `(delay, branch_prob)` tuple per outcome gives the fixed-probability
    # MIXTURE `Competing` (cause and timing independent); a BARE delay per
    # outcome (no branch probability) gives the racing-hazard `HazardCompeting`
    # (the winning probability is DERIVED from the hazards, timing coupled).
    # Mixing the two shapes is rejected EXCEPT the RESIDUAL form (every outcome
    # but the LAST carries a probability, the last is bare): the last outcome
    # then takes the residual `1 - sum(of the others)`, so the user need not
    # write a probability they can derive (and that must agree with the rest).
    if all(_is_prob_payload, payloads)
        return Competing(outcomes...)
    elseif all(_is_bare_payload, payloads)
        return HazardCompeting(outcomes...)
    elseif _is_residual_shape(payloads)
        return Competing(_fill_residual_outcome(outcomes)...)
    end
    throw(ArgumentError(
        "competing outcomes must ALL carry a branch probability " *
        "(`name => (delay, prob)`, the fixed-probability mixture) OR ALL " *
        "omit it (`name => delay`, the racing-hazard node), or carry a " *
        "probability on every outcome BUT the last (`name => delay`), which " *
        "then takes the residual `1 - sum(others)`; the given mix is unclear"))
end

# The RESIDUAL mixture shape: every outcome but the LAST carries a `(delay,
# prob)` pair and the last is a BARE delay, so the last outcome's probability is
# the residual `1 - sum(of the others)`. A bare delay ANYWHERE BUT the last (or
# more than one bare delay) is NOT the residual form: it falls through to the
# ambiguous-mix error so a misplaced omission is rejected clearly rather than
# silently treated as a hazard or a residual. At least two outcomes hold by the
# caller's guard, so the leading prob-payload set is non-empty.
function _is_residual_shape(payloads::Tuple)
    n = length(payloads)
    all(_is_prob_payload, Base.front(payloads)) &&
        _is_bare_payload(payloads[n])
end

# Resolve a residual-shape outcome list into the all-explicit `(delay, prob)`
# form: keep every leading `(delay, prob)` outcome and give the last (bare
# delay) outcome the residual probability `1 - sum(of the others)`. The residual
# rides the leading probabilities' element type, so a `logistic(Xβ)`/sampled
# leading prob keeps the last outcome a DIFFERENTIABLE function of the others
# (the residual flows the same `Dual`/tracked type). The leading probabilities
# are bounds- and sum-checked (each in `[0, 1]`, summing to `<= 1`) so the
# residual is a valid probability; an over-one leading set errors clearly here
# rather than yielding a negative residual the downstream bounds check catches.
function _fill_residual_outcome(outcomes::Tuple)
    n = length(outcomes)
    leading = Base.front(outcomes)
    last_pair = outcomes[n]
    leading_probs = Tuple(_competing_prob(o.second) for o in leading)
    _validate_branch_prob_bounds(leading_probs)
    total = sum(leading_probs)
    (total <= 1 + 1e-6) || throw(ArgumentError(
        "the leading competing branch probabilities sum to $total, " *
        "exceeding one, so the residual last-outcome probability `1 - sum` " *
        "is negative; the probabilities must leave non-negative residual mass"))
    residual = one(total) - total
    filled_last = last_pair.first => (last_pair.second, residual)
    return (leading..., filled_last)
end

# A `(delay, branch_prob)` mixture payload vs a bare-delay hazard payload. A
# competing OUTCOME delay may be a plain univariate leaf OR a composer SUBTREE
# (`Sequential` / `Parallel` / `Select` / nested `Competing`, the non-terminal
# branch of #466 Feature 3); `_is_competing_branch` (defined in `nesting.jl`, once
# those types exist) is the runtime admit-check, so the predicates stay value-based
# rather than referencing the later-loaded composer types in their signatures. A
# `NoEvent` marker is admitted only in the mixture (it carries the no-event mass
# `q`); a bare `NoEvent` in a hazard node has no hazard and is rejected by the
# `HazardCompeting` constructor.
_is_prob_payload(p::Tuple{Any, <:Real}) = _is_competing_branch(p[1])
_is_prob_payload(::Any) = false
_is_bare_payload(x) = _is_competing_branch(x)

function _competing_delay(payload::Tuple{Any, <:Real})
    _is_competing_branch(payload[1]) || throw(ArgumentError(
        "each competing outcome payload must be a `(delay, branch_prob)` tuple " *
        "whose delay is a univariate distribution or a composer subtree; got " *
        "$(typeof(payload[1]))"))
    return payload[1]
end
function _competing_delay(payload)
    throw(ArgumentError(
        "each competing outcome payload must be a `(delay, branch_prob)` " *
        "tuple; got $(typeof(payload))"))
end

_competing_prob(payload::Tuple{Any, <:Real}) = payload[2]

# Each branch probability must lie in `[0, 1]`. The bounds carry a small
# tolerance so a saturating covariate prob (`logistic(Xβ)` evaluating to a hair
# past 0 or 1 under AD/sampling) is accepted rather than spuriously rejected.
# Comparisons are value-based, so an AD `Dual`/tracked `branch_probs` is compared
# on its value WITHOUT being stripped of its derivative information.
function _validate_branch_prob_bounds(branch_probs::Tuple)
    tol = 1e-6
    for p in branch_probs
        (p >= -tol && p <= 1 + tol) ||
            throw(ArgumentError(
                "each branch probability must lie in [0, 1]; got $p"))
    end
    return nothing
end

# The branch probabilities must additionally sum to one. Applied on the
# user-facing `Pair...` constructor and at `as_mixture` (which lowers to a
# `Categorical` and so needs a normalised weight set), but NOT in the inner
# constructor, where a prior-sampled (unnormalised) weight set is legitimate.
# The `isapprox` sum check is value-based, so an AD `Dual` is not stripped.
function _validate_branch_probs_sum(branch_probs::Tuple)
    total = sum(branch_probs)
    isapprox(total, 1; atol = 1e-6) ||
        throw(ArgumentError(
            "competing branch probabilities sum to $total, not one"))
    return nothing
end

# ---------------------------------------------------------------------------
# Shared self-dispatch scoring
# ---------------------------------------------------------------------------
#
# The Turing-free arithmetic of the `Competing` self-dispatch (decision 2),
# factored here so BOTH the top-level `composed_distribution_model(d::Competing,
# row)` (the DynamicPPL extension) and the NESTED tree scorer use ONE
# implementation rather than two parallel copies. Each helper consumes already-
# resolved inputs (the observed-outcome index or `nothing`, the gap from the
# node's anchor, and the per-record branch probabilities) and returns a plain
# log density, so the extension supplies the row plumbing and these supply the
# scoring. The probabilities keep their (possibly AD `Dual`) element type so a
# covariate CFR `logistic(Xβ)` differentiates through the node.

# Per-record branch probabilities must each lie in `[0, 1]` and sum to one.
# Delegates to the same bounds and sum validators the stored `branch_probs` use
# (`_validate_branch_prob_bounds` then `_validate_branch_probs_sum`) so the
# per-record override and the node share ONE validation path; the tolerance and
# AD-`Dual`-preserving, value-based comparisons live in those helpers.
function _validate_record_probs(probs)
    _validate_branch_prob_bounds(probs)
    _validate_branch_probs_sum(probs)
    return nothing
end

# Coerce a per-record branch-probability override to the node's outcome order. A
# `NamedTuple` must name exactly the outcomes; a scalar is the FIRST outcome's
# probability of a two-outcome node (`(p, 1 - p)`). The element type is preserved
# so a `logistic(Xβ)` `Dual` flows through.
function _coerce_branch_probs(c::Competing, bp::NamedTuple)
    Set(keys(bp)) == Set(c.names) || throw(ArgumentError(
        "per-record branch_probs must name exactly the outcomes " *
        "$(collect(c.names)); got $(collect(keys(bp)))"))
    probs = map(n -> bp[n], c.names)
    _validate_record_probs(probs)
    return probs
end

function _coerce_branch_probs(c::Competing, p::Real)
    length(c.names) == 2 || throw(ArgumentError(
        "a scalar per-record branch_probs is only defined for a two-outcome " *
        "Competing (the first outcome's probability); node has " *
        "$(length(c.names)) outcomes, pass a NamedTuple instead"))
    probs = (p, one(p) - p)
    _validate_record_probs(probs)
    return probs
end

# Condition on the observed outcome `i`: `log(p[i]) + logpdf(delay[i], gap)`,
# the observed branch's own (censored) logpdf at its gap from the node's anchor.
# `delay` is optionally pre-censored/truncated by the caller (the same delay the
# top-level path scores), so this is purely the conditioned-branch arithmetic.
function _competing_condition_logpdf(probs, delay, gap, i::Int)
    return log(probs[i]) + logpdf(delay, gap)
end

# Marginalise an unknown outcome at a resolution time `t`: the branch-prob-
# weighted mixture log-density `log Σ_i p_i f_i(t)` via the log-sum-exp of
# `log p_i + logpdf(delay_i, t)`. `delays` may be pre-censored/truncated by the
# caller (matching the conditioned path's per-record horizon), else the node's
# delays.
#
# Computed directly, NOT via `MixtureModel(delays, float.(probs))`: `float.`
# strips an AD `Dual`/tracked type from the probabilities, breaking the gradient
# through a sampled / `logistic(Xβ)` branch probability on the MARGINALISED path.
# The explicit reduction keeps the probabilities' element type, so a `Dual`
# propagates exactly as it does on the conditioned path. A zero probability
# contributes no term (its `log` is `-Inf`); an all-zero set returns `-Inf`. A
# no-event branch carries no density at a finite observed time `t` (its mass is
# the survival, not a density term), so it contributes `-Inf` and is skipped
# here, leaving the marginal as the sum over the REAL outcomes.
function _competing_logmix(probs, delays, t)
    n = length(probs)
    terms = ntuple(
        i -> _is_no_event(delays[i]) ?
             oftype(log(probs[i]), -Inf) :
             log(probs[i]) + logpdf(delays[i], t), n)
    m = maximum(terms)
    isfinite(m) || return m
    s = zero(m)
    @inbounds for term in terms
        s += exp(term - m)
    end
    return m + log(s)
end

@doc "

Lower a [`Competing`](@ref) node to a `Distributions.MixtureModel`.

Returns the `MixtureModel` over the outcome delays weighted by the branch
probabilities, the marginal time-to-resolution regardless of which outcome
occurs.

# Examples
```@example
using CensoredDistributions, Distributions

node = Competing(:death => (Gamma(1.5, 1.0), 0.3),
    :disch => (Gamma(2.0, 1.5), 0.7))
as_mixture(node)
```

# See also
- [`Competing`](@ref): the composer type
"
function as_mixture(c::Competing)
    # A non-terminal node (a composer-valued outcome) is multivariate: no single
    # marginal time-to-resolution exists, so the scalar lowering is rejected.
    _is_nonterminal(c) && _nonterminal_marginal_error("as_mixture")
    # A no-event branch makes the OBSERVED-time mass `< 1` (a defective marginal),
    # so there is no proper `MixtureModel` over the observed delays: the node is
    # multivariate / sub-stochastic and is scored only through the event-vector
    # path. Reject the scalar lowering with a clear message.
    _has_no_event(c) && _no_event_marginal_error("as_mixture")
    # The `Categorical` inside the `MixtureModel` needs a normalised weight set,
    # so reject an unnormalised `Competing` (e.g. one built directly with
    # branch probabilities that do not sum to one) with a clear error rather
    # than the confusing `DomainError` `Categorical` would otherwise throw.
    _validate_branch_probs_sum(c.branch_probs)
    return MixtureModel(
        collect(c.delays), collect(float.(c.branch_probs)))
end

# A defective-marginal (no-event) competing node has no scalar `logpdf` / `mean`
# / `as_mixture`: its observed-time mass is `< 1`, so it is multivariate and
# scored only through the event-vector path. Errors with a clear message.
function _no_event_marginal_error(what::AbstractString)
    throw(ArgumentError(
        "a Competing node with a no-event branch is a defective marginal " *
        "(its observed-time mass is < 1) and has no scalar `$(what)`; score it " *
        "through the event-vector path (its observed-outcome / no-event record)"))
end

# A NON-TERMINAL (composer-outcome) competing node is MULTIVARIATE: an outcome's
# subtree spans several event slots, so there is no single marginal
# time-to-resolution and no scalar `logpdf` / `mean` / `as_mixture`. It is scored
# only through the event-vector path (its outcome subtree's slice), so the scalar
# methods error with a clear message pointing at the event-vector / NamedTuple
# path. (#466 Feature 3.) Defined AFTER `as_mixture` so the `@doc` block above
# `as_mixture` attaches to it (an intervening function definition would steal the
# docstring and leave `as_mixture` undocumented, an Aqua failure).
function _nonterminal_marginal_error(what::AbstractString)
    throw(ArgumentError(
        "a non-terminal Competing node (an outcome whose payload is a composer " *
        "subtree) is multivariate and has no scalar `$(what)`; score it through " *
        "the event-vector path (nest it in a `compose(...)` tree and pass the " *
        "outcome subtree's event slots, or use `event_names` / NamedTuple I/O)"))
end

params(c::Competing) = (map(params, c.delays), c.branch_probs)

# The univariate interface delegates to the mixture lowering, so a `Competing`
# behaves as the marginal time-to-resolution wherever a distribution is needed.
Base.minimum(c::Competing) = minimum(as_mixture(c))
Base.maximum(c::Competing) = maximum(as_mixture(c))
insupport(c::Competing, x::Real) = insupport(as_mixture(c), x)
mean(c::Competing) = mean(as_mixture(c))
var(c::Competing) = var(as_mixture(c))

@doc "

Log probability density of the competing-outcome marginal at `x`.

Routed through the AD-safe `_competing_logmix` reduction rather than
`logpdf(as_mixture(c), x)`: `as_mixture` does `float.(branch_probs)`, which
strips an AD `Dual`/tracked type from the branch probabilities, breaking the
gradient w.r.t. a covariate case-fatality term (`logistic(Xβ)`) when a
`Competing` is scored as a leaf of a plain (non-censored) `compose(...)` tree.
The explicit log-sum-exp keeps the probabilities' element type, so a `Dual`
propagates exactly as on the censored-tree scorer.

See also: [`as_mixture`](@ref)
"
function logpdf(c::Competing, x::Real)
    _is_nonterminal(c) && _nonterminal_marginal_error("logpdf")
    _has_no_event(c) && _no_event_marginal_error("logpdf")
    return _competing_logmix(c.branch_probs, c.delays, x)
end

@doc "

Probability density of the competing-outcome marginal at `x`.

`exp` of the AD-safe [`logpdf`](@ref), so branch-prob gradients survive (see the
`logpdf` note on why `as_mixture` is avoided on a differentiated path).

See also: [`logpdf`](@ref)
"
pdf(c::Competing, x::Real) = exp(logpdf(c, x))

@doc "

Cumulative distribution function of the competing-outcome marginal at `x`.

The branch-prob-weighted mixture cdf `Σ_i p_i F_i(x)`, summed directly so the
probabilities keep their (possibly AD `Dual`) element type rather than being
stripped by `as_mixture`'s `float.(branch_probs)`. This keeps the cdf AD-safe on
a differentiated path (e.g. a censored-survival term), matching `logpdf`.

See also: [`as_mixture`](@ref)
"
function cdf(c::Competing, x::Real)
    _is_nonterminal(c) && _nonterminal_marginal_error("cdf")
    _has_no_event(c) && _no_event_marginal_error("cdf")
    return sum(ntuple(
        i -> c.branch_probs[i] * cdf(c.delays[i], x), length(c.branch_probs)))
end

@doc "

Sample the competing-outcome marginal time-to-resolution.

See also: [`as_mixture`](@ref)
"
Base.rand(rng::AbstractRNG, c::Competing) = rand(rng, as_mixture(c))
Base.rand(c::Competing) = rand(default_rng(), c)

@doc "

Sample a competing outcome AND its time, returning `(name, time)`.

Unlike the univariate [`rand`](@ref) (the marginal time-to-resolution, which
discards which outcome occurred), this draws the resolved outcome from the branch
probabilities and the time from that outcome's own delay, so the chosen outcome
is retained. Used by the full-path tree simulation, where a `Competing` node
resolves to a single named outcome.

# Arguments
- `rng`: random number generator (the no-`rng` method uses the global default).
- `c`: the [`Competing`](@ref) node to sample an outcome from.

# Examples
```@example
using CensoredDistributions, Distributions, Random

node = Competing(:death => (Gamma(1.5, 1.0), 0.3),
    :disch => (Gamma(2.0, 1.5), 0.7))
name, time = rand_outcome(MersenneTwister(1), node)
```

See also: [`Competing`](@ref), [`rand`](@ref)
"
function rand_outcome(rng::AbstractRNG, c::Competing)
    i = _sample_branch(rng, c.branch_probs)
    # A no-event win yields `missing` (no event time recorded); a real outcome
    # draws its own delay.
    _is_no_event(c.delays[i]) && return c.names[i], missing
    return c.names[i], rand(rng, c.delays[i])
end

rand_outcome(c::Competing) = rand_outcome(default_rng(), c)

@doc "

Print a [`Competing`](@ref) node as a recursive indented tree, labelling each
outcome with its name and branch probability and descending into any nested
composer outcome so the whole structure is shown at once.

See also: [`Competing`](@ref)
"
function Base.show(io::IO, ::MIME"text/plain", c::Competing)
    _show_composer_tree(io, c)
    return nothing
end

function Base.show(io::IO, c::Competing)
    parts = ["$(c.names[k])@$(c.branch_probs[k])" for k in 1:_n_branches(c)]
    print(io, "Competing(", join(parts, " | "), ")")
    return nothing
end

# ============================================================================
# HazardCompeting: competing risks by racing hazards (dual to convolve)
# ============================================================================
#
# Where `convolve_distributions` SUMS independent delays (events in series, the
# total time through a chain), competing RISKS take the MINIMUM of racing latent
# delays (events compete, the first wins). `HazardCompeting` is that combinator:
# given cause-specific delays `D_1..D_n` it represents
#
#   - the marginal `any-event` time `T = min_k D_k`, a univariate with survival
#     `S(t) = ∏_k S_k(t)` and density `f(t) = ∑_j f_j(t) ∏_{k≠j} S_k(t)` (so it
#     nests as a leaf like a convolved chain), and
#   - the cause-resolved split (named outcomes) for the multivariate / event view:
#     observing `(cause j, time t)` scores `f_j(t) ∏_{k≠j} S_k(t)`.
#
# Unlike the mixture `Competing` (pick a branch by a FIXED probability, then draw
# its delay; cause and timing INDEPENDENT) the winning probability here is DERIVED
# from the hazards (`P(cause = j) = ∫ f_j ∏_{k≠j} S_k`) and timing is COUPLED.
#
# The three duals that MUST agree (the acceptance test):
#   - `rand`: draw a latent time per cause, return `(argmin cause, min time)`.
#   - `logpdf`: the competing-risks likelihood, marginal `log ∑_j f_j ∏_{k≠j} S_k`
#     or cause-resolved `log f_j + ∑_{k≠j} log S_k`.
#   - forward `convolve_distributions(stack, series)`: per-outcome sub-density
#     stream `series ⊛ pmf(f_j ∏_{k≠j} S_k)`, sub-stochastic (NOT renormalised).
#
# Ships against plain `Distributions.ccdf` / `logccdf` / `logpdf`, so a stock
# `Gamma`/`LogNormal` leaf AND a #470 SurvivalDistributions leaf both race. The
# logpdf is a log-sum-exp of `logpdf` + `logccdf` terms (AD-safe, no `float`
# stripping).

@doc "

Competing risks by racing hazards: the dual of [`convolve_distributions`](@ref)
under MINIMUM instead of sum.

Given cause-specific delay distributions `D_1, ..., D_n`, `HazardCompeting`
represents the first-event time `T = min_k D_k` together with which cause won.
The marginal `any-event` survival is `∏_k S_k(t)` and density
`∑_j f_j(t) ∏_{k≠j} S_k(t)`, so it nests as a univariate leaf. Observing a
resolved `(cause j, time t)` scores `f_j(t) ∏_{k≠j} S_k(t)`. The winning
probability of each cause is DERIVED from the hazards
(`P(cause = j) = ∫ f_j ∏_{k≠j} S_k`), NOT a free parameter — this is the key
difference from the fixed-probability mixture [`Competing`](@ref).

Build it with the shared [`competing`](@ref) constructor by giving BARE delays
(no branch probabilities): `competing(:death => D1, :recover => D2)`.

# Fields
- `names`: tuple of the competing outcome names (`Symbol`s).
- `delays`: tuple of the cause-specific delay distributions.

# See also
- [`competing`](@ref): the shared constructor (omit probabilities for this type).
- [`Competing`](@ref): the fixed-probability mixture sibling.
- [`winning_probabilities`](@ref): the derived per-cause winning probabilities.
- [`convolve_distributions`](@ref): the sum dual (events in series).
"
struct HazardCompeting{C <: Tuple, D <: Tuple} <: AbstractCompeting
    "Tuple of the competing outcome names (`Symbol`s)."
    names::C
    "Tuple of the cause-specific delay distributions."
    delays::D

    function HazardCompeting(names::C, delays::D) where {C <: Tuple, D <: Tuple}
        length(names) >= 2 ||
            throw(ArgumentError("HazardCompeting needs at least two outcomes"))
        length(names) == length(delays) || throw(ArgumentError(
            "HazardCompeting names and delays must have equal length; got " *
            "$(length(names)) and $(length(delays))"))
        all(n -> n isa Symbol, names) ||
            throw(ArgumentError("each competing outcome name must be a Symbol"))
        any(_is_no_event, delays) && throw(ArgumentError(
            "a racing-hazard competing node has no no-event branch: the " *
            "no-event probability is DERIVED as the survival ∏ S_k(horizon). " *
            "Use the fixed-probability `Competing` for an explicit no-event mass"))
        return new{C, D}(names, delays)
    end
end

@doc "

Build a racing-hazard [`HazardCompeting`](@ref) node from `name => delay`
outcomes (bare delays, NO branch probabilities).

Each outcome is `name => delay`. The winning probability of each cause is derived
from the hazards, so no branch probability is supplied (that is what selects this
type over the fixed-probability mixture [`Competing`](@ref)). At least two
outcomes are required.

# Examples
```@example
using CensoredDistributions, Distributions

node = HazardCompeting(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
winning_probabilities(node)
```

# See also
- [`competing`](@ref): the shared constructor.
- [`Competing`](@ref): the fixed-probability mixture sibling.
"
function HazardCompeting(outcomes::Pair...)
    length(outcomes) >= 2 ||
        throw(ArgumentError("HazardCompeting needs at least two outcomes"))
    names = Tuple(o.first for o in outcomes)
    delays = Tuple(o.second for o in outcomes)
    all(_is_competing_branch, delays) || throw(ArgumentError(
        "each racing-hazard outcome payload must be a bare delay distribution " *
        "or composer subtree (no branch probability); got a `(delay, prob)` " *
        "tuple? use `Competing`"))
    return HazardCompeting(names, delays)
end

params(c::HazardCompeting) = map(params, c.delays)

# The marginal any-event distribution `T = min_k D_k` is univariate: its survival
# is `∏_k S_k(t)` and its support is the intersection of the cause supports'
# lower bounds (the soonest any cause can fire) up to the largest cause maximum.
Base.minimum(c::HazardCompeting) = maximum(map(minimum, c.delays))
Base.maximum(c::HazardCompeting) = maximum(map(maximum, c.delays))
function insupport(c::HazardCompeting, x::Real)
    return minimum(c) <= x <= maximum(c)
end

# Survival of the marginal any-event time: `log ∏_k S_k(t) = Σ_k logccdf_k(t)`,
# summed directly so a `Dual`/tracked leaf param propagates (no `float` strip).
# Each term goes through `_logccdf_ad_safe` so a `Gamma` survival differentiates
# w.r.t. its shape/scale (the stock `logccdf(::Gamma)` has no `Dual`-shape rule).
function _hazard_logsurvival(c::HazardCompeting, t::Real)
    return sum(ntuple(k -> _logccdf_ad_safe(c.delays[k], t), _n_branches(c)))
end

# Cause-resolved log sub-density `log f_j(t) ∏_{k≠j} S_k(t) = log f_j(t) +
# Σ_{k≠j} logccdf_k(t)`, the likelihood term for an observed `(cause j, time t)`.
# Equivalently `logpdf_j(t) - logccdf_j(t) + Σ_k logccdf_k(t)` (the hazard form),
# but written as the explicit `≠ j` sum to avoid an `Inf - Inf` when a cause's
# survival underflows. AD-safe (`_logccdf_ad_safe` per term; the leaf params flow
# through).
function _hazard_cause_logpdf(c::HazardCompeting, j::Int, t::Real)
    n = _n_branches(c)
    return logpdf(c.delays[j], t) +
           sum(ntuple(
        k -> k == j ? zero(_logccdf_ad_safe(c.delays[k], t)) :
             _logccdf_ad_safe(c.delays[k], t), n))
end

@doc "

Log density of the racing-hazard marginal any-event time `T = min_k D_k`.

The marginal density is `∑_j f_j(t) ∏_{k≠j} S_k(t)`; this is its log via the
log-sum-exp of the cause-resolved sub-densities, AD-safe (the leaf params
propagate, no `float` stripping).

See also: [`HazardCompeting`](@ref), [`winning_probabilities`](@ref)
"
function logpdf(c::HazardCompeting, t::Real)
    _is_nonterminal(c) && _nonterminal_marginal_error("logpdf")
    n = _n_branches(c)
    terms = ntuple(j -> _hazard_cause_logpdf(c, j, t), n)
    m = maximum(terms)
    isfinite(m) || return m
    s = zero(m)
    @inbounds for term in terms
        s += exp(term - m)
    end
    return m + log(s)
end

pdf(c::HazardCompeting, t::Real) = exp(logpdf(c, t))

# Mean / variance of the marginal any-event time `T = min_k D_k`, by AD-safe
# fixed-node Gauss-Legendre quadrature of the survival `∏ S_k` (`E[T] = ∫ S(t)
# dt` for a non-negative `T`, `E[T²] = ∫ 2t S(t) dt`). A racing node's support
# floor may be positive; the integral runs from zero over the survival so the
# `E[T] = ∫ S` identity holds for a non-negative time.
function _hazard_marginal_window(c::HazardCompeting)
    hi = float(maximum(c))
    isfinite(hi) && return hi
    return _hazard_quad_window(c)
end

function mean(c::HazardCompeting)
    _is_nonterminal(c) && _nonterminal_marginal_error("mean")
    hi = _hazard_marginal_window(c)
    return gl_integrate(zero(hi), hi, _PRIMARY_GL) do t
        exp(_hazard_logsurvival(c, t))
    end
end

function var(c::HazardCompeting)
    _is_nonterminal(c) && _nonterminal_marginal_error("var")
    hi = _hazard_marginal_window(c)
    m = mean(c)
    e2 = gl_integrate(zero(hi), hi, _PRIMARY_GL) do t
        2 * t * exp(_hazard_logsurvival(c, t))
    end
    # Two independent quadratures can leave `e2 - m^2` a tiny negative for a
    # near-degenerate node; clamp to keep the variance non-negative.
    diff = e2 - m^2
    return max(zero(diff), diff)
end

@doc "

Survival of the racing-hazard marginal any-event time at `t`: `∏_k S_k(t)`.

See also: [`HazardCompeting`](@ref)
"
ccdf(c::HazardCompeting, t::Real) = exp(_hazard_logsurvival(c, t))
logccdf(c::HazardCompeting, t::Real) = _hazard_logsurvival(c, t)

# A racing-hazard node is a univariate leaf for the survival surface (#465 / the
# forward path): its AD-safe survival is just `_hazard_logsurvival`, so an outer
# `_logccdf_ad_safe`/`_ccdf_ad_safe` query (e.g. a parent racing node) recurses
# through the already-AD-safe terms rather than the stock `logccdf`.
_logccdf_ad_safe(c::HazardCompeting, t::Real) = _hazard_logsurvival(c, t)
_ccdf_ad_safe(c::HazardCompeting, t::Real) = exp(_hazard_logsurvival(c, t))
cdf(c::HazardCompeting, t::Real) = -expm1(_hazard_logsurvival(c, t))
function logcdf(c::HazardCompeting, t::Real)
    return log1mexp(_hazard_logsurvival(c, t))
end

@doc "

Sample the racing-hazard marginal any-event time `min_k D_k`.

See also: [`rand_outcome`](@ref) to retain WHICH cause won.
"
function Base.rand(rng::AbstractRNG, c::HazardCompeting)
    return rand_outcome(rng, c)[2]
end
Base.rand(c::HazardCompeting) = rand(default_rng(), c)

@doc "

Sample a racing-hazard outcome AND its time, returning `(name, time)`: draw a
latent time per cause and return the `argmin` cause with its `min` time.

This is the generative dual of the [`logpdf`](@ref) (`f_j ∏_{k≠j} S_k`) and of
the forward `convolve_distributions` stream: the Monte Carlo winning-cause
frequencies match the derived [`winning_probabilities`](@ref) and the forward
per-outcome stream masses.

# Arguments
- `rng`: random number generator (the no-`rng` method uses the global default).
- `c`: the [`HazardCompeting`](@ref) node to sample a winning cause from.

# Examples
```@example
using CensoredDistributions, Distributions, Random

node = competing(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
name, time = rand_outcome(MersenneTwister(1), node)
```

See also: [`HazardCompeting`](@ref), [`winning_probabilities`](@ref)
"
function rand_outcome(rng::AbstractRNG, c::HazardCompeting)
    n = _n_branches(c)
    best_i = 1
    best_t = rand(rng, c.delays[1])
    @inbounds for k in 2:n
        t = rand(rng, c.delays[k])
        if t < best_t
            best_t = t
            best_i = k
        end
    end
    return c.names[best_i], best_t
end
rand_outcome(c::HazardCompeting) = rand_outcome(default_rng(), c)

@doc "

The DERIVED per-cause winning probabilities of a racing-hazard
[`HazardCompeting`](@ref) node: `P(cause = j) = ∫ f_j(t) ∏_{k≠j} S_k(t) dt`,
returned as a `NamedTuple` keyed by the outcome names.

Computed by AD-safe fixed-node Gauss-Legendre quadrature of the cause-resolved
sub-density over the marginal support. The probabilities are sub-stochastic-free
(they sum to one for proper, eventually-certain causes); a node whose causes can
leave residual survival at `+∞` (a defective cause) sums to less than one, the
deficit being the never-resolved mass.

# Examples
```@example
using CensoredDistributions, Distributions

node = competing(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
winning_probabilities(node)
```

See also: [`HazardCompeting`](@ref), [`occurrence_probability`](@ref)
"
function winning_probabilities(c::HazardCompeting)
    _is_nonterminal(c) && _nonterminal_marginal_error("winning_probabilities")
    lo = float(minimum(c))
    hi_raw = float(maximum(c))
    # Bind `hi` UNCONDITIONALLY (a ternary, not `isfinite(hi) || (hi = ...)`): the
    # short-circuit-assignment form leaves `hi` only conditionally assigned, and the
    # `ntuple` closure below that captures it then trips JET's `local variable hi is
    # not defined` (the closure cannot prove the assignment branch ran). An
    # unbounded cause support falls back to a finite high-quantile quad window.
    hi = isfinite(hi_raw) ? hi_raw : lo + _hazard_quad_window(c)
    n = _n_branches(c)
    probs = ntuple(n) do j
        gl_integrate(lo, hi, _PRIMARY_GL) do t
            exp(_hazard_cause_logpdf(c, j, t))
        end
    end
    return NamedTuple{c.names}(probs)
end

# A finite quadrature window for a cause with unbounded support: a high quantile
# of the soonest-firing marginal. Uses the largest cause `0.9999` quantile so the
# tail beyond it carries negligible mass for the winning-probability integral.
function _hazard_quad_window(c::HazardCompeting)
    return maximum(map(d -> quantile(d, 0.9999), c.delays))
end

@doc "

The probability that ANY (non-no-event) outcome occurs for a competing node.

For a racing-hazard [`HazardCompeting`](@ref) node `occurrence_probability` is the
sum of the derived [`winning_probabilities`](@ref) (one for proper,
eventually-certain causes; the resolved mass for a defective node). For a
fixed-probability [`Competing`](@ref) node it is one minus the no-event branch
mass.

# Examples
```@example
using CensoredDistributions, Distributions

node = competing(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
occurrence_probability(node)
```

# See also
- [`winning_probabilities`](@ref): the per-outcome winning split.
"
function occurrence_probability(c::HazardCompeting)
    return sum(values(winning_probabilities(c)))
end

@doc "

The probability that the named outcome occurs for a fixed-probability
[`Competing`](@ref) node: its branch probability (the no-event branch's mass is
the non-occurrence probability), returned as a `NamedTuple`.

See also: [`occurrence_probability`](@ref)
"
function winning_probabilities(c::Competing)
    return NamedTuple{c.names}(c.branch_probs)
end

@doc "

The probability that ANY (non-no-event) outcome occurs for a fixed-probability
[`Competing`](@ref) node: one minus the no-event branch mass.

See also: [`winning_probabilities`](@ref)
"
function occurrence_probability(c::Competing)
    total = zero(float(eltype(c.branch_probs)))
    @inbounds for k in 1:_n_branches(c)
        _is_no_event(c.delays[k]) && continue
        total += c.branch_probs[k]
    end
    return total
end

# ----------------------------------------------------------------------------
# Cause-resolved sub-density leaf (for the forward convolve stream)
# ----------------------------------------------------------------------------
#
# The forward `convolve_distributions(stack, series)` per-outcome stream of a
# racing-hazard node is `series ⊛ pmf(f_j ∏_{k≠j} S_k)`, sub-stochastic: each
# outcome's mass equals its DERIVED winning probability, the deficit being the
# competing fraction. `_HazardCauseDelay` is the cause-resolved sub-density of one
# cause `j` of a [`HazardCompeting`](@ref) node as a (defective) univariate
# distribution: its `pdf` is `f_j(t) ∏_{k≠j} S_k(t)` and its `cdf` is that
# sub-density integrated from the support floor to `t` (the cause-`j` winning
# probability accumulated by time `t`). The convolve layer discretises it through
# `interval_censored`, so the resulting masses are EXACTLY the sub-stochastic
# per-outcome stream (no renormalise). AD-safe: the leaf params flow through the
# log-sum / quadrature.
struct _HazardCauseDelay{H <: HazardCompeting} <: UnivariateDistribution{Continuous}
    node::H
    cause::Int
end

Base.minimum(d::_HazardCauseDelay) = minimum(d.node)
Base.maximum(d::_HazardCauseDelay) = maximum(d.node)
function insupport(d::_HazardCauseDelay, x::Real)
    return insupport(d.node, x)
end
function logpdf(d::_HazardCauseDelay, t::Real)
    return _hazard_cause_logpdf(d.node, d.cause, t)
end
pdf(d::_HazardCauseDelay, t::Real) = exp(logpdf(d, t))

# The cause-`j` winning probability accumulated by `t`: `∫_lo^t f_j ∏_{k≠j} S_k`.
# Fixed-node Gauss-Legendre over the support floor to `t` (AD-safe; the leaf
# params flow through the integrand). A `t` at or below the floor has zero mass.
function cdf(d::_HazardCauseDelay, t::Real)
    lo = float(minimum(d.node))
    t <= lo && return zero(float(t))
    return gl_integrate(lo, float(t), _PRIMARY_GL) do u
        exp(_hazard_cause_logpdf(d.node, d.cause, u))
    end
end

@doc "

Print a [`HazardCompeting`](@ref) node as a recursive indented tree.

See also: [`HazardCompeting`](@ref)
"
function Base.show(io::IO, ::MIME"text/plain", c::HazardCompeting)
    _show_composer_tree(io, c)
    return nothing
end

function Base.show(io::IO, c::HazardCompeting)
    parts = ["$(c.names[k])~$(c.delays[k])" for k in 1:_n_branches(c)]
    print(io, "HazardCompeting(", join(parts, " | "), ")")
    return nothing
end
