# ============================================================================
# Resolve: a generic disjunctive composer over plain distributions
# ============================================================================
#
# `Resolve(:a => (d, p), :b => (d, p), ...)` composes any
# `UnivariateDistribution`s into one_of outcomes: exactly ONE outcome occurs,
# governed by branch probabilities that sum to one. It lowers to a
# `Distributions.MixtureModel` over the outcome delays weighted by those
# probabilities, so the realisation is a single time (the marginal
# time-to-resolution). Because it stays univariate it nests inside
# [`Sequential`](@ref) / [`Parallel`](@ref) as an ordinary child. This layer adds
# NO censored-internal behaviour: the generic composition only.

# ----------------------------------------------------------------------------
# AbstractOneOf: the shared supertype for the one_of-outcome composers
# ----------------------------------------------------------------------------
#
# Two one_of-outcome composers share one event-tree behaviour and so one
# supertype: the fixed-probability MIXTURE [`Resolve`](@ref) (cause and timing
# independent) and the racing-hazard [`Compete`](@ref) (the branch
# probability DERIVED from the hazards, timing coupled). The shared `compete(
# ...)` constructor builds the right one: a `Compete` when NO branch
# probabilities are given, a `Resolve` when they are. The tree walkers
# (`tree_events.jl` name layout, `nesting.jl` `_event_child_nleaves`,
# `censored_specialisations.jl` scoring / `rand`, the introspection) dispatch on
# `AbstractOneOf` wherever the behaviour is shared (one event slot per
# outcome, the shared origin, the per-outcome `rand`), and on the concrete type
# only where the SCORING arithmetic differs (mixture weight vs hazard survival).
abstract type AbstractOneOf <: UnivariateDistribution{Continuous} end

# Outcome names, one per one_of outcome. Both concrete types store `names`.
component_names(c::AbstractOneOf) = c.names
_n_branches(c::AbstractOneOf) = length(c.names)

@doc "

Marker distribution for a NO-EVENT (absorbing) outcome of a [`resolve`](@ref)
node: the outcome where *nothing happens* and no event time is written.

A `none => (NoEvent(), q)` branch carries no delay; its mass `q` is the
probability that no event occurs. On `rand` a no-event win yields `missing` (no
time recorded). On `logpdf` an OBSERVED non-occurrence (an explicit `no event by
the horizon` record) scores the survival term `log q` (mixture) or the
racing-hazard survival `∏ S_k`; a latent non-occurrence (a record whose no-event
slot is simply missing) contributes no one_of term.

`NoEvent` is a degenerate placeholder, not a sampling distribution: it has no
support and errors if asked for a density or a draw. It exists only to MARK the
absorbing branch so the one_of node carries its mass `q`.

# See also
- [`resolve`](@ref): the fixed-probability constructor.
- [`Resolve`](@ref): the mixture one_of node.
"
struct NoEvent <: UnivariateDistribution{Continuous} end

# `NoEvent` is a marker, not a sampleable density: every density / draw / support
# query errors with a clear message so a stray use surfaces immediately rather
# than silently scoring a degenerate term. The one_of scorers special-case the
# no-event branch (its mass is a survival term, never `logpdf(NoEvent(), .)`).
function _no_event_error()
    throw(ArgumentError(
        "NoEvent is a marker for a one_of no-event branch and has no density or " *
        "support; its mass is scored as a survival term by the one_of node"))
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

# Whether a one_of node carries a no-event branch (one of its delays is the
# marker). Such a node is a DEFECTIVE marginal (the observed-time mass is `< 1`),
# so its scalar `logpdf` / `mean` / `as_mixture` error: it is multivariate /
# sub-stochastic, scored only through the event-vector path.
_has_no_event(c::AbstractOneOf) = any(_is_no_event, c.delays)

# `_is_composer_outcome` / `_is_nonterminal` (the non-terminal one_of predicate,
# #466 Feature 3) reference `Sequential` / `Parallel` / `Choose`, which are loaded
# AFTER this file, so they are defined in `nesting.jl` (loaded once all composer
# types exist) rather than here.

@doc "

Resolve outcomes composed from any univariate distributions: exactly one of
several outcomes occurs, governed by branch probabilities summing to one.

`Resolve` names each one_of outcome, its delay distribution, and the branch
probability of that outcome. It lowers to a `Distributions.MixtureModel`
(see [`as_mixture`](@ref)) over the outcome delays weighted by the branch
probabilities, so the realisation is a single time and the type is univariate.
A death-versus-recovery competition makes the death branch probability the
case-fatality ratio.

Being univariate, a `Resolve` nests as a child of [`Sequential`](@ref) or
[`Parallel`](@ref). This is the plain generic composition; per-record outcome
selection and censoring are not part of this type.

# Fields
- `names`: tuple of the one_of outcome names (`Symbol`s).
- `delays`: tuple of the one_of outcome delay distributions.
- `branch_probs`: tuple of the branch probabilities, summing to one.

# See also
- [`as_mixture`](@ref): the `MixtureModel` lowering
- [`Sequential`](@ref): a chain of additive steps
- [`Parallel`](@ref): independent branches
"
struct Resolve{C <: Tuple, D <: Tuple, P <: Tuple} <: AbstractOneOf
    "Tuple of the one_of outcome names (`Symbol`s)."
    names::C
    "Tuple of the one_of outcome delay distributions."
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
    # including the DynamicPPL extension that rebuilds a `Resolve` from branch
    # probabilities sampled INDEPENDENTLY from priors. Those sampled probs are
    # in `[0, 1]` but need NOT sum to one (the AD-safe `_one_of_logmix`
    # scorer handles an unnormalised weight set), so the sum-to-one requirement
    # is enforced at the user-facing `Pair...` constructor and at `as_mixture`
    # (which DOES need a normalised `Categorical`), not here.
    function Resolve(names::C, delays::D, branch_probs::P) where {
            C <: Tuple, D <: Tuple, P <: Tuple}
        length(names) >= 2 ||
            throw(ArgumentError("Resolve needs at least two outcomes"))
        (length(names) == length(delays) == length(branch_probs)) ||
            throw(ArgumentError(
                "Resolve names, delays and branch_probs must have equal " *
                "length; got $(length(names)), $(length(delays)), " *
                "$(length(branch_probs))"))
        _validate_branch_prob_bounds(branch_probs)
        return new{C, D, P}(names, delays, branch_probs)
    end
end

@doc "

Build a [`Resolve`](@ref) node from `name => (delay, branch_prob)` outcomes.

Each outcome is `name => (delay, branch_prob)`: the outcome name (a `Symbol`),
its delay distribution, and the probability that this outcome occurs. The branch
probabilities must each lie in ``[0, 1]`` and sum to one, and at least two
outcomes are required.

# Examples
```@example
using CensoredDistributions, Distributions

cfr = 0.3
node = Resolve(:death => (Gamma(1.5, 1.0), cfr),
    :disch => (Gamma(2.0, 1.5), 1 - cfr))
mean(node)
```

# See also
- [`Resolve`](@ref): the composer type
- [`as_mixture`](@ref): the `MixtureModel` lowering
"
function Resolve(outcomes::Pair...)
    length(outcomes) >= 2 ||
        throw(ArgumentError("Resolve needs at least two outcomes"))
    names = Tuple(o.first for o in outcomes)
    payloads = Tuple(o.second for o in outcomes)
    all(n -> n isa Symbol, names) ||
        throw(ArgumentError("each one_of outcome name must be a Symbol"))
    delays = Tuple(_one_of_delay(p) for p in payloads)
    branch_probs = Tuple(_one_of_prob(p) for p in payloads)
    # The inner constructor validates the bounds and structure; the user-facing
    # constructor additionally requires the probabilities to sum to one.
    _validate_branch_probs_sum(branch_probs)
    return Resolve(names, delays, branch_probs)
end

@doc "

Build a fixed-probability [`Resolve`](@ref) node from
`name => (delay, branch_prob)` outcomes: exactly one outcome RESOLVES, with cause
independent of timing.

Each outcome is `name => (delay, branch_prob)`; the branch probabilities must
each lie in ``[0, 1]`` and sum to one, and at least two outcomes are required.

The LAST outcome's probability may be OMITTED (a bare `name => delay`): it then
takes the residual `1 - sum(of the others)`, so a probability that is fully
determined by the rest need not be written out (and cannot disagree with them).
The leading probabilities must sum to at most one. Omitting any outcome but the
last, or more than one, is rejected. To omit EVERY probability (a racing-hazard
node where the winning probability is derived from the hazards) use
[`compete`](@ref) instead.

# Arguments
- `outcomes`: two or more `name => (delay, branch_prob)` pairs, each giving the
  outcome name (a `Symbol`), its delay distribution, and the probability that
  the outcome occurs. The last pair's probability may be omitted (a bare
  `name => delay`), taking the residual `1 - sum(of the others)`.

# Examples
```@example
using CensoredDistributions, Distributions

cfr = 0.3
node = resolve(:death => (Gamma(1.5, 1.0), cfr),
    :disch => (Gamma(2.0, 1.5), 1 - cfr))
mean(node)
```

```@example
using CensoredDistributions, Distributions

# The discharge probability is the residual `1 - cfr`, so it is omitted.
cfr = 0.3
node = resolve(:death => (Gamma(1.5, 1.0), cfr),
    :disch => Gamma(2.0, 1.5))
mean(node)
```

# See also
- [`Resolve`](@ref): the composer type
- [`compete`](@ref): the racing-hazard sibling constructor (bare delays)
- [`choose`](@ref): the data-selected disjunction sibling constructor
- [`as_mixture`](@ref): the `MixtureModel` lowering
- [`compose`](@ref): the front-end that nests a `Resolve` as a branch
- [`Sequential`](@ref), [`Parallel`](@ref): the sibling composers
"
function resolve(outcomes::Pair...)
    length(outcomes) >= 2 ||
        throw(ArgumentError("resolve needs at least two outcomes"))
    payloads = Tuple(o.second for o in outcomes)
    # `resolve` builds the fixed-probability MIXTURE `Resolve` (cause and timing
    # independent): every outcome carries a `(delay, branch_prob)` pair, OR every
    # outcome but the LAST does and the last is a bare delay taking the residual
    # `1 - sum(of the others)`. ALL-bare delays are the racing-hazard form and
    # belong to `compete`; a misplaced omission is rejected as an unclear mix.
    if all(_is_prob_payload, payloads)
        return Resolve(outcomes...)
    elseif _is_residual_shape(payloads)
        return Resolve(_fill_residual_outcome(outcomes)...)
    elseif all(_is_bare_payload, payloads)
        throw(ArgumentError(
            "`resolve` builds the fixed-probability split and needs a branch " *
            "probability on each outcome (`name => (delay, prob)`); every " *
            "outcome here is a bare delay. For the racing-hazard node (winning " *
            "probability derived from the hazards) use `compete` instead"))
    end
    throw(ArgumentError(
        "`resolve` outcomes must ALL carry a branch probability " *
        "(`name => (delay, prob)`), or carry one on every outcome BUT the " *
        "last (`name => delay`), which then takes the residual " *
        "`1 - sum(others)`; the given mix is unclear. For an all-bare-delay " *
        "racing-hazard node use `compete`"))
end

@doc "

Build a racing-hazard [`Compete`](@ref) node from bare `name => delay`
outcomes: the cause-specific delays RACE, the first wins, and the winning
probability of each cause is DERIVED from the hazards (cause coupled to timing).

Each outcome is `name => delay` (a bare delay, NO branch probability). At least
two outcomes are required. To give an explicit fixed probability per outcome (a
mixture where cause is independent of timing) use [`resolve`](@ref) instead.

# Arguments
- `outcomes`: two or more bare `name => delay` pairs, each giving the outcome
  name (a `Symbol`) and its cause-specific delay distribution (no branch
  probability).

# Examples
```@example
using CensoredDistributions, Distributions

node = compete(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
winning_probabilities(node)
```

# See also
- [`Compete`](@ref): the composer type
- [`resolve`](@ref): the fixed-probability sibling constructor (`(delay, prob)`)
- [`choose`](@ref): the data-selected disjunction sibling constructor
- [`winning_probabilities`](@ref): the derived per-cause winning probabilities
- [`compose`](@ref): the front-end that nests the node as a branch
"
function compete(outcomes::Pair...)
    length(outcomes) >= 2 ||
        throw(ArgumentError("compete needs at least two outcomes"))
    payloads = Tuple(o.second for o in outcomes)
    # `compete` builds the racing-hazard `Compete`: every outcome is a
    # BARE delay (no branch probability), the winning probability being derived
    # from the hazards. A `(delay, prob)` pair anywhere is the fixed-probability
    # mixture and belongs to `resolve`.
    if all(_is_bare_payload, payloads)
        return Compete(outcomes...)
    end
    throw(ArgumentError(
        "`compete` builds the racing-hazard node and needs a bare delay per " *
        "outcome (`name => delay`, no branch probability); a `(delay, prob)` " *
        "pair was given. For the fixed-probability split (an explicit per-" *
        "outcome probability) use `resolve` instead"))
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
    leading_probs = Tuple(_one_of_prob(o.second) for o in leading)
    _validate_branch_prob_bounds(leading_probs)
    total = sum(leading_probs)
    (total <= 1 + 1e-6) || throw(ArgumentError(
        "the leading one_of branch probabilities sum to $total, " *
        "exceeding one, so the residual last-outcome probability `1 - sum` " *
        "is negative; the probabilities must leave non-negative residual mass"))
    residual = one(total) - total
    filled_last = last_pair.first => (last_pair.second, residual)
    return (leading..., filled_last)
end

# A `(delay, branch_prob)` mixture payload vs a bare-delay hazard payload. A
# one_of OUTCOME delay may be a plain univariate leaf OR a composer SUBTREE
# (`Sequential` / `Parallel` / `Choose` / nested `Resolve`, the non-terminal
# branch of #466 Feature 3); `_is_one_of_branch` (defined in `nesting.jl`, once
# those types exist) is the runtime admit-check, so the predicates stay value-based
# rather than referencing the later-loaded composer types in their signatures. A
# `NoEvent` marker is admitted only in the mixture (it carries the no-event mass
# `q`); a bare `NoEvent` in a hazard node has no hazard and is rejected by the
# `Compete` constructor.
_is_prob_payload(p::Tuple{Any, <:Real}) = _is_one_of_branch(p[1])
_is_prob_payload(::Any) = false
_is_bare_payload(x) = _is_one_of_branch(x)

function _one_of_delay(payload::Tuple{Any, <:Real})
    _is_one_of_branch(payload[1]) || throw(ArgumentError(
        "each one_of outcome payload must be a `(delay, branch_prob)` tuple " *
        "whose delay is a univariate distribution or a composer subtree; got " *
        "$(typeof(payload[1]))"))
    return payload[1]
end
function _one_of_delay(payload)
    throw(ArgumentError(
        "each one_of outcome payload must be a `(delay, branch_prob)` " *
        "tuple; got $(typeof(payload))"))
end

_one_of_prob(payload::Tuple{Any, <:Real}) = payload[2]

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
            "one_of branch probabilities sum to $total, not one"))
    return nothing
end

# ---------------------------------------------------------------------------
# Shared self-dispatch scoring
# ---------------------------------------------------------------------------
#
# The Turing-free arithmetic of the `Resolve` self-dispatch (decision 2),
# factored here so BOTH the top-level `composed_distribution_model(d::Resolve,
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
function _coerce_branch_probs(c::Resolve, bp::NamedTuple)
    Set(keys(bp)) == Set(c.names) || throw(ArgumentError(
        "per-record branch_probs must name exactly the outcomes " *
        "$(collect(c.names)); got $(collect(keys(bp)))"))
    probs = map(n -> bp[n], c.names)
    _validate_record_probs(probs)
    return probs
end

function _coerce_branch_probs(c::Resolve, p::Real)
    length(c.names) == 2 || throw(ArgumentError(
        "a scalar per-record branch_probs is only defined for a two-outcome " *
        "Resolve (the first outcome's probability); node has " *
        "$(length(c.names)) outcomes, pass a NamedTuple instead"))
    probs = (p, one(p) - p)
    _validate_record_probs(probs)
    return probs
end

# Condition on the observed outcome `i`: `log(p[i]) + logpdf(delay[i], gap)`,
# the observed branch's own (censored) logpdf at its gap from the node's anchor.
# `delay` is optionally pre-censored/truncated by the caller (the same delay the
# top-level path scores), so this is purely the conditioned-branch arithmetic.
function _one_of_condition_logpdf(probs, delay, gap, i::Int)
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
function _one_of_logmix(probs, delays, t)
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

Lower a [`Resolve`](@ref) node to a `Distributions.MixtureModel`.

Returns the `MixtureModel` over the outcome delays weighted by the branch
probabilities, the marginal time-to-resolution regardless of which outcome
occurs.

# Examples
```@example
using CensoredDistributions, Distributions

node = Resolve(:death => (Gamma(1.5, 1.0), 0.3),
    :disch => (Gamma(2.0, 1.5), 0.7))
as_mixture(node)
```

# See also
- [`Resolve`](@ref): the composer type
"
function as_mixture(c::Resolve)
    # A non-terminal node (a composer-valued outcome) is multivariate: no single
    # marginal time-to-resolution exists, so the scalar lowering is rejected.
    _is_nonterminal(c) && _nonterminal_marginal_error("as_mixture")
    # A no-event branch makes the OBSERVED-time mass `< 1` (a defective marginal),
    # so there is no proper `MixtureModel` over the observed delays: the node is
    # multivariate / sub-stochastic and is scored only through the event-vector
    # path. Reject the scalar lowering with a clear message.
    _has_no_event(c) && _no_event_marginal_error("as_mixture")
    # The `Categorical` inside the `MixtureModel` needs a normalised weight set,
    # so reject an unnormalised `Resolve` (e.g. one built directly with
    # branch probabilities that do not sum to one) with a clear error rather
    # than the confusing `DomainError` `Categorical` would otherwise throw.
    _validate_branch_probs_sum(c.branch_probs)
    return MixtureModel(
        collect(c.delays), collect(float.(c.branch_probs)))
end

# A defective-marginal (no-event) one_of node has no scalar `logpdf` / `mean`
# / `as_mixture`: its observed-time mass is `< 1`, so it is multivariate and
# scored only through the event-vector path. Errors with a clear message.
function _no_event_marginal_error(what::AbstractString)
    throw(ArgumentError(
        "a Resolve node with a no-event branch is a defective marginal " *
        "(its observed-time mass is < 1) and has no scalar `$(what)`; score it " *
        "through the event-vector path (its observed-outcome / no-event record)"))
end

# A NON-TERMINAL (composer-outcome) one_of node is MULTIVARIATE: an outcome's
# subtree spans several event slots, so there is no single marginal
# time-to-resolution and no scalar `logpdf` / `mean` / `as_mixture`. It is scored
# only through the event-vector path (its outcome subtree's slice), so the scalar
# methods error with a clear message pointing at the event-vector / NamedTuple
# path. (#466 Feature 3.) Defined AFTER `as_mixture` so the `@doc` block above
# `as_mixture` attaches to it (an intervening function definition would steal the
# docstring and leave `as_mixture` undocumented, an Aqua failure).
function _nonterminal_marginal_error(what::AbstractString)
    throw(ArgumentError(
        "a non-terminal Resolve node (an outcome whose payload is a composer " *
        "subtree) is multivariate and has no scalar `$(what)`; score it through " *
        "the event-vector path (nest it in a `compose(...)` tree and pass the " *
        "outcome subtree's event slots, or use `event_names` / NamedTuple I/O)"))
end

params(c::Resolve) = (map(params, c.delays), c.branch_probs)

# The univariate interface delegates to the mixture lowering, so a `Resolve`
# behaves as the marginal time-to-resolution wherever a distribution is needed.
Base.minimum(c::Resolve) = minimum(as_mixture(c))
Base.maximum(c::Resolve) = maximum(as_mixture(c))
insupport(c::Resolve, x::Real) = insupport(as_mixture(c), x)
mean(c::Resolve) = mean(as_mixture(c))
var(c::Resolve) = var(as_mixture(c))

@doc "

Log probability density of the one_of-outcome marginal at `x`.

Routed through the AD-safe `_one_of_logmix` reduction rather than
`logpdf(as_mixture(c), x)`: `as_mixture` does `float.(branch_probs)`, which
strips an AD `Dual`/tracked type from the branch probabilities, breaking the
gradient w.r.t. a covariate case-fatality term (`logistic(Xβ)`) when a
`Resolve` is scored as a leaf of a plain (non-censored) `compose(...)` tree.
The explicit log-sum-exp keeps the probabilities' element type, so a `Dual`
propagates exactly as on the censored-tree scorer.

See also: [`as_mixture`](@ref)
"
function logpdf(c::Resolve, x::Real)
    _is_nonterminal(c) && _nonterminal_marginal_error("logpdf")
    _has_no_event(c) && _no_event_marginal_error("logpdf")
    return _one_of_logmix(c.branch_probs, c.delays, x)
end

@doc "

Probability density of the one_of-outcome marginal at `x`.

`exp` of the AD-safe [`logpdf`](@ref), so branch-prob gradients survive (see the
`logpdf` note on why `as_mixture` is avoided on a differentiated path).

See also: [`logpdf`](@ref)
"
pdf(c::Resolve, x::Real) = exp(logpdf(c, x))

@doc "

Cumulative distribution function of the one_of-outcome marginal at `x`.

The branch-prob-weighted mixture cdf `Σ_i p_i F_i(x)`, summed directly so the
probabilities keep their (possibly AD `Dual`) element type rather than being
stripped by `as_mixture`'s `float.(branch_probs)`. This keeps the cdf AD-safe on
a differentiated path (e.g. a censored-survival term), matching `logpdf`.

See also: [`as_mixture`](@ref)
"
function cdf(c::Resolve, x::Real)
    _is_nonterminal(c) && _nonterminal_marginal_error("cdf")
    _has_no_event(c) && _no_event_marginal_error("cdf")
    return sum(ntuple(
        i -> c.branch_probs[i] * cdf(c.delays[i], x), length(c.branch_probs)))
end

@doc "

Sample the one_of-outcome marginal time-to-resolution.

See also: [`as_mixture`](@ref)
"
Base.rand(rng::AbstractRNG, c::Resolve) = rand(rng, as_mixture(c))
Base.rand(c::Resolve) = rand(default_rng(), c)

@doc "

Sample a one_of outcome AND its time, returning `(name, time)`.

Unlike the univariate [`rand`](@ref) (the marginal time-to-resolution, which
discards which outcome occurred), this draws the resolved outcome from the branch
probabilities and the time from that outcome's own delay, so the chosen outcome
is retained. Used by the full-path tree simulation, where a `Resolve` node
resolves to a single named outcome.

# Arguments
- `rng`: random number generator (the no-`rng` method uses the global default).
- `c`: the [`Resolve`](@ref) node to sample an outcome from.

# Examples
```@example
using CensoredDistributions, Distributions, Random

node = Resolve(:death => (Gamma(1.5, 1.0), 0.3),
    :disch => (Gamma(2.0, 1.5), 0.7))
name, time = rand_outcome(MersenneTwister(1), node)
```

See also: [`Resolve`](@ref), [`rand`](@ref)
"
function rand_outcome(rng::AbstractRNG, c::Resolve)
    i = _sample_branch(rng, c.branch_probs)
    # A no-event win yields `missing` (no event time recorded); a real outcome
    # draws its own delay.
    _is_no_event(c.delays[i]) && return c.names[i], missing
    return c.names[i], rand(rng, c.delays[i])
end

rand_outcome(c::Resolve) = rand_outcome(default_rng(), c)

@doc "

Print a [`Resolve`](@ref) node as a recursive indented tree, labelling each
outcome with its name and branch probability and descending into any nested
composer outcome so the whole structure is shown at once.

See also: [`Resolve`](@ref)
"
function Base.show(io::IO, ::MIME"text/plain", c::Resolve)
    _show_composer_tree(io, c)
    return nothing
end

function Base.show(io::IO, c::Resolve)
    parts = ["$(c.names[k])@$(c.branch_probs[k])" for k in 1:_n_branches(c)]
    print(io, "Resolve(", join(parts, " | "), ")")
    return nothing
end

# ----------------------------------------------------------------------------
# Fixed-probability outcome probabilities (the Compete duals live in
# hazard_one_of.jl)
# ----------------------------------------------------------------------------

@doc "

The probability that the named outcome occurs for a fixed-probability
[`Resolve`](@ref) node: its branch probability (the no-event branch's mass is
the non-occurrence probability), returned as a `NamedTuple`.

See also: [`occurrence_probability`](@ref)
"
function winning_probabilities(c::Resolve)
    return NamedTuple{c.names}(c.branch_probs)
end

@doc "

The probability that ANY (non-no-event) outcome occurs for a fixed-probability
[`Resolve`](@ref) node: one minus the no-event branch mass.

See also: [`winning_probabilities`](@ref)
"
function occurrence_probability(c::Resolve)
    total = zero(float(eltype(c.branch_probs)))
    @inbounds for k in 1:_n_branches(c)
        _is_no_event(c.delays[k]) && continue
        total += c.branch_probs[k]
    end
    return total
end
