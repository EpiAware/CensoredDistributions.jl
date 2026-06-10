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
struct Competing{C <: Tuple, D <: Tuple, P <: Tuple} <:
       UnivariateDistribution{Continuous}
    "Tuple of the competing outcome names (`Symbol`s)."
    names::C
    "Tuple of the competing outcome delay distributions."
    delays::D
    "Tuple of the branch probabilities, summing to one."
    branch_probs::P

    # Validate the structural invariants in the INNER constructor so EVERY
    # construction path (the `Pair...` outer constructor, equality round-trips,
    # `update`, `intervene` and direct struct calls) is checked, rather than
    # silently building a malformed node whose failure only surfaces later as a
    # confusing `DomainError` from `Categorical` inside `as_mixture`.
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

# Arguments
- `outcomes`: two or more `name => (delay, branch_prob)` pairs, each giving the
  outcome name (a `Symbol`), its delay distribution, and the probability that
  the outcome occurs.

# Examples
```@example
using CensoredDistributions, Distributions

cfr = 0.3
node = competing(:death => (Gamma(1.5, 1.0), cfr),
    :disch => (Gamma(2.0, 1.5), 1 - cfr))
mean(node)
```

# See also
- [`Competing`](@ref): the composer type
- [`as_mixture`](@ref): the `MixtureModel` lowering
- [`compose`](@ref): the front-end that nests a `Competing` as a branch
- [`Sequential`](@ref), [`Parallel`](@ref): the sibling composers
"
competing(outcomes::Pair...) = Competing(outcomes...)

function _competing_delay(payload::Tuple{<:UnivariateDistribution, <:Real})
    return payload[1]
end
function _competing_delay(payload)
    throw(ArgumentError(
        "each competing outcome payload must be a `(delay, branch_prob)` " *
        "tuple; got $(typeof(payload))"))
end

_competing_prob(payload::Tuple{<:UnivariateDistribution, <:Real}) = payload[2]

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

_n_branches(c::Competing) = length(c.names)

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

# Per-record branch probabilities must each lie in `[0, 1]` and sum to one. The
# bounds carry a small tolerance so a saturating covariate prob (`logistic(Xβ)`
# evaluating to a hair past 0 or 1 under AD/sampling) is accepted rather than
# spuriously rejected mid-gradient; a genuinely out-of-range value still errors.
# Comparisons are value-based, so an AD `Dual` is compared on its value.
function _validate_record_probs(probs)
    tol = 1e-6
    all(p -> p >= -tol && p <= 1 + tol, probs) || throw(ArgumentError(
        "each per-record branch probability must lie in [0, 1]; got " *
        "$(collect(probs))"))
    isapprox(sum(probs), 1; atol = 1e-6) || throw(ArgumentError(
        "per-record branch probabilities sum to $(sum(probs)), not one"))
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

# Marginalise an unknown outcome at the resolution time `t`: the branch-prob-
# weighted mixture log-density `log Σ_i p_i f_i(t)`. `delays` may be pre-
# censored/truncated by the caller (matching the conditioned path's per-record
# horizon), else the node's delays.
#
# Computed as `logsumexp_i (log p_i + logpdf(delay_i, t))` directly, NOT via
# `MixtureModel(delays, float.(probs))`: `float.(probs)` strips an AD `Dual`/
# tracked type from the probabilities, breaking the gradient through a
# sampled / `logistic(Xβ)` branch probability on the MARGINALISED path. The
# explicit reduction keeps the probabilities' element type, so a `Dual`
# propagates exactly as it does on the conditioned path.
function _competing_marginal_logpdf(probs, delays, t)
    return _competing_logmix(probs, delays, t)
end

# `log Σ_i p_i f_i(t)` via the log-sum-exp of `log p_i + logpdf(delay_i, t)`,
# preserving the probabilities' (possibly `Dual`) element type. A zero
# probability contributes no term (its `log` is `-Inf`); an all-zero set returns
# `-Inf`.
function _competing_logmix(probs, delays, t)
    n = length(probs)
    terms = ntuple(i -> log(probs[i]) + logpdf(delays[i], t), n)
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
    # The `Categorical` inside the `MixtureModel` needs a normalised weight set,
    # so reject an unnormalised `Competing` (e.g. one built directly with
    # branch probabilities that do not sum to one) with a clear error rather
    # than the confusing `DomainError` `Categorical` would otherwise throw.
    _validate_branch_probs_sum(c.branch_probs)
    return MixtureModel(
        collect(c.delays), collect(float.(c.branch_probs)))
end

params(c::Competing) = (map(params, c.delays), c.branch_probs)

# Outcome names, one per competing delay.
component_names(c::Competing) = c.names

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
logpdf(c::Competing, x::Real) = _competing_logmix(c.branch_probs, c.delays, x)

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
