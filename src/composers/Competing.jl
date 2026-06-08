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
# NO censored-internal behaviour (#329): the generic composition only.

@doc raw"

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
end

@doc raw"

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
    _validate_branch_probs(branch_probs)
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

function _validate_branch_probs(branch_probs::Tuple)
    for p in branch_probs
        (p >= 0 && p <= 1) ||
            throw(ArgumentError(
                "each branch probability must lie in [0, 1]; got $p"))
    end
    total = sum(branch_probs)
    isapprox(total, 1; atol = 1e-8) ||
        throw(ArgumentError(
            "competing branch probabilities sum to $total, not one"))
    return nothing
end

_n_branches(c::Competing) = length(c.names)

# ---------------------------------------------------------------------------
# Shared self-dispatch scoring (#329, #333)
# ---------------------------------------------------------------------------
#
# The Turing-free arithmetic of the `Competing` self-dispatch (#329 decision 2),
# factored here so BOTH the top-level `composed_distribution_model(d::Competing,
# row)` (the DynamicPPL extension) and the NESTED tree scorer (#333) use ONE
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
# weighted mixture `logpdf`. `delays` may be pre-censored/truncated by the caller
# (matching the conditioned path's per-record horizon), else the node's delays.
function _competing_marginal_logpdf(probs, delays, t)
    mix = MixtureModel(collect(delays), collect(float.(probs)))
    return logpdf(mix, t)
end

@doc raw"

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

See also: [`as_mixture`](@ref)
"
logpdf(c::Competing, x::Real) = logpdf(as_mixture(c), x)

@doc "

Probability density of the competing-outcome marginal at `x`.

See also: [`logpdf`](@ref)
"
pdf(c::Competing, x::Real) = pdf(as_mixture(c), x)

@doc "

Cumulative distribution function of the competing-outcome marginal at `x`.

See also: [`as_mixture`](@ref)
"
cdf(c::Competing, x::Real) = cdf(as_mixture(c), x)

@doc "

Sample the competing-outcome marginal time-to-resolution.

See also: [`as_mixture`](@ref)
"
Base.rand(rng::AbstractRNG, c::Competing) = rand(rng, as_mixture(c))
Base.rand(c::Competing) = rand(default_rng(), c)

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
