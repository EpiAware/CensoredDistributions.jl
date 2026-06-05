# ============================================================================
# Competing-outcome (disjunctive) node for the event tree (#300)
#
# At a disjunctive node exactly ONE of several competing outcomes occurs (for
# example an admission resolves to death XOR discharge), governed by branch
# probabilities that sum to one. The branch probability is a first-class model
# quantity: with a death-versus-recovery competition the death branch
# probability is the case-fatality ratio, fitted jointly with the outcome
# delays rather than as a bolted-on logistic block.
#
# A `Competing` value is placed in the `delay` cell of one edge-list row whose
# `parent` is the disjunctive node. It names each competing `child`, that
# child's delay distribution, and the branch probability of that outcome. The
# node lowers to a `Distributions.MixtureModel` over the branch delays weighted
# by the branch probabilities: that mixture is the marginal time-to-resolution
# at the node regardless of which outcome occurs, and it drives sampling (which
# branch, then its delay). The per-record likelihood instead selects the
# realised branch from the observed outcome (see `EventTree.jl`), so the
# observed-outcome label, not the mixture marginal, carries the branch
# information.
#
# Branch probabilities may be fixed `Real`s or sampled scalars supplied by a
# probabilistic-programming model; a covariate (logistic-regression) CFR is left
# to plain Turing and is NOT part of this grammar.
# ============================================================================

@doc """

A competing-outcome (disjunctive) node of an event tree: exactly one of several
competing child outcomes occurs, governed by branch probabilities.

Placed in the `delay` cell of one edge-list row whose `parent` is the
disjunctive node, a `Competing` names each competing `child` event, that
child's `delay` distribution, and the `branch_prob` probability that the
outcome occurs. The branch probabilities must sum to one. With a
death-versus-recovery competition the death branch probability is the
case-fatality ratio.

The node lowers to a `Distributions.MixtureModel` over the branch delays
weighted by the branch probabilities (see [`as_mixture`](@ref)); that mixture
is the marginal time-to-resolution and drives sampling. The per-record
likelihood selects the realised branch from the observed outcome.

# Fields

`children` is the tuple of competing child event names, `delays` the tuple of
their delay distributions, and `branch_probs` the tuple of their branch
probabilities (summing to one).

# See also
- [`primary_censored`](@ref): builds the tree from an edge list carrying this
  node
- [`as_mixture`](@ref): the `MixtureModel` lowering
"""
struct Competing{C <: Tuple, D <: Tuple, P <: Tuple}
    "Tuple of the competing child event names."
    children::C
    "Tuple of the competing child delay distributions."
    delays::D
    "Tuple of the branch probabilities, summing to one."
    branch_probs::P
end

@doc """

Build a [`Competing`](@ref) node from competing `child => (delay, branch_prob)`
outcome specifications.

Each outcome is given as `child => (delay, branch_prob)`: the competing child
event name (a `Symbol`), its delay distribution, and the branch probability
that this outcome occurs. The branch probabilities must sum to one and each
must lie in `[0, 1]`. At least two competing outcomes are required.

# Arguments
- `outcomes`: the competing outcomes, each a `child => (delay, branch_prob)`
  pair. `child` is a `Symbol`, `delay` is a continuous
  `UnivariateDistribution`, and `branch_prob` is the outcome probability.

# Examples
```@example
using CensoredDistributions, Distributions

# Admission resolves to death (with case-fatality ratio 0.3) or discharge.
cfr = 0.3
node = Competing(:death => (Gamma(1.5, 1.0), cfr),
    :disch => (Gamma(2.0, 1.5), 1 - cfr))
```

# See also
- [`Competing`](@ref): the node type
- [`as_mixture`](@ref): the `MixtureModel` lowering
"""
function Competing(outcomes::Pair...)
    length(outcomes) >= 2 ||
        throw(ArgumentError(
            "a Competing node needs at least two competing outcomes"))
    children = Tuple(o.first for o in outcomes)
    payloads = Tuple(o.second for o in outcomes)
    for c in children
        c isa Symbol ||
            throw(ArgumentError("each competing child must be a Symbol"))
    end
    delays = Tuple(_competing_delay(p) for p in payloads)
    branch_probs = Tuple(_competing_prob(p) for p in payloads)
    _validate_branch_probs(branch_probs)
    return Competing(children, delays, branch_probs)
end

# Pull the delay distribution out of a `(delay, branch_prob)` outcome payload.
function _competing_delay(payload::Tuple{<:UnivariateDistribution, <:Real})
    return payload[1]
end
function _competing_delay(payload)
    throw(ArgumentError(
        "each competing outcome payload must be a `(delay, branch_prob)` " *
        "tuple; got $(typeof(payload))"))
end

# Pull the branch probability out of a `(delay, branch_prob)` outcome payload.
_competing_prob(payload::Tuple{<:UnivariateDistribution, <:Real}) = payload[2]

# Validate that the branch probabilities each lie in [0, 1] and sum to one.
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

# Number of competing outcomes.
_n_branches(c::Competing) = length(c.children)

@doc """

Lower a [`Competing`](@ref) node to a `Distributions.MixtureModel`.

Returns the `MixtureModel` over the branch delays weighted by the branch
probabilities. This is the marginal time-to-resolution at the node regardless
of which outcome occurs, used for sampling and as the node's marginal delay.

# Examples
```@example
using CensoredDistributions, Distributions

node = Competing(:death => (Gamma(1.5, 1.0), 0.3),
    :disch => (Gamma(2.0, 1.5), 0.7))
mix = as_mixture(node)
mean(mix)
```

# See also
- [`Competing`](@ref): the node type
"""
function as_mixture(c::Competing)
    return MixtureModel(collect(c.delays), collect(float.(c.branch_probs)))
end

# Numeric element type carrying any AD tangent in the branch delays and the
# branch probabilities, recovered from the parameters rather than the sample
# `eltype` (which would drop a Dual). Mirrors the event-tree helper.
function _competing_param_eltype(c::Competing)
    T = float(mapreduce(_parallel_param_eltype, promote_type, c.delays))
    for p in c.branch_probs
        T = promote_type(T, float(typeof(p)))
    end
    return T
end

# ---------------------------------------------------------------------------
# show: a readable competing-outcome node
# ---------------------------------------------------------------------------

@doc """

Print a [`Competing`](@ref) node as its competing `child` outcomes, each
annotated with its branch probability and delay distribution.

See also: [`Competing`](@ref)
"""
function Base.show(io::IO, ::MIME"text/plain", c::Competing)
    println(io, "Competing node with $(_n_branches(c)) outcomes")
    for k in 1:_n_branches(c)
        last = k == _n_branches(c)
        branch = last ? "└─ " : "├─ "
        println(io, "  ", branch,
            "$(c.children[k]) (p = $(c.branch_probs[k])): $(c.delays[k])")
    end
    return nothing
end

# Compact single-line show, used when a `Competing` appears inside an edge-list
# row or a larger structure.
function Base.show(io::IO, c::Competing)
    parts = ["$(c.children[k])@$(c.branch_probs[k])" for k in 1:_n_branches(c)]
    print(io, "Competing(", join(parts, " | "), ")")
    return nothing
end
