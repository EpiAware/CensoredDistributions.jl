# ============================================================================
# Sequential: a generic chain composer over plain distributions
# ============================================================================
#
# `Sequential(d1, d2, ...)` composes any `UnivariateDistribution`s into a chain
# of additive steps. A realisation is the vector of cumulative event times
# `[0, d1, d1 + d2, ...]`: the origin sits at zero and each step adds an
# independent draw from the next component. The components are plain
# distributions and may themselves be composers, so chains nest recursively;
# that nesting is the tree. This layer adds NO censored-internal behaviour
# (#329): it is the generic composition only.

@doc raw"

A chain of independent steps composed from any univariate distributions.

`Sequential` links events ``E_0 \to E_1 \to \dots \to E_k`` through independent
step distributions ``D_1, \dots, D_k``. A realisation is the flat vector of step
values ``[v_1, \dots, v_k]`` (one value per step). A step may itself be a
[`Sequential`](@ref), [`Parallel`](@ref) or [`Competing`](@ref) composer, in
which case it contributes its own flat sub-vector, so chains nest recursively and
the nesting is the tree.

`logpdf` sums the per-step log-densities over the matching slices of the value
vector:

```math
\log f(v_1, \dots, v_k) = \sum_{i=1}^{k} \log f_{D_i}(v_i).
```

This is the plain generic composition; censoring and per-record marginalisation
are not part of this type. Cumulative event times, if wanted, are the running
sum of the step values.

# See also
- [`Parallel`](@ref): independent branches
- [`Competing`](@ref): exactly one of several outcomes
"
struct Sequential{C <: Tuple} <: Distribution{Multivariate, Continuous}
    "Tuple of the step distributions ``D_1, \\dots, D_k`` (each univariate or a
    nested composer)."
    components::C

    function Sequential(components::C) where {C <: Tuple}
        length(components) >= 1 ||
            throw(ArgumentError("Sequential needs at least one component"))
        all(_is_composable, components) ||
            throw(ArgumentError(
                "every Sequential component must be a UnivariateDistribution " *
                "or a nested composer"))
        new{C}(components)
    end
end

@doc raw"

Compose univariate distributions into a [`Sequential`](@ref) chain.

Each argument is a step distribution; the realisation is the vector of
cumulative event times. Pass components as positional arguments or a single
vector/tuple. Any [`Parallel`](@ref) or [`Competing`](@ref) child nests.

# Examples
```@example
using CensoredDistributions, Distributions

d = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
rand(d)
```

# See also
- [`Sequential`](@ref): the composer type
"
Sequential(c1, cs...) = Sequential((c1, cs...))
Sequential(components::AbstractVector) = Sequential(Tuple(components))

# Total number of leaf values in a realisation (sum over nested children).
Base.length(d::Sequential) = _nleaves(d.components)

function Base.eltype(::Type{<:Sequential{C}}) where {C <: Tuple}
    return mapreduce(eltype, promote_type, fieldtypes(C))
end

params(d::Sequential) = map(params, d.components)

@doc "

Log probability density of a chain's step-value vector.

See also: [`Sequential`](@ref)
"
function logpdf(d::Sequential, x::AbstractVector)
    length(x) == length(d) || throw(DimensionMismatch(
        "expected $(length(d)) step values, got $(length(x))"))
    return _composite_logpdf(d.components, x)
end

@doc "

Probability density of a chain's step-value vector.

See also: [`logpdf`](@ref)
"
pdf(d::Sequential, x::AbstractVector) = exp(logpdf(d, x))

@doc "

Sample a chain's step-value vector (one value per step, nested children
contributing their own sub-vectors).

See also: [`Sequential`](@ref)
"
Base.rand(rng::AbstractRNG, d::Sequential) = _composite_rand(rng, d.components, float(eltype(d)))

Base.rand(d::Sequential) = rand(default_rng(), d)
sampler(d::Sequential) = d

@doc "

Print a [`Sequential`](@ref) chain as its ordered step distributions.

See also: [`Sequential`](@ref)
"
function Base.show(io::IO, ::MIME"text/plain", d::Sequential)
    n = length(d.components)
    println(io, "Sequential chain of $n steps")
    for i in 1:n
        branch = i == n ? "└─ " : "├─ "
        println(io, "  ", branch, d.components[i])
    end
    return nothing
end

function Base.show(io::IO, d::Sequential)
    print(io, "Sequential(", join(string.(d.components), " -> "), ")")
    return nothing
end
