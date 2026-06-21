# ============================================================================
# Parallel: a generic branch composer over plain distributions
# ============================================================================
#
# `Parallel(d1, d2, ...)` composes any `UnivariateDistribution`s into a set of
# independent branches sharing one origin. A realisation is the vector of branch
# values `[v1, v2, ...]`, one per branch. The branches are plain, independent
# distributions and may themselves be composers, so trees nest recursively; that
# nesting is the tree. This layer adds NO censored-internal behaviour: in
# particular it does NOT couple the branches through a shared latent origin (that
# shared-origin specialisation is layered elsewhere). Here the joint is simply
# the product of the branch densities.

@doc raw"

Independent branches composed from any univariate distributions.

`Parallel` places ``n`` branch distributions ``D_1, \dots, D_n`` off one origin,
with the realisation the vector of branch values ``[v_1, \dots, v_n]``. A branch
may itself be a [`Sequential`](@ref), [`Parallel`](@ref) or [`Resolve`](@ref)
composer, so trees nest recursively and the nesting is the tree.

`logpdf` is the sum of the per-branch log-densities,

```math
\log f(v_1, \dots, v_n) = \sum_{i=1}^{n} \log f_{D_i}(v_i).
```

The branches are independent here: this is the plain generic composition. The
shared-origin coupling (where every branch shares one latent primary event) is a
censored specialisation layered on top elsewhere, not part of this type.

# Fields
- `components`: tuple of the branch distributions (each univariate or a nested
  composer).
- `names`: tuple of the branch names (`Symbol`s), one per component; the
  `compose` front-ends thread the user's names through, positional construction
  assigns `:branch_1, :branch_2, ...`.

# See also
- [`Sequential`](@ref): a chain of additive steps
- [`Resolve`](@ref): exactly one of several outcomes
"
struct Parallel{C <: Tuple, N <: Tuple} <:
       Distribution{Multivariate, Continuous}
    "Tuple of the branch distributions (each univariate or a nested composer)."
    components::C
    "Tuple of the branch names (`Symbol`s), one per component. The `compose`
    NamedTuple/table front-ends use the user's keys; positional construction
    assigns `:branch_1, :branch_2, ...`."
    names::N

    function Parallel(components::C, names::N) where {C <: Tuple, N <: Tuple}
        length(components) >= 1 ||
            throw(ArgumentError("Parallel needs at least one branch"))
        all(_is_composable, components) ||
            throw(ArgumentError(
                "every Parallel branch must be a UnivariateDistribution or " *
                "a nested composer"))
        length(names) == length(components) ||
            throw(ArgumentError(
                "Parallel names must match the number of components"))
        all(n -> n isa Symbol, names) ||
            throw(ArgumentError("every Parallel name must be a Symbol"))
        new{C, N}(components, names)
    end
end

# Positional construction assigns default `:branch_i` names.
function Parallel(components::C) where {C <: Tuple}
    return Parallel(components, _default_names(:branch, length(components)))
end

@doc "

Compose univariate distributions into [`Parallel`](@ref) branches.

Each argument is a branch distribution; the realisation is the vector of branch
values. Pass branches as positional arguments or a single vector/tuple. Any
[`Sequential`](@ref) or [`Resolve`](@ref) child nests.

# Examples
```@example
using CensoredDistributions, Distributions

d = Parallel(Gamma(2.0, 1.0), LogNormal(1.0, 0.5))
rand(d)
```

# See also
- [`Parallel`](@ref): the composer type
"
Parallel(c1, cs...) = Parallel((c1, cs...))
Parallel(components::AbstractVector) = Parallel(Tuple(components))

@doc "

Compose univariate distributions into [`Parallel`](@ref) branches.

Lowercase verb mirroring [`sequential`](@ref) / [`resolve`](@ref): the public
constructor for a [`Parallel`](@ref) branch set. Pass branch distributions
positionally (default names `:branch_1, :branch_2, ...`) or `name => dist` pairs
to name the branches; a branch may itself be a [`Sequential`](@ref),
[`Resolve`](@ref) or nested set. Prefer this verb over the bare struct
constructor.

# Arguments
- `branches`: the branch distributions, either as positional distributions or as
  `name => dist` pairs naming each branch.

# Examples
```@example
using CensoredDistributions, Distributions

d = parallel(:admit => Gamma(2.0, 1.0), :notif => LogNormal(1.0, 0.5))
event_names(d)
```

# See also
- [`Parallel`](@ref): the composer type
- [`sequential`](@ref), [`resolve`](@ref), [`compete`](@ref), [`choose`](@ref): the sibling constructors
- [`compose`](@ref): the NamedTuple/table/matrix front-end
"
function parallel(branches::Pair...)
    length(branches) >= 1 ||
        throw(ArgumentError("parallel needs at least one branch"))
    names = Tuple(b.first for b in branches)
    all(n -> n isa Symbol, names) ||
        throw(ArgumentError("each parallel branch name must be a Symbol"))
    dists = Tuple(b.second for b in branches)
    return Parallel(dists, names)
end

parallel(b1, bs...) = Parallel((b1, bs...))
parallel(branches::AbstractVector) = Parallel(Tuple(branches))

# Total number of leaf values in a realisation (sum over nested children).
Base.length(d::Parallel) = _nleaves(d.components)

function Base.eltype(::Type{<:Parallel{C}}) where {C <: Tuple}
    return mapreduce(eltype, promote_type, fieldtypes(C))
end

# Branch names, one per component.
component_names(d::Parallel) = d.names

@doc "

Nested, name-keyed parameters of the branches.

Returns a `NamedTuple` keyed by the branch names, each value the `params` of that
branch (recursing into nested composers; a leaf delegates to its standard/
extended `Distributions.params`). This nested form is for prior introspection
via [`params_table`](@ref); a composed distribution reconstructs through
[`compose`](@ref), not through `Distribution(params...)`.

See also: [`params_table`](@ref), [`event_names`](@ref), [`event`](@ref)
"
params(d::Parallel) = _composed_params(d)

@doc "

Log probability density of a branch-value vector, summed over branches.

See also: [`Parallel`](@ref)
"
function logpdf(d::Parallel, x::AbstractVector)
    length(x) == length(d) || throw(DimensionMismatch(
        "logpdf(::Parallel) expects a length-$(length(d)) branch-value " *
        "vector (one value per branch); got length $(length(x))"))
    return _composite_logpdf(d.components, x)
end

@doc "

Probability density of a branch-value vector.

See also: [`logpdf`](@ref)
"
pdf(d::Parallel, x::AbstractVector) = exp(logpdf(d, x))

@doc "

Sample a branch realisation. For plain branches this is one value per branch
(nested children contributing their sub-vectors). For flat censored branches
sharing one latent origin it is the full event-time vector
`[origin, observed_1, ...]`. For a NESTED tree (a branch is itself a composer, or
a [`Resolve`](@ref) branch) it is a NAMED event record keyed by
`_flat_event_names`: one shared origin draw, each branch hung off it and
each Resolve outcome sampled (the unsampled outcomes `missing`). See the
censored-specialisation [`rand`](@ref) method.

See also: [`Parallel`](@ref)
"
Base.rand(rng::AbstractRNG, d::Parallel) = _named_composer_rand(rng, d)

Base.rand(d::Parallel) = rand(default_rng(), d)
sampler(d::Parallel) = d

@doc "

Print a [`Parallel`](@ref) composer as a recursive indented tree, descending
into any nested composer children so the whole structure is shown at once.

See also: [`Parallel`](@ref)
"
function Base.show(io::IO, ::MIME"text/plain", d::Parallel)
    _show_composer_tree(io, d)
    return nothing
end

function Base.show(io::IO, d::Parallel)
    print(io, "Parallel(", join(string.(d.components), " | "), ")")
    return nothing
end
