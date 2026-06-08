# ============================================================================
# Select: a data-selected disjunction over independent sub-distributions
# ============================================================================
#
# `Select(:a => d_a, :b => d_b, ...; selector = :kind)` holds N NAMED
# alternatives, each an INDEPENDENT (sub-)distribution. Unlike
# [`Competing`](@ref) / [`Parallel`](@ref), the alternatives share NO common
# origin and carry NO branch probability: a single DATA field (the `selector`,
# e.g. `:kind`) picks WHICH alternative scores / `rand`s / dispatches for a
# given record. This is the index-vs-sourced split in the hanta model (#323,
# #356): an `index` case (origin = its own infection) versus a `sourced` case
# (origin = the source's onset, coupled). Neither `Parallel` (shared origin,
# product) nor `Competing` (shared origin, probabilistic mixture) fits; this is
# a disjunctive node selected by data.
#
# The hot path is TYPE-STABLE: the selected alternative is found by a
# hand-rolled recursion over the name tuple (a function barrier on each
# concrete alternative type), NOT a runtime type lookup or a `Dict` that breaks
# inference. No `AbstractTrees` in the hot path. AD flows through the selected
# alternative's own `logpdf`.

@doc raw"

A data-selected disjunction over independent named alternatives.

`Select` holds ``n`` NAMED alternatives ``D_1, \dots, D_n``, each an independent
sub-distribution, and a `selector` naming the DATA field that picks which
alternative applies to a record. Exactly one alternative is active per record,
chosen by the selector value, NOT by a branch probability and NOT off a shared
origin. This is the disjunctive split that neither [`Parallel`](@ref) (shared
origin, product over branches) nor [`Competing`](@ref) (shared origin,
probabilistic mixture) expresses: the alternatives are genuinely independent
sub-models with different origins, and the data says which one generated the
record.

Scoring, sampling, and model dispatch all route to the SELECTED alternative.
`logpdf(d, x; kind)` and `rand(d; kind)` take the chosen name as the `kind`
keyword; there is no default, so the caller MUST supply one (a `Select` has no
single distribution to score or sample without a selection). The selection walk
is type-stable: the selected alternative is found by a hand-rolled recursion
over the name tuple that barriers into the chosen alternative's concrete type,
so inference of the hot-path `logpdf` is preserved.

An alternative may itself be any distribution or a nested composer
([`Sequential`](@ref), [`Parallel`](@ref), [`Competing`](@ref), or another
`Select`), so a data-selected split nests inside a tree like any other node.

# Fields
- `names`: tuple of the alternative names (`Symbol`s).
- `alternatives`: tuple of the alternative distributions, one per name.
- `selector`: the row field name (`Symbol`) whose value selects an alternative.

# See also
- [`select`](@ref): friendly constructor over `name => dist` pairs
- [`Competing`](@ref): exactly one of several shared-origin outcomes (mixture)
- [`Parallel`](@ref): independent shared-origin branches (product)
"
struct Select{N, K <: NTuple{N, Symbol}, A <: Tuple} <:
       Distribution{Multivariate, Continuous}
    "Tuple of the alternative names (`Symbol`s)."
    names::K
    "Tuple of the alternative distributions, one per name."
    alternatives::A
    "The row field name (`Symbol`) whose value selects an alternative."
    selector::Symbol

    function Select(names::K, alternatives::A, selector::Symbol) where {
            N, K <: NTuple{N, Symbol}, A <: Tuple}
        N >= 2 ||
            throw(ArgumentError("Select needs at least two alternatives"))
        length(alternatives) == N ||
            throw(ArgumentError(
                "Select needs one alternative per name; got $N names and " *
                "$(length(alternatives)) alternatives"))
        allunique(names) ||
            throw(ArgumentError("Select alternative names must be unique"))
        all(_is_composable, alternatives) ||
            throw(ArgumentError(
                "every Select alternative must be a UnivariateDistribution " *
                "or a nested composer"))
        new{N, K, A}(names, alternatives, selector)
    end
end

@doc raw"

Build a [`Select`](@ref) data-selected disjunction from `name => dist`
alternatives.

Each alternative is `name => dist`: the alternative name (a `Symbol`) and its
independent sub-distribution. The `selector` keyword names the DATA field a
record carries to pick an alternative (default `:kind`). At least two
alternatives are required and their names must be unique.

# Arguments
- `alternatives`: the `name => dist` pairs, each an independent sub-distribution
  (a `UnivariateDistribution` or a nested composer).

# Keyword Arguments
- `selector`: the row field name (`Symbol`) whose value picks an alternative
  (default `:kind`).

# Examples
```@example
using CensoredDistributions, Distributions

# An index case (its own origin) vs a sourced case (a longer coupled delay),
# selected by the row's `:kind` field.
d = select(:index => primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)),
    :sourced => primary_censored(Gamma(4.0, 1.5), Uniform(0, 1)))

# Score the alternative the data names.
logpdf(d, 3.0; kind = :index)
```

# See also
- [`Select`](@ref): the disjunction type
"
function select(alternatives::Pair...; selector::Symbol = :kind)
    length(alternatives) >= 2 ||
        throw(ArgumentError("select needs at least two alternatives"))
    names = Tuple(a.first for a in alternatives)
    all(n -> n isa Symbol, names) ||
        throw(ArgumentError("each Select alternative name must be a Symbol"))
    dists = Tuple(a.second for a in alternatives)
    return Select(names, dists, selector)
end

_n_alternatives(::Select{N}) where {N} = N

# --- Type-stable selection -------------------------------------------------
#
# `_pick(d, kind)` returns the alternative distribution whose name `=== kind`,
# by a hand-rolled recursion over the name/alternative tuples. Each step
# compares one `Symbol` and either returns the matching alternative or recurses
# on the tail. The recursion is over tuples of constant length, so the compiler
# union-splits / specialises it and the matching alternative is returned, then a
# downstream `logpdf`/`rand` barriers into its concrete type. This is NOT a
# runtime `Dict`/type lookup: no boxing, and a `kind` known at the call boundary
# keeps the hot path inferable.
@inline _pick(d::Select, kind::Symbol) = _pick_recurse(d.names, d.alternatives, kind)

@inline function _pick_recurse(
        names::Tuple, alternatives::Tuple, kind::Symbol)
    return first(names) === kind ? first(alternatives) :
           _pick_recurse(Base.tail(names), Base.tail(alternatives), kind)
end

# Base case: the name was not found in any alternative.
@inline _pick_recurse(::Tuple{}, ::Tuple{},
    kind::Symbol) = throw(ArgumentError("Select has no alternative named $(repr(kind))"))

# The length of a realisation is the SELECTED alternative's length. Without a
# selection there is no single length; `length(d)` errors to flag that a
# selection is required (mirroring `logpdf`/`rand`).
function Base.length(::Select)
    throw(ArgumentError(
        "length(::Select) needs a selection; the realisation length is the " *
        "selected alternative's. Use `length(CensoredDistributions._pick(" *
        "d, kind))` or pass `kind` to `logpdf`/`rand`."))
end

params(d::Select) = map(params, d.alternatives)

@doc "

Log probability density of the SELECTED alternative at `x`.

`Select` is a data-selected disjunction, so scoring requires naming the active
alternative through the `kind` keyword; there is no default. The selection walk
is type-stable and the score is the selected alternative's own `logpdf`.

# Examples
```@example
using CensoredDistributions, Distributions

d = select(:short => Gamma(2.0, 1.0), :long => Gamma(5.0, 1.0))
logpdf(d, 3.0; kind = :short)
```

See also: [`Select`](@ref)
"
# A scalar `x` scores a univariate selected alternative; a vector `x` scores a
# (possibly composer) selected alternative whose realisation is a flat vector.
# Both route through the type-stable `_pick`. Typing `x` keeps these methods
# distinct from the generic multivariate `logpdf(::Distribution, ::AbstractArray)`
# batch methods (the Aqua ambiguity check), since a `Select`'s active dimension
# is the selected alternative's, not fixed.
function logpdf(
        d::Select, x::Real; kind::Union{Symbol, Nothing} = nothing)
    return _select_logpdf(d, x, kind)
end
function logpdf(d::Select, x::AbstractVector{<:Real};
        kind::Union{Symbol, Nothing} = nothing)
    return _select_logpdf(d, x, kind)
end

function _select_logpdf(d::Select, x, kind)
    kind === nothing && throw(ArgumentError(
        "logpdf(::Select, x) needs a `kind` selecting the alternative"))
    return logpdf(_pick(d, kind), x)
end

@doc "

Probability density of the SELECTED alternative at `x`.

See also: [`logpdf`](@ref)
"
function pdf(d::Select, x::Real; kind::Union{Symbol, Nothing} = nothing)
    return exp(logpdf(d, x; kind = kind))
end
function pdf(d::Select, x::AbstractVector{<:Real};
        kind::Union{Symbol, Nothing} = nothing)
    return exp(logpdf(d, x; kind = kind))
end

@doc "

Sample the SELECTED alternative.

With a `kind` the draw is that alternative's own `rand` (a full named event
record if the alternative is a composed tree). WITHOUT a `kind` — the
forward-simulation path, where no data names the branch — an alternative is
sampled uniformly and its draw returned, so `rand`/[`predict_events`](@ref)
produces a full path for a `Select` top with no manual selection.

See also: [`Select`](@ref)
"
function Base.rand(
        rng::AbstractRNG, d::Select; kind::Union{Symbol, Nothing} = nothing)
    chosen = kind === nothing ? d.names[rand(rng, 1:_n_alternatives(d))] : kind
    return rand(rng, _pick(d, chosen))
end

function Base.rand(d::Select; kind::Union{Symbol, Nothing} = nothing)
    rand(default_rng(), d; kind = kind)
end

@doc "

Print a [`Select`](@ref) node as its selector and named alternatives.

See also: [`Select`](@ref)
"
function Base.show(io::IO, ::MIME"text/plain", d::Select)
    n = _n_alternatives(d)
    println(io,
        "Select node of $n alternatives (selector = $(repr(d.selector)))")
    for k in 1:n
        branch = k == n ? "└─ " : "├─ "
        println(io, "  ", branch, "$(d.names[k]): $(d.alternatives[k])")
    end
    return nothing
end

function Base.show(io::IO, d::Select)
    parts = ["$(d.names[k])" for k in 1:_n_alternatives(d)]
    print(io, "Select(", join(parts, " | "),
        "; selector=", repr(d.selector), ")")
    return nothing
end
