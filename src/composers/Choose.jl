# ============================================================================
# Choose: a data-selected disjunction over independent sub-distributions
# ============================================================================
#
# `Choose(:a => d_a, :b => d_b, ...; selector = :kind)` holds N NAMED
# alternatives, each an INDEPENDENT (sub-)distribution. Unlike
# [`Resolve`](@ref) / [`Parallel`](@ref), the alternatives share NO common
# origin and carry NO branch probability: a single DATA field (the `selector`,
# e.g. `:kind`) picks WHICH alternative scores / `rand`s / dispatches for a
# given record. This is the index-vs-sourced split in the hanta model:
# an `index` case (origin = its own infection) versus a `sourced` case
# (origin = the source's onset, coupled). Neither `Parallel` (shared origin,
# product) nor `Resolve` (shared origin, probabilistic mixture) fits; this is
# a disjunctive node selected by data.
#
# The hot path is TYPE-STABLE: the selected alternative is found by a
# hand-rolled recursion over the name tuple (a function barrier on each
# concrete alternative type), NOT a runtime type lookup or a `Dict` that breaks
# inference. No `AbstractTrees` in the hot path. AD flows through the selected
# alternative's own `logpdf`.

@doc raw"

A data-selected disjunction over independent named alternatives.

`Choose` holds ``n`` NAMED alternatives ``D_1, \dots, D_n``, each an independent
sub-distribution, and a `selector` naming the DATA field that picks which
alternative applies to a record. Exactly one alternative is active per record,
chosen by the selector value, NOT by a branch probability and NOT off a shared
origin. This is the disjunctive split that neither [`Parallel`](@ref) (shared
origin, product over branches) nor [`Resolve`](@ref) (shared origin,
probabilistic mixture) expresses: the alternatives are genuinely independent
sub-models with different origins, and the data says which one generated the
record.

Scoring and model dispatch route to the SELECTED alternative: `logpdf(d, x; kind)`
takes the chosen name as the `kind` keyword (no default — a `Choose` has no single
distribution to score without a selection). Sampling has two forms: `rand(d; kind)`
draws the named alternative directly, while a bare `rand(d)` (no `kind`, the
forward-simulation path) samples an alternative UNIFORMLY and returns a
self-describing record tagging which was drawn (the `selector` field set to its
name plus the alternative's own draw), so `logpdf(d, rand(d))` round-trips with no
`kind`. The selection walk is type-stable: the selected alternative is found by a
hand-rolled recursion over the name tuple that barriers into the chosen
alternative's concrete type, so inference of the hot-path `logpdf` is preserved.

An alternative may itself be any distribution or a nested composer
([`Sequential`](@ref), [`Parallel`](@ref), [`Resolve`](@ref), or another
`Choose`), so a composed tree nests INSIDE a data-selected split. The reverse — a
`Choose` as a child of a `Sequential` / `Parallel` / `compose` composer — is not
currently supported (a `Choose` has no fixed contribution length) and is rejected.

For prior introspection ([`params_table`](@ref), [`build_priors`](@ref),
[`update`](@ref)) the alternatives' parameters are namespaced per alternative:
independent per-branch params live under their alternative name (`index.…` /
`sourced.…`), so each branch's parameters are inventoried and sampled separately.
A parameter tied across alternatives via [`shared`](@ref)`(:tag, ...)` is keyed
once by its `tag` and is inventoried once and sampled once, so the tied value is
shared by every alternative that uses it.

# Fields
- `names`: tuple of the alternative names (`Symbol`s).
- `alternatives`: tuple of the alternative distributions, one per name.
- `selector`: the row field name (`Symbol`) whose value selects an alternative.

# See also
- [`choose`](@ref): friendly constructor over `name => dist` pairs
- [`Resolve`](@ref): exactly one of several shared-origin outcomes (mixture)
- [`Parallel`](@ref): independent shared-origin branches (product)
"
struct Choose{N, K <: NTuple{N, Symbol}, A <: Tuple} <:
       Distribution{Multivariate, Continuous}
    "Tuple of the alternative names (`Symbol`s)."
    names::K
    "Tuple of the alternative distributions, one per name."
    alternatives::A
    "The row field name (`Symbol`) whose value selects an alternative."
    selector::Symbol

    function Choose(names::K, alternatives::A, selector::Symbol) where {
            N, K <: NTuple{N, Symbol}, A <: Tuple}
        N >= 2 ||
            throw(ArgumentError("Choose needs at least two alternatives"))
        length(alternatives) == N ||
            throw(ArgumentError(
                "Choose needs one alternative per name; got $N names and " *
                "$(length(alternatives)) alternatives"))
        allunique(names) ||
            throw(ArgumentError("Choose alternative names must be unique"))
        all(_is_composable, alternatives) ||
            throw(ArgumentError(
                "every Choose alternative must be a UnivariateDistribution " *
                "or a nested composer"))
        new{N, K, A}(names, alternatives, selector)
    end
end

@doc "

Build a [`Choose`](@ref) data-selected disjunction from `name => dist`
alternatives.

Each alternative is `name => dist`: the alternative name (a `Symbol`) and its
independent sub-distribution. The `selector` keyword names the DATA field a
record carries to pick an alternative (default `:kind`). At least two
alternatives are required and their names must be unique.

# Arguments
- `alternatives`: the `name => dist` pairs, each an independent sub-distribution
  (a `UnivariateDistribution` or a nested composer). A single named tuple
  `(name = dist, …)` is the equivalent positional spelling for hand-written
  alternatives, kept separate from the `selector` keyword; use Pairs for
  data-driven or computed names.

# Keyword Arguments
- `selector`: the row field name (`Symbol`) whose value picks an alternative
  (default `:kind`).

# Examples
```@example
using CensoredDistributions, Distributions

# An index case (its own origin) vs a sourced case (a longer coupled delay),
# selected by the row's `:kind` field. The named tuple alternatives stay
# positional, so the `selector` keyword reads separately.
d = choose((index = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)),
        sourced = primary_censored(Gamma(4.0, 1.5), Uniform(0, 1)));
    selector = :kind)

# The equivalent Pairs form, for data-driven or computed names.
d == choose(:index => primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)),
    :sourced => primary_censored(Gamma(4.0, 1.5), Uniform(0, 1)))

# Score the alternative the data names.
logpdf(d, 3.0; kind = :index)
```

# See also
- [`Choose`](@ref): the disjunction type
- [`resolve`](@ref), [`compete`](@ref): the one-of sibling constructors
- [`sequential`](@ref), [`parallel`](@ref): the sibling composers
- [`compose`](@ref): the NamedTuple/table/matrix front-end
"
function choose(alternatives::Pair...; selector::Symbol = :kind)
    length(alternatives) >= 2 ||
        throw(ArgumentError(
            "choose needs at least two alternatives"))
    names = Tuple(a.first for a in alternatives)
    all(n -> n isa Symbol, names) ||
        throw(ArgumentError("each choose alternative name must be a Symbol"))
    dists = Tuple(a.second for a in alternatives)
    return Choose(names, dists, selector)
end

# Positional NamedTuple spelling: `(a = d1, …)` lowers to `:a => d1, …` Pairs,
# kept separate from the `selector` keyword (the alternatives are positional, so
# `selector` is not consumed as an alternative).
function choose(alternatives::NamedTuple; selector::Symbol = :kind)
    return choose(_nt_pairs(alternatives)...; selector = selector)
end

_n_alternatives(::Choose{N}) where {N} = N

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
@inline _pick(d::Choose, kind::Symbol) = _pick_recurse(d.names, d.alternatives, kind)

@inline function _pick_recurse(
        names::Tuple, alternatives::Tuple, kind::Symbol)
    return first(names) === kind ? first(alternatives) :
           _pick_recurse(Base.tail(names), Base.tail(alternatives), kind)
end

# Base case: the name was not found in any alternative.
@inline _pick_recurse(::Tuple{}, ::Tuple{},
    kind::Symbol) = throw(ArgumentError("Choose has no alternative named $(repr(kind))"))

# The length of a realisation is the SELECTED alternative's length. Without a
# selection there is no single length; `length(d)` errors to flag that a
# selection is required (mirroring `logpdf`/`rand`).
function Base.length(::Choose)
    throw(ArgumentError(
        "length(::Choose) needs a selection; the realisation length is the " *
        "selected alternative's. Use `length(CensoredDistributions._pick(" *
        "d, kind))` or pass `kind` to `logpdf`/`rand`."))
end

# The PUBLIC `params(::Choose)` is POSITIONAL, mirroring `params(::Resolve)`: a
# tuple of each alternative's `params` in alternative order. The NAME-keyed nested
# params tree (what prior introspection threads when a `Choose` is a child) goes
# through `_choose_params`/`_child_params` in `introspection.jl`, keyed by the
# alternative names so a nested `Choose` yields a name-keyed subtree.
params(d::Choose) = map(params, d.alternatives)

@doc "

Log probability density of the SELECTED alternative at `x`.

`Choose` is a data-selected disjunction, so scoring requires naming the active
alternative through the `kind` keyword; there is no default. The selection walk
is type-stable and the score is the selected alternative's own `logpdf`.

# Examples
```@example
using CensoredDistributions, Distributions

d = choose(:short => Gamma(2.0, 1.0), :long => Gamma(5.0, 1.0))
logpdf(d, 3.0; kind = :short)
```

See also: [`Choose`](@ref)
"
# A scalar `x` scores a univariate selected alternative; a vector `x` scores a
# (possibly composer) selected alternative whose realisation is a flat vector.
# Both route through the type-stable `_pick`. Typing `x` keeps these methods
# distinct from the generic multivariate `logpdf(::Distribution, ::AbstractArray)`
# batch methods (the Aqua ambiguity check), since a `Choose`'s active dimension
# is the selected alternative's, not fixed.
function logpdf(
        d::Choose, x::Real; kind::Union{Symbol, Nothing} = nothing)
    return _choose_logpdf(d, x, kind)
end
function logpdf(d::Choose, x::AbstractVector{<:Real};
        kind::Union{Symbol, Nothing} = nothing)
    return _choose_logpdf(d, x, kind)
end

function _choose_logpdf(d::Choose, x, kind)
    kind === nothing && throw(ArgumentError(
        "logpdf(::Choose, x) needs a `kind` choose the alternative"))
    return logpdf(_pick(d, kind), x)
end

@doc "

Score a self-describing [`Choose`](@ref) record (the shape `rand(d)` returns).

A bare `rand(d)` draw is a `NamedTuple` whose `selector` field names the drawn
alternative and whose remaining fields are that alternative's labelled draw, so
`logpdf(d, rand(d))` round-trips with no `kind` argument: the selector field is
read to pick the alternative, then the rest of the record is scored under that
alternative's OWN `logpdf`. A LEAF alternative's value rides in the `:value`
field; a composer alternative is scored on its own labelled record fields. A
column table of such records is summed per row; a vector of them scores
per-record (`sum(logpdf.(Ref(d), draws))`).

See also: [`Choose`](@ref), [`rand`](@ref)
"
function logpdf(d::Choose, x::NamedTuple)
    # A column table (a `NamedTuple` of vectors) is a MULTI-record source: sum the
    # per-record scorer over its rows. A single tagged record scores below.
    Tables.istable(x) && return sum(logpdf(d, r)
    for r in Tables.namedtupleiterator(x))
    haskey(x, d.selector) || throw(ArgumentError(
        "logpdf(::Choose, record) needs the selector field $(repr(d.selector)) " *
        "to name the drawn alternative; pass `kind` for a bare value instead"))
    kind = x[d.selector]
    kind isa Symbol || throw(ArgumentError(
        "the Choose selector field $(repr(d.selector)) must hold a Symbol " *
        "naming the alternative; got $(typeof(kind))"))
    inner = _drop_named_field(x, d.selector)
    return _choose_record_logpdf(_pick(d, kind), inner)
end

# Score the chosen alternative on its slice of the record (the selector field
# dropped), through the alternative's OWN `logpdf` so the SAME shape its `rand`
# produced round-trips. A LEAF alternative scores its single `:value` field; a
# composer / one_of alternative scores its labelled record (a NamedTuple it
# converts by name internally), matching the alternative's native `rand`/`logpdf`
# pair rather than the event-vector table path (which a plain composer's
# per-value record does not match).
function _choose_record_logpdf(d::UnivariateDistribution, inner::NamedTuple)
    haskey(inner, :value) || throw(ArgumentError(
        "a leaf Choose alternative record needs a `value` field; got fields " *
        "$(collect(keys(inner)))"))
    return logpdf(d, inner.value)
end
_choose_record_logpdf(d, inner::NamedTuple) = logpdf(d, inner)

@doc "

Probability density of the SELECTED alternative at `x`.

See also: [`logpdf`](@ref)
"
function pdf(d::Choose, x::Real; kind::Union{Symbol, Nothing} = nothing)
    return exp(logpdf(d, x; kind = kind))
end
function pdf(d::Choose, x::AbstractVector{<:Real};
        kind::Union{Symbol, Nothing} = nothing)
    return exp(logpdf(d, x; kind = kind))
end

@doc "

Sample a [`Choose`](@ref), returning a self-describing record tagging which
alternative was drawn.

WITHOUT a `kind` (the forward-simulation path, where no data names the branch) an
alternative is sampled UNIFORMLY and the result is a `NamedTuple` carrying the
`selector` field set to the drawn alternative's name PLUS that alternative's own
draw, so the record identifies which alternative fired and feeds straight back
into [`logpdf`](@ref) with no extra arguments. A LEAF alternative's value is
labelled `:value`; a composer alternative contributes its own flat event-record
fields. This mirrors the [`Resolve`](@ref) / [`Compete`](@ref) standalone `rand`
(a self-describing record of the outcome that fired), the `Choose` analogue being
data-SELECTED rather than probabilistic, so the tag is the chosen `kind`. For `n`
independent draws use the count form `rand(c, n)`.

WITH a `kind` (explicit selection) the draw is that alternative's OWN `rand`
returned directly (a scalar for a leaf, a labelled NamedTuple for a composer),
NOT wrapped in a selector tag: the caller already named the alternative, so this
is the in-tree / committed-selection path (`logpdf(d, draw; kind)` scores it).

See also: [`Choose`](@ref), [`logpdf`](@ref)
"
function Base.rand(
        rng::AbstractRNG, d::Choose; kind::Union{Symbol, Nothing} = nothing)
    # Explicit selection: return the chosen alternative's raw draw (the caller
    # already knows the kind, so no selector tag is added — the committed /
    # in-tree path).
    kind === nothing || return rand(rng, _pick(d, kind))
    # Forward simulation: pick uniformly and return a SELF-DESCRIBING record that
    # tags the drawn alternative (`selector => name`) so `logpdf(d, rand(d))`
    # recovers the alternative with no extra argument.
    chosen = d.names[rand(rng, 1:_n_alternatives(d))]
    return _choose_tagged_record(d, chosen, rand(rng, _pick(d, chosen)))
end

function Base.rand(d::Choose; kind::Union{Symbol, Nothing} = nothing)
    rand(default_rng(), d; kind = kind)
end

# Build the self-describing record of a bare `Choose` draw: the `selector` field
# set to the drawn alternative's `name`, MERGED with that alternative's labelled
# draw. A LEAF alternative's scalar draw is labelled `:value` (matching the
# univariate-leaf record key the vectorised path uses); a composer / one_of
# alternative already returns a labelled `NamedTuple` that merges in directly.
# The selector field comes first so the tag is read off the front.
function _choose_tagged_record(d::Choose, name::Symbol, draw)
    return merge(NamedTuple{(d.selector,)}((name,)), _choose_draw_fields(draw))
end
_choose_draw_fields(draw::NamedTuple) = draw
_choose_draw_fields(draw) = (; value = draw)

@doc "

Print a [`Choose`](@ref) node as its selector and named alternatives.

See also: [`Choose`](@ref)
"
function Base.show(io::IO, ::MIME"text/plain", d::Choose)
    return _show_composer_tree(io, d)
end

function Base.show(io::IO, d::Choose)
    parts = ["$(d.names[k])" for k in 1:_n_alternatives(d)]
    print(io, "Choose(", join(parts, " | "),
        "; selector=", repr(d.selector), ")")
    return nothing
end
