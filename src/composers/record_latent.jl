# ---------------------------------------------------------------------------
# Vectorised LATENT scoring (stacked primary priors + vectorised conditional)
# ---------------------------------------------------------------------------
#
# A `latent`-wrapped LEAF carries ONE latent primary per record; a latent CHAIN
# (`latent(Sequential(...))` with `k` edges) carries `k` latent values per record
# (the origin draw plus one independent intermediate GAP per non-terminal edge),
# with the terminal event conditioned on the reconstructed chain. A single `~`
# cannot half-sample (the latents) and half-condition (the observed events), so
# the vectorised latent flow is a two-statement pair driven by these helpers:
#
#   primaries ~ product_distribution(latent_primary_priors(d, rows))
#   @addlogprob! latent_observed_logpdf(d, rows, primaries)
#
# `latent_primary_priors` returns the STACKED priors of every latent row's latent
# values, FLATTENED in row order: a leaf row contributes one prior (its origin
# primary), a `k`-edge chain row contributes `k` priors (the origin primary then
# the first `k - 1` DECLARED edges). `primaries` carries the matching draws in the
# same flat order, so a chain row reads a CONTIGUOUS `k`-slot block.
# `latent_observed_logpdf` scores the WHOLE table given those sampled latents:
# a leaf row conditions its observed event on its matched primary through the
# delay at the implied gap; a chain row reconstructs its event times (E_0 = the
# origin draw, E_i = E_{i-1} + gap_i) and conditions the terminal event on the
# last DECLARED edge at the final gap; a MARGINAL row (an index alternative in a
# mixed Choose table) scores through its marginal record `logpdf`, so one
# `@addlogprob!` covers the mixed table. This is a VECTORISED form (a broadcast
# over the rows), not a per-record submodel loop, so it differentiates under
# ForwardDiff and Mooncake reverse. The chain's INTERMEDIATE gaps are sampled
# INDEPENDENTLY (each off its own DECLARED edge) so they ride one
# `product_distribution`; the chaining (the running sum) is pure arithmetic in
# the conditional, and the shift Jacobian is 1, so the joint equals the
# per-record latent chain model exactly.

@doc "

The stacked priors of a latent table's latent values, flattened in row order.

For a [`latent`](@ref)-wrapped leaf each latent record carries one latent primary
event; for a latent CHAIN (`latent(Sequential(...))` with `k` edges) it carries
`k` latent values (the origin draw plus one intermediate gap per non-terminal
edge). `latent_primary_priors(d, rows)` returns the vector of those priors,
FLATTENED in row order and restricted to the LATENT rows: a leaf row contributes
one prior, a `k`-edge chain row contributes `k` priors (the origin primary then
the first `k - 1` declared edges), and a marginal row (an `index` alternative in a
mixed [`Choose`](@ref) table) contributes none. The result is the input to a
single `primaries ~ product_distribution(latent_primary_priors(d, rows))`,
sampling every latent value at once.

An all-marginal table (no latent rows, e.g. every record an `index` alternative)
returns an EMPTY prior vector. `product_distribution` of an empty vector is a
degenerate product that throws on `rand`, so guard the empty case at the call
site (skip the `primaries ~ ...` statement when `latent_primary_priors(d, rows)`
is empty: there is nothing latent to sample).

# Arguments
- `d`: a latent leaf or latent chain, or a [`Choose`](@ref) with latent
  alternative(s).
- `rows`: a Tables.jl row source of records keyed by event name.

# Examples
```@example
using CensoredDistributions, Distributions
using CensoredDistributions: latent, latent_primary_priors

d = latent(primary_censored(Gamma(4.0, 1.5), Uniform(0, 1)))
priors = latent_primary_priors(d, [(delay = 3.0,), (delay = 5.0,)])
length(priors)
```

# See also
- [`latent_observed_logpdf`](@ref): the matching vectorised observed
  conditional.
"
function latent_primary_priors(d, rows)
    rowvec = collect(Tables.rows(rows))
    # The concrete prior element type is fixed by `d`'s STATIC structure (the
    # leaf's primary event, or a chain's origin primary and bare edge cores, or a
    # Choose's latent alternatives), independent of any row VALUE. Determining it
    # up front lets us fill a `Vector{T}` directly, so the return type is
    # inferrable and `product_distribution(latent_primary_priors(...))` is type-
    # stable: Mooncake then compiles one tight gradient rule (fast cold start)
    # instead of a broad/dynamic one off a `Vector{Any}` narrowed at runtime.
    # The values are byte-identical to the previous `_narrow` build.
    T = _latent_prior_eltype(d)
    priors = Vector{T}()
    for row in rowvec
        nt = _row_namedtuple(row)
        alt = _latent_alternative(d, nt)
        alt === nothing && continue
        append!(priors, _latent_row_priors(alt))
    end
    return priors
end

# The concrete element type of the stacked latent priors, derived from `d`'s
# type alone (not from any row). A latent leaf/chain contributes the prior types
# of `_latent_row_priors`; a Choose promotes over its LATENT alternatives (the
# only ones contributing priors), so a homogeneous table (one latent alternative,
# all the same primary-event type) stays concretely typed and a heterogeneous one
# (mixed primary-event types, or a chain's origin-plus-edge mix) widens to the
# promotion, exactly as the previous runtime `_narrow` did. An all-marginal Choose
# (no latent alternative) has no priors, so the element type is irrelevant; it
# falls back to the leaf/chain type, and the returned vector is empty regardless.
_latent_prior_eltype(d::Latent) = _row_priors_eltype(d)
_latent_prior_eltype(::UnivariateDistribution) = Union{}
function _latent_prior_eltype(d::Choose)
    Ts = map(_latent_prior_eltype, d.alternatives)
    return reduce(_promote_prior_type, Ts; init = Union{})
end

# The promoted type of a LATENT alternative's prior tuple, derived from its
# wrapped node type. `_latent_prior_eltype` only routes a Latent alternative here
# (a non-latent alternative is `Union{}`), so a bare `UnivariateDistribution`
# here is a latent LEAF's wrapped delay: its single prior is the primary event,
# matching `_latent_row_priors(::UnivariateDistribution)`.
_row_priors_eltype(alt::Latent) = _row_priors_eltype(alt.dist)
_row_priors_eltype(alt::UnivariateDistribution) = typeof(get_primary_event(alt))
function _row_priors_eltype(chain::Sequential)
    origin = _origin_primary_event(_first_origin_node(chain))
    edges = map(_bare_latent_edge, Base.front(chain.components))
    return reduce(_promote_prior_type, map(typeof, edges);
        init = typeof(origin))
end

# Promote two prior element types the way `promote_type`/`_narrow` would, but with
# `Union{}` (no contribution) acting as the identity so an all-marginal or empty
# branch never widens the type.
_promote_prior_type(::Type{Union{}}, ::Type{T}) where {T} = T
_promote_prior_type(::Type{T}, ::Type{Union{}}) where {T} = T
_promote_prior_type(::Type{Union{}}, ::Type{Union{}}) = Union{}
_promote_prior_type(::Type{S}, ::Type{T}) where {S, T} = promote_type(S, T)

# The latent priors of one latent row's alternative, flattened. A latent LEAF
# contributes its single origin primary; a latent CHAIN contributes the origin
# primary then the first `k - 1` edges as the intermediate gap priors, so a
# `k`-edge chain row stacks `k` priors. The endpoint-observed chain SAMPLES the
# origin and every intermediate, so each gap `E_i - E_{i-1}` is distributed as the
# BARE edge core: when both endpoints of an edge are sampled latents the
# marginal convolves the bare cores, so the latent gap prior and the terminal
# conditional must be bare too for marginal == latent (no spurious primary smear).
_latent_row_priors(alt::Latent) = _latent_row_priors(alt.dist)
_latent_row_priors(alt::UnivariateDistribution) = (get_primary_event(alt),)
function _latent_row_priors(chain::Sequential)
    origin = _origin_primary_event(_first_origin_node(chain))
    edges = map(_bare_latent_edge, Base.front(chain.components))
    return (origin, edges...)
end

# The number of latent values one latent row's alternative carries (the size of
# its contiguous block in `primaries`): one for a leaf, `k` for a `k`-edge chain.
_latent_row_width(alt::Latent) = _latent_row_width(alt.dist)
_latent_row_width(::UnivariateDistribution) = 1
_latent_row_width(chain::Sequential) = length(chain.components)

@doc "

The vectorised observed conditional of a latent table given sampled primaries.

`latent_observed_logpdf(d, rows, primaries)` scores the whole table in one
contribution: a latent LEAF row conditions its observed event on the matched
sampled primary through the delay at the implied gap (`logpdf(get_dist(alt),
y - p)`); a latent CHAIN row reconstructs its event times from its contiguous
block of `primaries` (`E_0` = the origin draw, `E_i = E_{i-1} + gap_i`) and
conditions the terminal event on the last declared edge at the final gap; and a
MARGINAL row (an `index` alternative in a mixed [`Choose`](@ref) table) scores
through its marginal record `logpdf`. The `primaries` are the draws from
`product_distribution(`[`latent_primary_priors`](@ref)`(d, rows))`, flattened in
latent-row order (a leaf row reads one value, a `k`-edge chain row reads a `k`-
slot block); a per-record `weight`/`count` scales each row's contribution. This
is the second statement of the vectorised latent pair, added with `@addlogprob!`.

# Arguments
- `d`: a latent leaf or latent chain, or a [`Choose`](@ref) with latent
  alternative(s).
- `rows`: the same Tables.jl row source passed to
  [`latent_primary_priors`](@ref).
- `primaries`: the sampled latent values, flattened in latent-row order (one per
  leaf row, `k` per `k`-edge chain row).

# Examples
```@example
using CensoredDistributions, Distributions
using CensoredDistributions: latent, latent_observed_logpdf

d = latent(primary_censored(Gamma(4.0, 1.5), Uniform(0, 1)))
rows = [(delay = 3.0,), (delay = 5.0,)]
latent_observed_logpdf(d, rows, [0.3, 0.6])
```

# See also
- [`latent_primary_priors`](@ref): the matching stacked primary priors.
"
function latent_observed_logpdf(d, rows, primaries)
    rowvec = collect(Tables.rows(rows))
    total = zero(_latent_acc_type(primaries))
    k = 0
    for row in rowvec
        nt = _row_namedtuple(row)
        alt = _latent_alternative(d, nt)
        w = _row_weight_field(nt, nothing)
        if alt === nothing
            # A marginal row scores through its marginal record logpdf, so one
            # contribution covers a mixed Choose table.
            total += _marginal_row_logpdf(d, nt)
        else
            # The row's latent values are a contiguous block of `primaries`; its
            # width is one for a leaf, `k` for a `k`-edge chain.
            width = _latent_row_width(alt)
            block = view(primaries, (k + 1):(k + width))
            k += width
            lp = _latent_row_observed_logpdf(alt, _latent_row_events(d, nt),
                block)
            total += _weight_lp(lp, w)
        end
    end
    return total
end

# The observed conditional of one latent row given its block of sampled latents.
# A latent LEAF conditions its single observed value `y` on its primary `p`
# through the delay gap `y - p`. A latent CHAIN reconstructs the event times from
# the origin draw and the intermediate gaps, then conditions the terminal
# observed event on the last declared edge at the final gap.
function _latent_row_observed_logpdf(alt::Latent, events, block)
    _latent_row_observed_logpdf(alt.dist, events, block)
end
function _latent_row_observed_logpdf(alt::UnivariateDistribution, events, block)
    y = only(events)
    return logpdf(get_dist(alt), y - block[1])
end
function _latent_row_observed_logpdf(chain::Sequential, events, block)
    # `events` is the chain's flat event vector `[E_0, ..., E_k]`; only the
    # terminal is observed for the endpoint-observed chain (origin and EVERY
    # intermediate sampled). Reconstruct the latent event times from the origin
    # draw and the intermediate gaps, then condition the observed terminal on the
    # last edge. The bare-core rule matches the per-record `_latent_edge`: when
    # there is a sampled INTERMEDIATE before the terminal (k >= 2) the whole
    # observed->observed segment is a marginalised run, so the terminal edge
    # scores the BARE core; a SINGLE-edge chain (k == 1) is the origin->terminal
    # segment and keeps its DECLARED censoring with the floored sampled origin.
    edges = chain.components
    k = length(edges)
    prev = block[1]
    @inbounds for i in 2:k
        prev += block[i]
    end
    terminal = _the_terminal_observed(events)
    if k >= 2
        return logpdf(_bare_latent_edge(edges[k]), terminal - prev)
    end
    iv = _leaf_interval(edges[k])
    shift = iv === nothing ? prev : _apply_leaf_interval(prev, iv)
    return logpdf(edges[k], terminal - shift)
end

# The terminal (last) observed value of a chain's flat event vector. The
# endpoint-observed chain row observes only its terminal event; an intermediate
# observed event would need the per-segment conditioning of the per-record model,
# which the vectorised chain path does not cover, so that is rejected.
function _the_terminal_observed(events)
    obs = filter(!ismissing, events)
    length(obs) == 1 || throw(ArgumentError(
        "the vectorised latent chain path scores the endpoint-observed chain " *
        "(only the terminal event observed); got $(length(obs)) observed " *
        "events"))
    return only(obs)
end

# The latent alternative scoring a row, or `nothing` when the row is marginal. A
# top-level Latent leaf is latent for every row; a Choose reads the row's
# selector and returns the selected alternative only when it is a Latent.
_latent_alternative(d::Latent, ::NamedTuple) = d
_latent_alternative(::UnivariateDistribution, ::NamedTuple) = nothing
function _latent_alternative(d::Choose, row::NamedTuple)
    chosen = _pick(d, _choose_kind(d, row))
    return chosen isa Latent ? chosen : nothing
end

# The selector value of a Choose row, validated to be a Symbol naming an
# alternative (mirroring the per-record Choose path).
function _choose_kind(d::Choose, row::NamedTuple)
    kind = row[d.selector]
    kind isa Symbol || throw(ArgumentError(
        "the Choose selector field $(repr(d.selector)) must hold a Symbol " *
        "naming the alternative; got $(typeof(kind))"))
    return kind
end

# The observed event value(s) of a latent row, matched to the selected
# alternative. A latent LEAF carries one observed event (a single value); a
# latent CHAIN carries its flat event vector `[E_0, ..., E_k]` (the terminal
# observed, intermediates missing). The selector field is stripped first under a
# Choose so the alternative sees only its own events.
_latent_row_events(d::Latent, row::NamedTuple) = _latent_alt_events(d.dist, row)
function _latent_row_events(d::Choose, row::NamedTuple)
    inner = _drop_named_field(row, d.selector)
    alt = _pick(d, _choose_kind(d, row))
    return _latent_alt_events(_unwrap_latent(alt), inner)
end

# The wrapped node of a (possibly latent) alternative, so the event-vector
# extraction dispatches on the leaf-vs-chain structure.
_unwrap_latent(alt::Latent) = alt.dist
_unwrap_latent(alt) = alt

# A latent leaf alternative's single observed value; a latent chain
# alternative's flat event vector matched by name.
_latent_alt_events(::UnivariateDistribution, row::NamedTuple) = (_the_observed_value(row),)
_latent_alt_events(chain::Sequential, row::NamedTuple) = _row_event_vector(chain, row)

# The single observed event value of a latent leaf row: the lone non-reserved,
# non-selector field.
_latent_observed_value(d::Latent, row::NamedTuple) = _the_observed_value(row)
function _latent_observed_value(d::Choose, row::NamedTuple)
    inner = _drop_named_field(row, d.selector)
    return _the_observed_value(inner)
end

function _the_observed_value(row::NamedTuple)
    ev = _row_event_vector(row)
    length(ev) == 1 || throw(ArgumentError(
        "a latent leaf record takes one observed event value; got " *
        "$(length(ev))"))
    return ev[1]
end

# The marginal log-density of a non-latent row in a mixed Choose table: the
# selected (marginal) alternative scored at its single observed value, weighted.
function _marginal_row_logpdf(d::Choose, row::NamedTuple)
    chosen = _pick(d, _choose_kind(d, row))
    inner = _drop_named_field(row, d.selector)
    rec = _alternative_record(chosen, inner)
    return logpdf(rec, _record_obs_value(rec))
end

# The observed value(s) a record scores at, with missing slots zeroed (the
# marginalising logpdf ignores them), as the `~`-supplied value would be.
function _record_obs_value(rec)
    return [e === missing ? 0.0 : Float64(e) for e in rec.events]
end

# Accumulator element type for the vectorised latent sum: the primaries' element
# type (carrying any AD `Dual`/tracked type), widened to float.
_latent_acc_type(primaries) = float(eltype(primaries))
_latent_acc_type(primaries::AbstractVector{Any}) = Float64

# Narrow an `Any[]` vector of priors to its concrete element type so
# `product_distribution` builds a typed product (and the draws are concretely
# typed). An empty list (no latent rows) returns an empty typed vector.
function _narrow(xs::Vector)
    isempty(xs) && return Union{}[]
    T = mapreduce(typeof, promote_type, xs)
    return collect(T, xs)
end
