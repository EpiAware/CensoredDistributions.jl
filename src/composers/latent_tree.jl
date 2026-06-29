# ============================================================================
# latent_segments(tree): the marginal -> latent wrapper over a composed tree
# ============================================================================
#
# `latent_segments` lowers a composed tree (`Sequential` / `Parallel` /
# `Resolve` / `Choose`) into the per-segment latent scoring structure the
# vectorised latent path (`record_latent.jl`) consumes, so a composed model
# swaps from its marginal form to its (density-identical) latent form by
# wrapping the same object.

@doc "

Lower a composed tree to its per-segment latent scoring structure.

`latent_segments(tree)` turns a marginal composed distribution (a
[`Sequential`](@ref), [`Parallel`](@ref), [`Resolve`](@ref) or [`Choose`](@ref)
tree) into the latent form that samples each event rather than integrating it
out, returned as a [`Choose`](@ref) of single-edge [`latent`](@ref) chains keyed
by a `:kind` selector. Each leaf segment of the tree (an `origin -> target`
edge, or a Resolve outcome hanging off its anchor) becomes one alternative
`latent(sequential(name => leaf))`, named `origin_target` from the segment's
event names, so a record routes its observed segments to the matching
alternative. Pair it with [`latent_records`](@ref), which derives the
per-segment rows from the marginal records; the resulting `Choose` + rows feed
the vectorised [`latent_primary_priors`](@ref) / [`latent_observed_logpdf`](@ref)
path.

This is the marginal -> latent wrapper: a Turing model scores its latent form by
swapping the marginal tree for `latent_segments(tree)` and the marginal records
for `latent_records(tree, rows)`, with the rest of the model (the parameter
block, regression and priors) unchanged. The marginal and latent forms are
density-identical (the project invariant), so the latent fit recovers the same
parameters as the marginal fit.

This is distinct from `latent(tree)`, which wraps the whole tree as one
multivariate [`Latent`](@ref) per-event view (`rand`/`mean(latent(tree))`);
`latent_segments` instead decomposes the tree into the independent
observed-to-observed segments a record set scores against.

# Arguments
- `d`: a composed tree ([`Sequential`](@ref), [`Parallel`](@ref),
  [`Resolve`](@ref) or [`Choose`](@ref)) to lower to its latent segments.

# Examples
```@example
using CensoredDistributions, Distributions

leaf(d) = double_interval_censored(d; primary_event = Uniform(0, 1),
    interval = 1.0)
tree = compose((onset_admit = leaf(Gamma(1.2, 3.0)),
    onset_notif = leaf(Gamma(0.7, 20.0))))
segs = latent_segments(tree)
event_names(segs)
```

# See also
- [`latent_records`](@ref): derive the per-segment latent rows from records.
- [`latent_primary_priors`](@ref), [`latent_observed_logpdf`](@ref): the
  vectorised path the wrapper feeds.
- [`latent`](@ref): the whole-tree per-event view this complements.
"
function latent_segments(d::AbstractComposedDistribution)
    segs = _tree_segments(d)
    isempty(segs) && throw(ArgumentError(
        "the tree has no leaf segments to make latent"))
    length(segs) == 1 && return latent(sequential(
        segs[1].name => segs[1].leaf))
    alts = map(segs) do s
        s.name => latent(sequential(s.name => s.leaf))
    end
    return choose(alts...; selector = :kind)
end

# A leaf segment of a composed tree: a single edge between two events. `name` is
# the `origin_target` underscored event name (the `Choose` alternative key and
# the segment row's `:kind`); `origin`/`target` are the segment's event names;
# `leaf` is the edge's delay distribution; `branch_prob` names the reserved
# `branch_probs` field key when the segment is a Resolve outcome (so its
# per-record branch probability is read), else `nothing`.
struct _Segment{L}
    name::Symbol
    origin::Symbol
    target::Symbol
    leaf::L
    branch_prob::Union{Symbol, Nothing}
end

# Collect the leaf segments of a composed tree in depth-first event order,
# mirroring the `_walk_targets!` / `_walk_edge!` event-name walk (tree_events.jl)
# so the segment origin/target names match `event_names`. A `Sequential` threads
# the terminal event forward step to step; a `Parallel` hangs every branch off
# the shared origin; a `Resolve` outcome hangs off its anchor and carries its
# branch probability.
function _tree_segments(d::AbstractComposedDistribution)
    segs = _Segment[]
    origin = _segment_root_origin(d)
    _collect_segments!(segs, d, origin)
    return segs
end

# The root origin event name of a tree (E_0), reusing the event-name walker so
# the segment names align with `event_names`.
function _segment_root_origin(d::AbstractMultiChild)
    return _flat_event_names(d)[1]
end
_segment_root_origin(c::AbstractOneOf) = _flat_event_names(c)[1]
_segment_root_origin(d::Choose) = _segment_root_origin(
    _flat_choose_alternative(d))

function _collect_segments!(segs, d::Sequential, origin::Symbol)
    prev = origin
    enames = component_names(d)
    for i in eachindex(d.components)
        prev = _collect_edge_segments!(
            segs, enames[i], d.components[i], prev)
    end
    return nothing
end

function _collect_segments!(segs, d::Parallel, origin::Symbol)
    enames = component_names(d)
    for i in eachindex(d.components)
        _collect_edge_segments!(segs, enames[i], d.components[i], origin)
    end
    return nothing
end

# A standalone Resolve / Compete node hangs every outcome off the shared origin.
function _collect_segments!(segs, c::AbstractOneOf, origin::Symbol)
    for k in eachindex(c.names)
        _collect_outcome_segment!(
            segs, c.names[k], c.delays[k], c.branch_probs[k], origin)
    end
    return nothing
end

# A standalone Choose node routes to one alternative of a shared event-slot
# width; its default (first) alternative defines the segment layout (the
# alternatives share it), mirroring the nested-Choose edge walk.
function _collect_segments!(segs, d::Choose, origin::Symbol)
    _collect_segments!(segs, _flat_choose_alternative(d), origin)
    return nothing
end

# One edge of a composer. A leaf edge is one segment named by its split edge
# name; a nested composer recurses and returns its terminal event for a
# following chain step.
function _collect_edge_segments!(segs, edge_name::Symbol,
        child::UnivariateDistribution, origin::Symbol)
    split = _split_edge_name(edge_name)
    target = split === nothing ? edge_name : split[2]
    push!(segs, _Segment(
        Symbol(origin, :_, target), origin, target, child, nothing))
    return target
end

function _collect_edge_segments!(segs, edge_name::Symbol,
        child::AbstractMultiChild, origin::Symbol)
    _collect_segments!(segs, child, origin)
    return _nested_terminal_name(child, segs, origin)
end

# A nested Resolve edge: every outcome hangs off the chain's current event
# (`origin`), carrying its branch probability. A Resolve is terminal for a
# following chain step, so the terminal name is the anchor it hangs off.
function _collect_edge_segments!(segs, edge_name::Symbol,
        child::AbstractOneOf, origin::Symbol)
    for k in eachindex(child.names)
        _collect_outcome_segment!(
            segs, child.names[k], child.delays[k], child.branch_probs[k],
            origin)
    end
    return origin
end

# A nested Choose edge routes to one alternative of a shared event-slot width;
# its default alternative defines the segment layout (the alternatives share it).
function _collect_edge_segments!(segs, edge_name::Symbol, child::Choose,
        origin::Symbol)
    return _collect_edge_segments!(
        segs, edge_name, _flat_choose_alternative(child), origin)
end

# One Resolve outcome segment: the outcome name is the target event, anchored at
# `origin`, carrying the outcome key as the `branch_prob` selector so the
# per-record branch probability is read from the reserved `branch_probs` field.
# A composer outcome recurses (anchored at the outcome's resolution event).
function _collect_outcome_segment!(segs, oname::Symbol,
        delay::UnivariateDistribution, ::Any, origin::Symbol)
    push!(segs, _Segment(
        Symbol(origin, :_, oname), origin, oname, delay, oname))
    return nothing
end

function _collect_outcome_segment!(segs, oname::Symbol,
        delay::AbstractMultiChild, ::Any, origin::Symbol)
    _collect_segments!(segs, delay, origin)
    return nothing
end

@doc "

Derive the per-segment latent rows from marginal records and the tree.

`latent_records(tree, rows)` turns each marginal record into its observed latent
segments, the per-edge rows the vectorised latent path scores against
[`latent_segments`](@ref)`(tree)`. A record contributes one segment row per
observed edge: the `origin -> target` edge is observed when both its event
columns are present, its delay the `target - origin` gap, tagged with the
`:kind` that selects its [`Choose`](@ref) alternative. The segment's origin event
is named `missing` (the sampled latent) and its observed event carries the gap.
A Resolve outcome segment additionally copies the record's reserved
`branch_probs` field, so the vectorised [`latent_observed_logpdf`](@ref) adds the
per-record branch (case-fatality) log-probability for the recorded outcome.

The rows consume the same record schema the marginal
[`composed_distribution_model`](@ref) does (the event columns keyed by
[`event_names`](@ref) plus the reserved `branch_probs`), so a latent model is
the marginal model with `latent_segments(tree)` in place of the tree and
`latent_records` in place of the raw rows, scored by the vectorised pair. The
marginal and latent forms are density-identical (the project invariant).

# Arguments
- `tree`: the marginal composed tree (the same tree passed to
  [`latent_segments`](@ref) and the marginal model), used to read the segment
  layout.
- `rows`: the marginal records (each a `NamedTuple` keyed by
  [`event_names`](@ref) plus optional reserved fields).

# Examples
```@example
using CensoredDistributions, Distributions

leaf(d) = double_interval_censored(d; primary_event = Uniform(0, 1),
    interval = 1.0)
tree = compose((onset_admit = leaf(Gamma(1.2, 3.0)),
    onset_notif = leaf(Gamma(0.7, 20.0))))
rows = [(onset = 0.0, admit = 4.0, notif = 18.0)]
latent_records(tree, rows)
```

# See also
- [`latent_segments`](@ref): the matching tree wrapper.
- [`latent_primary_priors`](@ref), [`latent_observed_logpdf`](@ref): the
  vectorised path these rows feed.
"
function latent_records(tree::Union{Sequential, Parallel, AbstractOneOf,
            Choose}, rows)
    segs = _tree_segments(tree)
    out = NamedTuple[]
    for row in Tables.rows(rows)
        nt = _row_namedtuple(row)
        _append_record_segments!(out, segs, nt)
    end
    return out
end

# Append the observed segment rows of one marginal record. A segment is observed
# when both its origin and target event values are present; its delay is the
# `target - origin` gap. A Resolve outcome segment copies the record's reserved
# `branch_probs` so its per-record branch probability scores.
function _append_record_segments!(out, segs, nt::NamedTuple)
    for s in segs
        o = _event_value(nt, s.origin)
        t = _event_value(nt, s.target)
        (o === missing || t === missing) && continue
        gap = Float64(t) - Float64(o)
        row = (; kind = s.name, s.origin => missing, s.target => gap)
        if s.branch_prob !== nothing && haskey(nt, :branch_probs)
            p = getproperty(nt.branch_probs, s.branch_prob)
            row = merge(row, (; branch_prob = p))
        end
        push!(out, row)
    end
    return nothing
end

# The event value of a record by name: the column when present, else `missing`
# (an absent event is unobserved, dropping its segments).
function _event_value(nt::NamedTuple, name::Symbol)
    haskey(nt, name) || return missing
    return nt[name]
end
