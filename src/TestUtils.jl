# ============================================================================
# TestUtils: a public interface-conformance harness for composers / leaves
# ============================================================================
#
# `CensoredDistributions.TestUtils.test_interface(d)` runs ONE interface
# checklist over a composed distribution (or bare leaf), so a downstream author
# writing a new leaf / composer can drop it into their own `@testset` to verify
# conformance. The package itself runs it over a fixture set in
# `test/interfaces.jl` (see `example_fixtures`).
#
# The harness is deliberately dependency-light: it uses `Test` (a stdlib) and the
# package's own public surface. It returns the `@testset` result so a caller can
# assert on it.

"""
    CensoredDistributions.TestUtils

Public interface-conformance harness for composers and leaves.

`TestUtils.test_interface(d)` runs one interface checklist over a composed
distribution (or bare leaf), so a downstream author writing a new leaf or
composer can drop it into their own `@testset` to verify conformance against the
package's public interface. `test_node_interface(node)` is the companion check
for a new composer node, asserting its `child_nleaves` / `child_logpdf` /
`child_rand!` methods round-trip on a flat event vector. [`test_interface`](@ref),
`example_fixtures`, [`test_rejects_invalid`](@ref) and
[`test_node_interface`](@ref) are exported from this submodule.
"""
module TestUtils

using Random: Random, AbstractRNG, Xoshiro
using Test: Test, @testset, @test, @test_nowarn, @test_throws
using Distributions: Distributions, mean, var, std, logpdf, cdf, params,
                     UnivariateDistribution
import Tables

using ..CensoredDistributions: CensoredDistributions, Sequential, Parallel,
                               Resolve, AbstractOneOf,
                               Choose, Latent, compose, latent,
                               double_interval_censored, primary_censored,
                               event, event_names, event_tree, params_table,
                               observed_distribution, endpoint,
                               child_nleaves, child_logpdf, child_rand!

export test_interface, example_fixtures, test_rejects_invalid,
       test_node_interface

# --- per-fixture descriptor -------------------------------------------------
#
# A fixture is the distribution plus the metadata the checklist needs that is not
# recoverable from the object alone: a known event-name `path` to round-trip
# through `event`, an in-support `draw` to score, a `kind` selector for a
# `Choose`, the shape of the OVERALL `mean(d)` moment, and whether the per-event
# `mean(latent(d))` view applies.

Base.@kwdef struct InterfaceFixture{D}
    name::String
    dist::D
    "An in-support realisation to score (a scalar for univariate, a Vector or a
    NamedTuple/record for multivariate)."
    draw::Any = nothing
    "A known `event` path (tuple of Symbols) that must round-trip, or `nothing`."
    path::Union{Nothing, Tuple} = nothing
    "The `kind` keyword for a `Choose` fixture, or `nothing`."
    kind::Union{Nothing, Symbol} = nothing
    "Whether the node's `rand` is a univariate scalar (a leaf / `Resolve`)."
    univariate::Bool = false
    "The shape of the OVERALL `mean(d)`/`var(d)`/`std(d)` moment: `:scalar` for a
    univariate-collapsible node (a leaf, `Sequential`, `Resolve`), `:vector`
    for a genuinely multivariate `Parallel` (a per-endpoint Vector), or `:none`
    to skip the overall-moment check (a `Choose`, or a node with no closed-form
    moment)."
    overall::Symbol = :scalar
    "Whether the per-event `mean(latent(d))` view applies: a full per-event Vector
    matching `rand(latent(d))`. True for the composers (`Sequential`/`Parallel`
    and trees rooted in them), false for a bare leaf / `Choose` / `Resolve`."
    latent_moments::Bool = false
    "Whether the node collapses to a univariate endpoint via
    `observed_distribution` (a chain / univariate)."
    has_endpoint::Bool = true
end

# --- the checklist ----------------------------------------------------------

@doc """

Run the public interface-conformance checklist over a composed distribution.

`test_interface(d; name)` runs one `@testset` of interface assertions against `d`
(a composed distribution or a bare leaf), so a downstream author writing a new
leaf / composer can verify conformance by dropping it into their own tests.

The checklist asserts, where applicable to the node's shape:

- `length` is defined (multivariate) and a `rand(d)` realisation has matching
  length;
- the OVERALL `mean` / `var` / `std` are shaped as the fixture's `overall`
  declares (a scalar for a univariate-collapsible node, a per-endpoint Vector for
  a `Parallel`), and where `latent_moments` is set the per-event
  `mean(latent(d))` / `var(latent(d))` / `std(latent(d))` are a full Vector
  matching `rand(latent(d))`;
- `logpdf` is finite on the supplied in-support `draw`;
- a univariate `cdf` is monotone and in `[0, 1]`;
- `params` works and `params_table` is a Tables.jl table
  (`Tables.istable(params_table(d))`);
- `event_names` (flat) and `event_tree` agree in leaf count;
- `event(d, path...)` round-trips the supplied known path;
- `observed_distribution` / `endpoint` collapses a chain to a univariate scalar.

Pass the fixture metadata (an `example_fixtures` entry, or the keyword
arguments directly) so the harness knows the in-support `draw`, a known `event`
`path`, a `Choose` `kind`, the `overall` moment shape, and whether the per-event
`latent_moments` view applies. Returns the `@testset` object.

# Examples
```julia
using CensoredDistributions, Distributions
using CensoredDistributions.TestUtils: test_interface

d = compose((onset_admit = Gamma(2.0, 1.0), admit_death = LogNormal(0.5, 0.4)))
test_interface(d; draw = rand(d), path = (:onset_admit,),
    overall = :vector, latent_moments = true, has_endpoint = false)
```
""" function test_interface end

function test_interface(d; name::AbstractString = string(nameof(typeof(d))),
        draw = nothing, path::Union{Nothing, Tuple} = nothing,
        kind::Union{Nothing, Symbol} = nothing, univariate::Bool = false,
        overall::Symbol = :scalar, latent_moments::Bool = false,
        has_endpoint::Bool = true)
    fix = InterfaceFixture(; name = name, dist = d, draw = draw, path = path,
        kind = kind, univariate = univariate, overall = overall,
        latent_moments = latent_moments, has_endpoint = has_endpoint)
    return test_interface(fix)
end

function test_interface(fix::InterfaceFixture)
    d = fix.dist
    return @testset "interface: $(fix.name)" begin
        _check_select(d, fix)
        _check_moments_and_rand(d, fix)
        _check_logpdf(d, fix)
        _check_cdf(d, fix)
        _check_params(d)
        _check_event_names(d, fix)
        _check_event_path(d, fix)
        _check_endpoint(d, fix)
    end
end

# A Choose needs a selection for length/rand/logpdf, so it is checked on its
# own track (the generic moment / logpdf checks are skipped for it).
_is_select(::Choose) = true
_is_select(::Any) = false

function _check_select(d::Choose, fix)
    @testset "select" begin
        fix.kind === nothing && return
        chosen = event(d, fix.kind)
        # The selected alternative round-trips through `event` and scores.
        @test fix.draw === nothing ||
              isfinite(logpdf(d, fix.draw; kind = fix.kind))
    end
    return nothing
end
_check_select(::Any, fix) = nothing

# The moments have two tiers: the OVERALL `mean(d)` (a scalar for a
# univariate-collapsible node, a per-endpoint NamedTuple for a `Parallel`) and
# the per-event `mean(latent(d))` (a NamedTuple matching `rand(latent(d))`). Any
# MULTIVARIATE composed output is a labelled `NamedTuple`; a univariate
# (collapsible) output stays a scalar. The harness exercises `rand(d)` and
# asserts each tier the fixture declares applicable.
function _check_moments_and_rand(d, fix)
    _is_select(d) && return nothing
    @testset "moments and rand" begin
        r = rand(d)
        # A multivariate composer realisation is a labelled NamedTuple; a
        # univariate node is a bare scalar.
        if fix.univariate
            @test r isa Real
        else
            @test r isa NamedTuple
        end
        # Overall moment shape.
        if fix.overall === :scalar
            @test mean(d) isa Real
            @test var(d) isa Real
            @test std(d) isa Real
        elseif fix.overall === :vector
            # A genuinely multivariate `Parallel`: the per-endpoint moment is a
            # labelled NamedTuple keyed by the endpoint names.
            m = mean(d)
            v = var(d)
            s = std(d)
            @test m isa NamedTuple
            @test v isa NamedTuple
            @test s isa NamedTuple
            @test keys(m) == keys(v) == keys(s)
        end
        # Per-event (latent) moment: a full NamedTuple matching rand(latent(d)),
        # keyed by the same event names.
        if fix.latent_moments
            ld = latent(d)
            lr = rand(ld)
            lm = mean(ld)
            lv = var(ld)
            ls = std(ld)
            @test lr isa NamedTuple
            @test lm isa NamedTuple
            @test lv isa NamedTuple
            @test ls isa NamedTuple
            @test keys(lm) == keys(lr)
            @test keys(lv) == keys(lr)
            @test keys(ls) == keys(lr)
        end
    end
    return nothing
end

function _check_logpdf(d, fix)
    _is_select(d) && return nothing
    fix.draw === nothing && return nothing
    @testset "logpdf finite on an in-support draw" begin
        @test isfinite(_score(d, fix.draw))
    end
    return nothing
end

# Score `draw` under `d`, accommodating the two composer realisation layouts: a
# censored composer scores an EVENT vector (a `Vector{Union{Missing, Float64}}`,
# a `NamedTuple` event record collected into one), a plain composer scores its
# per-value vector, and a univariate node a scalar. Tries the draw as-is first,
# then a Missing-admitting event vector, so both a `rand(d)` draw and a
# hand-built event vector work.
function _score(d, draw::Real)
    return logpdf(d, draw)
end
# A labelled NamedTuple draw scores directly: the composer / latent-leaf `logpdf`
# accepts a NamedTuple and converts it to the scored vector BY NAME internally.
function _score(d, draw::NamedTuple)
    return logpdf(d, draw)
end
function _score(d, draw::AbstractVector)
    try
        return logpdf(d, draw)
    catch err
        err isa Union{DimensionMismatch, ArgumentError} || rethrow()
        return logpdf(d, _missing_vec(draw))
    end
end

function _missing_vec(xs)
    out = Vector{Union{Missing, Float64}}(undef, length(xs))
    for (i, x) in enumerate(xs)
        out[i] = x === missing ? missing : Float64(x)
    end
    return out
end

# --- deterministic in-support event draw ------------------------------------
#
# A raw `rand(d)` is not a sound in-support draw for the "logpdf finite" check:
# a censored `Parallel` (and the nested `Resolve` tree) floors each branch gap
# to its interval, so a random draw can land a gap on (or before) the
# continuous core's support edge, where the marginal-core density is zero and
# `logpdf(d, draw)` is `-Inf`. The check then fails intermittently. This builds a
# DETERMINISTIC event vector that is guaranteed in-support: it walks the same
# tree shape the scorer consumes (mirroring `_composer_rand` / `_tree_rand!`),
# but places each leaf event at `origin + round(mean(core)) + 1`, a strictly
# positive gap clear of every continuous core's lower edge. The shared origin is
# `0.0` (in the `Uniform`/primary support, density positive). A `Resolve` node
# observes its FIRST outcome and leaves the others `missing`, the same one-
# observed-outcome record `rand` produces, so the unobserved-outcome `missing`
# slot is exercised. The result feeds straight into `_score` like any draw.

# A strictly-positive in-support gap for a (possibly censored) edge: round the
# continuous core's mean up by one so the gap clears the core's lower support
# edge even after the scorer floors it to the leaf interval.
function _insupport_gap(d)
    core = CensoredDistributions._marginal_core(d)
    return round(Float64(mean(core))) + 1.0
end

# A deterministic in-support event vector for a censored composer, or the
# `_insupport_gap` scalar for a univariate node. Mirrors the layout `_score`
# scores: `[origin, leaf events...]` in depth-first order, an unobserved
# `Resolve` outcome left `missing`. The shared origin is fixed at `0.0`.
function _insupport_event_draw(d::Union{Sequential, Parallel})
    out = Vector{Union{Missing, Float64}}(
        missing, CensoredDistributions._event_nleaves(d.components) + 1)
    out[1] = 0.0
    _fill_insupport!(out, d, 0.0, 2)
    return out
end
_insupport_event_draw(d::UnivariateDistribution) = _insupport_gap(d)

# Fill the event slots of composer `d` hanging off absolute time `origin` from
# `event_start`, returning the next free index. A `Sequential` threads its
# terminal time step to step; a `Parallel` hangs every branch off the shared
# origin (matching `_tree_rand!`).
function _fill_insupport!(out, d::Sequential, origin, event_start)
    idx = event_start
    o = origin
    for step in d.components
        idx, term = _fill_insupport_step!(out, step, o, idx)
        o = term
    end
    return idx
end
function _fill_insupport!(out, d::Parallel, origin, event_start)
    idx = event_start
    for branch in d.components
        idx, _ = _fill_insupport_step!(out, branch, origin, idx)
    end
    return idx
end

# One step/branch hanging off `origin`, returning `(next_idx, terminal_time)`
# (the time a following chain step hangs off): its own event for a leaf, the
# shared origin for a `Parallel`/`Resolve`, the last step for a `Sequential`.
function _fill_insupport_step!(
        out, step::Union{Sequential, Parallel}, origin, idx)
    next = _fill_insupport!(out, step, origin, idx)
    term = step isa Parallel ? origin :
           out[idx + CensoredDistributions._terminal_offset(step)]
    return next, term
end
function _fill_insupport_step!(out, step::AbstractOneOf, origin, idx)
    # Observe the FIRST REAL outcome (a positive in-support gap off the anchor) and
    # leave the others missing, the one-observed-outcome record the scorer
    # expects; the unobserved slots exercise the `missing` path. A no-event branch
    # carries no delay, so the first non-no-event outcome is observed.
    k = findfirst(d -> !CensoredDistributions._is_no_event(d), step.delays)
    out[idx + k - 1] = origin + _insupport_gap(step.delays[k])
    return idx + CensoredDistributions._n_branches(step), origin
end
function _fill_insupport_step!(out, step::UnivariateDistribution, origin, idx)
    y = origin + _insupport_gap(step)
    out[idx] = y
    return idx + 1, y
end
# A nested `Choose` commits to its FIRST alternative on the flat (data-free) event
# path, matching `_flat_select_alternative`: the alternatives share one event-slot
# width, so the committed alternative fills the slot and a following step hangs off
# its terminal time whichever alternative is chosen.
function _fill_insupport_step!(out, step::Choose, origin, idx)
    return _fill_insupport_step!(out, first(step.alternatives), origin, idx)
end

# A univariate node's cdf is monotone and lives in [0, 1].
function _check_cdf(d, fix)
    fix.univariate || return nothing
    @testset "univariate cdf monotone in [0, 1]" begin
        xs = range(0.0, 30.0; length = 12)
        cs = [cdf(d, x) for x in xs]
        @test all(c -> 0.0 - 1e-8 <= c <= 1.0 + 1e-8, cs)
        @test issorted(cs)
    end
    return nothing
end

function _check_params(d)
    @testset "params / params_table" begin
        @test_nowarn params(d)
        if d isa Union{Sequential, Parallel, Resolve, Choose}
            tbl = params_table(d)
            @test Tables.istable(tbl)
        end
    end
    return nothing
end

# The flat `event_names` and the nested `event_tree` must agree in LEAF count:
# every `event_tree` leaf (a Resolve outcome / a leaf delay) has its own flat
# slot, plus the flat origin event, so `length(flat) == leaves + 1`. A `Choose`
# (standalone or nested as a composer child) shares ONE flat slot across its
# alternatives, while its `event_tree` carries every alternative name, so the
# leaf-count equality does not hold; for a Choose-containing node the check is
# that the flat count matches the actual flat EVENT layout and both are
# non-empty.
function _check_event_names(d, fix)
    d isa Union{Sequential, Parallel, Resolve, Choose} || return nothing
    @testset "event_names / event_tree leaf count" begin
        flat = event_names(d)
        tree = event_tree(d)
        if d isa Choose
            @test !isempty(flat)
            @test !isempty(keys(tree))
        elseif _contains_select(d)
            # A nested Choose collapses its alternatives to one shared flat slot,
            # so the flat count tracks the event layout, not the tree leaf count.
            @test length(flat) ==
                  CensoredDistributions._event_nleaves(d.components) + 1
            @test !isempty(keys(tree))
        else
            @test length(flat) == _tree_leaf_count(tree) + 1
        end
    end
    return nothing
end

# Whether a composer tree contains a nested `Choose` anywhere (its alternatives
# share one flat event slot, so the tree-vs-flat leaf-count equality is relaxed).
_contains_select(::Choose) = true
_contains_select(c::Union{Sequential, Parallel}) = any(_contains_select, c.components)
_contains_select(c::AbstractOneOf) = any(_contains_select, c.delays)
_contains_select(c::Latent) = _contains_select(c.dist)
_contains_select(::Any) = false

# Count the leaves of an `event_tree` (a nested NamedTuple keyed to leaf names).
_tree_leaf_count(x::Symbol) = 1
function _tree_leaf_count(nt::NamedTuple)
    return sum(_tree_leaf_count, values(nt))
end

function _check_event_path(d, fix)
    fix.path === nothing && return nothing
    @testset "event round-trips a known path" begin
        @test_nowarn event(d, fix.path...)
    end
    return nothing
end

# `observed_distribution` / `endpoint` collapses a chain to a univariate scalar.
# A node with several independent endpoints (a `Parallel`, a nested tree rooted
# in one) has no single observed scalar, so the check is skipped for it (both via
# the `has_endpoint` fixture flag and a `hasmethod` guard for the keyword entry).
function _check_endpoint(d, fix)
    fix.has_endpoint || return nothing
    hasmethod(observed_distribution, Tuple{typeof(d)}) || return nothing
    @testset "observed_distribution / endpoint collapses a chain" begin
        obs = observed_distribution(d)
        @test obs isa UnivariateDistribution
        @test endpoint(d) === obs || endpoint(d) == obs
    end
    return nothing
end

# --- the package's own fixture set ------------------------------------------

@doc """

The example fixture set every composer shape, for [`test_interface`](@ref).

Returns a `Vector` of [`test_interface`](@ref)-ready fixtures covering: a bare
censored leaf, `Sequential`, `Parallel`, `Resolve`, `choose`, nested mixes,
censored leaves, and a `latent`-wrapped case. The package runs the conformance
checklist over these in `test/interfaces.jl`; a downstream author can read them
as worked examples of the metadata `test_interface` expects.
""" function example_fixtures end

function example_fixtures()
    dic(x) = double_interval_censored(
        x; primary_event =
        Distributions.Uniform(0, 1), interval = 1.0)
    G = Distributions.Gamma
    LN = Distributions.LogNormal

    leaf = dic(G(2.0, 1.0))
    # A real Sequential chain (a `compose` of a `Vector` step), which DOES have a
    # single observed endpoint (the convolution of its steps).
    seq = Sequential((dic(G(2.0, 1.0)), dic(G(1.5, 2.0))),
        (:onset_admit, :admit_death))
    par = Parallel(dic(G(2.0, 1.0)), dic(G(1.5, 2.0)))
    # A standalone Resolve's scalar moment lowers through `as_mixture`, which has
    # no analytic moment for a CENSORED leaf, so the bare-Resolve fixture uses
    # plain delays (a censored Resolve is still exercised nested in `nested`).
    comp = Resolve(:death => (G(2.0, 3.5), 0.4),
        :discharge => (G(1.0, 8.0), 0.6))
    nested_comp = Resolve(:death => (dic(G(2.0, 3.5)), 0.4),
        :discharge => (dic(G(1.0, 8.0)), 0.6))
    nested = compose((
        admit_path = compose((onset_admit = dic(G(1.2, 3.0)),
            admit_resolution = nested_comp)),
        onset_notif = dic(G(0.7, 20.0))))
    sel = CensoredDistributions.choose(:index => dic(G(2.0, 1.0)),
        :sourced => compose((a = dic(G(4.0, 1.5)), b = dic(G(1.0, 2.0)))))
    # A `Choose` with equal-width alternatives nested AS a composer child (#424):
    # the Parallel admits it and the flat event path commits to its first
    # alternative. There is no closed-form moment for the Choose branch, so the
    # overall moment and per-event latent view are skipped.
    sel_child = CensoredDistributions.choose(:a => dic(G(2.0, 1.0)),
        :b => dic(G(1.5, 2.0)))
    sel_in_par = Parallel(dic(G(2.0, 1.0)), sel_child)
    # `latent` over a single primary-censored leaf is the documented use: a
    # multivariate `[primary, observed]` scored event-by-event (no summary moment,
    # no single observed endpoint).
    lat = latent(primary_censored(G(2.0, 1.0), Distributions.Uniform(0, 1)))

    return InterfaceFixture[
        # A plain leaf has the full univariate interface (scalar moment + cdf),
        # no latent per-event view.
        InterfaceFixture(; name = "bare plain leaf", dist = G(2.0, 1.0),
            draw = 3.0, univariate = true, overall = :scalar),
        # A bare CENSORED leaf scores and has a monotone cdf, but no analytic
        # summary moment (censoring has no closed-form mean), so the overall
        # moment check is skipped here.
        InterfaceFixture(; name = "bare censored leaf", dist = leaf,
            draw = 3.0, univariate = true, overall = :none),
        # A Sequential collapses to its overall scalar moment (the convolved
        # total), with the full per-event vector via `latent`.
        InterfaceFixture(; name = "Sequential", dist = seq,
            draw = rand(seq), path = (:onset_admit,), overall = :scalar,
            latent_moments = true),
        # A Parallel is genuinely multivariate: the overall moment is a
        # per-endpoint Vector, with the full per-event vector via `latent`. It has
        # several independent endpoints and so no single observed scalar;
        # `observed_distribution` is not defined for it. The `draw` is a
        # DETERMINISTIC in-support event vector (not `rand(par)`): a random
        # censored-branch draw can floor a gap onto the core's support edge and
        # score `-Inf`, so the "logpdf finite" check would fail intermittently.
        InterfaceFixture(; name = "Parallel", dist = par,
            draw = _insupport_event_draw(par),
            overall = :vector, latent_moments = true, has_endpoint = false),
        InterfaceFixture(; name = "Resolve", dist = comp, draw = 4.0,
            path = (:death,), univariate = true, overall = :scalar),
        InterfaceFixture(; name = "choose", dist = sel, draw = 3.0,
            kind = :index, path = (:index,), overall = :none,
            has_endpoint = false),
        # A `Choose` nested as a composer child (#424): the Parallel admits it and
        # the flat event path commits to the Choose's first alternative.
        InterfaceFixture(; name = "Choose-in-Parallel", dist = sel_in_par,
            draw = _insupport_event_draw(sel_in_par), path = (:branch_1,),
            overall = :none, latent_moments = false, has_endpoint = false),
        # A nested tree branches off a shared origin (a Parallel at its root), so
        # its overall moment is a per-endpoint Vector and it has no single
        # collapsed endpoint; the full per-event vector is via `latent`. The
        # `draw` is a DETERMINISTIC in-support event vector (not `rand(nested)`):
        # its nested `Resolve` observes one outcome and leaves the other
        # `missing`, the one-observed-outcome record the scorer expects, with
        # every gap clear of its core's support edge so `logpdf` is finite.
        InterfaceFixture(; name = "nested mix", dist = nested,
            draw = _insupport_event_draw(nested),
            path = (:admit_path, :admit_resolution),
            overall = :vector, latent_moments = true, has_endpoint = false),
        # `latent` over a single primary-censored leaf is scored event-by-event
        # with no summary moment.
        InterfaceFixture(; name = "latent-wrapped", dist = lat,
            draw = rand(lat), overall = :none, has_endpoint = false)
    ]
end

# --- construction rejection -------------------------------------------------

@doc """

Assert each composer rejects invalid construction in its inner constructor.

`test_rejects_invalid()` checks that the standard composers validate their
structural invariants on EVERY construction path (the inner constructor), so a
malformed node errors at build time rather than later: `Sequential` needs at
least one component, `Parallel`/`Resolve`/`Choose` at least two children,
`Resolve` branch probabilities in `[0, 1]`, and unique `Choose` names. Returns
the `@testset` object.
""" function test_rejects_invalid()
    G = Distributions.Gamma
    return @testset "construction rejects invalid input" begin
        # Resolve: out-of-range branch prob, and fewer than two outcomes.
        @test_throws ArgumentError Resolve((:a, :b), (G(2.0, 1.0), G(1.0, 1.0)),
            (1.5, -0.5))
        @test_throws ArgumentError Resolve(:only => (G(2.0, 1.0), 1.0))
        # Sequential: empty.
        @test_throws ArgumentError Sequential((), ())
        # Choose: needs at least two, and unique names.
        @test_throws ArgumentError CensoredDistributions.choose(
            :only => G(2.0, 1.0))
        @test_throws ArgumentError CensoredDistributions.choose(
            :a => G(2.0, 1.0), :a => G(1.0, 1.0))
    end
end

# --- composer-node contract -------------------------------------------------

@doc """

Assert a composer node satisfies the public node-extension contract.

`test_node_interface(node)` checks the three methods a new composer node
implements (`child_nleaves`, `child_logpdf`, `child_rand!`, see
[Extending the composer toolkit](@ref extending-composer)) round-trip on a flat
event vector, the same way the composers walk one. It asserts that

- `child_nleaves(node)` is a positive `Int`;
- `child_rand!` fills exactly the node's `offset + 1 : offset + n` slot, leaving
  any padding either side untouched (so a node placed in a wider vector writes
  only its own slice);
- `child_logpdf(node, x, offset, n)` is a finite scalar on that drawn vector and
  does not depend on the surrounding padding (placing the node at a different
  `offset` in a wider vector scores the same).

Pass `offset` and `pad` to place the node inside a wider vector (the default
`offset = 1`, `pad = 1` brackets the slot on both sides), and `rng` for a
reproducible draw. Returns the `@testset` object.

# Examples
```julia
using CensoredDistributions, Distributions
using CensoredDistributions.TestUtils: test_node_interface

node = compose((onset_admit = Gamma(2.0, 1.0), admit_death = LogNormal(0.5, 0.4)))
test_node_interface(node)
```
""" function test_node_interface end

function test_node_interface(node; name::AbstractString =
        string(nameof(typeof(node))),
        offset::Int = 1, pad::Int = 1,
        rng::AbstractRNG = Xoshiro(1))
    return @testset "node interface: $name" begin
        # child_nleaves: a positive flat-slot count.
        n = child_nleaves(node)
        @test n isa Int
        @test n >= 1

        # child_rand!: fills exactly the node's slot in a wider vector. A NaN
        # sentinel marks the untouched cells; only the node's slice should be
        # overwritten with finite draws.
        len = offset + n + pad
        out = fill(NaN, len)
        ret = child_rand!(out, offset, rng, node)
        @test ret === nothing
        slot = (offset + 1):(offset + n)
        @test all(isfinite, @view out[slot])
        # The padding either side is left untouched.
        @test all(isnan, @view out[1:offset])
        @test all(isnan, @view out[(offset + n + 1):len])

        # child_logpdf: a finite scalar over the drawn slot.
        lp = child_logpdf(node, out, offset, n)
        @test lp isa Real
        @test isfinite(lp)

        # Position-independence: scoring the same draw at offset 0 in a tight
        # vector gives the same value (the node reads only its own slice).
        tight = out[slot]
        @test child_logpdf(node, tight, 0, n) ≈ lp
    end
end

end # module TestUtils
