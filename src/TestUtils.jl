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
                               Resolve, Compete, AbstractOneOf, NoEvent,
                               Choose, Latent, compose, latent,
                               double_interval_censored, primary_censored,
                               interval_censored, resolve, compete,
                               affine, modify, weight, thin, difference,
                               convolve_distributions, truncate_to_horizon,
                               from_moments, MomentParams,
                               Affine, Modified, Weighted, Transformed,
                               Convolved, Difference, ExponentiallyTilted,
                               PrimaryCensored, IntervalCensored,
                               event, event_names, event_tree, params_table,
                               observed_distribution, endpoint,
                               child_nleaves, child_logpdf, child_rand!

export test_interface, example_fixtures, test_rejects_invalid,
       test_node_interface, test_ad_safety, registry_types,
       test_registry_coverage

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
    "Whether the distribution is DEFECTIVE / sub-stochastic: its total mass is
    `< 1` (a `thin` leaf, a `modify`/`Modified` with sub-stochastic mass, a
    `Resolve`/`Compete` with a no-event branch). A defective univariate leaf's
    `rand` may return `missing` (the no-event outcome), so the `rand isa Real`
    assertion is relaxed to `Real`-or-`missing`, and its `cdf` tends to its
    deficit-adjusted mass `< 1` rather than `1`."
    defective::Bool = false
    "For a DEFECTIVE univariate leaf, the expected total density mass (the
    reporting probability `p` of a `thin`, say), checked by integrating the pdf
    over `[0, integrate_upper]`; `nothing` skips the sub-density-mass check (a
    non-defective node, or one whose mass has no simple closed form)."
    subdensity_mass::Union{Nothing, Float64} = nothing
    "Upper limit for the sub-density mass quadrature (the defective-mass check)."
    integrate_upper::Float64 = 200.0
    "An AD-safety probe `(f, θ)`: a closure `f(θ::Vector) -> Real` reconstructing
    the node from a parameter vector and returning a scalar log density, plus an
    in-support point `θ`. When supplied (and an `ad_gradient` backend is passed to
    `test_interface`), the harness asserts the gradient is finite — `logpdf`
    differentiability is a contract, not an ad-hoc fixture. `nothing` skips it."
    ad::Union{Nothing, Tuple{Function, Vector{Float64}}} = nothing
    "Whether to exercise the missing-sentinel round-trip: a simulated record with
    `missing` slots fed straight back into scoring (north-star tenet 8). Applies
    to a multivariate composer whose `rand`/event vector can carry a `missing`
    slot (a censored `Parallel`/tree, a `Resolve` with an unobserved outcome)."
    missing_record::Bool = false
    "Whether `rand(d)` returns a self-describing NamedTuple OUTCOME RECORD even
    though the node has scalar marginal moments / cdf (`univariate = true`). True
    for a standalone terminal `Resolve` / `Compete`: its `rand` returns the named
    record of the outcome that fired (the fired slot present, the others
    `missing`), while `mean`/`var`/`cdf` stay the scalar marginal via
    `as_mixture`. The `rand isa Real` shape assertion is relaxed to a NamedTuple
    record."
    record_rand::Bool = false
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
        has_endpoint::Bool = true, defective::Bool = false,
        subdensity_mass::Union{Nothing, Real} = nothing,
        integrate_upper::Real = 200.0,
        ad::Union{Nothing, Tuple{Function, Vector{Float64}}} = nothing,
        missing_record::Bool = false, record_rand::Bool = false,
        ad_gradient = nothing)
    fix = InterfaceFixture(; name = name, dist = d, draw = draw, path = path,
        kind = kind, univariate = univariate, overall = overall,
        latent_moments = latent_moments, has_endpoint = has_endpoint,
        defective = defective,
        subdensity_mass = subdensity_mass === nothing ? nothing :
                          Float64(subdensity_mass),
        integrate_upper = Float64(integrate_upper), ad = ad,
        missing_record = missing_record, record_rand = record_rand)
    return test_interface(fix; ad_gradient = ad_gradient)
end

# `ad_gradient` is an INJECTED gradient backend (e.g. `ForwardDiff.gradient`):
# the harness lives in `src` and is dependency-light (no AD dep of its own), so a
# caller in the test env passes the backend it has loaded. `nothing` skips the
# AD-safety contract check (still reported as a skip).
function test_interface(fix::InterfaceFixture; ad_gradient = nothing)
    d = fix.dist
    return @testset "interface: $(fix.name)" begin
        _check_choose(d, fix)
        _check_moments_and_rand(d, fix)
        _check_logpdf(d, fix)
        _check_cdf(d, fix)
        _check_subdensity(d, fix)
        _check_params(d)
        _check_event_names(d, fix)
        _check_event_path(d, fix)
        _check_endpoint(d, fix)
        _check_missing_roundtrip(d, fix)
        _check_ad(d, fix, ad_gradient)
    end
end

# A Choose needs a selection for length/rand/logpdf, so it is checked on its
# own track (the generic moment / logpdf checks are skipped for it).
_is_choose(::Choose) = true
_is_choose(::Any) = false

function _check_choose(d::Choose, fix)
    @testset "choose" begin
        fix.kind === nothing && return
        chosen = event(d, fix.kind)
        # The selected alternative round-trips through `event` and scores.
        @test fix.draw === nothing ||
              isfinite(logpdf(d, fix.draw; kind = fix.kind))
    end
    return nothing
end
_check_choose(::Any, fix) = nothing

# The moments have two tiers: the OVERALL `mean(d)` (a scalar for a
# univariate-collapsible node, a per-endpoint NamedTuple for a `Parallel`) and
# the per-event `mean(latent(d))` (a NamedTuple matching `rand(latent(d))`). Any
# MULTIVARIATE composed output is a labelled `NamedTuple`; a univariate
# (collapsible) output stays a scalar. The harness exercises `rand(d)` and
# asserts each tier the fixture declares applicable.
function _check_moments_and_rand(d, fix)
    _is_choose(d) && return nothing
    @testset "moments and rand" begin
        r = rand(d)
        # A multivariate composer realisation is a labelled NamedTuple; a
        # univariate node is a bare scalar. A DEFECTIVE univariate leaf's `rand`
        # may return `missing` (the no-event outcome), so its realisation is
        # `Real`-or-`missing`. A standalone terminal `Resolve` / `Compete`
        # (`record_rand`) has scalar marginal moments but its `rand` returns the
        # self-describing NamedTuple OUTCOME RECORD (which outcome fired), so its
        # realisation is a NamedTuple even though it is univariate-collapsible.
        if fix.record_rand
            @test r isa NamedTuple
        elseif fix.univariate
            @test fix.defective ? (r isa Real || r === missing) : r isa Real
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
    _is_choose(d) && return nothing
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
# path, matching `_flat_choose_alternative`: the alternatives share one event-slot
# width, so the committed alternative fills the slot and a following step hangs off
# its terminal time whichever alternative is chosen.
function _fill_insupport_step!(out, step::Choose, origin, idx)
    return _fill_insupport_step!(out, first(step.alternatives), origin, idx)
end

# A univariate node's cdf is monotone and lives in [0, 1]. A DEFECTIVE leaf's
# cdf tends to its mass `< 1`, so it stays `<= 1` and monotone all the same.
function _check_cdf(d, fix)
    fix.univariate || return nothing
    @testset "univariate cdf monotone in [0, 1]" begin
        xs = range(0.0, 30.0; length = 12)
        cs = [cdf(d, x) for x in xs]
        @test all(c -> 0.0 - 1e-8 <= c <= 1.0 + 1e-8, cs)
        @test issorted(cs)
        # A defective leaf is sub-stochastic: its cdf tends to a deficit-adjusted
        # mass strictly below 1 (the no-report mass leaves the observed stream).
        if fix.defective
            @test cdf(d, 1.0e6) <= 1.0 + 1e-8
        end
    end
    return nothing
end

# A DEFECTIVE (sub-stochastic) univariate leaf integrates its pdf to a total mass
# `<= 1`, the reporting probability the deficit accounts for. When the expected
# `subdensity_mass` is supplied the harness pins the integrated mass to it; the
# quadrature is a coarse midpoint rule over `[0, integrate_upper]` (the harness
# stays dependency-light, so no QuadGK), tolerant enough to catch a gross
# mass error without a quadrature dep.
function _check_subdensity(d, fix)
    (fix.univariate && fix.defective) || return nothing
    @testset "defective pdf integrates to its sub-stochastic mass" begin
        mass = _midpoint_mass(d, fix.integrate_upper)
        @test mass <= 1.0 + 1e-2
        if fix.subdensity_mass !== nothing
            @test isapprox(mass, fix.subdensity_mass; atol = 5e-3)
        end
    end
    return nothing
end

# Coarse composite-midpoint integral of the pdf over `[0, upper]`. Dense enough
# (4000 panels) to land a smooth delay density's mass within a few 1e-3.
function _midpoint_mass(d, upper::Real)
    n = 4000
    h = upper / n
    acc = 0.0
    for i in 1:n
        x = (i - 0.5) * h
        acc += Distributions.pdf(d, x)
    end
    return acc * h
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

# The flat EVENT PATH and the nested `event_tree` must agree in LEAF count:
# every `event_tree` leaf (a Resolve outcome / a leaf delay) has its own flat
# slot, plus the flat origin event, so `length(flat) == leaves + 1`. A `Choose`
# (standalone or nested as a composer child) shares ONE flat slot across its
# alternatives, while its `event_tree` carries every alternative name, so the
# leaf-count equality does not hold; for a Choose-containing node the check is
# that the flat count matches the actual flat EVENT layout and both are
# non-empty. This tests the STRUCTURAL flat event path (`_flat_event_names`,
# origin + one per leaf), not the public `event_names` record-key space, which
# for a PLAIN composer drops the origin and follows the branch names.
function _check_event_names(d, fix)
    d isa Union{Sequential, Parallel, Resolve, Choose} || return nothing
    @testset "event_names / event_tree leaf count" begin
        tree = event_tree(d)
        if d isa Choose
            # A Choose has no shared origin / flat path; its record keys are the
            # alternative names, which must be non-empty.
            @test !isempty(event_names(d))
            @test !isempty(keys(tree))
        elseif _contains_choose(d)
            # A nested Choose collapses its alternatives to one shared flat slot,
            # so the flat count tracks the event layout, not the tree leaf count.
            @test length(CensoredDistributions._flat_event_names(d)) ==
                  CensoredDistributions._event_nleaves(d.components) + 1
            @test !isempty(keys(tree))
        else
            @test length(CensoredDistributions._flat_event_names(d)) ==
                  _tree_leaf_count(tree) + 1
        end
    end
    return nothing
end

# Whether a composer tree contains a nested `Choose` anywhere (its alternatives
# share one flat event slot, so the tree-vs-flat leaf-count equality is relaxed).
_contains_choose(::Choose) = true
_contains_choose(c::Union{Sequential, Parallel}) = any(_contains_choose, c.components)
_contains_choose(c::AbstractOneOf) = any(_contains_choose, c.delays)
_contains_choose(c::Latent) = _contains_choose(c.dist)
_contains_choose(::Any) = false

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

# Missing-sentinel round-trip (north-star tenet 8): a simulated record carries
# `missing` in an unobserved slot (a censored composer's `rand`, a `Resolve`
# outcome that did not fire), and that SAME record must feed straight back into
# scoring and yield a finite log density. This pins the simulate -> score loop on
# a `missing`-bearing record: a `rand(d)` draw is collected into an event vector,
# one observed slot is blanked to `missing`, and the blanked record is scored.
function _check_missing_roundtrip(d, fix)
    fix.missing_record || return nothing
    @testset "missing-sentinel record round-trips into scoring" begin
        # The fixture's in-support event draw already carries the censored layout
        # the scorer consumes (with an unobserved `Resolve` slot left `missing`),
        # so it IS a missing-bearing record where the node has an unobserved
        # outcome; otherwise blank the last real slot to a `missing` sentinel.
        rec = _missing_record(fix.draw)
        @test any(ismissing, rec)
        @test isfinite(_score(d, rec))
    end
    return nothing
end

# Build a `missing`-bearing record from a draw: if it already carries a `missing`
# (a Resolve no-event slot), keep it; else blank the last finite slot.
function _missing_record(draw::AbstractVector)
    out = _missing_vec(draw)
    any(ismissing, out) && return out
    last_real = findlast(!ismissing, out)
    last_real === nothing || (out[last_real] = missing)
    return out
end
_missing_record(draw) = draw

# --- AD-safety contract -----------------------------------------------------

# `logpdf` must differentiate: the fixture's `ad = (f, θ)` reconstructs the node
# from a parameter vector and returns a scalar log density, and the INJECTED
# `ad_gradient` backend (e.g. `ForwardDiff.gradient`, passed from the test env)
# evaluates `∇f(θ)`, which must be finite. This makes differentiability a first-
# class contract rather than an ad-hoc per-fixture test. With no backend injected
# the check is skipped (still reported), so the harness keeps its `src` AD-dep
# free; the package's own suite injects ForwardDiff (and Mooncake where loaded).
function _check_ad(d, fix, ad_gradient)
    fix.ad === nothing && return nothing
    ad_gradient === nothing && return nothing
    f, θ = fix.ad
    @testset "logpdf is AD-differentiable (finite gradient)" begin
        g = ad_gradient(f, θ)
        @test g isa AbstractVector
        @test all(isfinite, g)
    end
    return nothing
end

@doc """

Assert a parameterised log density differentiates under an injected AD backend.

`test_ad_safety(f, θ; ad_gradient, name)` evaluates `ad_gradient(f, θ)` (e.g.
`ForwardDiff.gradient`) on a closure `f(θ::Vector) -> Real` reconstructing a
distribution from its parameter vector and returning a scalar log density, and
asserts the gradient is finite. AD-safety of `logpdf` is a contract: a leaf or
node that scores must also differentiate. Pass several backends through
`test_interface`'s `ad_gradient` or call this directly per backend (ForwardDiff
always, Mooncake / ReverseDiff where the test env loads them). Returns the
`@testset` object.
""" function test_ad_safety(f::Function, θ::Vector{Float64}; ad_gradient,
        name::AbstractString = "ad")
    return @testset "AD-safety: $name" begin
        g = ad_gradient(f, θ)
        @test g isa AbstractVector
        @test all(isfinite, g)
    end
end

# --- the package's own fixture set ------------------------------------------

@doc """

The example fixture set over every public type, for [`test_interface`](@ref).

Returns a `Vector` of [`test_interface`](@ref)-ready fixtures covering the full
public registry: a bare (plain / censored) leaf, the composer shapes
(`Sequential`, `Parallel`, `Resolve`, `Compete`, `choose`), nested mixes, a
`latent`-wrapped case, the distribution-modifier / derived leaves (`affine`,
`modify` over the log / identity / logit-discrete links, `weight`, `thin`,
`Convolved`, `Difference`, `ExponentiallyTilted`), a defective no-event
`Resolve`, and the deep-nesting matrix (a `Sequential` of `Parallel`, a `Choose`
of `Sequential`s, a `Convolved` of a composed leaf, a truncation over a composed
chain). [`test_registry_coverage`](@ref) asserts these cover every public type.
The package runs the conformance checklist over these in `test/interfaces.jl`; a
downstream author can read them as worked examples of the metadata
`test_interface` expects (a `draw`, an `event` `path`, the `overall` moment
shape, an `ad` probe, the `defective` / `missing_record` flags).
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
    # A `Choose` with equal-width alternatives nested AS a composer child:
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

    # --- distribution-modifier / derived leaves (issue #666 registry) -------
    # Each is a UNIVARIATE leaf; the AD probe reconstructs it from a parameter
    # vector and scores a scalar logpdf, asserting differentiability.
    Uni = Distributions.Uniform
    aff = affine(G(2.0, 1.0); scale = 2.0, shift = 1.0)
    aff_ad = (
        θ -> Distributions.logpdf(
            affine(G(θ[1], θ[2]); scale = 2.0, shift = 1.0), 5.0),
        [2.0, 1.0])
    # modify, log link (proportional hazards): closed-form, no analytic moment.
    mod_log = modify(LN(1.5, 0.5), -log(2.0); link = log)
    mod_log_ad = (
        θ -> Distributions.logpdf(
            modify(LN(θ[1], θ[2]), -log(2.0); link = log), 2.0),
        [1.5, 0.5])
    # modify, identity link (additive hazards): a POSITIVE effect keeps the
    # modified hazard `h + β >= 0`, so the sub-survival stays in `[0, 1]` and the
    # cdf is monotone. A NEGATIVE identity-link effect drives the cdf negative and
    # non-monotone (the hazard goes sub-zero near the origin) — a real model-
    # validity bug tracked in issue #670, NOT a harness regression, so the fixture
    # uses the valid positive-effect regime.
    mod_id = modify(LN(1.5, 0.5), 0.1; link = identity)
    mod_id_ad = (
        θ -> Distributions.logpdf(
            modify(LN(θ[1], θ[2]), 0.1; link = identity), 2.0),
        [1.5, 0.5])
    # modify, logit link (discrete-time reporting hazard): dispatches on the
    # interval-censored discrete path with a per-bin effect vector.
    mod_logit = modify(
        interval_censored(LN(1.5, 0.5), 1.0), fill(0.2, 11); link = :logit)
    # weight: a per-record count weight on a leaf, no analytic moment.
    wtd = weight(G(2.0, 1.0), 3.0)
    wtd_ad = (θ -> Distributions.logpdf(weight(G(θ[1], θ[2]), 3.0), 3.0),
        [2.0, 1.0])
    # thin: a DEFECTIVE leaf (reporting probability `p = 0.3`). Its pdf integrates
    # to `p`, its cdf tends to `p`, and `rand` returns `missing` with probability
    # `1 - p`. The conditional-on-report moments are closed-form (`:scalar`).
    thn = thin(LN(1.5, 0.5), 0.3)
    thn_ad = (θ -> Distributions.logpdf(thin(LN(θ[1], θ[2]), 0.3), 2.0),
        [1.5, 0.5])
    # Convolved: the sum of two independent delays, a univariate leaf with a
    # closed-form (additive) mean.
    conv = convolve_distributions(G(2.0, 1.0), LN(0.5, 0.4))
    conv_ad = (
        θ -> Distributions.logpdf(
            convolve_distributions(G(θ[1], θ[2]), LN(0.5, 0.4)), 3.0),
        [2.0, 1.0])
    # Difference: Z = X - Y, two-sided (possibly negative) support, a derived
    # observation rather than a delay leaf. Its mean is the difference of means.
    diff = difference(G(2.0, 1.0), G(1.5, 2.0))
    diff_ad = (θ -> Distributions.logpdf(
            difference(G(θ[1], 1.0), G(1.5, 2.0)), 0.5),
        [2.0])
    # ExponentiallyTilted: a bounded exponentially-tilted leaf on `[min, max]`.
    et = ExponentiallyTilted(0.0, 5.0, 0.5)
    et_ad = (θ -> Distributions.logpdf(
            ExponentiallyTilted(0.0, 5.0, θ[1]), 2.0), [0.5])
    # MomentParams: a Gamma reparameterised by (mean, shape) via the generic
    # moment wrapper, density-identical to `Gamma(shape, mean / shape)`. The AD
    # closure differentiates the log density wrt the (mean, shape) pair.
    mg = from_moments(Distributions.Gamma; mean = 5.0, shape = 2.0)
    mg_ad = (
        θ -> Distributions.logpdf(
            from_moments(Distributions.Gamma; mean = θ[1], shape = θ[2]),
            4.0),
        [5.0, 2.0])
    # Compete (racing hazards): a UNIVARIATE time-to-first-event marginal, an
    # AbstractOneOf like Resolve. Plain (non-censored) delays so it has an
    # analytic-enough scalar mean.
    cmp = compete(:recovery => G(2.0, 1.0), :death => G(1.5, 2.0))
    # A DEFECTIVE Resolve with an explicit no-event branch: its observed-time mass
    # is `< 1` (the `:none` outcome carries the deficit, `occurrence_probability`
    # `= 0.4 < 1`). A standalone defective Resolve has NO scalar/marginal logpdf
    # (it is scored through the event-vector path only), so it is exercised
    # NESTED in a Parallel where the composer scores its event vector and its
    # unobserved `:none` outcome leaves a `missing` slot — the sub-stochastic
    # no-event semantics in the simulate -> score loop.
    res_def_node = resolve(:event => (dic(G(2.0, 1.0)), 0.4),
        :none => (NoEvent(), 0.6))
    res_def = Parallel(dic(G(1.5, 2.0)), res_def_node)

    # --- deep-nesting matrix (folds #645/#653 coverage into the harness) ----
    # A Sequential whose step is a Parallel: a chain whose first step fans out.
    seq_of_par = Sequential(
        (Parallel(dic(G(2.0, 1.0)), dic(G(1.5, 2.0))), dic(G(1.0, 3.0))),
        (:fanout, :tail))
    # A Choose of Sequentials: each alternative is a two-step chain. The
    # alternatives are PLAIN (uncensored) chains so the selected alternative
    # exposes the scalar event-vector `logpdf(d, x; kind)` (the censored
    # multivariate `Choose` alternative has no such scalar path yet — out of
    # scope here, mirroring the nested-scalar coverage in
    # `test/composers/nested_scalar_methods.jl`).
    choose_of_seq = CensoredDistributions.choose(
        :fast => Sequential((G(1.0, 1.0), G(1.0, 1.0)), (:a, :b)),
        :slow => Sequential((G(2.0, 2.0), G(2.0, 2.0)), (:c, :d)))
    # A Convolved of a composed (univariate) leaf: an Affine (shift+scale)
    # composed leaf convolved with a plain leaf, the sum of a transformed delay
    # and a delay. The plain second component keeps the convolution CDF
    # analytic / monotone (a double-censored second component adds quadrature
    # noise in the saturated tail — exercised by the dic fixtures already).
    conv_composed = convolve_distributions(
        affine(G(1.5, 2.0); shift = 0.5), LN(0.5, 0.4))
    # Right-truncation of a composed chain's observed TOTAL: the chain is
    # collapsed to its scalar combine-then-censor total via
    # `observed_distribution`, then right-truncated, giving a univariate
    # `Truncated` leaf. (A bare-node `truncate_to_horizon(seq)` now distributes
    # the truncation into the leaf cores and stays multivariate, #655; this
    # fixture intends the scalar total, so it uses the explicit collapse form.)
    trunc_composed = truncate_to_horizon(observed_distribution(seq), 20.0)

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
            overall = :vector, latent_moments = true, has_endpoint = false,
            missing_record = true),
        # A terminal `Resolve`: scalar marginal moments / cdf (via `as_mixture`),
        # but `rand` returns the named outcome record (which outcome fired), so
        # `record_rand` relaxes the `rand` shape check.
        InterfaceFixture(; name = "Resolve", dist = comp, draw = 4.0,
            path = (:death,), univariate = true, overall = :scalar,
            record_rand = true),
        InterfaceFixture(; name = "choose", dist = sel, draw = 3.0,
            kind = :index, path = (:index,), overall = :none,
            has_endpoint = false),
        # A `Choose` nested as a composer child: the Parallel admits it and
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
            overall = :vector, latent_moments = true, has_endpoint = false,
            missing_record = true),
        # `latent` over a single primary-censored leaf is scored event-by-event
        # with no summary moment.
        InterfaceFixture(; name = "latent-wrapped", dist = lat,
            draw = rand(lat), overall = :none, has_endpoint = false),

        # --- distribution-modifier / derived leaves (issue #666) ------------
        # Affine: a deterministic shift+scale leaf with a closed-form moment.
        InterfaceFixture(; name = "affine", dist = aff, draw = 5.0,
            univariate = true, overall = :scalar, ad = aff_ad),
        # modify, log link (proportional hazards): no analytic moment.
        InterfaceFixture(; name = "modify (log link)", dist = mod_log,
            draw = 2.0, univariate = true, overall = :none, ad = mod_log_ad),
        # modify, identity link (additive hazards), POSITIVE effect (the valid
        # regime; the negative-effect bug is #670).
        InterfaceFixture(; name = "modify (identity link)", dist = mod_id,
            draw = 2.0, univariate = true, overall = :none, ad = mod_id_ad),
        # modify, logit link (discrete-time reporting hazard): the discrete path.
        InterfaceFixture(; name = "modify (logit link, discrete)",
            dist = mod_logit, draw = 2.0, univariate = true, overall = :none),
        # Weighted: a per-record count weight on a leaf.
        InterfaceFixture(; name = "weight", dist = wtd, draw = 3.0,
            univariate = true, overall = :none, ad = wtd_ad),
        # thin: a DEFECTIVE leaf, pdf integrates to p = 0.3, rand may be missing.
        InterfaceFixture(; name = "thin (defective)", dist = thn, draw = 2.0,
            univariate = true, overall = :scalar, defective = true,
            subdensity_mass = 0.3, integrate_upper = 400.0, ad = thn_ad),
        # Convolved: the sum of two independent delays.
        InterfaceFixture(; name = "Convolved", dist = conv, draw = 3.0,
            univariate = true, overall = :scalar, ad = conv_ad),
        # Difference: Z = X - Y, two-sided support, a derived observation.
        InterfaceFixture(; name = "Difference", dist = diff, draw = 0.5,
            univariate = true, overall = :scalar, ad = diff_ad),
        # ExponentiallyTilted: a bounded exponentially-tilted leaf.
        InterfaceFixture(; name = "ExponentiallyTilted", dist = et, draw = 2.0,
            univariate = true, overall = :scalar, ad = et_ad),
        # MomentParams: a (mean, shape)-reparameterised Gamma leaf, scalar moment.
        InterfaceFixture(; name = "MomentParams", dist = mg, draw = 4.0,
            univariate = true, overall = :scalar, ad = mg_ad),
        # Compete (racing hazards): a univariate time-to-first-event marginal.
        # Scalar marginal moments, but `rand` returns the named winning-cause
        # record, so `record_rand` relaxes the `rand` shape check.
        InterfaceFixture(; name = "Compete", dist = cmp, draw = 2.0,
            path = (:recovery,), univariate = true, overall = :scalar,
            record_rand = true),
        # A DEFECTIVE Resolve (a no-event branch) nested in a Parallel: the
        # composer scores its event vector and the unobserved `:none` outcome
        # leaves a `missing` slot, exercising the sub-stochastic no-event
        # semantics in the simulate -> score loop.
        InterfaceFixture(; name = "Resolve (defective, no-event)",
            dist = res_def, draw = _insupport_event_draw(res_def),
            overall = :none, has_endpoint = false, missing_record = true),

        # --- deep-nesting matrix (#645/#653 folded into the harness) --------
        # A Sequential whose step is a Parallel (a chain that fans out).
        InterfaceFixture(; name = "deep: Sequential of Parallel",
            dist = seq_of_par, draw = _insupport_event_draw(seq_of_par),
            path = (:fanout,), overall = :none, has_endpoint = false,
            missing_record = true),
        # A Choose of Sequentials (the selected alternative is a chain).
        InterfaceFixture(; name = "deep: Choose of Sequentials",
            dist = choose_of_seq, draw = [1.0, 2.0], kind = :fast,
            path = (:fast,), overall = :none, has_endpoint = false),
        # A Convolved of a composed (univariate) leaf.
        InterfaceFixture(; name = "deep: Convolved of composed leaf",
            dist = conv_composed, draw = 3.0, univariate = true,
            overall = :none),
        # Right-truncation over a composed chain: a Truncated wrapper collapsing
        # the chain to its observed total, scored as a univariate scalar.
        InterfaceFixture(; name = "deep: truncated over composed",
            dist = trunc_composed, draw = 5.0, univariate = true,
            overall = :none, has_endpoint = false)
    ]
end

# --- registry completeness --------------------------------------------------

# The package's own public distribution / leaf / node types that
# `test_interface` is expected to exercise. Each must appear (possibly nested) in
# at least one `example_fixtures()` fixture, asserted by `test_registry_coverage`.
# A NEW public distribution type added without a fixture fails that meta-test.
#
# A handful of public `Distribution` subtypes are deliberately EXCLUDED from the
# `test_interface` registry, with a documented reason each:
#   - `NoEvent`        : an absorbing no-event MARKER, not a scorable delay; it
#                        only appears as a `Resolve`/`thin` branch (covered via
#                        the defective Resolve / thin fixtures).
#   - `Shared`         : a name-tag wrapper tying a leaf across branches; it is an
#                        introspection tag with no standalone scalar interface.
#   - `EventRecord`    : a per-record baked metadata carrier, exercised by the
#                        record-distributions tests, not a leaf.
#   - `PrimaryConditional` : the inverse-latent conditional, exercised by the
#                        primary-conditional tests; not a composer leaf.
# The set below is the registry the meta-test enforces.
@doc """

The public distribution / leaf / node types the fixture registry must cover.

`registry_types()` returns the `Vector` of the package's own public
`Distribution` types that [`test_interface`](@ref) is expected to exercise (the
composer shapes, the censoring leaves and the distribution-modifier / derived
leaves). [`test_registry_coverage`](@ref) asserts every entry appears in at least
one [`example_fixtures`](@ref) fixture, so a new public type added without a
fixture fails. A few infrastructure `Distribution` subtypes (`NoEvent`, `Shared`,
`EventRecord`, `PrimaryConditional`) are deliberately excluded, each with a
documented reason in the source.
""" function registry_types()
    return Type[
        # composer shapes
        Sequential, Parallel, Resolve, Compete, Choose,
        # censoring leaves
        PrimaryCensored, IntervalCensored, Latent,
        # distribution-modifier / derived leaves
        Affine, Modified, Weighted, Transformed, Convolved, Difference,
        ExponentiallyTilted, MomentParams
    ]
end

# Every concrete type appearing in a fixture's distribution, walked recursively
# through the composer children, censoring wrappers and modifier inners, so a
# type nested deep in a tree still counts as covered.
function _covered_types(fixtures)
    seen = Set{Type}()
    for fix in fixtures
        _collect_types!(seen, fix.dist)
    end
    return seen
end

function _collect_types!(seen, d)
    push!(seen, typeof(d))
    # Composer children.
    if d isa Union{Sequential, Parallel}
        for c in d.components
            _collect_types!(seen, c)
        end
    elseif d isa AbstractOneOf
        for c in d.delays
            _collect_types!(seen, c)
        end
    elseif d isa Choose
        for c in d.alternatives
            _collect_types!(seen, c)
        end
    elseif d isa Latent
        _collect_types!(seen, d.dist)
    end
    # Wrapper / modifier inners reached through the public `free_leaf` peel and
    # the specific inner fields, so a censored / affine / modified / weighted /
    # convolved / difference inner counts too.
    _collect_inner_types!(seen, d)
    return nothing
end

function _collect_inner_types!(seen, d)
    for f in (:dist, :d, :x, :y)
        if hasproperty(d, f)
            inner = getproperty(d, f)
            inner isa Distributions.Distribution && _collect_types!(seen, inner)
        end
    end
    # Convolved holds a tuple of components.
    if d isa Convolved
        for c in d.components
            _collect_types!(seen, c)
        end
    end
    return nothing
end

@doc """

Assert the fixture registry covers every public distribution / leaf type.

`test_registry_coverage(fixtures = example_fixtures())` checks that every type in
[`registry_types`](@ref) appears (possibly nested) in at least one fixture, so a
NEW public distribution type added without a `test_interface` fixture fails here.
The walk descends composer children, censoring wrappers and modifier inners.
Returns the `@testset` object.
""" function test_registry_coverage(fixtures = example_fixtures())
    covered = _covered_types(fixtures)
    # A covered concrete type matches a registry entry if it is that type or a
    # parametric instance of it (`PrimaryCensored{...} <: PrimaryCensored`).
    is_covered(T) = any(c -> c <: T, covered)
    return @testset "fixture registry covers every public type" begin
        for T in registry_types()
            @test is_covered(T)
        end
    end
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
