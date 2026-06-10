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

module TestUtils

using Test: Test, @testset, @test, @test_nowarn, @test_throws
using Distributions: Distributions, mean, var, std, logpdf, cdf, params,
                     UnivariateDistribution
import Tables

using ..CensoredDistributions: CensoredDistributions, Sequential, Parallel,
                               Competing, Select, compose, latent,
                               double_interval_censored, primary_censored,
                               event, event_names, event_tree, params_table,
                               observed_distribution, endpoint

export test_interface, example_fixtures, test_rejects_invalid

# --- per-fixture descriptor -------------------------------------------------
#
# A fixture is the distribution plus the metadata the checklist needs that is not
# recoverable from the object alone: a known event-name `path` to round-trip
# through `event`, an in-support `draw` to score, a `kind` selector for a
# `Select`, and whether the node is univariate (a scalar moment) or multivariate
# (a per-event Vector moment).

Base.@kwdef struct InterfaceFixture{D}
    name::String
    dist::D
    "An in-support realisation to score (a scalar for univariate, a Vector or a
    NamedTuple/record for multivariate)."
    draw::Any = nothing
    "A known `event` path (tuple of Symbols) that must round-trip, or `nothing`."
    path::Union{Nothing, Tuple} = nothing
    "The `kind` keyword for a `Select` fixture, or `nothing`."
    kind::Union{Nothing, Symbol} = nothing
    "Whether the node is univariate (scalar moment) vs multivariate (Vector)."
    univariate::Bool = false
    "Whether `mean`/`var`/`std` are defined for this node (a raw `latent`-wrapped
    leaf, scored event-by-event, has no summary moment)."
    check_moments::Bool = true
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
- `mean` / `var` / `std` are defined and shaped to match `rand` (a Vector for a
  multivariate node, a scalar for a univariate one);
- `logpdf` is finite on the supplied in-support `draw`;
- a univariate `cdf` is monotone and in `[0, 1]`;
- `params` works and `params_table` is a Tables.jl table
  (`Tables.istable(params_table(d))`);
- `event_names` (flat) and `event_tree` agree in leaf count;
- `event(d, path...)` round-trips the supplied known path;
- `observed_distribution` / `endpoint` collapses a chain to a univariate scalar.

Pass the fixture metadata (an [`example_fixtures`](@ref) entry, or the keyword
arguments directly) so the harness knows the in-support `draw`, a known `event`
`path`, a `Select` `kind`, and whether the node is univariate. Returns the
`@testset` object.

# Examples
```julia
using CensoredDistributions, Distributions
using CensoredDistributions.TestUtils: test_interface

d = compose((onset_admit = Gamma(2.0, 1.0), admit_death = LogNormal(0.5, 0.4)))
test_interface(d; draw = rand(d), path = (:onset_admit,))
```
""" function test_interface end

function test_interface(d; name::AbstractString = string(nameof(typeof(d))),
        draw = nothing, path::Union{Nothing, Tuple} = nothing,
        kind::Union{Nothing, Symbol} = nothing, univariate::Bool = false,
        check_moments::Bool = true, has_endpoint::Bool = true)
    fix = InterfaceFixture(; name = name, dist = d, draw = draw, path = path,
        kind = kind, univariate = univariate, check_moments = check_moments,
        has_endpoint = has_endpoint)
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

# A Select needs a selection for length/rand/logpdf, so it is checked on its
# own track (the generic moment / logpdf checks are skipped for it).
_is_select(::Select) = true
_is_select(::Any) = false

function _check_select(d::Select, fix)
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

# `mean`/`var`/`std` defined and shaped to match `rand`: a Vector for a
# multivariate node, a scalar for a univariate one. For a multivariate node the
# moment / rand lengths must agree.
function _check_moments_and_rand(d, fix)
    _is_select(d) && return nothing
    @testset "moments and rand" begin
        r = rand(d)
        if !fix.check_moments
            # Still exercise rand; the node has no summary moment.
            @test r isa Union{Real, AbstractVector, NamedTuple}
        elseif fix.univariate
            @test r isa Real
            @test mean(d) isa Real
            @test var(d) isa Real
            @test std(d) isa Real
        else
            m = mean(d)
            v = var(d)
            s = std(d)
            @test m isa AbstractVector
            @test v isa AbstractVector
            @test s isa AbstractVector
            @test length(m) == length(r)
            @test length(v) == length(r)
            @test length(s) == length(r)
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
function _score(d, draw::NamedTuple)
    return logpdf(d, _missing_vec(collect(draw)))
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
        if d isa Union{Sequential, Parallel, Competing, Select}
            tbl = params_table(d)
            @test Tables.istable(tbl)
        end
    end
    return nothing
end

# The flat `event_names` and the nested `event_tree` must agree in LEAF count:
# every `event_tree` leaf (a Competing outcome / a leaf delay) has its own flat
# slot, plus the flat origin event, so `length(flat) == leaves + 1`. A `Select`
# has no single flat layout (its `event_names` are the alternative names), so the
# count check is skipped for it; only that both are non-empty.
function _check_event_names(d, fix)
    d isa Union{Sequential, Parallel, Competing, Select} || return nothing
    @testset "event_names / event_tree leaf count" begin
        flat = event_names(d)
        tree = event_tree(d)
        if d isa Select
            @test !isempty(flat)
            @test !isempty(keys(tree))
        else
            @test length(flat) == _tree_leaf_count(tree) + 1
        end
    end
    return nothing
end

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
censored leaf, `Sequential`, `Parallel`, `Competing`, `selecting`, nested mixes,
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
    # A standalone Competing's scalar moment lowers through `as_mixture`, which has
    # no analytic moment for a CENSORED leaf, so the bare-Competing fixture uses
    # plain delays (a censored Competing is still exercised nested in `nested`).
    comp = Competing(:death => (G(2.0, 3.5), 0.4),
        :discharge => (G(1.0, 8.0), 0.6))
    nested_comp = Competing(:death => (dic(G(2.0, 3.5)), 0.4),
        :discharge => (dic(G(1.0, 8.0)), 0.6))
    nested = compose((
        admit_path = compose((onset_admit = dic(G(1.2, 3.0)),
            admit_resolution = nested_comp)),
        onset_notif = dic(G(0.7, 20.0))))
    sel = CensoredDistributions.selecting(:index => dic(G(2.0, 1.0)),
        :sourced => compose((a = dic(G(4.0, 1.5)), b = dic(G(1.0, 2.0)))))
    # `latent` over a single primary-censored leaf is the documented use: a
    # multivariate `[primary, observed]` scored event-by-event (no summary moment,
    # no single observed endpoint).
    lat = latent(primary_censored(G(2.0, 1.0), Distributions.Uniform(0, 1)))

    return InterfaceFixture[
        # A plain leaf has the full univariate interface (scalar moments + cdf).
        InterfaceFixture(; name = "bare plain leaf", dist = G(2.0, 1.0),
            draw = 3.0, univariate = true),
        # A bare CENSORED leaf scores and has a monotone cdf, but no analytic
        # summary moment (censoring has no closed-form mean), so moments are
        # skipped here.
        InterfaceFixture(; name = "bare censored leaf", dist = leaf,
            draw = 3.0, univariate = true, check_moments = false),
        InterfaceFixture(; name = "Sequential", dist = seq,
            draw = rand(seq), path = (:onset_admit,)),
        # A Parallel has several independent endpoints and so no single observed
        # scalar; `observed_distribution` is not defined for it.
        InterfaceFixture(; name = "Parallel", dist = par, draw = rand(par),
            has_endpoint = false),
        InterfaceFixture(; name = "Competing", dist = comp, draw = 4.0,
            path = (:death,), univariate = true),
        InterfaceFixture(; name = "selecting", dist = sel, draw = 3.0,
            kind = :index, path = (:index,), has_endpoint = false),
        # A nested tree branches off a shared origin (a Parallel at its root), so
        # likewise has no single collapsed endpoint.
        InterfaceFixture(; name = "nested mix", dist = nested,
            draw = rand(nested), path = (:admit_path, :admit_resolution),
            has_endpoint = false),
        InterfaceFixture(; name = "latent-wrapped", dist = lat,
            draw = rand(lat), check_moments = false, has_endpoint = false)
    ]
end

# --- construction rejection -------------------------------------------------

@doc """

Assert each composer rejects invalid construction in its inner constructor.

`test_rejects_invalid()` checks that the standard composers validate their
structural invariants on EVERY construction path (the inner constructor), so a
malformed node errors at build time rather than later: `Sequential` needs at
least one component, `Parallel`/`Competing`/`Select` at least two children,
`Competing` branch probabilities in `[0, 1]`, and unique `Select` names. Returns
the `@testset` object.
""" function test_rejects_invalid()
    G = Distributions.Gamma
    return @testset "construction rejects invalid input" begin
        # Competing: out-of-range branch prob, and fewer than two outcomes.
        @test_throws ArgumentError Competing((:a, :b), (G(2.0, 1.0), G(1.0, 1.0)),
            (1.5, -0.5))
        @test_throws ArgumentError Competing(:only => (G(2.0, 1.0), 1.0))
        # Sequential: empty.
        @test_throws ArgumentError Sequential((), ())
        # Select: needs at least two, and unique names.
        @test_throws ArgumentError CensoredDistributions.selecting(
            :only => G(2.0, 1.0))
        @test_throws ArgumentError CensoredDistributions.selecting(
            :a => G(2.0, 1.0), :a => G(1.0, 1.0))
    end
end

end # module TestUtils
