# Per-extension in-process method-ambiguity checks (#654).
#
# `Aqua.test_ambiguities(CensoredDistributions)` (see `Aqua.jl`) runs in a
# SUBPROCESS with NO package extensions loaded, so it never exercises any
# extension's method table. #653 added an in-process `detect_ambiguities` check
# for the SurvivalDistributions extension specifically (in
# `test/integration/SurvivalDistributions.jl`); this file GENERALISES that to
# every extension whose trigger packages the main test env can load — loading
# the trigger package(s), then asserting the extension introduces no method
# ambiguity ON THE PACKAGE'S OWN SURFACE.
#
# # The on-surface filter (and why it is needed)
#
# `detect_ambiguities(CensoredDistributions, ext)` reports EVERY ambiguous pair
# in the combined method table, including pairs whose collision is owned by a
# THIRD-PARTY package, not by this package or its extension. The dominant such
# phantom is `SymbolicsDistributionsExt` (pulled in transitively by Catalyst ->
# Symbolics): it defines `logpdf/cdf/logcdf/quantile(::Distribution, ::Num)`,
# which is ambiguous with EVERY concrete `f(::SomeConcreteDist, ::Real)` method
# in Distributions and in this package (a `Num`/`Real` overlap). Those are
# pre-existing phantoms of the Symbolics integration — they would collide with
# any package defining such a method — and are out of scope here.
#
# The filter therefore keeps only pairs where BOTH methods live on the target
# surface (this package, its extension, or the trigger package itself), which
# excludes the `SymbolicsDistributionsExt`-owned phantom side while still
# catching a real ambiguity the package or its extension introduces. This is the
# same scoping the SurvivalDistributions check in #653 uses, lifted to a shared
# helper here.
#
# # Quarantined REAL findings (do NOT weaken the contract to hide them)
#
# Two extensions DO carry a real, extension-only ambiguity that Aqua cannot see.
# They are surfaced as issues and QUARANTINED as `@test_broken` (referencing the
# issue), not silenced:
#   - ForwardDiff: 6 ambiguities in the `_gamma_cdf` `Dual` overload set
#     (`ext/CensoredDistributionsForwardDiffExt.jl`) — issue #672.
#   - DynamicPPL: a reachable ambiguity in `composed_distribution_model` over a
#     `latent(primary_censored(...))` batch (a `MethodError` at the user API) —
#     issue #673.
# Flipping either fix turns its `@test_broken` green; that is the signal the bug
# is gone.

@testsnippet ExtAmbiguityHelper begin
    using CensoredDistributions
    using Test: detect_ambiguities

    # The method's defining module name starts with one of the allowed surface
    # prefixes (this package + its extensions all start "CensoredDistributions";
    # a trigger package is named explicitly).
    function _on_surface(m, prefixes)
        mn = string(m.module)
        return any(p -> startswith(mn, p), prefixes)
    end

    # The TOTAL (unfiltered) ambiguity count over `(CensoredDistributions, ext)`,
    # used by the Catalyst check to prove the Symbolics phantoms are present (so
    # the on-surface filter is doing real work, not trivially empty).
    function raw_ambiguity_count(extname::Symbol)
        ext = Base.get_extension(CensoredDistributions, extname)
        ext === nothing && error("extension $extname is not loaded")
        return length(
            detect_ambiguities(CensoredDistributions, ext; recursive = false))
    end

    # The ambiguous pairs over `(CensoredDistributions, ext)` that this package
    # or its extension OWNS — both methods on the allowed surface, so a
    # third-party phantom (e.g. `SymbolicsDistributionsExt`'s `::Num` methods) is
    # filtered out. `extname` is the extension module symbol; `prefixes` the
    # allowed surface module-name prefixes (always includes
    # "CensoredDistributions").
    function on_surface_ambiguities(extname::Symbol,
            prefixes::Vector{String} = ["CensoredDistributions"])
        ext = Base.get_extension(CensoredDistributions, extname)
        ext === nothing && error("extension $extname is not loaded")
        amb = detect_ambiguities(CensoredDistributions, ext; recursive = false)
        return filter(
            p -> _on_surface(p[1], prefixes) && _on_surface(p[2], prefixes),
            amb)
    end
end

@testitem "ext ambiguities: Integrals" tags=[:quality] setup=[ExtAmbiguityHelper] begin
    import Integrals

    # The Integrals extension adds the pluggable `integrate` method; it
    # introduces no ambiguity on the package + Integrals surface.
    @test isempty(on_surface_ambiguities(
        :CensoredDistributionsIntegralsExt,
        ["CensoredDistributions", "Integrals"]))
end

@testitem "ext ambiguities: SurvivalDistributions" tags=[:quality] setup=[ExtAmbiguityHelper] begin
    import SurvivalDistributions

    # The SurvivalDistributions extension adds the AD-safe GeneralizedGamma
    # `logcdf` routing; clean on the package + SurvivalDistributions +
    # Distributions surface (the original #653 check, now via the shared helper).
    @test isempty(on_surface_ambiguities(
        :CensoredDistributionsSurvivalDistributionsExt,
        ["CensoredDistributions", "SurvivalDistributions", "Distributions"]))
end

@testitem "ext ambiguities: FlexiChains" tags=[:quality] setup=[ExtAmbiguityHelper] begin
    import DynamicPPL, FlexiChains

    # The FlexiChains extension adds the vectorised chain readers; clean on the
    # package surface.
    @test isempty(
        on_surface_ambiguities(:CensoredDistributionsFlexiChainsExt))
end

@testitem "ext ambiguities: ChainRulesCore + Mooncake (AD)" tags=[:quality] setup=[ExtAmbiguityHelper] begin
    import Mooncake  # pulls in ChainRulesCore, triggering both extensions

    # The reverse-mode AD extensions (the ChainRules rule and the Mooncake
    # wrapper) are clean on the package surface.
    @test isempty(
        on_surface_ambiguities(:CensoredDistributionsChainRulesCoreExt))
    @test isempty(
        on_surface_ambiguities(:CensoredDistributionsMooncakeExt))
end

@testitem "ext ambiguities: Catalyst (Symbolics phantoms filtered)" tags=[:quality] setup=[ExtAmbiguityHelper] begin
    import Catalyst

    # Catalyst transitively loads Symbolics and its `SymbolicsDistributionsExt`,
    # which defines `logpdf/cdf/...(::Distribution, ::Num)` — ambiguous with
    # every concrete `f(::SomeDist, ::Real)` method (a `Num`/`Real` overlap), a
    # pre-existing phantom of the Symbolics integration, NOT introduced by this
    # extension. The raw count is large; the on-surface filter (both methods on
    # the package surface) removes the phantom side and asserts the Catalyst
    # bridge itself adds no ambiguity.
    #
    # Sanity first: there ARE phantom pairs (the Symbolics integration is
    # loaded), so the on-surface filter is doing real work, not trivially empty.
    @test raw_ambiguity_count(:CensoredDistributionsCatalystExt) > 0
    @test isempty(
        on_surface_ambiguities(:CensoredDistributionsCatalystExt))
end

@testitem "ext ambiguities: ForwardDiff (quarantined, issue #672)" tags=[:quality] setup=[ExtAmbiguityHelper] begin
    import ForwardDiff

    # REAL finding (issue #672): the `_gamma_cdf` `Dual` overload set in
    # `ext/CensoredDistributionsForwardDiffExt.jl` has 6 method ambiguities in
    # its partial-Dual signatures. They are benign on the happy path (the
    # all-Dual covering methods resolve concrete same-tag calls, and the
    # `test/ad` ForwardDiff correctness suite passes), but the method table is
    # pairwise-ambiguous. Quarantined as `@test_broken` rather than hidden;
    # collapsing the overloads (#672) turns this green.
    @test_broken isempty(on_surface_ambiguities(
        :CensoredDistributionsForwardDiffExt,
        ["CensoredDistributions", "ForwardDiff"]))
end

@testitem "ext ambiguities: DynamicPPL (quarantined, issue #673)" tags=[:quality] setup=[ExtAmbiguityHelper] begin
    import DynamicPPL

    # REAL finding (issue #673): `composed_distribution_model` over a
    # `latent(primary_censored(...))` and a vector of rows matches both the
    # single-record `Latent{<:PrimaryCensored}` method and the batch
    # `Latent, ::AbstractVector` method ambiguously — a reachable `MethodError`
    # at the user API. Quarantined as `@test_broken`; the disambiguating batch
    # method (#673) turns this green.
    @test_broken isempty(
        on_surface_ambiguities(:CensoredDistributionsDynamicPPLExt))
end
