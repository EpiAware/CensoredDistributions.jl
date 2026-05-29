#!/usr/bin/env julia
# AD gradient tests for CensoredDistributions.
# Run with `task test-ad` or `julia --project=test/ad test/ad/runtests.jl`.
#
# Scenarios and backend lists are sourced from the ADFixtures path
# package at `test/ADFixtures`. Correctness is driven by
# `DifferentiationInterfaceTest.test_differentiation` against a
# ForwardDiff reference stored in each scenario's `res1` field;
# every other backend is checked for self-consistency against it.

using Test
using CensoredDistributions
using ADTypes
using DifferentiationInterface
using ForwardDiff
using ReverseDiff
using Enzyme
using Mooncake
using ADFixtures
import DifferentiationInterfaceTest as DIT

# Optional single-backend selection for the per-backend AD CI (#269).
# `CENSORED_AD_BACKEND` matches a name from `ADFixtures.backends()`
# (e.g. "Enzyme reverse"); unset or "all" runs every backend, which is
# what `task test-ad` does locally. Running one backend per CI job means
# a transiently unstable backend only reds its own badge.
const AD_SELECTED = get(ENV, "CENSORED_AD_BACKEND", "all")

_selected(name) = AD_SELECTED == "all" || AD_SELECTED == name

# Which `gamma_ad.jl` unit blocks to run for the selected backend.
# Backend-agnostic blocks (series accuracy, rrule guards) are tagged
# "core" and attached to the ForwardDiff run so they execute exactly
# once across the per-backend jobs.
function ad_unit_runs(family)
    AD_SELECTED == "all" && return true
    family == "core" && return AD_SELECTED == "ForwardDiff"
    return startswith(AD_SELECTED, family)
end

function check_broken(scenarios_list, backend)
    # Scenarios that `DIT.test_differentiation` cannot exercise — the
    # backend errors throughout (e.g. Enzyme, #225), the scenario is
    # globally broken (#217), or it errors for one backend only (e.g.
    # Mooncake forward on the `_gamma_cdf` path, #270). For each
    # scenario we try plain DI:
    #   - succeeds + matches reference → @test (passes; this is
    #     downgraded-but-working coverage)
    #   - fails or mismatches → @test_broken (the genuine broken case)
    for scen in scenarios_list
        ok = try
            g = DifferentiationInterface.gradient(scen.f, backend, scen.x)
            ref = scen.res1
            g isa AbstractVector && all(isfinite, g) && ref !== nothing &&
                isapprox(g, ref; rtol = 5e-2, atol = 1e-6)
        catch
            false
        end
        if ok
            @test ok
        else
            @test_broken ok
        end
    end
end

@testset "AD gradient scenarios" begin
    all_scenarios = ADFixtures.scenarios(with_reference = true)
    global_broken = Set(ADFixtures.broken_scenario_names())
    backend_broken = ADFixtures.backend_broken_scenarios()

    working = filter(e -> _selected(e.name), ADFixtures.working_backends())
    broken = filter(e -> _selected(e.name), ADFixtures.broken_backends())

    for entry in working
        per_backend = get(backend_broken, entry.name, Set{String}())
        ok = filter(
            s -> !(s.name in global_broken) && !(s.name in per_backend),
            all_scenarios)
        broken = filter(
            s -> s.name in global_broken || s.name in per_backend,
            all_scenarios)

        @testset "$(entry.name)" begin
            @testset "working scenarios" begin
                DIT.test_differentiation(
                    [entry.backend], ok;
                    correctness = true,
                    type_stability = :none,
                    logging = false,
                    rtol = 5e-2,
                    atol = 1e-6
                )
            end
            @testset "broken scenarios" begin
                check_broken(broken, entry.backend)
            end
        end
    end

    @testset "fully broken backends (#225)" begin
        for entry in broken
            check_broken(all_scenarios, entry.backend)
        end
    end
end

# Unit-level AD coverage that complements the scenario suite above:
# series baseline, Mooncake.TestUtils.test_rule, and defensive guards.
include(joinpath(@__DIR__, "gamma_ad.jl"))
