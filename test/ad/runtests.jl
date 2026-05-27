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

function check_broken(scenarios_list, backend)
    # Scenarios that `DIT.test_differentiation` cannot exercise — either
    # because the backend errors throughout, or because the backend
    # works via plain `DifferentiationInterface.gradient` but the
    # DIT-specific prepare path errors (observed with Mooncake on
    # primary-censored Gamma/Weibull scenarios). For each scenario we
    # try plain DI:
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

    for entry in ADFixtures.working_backends()
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
        for entry in ADFixtures.broken_backends()
            check_broken(all_scenarios, entry.backend)
        end
    end
end

# Unit-level AD coverage that complements the scenario suite above:
# series baseline, Mooncake.TestUtils.test_rule, and defensive guards.
include(joinpath(@__DIR__, "gamma_ad.jl"))
