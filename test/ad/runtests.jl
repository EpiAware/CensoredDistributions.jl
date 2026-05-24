#!/usr/bin/env julia
# AD gradient tests for CensoredDistributions.
# Run with `task test-ad` or `julia --project=test/ad test/ad/runtests.jl`.
#
# Correctness is driven by `DifferentiationInterfaceTest.test_differentiation`
# against a fixed-step central-difference reference stored in each scenario's
# `res1` field.

using Test
using CensoredDistributions
using Distributions
using ADTypes
using DifferentiationInterface
using ForwardDiff
using ReverseDiff
using Enzyme
using Mooncake
import DifferentiationInterfaceTest as DIT

include(joinpath(@__DIR__, "..", "..", "docs", "src", "getting-started",
    "tutorials", "ad_scenarios.jl"))

function check_broken(scenarios, backend)
    for scen in scenarios
        ok = try
            g = DifferentiationInterface.gradient(scen.f, backend, scen.x)
            g isa AbstractVector && all(isfinite, g) &&
                isapprox(g, scen.res1; rtol = 5e-2, atol = 1e-6)
        catch
            false
        end
        @test_broken ok
    end
end

@testset "AD gradients" begin
    all_scenarios = ad_scenarios(with_reference = true)
    global_broken = Set(ad_broken_scenario_names())
    backend_broken = ad_backend_broken_scenarios()

    for entry in ad_working_backends()
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
        for entry in ad_broken_backends()
            check_broken(all_scenarios, entry.backend)
        end
    end
end
