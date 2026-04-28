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

include(joinpath(@__DIR__, "scenarios.jl"))

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
    broken_names = Set(ad_broken_scenario_names())
    working_scenarios = filter(s -> !(s.name in broken_names), all_scenarios)
    broken_scenarios = filter(s -> s.name in broken_names, all_scenarios)
    working = [entry.backend for entry in ad_working_backends()]
    broken = [entry.backend for entry in ad_broken_backends()]

    @testset "working backends × working scenarios" begin
        DIT.test_differentiation(
            working, working_scenarios;
            correctness = true,
            type_stability = :none,
            logging = false,
            rtol = 5e-2,
            atol = 1e-6
        )
    end

    @testset "working backends × broken scenarios (#217)" begin
        for backend in working
            check_broken(broken_scenarios, backend)
        end
    end

    @testset "broken backends × all scenarios (#225)" begin
        for backend in broken
            check_broken(all_scenarios, backend)
        end
    end
end
