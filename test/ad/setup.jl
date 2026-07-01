# MANAGED by EpiAwarePackageTools.scaffold — do not edit by hand.
#
# AD-harness driver. Wires the shared EpiAwarePackageTools AD harness to the
# package's own AD-fixture registry (`ADFixtures` by convention), then exposes
# `test_working_backend` / `test_partial_backend` / `check_broken` as thin
# locals the scenario test items call. The registry — the actual scenarios,
# backend list, and broken/skip bookkeeping — is PACKAGE-OWNED (see
# `test/ad/scenarios.jl` and the package's `test/ADFixtures` registry); only
# this wiring is standard.

@testsnippet ADHelpers begin
    using ADTypes
    using DifferentiationInterface
    import DifferentiationInterfaceTest as DIT
    using EpiAwarePackageTools
    # The package's AD-fixture registry satisfying the `ADRegistry` contract.
    using ADFixtures
    # Backends the package tests; trim to those the package actually uses.
    using ForwardDiff, ReverseDiff, Enzyme, Mooncake

    const REG = ADFixtures

    # Drive a working backend over the registry's scenarios for a category.
    function test_working_backend(name; category::Symbol = :marginal)
        EpiAwarePackageTools.test_working_backend(REG, name;
            scenario_kwargs = (; category = category))
    end

    # Drive a partial backend (every scenario through `check_broken`).
    function test_partial_backend(name)
        EpiAwarePackageTools.test_partial_backend(REG, name)
    end

    # Re-export the shared `check_broken` for any bespoke scenario item.
    const check_broken = EpiAwarePackageTools.check_broken
end
