# Shared setup for the AD gradient test items. Scenarios and backend
# metadata come from the ADFixtures path package at `test/ADFixtures`,
# shared with the benchmark suite (`benchmark/src/ad_gradients.jl`) and the
# docs tutorial. Correctness is driven by
# `DifferentiationInterfaceTest.test_differentiation` against a ForwardDiff
# reference stored in each scenario's `res1` field.
#
# The generic run logic (working/partial backends, broken bookkeeping) lives
# in `EpiAwarePackageTools`; `ADFixtures` is the package-specific registry it
# drives. CD `main`'s `ADFixtures` satisfies the `ADRegistry` contract via
# `scenarios`, `backends`, `broken_scenario_names`, and
# `backend_broken_scenarios`; it defines no `backend_skip_scenarios`, which
# the harness treats as "no skipped scenarios". The thin wrappers below pin
# the CD-specific `rtol`/`atol` and the `scenario_intact = false` workaround,
# so the test items in `scenarios.jl` stay one-liners.

@testsnippet ADHelpers begin
    using ADTypes
    using DifferentiationInterface
    import DifferentiationInterfaceTest as DIT
    using ADFixtures
    using ForwardDiff, ReverseDiff, Enzyme, Mooncake
    using EpiAwarePackageTools: EpiAwarePackageTools

    # CD's reference tolerance, shared by the working and partial paths.
    const AD_RTOL = 5e-2
    const AD_ATOL = 1e-6

    # A working backend: hard correctness test on the scenarios it supports,
    # `@test_broken` on its known-broken scenarios (none today).
    #
    # `scenario_intact = false`: some scenarios carry a `Missing`-bearing
    # context vector as a `Constant`. DIT's default post-run equality check
    # compares the scenario structs with `==`, and comparing a vector that
    # contains `missing` returns `missing`, which `==` then uses in a boolean
    # context and errors. The gradients themselves are correct; only the
    # intactness check trips, so it is disabled. Other scenarios are
    # unaffected.
    function test_working_backend(name)
        EpiAwarePackageTools.test_working_backend(
            ADFixtures, name;
            rtol = AD_RTOL, atol = AD_ATOL,
            scenario_intact = false)
    end

    # A partial backend (#225): every scenario through the harness's
    # `check_broken`, so the supported subset passes and the rest are marked
    # broken.
    function test_partial_backend(name)
        EpiAwarePackageTools.test_partial_backend(
            ADFixtures, name; rtol = AD_RTOL, atol = AD_ATOL)
    end
end
