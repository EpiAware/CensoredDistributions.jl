# Shared setup for the AD gradient test items. Scenarios and backend
# metadata come from the ADFixtures path package at `test/ADFixtures`,
# shared with the benchmark suite (`benchmark/src/ad_gradients.jl`) and the
# docs tutorial. Correctness is driven by
# `DifferentiationInterfaceTest.test_differentiation` against a ForwardDiff
# reference stored in each scenario's `res1` field.
#
# The generic run logic (working/partial backends, broken bookkeeping) lives
# in `EpiAwareTestUtils`; `ADFixtures` is the package-specific registry it
# drives (it satisfies the `ADRegistry` contract: `scenarios`, `backends`,
# `broken_scenario_names`, `backend_broken_scenarios`,
# `backend_skip_scenarios`). The thin wrappers below pin the CD-specific
# `rtol`/`atol`, the `scenario_intact = false` workaround, and the
# marginal/latent scenario-group selector, so the test items in
# `scenarios.jl` stay one-liners.

@testsnippet ADHelpers begin
    using ADTypes
    using DifferentiationInterface
    import DifferentiationInterfaceTest as DIT
    using ADFixtures
    using ForwardDiff, ReverseDiff, Enzyme, Mooncake
    using EpiAwareTestUtils: EpiAwareTestUtils

    # CD's reference tolerance, shared by the working and partial paths.
    const AD_RTOL = 5e-2
    const AD_ATOL = 1e-6

    # A working backend: hard correctness test on the scenarios it supports,
    # `@test_broken` on its known-broken scenarios (none today). `category`
    # selects the scenario group and is forwarded to `ADFixtures.scenarios`:
    # `:marginal` (default) keeps the marginal AD sweep purely marginal, while
    # `:latent` runs the latent / augmented-primary group (`Latent*`,
    # `PrimaryConditional`). See `scenarios.jl` for the per-group test items.
    #
    # `scenario_intact = false`: some scenarios carry a `Missing`-bearing event
    # vector as a `Constant` context (the censored-composer marginalisation
    # path, #333). DIT's default post-run equality check compares the scenario
    # structs with `==`, and comparing a vector that contains `missing` returns
    # `missing`, which `==` then uses in a boolean context and errors. The
    # gradients themselves are correct; only the intactness check trips, so it
    # is disabled. Other scenarios are unaffected.
    function test_working_backend(name; category::Symbol = :marginal)
        EpiAwareTestUtils.test_working_backend(
            ADFixtures, name;
            rtol = AD_RTOL, atol = AD_ATOL,
            scenario_intact = false,
            scenario_kwargs = (; category = category))
    end

    # A partial backend (#225): every scenario through the harness's
    # `check_broken`, so the supported subset passes and the rest are marked
    # broken.
    function test_partial_backend(name)
        EpiAwareTestUtils.test_partial_backend(
            ADFixtures, name; rtol = AD_RTOL, atol = AD_ATOL)
    end
end
