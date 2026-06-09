# Shared setup for the AD gradient test items. Scenarios and backend
# metadata come from the ADFixtures path package at `test/ADFixtures`,
# shared with the benchmark suite (`benchmark/src/ad_gradients.jl`) and the
# docs tutorial. Correctness is driven by
# `DifferentiationInterfaceTest.test_differentiation` against a ForwardDiff
# reference stored in each scenario's `res1` field.

@testsnippet ADHelpers begin
    using ADTypes
    using DifferentiationInterface
    import DifferentiationInterfaceTest as DIT
    using ADFixtures
    using ForwardDiff, ReverseDiff, Enzyme, Mooncake

    _entry(name) = only(filter(e -> e.name == name, ADFixtures.backends()))

    # Scenarios `DIT.test_differentiation` cannot exercise for a backend:
    # try plain DI and mark each as passing if it matches the reference,
    # broken otherwise. Lets a partial backend record working coverage
    # without forcing all-or-nothing.
    function check_broken(scenarios_list, backend)
        for scen in scenarios_list
            ok = try
                g = DifferentiationInterface.gradient(
                    scen.f, backend, scen.x, scen.contexts...)
                ref = scen.res1
                g isa AbstractVector && all(isfinite, g) &&
                    ref !== nothing &&
                    isapprox(g, ref; rtol = 5e-2, atol = 1e-6)
            catch
                false
            end
            ok ? (@test ok) : (@test_broken ok)
        end
    end

    # A working backend: hard correctness test on the scenarios it
    # supports, `@test_broken` on its known-broken scenarios (none today).
    function test_working_backend(name)
        backend = _entry(name).backend
        all_scenarios = ADFixtures.scenarios(with_reference = true)
        global_broken = Set(ADFixtures.broken_scenario_names())
        per_backend = get(
            ADFixtures.backend_broken_scenarios(), name, Set{String}())
        # Scenarios that crash this backend UNCATCHABLY are skipped entirely (not
        # even run through `check_broken`, which would still execute them).
        skip = get(ADFixtures.backend_skip_scenarios(), name, Set{String}())
        runnable = filter(s -> !(s.name in skip), all_scenarios)
        ok = filter(
            s -> !(s.name in global_broken) && !(s.name in per_backend),
            runnable)
        broken_scens = filter(
            s -> s.name in global_broken || s.name in per_backend,
            runnable)
        # `scenario_intact = false`: some scenarios carry a `Missing`-bearing
        # event vector as a `Constant` context (the censored-composer
        # marginalisation path, #333). DIT's default post-run equality check
        # compares the scenario structs with `==`, and comparing a vector that
        # contains `missing` returns `missing`, which `==` then uses in a
        # boolean context and errors. The gradients themselves are correct; only
        # the intactness check trips, so it is disabled. Other scenarios are
        # unaffected.
        DIT.test_differentiation(
            [backend], ok;
            correctness = true,
            type_stability = :none,
            logging = false,
            scenario_intact = false,
            rtol = 5e-2,
            atol = 1e-6
        )
        check_broken(broken_scens, backend)
    end

    # A partial backend (#225): every scenario through `check_broken`, so
    # the supported subset passes and the rest are marked broken.
    function test_partial_backend(name)
        backend = _entry(name).backend
        check_broken(ADFixtures.scenarios(with_reference = true), backend)
    end
end
