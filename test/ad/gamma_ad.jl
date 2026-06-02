# Unit-level AD coverage for the gamma CDF rrule. Complements the
# scenario suite in `scenarios.jl`: those check end-to-end gradient
# agreement; these pin the implementation-level guarantees (series
# accuracy, Mooncake rule structure, defensive guards) that no scenario
# exercises directly.
#
# Each item is mode-agnostic (it exercises both modes of a backend's
# rules), so it is tagged for a single canonical backend and runs once
# across the per-backend CI jobs; the untagged `task test-ad` run executes
# every item.

@testitem "_grad_p_a_series matches FiniteDifferences" tags=[:ad, :forwarddiff] begin
    using SpecialFunctions: gamma_inc
    using FiniteDifferences: central_fdm
    using DifferentiationInterface: AutoFiniteDifferences, derivative
    using CensoredDistributions: _grad_p_a_series

    fd = AutoFiniteDifferences(; fdm = central_fdm(7, 1))

    # Restrict grid to (a, z) values where ∂P/∂a is large enough that the
    # finite-difference baseline is meaningful (i.e. away from
    # P-saturation). Includes k≪1 cases — the singular regime that breaks
    # naive implementations of the shape-parameter derivative. Lower bound
    # on a is set by the fdm probe radius (`central_fdm(7, 1)` reaches into
    # negative a if started below ~0.1).
    grid = [
        (0.1, 0.1), (0.1, 0.5), (0.1, 1.0),
        (0.3, 0.1), (0.3, 0.5), (0.3, 1.0),
        (0.5, 0.1), (0.5, 0.9), (0.5, 5.0),
        (1.0, 0.1), (1.0, 1.0), (1.0, 5.0),
        (2.3, 0.1), (2.3, 1.0), (2.3, 5.0),
        (10.0, 0.5), (10.0, 9.5), (10.0, 25.0),
        (50.0, 5.0), (50.0, 49.5)
    ]
    for (a, z) in grid
        truth = derivative(a -> first(gamma_inc(a, z)), fd, a)
        series = _grad_p_a_series(a, z)
        @test isapprox(series, truth; atol = 1e-10, rtol = 1e-10)
    end
end

@testitem "_gamma_cdf passes Mooncake.TestUtils.test_rule" tags=[
    :ad, :mooncake, :mooncake_reverse] begin
    # Mooncake's canonical rule test, run for both reverse and forward
    # mode. `@from_chainrules` (default mode) lifts the `rrule` into an
    # `rrule!!` and the `frule` into an `frule!!`, so both interfaces are
    # registered (#270). For each mode, verifies (a) the rule is actually
    # invoked (is_primitive = true asserts this) and (b) primal +
    # derivative match Richardson-extrapolated finite differences.
    using Random: MersenneTwister
    using Mooncake: Mooncake
    using CensoredDistributions: _gamma_cdf

    cases = [
        (2.3, 1.7, 1.9),
        (0.5, 2.0, 0.3),
        (5.0, 0.4, 1.0),
        (10.0, 1.0, 9.5),
        (0.3, 1.0, 0.5)
    ]
    for mode in (Mooncake.ReverseMode, Mooncake.ForwardMode),
        (k, θ, x) in cases

        Mooncake.TestUtils.test_rule(
            MersenneTwister(20260526),
            _gamma_cdf, k, θ, x;
            is_primitive = true,
            perf_flag = :none,
            mode = mode
        )
    end
end

@testitem "Enzyme direct rule on _gamma_cdf (issue #259)" tags=[
    :ad, :enzyme, :enzyme_reverse] begin
    # Pins the fix in CensoredDistributionsEnzymeExt: the original
    # `Enzyme.@import_rrule` lift returned the wrong ∂P/∂k (~8% off). The
    # direct EnzymeRules.augmented_primal/reverse + forward rule should now
    # match the ForwardDiff reference on both modes.
    using ADTypes: AutoEnzyme, AutoForwardDiff
    using DifferentiationInterface: gradient
    using Enzyme: Enzyme
    using ForwardDiff: ForwardDiff
    using CensoredDistributions: _gamma_cdf

    f(v) = _gamma_cdf(v[1], v[2], v[3])
    cases = [
        [2.3, 1.7, 1.9],
        [0.5, 2.0, 0.3],
        [5.0, 0.4, 1.0],
        [10.0, 1.0, 9.5]
    ]
    for input in cases
        ref = gradient(f, AutoForwardDiff(), input)
        g_rev = gradient(f, AutoEnzyme(mode = Enzyme.Reverse), input)
        g_fwd = gradient(f, AutoEnzyme(mode = Enzyme.Forward), input)
        @test isapprox(g_rev, ref; rtol = 1e-10, atol = 1e-12)
        @test isapprox(g_fwd, ref; rtol = 1e-10, atol = 1e-12)
    end
end

@testitem "Enzyme gamma rule (issue #263)" tags=[:ad, :enzyme, :enzyme_reverse] begin
    # Pins the `SpecialFunctions.gamma` rule in
    # CensoredDistributionsEnzymeExt. With only EnzymeSpecialFunctionsExt
    # loaded, Enzyme mis-lowers `gamma` to the `loggamma` known-op and
    # returns `ψ(x)` instead of `Γ(x) ψ(x)` — silently wrong by a factor
    # of `Γ(x)` in both modes. The analytical Gamma/Weibull
    # `primarycensored_cdf` paths call `gamma(k + 1)` / `gamma(1 + 1/k)`
    # outside the `_gamma_cdf` rule, so this gap corrupted the shape
    # partial of the whole pipeline.
    using ADTypes: AutoEnzyme, AutoForwardDiff
    using DifferentiationInterface: gradient
    using Enzyme: Enzyme
    using SpecialFunctions: gamma, digamma

    f(v) = gamma(v[1])
    for x in (0.7, 1.5, 2.0, 3.4, 7.0)
        truth = gamma(x) * digamma(x)
        g_rev = gradient(f, AutoEnzyme(mode = Enzyme.Reverse), [x])
        g_fwd = gradient(f, AutoEnzyme(mode = Enzyme.Forward), [x])
        @test isapprox(g_rev[1], truth; rtol = 1e-10, atol = 1e-12)
        @test isapprox(g_fwd[1], truth; rtol = 1e-10, atol = 1e-12)
    end
end

@testitem "_gamma_cdf rrule and _make_weibull_g zero-input guards" tags=[:ad, :forwarddiff] begin
    # Exercise the non-positive-input early-return branches that the
    # scenario suite never hits (all gradient grids use strictly positive
    # x / t). Without this, the rrule's x <= 0 path and the Weibull g's
    # t <= 0 guard appear as uncovered defensive code in patch coverage.
    using ChainRulesCore: rrule, NoTangent
    using CensoredDistributions: CensoredDistributions, _gamma_cdf
    _make_weibull_g = CensoredDistributions._make_weibull_g

    Ω, pb = rrule(_gamma_cdf, 2.0, 1.5, 0.0)
    @test Ω == 0.0
    @test pb(1.0) == (NoTangent(), 0.0, 0.0, 0.0)

    Ω_neg, pb_neg = rrule(_gamma_cdf, 2.0, 1.5, -0.5)
    @test Ω_neg == 0.0
    @test pb_neg(1.0) == (NoTangent(), 0.0, 0.0, 0.0)

    g = _make_weibull_g(2.0, 1.5)
    @test g(0.0) === 0.0
    @test g(-1.0) === 0.0
end
