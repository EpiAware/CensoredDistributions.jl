# Unit-level AD coverage for the gamma CDF rrule. Complements the
# scenario suite in `runtests.jl`: those check end-to-end gradient
# agreement; these pin the implementation-level guarantees (series
# accuracy, Mooncake rule structure, defensive guards) that no
# scenario exercises directly.

@testset "_grad_p_a_series matches FiniteDifferences" begin
    using SpecialFunctions: gamma_inc
    using FiniteDifferences: central_fdm
    using DifferentiationInterface: AutoFiniteDifferences, derivative
    using CensoredDistributions: _grad_p_a_series

    fd = AutoFiniteDifferences(; fdm = central_fdm(7, 1))

    # Restrict grid to (a, z) values where ∂P/∂a is large enough that the
    # finite-difference baseline is meaningful (i.e. away from
    # P-saturation). Includes k≪1 cases — the singular regime that breaks
    # naive implementations of the shape-parameter derivative. Lower bound
    # on a is set by the fdm probe radius (`central_fdm(7, 1)` reaches
    # into negative a if started below ~0.1).
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

@testset "_gamma_cdf passes Mooncake.TestUtils.test_rule" begin
    # Mooncake's canonical rule test. `mode = Mooncake.ReverseMode` skips
    # the forward-mode interface check — we only register an rrule via
    # @from_chainrules, no frule. Verifies (a) the rule is actually being
    # invoked (is_primitive = true asserts this) and (b) primal +
    # pullback match Richardson-extrapolated finite differences.
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
    for (k, θ, x) in cases
        Mooncake.TestUtils.test_rule(
            MersenneTwister(20260526),
            _gamma_cdf, k, θ, x;
            is_primitive = true,
            perf_flag = :none,
            mode = Mooncake.ReverseMode
        )
    end
end

@testset "_gamma_cdf rrule and _make_weibull_g zero-input guards" begin
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
