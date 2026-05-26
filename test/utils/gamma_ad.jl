@testitem "_gamma_p_series matches SpecialFunctions.gamma_inc" tags=[:ad] begin
    using SpecialFunctions: gamma_inc
    using CensoredDistributions: _gamma_p_series

    # Grid spanning small/large shape and the z<a / z≈a / z>a regimes
    # where the series converges at different rates. The k≪1 rows pin
    # the singular regime where t^(k-1) blows up at t=0 — the case that
    # historically broke implementations of the gamma CDF derivative.
    # All (a, z) here satisfy z < a + 1 — the regime where the absolutely-
    # convergent series converges. For z >= a + 1 the series catastrophically
    # underflows once z/a is large enough that z^a · e^{-z} / Γ(a+1) falls
    # below floatmin(Float64); _gamma_cdf dispatches to _gamma_q_cf there
    # and that path is exercised by the next testitem.
    grid = [
        (0.05, 0.001), (0.05, 0.1), (0.05, 1.0),
        (0.1, 0.01), (0.1, 0.5), (0.1, 1.0),
        (0.3, 0.01), (0.3, 0.5), (0.3, 1.0),
        (0.5, 0.1), (0.5, 0.9), (0.5, 1.4),
        (1.0, 0.1), (1.0, 1.0), (1.0, 1.9),
        (2.3, 0.1), (2.3, 1.0), (2.3, 3.0),
        (10.0, 0.5), (10.0, 5.0), (10.0, 10.5),
        (50.0, 5.0), (50.0, 30.0), (50.0, 50.5)
    ]
    for (a, z) in grid
        truth = first(gamma_inc(a, z))
        series = _gamma_p_series(a, z)
        @test isapprox(series, truth; atol = 1e-13, rtol = 1e-13)
    end
end

@testitem "_grad_p_a_series matches FiniteDifferences" tags=[:ad] begin
    using SpecialFunctions: gamma_inc
    using FiniteDifferences: central_fdm
    using DifferentiationInterface: AutoFiniteDifferences, derivative
    using CensoredDistributions: _grad_p_a_series

    fd = AutoFiniteDifferences(; fdm = central_fdm(7, 1))

    # Restrict grid to (a, z) values where ∂P/∂a is large enough that the
    # finite-difference baseline is meaningful (i.e. away from P-saturation).
    # Includes k≪1 cases — the singular regime that breaks naive
    # implementations of the shape-parameter derivative. Lower bound on a
    # is set by the fdm probe radius (`central_fdm(7, 1)` reaches into
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

@testitem "_gamma_cdf rrule matches FiniteDifferences (ReverseDiff & Mooncake)" tags=[:ad] begin
    using FiniteDifferences: central_fdm
    using DifferentiationInterface
    import ReverseDiff
    import Mooncake
    using CensoredDistributions: _gamma_cdf

    fd = AutoFiniteDifferences(; fdm = central_fdm(7, 1))
    backends = [
        ("ReverseDiff", AutoReverseDiff()),
        ("Mooncake", AutoMooncake(; config = nothing))
    ]

    # Wrap _gamma_cdf as a single-vector function so DI can request the
    # full gradient through one call.
    f(v) = _gamma_cdf(v[1], v[2], v[3])

    cases = [
        [2.3, 1.7, 1.9],
        [0.5, 2.0, 0.3],
        [5.0, 0.4, 1.0],
        [10.0, 1.0, 9.5],
        # Small shape values: t^(k-1) singular at t=0
        [0.1, 1.0, 0.5],
        [0.3, 1.0, 0.5],
        [0.6, 1.0, 1.0]
    ]
    for v in cases
        truth = gradient(f, fd, v)
        for (name, backend) in backends
            g = gradient(f, backend, v)
            @test isapprox(g, truth; atol = 1e-8, rtol = 1e-8)
        end
    end
end

@testitem "_gamma_cdf passes Mooncake.TestUtils.test_rule" tags=[:ad] begin
    # Mooncake's canonical rule test. `mode = Mooncake.ReverseMode`
    # skips the forward-mode interface check — we only register an
    # rrule via @from_chainrules, no frule. Verifies (a) the rule is
    # actually being invoked (is_primitive = true asserts this) and
    # (b) primal + pullback match Richardson-extrapolated finite
    # differences. Stronger structural guarantee than the DI-based
    # test below, which only checks end-to-end numerical agreement.
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

@testitem "primarycensored Gamma+Uniform numeric path differentiates across backends" tags=[:ad] begin
    # `_logcdf_ad_safe(::Gamma, ...)` routes the inner CDF through our
    # rrule so the integrand is differentiable; the outer integrator
    # then has to itself survive AD. Default GaussLegendre (fixed-node)
    # survives ForwardDiff/ReverseDiff/Mooncake. QuadGKJL retains
    # adaptive behaviour but only survives forward/reverse Dual paths.
    using FiniteDifferences: central_fdm
    using DifferentiationInterface
    import ForwardDiff
    import ReverseDiff
    import Mooncake
    using Distributions: Gamma, Uniform
    using Integrals: QuadGKJL
    using CensoredDistributions: primarycensored_cdf, NumericSolver

    fd = AutoFiniteDifferences(; fdm = central_fdm(7, 1))

    # method => backends that should pass on it
    method_backends = [
        (NumericSolver(),
            [
                ("ForwardDiff", AutoForwardDiff()),
                ("ReverseDiff", AutoReverseDiff()),
                ("Mooncake", AutoMooncake(; config = nothing))
            ]),
        (NumericSolver(QuadGKJL()),
            [
                ("ForwardDiff", AutoForwardDiff()),
                ("ReverseDiff", AutoReverseDiff())
            ])
    ]

    pe = Uniform(0.0, 2.0)
    cases = [
        (Float64[2.3, 1.7], 3.9),
        (Float64[1.5, 2.0], 4.0),
        (Float64[5.0, 0.8], 4.5),
        # Small shape values
        (Float64[0.3, 1.0], 2.5),
        (Float64[0.6, 1.5], 3.0)
    ]
    for (method, backends) in method_backends, (v, x) in cases

        f = let pe = pe, method = method, x = x
            v -> primarycensored_cdf(Gamma(v[1], v[2]), pe, x, method)
        end
        truth = gradient(f, fd, v)
        for (_, backend) in backends
            g = gradient(f, backend, v)
            @test isapprox(g, truth; atol = 1e-6, rtol = 1e-6)
        end
    end
end

@testitem "_gamma_cdf rrule and _make_weibull_g zero-input guards" tags=[:ad] begin
    # Exercise the non-positive-input early-return branches that no other
    # AD test hits (all the gradient-checking grids use strictly positive
    # x / t). Without this, the rrule's x <= 0 path and the Weibull g's
    # t <= 0 guard appear as uncovered defensive code in patch coverage.
    using ChainRulesCore: rrule, NoTangent
    using CensoredDistributions: CensoredDistributions, _gamma_cdf
    # _make_weibull_g lives in the primarycensored CDF module; reach it
    # via the parent module to avoid leaking it into the public surface.
    _make_weibull_g = CensoredDistributions._make_weibull_g

    # rrule(_gamma_cdf, k, θ, x) with x == 0: primal is zero, pullback
    # returns NoTangent + zero tangents for every input.
    Ω, pb = rrule(_gamma_cdf, 2.0, 1.5, 0.0)
    @test Ω == 0.0
    @test pb(1.0) == (NoTangent(), 0.0, 0.0, 0.0)

    Ω_neg, pb_neg = rrule(_gamma_cdf, 2.0, 1.5, -0.5)
    @test Ω_neg == 0.0
    @test pb_neg(1.0) == (NoTangent(), 0.0, 0.0, 0.0)

    # _make_weibull_g(k, λ)(t) for t <= 0 returns zero(t).
    g = _make_weibull_g(2.0, 1.5)
    @test g(0.0) === 0.0
    @test g(-1.0) === 0.0
end

@testitem "primarycensored Weibull+Uniform analytic gradient matches FiniteDifferences" tags=[:ad] begin
    # _make_weibull_g must route through _gamma_cdf (not _gamma_p_series
    # directly) so the ChainRules rrule + Mooncake @from_rrule
    # intercept. Without that, reverse-mode AD traces the series loop —
    # slower, and the regression isn't caught by primal-only tests.
    using FiniteDifferences: central_fdm
    using DifferentiationInterface
    import ReverseDiff
    import Mooncake
    using Distributions: Weibull, Uniform
    using CensoredDistributions: primarycensored_cdf, AnalyticalSolver

    fd = AutoFiniteDifferences(; fdm = central_fdm(7, 1))
    backends = [
        ("ReverseDiff", AutoReverseDiff()),
        ("Mooncake", AutoMooncake(; config = nothing))
    ]

    pe = Uniform(0.0, 2.0)
    method = AnalyticalSolver(nothing)
    f(v) = primarycensored_cdf(Weibull(v[1], v[2]), pe, v[3], method)

    cases = [
        [1.5, 1.0, 2.0],
        [2.0, 1.5, 3.0],
        [3.0, 0.8, 1.5]
    ]
    # FiniteDifferences baseline is noisier on the x-partial here because
    # the integrand contains (t/λ)^k, whose finite-difference probe
    # accumulates rounding error through the real-exponent power; the
    # backends agree with each other to ~1e-10.
    for v in cases
        truth = gradient(f, fd, v)
        for (_, backend) in backends
            g = gradient(f, backend, v)
            @test isapprox(g, truth; atol = 1e-6, rtol = 1e-6)
        end
    end
end

@testitem "primarycensored Gamma+Uniform analytic gradient matches FiniteDifferences" tags=[:ad] begin
    using FiniteDifferences: central_fdm
    using DifferentiationInterface
    import ReverseDiff
    import Mooncake
    using Distributions: Gamma, Uniform
    using Integrals: QuadGKJL
    using CensoredDistributions: primarycensored_cdf, AnalyticalSolver

    fd = AutoFiniteDifferences(; fdm = central_fdm(7, 1))
    backends = [
        ("ReverseDiff", AutoReverseDiff()),
        ("Mooncake", AutoMooncake(; config = nothing))
    ]

    pe = Uniform(0.0, 2.0)
    method = AnalyticalSolver(QuadGKJL())
    f(v) = primarycensored_cdf(Gamma(v[1], v[2]), pe, v[3], method)

    cases = [
        [2.3, 1.7, 3.9],
        [1.5, 2.0, 4.0],
        [5.0, 0.8, 4.5],
        # Small shape values
        [0.3, 1.0, 2.5],
        [0.6, 1.5, 3.0]
    ]
    for v in cases
        truth = gradient(f, fd, v)
        for (name, backend) in backends
            g = gradient(f, backend, v)
            @test isapprox(g, truth; atol = 1e-7, rtol = 1e-7)
        end
    end
end
