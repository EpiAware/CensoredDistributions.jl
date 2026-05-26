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

@testitem "_gamma_cdf gradient matches FiniteDifferences across AD backends" tags=[:ad] begin
    # Direct function-level coverage of every AD backend we wire up:
    # the ChainRules rrule (used by ReverseDiff and Mooncake via their
    # respective extensions) and the explicit Dual methods (used by
    # ForwardDiff via the ForwardDiff extension). Each backend gets the
    # same FiniteDifferences ground truth so a regression in any single
    # registration surfaces here rather than only in the pipeline tests.
    using FiniteDifferences: central_fdm
    using DifferentiationInterface
    import ForwardDiff
    import ReverseDiff
    import Mooncake
    using CensoredDistributions: _gamma_cdf

    fd = AutoFiniteDifferences(; fdm = central_fdm(7, 1))
    backends = [
        ("ForwardDiff", AutoForwardDiff()),
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

@testitem "_gamma_cdf ForwardDiff Dual dispatch covers every Dual/Real subset" tags=[:ad] begin
    # The ForwardDiff extension enumerates seven explicit Dual methods,
    # one per non-trivial subset of (k, θ, x) being a Dual rather than
    # a plain Real. The other AD tests only exercise the all-Dual
    # combination via `gradient(f, AutoForwardDiff(), [k, θ, x])`,
    # which leaves the six mixed-arity methods uncovered. Calling each
    # combination directly and comparing against the all-Float64 primal
    # both validates dispatch and lifts the patch coverage off those
    # otherwise-unhit lines.
    import ForwardDiff
    using ForwardDiff: Dual, value, partials
    using CensoredDistributions: _gamma_cdf

    k0, θ0, x0 = 2.3, 1.7, 1.9
    primal = _gamma_cdf(k0, θ0, x0)
    # 1-partial Dual along each axis lets us read off the per-arg partial
    # without going through `gradient`. Tag is opaque; tag value `nothing`
    # is the convention for tag-less testing.
    seed(v) = Dual{Nothing}(v, 1.0)

    dk = seed(k0)
    dθ = seed(θ0)
    dx = seed(x0)

    # All seven non-trivial Dual/Real subsets of (k, θ, x).
    combos = [
        (dk, dθ, dx),         # (Dual, Dual, Dual)
        (dk, dθ, x0),         # (Dual, Dual, Real)
        (dk, θ0, dx),         # (Dual, Real, Dual)
        (k0, dθ, dx),         # (Real, Dual, Dual)
        (dk, θ0, x0),         # (Dual, Real, Real)
        (k0, dθ, x0),         # (Real, Dual, Real)
        (k0, θ0, dx)          # (Real, Real, Dual)
    ]
    for (k, θ, x) in combos
        out = _gamma_cdf(k, θ, x)
        @test out isa Dual
        @test value(out)≈primal atol=1e-12 rtol=1e-12
        @test length(partials(out)) == 1
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
    # directly) so the ChainRules rrule + Mooncake @from_chainrules
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
