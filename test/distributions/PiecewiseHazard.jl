@testitem "PiecewiseHazard constructor validation" begin
    using Distributions

    d = piecewise_hazard([1.0, 3.0], [0.2, 0.8, 0.3])
    @test d isa PiecewiseHazard
    @test d.breaks == [1.0, 3.0]
    @test d.hazards == [0.2, 0.8, 0.3]
    @test eltype(d) == Float64

    # Mixed integer / float arguments promote to Float64.
    d2 = piecewise_hazard([1, 3], [1, 2, 1])
    @test eltype(d2) == Float64

    # A single hazard with no breakpoints is the constant-hazard case.
    d3 = piecewise_hazard(Float64[], [0.5])
    @test d3.hazards == [0.5]
    @test isempty(d3.breaks)

    # Error cases.
    # hazards must have one more element than breaks
    @test_throws ArgumentError piecewise_hazard([1.0, 3.0], [0.2, 0.8])
    # non-positive hazard
    @test_throws ArgumentError piecewise_hazard([1.0], [0.2, -0.1])
    @test_throws ArgumentError piecewise_hazard([1.0], [0.0, 0.5])
    # non-finite hazard
    @test_throws ArgumentError piecewise_hazard([1.0], [Inf, 0.5])
    # non-positive breakpoint
    @test_throws ArgumentError piecewise_hazard([0.0, 1.0], [0.2, 0.5, 0.3])
    # non-increasing breakpoints
    @test_throws ArgumentError piecewise_hazard([3.0, 1.0], [0.2, 0.5, 0.3])
    @test_throws ArgumentError piecewise_hazard([1.0, 1.0], [0.2, 0.5, 0.3])
end

@testitem "PiecewiseHazard basic interface methods" begin
    using Distributions

    d = piecewise_hazard([1.0, 3.0], [0.2, 0.8, 0.3])

    # params returns the grid and the hazard values.
    p = params(d)
    @test p == ([1.0, 3.0], [0.2, 0.8, 0.3])

    # Support is [0, ∞).
    @test minimum(d) == 0.0
    @test maximum(d) == Inf
    @test insupport(d, 0.0)
    @test insupport(d, 5.0)
    @test !insupport(d, -0.1)
    @test !insupport(d, Inf)
end

@testitem "PiecewiseHazard constant hazard equals Exponential" begin
    using Distributions

    # A single-piece hazard h is an Exponential with rate h, i.e. scale 1/h.
    for h in (0.25, 0.5, 1.0, 2.0)
        d = piecewise_hazard(Float64[], [h])
        e = Exponential(1 / h)
        for x in (0.0, 0.5, 1.0, 2.5, 5.0, 10.0)
            @test logpdf(d, x) ≈ logpdf(e, x)
            @test pdf(d, x) ≈ pdf(e, x)
            @test cdf(d, x) ≈ cdf(e, x)
            @test ccdf(d, x) ≈ ccdf(e, x)
            if x > 0
                @test logcdf(d, x) ≈ logcdf(e, x)
            end
            @test logccdf(d, x) ≈ logccdf(e, x)
        end
        for p in (0.1, 0.5, 0.9)
            @test quantile(d, p) ≈ quantile(e, p)
        end
        @test mean(d) ≈ mean(e)
        @test var(d) ≈ var(e)
        @test std(d) ≈ std(e)
        @test median(d) ≈ median(e)
    end
end

@testitem "PiecewiseHazard step hazard matches hand-computed S and f" begin
    using Distributions

    # h(t) = 0.2 on [0,1), 0.8 on [1,3), 0.3 on [3,∞).
    d = piecewise_hazard([1.0, 3.0], [0.2, 0.8, 0.3])

    # Hand-computed cumulative hazards:
    # H(0.5) = 0.2 * 0.5 = 0.1
    # H(2.0) = 0.2 * 1 + 0.8 * 1 = 1.0
    # H(4.0) = 0.2 * 1 + 0.8 * 2 + 0.3 * 1 = 2.1
    @test ccdf(d, 0.5) ≈ exp(-0.1)
    @test ccdf(d, 2.0) ≈ exp(-1.0)
    @test ccdf(d, 4.0) ≈ exp(-2.1)

    # Densities f(t) = h(t) S(t).
    @test pdf(d, 0.5) ≈ 0.2 * exp(-0.1)
    @test pdf(d, 2.0) ≈ 0.8 * exp(-1.0)
    @test pdf(d, 4.0) ≈ 0.3 * exp(-2.1)

    # CDF is the complement of the survival.
    @test cdf(d, 2.0) ≈ 1 - exp(-1.0)
    @test cdf(d, 2.0) + ccdf(d, 2.0) ≈ 1.0

    # Density vanishes below the support, survival is one at zero.
    @test pdf(d, -1.0) == 0.0
    @test logpdf(d, -1.0) == -Inf
    @test ccdf(d, 0.0) == 1.0
    @test cdf(d, 0.0) == 0.0
end

@testitem "PiecewiseHazard pdf integrates to 1" begin
    using Distributions
    using Integrals

    function integrate_pdf(d, upper)
        prob = IntegralProblem((x, p) -> pdf(d, x), (0.0, upper))
        sol = solve(prob, QuadGKJL(); reltol = 1e-9)
        return sol.u
    end

    test_cases = [
        piecewise_hazard(Float64[], [0.5]),
        piecewise_hazard([1.0, 3.0], [0.2, 0.8, 0.3]),
        piecewise_hazard([2.0], [1.0, 0.4]),
        piecewise_hazard([0.5, 1.0, 2.0, 4.0], [1.5, 0.3, 0.9, 0.2, 0.6])
    ]
    for d in test_cases
        # The tail beyond a high quantile contributes negligibly.
        upper = quantile(d, 1 - 1e-10)
        @test abs(integrate_pdf(d, upper) - 1.0) < 1e-7
    end
end

@testitem "PiecewiseHazard cdf is the integral of the pdf" begin
    using Distributions
    using Integrals

    function numerical_cdf(d, x)
        x <= 0 && return 0.0
        prob = IntegralProblem((t, p) -> pdf(d, t), (0.0, x))
        sol = solve(prob, QuadGKJL(); reltol = 1e-10)
        return sol.u
    end

    d = piecewise_hazard([1.0, 3.0], [0.2, 0.8, 0.3])
    for x in [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
        @test abs(numerical_cdf(d, x) - cdf(d, x)) < 1e-8
    end

    # The density is the derivative of the cdf.
    eps = 1e-6
    for x in [0.5, 1.5, 2.5, 4.0]
        fd = (cdf(d, x + eps) - cdf(d, x - eps)) / (2 * eps)
        @test abs(fd - pdf(d, x)) < 1e-4
    end
end

@testitem "PiecewiseHazard quantile inverts the cdf" begin
    using Distributions

    d = piecewise_hazard([1.0, 3.0], [0.2, 0.8, 0.3])

    @test quantile(d, 0.0) == 0.0
    @test quantile(d, 1.0) == Inf

    for p in [0.05, 0.25, 0.5, 0.75, 0.95]
        q = quantile(d, p)
        @test cdf(d, q) ≈ p
    end

    @test_throws ArgumentError quantile(d, -0.1)
    @test_throws ArgumentError quantile(d, 1.1)

    @test median(d) ≈ quantile(d, 0.5)
    @test cdf(d, median(d)) ≈ 0.5
end

@testitem "PiecewiseHazard analytic mean and variance match numerics" begin
    using Distributions
    using Integrals

    # E[T] = ∫ S, E[T²] = ∫ 2 t S, checked against quadrature.
    test_cases = [
        piecewise_hazard([1.0, 3.0], [0.2, 0.8, 0.3]),
        piecewise_hazard([2.0], [1.0, 0.4]),
        piecewise_hazard([0.5, 1.0, 2.0], [1.5, 0.3, 0.9, 0.6])
    ]
    for d in test_cases
        upper = quantile(d, 1 - 1e-12)
        mprob = IntegralProblem((t, p) -> ccdf(d, t), (0.0, upper))
        m_num = solve(mprob, QuadGKJL(); reltol = 1e-10).u
        s2prob = IntegralProblem((t, p) -> 2 * t * ccdf(d, t), (0.0, upper))
        s2_num = solve(s2prob, QuadGKJL(); reltol = 1e-10).u
        @test abs(mean(d) - m_num) < 1e-6
        @test abs(var(d) - (s2_num - m_num^2)) < 1e-5
        @test std(d) ≈ sqrt(var(d))
    end
end

@testitem "PiecewiseHazard random sampling matches the analytic law" begin
    using Distributions
    using Random
    using Statistics
    using HypothesisTests

    Random.seed!(42)
    test_cases = [
        piecewise_hazard(Float64[], [0.5]),
        piecewise_hazard([1.0, 3.0], [0.2, 0.8, 0.3]),
        piecewise_hazard([2.0], [1.0, 0.4])
    ]
    n = 20000
    for d in test_cases
        s = [rand(d) for _ in 1:n]
        @test all(>=(0.0), s)

        # The empirical CDF matches the analytic one.
        ks = ExactOneSampleKSTest(s, d)
        @test pvalue(ks) > 0.001

        @test abs(mean(s) - mean(d)) < 0.05 * mean(d)
        @test abs(std(s) - std(d)) < 0.05 * std(d)

        for p in [0.25, 0.5, 0.75, 0.9]
            @test abs(quantile(s, p) - quantile(d, p)) < 0.05 * mean(d)
        end
    end
end

@testitem "PiecewiseHazard composes as a leaf" begin
    using Distributions

    hz = piecewise_hazard([2.0, 5.0], [0.3, 0.6, 0.2])

    # Censoring wrappers.
    @test isfinite(logpdf(interval_censored(hz, 1.0), 2.0))
    @test isfinite(logpdf(primary_censored(hz, Uniform(0, 1)), 2.5))
    dic = double_interval_censored(hz; primary_event = Uniform(0, 1),
        upper = 20.0, interval = 1.0)
    @test isfinite(logpdf(dic, 3.0))

    # Truncation. `truncate_to_window(d, upper, δ)` keeps `[upper - δ, upper]`,
    # so 3.0 is in support for the window `[2, 6]`.
    @test isfinite(logpdf(truncate_to_horizon(hz, 8.0), 3.0))
    @test isfinite(logpdf(truncate_to_window(hz, 6.0, 4.0), 3.0))

    # A compose chain with the hazard leaf as a step.
    seq = compose((onset_admit = hz, admit_death = LogNormal(0.5, 0.4)))
    @test rand(seq) isa NamedTuple

    # The racing-hazard `compete` node multiplies branch survivals, so a
    # flexible cause-specific hazard drops straight in. Each cause carries a
    # positive winning probability summing to roughly one (the node integrates
    # the branch sub-densities numerically).
    race = compete(:death => hz, :recover => Gamma(2.0, 1.5))
    @test isfinite(logpdf(race, 3.0))
    wp = winning_probabilities(race)
    @test all(p -> p > 0, values(wp))
    @test isapprox(sum(values(wp)), 1.0; atol = 0.02)
end

@testitem "PiecewiseHazard interface conformance" begin
    using CensoredDistributions
    using CensoredDistributions.TestUtils: test_interface

    d = piecewise_hazard([1.0, 3.0], [0.2, 0.8, 0.3])
    test_interface(d; name = "PiecewiseHazard", draw = 2.0,
        univariate = true, overall = :scalar)
end

@testitem "PiecewiseHazard ForwardDiff gradient over hazards" begin
    using Distributions
    using ForwardDiff

    obs = [0.5, 1.2, 2.5, 3.8, 5.1]
    f(h) = sum(x -> logpdf(piecewise_hazard([1.0, 3.0], h), x), obs)
    g = ForwardDiff.gradient(f, [0.2, 0.8, 0.3])
    @test length(g) == 3
    @test all(isfinite, g)

    # Compare against a central finite difference.
    h0 = [0.2, 0.8, 0.3]
    eps = 1e-6
    for i in 1:3
        hp = copy(h0);
        hp[i] += eps
        hm = copy(h0);
        hm[i] -= eps
        fd = (f(hp) - f(hm)) / (2 * eps)
        @test abs(g[i] - fd) < 1e-4
    end
end
