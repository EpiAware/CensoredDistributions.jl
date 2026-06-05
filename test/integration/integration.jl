# Tests for the pluggable integration interface (issue #208): the core
# default `GaussLegendre` solver, its numerical equivalence to the prior
# Integrals.jl GaussLegendre path, and the optional Integrals.jl extension.

@testitem "Default GaussLegendre integrate is accurate" begin
    using CensoredDistributions
    using CensoredDistributions: integrate, GaussLegendre, gl_integrate

    solver = GaussLegendre(; n = 64)
    # ∫ x^2 dx over [0, 1] = 1/3
    @test integrate(solver, x -> x^2, 0.0, 1.0) ≈ 1 / 3 atol=1e-13
    # ∫ exp(x) dx over [0, 2] = e^2 - 1
    @test integrate(solver, x -> exp(x), 0.0, 2.0) ≈ exp(2) - 1 atol=1e-12
    # gl_integrate matches integrate through the default solver.
    @test gl_integrate(x -> x^2, 0.0, 1.0,
        CensoredDistributions._gl_rule(64)) ≈ integrate(solver, x -> x^2, 0.0, 1.0)
    # Degenerate window returns a typed zero.
    @test integrate(solver, x -> x^2, 1.0, 1.0) == 0.0
end

@testitem "Default path reproduces Integrals.jl GaussLegendre (n=64)" begin
    using CensoredDistributions
    using Distributions
    using Integrals: Integrals, IntegralProblem, solve

    # Reproduce the prior core path: the numeric primary-censored CDF used
    # `Integrals.GaussLegendre(; n = 64)` via `IntegralProblem`/`solve`.
    function reference_pc_cdf(dist, primary, x; n = 64)
        integrand(u, x) = exp(logcdf(dist, u) + logpdf(primary, x - u))
        lower = max(x - maximum(primary), minimum(dist))
        upper = x - minimum(primary)
        upper <= lower && return 0.0
        prob = IntegralProblem(integrand, (lower, upper), x)
        clamp(solve(prob, Integrals.GaussLegendre(; n = n))[1], 0.0, 1.0)
    end

    # A non-analytic pair (truncated Normal primary) forces the numeric
    # quadrature path, so the default `gl_integrate` is exercised.
    dist = Gamma(2.0, 1.5)
    primary = truncated(Normal(0.5, 0.3), 0.0, 1.0)
    d = primary_censored(dist, primary)
    for x in (1.0, 2.0, 3.0, 5.0)
        @test cdf(d, x) ≈ reference_pc_cdf(dist, primary, x) atol=1e-13
    end

    # Convolution numeric path: compare the package's `convolve_distributions`
    # CDF against an independent Integrals.jl GaussLegendre solve of the
    # same convolution integral.
    function reference_conv_cdf(c1, c2, x; n = 192)
        lower = max(minimum(c2), x - maximum(c1))
        upper = min(maximum(c2), x - minimum(c1))
        upper <= lower && return cdf(c2, x - maximum(c1)) > 0 ? 1.0 : 0.0
        cut = x - maximum(c1)
        saturated = cut > minimum(c2) ? cdf(c2, cut) : 0.0
        integrand(t, x) = cdf(c1, x - t) * pdf(c2, t)
        prob = IntegralProblem(integrand, (lower, upper), x)
        val = saturated + solve(prob, Integrals.GaussLegendre(; n = n))[1]
        clamp(val, 0.0, 1.0)
    end

    c1 = Gamma(2.0, 1.0)
    c2 = LogNormal(0.5, 0.4)
    dc = convolve_distributions(c1, c2)
    for x in (1.0, 2.0, 3.0, 5.0)
        @test cdf(dc, x) ≈ reference_conv_cdf(c1, c2, x) atol=1e-13
    end
end

@testitem "Integrals.jl extension routes QuadGKJL/HCubatureJL" begin
    using CensoredDistributions
    using CensoredDistributions: integrate, GaussLegendre
    using Integrals: QuadGKJL, HCubatureJL

    default = GaussLegendre(; n = 64)
    for f in (x -> x^2, x -> exp(x), x -> sin(x) + 1)
        ref = integrate(default, f, 0.0, 1.5)
        @test integrate(QuadGKJL(), f, 0.0, 1.5) ≈ ref atol=1e-10
        @test integrate(HCubatureJL(), f, 0.0, 1.5) ≈ ref atol=1e-10
    end
end

@testitem "primary_censored agrees across solvers via extension" begin
    using CensoredDistributions
    using Distributions
    using Integrals: QuadGKJL

    # Smooth, non-analytic pair: the default GaussLegendre solver and an
    # Integrals.jl QuadGKJL solver (passed via the extension) agree.
    dist = Gamma(2.0, 1.5)
    primary = truncated(Normal(0.5, 0.3), 0.0, 1.0)
    d_default = primary_censored(dist, primary)
    d_quadgk = primary_censored(dist, primary; solver = QuadGKJL())
    for x in (1.0, 2.0, 3.0, 5.0)
        @test cdf(d_default, x) ≈ cdf(d_quadgk, x) atol=1e-8
    end
end
