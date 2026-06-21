# Direct unit tests for the shared `_quantile_optimization` root finder
# (`src/utils/quantile_optimization.jl`). The public `quantile` methods on
# `PrimaryCensored` / `IntervalCensored` / `Convolved` route through it, but its
# accuracy / convergence / boundary / error contract was only exercised
# indirectly. These tests pin it on a known closed-form case.

@testitem "_quantile_optimization: known closed-form Normal quantile" begin
    using CensoredDistributions: _quantile_optimization
    using Distributions

    # Normal has an exact `quantile`, so the Nelder-Mead root of
    # `cdf(d, q) - p = 0` must match it to the solver tolerance.
    d = Normal(2.0, 1.5)
    for p in (0.05, 0.25, 0.5, 0.75, 0.95)
        q = _quantile_optimization(d, p)
        @test isapprox(q, quantile(d, p); atol = 1e-4)
        # Convergence is defined by the recovered CDF hitting `p`.
        @test isapprox(cdf(d, q), p; atol = 1e-5)
    end
end

@testitem "_quantile_optimization: boundary probabilities" begin
    using CensoredDistributions: _quantile_optimization
    using Distributions

    # p = 0 / p = 1 are handled analytically as the support endpoints, NOT
    # via the optimiser.
    d = Gamma(2.0, 1.5)
    @test _quantile_optimization(d, 0.0) == minimum(d)
    @test _quantile_optimization(d, 1.0) == maximum(d)
end

@testitem "_quantile_optimization: invalid probabilities throw" begin
    using CensoredDistributions: _quantile_optimization
    using Distributions

    d = Normal(0.0, 1.0)
    @test_throws ArgumentError _quantile_optimization(d, -0.1)
    @test_throws ArgumentError _quantile_optimization(d, 1.5)
    # NaN only rejected when `check_nan` is requested (the call-site default
    # leaves it off; the explicit check is opt-in).
    @test_throws ArgumentError _quantile_optimization(
        d, NaN; check_nan = true)
end

@testitem "_quantile_optimization: converges on a censored target" begin
    using CensoredDistributions: _quantile_optimization
    using Distributions

    # The real call site: a PrimaryCensored distribution whose CDF has no
    # closed-form inverse. Convergence means the recovered CDF equals `p` and
    # the quantile is monotone in `p`.
    d = primary_censored(LogNormal(1.5, 0.5), Uniform(0.0, 1.0))
    qs = Float64[]
    for p in (0.1, 0.3, 0.5, 0.7, 0.9)
        q = _quantile_optimization(d, p)
        @test isfinite(q)
        @test isapprox(cdf(d, q), p; atol = 1e-4)
        push!(qs, q)
    end
    @test issorted(qs)
end
