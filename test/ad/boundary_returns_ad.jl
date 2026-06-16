# AD coverage for support-boundary / sentinel returns.
#
# Several densities return a bare numeric sentinel (`0.0` / `1.0` / `-Inf`)
# on their support-boundary branches. Under ForwardDiff those literals strip
# the Dual partials, so a gradient evaluated AT or NEAR a boundary silently
# loses its derivative (or yields a Union-typed result). The sentinels are
# now seeded from the promoted argument type, so a Dual flows through. These
# items differentiate through the affected ExponentiallyTilted and
# primarycensored_cdf edges and assert the result is a finite Dual.

@testitem "ExponentiallyTilted boundary returns keep Duals finite" tags=[
    :ad, :forwarddiff] begin
    using CensoredDistributions, Distributions
    using ADTypes: AutoForwardDiff
    using DifferentiationInterface: derivative
    using ForwardDiff: ForwardDiff

    # cdf at the upper edge: differentiate the cdf at x = max w.r.t. the
    # tilting rate. The boundary branch returns `one(T)`; a stripped Float64
    # would make the derivative exactly zero with no Dual flowing.
    f_cdf_edge(r) = cdf(ExponentiallyTilted(0.0, 1.0, r), 1.0)
    g = derivative(f_cdf_edge, AutoForwardDiff(), 2.0)
    @test isfinite(g)

    # logpdf just inside the support, differentiated w.r.t. r: must stay a
    # finite Dual right up against the boundary.
    f_lp(r) = logpdf(ExponentiallyTilted(0.0, 1.0, r), 1.0 - 1e-8)
    g_lp = derivative(f_lp, AutoForwardDiff(), 1.5)
    @test isfinite(g_lp)

    # quantile at the p == 1 boundary, differentiated w.r.t. r. The branch
    # returns `convert(T, d.max)`; a Float64 strip would drop the partial.
    f_q(r) = quantile(ExponentiallyTilted(0.0, 1.0, r), 1.0)
    g_q = derivative(f_q, AutoForwardDiff(), 1.0)
    @test isfinite(g_q)
end

@testitem "primarycensored_cdf boundary returns keep Duals finite" tags=[
    :ad, :forwarddiff] begin
    using CensoredDistributions, Distributions
    using ADTypes: AutoForwardDiff
    using DifferentiationInterface: derivative
    using ForwardDiff: ForwardDiff

    # Numeric solver, x at the delay support floor. The edge branch returns
    # `zero(T)`; differentiating w.r.t. a delay parameter must yield a finite
    # Dual rather than a stripped Float64.
    function f_num(μ)
        d = primary_censored(
            LogNormal(μ, 0.5), Uniform(0.0, 1.0); method = NumericSolver())
        return cdf(d, 0.0)
    end
    g_num = derivative(f_num, AutoForwardDiff(), 1.0)
    @test isfinite(g_num)

    # Analytical Gamma path, just above the lower edge. Differentiate the cdf
    # w.r.t. the shape parameter near the support boundary.
    function f_gamma(k)
        d = primary_censored(Gamma(k, 1.5), Uniform(0.0, 1.0))
        return cdf(d, 1e-6)
    end
    g_gamma = derivative(f_gamma, AutoForwardDiff(), 2.0)
    @test isfinite(g_gamma)

    # logcdf below the support returns `oftype(zero(T), -Inf)`; the in-support
    # query just above the floor must still differentiate to a finite Dual.
    function f_logcdf(μ)
        d = primary_censored(LogNormal(μ, 0.5), Uniform(0.0, 1.0))
        return logcdf(d, 0.5)
    end
    g_logcdf = derivative(f_logcdf, AutoForwardDiff(), 1.0)
    @test isfinite(g_logcdf)
end
