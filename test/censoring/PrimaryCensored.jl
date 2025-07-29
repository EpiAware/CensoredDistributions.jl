
@testitem "Test struct type" begin
    using Distributions
    @test_throws MethodError primary_censored(Normal(1, 2), 7)
end

@testitem "Test constructor" begin
    using Distributions
    use_dist = primary_censored(LogNormal(3.5, 1.5), Uniform(1, 2))
    @test typeof(use_dist) <: CensoredDistributions.PrimaryCensored
end

@testitem "Default constructor (analytical solver)" begin
    using Distributions
    using Integrals

    dist = Gamma(2.0, 3.0)
    primary = Uniform(0.0, 1.0)
    d = primary_censored(dist, primary)

    @test d.dist === dist
    @test d.primary_event === primary
    @test d.method isa CensoredDistributions.AnalyticalSolver
    @test d.method.solver isa QuadGKJL
end

@testitem "Constructor with force_numeric" begin
    using Distributions
    using Integrals

    dist = LogNormal(1.5, 0.75)
    primary = Uniform(0.0, 1.0)
    d = primary_censored(dist, primary; force_numeric = true)

    @test d.dist === dist
    @test d.primary_event === primary
    @test d.method isa CensoredDistributions.NumericSolver
    @test d.method.solver isa QuadGKJL
end

@testitem "Constructor with custom solver" begin
    using Distributions
    using Integrals

    dist = Weibull(2.0, 1.5)
    primary = Uniform(0.0, 1.0)
    custom_solver = HCubatureJL()
    d = primary_censored(dist, primary; solver = custom_solver, force_numeric = false)

    @test d.method isa CensoredDistributions.AnalyticalSolver
    @test d.method.solver === custom_solver
end

@testitem "Test random generation" begin
    using Distributions
    use_dist = primary_censored(LogNormal(3.5, 1.5), Uniform(1, 2))
    @test length(rand(use_dist, 10)) == 10

    use_dist_trunc = truncated(use_dist, 3, 10)
    use_dist_trunc_rn = rand(use_dist_trunc, 1000000)
    @test length(use_dist_trunc_rn) ≈ 1e6
    @test maximum(use_dist_trunc_rn) <= 10
    @test minimum(use_dist_trunc_rn) >= 3
end

@tesitem "Params calls" begin
    using Distributions
    use_dist = primary_censored(LogNormal(3.5, 1.5), Uniform(1, 2))
    extracted_params = params(use_dist)
    @test length(extracted_params) == 4
    @test extracted_params[1] ≈ 3.5
    @test extracted_params[3] ≈ 1.0
end

@testitem "CDF constructor and end point" begin
    using Distributions

    use_dist = primary_censored(LogNormal(3.5, 1.5), Uniform(1, 2))
    @test cdf(use_dist, 1e8) ≈ 1.0
    @test logcdf(use_dist, -Inf) ≈ -Inf
end

@testitem "CDF against known Exponential analytical solution" begin
    using Distributions

    dist = Exponential(1.0)
    use_dist = primary_censored(dist, Uniform(0, 1))
    # Analytical solution for the pmf of observation in [0,1], [1,2], ...
    expected_pmf_uncond = [exp(-1)
                           [(1 - exp(-1)) * (exp(1) - 1) * exp(-s) for s in 1:9]]
    # Analytical solution for the cdf
    expected_cdf = [0.0; cumsum(expected_pmf_uncond)]
    # Calculated cdf
    calc_cdf = map(0:10) do t
        cdf(use_dist, t)
    end
    @test expected_cdf ≈ calc_cdf
end

@testitem "Test ccdf" begin
    using Distributions
    use_dist = primary_censored(LogNormal(3.5, 1.5), Uniform(1, 2))
    @test ccdf(use_dist, 1e8) ≈ 0.0
    @test ccdf(use_dist, 0.0) ≈ 1
    @test logccdf(use_dist, 0.0) ≈ 0.0
end

@testitem "Keyword argument constructor" begin
    using Distributions

    dist = LogNormal(1.5, 0.75)
    primary = Uniform(0, 2)

    # Test keyword version returns correct PrimaryCensored struct
    d_kw = primary_censored(dist; primary_event = primary)
    @test typeof(d_kw) <: CensoredDistributions.PrimaryCensored
    @test d_kw.dist === dist
    @test d_kw.primary_event === primary

    # Test default primary_event
    d_default = primary_censored(dist)
    @test typeof(d_default) <: CensoredDistributions.PrimaryCensored
    @test d_default.dist === dist
    @test d_default.primary_event == Uniform(0, 1)
end
