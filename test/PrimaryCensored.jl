
@testitem "Test struct type" begin
    using Distributions
    @test_throws MethodError primary_censored(Normal(1, 2), 7)
end

@testitem "Test constructor" begin
    using Distributions
    use_dist = primary_censored(LogNormal(3.5, 1.5), Uniform(1, 2))
    @test typeof(use_dist) <: CensoredDistributions.PrimaryCensored
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

@testitem "Test cdf method" begin
    using Distributions
    @testset "Constructor and end point" begin
        use_dist = primary_censored(LogNormal(3.5, 1.5), Uniform(1, 2))
        @test cdf(use_dist, 1e8) ≈ 1.0
        @test logcdf(use_dist, -Inf) ≈ -Inf
    end

    @testset "Check CDF function against known Exp(1) with uniform censor on primary" begin
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
end

@testitem "Test ccdf" begin
    using Distributions
    use_dist = primary_censored(LogNormal(3.5, 1.5), Uniform(1, 2))
    @test ccdf(use_dist, 1e8) ≈ 0.0
    @test ccdf(use_dist, 0.0) ≈ 1
    @test logccdf(use_dist, 0.0) ≈ 0.0
end
