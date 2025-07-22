@testitem "Test doublecensored structure equivalence - primary only" begin
    using Distributions
    
    # Test case 1: Primary censoring only
    uncensored_dist = LogNormal(1.5, 0.75)
    censoring_dist = Uniform(0, 1)
    
    # Manual approach
    manual_dist = primarycensored(uncensored_dist, censoring_dist)
    
    # Convenience function approach
    convenience_dist = doublecensored(uncensored_dist, censoring_dist)
    
    # Test they produce same type and structure
    @test typeof(manual_dist) == typeof(convenience_dist)
    @test params(manual_dist) == params(convenience_dist)
end

@testitem "Test doublecensored structure equivalence - with truncation" begin
    using Distributions
    
    # Test case 2: Primary censoring + truncation
    uncensored_dist = LogNormal(1.5, 0.75)
    censoring_dist = Uniform(0, 1)
    upper_bound = 10.0
    
    # Manual approach
    manual_dist = primarycensored(uncensored_dist, censoring_dist) |>
                  d -> truncated(d; upper=upper_bound)
    
    # Convenience function approach
    convenience_dist = doublecensored(uncensored_dist, censoring_dist; upper=upper_bound)
    
    # Test they produce same type
    @test typeof(manual_dist) == typeof(convenience_dist)
    
    # Test bounds are the same
    @test maximum(manual_dist) == maximum(convenience_dist)
    @test minimum(manual_dist) == minimum(convenience_dist)
end

@testitem "Test doublecensored structure equivalence - with discretisation" begin
    using Distributions
    
    # Test case 3: Primary censoring + discretisation
    uncensored_dist = LogNormal(1.5, 0.75)
    censoring_dist = Uniform(0, 1)
    interval_width = 1.0
    
    # Manual approach
    manual_dist = primarycensored(uncensored_dist, censoring_dist) |>
                  d -> discretise(d, interval_width)
    
    # Convenience function approach
    convenience_dist = doublecensored(uncensored_dist, censoring_dist; interval=interval_width)
    
    # Test they produce same type
    @test typeof(manual_dist) == typeof(convenience_dist)
end

@testitem "Test doublecensored structure equivalence - full pipeline" begin
    using Distributions
    
    # Test case 4: Full pipeline - Primary censoring + truncation + discretisation
    uncensored_dist = LogNormal(1.5, 0.75)
    censoring_dist = Uniform(0, 1)
    upper_bound = 10.0
    interval_width = 1.0
    
    # Manual approach (ensuring correct order: primary -> truncation -> discretisation)
    manual_dist = primarycensored(uncensored_dist, censoring_dist) |>
                  d -> truncated(d; upper=upper_bound) |>
                  d -> discretise(d, interval_width)
    
    # Convenience function approach
    convenience_dist = doublecensored(uncensored_dist, censoring_dist; 
                                    upper=upper_bound, interval=interval_width)
    
    # Test they produce same type
    @test typeof(manual_dist) == typeof(convenience_dist)
end

@testitem "Test doublecensored with both bounds" begin
    using Distributions
    
    # Test case 5: With both lower and upper bounds
    uncensored_dist = LogNormal(1.5, 0.75)
    censoring_dist = Uniform(0, 1)
    lower_bound = 2.0
    upper_bound = 8.0
    
    # Manual approach
    manual_dist = primarycensored(uncensored_dist, censoring_dist) |>
                  d -> truncated(d, lower_bound, upper_bound)
    
    # Convenience function approach
    convenience_dist = doublecensored(uncensored_dist, censoring_dist; 
                                    lower=lower_bound, upper=upper_bound)
    
    # Test they produce same type and bounds
    @test typeof(manual_dist) == typeof(convenience_dist)
    @test minimum(manual_dist) == minimum(convenience_dist) == lower_bound
    @test maximum(manual_dist) == maximum(convenience_dist) == upper_bound
end

@testitem "Test doublecensored return types" begin
    using Distributions
    
    # Test that function returns expected types for different combinations
    gamma_dist = Gamma(2, 1)
    uniform_censoring = Uniform(0, 1)
    
    # Primary only -> PrimaryCensored
    @test isa(doublecensored(gamma_dist, uniform_censoring), 
              CensoredDistributions.PrimaryCensored)
    
    # Primary + truncation -> Truncated{PrimaryCensored}
    @test isa(doublecensored(gamma_dist, uniform_censoring; upper=5), 
              Truncated)
    
    # Primary + discretisation -> Discretised{PrimaryCensored}
    @test isa(doublecensored(gamma_dist, uniform_censoring; interval=0.5), 
              CensoredDistributions.Discretised)
    
    # Full pipeline -> Discretised{Truncated{PrimaryCensored}}
    full_dist = doublecensored(gamma_dist, uniform_censoring; upper=5, interval=0.5)
    @test isa(full_dist, CensoredDistributions.Discretised)
end

@testitem "Test doublecensored with nothing parameters" begin
    using Distributions
    
    # Test with nothing values (should be equivalent to not providing the argument)
    gamma_dist = Gamma(2, 1)
    uniform_censoring = Uniform(0, 1)
    
    dist1 = doublecensored(gamma_dist, uniform_censoring)
    dist2 = doublecensored(gamma_dist, uniform_censoring; 
                          lower=nothing, upper=nothing, interval=nothing)
    
    @test typeof(dist1) == typeof(dist2)
    @test params(dist1) == params(dist2)
end