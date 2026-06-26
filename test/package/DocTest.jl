@testitem "Run docstring tests" begin
    using EpiAwarePackageTools: test_doctest
    # Run the package doctests through the shared kit wrapper.
    test_doctest(CensoredDistributions)
end
