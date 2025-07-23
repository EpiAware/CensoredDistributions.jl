using TestItemRunner

# Filter tests based on command line arguments
if "skip_jet" in ARGS
    # Skip JET tests (used in CI for performance)
    @run_package_tests filter=ti->!(:jet in ti.tags)
else
    # Run all tests (default for local development)
    @run_package_tests
end
