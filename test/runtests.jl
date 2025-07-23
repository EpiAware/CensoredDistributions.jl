using TestItemRunner

# Filter tests based on command line arguments
if "skip_quality" in ARGS
    # Skip quality tests (JET, Aqua, formatting) used in CI for performance
    @run_package_tests filter=ti->!(:quality in ti.tags)
else
    # Run all tests (default for local development)
    @run_package_tests
end
