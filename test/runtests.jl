using TestItemRunner

# Filter tests based on command line arguments
if "skip_quality" in ARGS
    # Skip quality tests (JET, Aqua, formatting) used in CI for performance
    @run_package_tests filter=ti->!(:quality in ti.tags)
elseif "readme_only" in ARGS
    # Run only README tests
    @run_package_tests filter=ti->:readme in ti.tags
elseif "optimization_only" in ARGS
    # Run only OptimizationExt tests
    @run_package_tests filter=ti->:optimization in ti.tags
elseif "optimization_unit" in ARGS
    # Run only OptimizationExt unit tests (fast)
    @run_package_tests filter=ti->:unit in ti.tags
elseif "optimization_failure" in ARGS
    # Run only OptimizationExt failure tests
    @run_package_tests filter=ti->:failure in ti.tags
elseif "optimization_recovery" in ARGS
    # Run only OptimizationExt parameter recovery tests
    @run_package_tests filter=ti->:recovery in ti.tags
else
    # Run all tests (default for local development)
    @run_package_tests
end
