using TestItemRunner

# Suppress hypergeometric function warnings from extreme parameter tests
using Logging

# Create a custom logger that filters out only HypergeometricFunctions warnings
struct FilteredLogger{T} <: Logging.AbstractLogger
    logger::T
end

Logging.min_enabled_level(logger::FilteredLogger) = Logging.min_enabled_level(logger.logger)

function Logging.shouldlog(logger::FilteredLogger, level, _module, group, id)
    # Filter out HypergeometricFunctions warnings
    if level >= Logging.Warn && _module == :HypergeometricFunctions
        return false
    end
    return Logging.shouldlog(logger.logger, level, _module, group, id)
end

function Logging.handle_message(logger::FilteredLogger, level, message, _module,
        group, id, file, line; maxlog = nothing, kwargs...)
    Logging.handle_message(logger.logger, level, message, _module, group,
        id, file, line; maxlog = maxlog, kwargs...)
end

# Set up the filtered logger
filtered_logger = FilteredLogger(Logging.current_logger())
Logging.global_logger(filtered_logger)

# Filter tests based on command line arguments
if "skip_quality" in ARGS
    # Skip quality tests (JET, Aqua, formatting) used in CI for performance
    @run_package_tests filter = ti -> !(:quality in ti.tags)
elseif "quality_only" in ARGS
    # Run only quality tests (Aqua, formatting, linting, doctests)
    @run_package_tests filter = ti -> :quality in ti.tags
elseif "readme_only" in ARGS
    # Run only README tests
    @run_package_tests filter = ti -> :readme in ti.tags
else
    # Run all tests (default for local development)
    @run_package_tests
end
