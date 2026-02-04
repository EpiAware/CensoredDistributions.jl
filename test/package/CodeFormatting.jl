@testitem "Code formatting" tags=[:quality] begin
    using Pkg
    # Run JuliaFormatter in a separate environment to isolate its dependencies
    # This prevents JuliaSyntax version conflicts with JET
    formatter_env = joinpath(@__DIR__, "..", "formatter")
    if isdir(formatter_env) && isfile(joinpath(formatter_env, "Project.toml"))
        # Instantiate the formatter environment
        Pkg.activate(formatter_env)
        Pkg.instantiate()
        Pkg.activate()  # Return to original environment

        # Run formatter check in the isolated environment
        # Pipe output to show formatting results in test logs
        result = run(
            pipeline(
                `julia --project=$formatter_env $(joinpath(formatter_env, "runtests.jl"))`,
                stdout = stdout,
                stderr = stderr
            );
            wait = true
        )
        @test result.exitcode == 0
    else
        @warn "Formatter test environment not found at $formatter_env"
        @test_skip "Formatter environment not found"
    end
end
