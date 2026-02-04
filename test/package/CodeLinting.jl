@testitem "Code linting" tags=[:quality] begin
    using Pkg
    # Skip on experimental/pre-release Julia where JET may not be compatible
    if VERSION >= v"1.10" && get(ENV, "JULIA_CI_EXPERIMENTAL", "false") != "true"
        # Run JET in a separate environment to isolate its dependencies
        # This prevents JET version conflicts from breaking package resolution
        jet_env = joinpath(@__DIR__, "..", "jet")
        if isdir(jet_env) && isfile(joinpath(jet_env, "Project.toml"))
            # Instantiate the JET environment
            Pkg.activate(jet_env)
            Pkg.instantiate()
            Pkg.activate()  # Return to original environment

            # Run JET tests in the isolated environment
            # Pipe output to show JET results in test logs
            result = run(
                pipeline(
                    `julia --project=$jet_env $(joinpath(jet_env, "runtests.jl"))`,
                    stdout = stdout,
                    stderr = stderr
                );
                wait = true
            )
            @test result.exitcode == 0
        else
            @warn "JET test environment not found at $jet_env"
            @test_skip "JET environment not found"
        end
    else
        @info "Skipping JET tests" VERSION experimental = get(
            ENV, "JULIA_CI_EXPERIMENTAL", "false"
        )
        @test_skip "JET skipped on experimental Julia"
    end
end
