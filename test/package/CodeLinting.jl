@testitem "Code linting" tags=[:quality] begin
    # Skip on experimental/pre-release Julia where JET may not be compatible
    if VERSION >= v"1.10" && get(ENV, "JULIA_CI_EXPERIMENTAL", "false") != "true"
        # Run JET in a separate environment to isolate its dependencies
        # This prevents JET version conflicts from breaking package resolution
        jet_env = joinpath(@__DIR__, "..", "jet")
        if isdir(jet_env) && isfile(joinpath(jet_env, "Project.toml"))
            # Instantiate the JET environment first (suppress output)
            inst_result = run(
                pipeline(
                    `julia --project=$jet_env -e "import Pkg; Pkg.instantiate()"`,
                    devnull
                );
                wait = true
            )
            if inst_result.exitcode == 0
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
                @info "JET environment instantiation failed, skipping" VERSION
            end
        else
            @warn "JET test environment not found at $jet_env"
        end
    else
        @info "Skipping JET tests" VERSION experimental = get(
            ENV, "JULIA_CI_EXPERIMENTAL", "false"
        )
    end
end
