#!/usr/bin/env julia
# JuliaFormatter code formatting tests
# Run from the test/formatter directory or with --project=test/formatter

using JuliaFormatter

# Get the project root (two levels up from test/formatter)
project_root = dirname(dirname(@__DIR__))

# Directories to check (exclude worktrees and dev directories)
dirs_to_check = [
    joinpath(project_root, "src"),
    joinpath(project_root, "test"),
    joinpath(project_root, "docs"),
    joinpath(project_root, "benchmark")
]

# Filter to only existing directories
dirs_to_check = filter(isdir, dirs_to_check)

# Check formatting without modifying files
all_formatted = all(dirs_to_check) do dir
    JuliaFormatter.format(dir; verbose = true, overwrite = false)
end

if all_formatted
    println("All files are properly formatted")
    exit(0)
else
    println("Some files are not properly formatted")
    println("Run `task format` or use pre-commit hooks to fix")
    exit(1)
end
