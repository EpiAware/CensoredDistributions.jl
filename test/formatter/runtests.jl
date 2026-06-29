#!/usr/bin/env julia
# MANAGED by EpiAwarePackageTools.scaffold — do not edit by hand.
#
# JuliaFormatter check, run in this isolated environment so its JuliaSyntax pin
# does not clash with JET. Checks the standard source trees without modifying
# them; exits non-zero if any file is not formatted.
#
#   julia --project=test/formatter test/formatter/runtests.jl

using JuliaFormatter

# Project root is two levels up from test/formatter.
project_root = dirname(dirname(@__DIR__))
dirs = filter(isdir,
    [joinpath(project_root, d) for d in ("src", "test", "docs", "benchmark")])

all_formatted = all(dirs) do dir
    JuliaFormatter.format(dir; verbose = true, overwrite = false)
end

if all_formatted
    println("All files are properly formatted")
    exit(0)
else
    println("Some files are not properly formatted")
    println("Run `task format` or the pre-commit hooks to fix")
    exit(1)
end
