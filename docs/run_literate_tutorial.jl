# MANAGED by EpiAwarePackageTools.scaffold — do not edit by hand.
# Run a single Literate tutorial in its own process with `execute = true`.
#
# Heavy tutorials (live MCMC fits + CairoMakie/PairPlots) accumulate native and
# memory state when executed back-to-back inside one long-lived Documenter
# process, which SIGSEGVs the full build. Executing each one here, in a fresh
# `julia --project=docs` subprocess, runs the code once per process so that
# state cannot accumulate. Literate embeds the captured outputs as static
# ````julia```` blocks (DocumenterFlavor + execute = true), so Documenter only
# renders them and never re-executes them.
#
# Usage: julia --project=docs docs/run_literate_tutorial.jl <input.jl> <outdir>

using Literate

input = ARGS[1]
outdir = ARGS[2]

Literate.markdown(
    input,
    outdir;
    flavor = Literate.DocumenterFlavor(),
    execute = true,
    mdstrings = true,
    credit = false
)
