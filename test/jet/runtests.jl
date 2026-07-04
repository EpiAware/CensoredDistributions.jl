#!/usr/bin/env julia
# MANAGED by EpiAwarePackageTools.scaffold — do not edit by hand.
#
# JET static-analysis runner, run in this isolated environment so JET's
# JuliaSyntax pin does not clash with the main test deps.
#
#   julia --project=test/jet test/jet/runtests.jl
#
# A package whose public surface is DynamicPPL `@model` functions gets spurious
# JET reports for every `~`/`:=` line (the tilde macro hides the assignment from
# JET). To suppress exactly those, drop a package-owned `test/jet/jet_config.jl`
# that defines `JET_REPORT_FILTER` (a `report -> Bool` predicate; a report is
# kept when it returns `true`). `EpiAwarePackageTools.dynamicppl_model_filter`
# is the ready-made filter for `@model` packages. Without the config the runner
# fails on any report (the strict default).

using JET
using EpiAwarePackageTools: dynamicppl_model_filter
using CensoredDistributions

const _CONFIG = joinpath(@__DIR__, "jet_config.jl")
isfile(_CONFIG) && include(_CONFIG)

if @isdefined(JET_REPORT_FILTER)
    result = JET.report_package(CensoredDistributions;
        target_modules = (CensoredDistributions,))
    kept = filter(JET_REPORT_FILTER, JET.get_reports(result))
    for r in kept
        @info "JET report (not filtered)" report = sprint(show, r)
    end
    isempty(kept) || error("JET found $(length(kept)) report(s)")
    println("JET: no reports survived the configured filter")
else
    JET.test_package(CensoredDistributions; target_modules = (CensoredDistributions,))
end
