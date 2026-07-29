# README structure check, run through the shared EpiAwarePackageTools
# helper so the required sections and their order stay defined in one
# place for the whole org (see `EpiAwarePackageTools.STANDARD_README_SECTIONS`).
#
# `test_readme_sections` asserts the README opens with a single H1 title,
# carries the `<!-- badges:start -->` / `<!-- badges:end -->` markers, and
# has a Why, Getting started, Documentation, Contributing and Citing
# section in that order. The standard-sections managed block is not
# adopted here (issue #914), so only the ordering check applies.
#
# The rest of this directory still holds the hand-written QA testsets
# (Aqua, CodeFormatting, CodeLinting, DocstringFormat, DocTest,
# ExplicitImports). PR #884 replaces all of them, and this file, with the
# managed scaffold version, which runs the same README check.

@testitem "Quality: README sections" tags=[:quality, :readme] begin
    using EpiAwarePackageTools: test_readme_sections
    test_readme_sections(joinpath(@__DIR__, "..", ".."))
end
