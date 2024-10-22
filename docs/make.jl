using Pkg: Pkg
Pkg.instantiate()

using Documenter
using PrimaryCensored
using Pluto: Configuration.CompilerOptions
using PlutoStaticHTML

include("changelog.jl")
include("pages.jl")
include("build.jl")

build("getting-started")
build("getting-started/tutorials")

DocMeta.setdocmeta!(
    PrimaryCensored, :DocTestSetup, :(using PrimaryCensored); recursive = true)

makedocs(; sitename = "PrimaryCensored.jl",
    authors = "Samuel Brand, Sam Abbott, and contributors",
    clean = true, doctest = false, linkcheck = true,
    warnonly = [:docs_block, :missing_docs, :linkcheck, :autodocs_block],
    modules = [PrimaryCensored],
    pages = pages,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        mathengine = Documenter.MathJax3(),
        size_threshold = 6000 * 2^10,
        size_threshold_warn = 2000 * 2^10
    )
)

deploydocs(
    repo = "github.com/epinowcast/PrimaryCensored.jl.git",
    target = "build",
    push_preview = true
)
