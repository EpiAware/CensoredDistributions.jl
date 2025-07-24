getting_started_pages = Any[
    "Overview" => "getting-started/index.md",
    "Installation" => "getting-started/installation.md",
    "Tutorials" => [
        "Analytical CDF Solutions" => "getting-started/tutorials/analytical-primarycensored-cdfs.md",
        "Maximum Likelihood Estimation" => "getting-started/tutorials/mle-fitting.md",
        "Fitting with Turing.jl" => "getting-started/tutorials/fitting-with-turing.md"
    ],
    "Julia" => "getting-started/julia.md",
    "FAQ" => "getting-started/faq.md"
]

module_pages = [
    "Public API" => "lib/public.md",
    "Internal API" => "lib/internals.md"
]

dev_pages = [
    "Overview" => "developer/index.md",
    "Contributing" => "developer/contributing.md",
    "Developer FAQ" => "developer/faq.md",
    "Release Process" => "developer/release-process.md"
]

pages = [
    "CensoredDistributions.jl: Primary event censored distributions" => "index.md",
    "Getting started" => getting_started_pages,
    "Modules" => module_pages,
    "Development" => dev_pages
]
