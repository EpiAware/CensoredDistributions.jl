getting_started_pages = Any[
    "Overview" => "getting-started/index.md",
    "Installation" => "getting-started/installation.md",
    "Quickstart" => "getting-started/quickstart.md",
    "Frequently asked questions" => "getting-started/faq.md",
    "Explainers" => [
        "Overview" => "getting-started/explainers/index.md"
    ],
    "Tutorials" => [
        "Overview" => "getting-started/tutorials/index.md",
        "Fitting distributions with PrimaryCensored and Turing" => "getting-started/tutorials/fitting-dists-with-Turing.md"
    ]
]

module_pages = [
    "Overview" => "lib/index.md",
    "Public API" => "lib/public.md",
    "Internal API" => "lib/internals.md"
]

developer_pages = [
    "Overview" => "developer/index.md",
    "Contributing" => "developer/contributing.md",
    "Release checklist" => "developer/checklist.md"
]

pages = [
    "PrimaryCensored.jl: Primary event censored distributions" => "index.md",
    "Getting started" => getting_started_pages,
    "What is CensoredDistributions.jl?" => "overview.md",
    "Modules" => module_pages,
    "Developers" => developer_pages,
    "release-notes.md"
]
