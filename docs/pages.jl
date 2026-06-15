getting_started_pages = Any[
    "Overview" => "getting-started/index.md",
    "Installation" => "getting-started/installation.md",
    "Tutorials" => [
        "Analytical CDF solutions" => "getting-started/tutorials/analytical-primarycensored-cdfs.md",
        "Exponentially tilted primary events" => "getting-started/tutorials/exponentially-tilted-primary-events.md",
        "Automatic differentiation backends" => "getting-started/tutorials/ad-backends.md",
        "Fitting with Turing.jl" => "getting-started/tutorials/fitting-with-turing.md",
        "Composing censored distributions" => "getting-started/tutorials/composer-toolkit.md",
        "SurvivalDistributions.jl delay families" => "getting-started/tutorials/survival-delay-families.md",
        "Rt renewal with delay convolution" => "getting-started/tutorials/rt-renewal-convolution.md",
        "Composed delay as ODE compartments" => "getting-started/tutorials/linear-chain-sir.md",
        "Epinowcast-style hazard nowcasting" => "getting-started/tutorials/epinowcast-nowcasting.md",
        "Fit marginal, sample event based" => "getting-started/tutorials/fit-marginal-sample-event-based.md",
        "Bundibugyo Ebola delays" => "getting-started/tutorials/bdbv-linelist-analysis.md",
        "Real-time Andes virus delays" => "getting-started/tutorials/andv-linelist-analysis.md",
        "Stratified Sierra Leone Ebola delays" => "getting-started/tutorials/ebola-stratified-delays.md"
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
    "Extending the composer toolkit" => "developer/extending.md",
    "Developer FAQ" => "developer/faq.md",
    "Release process" => "developer/release-process.md"
]

pages = [
    "Getting started" => getting_started_pages,
    "Modules" => module_pages,
    "Development" => dev_pages,
    "Release notes" => "release-notes.md"
]
