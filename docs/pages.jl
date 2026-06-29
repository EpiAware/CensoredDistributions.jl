getting_started_pages = Any[
    "Overview" => "getting-started/index.md",
    "Installation" => "getting-started/installation.md",
    "Tutorials" => [
        "Distributions and censoring" => [
            "Analytical CDF solutions" => "getting-started/tutorials/analytical-primarycensored-cdfs.md",
            "Exponentially tilted primary events" => "getting-started/tutorials/exponentially-tilted-primary-events.md",
            "SurvivalDistributions.jl delay families" => "getting-started/tutorials/survival-delay-families.md"
        ],
        "Composing and fitting" => [
            "Composing censored distributions" => "getting-started/tutorials/composer-toolkit.md",
            "Fitting with Turing.jl" => "getting-started/tutorials/fitting-with-turing.md",
            "Fit marginal, sample event based" => "getting-started/tutorials/fit-marginal-sample-event-based.md",
            "Automatic differentiation backends" => "getting-started/tutorials/ad-backends.md"
        ],
        "Case studies and applications" => [
            "Bundibugyo Ebola delays" => "getting-started/tutorials/bdbv-linelist-analysis.md",
            "Real-time Andes virus delays" => "getting-started/tutorials/andv-linelist-analysis.md",
            "Stratified Sierra Leone Ebola delays" => "getting-started/tutorials/ebola-stratified-delays.md",
            "Branching-process competing outcomes" => "getting-started/tutorials/branching-competing.md",
            "Recurrent multi-state transitions" => "getting-started/tutorials/recurrent-multistate.md",
            "Pairwise survival of transmission" => "getting-started/tutorials/pairwise-survival-transmission.md",
            "Rt renewal with delay convolution" => "getting-started/tutorials/rt-renewal-convolution.md",
            "Susceptibility-depleting renewal" => "getting-started/tutorials/renewal-susceptibility.md",
            "Epinowcast-style hazard nowcasting" => "getting-started/tutorials/epinowcast-nowcasting.md",
            "Composed delays as compartments" => "getting-started/tutorials/linear-chain-sir.md"
        ]
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
    "Interface contracts" => "developer/interface-contracts.md",
    "LogDensityProblems inference layer" => "developer/logdensity-layer.md",
    "Composable renewal step (design)" => "developer/renewal-step-611.md",
    "Recurrent multi-state design" => "developer/recurrent-multistate-545.md",
    "Developer FAQ" => "developer/faq.md",
    "Release process" => "developer/release-process.md"
]

pages = [
    "Getting started" => getting_started_pages,
    "API reference" => module_pages,
    "Development" => dev_pages,
    "Release notes" => "release-notes.md"
]
