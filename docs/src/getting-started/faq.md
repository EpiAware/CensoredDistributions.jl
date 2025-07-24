# [Frequently Asked Questions](@id faq)

This page contains frequently asked questions about CensoredDistributions.jl. If you have a question that is not answered here, please open a discussion on the GitHub repository.

## Working with Tutorials

### Q: How can I run the tutorial notebooks?

**A:** You have two options:

**Option 1: Copy and paste (easiest)**
- Copy code blocks from the online tutorials
- Paste into your Julia REPL or script
- Modify as needed for your analysis

**Option 2: Interactive notebooks**
1. Install Pluto: `] add Pluto`
2. Clone the repository: `git clone https://github.com/EpiAware/CensoredDistributions.jl.git`
3. Start Julia in the repository directory: `julia --project=docs`
4. Start Pluto: `using Pluto; Pluto.run()`
5. Open the `.jl` files in `docs/src/getting-started/tutorials/`

### Q: What are Pluto notebooks?

**A:** [Pluto.jl](https://plutojl.org/) notebooks are reactive - when you change one cell, all dependent cells automatically update. This is different from Jupyter notebooks. Each cell contains either a single expression or code wrapped in `begin...end` or `let...end` blocks.

## Using the Package

### Q: How do I create a primary censored distribution?

**A:** Use the `primary_censored` function:

```@example faq
using CensoredDistributions, Distributions

# Delay distribution (e.g., incubation period)
delay_dist = Gamma(2, 3)

# Primary event distribution (e.g., infection within a day)
primary_dist = Uniform(0, 1)

# Create censored distribution
censored_dist = primary_censored(delay_dist, primary_dist)
```

### Q: What's the difference between the different types of censoring?

**A:**
- **Primary event censoring**: The timing of the initial event in a delay distribution is uncertain. See the [tutorials](@ref tutorials) for detailed examples.
- **Interval censoring**: Continuous values are observed only within discrete intervals. See the [API documentation](@ref "Public API") for `interval_censored`.
- **Double interval censoring**: Combines both types of censoring. See `double_interval_censored` in the [API documentation](@ref "Public API").

### Q: How do I fit censored distributions to data?

**A:** See the Fitting with Turing.jl tutorial in the [tutorials](@ref tutorials) section for Bayesian inference examples using Turing.jl.

### Q: Which distributions have analytical solutions for better performance?

**A:** See the Analytical CDF Solutions tutorial in the [tutorials](@ref tutorials) section for details on which distribution combinations have optimized implementations.

### Q: Can I use this with automatic differentiation?

**A:** Yes! The package supports ForwardDiff.jl, ReverseDiff.jl, Zygote.jl, and Enzyme.jl for use with Turing.jl and other AD-based workflows.

## Common Issues

### Q: I get "Package not found" errors

**A:** Make sure you're in the right environment:

```julia
using Pkg
Pkg.activate(".")           # Activate current directory
Pkg.instantiate()           # Install dependencies
Pkg.add("CensoredDistributions")  # Add the package if needed
```

### Q: How do I cite this package?

**A:** Please cite the GitHub repository and mention the version you used. Citation information for the associated paper will be added when available.

### Q: I want to contribute to development

**A:** See the [Developer FAQ](@ref developer-faq) and [Contributing Guide](@ref contributing) for development-specific questions and guidelines.

## Getting Help

Still have questions?

- **Package-specific**: Open a [GitHub Discussion](https://github.com/EpiAware/CensoredDistributions.jl/discussions)
- **General Julia help**: [Julia Discourse](https://discourse.julialang.org/) or [Julia Slack](https://julialang.org/slack/)
- **Bug reports**: [GitHub Issues](https://github.com/EpiAware/CensoredDistributions.jl/issues)
