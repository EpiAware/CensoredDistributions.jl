# [Getting Started with Julia](@id julia)

This guide helps you set up Julia for working effectively with CensoredDistributions.jl, whether you're using the package for analysis or contributing to its development. It's aimed at people familiar with other technical computing languages (R, Python, MATLAB) but new to Julia workflows.

> [!NOTE]
> If you're familiar with other languages for technical computing, these [noteworthy differences](https://docs.julialang.org/en/v1/manual/noteworthy-differences/) may be useful.

## What this guide is and isn't

This isn't a comprehensive guide to learning Julia programming. Instead, we provide practical setup advice to get you working productively with CensoredDistributions.jl quickly.

**To learn Julia programming, we recommend:**
- [Julia Documentation - getting started](https://docs.julialang.org/en/v1/manual/getting-started/)
- [Julia learning resources](https://julialang.org/learning/)
- [Julia Discourse](https://discourse.julialang.org/) - community forum

## Julia Installation with juliaup

1. **Download juliaup**: This is the official cross-platform installer/updater for Julia. Go to the [juliaup GitHub repository](https://github.com/JuliaLang/juliaup) or [install.julialang.org](https://install.julialang.org) for installation instructions.

2. **Verify Installation**: Open a terminal and type `julia` to start the Julia REPL. You should see a Julia prompt `julia>`.

**Why juliaup?** Easy version management, automatic updates, and seamless switching between Julia versions for different projects.

👉 **Learn more**: [juliaup GitHub repository](https://github.com/JuliaLang/juliaup) for detailed usage instructions.

## Editor Setup: VSCode with Julia Extension

**Recommended setup:**

1. Install [Visual Studio Code](https://code.visualstudio.com/)
2. Install the [Julia extension](https://marketplace.visualstudio.com/items?itemName=julialang.language-julia)

**Key features:**
- **Integrated REPL**: Execute code directly from editor
- **Test Explorer**: Run individual tests from sidebar with coverage visualization
- **Debugging**: Set breakpoints, inspect variables, step through code
- **Plot viewer**: Dedicated pane for visualizations
- **Symbol navigation**: Go to definitions, find references

👉 **Learn more**: [Julia VSCode documentation](https://www.julia-vscode.org/docs/stable/)

## Julia Environments

Julia uses [**environments**](https://docs.julialang.org/en/v1/manual/code-loading/#Environments-1) to manage project dependencies. Each project can have isolated packages and versions.

**Key concepts:**
- `Project.toml`: Lists project dependencies
- `Manifest.toml`: Records exact versions (like a lockfile)
- Environments can be [stacked](https://docs.julialang.org/en/v1/manual/code-loading/#Environment-stacks) - global packages available to projects

👉 **Learn more**: [Julia Pkg documentation](https://pkgdocs.julialang.org/v1/environments/)

### Using environments from the REPL

```julia-repl
julia> ]                    # Enter package mode
(@v1.11) pkg> activate .    # Activate current directory as environment
(myproject) pkg> add SomePackage   # Add package to project
(myproject) pkg> status     # See what's installed
```

**Common commands:**
- `activate .` - activate current directory as environment
- `activate --temp` - create temporary environment for experiments
- `instantiate` - install all dependencies listed in Project.toml

## Recommended packages for your global Julia environment

Install these in your Julia version environment (e.g., `@v1.11`) to make them available across projects:

```julia-repl
julia> ]
(@v1.11) pkg> add Revise OhMyREPL BenchmarkTools TestEnv
```

**Recommended packages:**
- **Revise**: Automatic code reloading - essential for development
- **OhMyREPL**: Better REPL with syntax highlighting
- **BenchmarkTools**: Performance measurement
- **TestEnv**: Easy switching to test environments

## startup.jl configuration

Automatically load essential tools by creating `~/.julia/config/startup.jl`:

```julia
atreplinit() do repl
    # Load Revise for automatic code reloading
    try
        @eval using Revise
    catch e
        @warn "error while importing Revise" e
    end

    # Load OhMyREPL for better REPL experience
    try
        @eval using OhMyREPL
    catch e
        @warn "error while importing OhMyREPL" e
    end

    # Pkg convenience functions
    try
        @eval using Pkg
        @eval st() = Pkg.status()
        @eval up() = Pkg.update()
    catch e
        @warn "error while importing Pkg" e
    end
end
```

## Working with CensoredDistributions.jl

### Installing and using the package

```julia-repl
julia> ]
(@v1.11) pkg> add CensoredDistributions
julia> using CensoredDistributions

# Start using the package
julia> using Distributions
julia> primary_censored(Gamma(2, 1), Uniform(0, 1))
```

### Working with the tutorials

The [tutorials](@ref tutorials) include interactive Pluto notebooks. See the [FAQ](@ref faq) for detailed instructions on running the notebooks vs. copying code.

### Development workflow (for contributors)

If you want to contribute to the package, see the [Contributing Guide](@ref contributing) and [Developer FAQ](@ref developer-faq) for detailed guidance.

```bash
# Clone and enter package directory
git clone https://github.com/EpiAware/CensoredDistributions.jl.git
cd CensoredDistributions.jl

# Start Julia in package environment
julia --project=.
```

```julia-repl
julia> using CensoredDistributions  # Load package (automatically reloads with Revise)

# Make changes to source code - they reload automatically
# Test changes interactively
julia> primary_censored(Gamma(2, 1), Uniform(0, 1))
```

### Running tests

```julia-repl
julia> ]
(CensoredDistributions) pkg> test    # Run all tests

# Or from command line
julia --project=. -e 'using Pkg; Pkg.test()'
```

### Quick environment switching

```julia-repl
julia> using TestEnv
julia> TestEnv.activate()    # Switch to test environment with package available
```

## Common issues and solutions

**Package not found:**
```julia-repl
julia> ]
pkg> activate .        # Ensure correct environment
pkg> instantiate       # Install missing dependencies
```

**Changes not reflecting:**
- Ensure Revise is loaded in startup.jl
- Or restart Julia and reload your package

**Environment conflicts:**
```julia-repl
julia> ]
pkg> resolve    # Resolve version conflicts
pkg> update     # Update to compatible versions
```

## Next Steps

With this setup, you're ready to:
- Work through the [tutorials](@ref tutorials) to learn CensoredDistributions.jl
- Explore the [API documentation](@ref "Public API")
- [Contribute to the project](@ref contributing) if you're interested

## Additional Resources

**Community & Help:**
- [Julia Discourse](https://discourse.julialang.org/) - main community forum
- [Julia Slack](https://julialang.org/slack/) - real-time chat
- [JuliaCon](https://juliacon.org/) - annual conference with excellent talks

**Package Development:**
- [PkgTemplates.jl](https://github.com/JuliaCI/PkgTemplates.jl) - package templates with best practices
- [Julia Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/) - optimization guide
- [JuliaHub](https://juliahub.com/) - discover packages and documentation
