# [Getting Started with Julia](@id julia)

For installing Julia, setting up an editor, working with Julia environments, a
productive REPL, and common setup problems, see
[Using Julia](https://epiaware.org/using-julia) on the EpiAware site.
That guide is written once for the whole ecosystem, so it is not repeated
here.

This page covers only what is specific to CensoredDistributions.jl.

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

The [tutorials](@ref tutorials) are Literate.jl scripts that can be run
directly in the REPL or as standalone Julia scripts.
See the [FAQ](@ref faq) for instructions on running them.

## Next steps

- Work through the [tutorials](@ref tutorials) to learn
  CensoredDistributions.jl.
- Explore the [API documentation](@ref "Public API").
- Ready to contribute? The [Contributing guide](@ref contributing) and
  [Developer FAQ](@ref developer-faq) cover the development workflow —
  running tests, switching environments, and troubleshooting — with the
  package's actual `task`-based commands, not the generic advice above.
