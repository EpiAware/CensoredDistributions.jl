# [Developer FAQ](@id developer-faq)

This page contains frequently asked questions for developers and contributors to CensoredDistributions.jl.

## Development Environment

### Q: My code changes aren't reflecting when developing

**A:** Install and use Revise.jl for automatic code reloading:

```julia
using Pkg
Pkg.add("Revise")          # Install once
using Revise               # Load before your package
using CensoredDistributions # Now changes reload automatically
```

Better yet, add Revise to your `startup.jl` file as described in the [Julia setup guide](@ref julia).

### Q: I get "Package not found" errors during development

**A:** Make sure you're in the right environment:

```julia
using Pkg
Pkg.activate(".")           # Activate current directory
Pkg.instantiate()           # Install dependencies
Pkg.develop(PackageSpec(path="."))  # Add local package in dev mode
```

## Testing

### Q: Tests are failing or taking too long

**A:** For development, you can skip quality tests:

```bash
julia --project=test test/runtests.jl skip_quality
```

This runs core functionality tests without slow formatting/linting checks.

### Q: How do I run specific tests?

**A:** Use TestItemRunner for targeted testing:

```julia
using TestItemRunner
# Run tests for specific components
run_tests("test/censoring/")
# Run tests with specific tags
run_tests(filter=ti->(:unit in ti.tags))
```

### Q: How do I add new tests?

**A:** Create tests using the `@testitem` macro:

```julia
@testitem "My new feature" tags=[:unit] begin
    using CensoredDistributions, Distributions

    # Test your feature
    result = my_new_function()
    @test result isa SomeType
end
```

## Documentation

### Q: How do I build the documentation locally?

**A:** Use the documentation environment:

```bash
# Full build (includes Pluto notebook processing - slow)
julia --project=docs docs/make.jl

# Fast build for development (skips notebook processing)
julia --project=docs docs/make.jl --skip-notebooks

# Alternative: use environment variable
SKIP_NOTEBOOKS=true julia --project=docs docs/make.jl
```

The `--skip-notebooks` option is useful during development for quick documentation checks without waiting for slow Pluto notebook processing.

### Q: How do I add a new Pluto notebook tutorial?

**A:**
1. Create the `.jl` notebook in `docs/src/getting-started/tutorials/`
2. Add a `build("getting-started/tutorials")` call in `docs/make.jl` if not already present
3. Add the generated `.md` file path to `docs/pages.jl`

### Q: How do I update docstrings?

**A:** We use DocStringExtensions.jl for automatic documentation generation. The approach depends on content:

For docstrings with DocStringExtensions macros only:
```julia
@doc "
$(TYPEDSIGNATURES)

Compute the square of `x`.

# See also
- [`sqrt`](@ref): Inverse operation
"
function my_function(x::Real)
    return x^2
end
```

For docstrings with both macros and LaTeX math:
```julia
@doc """
$(TYPEDSIGNATURES)

Compute the function ``f(x) = x^2``.

# Mathematical formulation
The function computes: ``f(x) = x^2``
"""
function my_function(x::Real)
    return x^2
end
```

**Important:** Never use `@doc raw"` with DocStringExtensions macros as it prevents macro expansion.

## Code Quality

### Q: How do I run code quality checks?

**A:** Quality tests are included in the main test suite:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Individual quality tests can be found in `test/package/`.

### Q: My code doesn't pass formatting checks

**A:** The project uses automatic formatting. Check the pre-commit hooks or run formatting tools as specified in the contributing guide.

### Q: How do I check for type stability?

**A:** Use JET.jl for static analysis:

```julia
using JET
# Check specific function
@report_opt primary_censored(Gamma(2, 1), Uniform(0, 1))
# Check entire package
@report_package CensoredDistributions
```

## Performance

### Q: How do I benchmark my changes?

**A:** Use BenchmarkTools.jl:

```julia
using BenchmarkTools
pc = primary_censored(Gamma(2, 1), Uniform(0, 1))
@benchmark cdf(pc, 5.0)
```

For comprehensive benchmarking:

```bash
julia --project=benchmark benchmark/runbenchmarks.jl
```

### Q: How do I add analytical solutions for new distribution pairs?

**A:** See the Analytical CDF Solutions tutorial in the [tutorials](@ref tutorials) section for implementation patterns. Add methods to the appropriate files in `src/censoring/` and ensure they're properly tested.

## Contributing

### Q: How can I contribute to the package?

**A:** See our [Contributing Guide](@ref contributing) for details on setting up the development environment, running tests, code style guidelines, and submitting pull requests.

### Q: I found a bug or have a feature request

**A:**
- **Bugs**: File a GitHub issue with a minimal reproducible example
- **Feature requests**: Open a GitHub issue with rationale and use case
- **Questions**: Use GitHub Discussions for broader questions

### Q: How do I submit a pull request?

**A:**
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the full test suite: `julia --project=. -e 'using Pkg; Pkg.test()'`
5. Build documentation: `julia --project=docs docs/make.jl`
6. Submit a pull request with a clear description

### Q: What should I include in my pull request?

**A:**
- Clear description of changes and motivation
- Tests for new functionality
- Documentation updates if needed
- Ensure all CI checks pass

## Troubleshooting

### Q: The documentation build is failing

**A:** Common issues:
- Check that all cross-references are valid
- Ensure Pluto notebooks run without errors
- Verify all `@example` blocks execute successfully
- Check for missing dependencies in `docs/Project.toml`

### Q: Pluto notebooks won't start

**A:**
- Ensure you're using the docs environment: `julia --project=docs`
- Check that Pluto is installed: `] add Pluto`
- Try the shell script: `docs/pluto-scripts.sh`

### Q: I'm getting precompilation errors

**A:**
- Clear compiled cache: `julia -e 'using Pkg; Pkg.precompile()'`
- Reset environments: remove `Manifest.toml` and run `] instantiate`
- Check for version conflicts: `] resolve`

## Getting Help

For development-specific questions:

- **Code issues**: Open a [GitHub Discussion](https://github.com/EpiAware/CensoredDistributions.jl/discussions)
- **Bug reports**: [GitHub Issues](https://github.com/EpiAware/CensoredDistributions.jl/issues)
- **General Julia development**: [Julia Discourse](https://discourse.julialang.org/)
