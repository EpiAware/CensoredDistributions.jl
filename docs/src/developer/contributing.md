# [Contributing](@id contributing)

This page details the guidelines that should be followed when contributing to CensoredDistributions.jl.

## Getting Started

Before contributing, please:
1. Read the [Getting Started with Julia](@ref julia) guide if you're new to Julia development
2. Check out the [developer documentation](@ref developer) for advanced workflows
3. Review the project structure and development commands below

## Project Structure

CensoredDistributions.jl uses multiple environments for different purposes:

```
CensoredDistributions.jl/
├── Project.toml           # Main package environment
├── test/
│   ├── Project.toml       # Test environment with test dependencies
│   ├── runtests.jl        # Main test entry point
│   ├── censoring/         # Feature-specific tests
│   └── package/           # Quality tests (Aqua, DocTest, etc.)
├── docs/
│   ├── Project.toml       # Documentation environment
│   ├── make.jl           # Documentation build script
│   ├── pages.jl          # Documentation structure
│   └── src/              # Documentation source files
└── benchmark/
    ├── Project.toml       # Benchmark environment
    └── runbenchmarks.jl   # Benchmark suite
```

## Development Commands

### Running Tests

```bash
# Full test suite (recommended for CI and final checks)
julia --project=. -e 'using Pkg; Pkg.test()'

# Run tests directly (faster during development)
julia --project=test test/runtests.jl

# Skip quality tests for faster development iteration
julia --project=test test/runtests.jl skip_quality
```

### Building Documentation

```bash
# Build complete documentation (includes Pluto notebook conversion)
julia --project=docs docs/make.jl
```

### Running Benchmarks

```bash
# Execute benchmark suite
julia --project=benchmark benchmark/runbenchmarks.jl
```

## Testing Strategy

### Test Organisation

- **Unit tests**: Located in `test/censoring/` for each distribution type
- **Integration tests**: Test interactions between components
- **Quality tests**: Located in `test/package/` including:
  - Aqua.jl for code quality
  - DocTest.jl for documentation examples
  - Code formatting and linting checks

### Test Environment

The test environment (`test/Project.toml`) includes:
- Test-specific dependencies (TestItemRunner.jl, Test.jl)
- The main package in development mode
- Quality assurance tools

Use `skip_quality` argument during development to bypass slow quality checks:
```bash
julia --project=test test/runtests.jl skip_quality
```

## Documentation

### Pluto Notebooks

The tutorials use [Pluto.jl](https://plutojl.org/) notebooks located in `docs/src/getting-started/tutorials/`.
These are converted to markdown during the documentation build.

#### Working with Pluto notebooks

1. **Start Pluto**: Use the provided script `docs/pluto-scripts.sh` or:
   ```bash
   julia --project=docs -e 'using Pluto; Pluto.run()'
   ```

2. **Environment setup**: Notebooks should use the docs environment and develop the local package:
   ```julia
   # In notebook setup cell
   let
       docs_dir = (dirname ∘ dirname ∘ dirname)(@__DIR__)
       using Pkg: Pkg
       Pkg.activate(docs_dir)
       Pkg.develop(PackageSpec(path=dirname(docs_dir)))
       Pkg.instantiate()
   end
   ```

3. **Adding new notebooks**:
   - Add the notebook file to `docs/src/getting-started/tutorials/`
   - Add build call in `docs/make.jl`
   - Add the generated `.md` file to `docs/pages.jl`

### Documentation Structure

- `docs/src/getting-started/`: User-facing documentation
- `docs/src/lib/`: API documentation (auto-generated)
- `docs/src/developer/`: Developer and contributor documentation

## Branches and Workflow

- **Feature branches**: Create feature branches for new development
- **Main branch**: Features are merged into `main` when ready
- **Releases**: Automatic releases are created when versions are tagged

## Style Guide

This project follows the [SciML style guide](https://github.com/SciML/SciMLStyle).

Key points:
- Use descriptive variable names
- Follow Julia naming conventions (snake_case for variables, CamelCase for types)
- Write docstrings for exported functions
- Keep lines under 80 characters where possible
- Use consistent indentation (4 spaces)

## Code Quality

### Pre-commit Checklist

Before submitting a pull request:

1. **Run full tests**:
   ```bash
   julia --project=. -e 'using Pkg; Pkg.test()'
   ```

2. **Build documentation**:
   ```bash
   julia --project=docs docs/make.jl
   ```

3. **Check formatting**: The project uses automatic formatting tools

### Quality Tools

The project includes several quality assurance tools:
- **Aqua.jl**: Checks for common package issues
- **JET.jl**: Static analysis for type stability (available in developer environment)
- **DocTest.jl**: Ensures documentation examples work

## Adding New Features

1. **Write tests first**: Add tests in appropriate `test/` subdirectory
2. **Implement feature**: Add implementation in `src/`
3. **Document feature**: Add docstrings and update documentation if needed
4. **Test thoroughly**: Run full test suite
5. **Update changelog**: Add entry describing the change

## Advanced Development Resources

For advanced Julia development techniques beyond this project:

- **[Julia Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/)**: Official performance optimization guide
- **[JET.jl](https://github.com/aviatesk/JET.jl)**: Static analysis for type stability and optimization
- **[ProfileView.jl](https://github.com/timholy/ProfileView.jl)**: Visual profiling for performance analysis
- **[PkgTemplates.jl](https://github.com/JuliaCI/PkgTemplates.jl)**: Best practices for Julia package structure

## Getting Help

- **Questions**: Open a GitHub discussion
- **Bugs**: File a GitHub issue with minimal reproducible example
- **Feature requests**: Open a GitHub issue with rationale and use case
- **General Julia help**: See [Julia Discourse](https://discourse.julialang.org/) or [Julia Slack](https://julialang.org/slack/)

Thank you for contributing to CensoredDistributions.jl!
