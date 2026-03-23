# [Contributing](@id contributing)

This page details the guidelines that should be followed when contributing to CensoredDistributions.jl.

## Getting started

Before contributing, please:
1. Read the [Getting Started with Julia](@ref julia) guide if you're new to Julia development
2. **Install Task** for streamlined development workflows:
   - **macOS**: `brew install go-task/tap/go-task`
   - **Linux**: Download from [releases](https://github.com/go-task/task/releases) or use package manager
   - **Windows**: `winget install Task.Task` or download from releases
3. Check out the [developer documentation](@ref developer) for advanced workflows
4. Review the project structure and development commands below

## Project structure

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

## Development commands

This project includes a Taskfile for streamlined development workflows.

### Quick start with tasks

```bash
# Discover all available tasks
task --list

# Common development workflow
task setup     # One-time environment setup
task dev       # Fast tests + documentation
task precommit # Pre-commit validation

# Individual workflows
task test-fast    # Quick testing
task docs         # Full docs including Literate tutorials
task benchmark    # Run benchmarks
```

### Detailed commands

For advanced usage or when tasks don't cover specific needs, use the underlying Julia commands:

```bash
# Full test suite (recommended for CI and final checks)
julia --project=. -e 'using Pkg; Pkg.test()'

# Run tests directly (faster during development)
julia --project=test test/runtests.jl

# Skip quality tests for faster development iteration
julia --project=test test/runtests.jl skip_quality

# Build complete documentation (includes Literate tutorial processing)
julia --project=docs docs/make.jl

# Execute benchmark suite
julia --project=benchmark benchmark/runbenchmarks.jl
```

## Testing strategy

### Test organisation

- **Unit tests**: Located in `test/censoring/` for each distribution type
- **Integration tests**: Test interactions between components
- **Quality tests**: Located in `test/package/` including:
  - Aqua.jl for code quality
  - DocTest.jl for documentation examples
  - Code formatting and linting checks

### Test environment

The test environment (`test/Project.toml`) includes:
- Test-specific dependencies (TestItemRunner.jl, Test.jl)
- The main package in development mode
- Quality assurance tools

Use `skip_quality` argument during development to bypass slow quality checks:
```bash
julia --project=test test/runtests.jl skip_quality
```

## Documentation

### Literate.jl tutorials

The tutorials use [Literate.jl](https://fredrikekre.github.io/Literate.jl/) scripts located in `docs/src/getting-started/tutorials/`.
These are converted to markdown during the documentation build.

#### Working with tutorials

Tutorials are plain Julia `.jl` files using `md"""..."""` blocks for markdown.
You can run them directly in the REPL or as scripts.

1. **Adding new tutorials**:
   - Create a `.jl` file in `docs/src/getting-started/tutorials/`
   - Add a `Literate.markdown()` call in `docs/make.jl`
   - Add the generated `.md` file to `docs/pages.jl`

2. **Tutorial format**:
   ```julia
   md"""
   # Tutorial title

   Introduction text.
   """

   using SomePackage

   md"""
   ## Section
   """

   code_here()
   ```

### Documentation structure

- `docs/src/getting-started/`: User-facing documentation
- `docs/src/lib/`: API documentation (auto-generated)
- `docs/src/developer/`: Developer and contributor documentation

## Branches and workflow

- **Feature branches**: Create feature branches for new development
- **Main branch**: Features are merged into `main` when ready
- **Releases**: Automatic releases are created when versions are tagged

## Style guide

This project follows the [SciML style guide](https://github.com/SciML/SciMLStyle).

Key points:
- Use descriptive variable names
- Follow Julia naming conventions (snake_case for variables, CamelCase for types)
- Write docstrings for exported functions
- Keep lines under 80 characters where possible
- Use consistent indentation (4 spaces)

### Documentation standards

All docstrings use the DocStringExtensions.jl template system defined in `src/docstrings.jl`:

**Functions**: Use `$(TYPEDSIGNATURES)` for automatic signature generation:
```julia
@doc "
$(TYPEDSIGNATURES)

Brief description of the function.

# Arguments
- `param1`: Description (no type annotations needed)
- `param2`: Description

# Keyword Arguments
- `kwarg1`: Description
"
function my_function(param1, param2; kwarg1=default)
    # implementation
end
```

**Structs**: Use `$(TYPEDEF)` and `$(TYPEDFIELDS)` with inline field documentation:
```julia
@doc "
$(TYPEDEF)

Description of the struct.

$(TYPEDFIELDS)
"
struct MyStruct
    "Description of field1"
    field1::Type1
    "Description of field2"
    field2::Type2
end
```

**Key rules**:
- **Never use `@doc raw"`** - it bypasses the template system
- **Don't repeat type information** in argument descriptions since `$(TYPEDSIGNATURES)` shows them
- **Use `@doc "` (not `@doc """`)** to allow macro expansion
- **Document argument purpose**, not types

## Code quality

### Pre-commit checklist

Before submitting a pull request:

1. **Run pre-commit checks** (recommended):
   ```bash
   task precommit
   ```

2. **Or run individual checks**:
   ```bash
   task test       # Full test suite
   task docs-fast  # Build documentation
   ```

### Quality tools

The project includes several quality assurance tools:
- **Aqua.jl**: Checks for common package issues
- **JET.jl**: Static analysis for type stability (available in developer environment)
- **DocTest.jl**: Ensures documentation examples work

## Adding new features

1. **Write tests first**: Add tests in appropriate `test/` subdirectory
2. **Implement feature**: Add implementation in `src/`
3. **Document feature**: Add docstrings and update documentation if needed
4. **Test thoroughly**: Run full test suite
5. **Update changelog**: Add entry describing the change

## Advanced development resources

For advanced Julia development techniques beyond this project:

- **[Julia Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/)**: Official performance optimization guide
- **[JET.jl](https://github.com/aviatesk/JET.jl)**: Static analysis for type stability and optimization
- **[ProfileView.jl](https://github.com/timholy/ProfileView.jl)**: Visual profiling for performance analysis
- **[PkgTemplates.jl](https://github.com/JuliaCI/PkgTemplates.jl)**: Best practices for Julia package structure

## Getting help

- **Questions**: Open a GitHub discussion
- **Bugs**: File a GitHub issue with minimal reproducible example
- **Feature requests**: Open a GitHub issue with rationale and use case
- **General Julia help**: See [Julia Discourse](https://discourse.julialang.org/) or [Julia Slack](https://julialang.org/slack/)

Thank you for contributing to CensoredDistributions.jl!
