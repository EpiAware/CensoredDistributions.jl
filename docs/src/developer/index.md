# [Developer Documentation](@id developer)

Welcome to the CensoredDistributions.jl developer documentation.
This section contains guides and resources for contributors and maintainers.

## Contributing to CensoredDistributions.jl

New to contributing? Start here:

- **[Contributing Guide](@ref contributing)**: How to contribute code, documentation, and issues
- **[Release Process](@ref release-process)**: How we manage releases and versioning

For general Julia development setup, see [Getting Started with Julia](@ref julia) in the Getting Started section.

## Frequently Asked Questions

### How do I run tests faster during development?

You can skip the quality tests (which include Aqua.jl, DocTest, and other checks) by using:

```bash
julia --project=test test/runtests.jl skip_quality
```

This is useful during active development when you want quick feedback on your changes without waiting for the full quality suite.

---

*For users looking to get started with the package, see the [Getting Started](@ref getting-started) section.*
