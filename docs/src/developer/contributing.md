# [Contributing](@id contributing)

This page details the some of the guidelines that should be followed when contributing to this package. It is adapted from `Documenter.jl`.

## Branches

Are used for developing features. Once a feature is ready to be released, it is merged into the `main` branch and then a new release is automatically created.

## Style Guide

Follow the style of the surrounding text when making changes. When adding new features please try to stick to the following points whenever applicable. This project follows the
[SciML style guide](https://github.com/SciML/SciMLStyle).

## Tests

### Unit tests

As is conventional for Julia packages, unit tests are located at `test/*.jl` with the entrypoint
`test/runtests.jl`.

### End to end testing

Tests that build example package docs from source and inspect the results (end to end tests) are
located in `/test/examples`. The main entry points are `test/examples/make.jl` for building and
`test/examples/test.jl` for doing some basic checks on the generated outputs.

## Pluto usage in showcase documentation

Some of the showcase examples in `PrimaryCensored/docs/src/showcase` use [`Pluto.jl`](https://plutojl.org/) notebooks for the underlying computation. The output of the notebooks is rendered into HTML for inclusion in the documentation in two steps:
1. [`PlutoStaticHTML.jl`](https://github.com/rikhuijzer/PlutoStaticHTML.jl) converts the notebook with output into a machine-readable `.md` format.
2. [`Documenter.jl`](https://github.com/JuliaDocs/Documenter.jl) renders the `.md` file into HTML for inclusion in the documentation during the build process.

For other examples of using `Pluto` to generate documentation see the examples shown [here](https://plutostatichtml.huijzer.xyz/stable/#Documenter.jl).

### Running Pluto notebooks from `PrimaryCensored` locally

To run the `Pluto.jl` scripts in the `PrimaryCensored` documentation directly from the source code you can do these steps:

1. Install [`Pluto.jl`](https://plutojl.org/) locally. We recommend using the version of `Pluto` that is pinned in the `Project.toml` file defining the documentation environment.
2. Clone the `PrimaryCensored` repository.
3. Start `Pluto.jl` either from REPL (see the `Pluto.jl` documentation) or from the command line with the shell script `PrimaryCensored/docs/pluto-scripts.sh`.
4. From the `Pluto.jl` interface, navigate to the `Pluto.jl` script you want to run.

### Contributing to Pluto notebooks in `PrimaryCensored` documentation

#### Modifying an existing Pluto notebook
Committing changes to the `Pluto.jl` notebooks in the `EpiAware` documentation is the same as committing changes to any other part of the repository. However, please note that we expect the following features for the environment management of the notebooks:

1. Use the environment determined by the `Project.toml` file in the `PrimaryCensored/docs` directory. If you want extra packages, add them to this environment.
2. Use the version of `PrimaryCensored` that is used in these notebooks to be the version of `PrimaryCensored` on the branch being pull requested into `main`. To do this use the `Pkg.develop` function.

To do this you can use the following code snippet in the Pluto notebook:

```julia
# Determine the relative path to the `EpiAware/docs` directory
docs_dir = dirname(dirname(dirname(dirname(@__DIR__))))
# Determine the relative path to the `EpiAware` package directory
pkg_dir = dirname(docs_dir)

using Pkg: Pkg
Pkg.activate(docs_dir)
Pkg.develop(; path = pkg_dir)
Pkg.instantiate()
```

#### Adding a new Pluto notebook
Adding a new `Pluto.jl` notebook to the `PrimaryCensored` documentation is the same as adding any other file to the repository. However, in addition to following the guidelines for modifying an [existing notebook](#modifying-an-existing-pluto-notebook), please note that the new notebook is added to the set of notebook builds using `build` in the `PrimaryCensored/docs/make.jl` file. This will generate an `.md` of the same name as the notebook which can be rendered when `makedocs` is run. For this document to be added to the overall documentation the path to the `.md` file must be added to the `Pages` array defined in `PrimaryCensored/docs/pages.jl`.
