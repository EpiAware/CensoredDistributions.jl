# CensoredDistributions.jl

*Censored event tools for Distributions.jl*

## What is CensoredDistributions.jl?

`CensoredDistributions.jl` is a package for working with censored distributions. It extends the functionality of the [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) package to support additional censored distribution and utilities for working with them.  

## What can I do with CensoredDistributions.jl?

- Create distributions that are modified to account for primary event censoring, and within interval censoring.
- Discretise continuous distributions to create discrete distributions.
- Generate random samples from censored and discretised distributions
- Calculate the probability density function (PDF) and cumulative distribution function (CDF) of censored distributions.
- Calculate the PDF of discretised distributions.
- Calculate the mean, variance, and other moments of censored and discretised distributions.

## What packages work well with CensoredDistributions.jl?

- [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) provides the base functionality for working with distributions as well as tools for frequentist inference of distributions.
- [Turing.jl](https://github.com/TuringLang/Turing.jl) for Bayesian inference of censored distributions. `CensoredDistributions.jl` is designed (and tested) to work well with Turing.jl.

## Where to learn more

- Want to get started running code? Check out the [Getting Started Tutorials](@ref getting-started).
- Want to understand the API? Check out our [API Reference](@ref api-reference).
- Want to chat with someone about `CensoredDistributions`? Post on our [GitHub Discussions](https://github.com/epiaware/CensoredDistributions.jl/discussions).
- Want to contribute to `CensoredDistributions`? Check out our [Contributing guide](@ref contributing).
- Want to see our code? Check out our [GitHub Repository](https://github.com/epiaware/CensoredDistributions.jl/).

## Contributing

We welcome contributions and new contributors!
We particularly appreciate help on [identifying and identified issues](https://github.com/epiaware/CensoredDistributions.jl/issues).
Please check and add to the issues, and/or add a [pull request](https://github.com/epiaware/CensoredDistributions.jl/pulls) and see our [contributing guide](https://github.com/epiaware/.github/blob/main/CONTRIBUTING.md) for more information.

If you need a different underlying model for your work: `CensoredDistributions` provides a flexible framework for censored distributions in Julia, the language of the future.
The future the is now.

## Code of Conduct

Please note that the `CensoredDistributions` project is released with a [Contributor Code of Conduct](https://github.com/epiaware/.github/blob/main/CODE_OF_CONDUCT.md). By contributing to this project, you agree to abide by its terms.
