# CensoredDistributions.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://www.CensoredDistributions.epiaware.org/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://www.CensoredDistributions.epiaware.org/dev/)
[![Test](https://github.com/EpiAware/CensoredDistributions.jl/actions/workflows/test.yaml/badge.svg)](https://github.com/EpiAware/CensoredDistributions.jl/actions/workflows/test.yaml)
[![codecov](https://codecov.io/gh/EpiAware/CensoredDistributions.jl/graph/badge.svg)](https://codecov.io/gh/EpiAware/CensoredDistributions.jl)

[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![JET](https://img.shields.io/badge/%E2%9C%88%EF%B8%8F%20tested%20with%20-%20JET.jl%20-%20red)](https://github.com/aviatesk/JET.jl)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Additional censored event tools for Distributions.jl*

**Websites**: [Organization Website](https://www.epiaware.org/) | [Documentation](https://www.CensoredDistributions.epiaware.org/)

CensoredDistributions.jl Stats: ![CensoredDistributions Stars](https://img.shields.io/github/stars/EpiAware/CensoredDistributions.jl?style=social)

## What is CensoredDistributions.jl?

`CensoredDistributions.jl` is a package for working with censored distributions. It extends the functionality of the [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) package to support additional censored distribution and utilities for working with them.

## Why CensoredDistributions.jl?

- **Primary event censoring**: Model delay distributions where the initial event occurs within a time window (e.g., exposure periods in epidemiology).
- **Interval censoring**: Bin continuous distributions into discrete intervals (e.g., daily reporting) when exact values are unobserved.
- **Double interval censoring**: Combines both primary event and interval censoring for complex observation processes.
- **Distribution fitting**: Extends Distributions.jl's `fit` support with MLE fitting for primary censored and interval censored distributions (potentially truncated), plus Turing.jl integration for Bayesian inference.
- **Analytical solutions**: Provides analytical solutions where possible with numerical fallbacks for efficiency.

## What can I do with CensoredDistributions.jl?

- Create distributions that are modified to account for primary event censoring and interval censoring.
- Apply interval censoring to continuous distributions (both regular and arbitrary intervals).
- Generate random samples from censored distributions.
- Calculate the probability density function (PDF) and cumulative distribution function (CDF) of censored event distributions.
- Calculate the PDF of interval-censored distributions.
- Calculate the mean, variance, and other moments of censored event distributions.
- Fit censored event distributions using MLE methods and Bayesian inference with Turing.jl.

## Getting Started

The following example demonstrates how to create a double interval censored distribution:

```julia
using CensoredDistributions, Distributions, Plots

# Create a censored distribution accounting for primary and secondary censoring
original = Gamma(2, 3)
censored = double_interval_censored(original, Uniform(0, 1); upper = 15, interval = 1)

# Compare the distributions
x = 0:0.01:20
plot(x, pdf.(original, x), label = "Original Gamma", lw = 2)
plot!(x, pdf.(censored, x), label="Double Censored and right truncated", lw = 2)
```

## What packages work well with CensoredDistributions.jl?

- [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) provides the base functionality for working with distributions as well as tools for frequentist inference of distributions.
- [Turing.jl](https://github.com/TuringLang/Turing.jl) for Bayesian inference of censored distributions. `CensoredDistributions.jl` is designed (and tested) to work well with Turing.jl.

## Where to learn more

- Want to get started running code? Check out the [Getting Started Tutorials](https://www.CensoredDistributions.epiaware.org/getting-started/).
- Want to understand the API? Check out our [API Reference](https://www.CensoredDistributions.epiaware.org/lib/public/).
- Want to chat with someone about `CensoredDistributions`? Post on our [GitHub Discussions](https://github.com/EpiAware/CensoredDistributions.jl/discussions).
- Want to contribute to `CensoredDistributions`? Check out our [Developer Documentation](https://www.CensoredDistributions.epiaware.org/dev/developer/).
- Want to see our code? Check out our [GitHub Repository](https://github.com/EpiAware/CensoredDistributions.jl/).

## Contributing

We welcome contributions and new contributors!
We particularly appreciate help on [identifying and identified issues](https://github.com/EpiAware/CensoredDistributions.jl/issues).
Please check and add to the issues, and/or add a [pull request](https://github.com/EpiAware/CensoredDistributions.jl/pulls) and see our [developer documentation](https://www.CensoredDistributions.epiaware.org/dev/developer/) for more information.

## Code of Conduct

Please note that the `CensoredDistributions` project is released with a [Contributor Code of Conduct](https://github.com/EpiAware/.github/blob/main/CODE_OF_CONDUCT.md). By contributing to this project, you agree to abide by its terms.
