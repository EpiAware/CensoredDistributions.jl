# [Getting started](@id getting-started)

*Note that this section of the documentation is still under construction. Please see replications for the most up-to-date information. Please feel free to contribute to the documentation by submitting a pull request.*

Welcome to the `CensoredDistributions` documentation! This section is designed to help you get started with the package. It includes a frequently asked questions (FAQ) section, a series of explainers that provide a detailed overview of the platform and its features, and tutorials that will help you get started with `CensoredDistributions` for specific tasks. See the sidebar for the list of topics.

## Introduction to the problem

Delay distributions play a crucial role in various fields, including epidemiology, reliability analysis, and survival analysis.

## Loading the packages

```
# Import the package
using CensoredDistributions
using Distributions
using Random
using Plots

# Set the seed for reproducibility
Random.seed!(123)

```
Now we can set up the problem as described [here](https://primarycensored.epinowcast.org/dev/articles/primarycensored.html).

```
n = 1e4
meanlog = 1.5
sdlog = 0.75
obs_time = 10
pwindow = 1

# generate the distributions

primary_distribution = Uniform(1, 2)
delay_distribution = LogNormal(meanlog, sdlog)

```
Now we can generate our `CensoredDistributions` object

```

prim_dist = primarycensored(primary_distribution, delay_distribution)

```

We can then apply the truncation.

```

trunc_prime_dist = truncated(prim_dist, 1, 2)

```

And then we can apply our within interval censoring approach:

```
int_censored_dist = within_interval_censored(trunc_prime_dist, 2, 4)

hist(rand(int_censored_dist, 1000))

```

## Contributing

We welcome contributions and new contributors!
We particularly appreciate help on [identifying and identified issues](https://github.com/epiaware/CensoredDistributions.jl/issues).
Please check and add to the issues, and/or add a [pull request](https://github.com/epiaware/CensoredDistributions.jl/pulls) and see our [contributing guide](https://github.com/epiaware/.github/blob/main/CONTRIBUTING.md) for more information.

If you need a different underlying model for your work: `CensoredDistributions` provides a flexible framework for censored distributions in Julia, the language of the future.
The future the is now.


## Code of Conduct

Please note that the `CensoredDistributions` project is released with a [Contributor Code of Conduct](https://github.com/epiaware/.github/blob/main/CODE_OF_CONDUCT.md). By contributing to this project, you agree to abide by its terms.
