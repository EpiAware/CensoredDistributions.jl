# [Getting started](@id getting-started)

Welcome to the `CensoredDistributions` documentation! This section is designed to help you get started with the package. It includes a quickstart guide, frequently asked questions (FAQ) section, and tutorials that will help you get started with `CensoredDistributions` for specific tasks. See the sidebar for the list of topics.

# Introduction

Delay distributions play a crucial role in various fields, including epidemiology, reliability analysis, and survival analysis. These distributions describe the time between two events of interest, such as the incubation period of a disease or the time to failure of a component.
Accurately estimating and calculating these distributions is essential for understanding the underlying processes and making informed decisions.

The estimation of delay distributions often faces the following challenges:

- **Primary event within interval censoring**: The primary event (e.g., exposure to a pathogen or the start of a process) is often observed with some degree of interval censoring. This means that the exact time of the event is not known, but rather, it is known to have occurred within a certain time interval, commonly a day.
As a result, any distribution based on these primary events is a combination of the underlying true distribution and the censoring distribution.

- **Truncation**: The observation of delay distributions is often conditioned on the occurrence of the secondary event. This leads to a truncation of the observed distribution, as delays longer than the observation time are not captured in the data.
Consequently, the observed distribution is a combination of the underlying true distribution, the censoring distribution, and the observation time.

- **Secondary event within interval censoring**: The secondary event (e.g., symptom onset or the end of a process) is also frequently observed with within an interval.
This additional layer of censoring further complicates the estimation of the delay distribution.

- **Double event within interval censoring** Both the primary and secondary events are censored so that we know they occurred in an interval but not precisely when.

The `CensoredDistributions.jl` package aims to address these challenges by providing tools to manipulate primary censored delay distributions and to extend these distributions to account for both truncation and secondary event censoring. By accounting for the censoring and truncation present in the data, the package enables more accurate estimation and use of the underlying true distribution.

In this quickstart, we will provide a quick introduction to the main functions and concepts in the `CensoredDistributions.jl` package.
We will cover the mathematical formulation of the problem, demonstrate the usage of the key functions, and provide signposting on how to learn more.

## Packages used in this getting started guide

## Loading the packages

```@example getting-started
# Import the package
using CensoredDistributions
using Distributions
using Random
using Plots

# Set the seed for reproducibility
Random.seed!(123)
```

## Primary event censoring

The mathematical formulation for primary event censoring involves several key steps:

1. **Primary event times** ($p$) are generated from a specified primary event distribution, in this case uniform between 0 and 1:

```@example getting-started
primary_event = Uniform(0, 1)
```

2. **Delays** ($d$) are generated from a specified delay distribution, here a log-normal:

```@example getting-started
dist = LogNormal(1.5, 0.75)
```

This corresponds to: $d \sim \text{LogNormal}(1.5, 0.75), \quad d \geq 0$

```@example getting-started
x = 0:0.01:15
plot(x, pdf.(dist, x))
```

3. **Total delays** ($t$) are calculated by adding the primary event times and delays: $t = p + d$

Now we combine these two distributions to create a primary censored distribution:

```@example getting-started
prim_dist = primary_censored(dist, primary_event)
```

The primary event censored cumulative distribution function (CDF) is given by:

$$F_{\text{cens}}(q) = \int_{0}^{1} F(q - p) \cdot f_{\text{primary}}(p) \, dp$$

where $F$ is the CDF of the delay distribution and $f_{\text{primary}}$ is the PDF of the primary event times.

For theory explained in more detail, see the [primary_censored](https://primary_censored.epinowcast.org/dev/articles/primary_censored.html) documentation.

We can now generate a random sample from the primary distribution

```@example getting-started
Random.seed!(123)
rand(prim_dist, 10)
```

and plot the CDF compared to the unmodified distribution.

```@example getting-started
x = 0:0.01:15
plot(x, cdf.(dist, x), label="Uncensored")
plot!(x, cdf.(prim_dist, x), label="Primary censored")
```

## Truncation

Truncation is applied to ensure delays are within the specified range $[0, D]$, where $D$ is the maximum observable delay.
This step filters out delays longer than the observation time:

$$t_{\text{truncated}} = \{t \mid 0 \leq t < D\}$$

If the maximum delay $D$ is finite, the CDF is normalised by dividing by $F_{\text{cens}}(D)$:

$$F_{\text{cens,norm}}(q) = \frac{F_{\text{cens}}(q)}{F_{\text{cens}}(D)}$$

We can apply truncation using the normal `truncated` function from `Distributions.jl`:

```@example getting-started
trunc_prim_dist = truncated(prim_dist, upper=10)
```

We can again sample from the distribution

```@example getting-started
Random.seed!(123)
rand(trunc_prim_dist, 10)
```

or plot the CDFs of the different distributions:

```@example getting-started
x = 0:0.01:15
plot(x, cdf.(dist, x), label="Uncensored")
plot!(x, cdf.(prim_dist, x), label="Primary censored")
plot!(x, cdf.(trunc_prim_dist, x), label="Truncated and primary censored")
```

## Secondary interval censoring

We can now apply secondary interval censoring using the `interval_censored` function. This censors observations to fall within specified intervals.

The secondary event censoring process rounds the truncated delays to the nearest secondary event window ($\text{swindow}$):

$$t_{\text{valid}} = \lfloor \frac{t_{\text{truncated}}}{\text{swindow}} \rfloor \times \text{swindow}$$

For discrete data or when working with specific time windows (such as daily reporting delays), this step is particularly important. The primary event censored probability mass function (PMF), $f_{\text{cens}}(d)$, accounts for this secondary event censoring and is given by:

$$f_{\text{cens}}(d) = F_{\text{cens}}(d + \text{swindow}) - F_{\text{cens}}(d)$$

where $F_{\text{cens}}$ is the potentially right truncated primary event censored CDF and $\text{swindow}$ is the secondary event window.

```@example getting-started
int_censored_dist = interval_censored(trunc_prim_dist, 1)
```

Again we can sample from the distribution.

```@example getting-started
Random.seed!(123)
rand(int_censored_dist, 10)
```

or plot the CDFs of the different distributions.

```@example getting-started
x = 0:0.01:15
plot(x, cdf.(dist, x), label="Uncensored")
plot!(x, cdf.(prim_dist, x), label="Primary censored")
plot!(x, cdf.(trunc_prim_dist, x), label="Truncated and primary censored")
plot!(x, cdf.(int_censored_dist, x), label="Truncated, primary censored, and interval censored")
```

Neither the primary censored nor the interval censored distributions match the true distribution due to the censoring effects and truncation at the maximum observable delay, which biases both observed distributions towards shorter delays.

## Convenience Function: `double_interval_censored`

For common workflows involving the complete pipeline of primary censoring, truncation, and secondary interval censoring, the package provides a convenient `double_interval_censored` function that applies all transformations in the correct order (primary censoring → truncation → interval censoring):

```@example getting-started
# This is equivalent to the step-by-step approach above
double_censored_dist = double_interval_censored(Gamma(2, 1); upper=8, interval=2)
```

As with all the other functions, we can sample from the distribution

```@example getting-started
Random.seed!(123)
rand(double_censored_dist, 10)
```

or do any of the other common distribution operations.

## Key Package Features

In addition to these main functions, the package also includes:

- **Distributions.jl integration:** Full compatibility with the Distributions.jl ecosystem, supporting all standard distribution methods (`pdf`, `cdf`, `quantile`, `rand`, etc.).

- **Analytical solutions:** For common combinations of primary event and delay distributions (e.g., uniform primary events with gamma, lognormal, or Weibull delays), analytical solutions provide significant computational speedups compared to numerical integration.

- **Automatic differentiation compatibility:** Full support for automatic differentiation backends including ForwardDiff.jl, ReverseDiff.jl, Mooncake.jl, and Enzyme.jl for use in probabilistic programming and optimisation.

- **Type stability:** Efficient implementation with type-stable operations for high-performance computation.

## Learning more

### [Tutorials](@id tutorials)

For more information on the package and its integration with other packages, see the tutorials in this getting started section:

- **[Analytical CDF Solutions](tutorials/analytical-primarycensored-cdfs.md)**: Understanding analytical solutions for common distribution pairs
- **[Fitting with Turing.jl](tutorials/fitting-with-turing.md)**: Bayesian inference with censored distributions

### Methodological Background

For methodological background on delay distributions and censoring methods, see:

- **Park et al. (2024)**: ["Estimating epidemiological delay distributions for infectious diseases"](https://doi.org/10.1101/2024.01.12.24301247) - Provides detailed mathematical foundations and methodological guidance for delay distribution estimation.

- **Charniga et al. (2024)**: ["Best practices for estimating and reporting epidemiological delay distributions of infectious diseases using public health surveillance and healthcare data"](https://doi.org/10.48550/arXiv.2405.08841) - Offers best practices and practical advice when estimating delay distributions from real-world data.

- The [primarycensored R package documentation](https://primarycensored.epinowcast.org/dev/articles/primarycensored.html) provides additional theoretical context and examples.
