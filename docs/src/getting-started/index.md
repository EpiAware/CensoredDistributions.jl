# [Getting started](@id getting-started)


Welcome to the `CensoredDistributions` documentation! This section is designed to help you get started with the package. It includes a quickstart guide, frequently asked questions (FAQ) section, and tutorials that will help you get started with `CensoredDistributions` for specific tasks. See the sidebar for the list of topics.

## Quickstart

Delay distributions play a crucial role in various fields, including epidemiology, reliability analysis, and survival analysis. In epidemiology, delay distributions are used to model the time between the primary event (e.g. infection) and the secondary event (e.g. onset of symptoms). Both events are often subject to interval censoring, and the delay distribution is often truncated to the time of the secondary event. One of the main functions of `CensoredDistributions.jl` is to provide a flexible framework for working with distributions where one or both (this is commonly called double-censoring) of the events are subject to interval censoring.

## Loading the packages

```julia
# Import the package
using CensoredDistributions
using Distributions
using Random

# Set the seed for reproducibility
Random.seed!(123)
```

## Primary event censoring

For the primary event, we'll use a uniform distribution between 1 and 2.


```julia
primary_event = Uniform(0, 1)
```

For the delay distribution, we'll use a lognormal distribution with a mean of 1.5 and a standard deviation of 0.75.

```julia
dist = LogNormal(1.5, 0.75)
```

Now we combine these two distributions to create a primary censored distribution.

```julia
prim_dist = primarycensored(dist, primary_event)
```

For theory explained in more detail, see the [primarycensored](https://primarycensored.epinowcast.org/dev/articles/primarycensored.html) documentation.

We can now generate a random sample from the primary distribution or calculate the probability density function (PDF) and cumulative distribution function (CDF).

```julia
rand(prim_dist, 10)
```

and plot the CDF compared to the unmodified distribution.

```julia
x = 0:0.01:15
plot(x, cdf.(dist, x), label="Uncensored")
plot!(x, cdf.(prim_dist, x), label="Primary censored")
```

## Truncation

We can then apply the truncation using the normal `truncated` function from `Distributions.jl`.

```julia    
trunc_prim_dist = truncated(prim_dist, upper= 10)
```

We can again sample from the distribution.

```julia
rand(trunc_prim_dist, 10)
```

or plot the CDFs of the different distributions.

```julia
x = 0:0.01:15
plot(x, cdf.(dist, x), label="Uncensored")
plot!(x, cdf.(prim_dist, x), label="Primary censored")
plot!(x, cdf.(trunc_prim_dist, x), label="Truncated and primary censored")
```

## Secondary interval censoring

We can now apply secondary interval censoring using the `discretise` function. We call this discretisation as rather than specifying an interval for the secondary event, we specify intervals to round to.

```julia
int_censored_dist = discretise(trunc_prim_dist, 1)
```

Again we can sample from the distribution.

```julia
rand(int_censored_dist, 10)
```

or plot the CDFs of the different distributions.

```julia
x = 0:0.01:15
plot(x, cdf.(dist, x), label="Uncensored")
plot!(x, cdf.(prim_dist, x), label="Primary censored")
plot!(x, cdf.(trunc_prim_dist, x), label="Truncated and primary censored")
plot!(x, cdf.(int_censored_dist, x), label="Truncated, primary censored, and discretised")
```

## Learning more

For more information on the package and its integration with other packages, see the [tutorials](@ref tutorials).