### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ bb9c75db-6638-48fe-afcb-e78c4bcc057d
begin
    let
        docs_dir = dirname(dirname(@__DIR__))
        println(docs_dir)
        pkg_dir = dirname(docs_dir)
        println(pkg_dir)
        using Pkg: Pkg
        Pkg.activate(docs_dir)
        Pkg.develop(; path = pkg_dir)
        Pkg.instantiate()
    end
end

# ╔═╡ 3690c122-d630-4fd0-aaf2-aea9226df086
begin
    using DataFrames
    using Turing
    using Distributions
    using Random
    using CairoMakie
    using StatsBase
    using PrimaryCensored
end

# ╔═╡ 30511a27-984e-40b7-9b1e-34bc87cb8d56
md"""
# Fitting distributions using PrimaryCensored.jl and Turing.jl

## Introduction

### What are we going to do in this exercise

We'll demonstrate how to use `PrimaryCensored.jl` in conjunction with Turing.jl for Bayesian inference of epidemiological delay distributions. We'll cover the following key points:

1. Simulating censored delay distribution data
2. Fitting a naive model using Turing.jl
3. Evaluating the naive model's performance
4. Fitting an improved model using `PrimaryCensored.jl` functionality
5. Comparing the `PrimaryCensored.jl` model's performance to the naive model

## What might I need to know before starting

This vignette builds on the concepts introduced in the [Getting Started with PrimaryCensored.jl](FIXME.html) vignette and assumes familiarity with using Turing.jl.

## Packages used
We use CairoMakie for plotting, Turing for probabilistic programming, and the _classics_ DataFrames, Random, StatsBase (for the ecdf function). We install the PrimaryCensored.jl packages from the github repo.
"""

# ╔═╡ 80ce4590-e51a-457c-8497-d16916688656

# ╔═╡ 320f081c-b26e-495a-9887-56c4212d1ed7

# ╔═╡ 960b18ef-ab1d-417f-853e-d05d2cddc78f
# FIXME: PrimaryCensored.jl installation

# ╔═╡ 932b1cac-1705-43fb-bdae-91b97e39aef3
#Pkg.activate(temp = true)

# ╔═╡ b84fe36a-e673-48a6-a57f-c534608ae9bb
#Pkg.add(url = "https://github.com/epinowcast/PrimaryCensored.jl", rev = "main")

# ╔═╡ 842769b4-a499-41f6-baf1-3b54f14262d4
#using PrimaryCensored

# ╔═╡ 8c77f9db-72a7-4a09-88fa-ac3fea030f84
# CairoMakie.activate!(type = "svg")  # necessary ?

# ╔═╡ cb58eb18-aabb-4daf-94dc-a4b101791fdd
# CairoMakie.inline!(true) # necessary ?

# ╔═╡ c5ec0d58-ce3d-4b0b-a261-dbd37b119f71
md"""
# Simulating censored and truncated delay distribution data

We'll start by simulating some censored and truncated delay distribution data. We'll use the `primarycensored` function.
"""

# ╔═╡ b4409687-7bee-4028-824d-03b209aee68d
Random.seed!(1234) # Set seed for reproducibility

# ╔═╡ 30e99e77-aad1-43e8-9284-ab0bf8ae741f
md"Define the parameters for the simulation"

# ╔═╡ 28bcd612-19f6-4e25-b6df-cb43df4f2a73
n = 2000

# ╔═╡ 04e414ab-c790-4d31-b216-18776534a287
meanlog = 1.5

# ╔═╡ 54700ad7-6b2a-440f-903a-c126b4c60c0e
sdlog = 0.75

# ╔═╡ 767a58ed-9d7b-41db-a488-10f98a777474
md"we generate varying pwindow, swindow, and obs_time lengths"

# ╔═╡ 35472e04-e096-4948-a218-3de53923f271
pwindows = rand(1:2, n)

# ╔═╡ 2d0ca6e6-0333-4aec-93d4-43eb9985dc14
swindows = rand(1:2, n)

# ╔═╡ 6465e51b-8d71-4c85-ba40-e6d230aa53b1
obs_times = rand(8:10, n)

# ╔═╡ 2e04be98-625f-45f4-bf5e-a0074ea1ea01
md"Let's generates all the $n$ samples:"

# ╔═╡ a063cf93-9cd2-4c8b-9c0d-87075d1fa20d
samples = Vector{Int64}(undef, n);

# ╔═╡ 51f266ce-ffca-4ffe-aae5-ab0fa0e16479
for i in 1:n
    primarycensored_dist = primarycensored(
        LogNormal(meanlog, sdlog), Uniform(0, pwindows[i]))
    trunc_primarycensored_dist = truncated(primarycensored_dist, 0, obs_times[i])

    # make it within range, normally using truncated here. FIXME
    #trc_evt = min(rand(d), )
    samples[i] = (floor(rand(trunc_primarycensored_dist)) / swindows[i]) * swindows[i]
end

# ╔═╡ 8ea6883e-f157-4faf-9409-f91dd2581464
samples

# ╔═╡ 50757759-9ec3-42d0-a765-df212642885a
md"Create a dataframe with the data we just generated"

# ╔═╡ 5aed77d3-5798-4538-b3eb-3f4ce43d0423
delay_data = DataFrame(
    pwindow = pwindows,
    obs_time = obs_times,
    observed_delay = samples,
    observed_delay_upper = [min(obs_times[i], samples[i] + swindows[i]) for i in 1:n]
)

# ╔═╡ 06d2d1a0-883f-4f79-b94d-e4c94647e208
first(delay_data, 6)

# ╔═╡ bfef454d-3664-49de-b8ae-10c529f5a64b
md"Aggregate to unique combinations and count occurrences"

# ╔═╡ cd9e4e56-301d-44b8-b013-9cf9196e7e90

# ╔═╡ 181e3bbd-d95b-4c50-87a9-75f4851e85a1
delay_counts = combine(
    groupby(delay_data, [:pwindow, :obs_time, :observed_delay, :observed_delay_upper]),
    nrow => :n)

# ╔═╡ 993f1f74-4a55-47a7-9e3e-c725cba13c0a
md"""
Compare the samples with and without secondary censoring to the true distribution. First let's calculate empirical CDF:
"""

# ╔═╡ 5a6d605d-bff6-4b7d-97f0-ca35750411d3
empirical_cdf = ecdf(samples)

# ╔═╡ 2b773594-5187-45bc-96f4-22a3d726b7d2
# Create a sequence of x values for the theoretical CDF
x_seq = range(minimum(samples), stop = maximum(samples), length = 100)

# ╔═╡ a5b04acc-acc5-4d4d-8871-09d54caab185
# Calculate theoretical CDF using log-normal distribution
theoretical_cdf = cdf.(LogNormal(meanlog, sdlog), x_seq)

# ╔═╡ 425cdd47-c69f-4bd8-9582-97fe6433891c
# Create a long format DataFrame for plotting
cdf_data = DataFrame(
    x = vcat(x_seq, x_seq),
    probability = vcat(empirical_cdf.(x_seq), theoretical_cdf),
    type = repeat(["Observed", "Theoretical"], inner = length(x_seq))
)

# ╔═╡ fb6dc898-21a9-4f8d-aa14-5b45974c2242
begin
    # Plot using CairoMakie
    fig = Figure(size = (800, 600))

    ax = Axis(fig[1, 1], title = "Comparison of Observed vs Theoretical CDF",
        xlabel = "Delay", ylabel = "Cumulative Probability", titlealign = :center)

    # Plot the empirical and theoretical CDF
    lines!(ax, cdf_data.x[1:100], cdf_data.probability[1:100],
        color = :dodgerblue, linewidth = 2, label = "Observed")
    lines!(ax, cdf_data.x[101:end], cdf_data.probability[101:end],
        color = :black, linewidth = 2, label = "Theoretical")

    # Add vertical dashed lines for observed and theoretical means
    mean_observed = mean(samples)
    mean_theoretical = exp(meanlog + sdlog^2 / 2)

    vlines!(ax, [mean_observed], color = :dodgerblue, linestyle = :dash,
        linewidth = 1.5, label = "Mean Observed")
    vlines!(ax, [mean_theoretical], color = :black, linestyle = :dash,
        linewidth = 1.5, label = "Mean Theoretical")

    # Customize the legend and appearance
    axislegend(ax, position = :rb)

    # Show the plot
    fig
end

# ╔═╡ 9c8aebbe-8606-41e7-8e86-23129b1cbc8d
md"""
We've aggregated the data to unique combinations of `pwindow`, `swindow`, and `obs_time` and counted the number of occurrences of each `observed_delay` for each combination. This is the data we will use to fit our model.
"""

# ╔═╡ 91279812-9848-48bc-9258-b6f86c9fe923
md"""
Fitting a naive model using Turing.jl

We'll start by fitting a naive model using Turing.jl. We'll use the `Turing.jl` package to interface with Turing.jl. We define the model in a string and then write it to a file as in the [How to use primarycensoreddist with Stan](using-stan-tools.html) vignette.
"""

# ╔═╡ d995059c-81f7-441c-8695-6ba08c90a1f8
N = nrow(delay_counts)

# ╔═╡ a59d820c-616d-42ff-a0b3-d495d9f529f8
y = delay_counts.observed_delay .+ 1e-6  # Adding small constant to avoid log(0)

# ╔═╡ a257ce07-efbe-45e1-a8b0-ada40c29de8d
@model naive_model(; y, n, N) = begin
    # Priors
    mu ~ Normal(1, 1)
    sigma ~ truncated(Normal(0.5, 1), 0, Inf)

    # Likelihood
    for i in 1:N
        Turing.@addlogprob! n[i] * logpdf(LogNormal(mu, sigma), y[i])
    end
end

# ╔═╡ 49846128-379c-4c3b-9ec1-567ffa92e079

# ╔═╡ 4cf596f1-0042-4990-8d0a-caa8ba1db0c7
model = naive_model(y = y, n = delay_counts.n, N = nrow(delay_counts))

# ╔═╡ 5700a6c2-d6b1-4506-b776-7ba485262030
chain = sample(model, NUTS(0.65), 4000; chains = 4)

# ╔═╡ ac3407d6-26ed-4aa4-a7c9-40c529205915
describe(chain)

# ╔═╡ Cell order:
# ╠═30511a27-984e-40b7-9b1e-34bc87cb8d56
# ╠═bb9c75db-6638-48fe-afcb-e78c4bcc057d
# ╠═3690c122-d630-4fd0-aaf2-aea9226df086
# ╠═80ce4590-e51a-457c-8497-d16916688656
# ╠═320f081c-b26e-495a-9887-56c4212d1ed7
# ╠═960b18ef-ab1d-417f-853e-d05d2cddc78f
# ╠═932b1cac-1705-43fb-bdae-91b97e39aef3
# ╠═b84fe36a-e673-48a6-a57f-c534608ae9bb
# ╠═842769b4-a499-41f6-baf1-3b54f14262d4
# ╠═8c77f9db-72a7-4a09-88fa-ac3fea030f84
# ╠═cb58eb18-aabb-4daf-94dc-a4b101791fdd
# ╠═c5ec0d58-ce3d-4b0b-a261-dbd37b119f71
# ╠═b4409687-7bee-4028-824d-03b209aee68d
# ╠═30e99e77-aad1-43e8-9284-ab0bf8ae741f
# ╠═28bcd612-19f6-4e25-b6df-cb43df4f2a73
# ╠═04e414ab-c790-4d31-b216-18776534a287
# ╠═54700ad7-6b2a-440f-903a-c126b4c60c0e
# ╠═767a58ed-9d7b-41db-a488-10f98a777474
# ╠═35472e04-e096-4948-a218-3de53923f271
# ╠═2d0ca6e6-0333-4aec-93d4-43eb9985dc14
# ╠═6465e51b-8d71-4c85-ba40-e6d230aa53b1
# ╠═2e04be98-625f-45f4-bf5e-a0074ea1ea01
# ╠═a063cf93-9cd2-4c8b-9c0d-87075d1fa20d
# ╠═51f266ce-ffca-4ffe-aae5-ab0fa0e16479
# ╠═8ea6883e-f157-4faf-9409-f91dd2581464
# ╠═50757759-9ec3-42d0-a765-df212642885a
# ╠═5aed77d3-5798-4538-b3eb-3f4ce43d0423
# ╠═06d2d1a0-883f-4f79-b94d-e4c94647e208
# ╠═bfef454d-3664-49de-b8ae-10c529f5a64b
# ╠═cd9e4e56-301d-44b8-b013-9cf9196e7e90
# ╠═181e3bbd-d95b-4c50-87a9-75f4851e85a1
# ╠═993f1f74-4a55-47a7-9e3e-c725cba13c0a
# ╠═5a6d605d-bff6-4b7d-97f0-ca35750411d3
# ╠═2b773594-5187-45bc-96f4-22a3d726b7d2
# ╠═a5b04acc-acc5-4d4d-8871-09d54caab185
# ╠═425cdd47-c69f-4bd8-9582-97fe6433891c
# ╠═fb6dc898-21a9-4f8d-aa14-5b45974c2242
# ╠═9c8aebbe-8606-41e7-8e86-23129b1cbc8d
# ╠═91279812-9848-48bc-9258-b6f86c9fe923
# ╠═d995059c-81f7-441c-8695-6ba08c90a1f8
# ╠═a59d820c-616d-42ff-a0b3-d495d9f529f8
# ╠═a257ce07-efbe-45e1-a8b0-ada40c29de8d
# ╠═49846128-379c-4c3b-9ec1-567ffa92e079
# ╠═4cf596f1-0042-4990-8d0a-caa8ba1db0c7
# ╠═5700a6c2-d6b1-4506-b776-7ba485262030
# ╠═ac3407d6-26ed-4aa4-a7c9-40c529205915
