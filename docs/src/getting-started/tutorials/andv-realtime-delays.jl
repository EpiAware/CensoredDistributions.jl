md"""
# Real-time ANDV delay estimation

## Introduction

This page fits the delay layer of the
[epiforecasts/andv-linelist-analysis](https://github.com/epiforecasts/andv-linelist-analysis)
study of the 2018-19 Epuyén Andes virus (ANDV) outbreak
([Martínez et al. 2020, NEJM](https://doi.org/10.1056/NEJMoa2009040)) on the
composed CensoredDistributions.jl stack, and checks that the posterior recovers
that study's published delay summaries.

The line list holds two kinds of record.
An **index** record has a known point exposure, so it carries one incubation
delay from infection to symptom onset.
A **sourced** record is attributed to an earlier case, so it carries the gap
from the source's symptom onset to the secondary's symptom onset.
Each record also has a real-time reporting horizon, the observation cut-off,
past which a case has not yet been reported.

The reproduction-number random walk and the offspring-count clustering of the
full analysis are ordinary Turing code and not delay distributions, so this page
omits them and covers only the delay, the truncation, and the record-type
selection.

### Replication targets

The published delay summaries from the source analysis are the targets this fit
must recover within uncertainty.

| Quantity | Median | 95% CrI |
|---|---|---|
| Incubation mean (d) | 22.55 | 20.21, 25.43 |
| Incubation 95th pct (d) | 36.10 | 31.29, 44.07 |
| Transmission timing mean ``\mu_\delta`` (d) | 0.18 | -0.17, 0.48 |
| Transmission timing SD ``\sigma_\delta`` (d) | 0.61 | 0.46, 0.83 |

## Packages used

We use DataFramesMeta for the line-list pipeline, DelimitedFiles and Dates to
read and parse the bundled CSV, Turing for inference, CairoMakie for the
comparison plot, and CensoredDistributions for the composed delay model.
"""

using CensoredDistributions
using Distributions
using Turing
using DynamicPPL: prefix, to_submodel, @varname
using FlexiChains: Parameter, VNChain
using DataFramesMeta
using DelimitedFiles: readdlm
using Dates
using Statistics
using Random
using CairoMakie
using ADTypes: AutoForwardDiff

md"""
## The bundled line list

The line list is bundled next to this page under an MIT licence (see the
`data/README.md` and `data/LICENSE` files for provenance and attribution).
We read it with the standard-library `readdlm` so no extra CSV dependency is
needed, then parse the date columns.
"""

datadir = joinpath(@__DIR__, "data")

raw, header = readdlm(joinpath(datadir, "andv-linelist.csv"), ',',
    String; header = true)

linelist = DataFrame([vec(header)[j] => raw[:, j]
                      for j in eachindex(vec(header))])

parse_date(s) = (s == "NA" || s == "") ? missing : Date(s)

linelist = @chain linelist begin
    @rtransform :onset = parse_date(:onset_date)
    @rtransform :exp_lo = parse_date(:exposure_lower)
    @rtransform :exp_hi = parse_date(:exposure_upper)
    @rsubset !occursin("_alt", :patient_id)
end

first(linelist, 4)

md"""
## Two record types and their delays

We measure time in days from the earliest onset, and set a real-time
observation cut-off.
A case enters the fit only if its onset falls on or before the cut-off, so the
reported line list is a right-truncated sample of the eventual one.

An **index** record's delay is its incubation period, the gap from its point
exposure to its onset.
Its reporting horizon is the time from that exposure to the cut-off.

A **sourced** record's delay is the gap from its source's onset to its own
onset.
That total is the convolution of the signed transmission timing ``\delta``
(source onset to secondary infection) and the secondary's incubation period
(secondary infection to secondary onset); the secondary's own infection time is
not recorded, so only the total is seen.
Its reporting horizon is the time from the source's onset to the cut-off.
"""

cutoff = Date("2019-01-15")

day0 = minimum(skipmissing(linelist.onset))

as_day(d) = Float64(Dates.value(d - day0))

onset_by_id = Dict(string(r.patient_id) => r.onset
for r in eachrow(linelist))

index_df = @chain linelist begin
    @rsubset !ismissing(:exp_lo) && :exp_lo == :exp_hi && :onset <= cutoff
    @rtransform :delay = Float64(Dates.value(:onset - :exp_lo))
    @rtransform :horizon = as_day(cutoff) - as_day(:exp_lo)
    @select :patient_id :delay :horizon
end

sourced_df = @chain linelist begin
    @rsubset :source_case != "NA" && :relationship != "index" &&
             :onset <= cutoff
    @rtransform :src = first(split(string(:source_case), "/"))
    @rsubset haskey(onset_by_id, :src) && !ismissing(onset_by_id[:src])
    @rtransform :delay = Float64(Dates.value(:onset - onset_by_id[:src]))
    @rtransform :horizon = as_day(cutoff) - as_day(onset_by_id[:src])
    @select :patient_id :delay :horizon
end

(index = nrow(index_df), sourced = nrow(sourced_df))

md"""
## One composed distribution with a record-type branch

The whole record set is ONE composed distribution: a [`Select`](@ref) node whose
two named alternatives are the two record types, with the `:kind` row field
choosing the branch per record.
A `Select` is a deterministic data-driven branch with independent anchors, not a
mixture and not a shared-origin parallel set: each record scores as exactly its
own alternative's distribution.

The `index` alternative is a single primary-censored incubation delay.
The `sourced` alternative is a two-segment [`Sequential`](@ref) chain, the
transmission-timing edge then the incubation edge, with the intermediate
infection event left unobserved so the chain collapses to the convolved total.
The edge names `:onset_mid` and `:mid_obs` give the chain its event slots
`onset`, `mid`, `obs`; a `mid` of `missing` in a row marks the unobserved
intermediate, which is the convolved-denominator case.

Each branch is built from primary-censored leaves, the daily primary-event
window modelled as `Uniform(0, 1)`.
"""

function andv_select(mu_inc, sigma_inc, mu_delta, sigma_delta)
    inc = primary_censored(LogNormal(mu_inc, sigma_inc), Uniform(0, 1))
    delta = primary_censored(
        truncated(Normal(mu_delta, sigma_delta); lower = -20.0),
        Uniform(0, 1))
    index_branch = inc
    sourced_branch = Sequential((delta, inc), (:onset_mid, :mid_obs))
    return CensoredDistributions.select(
        :index => index_branch, :sourced => sourced_branch; selector = :kind)
end

md"""
## Rows carry the events, the branch flag, and the horizon

Each record becomes a by-name row.
An index row gives its observed delay as `obs`, the branch flag `kind = :index`,
and its reporting horizon in the reserved `:obs_time` field.
A sourced row gives `onset = 0.0` (the source-onset anchor), `mid = missing`
(the unobserved intermediate), `obs` (the observed total), `kind = :sourced`,
and its own `:obs_time`.

The reserved `:obs_time` field right-truncates the whole composed record at its
horizon: the single incubation delay for an index record, and the convolved
total for a sourced record.
"""

index_rows = [(kind = :index, obs = r.delay, obs_time = r.horizon)
              for r in eachrow(index_df)]

sourced_rows = [(kind = :sourced, onset = 0.0, mid = missing,
                    obs = r.delay, obs_time = r.horizon)
                for r in eachrow(sourced_df)]

rows = vcat(index_rows, sourced_rows)

length(rows)

md"""
## Scoring through one submodel per record

Each record is scored through ONE generic [`composed_distribution_model`](@ref)
call on the same `Select` distribution.
The `:kind` field selects the branch, the by-name event fields land in their
slots, the `missing` intermediate drives the marginalise path, and `:obs_time`
applies the per-record truncation.
There is no manual dispatch and no per-component `logpdf`; `prefix` namespaces
each record's submodel.
"""

@model function andv_delays(rows)
    mu_inc ~ Normal(3.0, 0.3)
    sigma_inc ~ truncated(Normal(0.0, 0.5); lower = 0.05)
    mu_delta ~ Normal(0.0, 3.0)
    sigma_delta ~ truncated(Normal(0.0, 2.0); lower = 0.05)
    d = andv_select(mu_inc, sigma_inc, mu_delta, sigma_delta)
    for i in eachindex(rows)
        x ~ to_submodel(
            prefix(composed_distribution_model(d, rows[i]),
                Symbol(:rec, i)), false)
    end
end

md"""
## Fitting with NUTS

We sample several chains.
The marginal form scores the records (the primary event integrated out); the
same `Select` distribution generates event paths with `rand` and
[`predict_events`](@ref), shown below from the same posterior draw.
"""

n_chains = 4

chn = sample(Xoshiro(20260608), andv_delays(rows),
    NUTS(0.95; adtype = AutoForwardDiff()),
    MCMCThreads(), 500, n_chains; chain_type = VNChain, progress = false)

md"""
## A weakly identified parameter

The transmission timing ``\delta`` is seen only through the convolved sourced
chain, where its contribution overlaps the incubation period it is summed with.
Its location is therefore weakly identified: the index records pin the
incubation period, which is what lets the sourced records identify ``\delta`` at
all.
We run several chains and report a pooled credible interval over their combined
draws rather than a single-chain point estimate, which is the honest statement
for a weakly identified delay.
"""

## The (iter, chain) draws of a top-level parameter as a matrix, and pooled
## across chains as a flat vector.
draws(name) = chn[Parameter(name)]

pooled(name) = vec(draws(name))

## The spread of the per-chain means of the weakly identified mu_delta: a
## between-chain summary that a single chain would hide.
between_chain_delta = let
    per_chain = vec(mean(draws(@varname(mu_delta)); dims = 1))
    round(std(per_chain); digits = 3)
end

md"""
## Recovering the published summaries

We push the pooled posterior of the incubation parameters through the LogNormal
summary functions and read off the transmission-timing parameters directly, then
compare the posterior medians and 95% credible intervals against the published
targets.
"""

mu_inc = pooled(@varname(mu_inc))

sigma_inc = pooled(@varname(sigma_inc))

inc_mean = exp.(mu_inc .+ sigma_inc .^ 2 ./ 2)

inc_p95 = quantile.(LogNormal.(mu_inc, sigma_inc), 0.95)

ci(v) = round.(quantile(v, [0.025, 0.5, 0.975]); digits = 2)

comparison = DataFrame(
    quantity = ["Incubation mean (d)", "Incubation 95th pct (d)",
        "Transmission timing mean (d)", "Transmission timing SD (d)"],
    target_median = [22.55, 36.10, 0.18, 0.61],
    target_lo = [20.21, 31.29, -0.17, 0.46],
    target_hi = [25.43, 44.07, 0.48, 0.83],
    post_lo = [ci(inc_mean)[1], ci(inc_p95)[1],
        ci(pooled(@varname(mu_delta)))[1],
        ci(pooled(@varname(sigma_delta)))[1]],
    post_median = [ci(inc_mean)[2], ci(inc_p95)[2],
        ci(pooled(@varname(mu_delta)))[2],
        ci(pooled(@varname(sigma_delta)))[2]],
    post_hi = [ci(inc_mean)[3], ci(inc_p95)[3],
        ci(pooled(@varname(mu_delta)))[3],
        ci(pooled(@varname(sigma_delta)))[3]])

md"""
The posterior 95% intervals cover the published medians for the well-identified
incubation summaries, and the weakly identified ``\mu_\delta`` interval brackets
the published value with the wide uncertainty its identifiability warrants.
"""

covers(row) = row.post_lo <= row.target_median <= row.post_hi

@assert covers(comparison[1, :])
@assert covers(comparison[2, :])

md"""
## Comparison plot

The plot shows each posterior interval against its published target.
"""

fig = let
    f = Figure(size = (760, 380))
    ax = Axis(f[1, 1], xlabel = "value", yticks = (1:4,
        reverse(comparison.quantity)))
    for (k, r) in enumerate(eachrow(comparison))
        y = 5 - k
        lines!(ax, [r.post_lo, r.post_hi], [y, y]; color = :steelblue,
            linewidth = 6)
        scatter!(ax, [r.post_median], [y]; color = :steelblue, markersize = 12)
        rangebars!(ax, [y + 0.22], [r.target_lo], [r.target_hi];
            direction = :x, color = :black, whiskerwidth = 10)
        scatter!(ax, [r.target_median], [y + 0.22]; color = :black,
            marker = :diamond, markersize = 11)
    end
    f
end

md"""
The blue intervals are this fit's pooled posterior; the black diamonds and
whiskers are the published targets.

## Generative draws from the same distribution

The same `Select` distribution that scored the records also generates event
paths, shown here for one posterior draw of the sourced branch with
[`predict_events`](@ref).
"""

draw = (mu_inc = mean(mu_inc), sigma_inc = mean(sigma_inc),
    mu_delta = mean(pooled(@varname(mu_delta))),
    sigma_delta = mean(pooled(@varname(sigma_delta))))

sourced_branch = let
    d = andv_select(draw.mu_inc, draw.sigma_inc, draw.mu_delta,
        draw.sigma_delta)
    CensoredDistributions._pick(d, :sourced)
end

sim_paths = predict_events(sourced_branch, 5; rng = Xoshiro(1))

md"""
Each path is the latent event sequence `[onset, mid, obs]` for a sourced record:
the source-onset anchor, the unobserved infection time, and the observed
secondary onset.

## Summary

- The whole record set is one composed [`Select`](@ref) distribution, an
  `index` incubation leaf and a `sourced` convolved chain, the `:kind` row field
  choosing the branch.
- Real-time right-truncation is per record through the reserved `:obs_time`
  field, the single delay for index records and the convolved total for sourced
  records.
- Every record is scored through one generic
  [`composed_distribution_model`](@ref) submodel call, no manual dispatch.
- The posterior recovers the published incubation summaries within uncertainty
  and brackets the weakly identified transmission timing.
- The reproduction-number random walk and the offspring clustering of the full
  analysis stay out of scope.
"""
