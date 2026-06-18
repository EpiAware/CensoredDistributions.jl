md"""
# [Pairwise survival analysis of transmission (Kenah)](@id pairwise-survival-transmission)

## Introduction

Classical survival analysis is per individual: each person has one failure
time and one hazard.
Infectious disease breaks that independence — whether and when one person is
infected depends on who else is infected and when.
Kenah's pairwise survival framework restores a survival-analysis likelihood by
working over *ordered pairs* of individuals rather than individuals
[kenah2011contact](@cite).

For an ordered pair ``(i, j)`` the **contact interval** ``\tau_{ij}`` is the
time from the onset of infectiousness in ``i`` to *infectious contact* from
``i`` to ``j`` — a contact that would infect ``j`` if ``j`` were still
susceptible.
The contact interval has a hazard of infectious contact ``\lambda(\tau)``,
and its distribution is the quantity we want to estimate.
A susceptible ``j`` is infected at the minimum contact time over all its
infectious sources, and *who-infected-whom* is the ``\arg\min``.
This is exactly a **competing risks across sources** problem: the sources race,
the first contact wins, and timing and cause are coupled.

This page maps that structure directly onto the CensoredDistributions composer
stack.

### What are we going to do in this exercise

We fit the 1861 Hagelloch measles outbreak, the canonical dataset for this
method, and recover the **transmission coefficients** ``\beta`` of Kenah's
hazard-of-infectious-contact regression — the within-household and
within-class effects, which are the result to trust here.
We also read off an absolute household reproduction number, but treat it as an
illustration of the mapping rather than a calibrated estimate, because this
compact version omits infectious-period gating (explained where we compute it).

1. Map each Kenah pairwise concept onto a composed primitive:

| Kenah pairwise concept | Composed primitive |
|---|---|
| contact interval ``\tau_{ij}`` with hazard ``\lambda_0(\tau)\,e^{\beta' x_{ij}}`` | a `Distributions.Weibull` leaf with a per-pair rate |
| susceptible ``j`` infected at ``\min_i \tau_{ij}``, source ``=\arg\min`` | racing-hazard [`compete`](@ref) node ([`Compete`](@ref)) |
| pair right-censored (``j`` infected elsewhere / source recovers / study ends) | the racing-hazard survival ``\prod_k S_k`` (`logccdf`) |
| contact-interval observation windows (dates to the day) | a [`double_interval_censored`](@ref) leaf (sketched in the refinements) |
| transmission coefficients ``\beta`` (household, class, …) | a per-pair log-rate regression ``\lambda_0 e^{\beta' x_{ij}}`` |

2. Identify the baseline contact interval as a stock `Weibull` leaf and put a
   log-rate regression ``\lambda_0 e^{\beta' x_{ij}}`` on its rate.
3. Build the pairwise survival likelihood as a racing-hazard compete node.
4. Fit the Hagelloch data, recover the transmission coefficients ``\beta`` and
   the within-household effect, and read off an illustrative household ``R_0``.

### What might I need to know before starting

This tutorial builds on [Getting Started with
CensoredDistributions.jl](@ref getting-started) and the composer reference,
[Composing censored distributions](@ref composer-toolkit).

## The method

Kenah models the **hazard of infectious contact** for an ordered pair as a
*regression* on pair covariates, not as a handful of free rates
[kenah2011contact](@cite).
The contact interval of pair ``(i, j)`` has hazard

```math
\lambda_{ij}(\tau \mid x_{ij}) = \lambda_0(\tau)\, e^{\beta' x_{ij}},
```

a baseline hazard ``\lambda_0(\tau)`` shared by every pair times a
**proportional** factor ``e^{\beta' x_{ij}}`` set by the pair's covariates
``x_{ij}`` (does the pair share a household, a school class, …).
The vector ``\beta`` holds the **transmission coefficients**: each ``\beta_m`` is
the log-hazard-ratio of infectious contact for covariate ``m`` (a unit increase
in ``x_m`` multiplies the contact hazard by ``e^{\beta_m}``), and ``\beta`` is
what we estimate.

We take the baseline as the `transtat` Weibull, a rate ``\lambda_0`` and a shape
``\gamma`` with baseline cumulative hazard

```math
H_0(\tau) = (\lambda_0 \tau)^\gamma,
\qquad S_0(\tau) = e^{-(\lambda_0 \tau)^\gamma},
\qquad h_0(\tau) = \lambda_0 \gamma (\lambda_0 \tau)^{\gamma - 1}.
```

``\gamma > 1`` gives an increasing baseline hazard (infectiousness that builds
after onset), ``\gamma < 1`` a decreasing one.
The Weibull is the rare family that is both a proportional-hazards and an
accelerated-failure-time model, so the log-rate regression
``\lambda_{ij} = \lambda_0\, e^{\beta' x_{ij}}`` keeps each pair's contact
interval a Weibull: a stock `Distributions.Weibull(γ, 1/λ_{ij})` whose rate
carries the linear predictor.
A single binary household covariate recovers the two-rate special case exactly,
``\log(\lambda_w / \lambda_b) = \beta_\text{household}``, but the same structure
takes any number of pair covariates.
Any `UnivariateDistribution` whose `logpdf` and `logccdf` differentiate under
the chosen AD backend is a valid cause-specific delay for the racing-hazard
node, so a richer baseline — a
[SurvivalDistributions.jl](@ref survival-delay-families) leaf or a
piecewise-constant hazard — drops in once its survival is AD-safe; some
families need the package's AD-safe routing for that (the SurvivalDistributions
extension adds it for `GeneralizedGamma`).
We use the Weibull baseline here because the `transtat` contact interval *is* a
Weibull and the regression rides on its rate.

For a susceptible ``j`` infected at time ``t_j``, let ``R(j)`` be the set of
sources infectious before ``t_j`` and write ``g_{ij} = t_j - o_i`` for the gap
since source ``i``'s infectiousness onset ``o_i``.
The pairwise likelihood contribution of ``j`` is the racing-hazard
cause-resolved marginal density

```math
\sum_{i \in R(j)} f(g_{ij}) \prod_{k \in R(j),\, k \ne i} S(g_{kj}),
```

i.e. *some* source contacted ``j`` at ``t_j`` while every other at-risk source
had not yet — the `logpdf` of the racing-hazard
[`compete`](@ref) node over ``R(j)``.
A pair ``(i, j)`` where ``j`` is never infected by ``i`` (``j`` is infected by
another source first, ``i`` recovers, or the study ends) contributes the
**survival** ``S(\text{at-risk duration})`` — the racing-hazard
`logccdf`, which is the product ``\prod_k S_k`` of the at-risk pair
survivals.
These survival terms are the no-event / right-censored branch of the
likelihood, and they are what make this a *survival* analysis rather than a
binomial secondary-attack-rate calculation [kenah2011contact](@cite).

## Packages used
"""

using CSV, DataFramesMeta, Dates
using CensoredDistributions, Distributions
using Turing, Random, Statistics
using LinearAlgebra: dot, I
using FlexiChains: Parameter, Extra, rhat, ess
import Mooncake
using ADTypes: AutoMooncake

md"""
## The contact-interval leaf and its rate regression

A contact interval with cumulative hazard ``(\lambda\tau)^\gamma`` is a stock
`Distributions.Weibull`: a `transtat` rate ``\lambda`` and shape ``\gamma`` are a
Weibull of shape ``\gamma`` and scale ``1/\lambda``.
`contact_interval` is just that mapping, so the rest of the page works in the
natural ``(\lambda, \gamma)`` hazard parameterisation while the leaf is a tested
library distribution with AD-safe `logpdf`/`logccdf` (the survival the
racing-hazard node needs) and a `quantile`/`rand` for simulation.

Kenah's regression rides on the rate: a pair's rate is the baseline rate scaled
by its linear predictor, ``\lambda_{ij} = \lambda_0\, e^{\beta' x_{ij}}``.
`pair_rate` evaluates that log-rate regression, so a pair leaf is
`contact_interval(pair_rate(log_lambda0, beta, x), gamma)`.
"""

contact_interval(lambda::Real, gamma::Real) = Weibull(gamma, 1 / lambda)

## Pair rate λ_ij = exp(log λ0 + β'x): the baseline log-rate plus the linear
## predictor over the pair covariates, exponentiated to a positive rate.
function pair_rate(log_lambda0::Real, beta::AbstractVector, x::AbstractVector)
    return exp(log_lambda0 + dot(beta, x))
end

md"""
We can read the hazard form straight off the Weibull to confirm the mapping: the
survival ``S(\tau) = e^{-(\lambda\tau)^\gamma}`` and the cumulative hazard
``-\log S(\tau) = (\lambda\tau)^\gamma`` are exactly the `transtat`
parameterisation, and a unit covariate scales the rate by ``e^{\beta_m}``.
"""

let log_lambda0 = log(0.1), beta = [1.4], gamma = 1.6, tau = 3.0
    rate_within = pair_rate(log_lambda0, beta, [1.0])  # share covariate
    rate_between = pair_rate(log_lambda0, beta, [0.0])  # baseline
    d = contact_interval(rate_within, gamma)
    (; survival = ccdf(d, tau),
        cumhazard = -logccdf(d, tau),
        transtat_cumhazard = (rate_within * tau)^gamma,
        rate_ratio = rate_within / rate_between,
        exp_beta = exp(beta[1]))
end

md"""
## A racing-hazard node over sources

When several sources became infectious *at the same time*, the susceptible
faces a racing hazard on a shared clock: a [`compete`](@ref) node built from
*bare* contact-interval delays (no branch probabilities), which selects the
racing-hazard [`Compete`](@ref) type — the winning source is *derived*
from the hazards, not a free parameter.
The node's `logpdf` is the cause-resolved marginal density
``\sum_i f(\tau) \prod_{k \ne i} S(\tau)``, its `logccdf` is the joint
survival ``\prod_k S_k``, and [`winning_probabilities`](@ref) is the per-source
``\arg\min`` (who-infected-whom) split.
"""

let node = compete(:near => contact_interval(0.4, 1.6),
        :far => contact_interval(0.1, 1.6))
    (; marginal_logpdf = logpdf(node, 2.0),
        joint_logsurvival = logccdf(node, 2.0),
        winning = winning_probabilities(node))
end

md"""
## Anchoring a source's contact interval on the outbreak clock

In a real outbreak the sources do *not* share a clock: each source ``i`` starts
its own contact-interval clock at its infectiousness onset ``o_i``, so on the
*outbreak* clock source ``i``'s contact time is ``o_i + \tau``.
This is a deterministic shift of the contact-interval leaf by the source's
onset, which is exactly the additive special case of the exported
[`affine`](@ref) primitive ``Y = \text{scale}\cdot X + \text{shift}``:
`affine(leaf; shift = o_i)` reads the underlying leaf at the gap ``t - o_i`` when
evaluated at an outbreak time ``t``, and nests as a leaf in a
[`compete`](@ref) node like any other distribution.
This lets a racing-hazard node over sources be scored at the *susceptible's*
infection time directly, with every source automatically read at its own gap
``g_{ij} = t_j - o_i`` — the node's `logpdf` at ``t_j`` is then exactly the
cause-resolved pairwise density ``\sum_i f(g_{ij}) \prod_{k \ne i} S(g_{kj})``.
"""

md"""
Two sources with different onsets, scored at the susceptible's infection time:
the node's marginal `logpdf` reads each source at its own gap, and the joint
survival is the product of the per-source survivals.
"""

let onset_near = 1.0, onset_far = 2.5, t_infect = 6.0
    node = compete(
        :near => affine(contact_interval(0.4, 1.6); shift = onset_near),
        :far => affine(contact_interval(0.1, 1.6); shift = onset_far))
    (; marginal_logpdf = logpdf(node, t_infect),
        joint_logsurvival = logccdf(node, t_infect))
end

md"""
## The Hagelloch line list

The 1861 Hagelloch measles outbreak (n = 188 children) is the standard test
bed for transmission survival methods [kenah2011contact, neal2004statistical](@cite):
every case carries a household (`family_ID`), a school class (`class`), a
putative infector (`infector`), and dated symptom milestones.
The household and class give us two **pair covariates** — does a source and
susceptible share a household, do they share a school class — the regressors
``x_{ij}`` of the contact-hazard regression.
We take the **date of prodrome** as the onset of infectiousness — the event
that starts each source's contact-interval clock — and measure time in days
from the first prodrome in the outbreak.
"""

datadir = joinpath(@__DIR__, "data", "hagelloch")

raw = CSV.read(joinpath(datadir, "linelist.csv"), DataFrame;
    missingstring = ["NA", ""])

md"""
We parse the dates to a common day origin, keep the columns the pairwise model
needs, and sort by onset so a source always precedes the people it can infect.
The infectiousness onset is the prodrome date; a case with no prodrome date is
dropped.
"""

origin = minimum(skipmissing(Date.(raw.date_of_prodrome)))
day_of(x) = ismissing(x) ? missing : Float64(Dates.value(Date(x) - origin))

cases = @chain raw begin
    @rtransform begin
        :onset = day_of(:date_of_prodrome)
        :household = :family_ID
    end
    @subset .!ismissing.(:onset)
    @select(:case_ID, :infector, :onset, :household, :age, :class)
    @orderby(:onset)
end

n = nrow(cases)
(; n_cases = n, n_households = length(unique(cases.household)),
    first_onset = origin, span_days = maximum(cases.onset))

md"""
Every case here is an *observed infection*: Hagelloch is a closed outbreak with
no surviving susceptibles recorded, so the right-censoring in this likelihood
comes from pairs whose source had not yet contacted the susceptible by the time
some *other* source did (the racing survivors), not from never-infected
children.
The study horizon `T_end` closes the at-risk window for the last infections.
"""

onset = collect(cases.onset)
household = collect(cases.household)
class = collect(cases.class)
T_end = maximum(onset)
(; T_end, mean_household_size = n / length(unique(household)),
    n_classes = length(unique(class)))

md"""
## The pairwise likelihood

For each infected susceptible ``j`` we assemble its at-risk source set
``R(j)`` — every case whose infectiousness onset precedes ``t_j`` — anchor each
source's contact interval at its onset, and score ``j`` through a racing-hazard
[`compete`](@ref) node over those anchored sources, evaluated at ``j``'s
infection time ``t_j``.
Each pair ``(i, j)`` carries a covariate vector ``x_{ij}`` (does the pair share a
household, a school class) and its contact-interval rate is the log-rate
regression ``\lambda_{ij} = \lambda_0\, e^{\beta' x_{ij}}``, with a baseline rate
``\lambda_0`` and shape ``\gamma`` shared across pairs.
The coefficients ``\beta`` are the **transmission coefficients**: ``\beta_m`` is
the log-hazard-ratio of infectious contact for covariate ``m``, the central
estimands.

`pairwise_loglik` walks the cases once.
Each susceptible's term is the racing-hazard node's marginal `logpdf` at
``t_j``: this is the cause-resolved sum over the at-risk sources,
``\sum_i f(g_{ij}) \prod_{k \ne i} S(g_{kj})``, so the winning source's density
is multiplied by the **survival** ``S`` of every *non*-winning at-risk source.
Those survivals are the right-censoring of the racing pairs — every source that
*could* have infected ``j`` but did not is automatically a survival term inside
the same node, exactly the no-event branch of the racing-hazard likelihood.
The first case has no prior source and seeds the outbreak, so it contributes no
pairwise term (its introduction is exogenous).
"""

## Pair covariates x_ij for the contact-hazard regression: a share-household
## indicator and a share-(non-zero)-class indicator. Class 0 is "no school
## class", so a shared class only counts when both children are in the same
## actual class. Extend `x_ij` here to add more pair covariates (age gap,
## spatial distance, …) and `beta` grows to match.
function pair_covariates(household, class, i, j)
    same_household = household[i] == household[j] ? 1.0 : 0.0
    same_class = (class[i] == class[j] && class[i] != 0) ? 1.0 : 0.0
    return [same_household, same_class]
end

function source_delays(
        log_lambda0, beta, gamma, onset, household, class, sources, j)
    return Tuple(
        affine(
            contact_interval(
                pair_rate(log_lambda0, beta,
                    pair_covariates(household, class, i, j)),
                gamma);
            shift = onset[i])
    for i in sources)
end

function source_node(
        log_lambda0, beta, gamma, onset, household, class, sources, j)
    delays = source_delays(
        log_lambda0, beta, gamma, onset, household, class, sources, j)
    names = ntuple(k -> Symbol(:src, k), length(sources))
    return compete((names[k] => delays[k] for k in eachindex(sources))...)
end

## Count the sources infectious before j's own infection (j's at-risk set size).
function n_sources_at_risk(onset, j)
    return count(i -> i != j && onset[i] < onset[j], eachindex(onset))
end

md"""
The at-risk structure — which sources can infect each susceptible, the gap to
each, and the pair covariates ``x_{ij}`` — does not depend on the parameters, so
we precompute it once.
`atrisk_pairs` returns, per infected susceptible, the vector of source gaps
``g_{ij} = t_j - o_i`` and a matching vector of covariate vectors ``x_{ij}``; the
seed case (no prior source) is dropped.
"""

function atrisk_pairs(onset, household, class)
    pairs = Vector{Tuple{Vector{Float64}, Vector{Vector{Float64}}}}()
    for j in eachindex(onset)
        tj = onset[j]
        gaps = Float64[]
        covs = Vector{Float64}[]
        for i in eachindex(onset)
            (i == j || onset[i] >= tj) && continue
            push!(gaps, tj - onset[i])
            push!(covs, pair_covariates(household, class, i, j))
        end
        isempty(gaps) || push!(pairs, (gaps, covs))
    end
    return pairs
end

pairs = atrisk_pairs(onset, household, class)
(; n_susceptibles_scored = length(pairs),
    n_ordered_pairs = sum(length(first(p)) for p in pairs),
    n_covariates = length(first(first(pairs)[2])))

md"""
For one susceptible the racing-hazard marginal at ``t_j`` is the log-sum-exp of
each source's cause-resolved log sub-density
``\log f(g_{ij}) + \sum_{k \ne i} \log S(g_{kj})``.
This is *exactly* `logpdf(source_node(...), t_j)`: `source_node` builds the
racing-hazard [`compete`](@ref) node over the anchored sources and its marginal
`logpdf` is the per-susceptible likelihood. We verify that equality below, then
fit through the direct reduction `susceptible_loglik`.

The package's higher-level Turing tooling
([`composed_distribution_model`](@ref), which scores a record — or a whole table
of records in one `~` — against a composed distribution) covers a *fixed* record
graph scored against observed events, where the node structure and its event
names are the same for every record. The pairwise likelihood here is not that shape: the
racing node is *rebuilt per susceptible* over a different at-risk source set
``R(j)`` (a different arity, different onsets, different pair covariates),
and the susceptible's infection time is scored as the node's marginal rather than
as a named-event record. So the right composed primitive is the racing-hazard
[`compete`](@ref) node — which the package *does* supply and which the
agreement check below confirms is the same likelihood — and the manual part that
remains is only the per-susceptible loop and the log-sum-exp reduction.
`Compete`'s own `logpdf` is itself an AD-safe log-sum-exp, so fitting
through `logpdf(source_node(...), t_j)` differentiates cleanly under the
Mooncake backend used here; `susceptible_loglik` is a *per-susceptible inlining*
of that node `logpdf`, equivalent to it (checked below), that skips rebuilding a
fresh `compete` node of a new arity for each susceptible on every gradient
step.
Each source's rate is the regression ``\lambda_0\, e^{\beta' x_{ij}}`` evaluated
at the pair's covariates; the shared shape ``\gamma`` carries the hazard's time
shape.
"""

function susceptible_loglik(log_lambda0, beta, gamma, gaps, covs)
    leaf(k) = contact_interval(pair_rate(log_lambda0, beta, covs[k]), gamma)
    ## joint survival ∏ S_k(g_kj) = Σ logccdf, shared by every cause term
    total_logsurv = sum(logccdf(leaf(k), gaps[k]) for k in eachindex(gaps))
    ## cause-resolved term for source k: log f_k - log S_k + Σ_i log S_i.
    function cause_term(k)
        l = leaf(k)
        return logpdf(l, gaps[k]) - logccdf(l, gaps[k]) + total_logsurv
    end
    ## log-sum-exp over the sources in one pass, tracking the running max and the
    ## accumulated sum so no per-source vector is allocated on the AD path.
    running_max = cause_term(1)
    acc = one(running_max)
    for k in 2:length(gaps)
        term = cause_term(k)
        if term > running_max
            acc = acc * exp(running_max - term) + one(acc)
            running_max = term
        else
            acc += exp(term - running_max)
        end
    end
    return running_max + log(acc)
end

md"""
The direct computation agrees with the racing-hazard `compete` node it stands
in for. For one Hagelloch susceptible with several at-risk sources we build the
node with `source_node` and check its marginal at ``t_j`` against
`susceptible_loglik`.
"""

let j = findfirst(j -> n_sources_at_risk(onset, j) >= 3, eachindex(onset)),
    srcs = [i for i in eachindex(onset) if i != j && onset[i] < onset[j]]

    log_lambda0, beta, gamma = log(0.05), [1.4, 0.5], 1.3
    node = source_node(
        log_lambda0, beta, gamma, onset, household, class, srcs, j)
    gaps = [onset[j] - onset[i] for i in srcs]
    covs = [pair_covariates(household, class, i, j) for i in srcs]
    (; n_sources = length(srcs),
        node_logpdf = logpdf(node, onset[j]),
        direct_logpdf = susceptible_loglik(log_lambda0, beta, gamma, gaps, covs))
end

md"""
The two agree (the composer node and the direct reduction are the same
likelihood), so the fit can use the fast direct form. The full likelihood sums
the per-susceptible term over the precomputed at-risk pairs.
"""

function pairwise_loglik(log_lambda0, beta, gamma, pairs)
    lp = zero(log_lambda0)
    for (gaps, covs) in pairs
        lp += susceptible_loglik(log_lambda0, beta, gamma, gaps, covs)
    end
    return lp
end

md"""
The racing-hazard node's marginal `logpdf` sums the cause-resolved sub-densities
over the sources, so we do not need to know which source actually won to fit the
contact-interval parameters; the unknown infector is marginalised.

## Fit with Turing

We put a weakly-informative prior on the baseline log-rate ``\log\lambda_0``,
mean-zero Normal priors on the transmission coefficients ``\beta`` (so the data
drive each log-hazard-ratio away from no effect), and a positive prior on the
shape, and score the whole line list with `Turing.@addlogprob!`.
The shape prior is centred at one (a constant baseline hazard, the exponential
special case) so the data drive any departure.
The likelihood is differentiated with Mooncake reverse mode (`AutoMooncake`),
the package's preferred reverse-mode backend: the pairwise loglik is a pure-Julia
log-sum-exp over the Weibull `logpdf`/`logccdf`, with no string or control-flow
operations that Mooncake cannot trace, so it compiles a rule cleanly here.
"""

@model function hagelloch_pairwise(pairs, n_cov)
    log_lambda0 ~ Normal(log(0.03), 1.0)
    beta ~ MvNormal(zeros(n_cov), 1.5^2 * I)
    gamma ~ truncated(Normal(1.0, 0.5); lower = 0.2)
    Turing.@addlogprob! pairwise_loglik(log_lambda0, beta, gamma, pairs)
end

n_cov = length(first(first(pairs)[2]))
rng = MersenneTwister(2024)
chain = sample(rng, hagelloch_pairwise(pairs, n_cov),
    NUTS(0.8; adtype = AutoMooncake(; config = nothing)),
    MCMCThreads(), 1000, 4; progress = false)

md"""
We check the sampler before reading the estimands: across the four chains we
take the worst-case ``\hat{R}`` (the between- over within-chain variance ratio,
which should sit near one) and the smallest effective sample size over the model
parameters, and count any divergent transitions flagged by NUTS.
"""

model_params = [k for k in keys(rhat(chain)) if k isa Parameter]
divergences = sum(skipmissing(vec(chain[Extra(:numerical_error)])))
(; max_rhat = maximum(rhat(chain)[k] for k in model_params),
    min_ess = minimum(ess(chain)[k] for k in model_params),
    n_divergences = divergences)

md"""
The posterior summaries for the baseline rate, the transmission coefficients
``\beta`` (household, class), and the shape:
"""

## The chain stores `beta` as one vector-valued draw per sample, so `vec(chain
## [:beta])` is a vector of `[β₁, …]` vectors; pull covariate m's marginal out.
covariate_names = (:household, :class)
beta_draws(m) = getindex.(vec(chain[:beta]), m)
post = (; log_lambda0 = mean(chain[:log_lambda0]),
    beta = [mean(beta_draws(m)) for m in 1:n_cov],
    gamma = mean(chain[:gamma]))

md"""
## Estimands: transmission coefficients and ``R_0``

The **transmission coefficients** ``\beta`` are the headline estimands, and they
are the part of this fit to trust: they are *relative* quantities (a log-hazard
ratio between pair types) and so are insensitive to the absolute scale of the
at-risk denominator discussed below.
``\beta_m`` is the log-hazard-ratio of infectious contact for covariate ``m``, so
``e^{\beta_m}`` is the multiplicative effect on the contact *rate*: sharing a
household (or a school class) multiplies the rate of infectious contact by
``e^{\beta_\text{household}}`` (or ``e^{\beta_\text{class}}``).
Both effects recover well — strong within-household and within-class contact
hazards, with ``\gamma > 1`` (infectiousness building after prodrome) — and
match the Kenah (2011) and Neal & Roberts (2004) picture qualitatively.
Because a Weibull's cumulative hazard scales as the rate raised to the shape, the
same effect on the *cumulative-hazard* scale (the within-window contact
probability) is ``e^{\gamma \beta_m}``; the single-covariate special case
recovers the old within-household hazard ratio ``(\lambda_w/\lambda_b)^\gamma =
e^{\gamma\beta_\text{household}}``.
"""

beta_chains = [beta_draws(m) for m in 1:n_cov]
gamma_chain = vec(chain[:gamma])
beta_summary = [(; covariate = covariate_names[m],
                    beta_mean = mean(beta_chains[m]),
                    beta_90 = quantile(beta_chains[m], (0.05, 0.95)),
                    rate_hazard_ratio = mean(exp.(beta_chains[m])),
                    cumhazard_ratio = mean(exp.(gamma_chain .* beta_chains[m])))
                for m in 1:n_cov]

md"""
``R_0`` in the pairwise framework is the expected number of infectious contacts
a case makes with *susceptible* others over its infectious period
[kenah2011contact](@cite).
For each susceptible the per-pair probability of an infectious contact within an
infectious window of length ``w`` is the contact-interval cdf ``F(w)``, so the
**household reproduction number** — the expected secondary infections within a
case's own household — is ``F(w)`` at the within-household rate
``\lambda_0\, e^{\beta_\text{household}}`` times the mean number of household
susceptibles.
We take a measles-typical infectious window of `w = 8` days
[neal2004statistical](@cite); a community ``R_0`` would add the between-household
contacts over the larger susceptible pool, governed by the baseline rate
``\lambda_0``.

Unlike the ``\beta`` ratios, this is an *absolute* estimand, and we show it as an
illustration of the mapping, not as a reliable measles ``R_0``.
It comes out well below the historical within-household secondary attack rate for
measles (``\approx 0.6\text{–}0.8``), and the reason is structural: every prior
case stays an at-risk source for every later susceptible forever, because this
page does *not* gate sources by an infectious period.
That inflates the surviving-pair denominator, pulls the baseline rate
``\lambda_0`` (and hence ``F(w)`` and ``R_\text{household}``) low, and biases the
absolute number downward.
Gating each source with a recovery/rash-onset window (a recovery-time leaf, noted
in the refinements below) would shrink the denominator and move ``\lambda_0`` and
``R_\text{household}`` into a measles-plausible range; that is out of scope here,
so read the value below as a lower bound that demonstrates the calculation rather
than a calibrated estimate.
"""

mean_hh_susc = n / length(unique(household)) - 1
window = 8.0                            # measles infectious window (days)
log_lambda0_chain = vec(chain[:log_lambda0])
beta_hh_chain = beta_chains[1]          # household is the first covariate
R_household_samples = map(eachindex(log_lambda0_chain)) do s
    rate_within = exp(log_lambda0_chain[s] + beta_hh_chain[s])
    cdf(contact_interval(rate_within, gamma_chain[s]), window) * mean_hh_susc
end
(; R_household_mean = mean(R_household_samples),
    R_household_90 = quantile(R_household_samples, (0.05, 0.95)))

md"""
## Compare to the who-infected-whom truth

Hagelloch records a *putative* infector for each case, reconstructed from the
outbreak investigation.
We never used it in the fit — the racing-hazard marginal marginalises over the
unknown source — but we can check that the fitted contact intervals assign high
``\arg\min`` probability to the recorded infector.

The racing-hazard node's [`winning_probabilities`](@ref) integrates the
cause-resolved split over a *shared* support floor, which assumes every cause
can fire from the same earliest time; with anchored sources that floor differs
per cause, so we read the ``\arg\min`` split by Monte Carlo instead — drawing a
latent contact time from each anchored source and counting how often each
source's contact is the earliest.
This is the generative dual of the racing marginal (and the basis of
[`winning_probabilities`](@ref) for a shared-floor node).
For each case with a recorded infector we estimate the probability the model
places on the *recorded* source winning the race.
"""

function argmin_source_probs(rng, node_delays, draws)
    n = length(node_delays)
    wins = zeros(Int, n)
    for _ in 1:draws
        best_i, best_t = 1, rand(rng, node_delays[1])
        for i in 2:n
            t = rand(rng, node_delays[i])
            t < best_t && ((best_i, best_t) = (i, t))
        end
        wins[best_i] += 1
    end
    return wins ./ draws
end

function recorded_infector_mass(
        rng, post, cases, onset, household, class; draws = 2000)
    id_to_pos = Dict(id => k for (k, id) in enumerate(cases.case_ID))
    masses = Float64[]
    for j in 1:nrow(cases)
        inf_id = cases.infector[j]
        (ismissing(inf_id) || !haskey(id_to_pos, inf_id)) && continue
        tj = onset[j]
        sources = [i for i in eachindex(onset) if i != j && onset[i] < tj]
        isempty(sources) && continue
        delays = source_delays(post.log_lambda0, post.beta, post.gamma,
            onset, household, class, sources, j)
        probs = argmin_source_probs(rng, delays, draws)
        k = findfirst(==(id_to_pos[inf_id]), sources)
        isnothing(k) && continue
        push!(masses, probs[k])
    end
    return masses
end

masses = recorded_infector_mass(rng, post, cases, onset, household, class)
chance = [1 / n_sources_at_risk(onset, j)
          for j in eachindex(onset) if n_sources_at_risk(onset, j) > 0]
(; n_pairs_checked = length(masses),
    mean_recorded_infector_prob = mean(masses),
    chance_baseline = mean(chance))

md"""
The model places more ``\arg\min`` mass on the *recorded* infector than the
chance baseline (one over the number of at-risk sources): the fitted
contact-hazard regression, driven only by timing and the household/class
covariates and never shown the recorded tree, points at the recorded infector
several times more often than chance.
This is a *minority* of the mass, not a confident reconstruction, since the same
un-gated at-risk window that biases ``R_\text{household}`` low also leaves many
old sources competing for each susceptible, so read it as evidence that the
timing-and-covariate signal is informative about who-infected-whom, not as a
recovered transmission tree.

## Mapping back to `transtat`

This page is a faithful but deliberately compact rendering of Kenah's pairwise
survival method [kenah2011contact](@cite) on the composer stack:

- the **contact interval** is a stock `Weibull` baseline whose rate carries
  Kenah's log-rate regression ``\lambda_0\, e^{\beta' x_{ij}}``, so the
  transmission coefficients ``\beta`` are estimated directly; because the racing
  node needs only a `logpdf` and a `logccdf`, any richer baseline with an
  AD-safe survival (a [SurvivalDistributions.jl](@ref survival-delay-families)
  leaf, a piecewise-constant hazard) drops in;
- the **racing across sources** is the racing-hazard [`compete`](@ref) node,
  whose cause-resolved marginal is the pairwise likelihood and whose
  ``\arg\min`` draws give the who-infected-whom posterior;
- the **right-censoring** of non-winning pairs is the racing survival
  ``\prod_k S_k``, the no-event branch of the compete likelihood.

The page stays compact, leaving several `transtat` features for the reader to
add on the same likelihood structure: an infectious-period leaf that gates each
source's survival term, an external community hazard as an extra racing branch,
more pair covariates as a longer ``x_{ij}``, left-truncation as a conditioning
survival, and day-resolution censoring by wrapping each anchored leaf in
[`double_interval_censored`](@ref).
"""
