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
method, recover the contact-interval parameters and ``R_0``, and read off a
within-household hazard ratio.

1. Map each Kenah pairwise concept onto a composed primitive:

| Kenah pairwise concept | Composed primitive |
|---|---|
| contact interval ``\tau_{ij}`` with hazard ``\lambda(\tau)`` | a custom in-doc `UnivariateDistribution` leaf |
| susceptible ``j`` infected at ``\min_i \tau_{ij}``, source ``=\arg\min`` | racing-hazard [`competing`](@ref) ([`HazardCompeting`](@ref)) |
| pair right-censored (``j`` infected elsewhere / source recovers / study ends) | the racing-hazard survival ``\prod_k S_k`` (`logccdf`) |
| contact-interval observation windows (dates to the day) | a [`double_interval_censored`](@ref) leaf (sketched in the refinements) |
| covariate hazard ratios (within vs between household) | a per-pair scale on the leaf |

2. Define the contact-interval distribution as a custom hazard leaf.
3. Build the pairwise survival likelihood as a racing-hazard competing node.
4. Fit the Hagelloch data and recover the parameters, ``R_0`` and the
   within-household hazard ratio.

### What might I need to know before starting

This tutorial builds on [Getting Started with
CensoredDistributions.jl](@ref getting-started) and the composer reference,
[Composing censored distributions](@ref composer-toolkit).

## The method

The contact-interval distribution we use is the `transtat` Weibull
parameterisation [kenah2011contact](@cite): a rate ``\lambda`` and a shape
``\gamma`` with cumulative hazard

```math
H(\tau) = (\lambda \tau)^\gamma,
\qquad S(\tau) = e^{-(\lambda \tau)^\gamma},
\qquad h(\tau) = \lambda \gamma (\lambda \tau)^{\gamma - 1}.
```

``\gamma > 1`` gives an increasing hazard of infectious contact (infectiousness
that builds after onset), ``\gamma < 1`` a decreasing one.
This is identical to a `Distributions.Weibull(γ, 1/λ)`, but we define it from
its hazard as a custom leaf — it doubles as a worked example of the
*user-extensibility* of the composer stack: any `UnivariateDistribution` that
reports a `logpdf` and a `logccdf` is a valid cause-specific delay for the
racing-hazard node, no new package feature required.

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
[`competing`](@ref) node over ``R(j)``.
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
using ADTypes: AutoForwardDiff
import Distributions: logpdf, logccdf, pdf, cdf, ccdf, quantile
import Base: minimum, maximum, rand

md"""
## The contact-interval leaf

The contact interval is a custom `UnivariateDistribution` defined from its
cumulative hazard ``H(\tau) = (\lambda\tau)^\gamma``.
We give it the survival functions the racing-hazard node needs
(`logccdf`/`ccdf`), the density (`logpdf`/`pdf`), and `quantile`/`rand` for
simulation.
Everything is written through `log1p`/`expm1`/`log` so the leaf is AD-safe and
differentiates with respect to ``\lambda`` and ``\gamma`` under Turing.
"""

struct ContactInterval{T <: Real} <: ContinuousUnivariateDistribution
    "Rate parameter ``\\lambda`` of the contact-interval hazard."
    lambda::T
    "Shape parameter ``\\gamma`` of the contact-interval hazard."
    gamma::T
end

minimum(::ContactInterval) = 0.0
maximum(::ContactInterval) = Inf

## Cumulative hazard H(τ) = (λτ)^γ and the survival S = exp(-H).
_cumhazard(d::ContactInterval, t::Real) = (d.lambda * t)^d.gamma
function logccdf(d::ContactInterval, t::Real)
    return t <= 0 ? zero(float(t)) : -_cumhazard(d, t)
end
ccdf(d::ContactInterval, t::Real) = exp(logccdf(d, t))
cdf(d::ContactInterval, t::Real) = -expm1(logccdf(d, t))

## Density f(τ) = h(τ) S(τ) with hazard h(τ) = λγ(λτ)^(γ-1).
function logpdf(d::ContactInterval, t::Real)
    t <= 0 && return oftype(float(t), -Inf)
    loghazard = log(d.lambda) + log(d.gamma) +
                (d.gamma - 1) * log(d.lambda * t)
    return loghazard - _cumhazard(d, t)
end
pdf(d::ContactInterval, t::Real) = exp(logpdf(d, t))

## Inverse-cdf draw: H(τ) = -log(1-p) ⇒ τ = (1/λ)(-log(1-p))^(1/γ).
function quantile(d::ContactInterval, p::Real)
    return (1 / d.lambda) * (-log1p(-p))^(1 / d.gamma)
end
rand(rng::AbstractRNG, d::ContactInterval) = quantile(d, rand(rng))

md"""
The leaf is just a `Distributions.Weibull(γ, 1/λ)` written from its hazard, so
we can check it against the stock distribution: the density, survival, and
quantiles agree to machine precision.
"""

let d = ContactInterval(0.4, 1.6), w = Weibull(1.6, 1 / 0.4)
    (; dlogpdf = logpdf(d, 3.0) - logpdf(w, 3.0),
        dlogccdf = logccdf(d, 3.0) - logccdf(w, 3.0),
        dquantile = quantile(d, 0.7) - quantile(w, 0.7))
end

md"""
## A racing-hazard node over sources

When several sources became infectious *at the same time*, the susceptible
faces a racing hazard on a shared clock: a [`competing`](@ref) node built from
*bare* contact-interval delays (no branch probabilities), which selects the
racing-hazard [`HazardCompeting`](@ref) type — the winning source is *derived*
from the hazards, not a free parameter.
The node's `logpdf` is the cause-resolved marginal density
``\sum_i f(\tau) \prod_{k \ne i} S(\tau)``, its `logccdf` is the joint
survival ``\prod_k S_k``, and [`winning_probabilities`](@ref) is the per-source
``\arg\min`` (who-infected-whom) split.
"""

let node = competing(:near => ContactInterval(0.4, 1.6),
        :far => ContactInterval(0.1, 1.6))
    (; marginal_logpdf = logpdf(node, 2.0),
        joint_logsurvival = logccdf(node, 2.0),
        winning = winning_probabilities(node))
end

md"""
## Anchoring a source's contact interval on the outbreak clock

In a real outbreak the sources do *not* share a clock: each source ``i`` starts
its own contact-interval clock at its infectiousness onset ``o_i``, so on the
*outbreak* clock source ``i``'s contact time is ``o_i + \tau``.
`Anchored` shifts a contact-interval leaf by a source's onset, a second tiny
custom leaf: evaluating it at an outbreak time ``t`` reads the underlying leaf
at the gap ``t - o_i``.
This lets a racing-hazard node over sources be scored at the *susceptible's*
infection time directly, with every source automatically read at its own gap
``g_{ij} = t_j - o_i`` — the node's `logpdf` at ``t_j`` is then exactly the
cause-resolved pairwise density ``\sum_i f(g_{ij}) \prod_{k \ne i} S(g_{kj})``.
"""

struct Anchored{D <: UnivariateDistribution, T <: Real} <:
       ContinuousUnivariateDistribution
    "The contact-interval leaf measured from the source's onset."
    leaf::D
    "The source's infectiousness-onset time on the outbreak clock."
    onset::T
end

minimum(d::Anchored) = d.onset + minimum(d.leaf)
maximum(d::Anchored) = d.onset + maximum(d.leaf)
logpdf(d::Anchored, t::Real) = logpdf(d.leaf, t - d.onset)
pdf(d::Anchored, t::Real) = exp(logpdf(d, t))
logccdf(d::Anchored, t::Real) = logccdf(d.leaf, t - d.onset)
ccdf(d::Anchored, t::Real) = exp(logccdf(d, t))
cdf(d::Anchored, t::Real) = cdf(d.leaf, t - d.onset)
quantile(d::Anchored, p::Real) = d.onset + quantile(d.leaf, p)
rand(rng::AbstractRNG, d::Anchored) = d.onset + rand(rng, d.leaf)

md"""
Two sources with different onsets, scored at the susceptible's infection time:
the node's marginal `logpdf` reads each source at its own gap, and the joint
survival is the product of the per-source survivals.
"""

let onset_near = 1.0, onset_far = 2.5, t_infect = 6.0
    node = competing(:near => Anchored(ContactInterval(0.4, 1.6), onset_near),
        :far => Anchored(ContactInterval(0.1, 1.6), onset_far))
    (; marginal_logpdf = logpdf(node, t_infect),
        joint_logsurvival = logccdf(node, t_infect))
end

md"""
## The Hagelloch line list

The 1861 Hagelloch measles outbreak (n = 188 children) is the standard test
bed for transmission survival methods [kenah2011contact, neal2004statistical](@cite):
every case carries a household (`family_ID`), a putative infector
(`infector`), and dated symptom milestones.
We take the **date of prodrome** as the onset of infectiousness — the event
that starts each source's contact-interval clock — and measure time in days
from the first prodrome in the outbreak.
"""

datadir = joinpath(@__DIR__, "hagelloch-data")

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
T_end = maximum(onset)
(; T_end, mean_household_size = n / length(unique(household)))

md"""
## The pairwise likelihood

For each infected susceptible ``j`` we assemble its at-risk source set
``R(j)`` — every case whose infectiousness onset precedes ``t_j`` — anchor each
source's contact interval at its onset, and score ``j`` through a racing-hazard
[`competing`](@ref) node over those anchored sources, evaluated at ``j``'s
infection time ``t_j``.
The contact-interval scale depends on whether the pair shares a household: a
within-household rate ``\lambda_w`` and a between-household rate
``\lambda_b``, with a shared shape ``\gamma``.
Their ratio is the **within-household hazard ratio**, the central covariate
estimand.

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

function source_scale(lambda_w, lambda_b, household, i, j)
    return household[i] == household[j] ? lambda_w : lambda_b
end

function source_delays(lambda_w, lambda_b, gamma, onset, household, sources, j)
    return Tuple(
        Anchored(
            ContactInterval(
                source_scale(lambda_w, lambda_b, household, i, j), gamma),
            onset[i])
    for i in sources)
end

function source_node(lambda_w, lambda_b, gamma, onset, household, sources, j)
    delays = source_delays(
        lambda_w, lambda_b, gamma, onset, household, sources, j)
    names = ntuple(k -> Symbol(:src, k), length(sources))
    return competing((names[k] => delays[k] for k in eachindex(sources))...)
end

## Count the sources infectious before j's own infection (j's at-risk set size).
function n_sources_at_risk(onset, j)
    return count(i -> i != j && onset[i] < onset[j], eachindex(onset))
end

md"""
The at-risk structure — which sources can infect each susceptible, the gap to
each, and whether the pair shares a household — does not depend on the
parameters, so we precompute it once.
`atrisk_pairs` returns, per infected susceptible, the vector of source gaps
``g_{ij} = t_j - o_i`` and a matching within-household flag; the seed case (no
prior source) is dropped.
"""

function atrisk_pairs(onset, household)
    pairs = Vector{Tuple{Vector{Float64}, Vector{Bool}}}()
    for j in eachindex(onset)
        tj = onset[j]
        gaps = Float64[]
        within = Bool[]
        for i in eachindex(onset)
            (i == j || onset[i] >= tj) && continue
            push!(gaps, tj - onset[i])
            push!(within, household[i] == household[j])
        end
        isempty(gaps) || push!(pairs, (gaps, within))
    end
    return pairs
end

pairs = atrisk_pairs(onset, household)
(; n_susceptibles_scored = length(pairs),
    n_ordered_pairs = sum(length(first(p)) for p in pairs))

md"""
For one susceptible the racing-hazard marginal at ``t_j`` is the log-sum-exp of
each source's cause-resolved log sub-density
``\log f(g_{ij}) + \sum_{k \ne i} \log S(g_{kj})``.
This is exactly `logpdf(source_node(...), t_j)`, written out directly here so the
likelihood evaluates the contact-interval leaf functions without rebuilding a
composer node on every gradient step (the closed Hagelloch fit scores every
ordered at-risk pair).
The within-household flag picks the rate ``\lambda_w`` or ``\lambda_b``; the
shared shape ``\gamma`` carries the hazard's time shape.
"""

function susceptible_loglik(lambda_w, lambda_b, gamma, gaps, within)
    leaf(k) = ContactInterval(within[k] ? lambda_w : lambda_b, gamma)
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
The direct computation agrees with the racing-hazard `competing` node it stands
in for. For one Hagelloch susceptible with several at-risk sources we build the
node with `source_node` and check its marginal at ``t_j`` against
`susceptible_loglik`.
"""

let j = findfirst(j -> n_sources_at_risk(onset, j) >= 3, eachindex(onset)),
    srcs = [i for i in eachindex(onset) if i != j && onset[i] < onset[j]]

    node = source_node(0.2, 0.05, 1.3, onset, household, srcs, j)
    gaps = [onset[j] - onset[i] for i in srcs]
    within = [household[i] == household[j] for i in srcs]
    (; n_sources = length(srcs),
        node_logpdf = logpdf(node, onset[j]),
        direct_logpdf = susceptible_loglik(0.2, 0.05, 1.3, gaps, within))
end

md"""
The two agree (the composer node and the direct reduction are the same
likelihood), so the fit can use the fast direct form. The full likelihood sums
the per-susceptible term over the precomputed at-risk pairs.
"""

function pairwise_loglik(lambda_w, lambda_b, gamma, pairs)
    lp = zero(lambda_w)
    for (gaps, within) in pairs
        lp += susceptible_loglik(lambda_w, lambda_b, gamma, gaps, within)
    end
    return lp
end

md"""
The racing-hazard node's marginal `logpdf` sums the cause-resolved sub-densities
over the sources, so we do not need to know which source actually won to fit the
contact-interval parameters; the unknown infector is marginalised.

The right-censoring is implicit in the racing marginal: every source that
*could* have infected ``j`` but did not contributes its survival ``S`` inside
the same node, so every at-risk pair that did **not** win is already a survival
term — the no-event branch of the racing-hazard likelihood.
"""

md"""
## Fit with Turing

We put weakly-informative positive priors on the two rates and the shape and
score the whole line list with `Turing.@addlogprob!`.
The shape prior is centred at one (a constant hazard, the exponential special
case) so the data drive any departure.
"""

@model function hagelloch_pairwise(pairs)
    lambda_w ~ truncated(Normal(0.2, 0.2); lower = 1e-3)
    lambda_b ~ truncated(Normal(0.03, 0.05); lower = 1e-3)
    gamma ~ truncated(Normal(1.0, 0.5); lower = 0.2)
    Turing.@addlogprob! pairwise_loglik(lambda_w, lambda_b, gamma, pairs)
end

rng = MersenneTwister(2024)
chain = sample(rng, hagelloch_pairwise(pairs),
    NUTS(0.8; adtype = AutoForwardDiff()), 400; progress = false)

md"""
The posterior summaries for the contact-interval parameters:
"""

post = (; lambda_w = mean(chain[:lambda_w]),
    lambda_b = mean(chain[:lambda_b]),
    gamma = mean(chain[:gamma]))

md"""
## Estimands: hazard ratio and ``R_0``

The **within-household hazard ratio** is the ratio of the two rates raised to
the shape, ``(\lambda_w / \lambda_b)^\gamma`` for the cumulative-hazard
parameterisation — children in the same household experience a much higher
hazard of infectious contact than between households.
"""

hr_samples = (vec(chain[:lambda_w]) ./ vec(chain[:lambda_b])) .^
             vec(chain[:gamma])
(; hazard_ratio_mean = mean(hr_samples),
    hazard_ratio_90 = quantile(hr_samples, (0.05, 0.95)))

md"""
``R_0`` in the pairwise framework is the expected number of infectious contacts
a case makes with *susceptible* others over its infectious period
[kenah2011contact](@cite).
For each susceptible the per-pair probability of an infectious contact within an
infectious window of length ``w`` is the contact-interval cdf ``F(w)``, so the
**household reproduction number** — the expected secondary infections within a
case's own household — is ``F_w(w)`` (within-household rate) times the mean
number of household susceptibles.
We take a measles-typical infectious window of `w = 8` days
[neal2004statistical](@cite); a community ``R_0`` would add the between-household
contacts over the larger susceptible pool, governed by ``\lambda_b``.
"""

mean_hh_susc = n / length(unique(household)) - 1
window = 8.0                            # measles infectious window (days)
R_household_samples = map(eachindex(vec(chain[:lambda_w]))) do s
    lw = vec(chain[:lambda_w])[s]
    g = vec(chain[:gamma])[s]
    cdf(ContactInterval(lw, g), window) * mean_hh_susc
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
        rng, post, cases, onset, household; draws = 2000)
    id_to_pos = Dict(id => k for (k, id) in enumerate(cases.case_ID))
    masses = Float64[]
    for j in 1:nrow(cases)
        inf_id = cases.infector[j]
        (ismissing(inf_id) || !haskey(id_to_pos, inf_id)) && continue
        tj = onset[j]
        sources = [i for i in eachindex(onset) if i != j && onset[i] < tj]
        isempty(sources) && continue
        delays = source_delays(post.lambda_w, post.lambda_b, post.gamma,
            onset, household, sources, j)
        probs = argmin_source_probs(rng, delays, draws)
        k = findfirst(==(id_to_pos[inf_id]), sources)
        isnothing(k) && continue
        push!(masses, probs[k])
    end
    return masses
end

masses = recorded_infector_mass(rng, post, cases, onset, household)
chance = [1 / n_sources_at_risk(onset, j)
          for j in eachindex(onset) if n_sources_at_risk(onset, j) > 0]
(; n_pairs_checked = length(masses),
    mean_recorded_infector_prob = mean(masses),
    chance_baseline = mean(chance))

md"""
The model places more ``\arg\min`` mass on the *recorded* infector than the
chance baseline (one over the number of at-risk sources), so the fitted
contact-interval distribution — driven only by timing and household structure —
recovers a transmission tree consistent with the outbreak reconstruction
without ever being shown it.

## Mapping back to `transtat`

This page is a faithful but deliberately compact rendering of Kenah's pairwise
survival method [kenah2011contact](@cite) on the composer stack:

- the **contact interval** is a custom hazard leaf, so any parametric or
  semiparametric hazard family (Weibull, log-logistic, a
  [SurvivalDistributions.jl](@ref survival-delay-families) leaf, a
  piecewise-constant hazard) drops in unchanged;
- the **racing across sources** is the racing-hazard [`competing`](@ref) node,
  whose cause-resolved marginal is the pairwise likelihood and whose
  ``\arg\min`` draws give the who-infected-whom posterior;
- the **right-censoring** of non-winning pairs is the racing survival
  ``\prod_k S_k``, the no-event branch of the competing likelihood.

Compared with the `transtat` reference implementation, several refinements are
left for the maintainer's specific integration:

- `transtat` separates the **infectious period** from the contact interval, so
  a source stops contributing hazard once it recovers; here the at-risk window
  is closed only by the study horizon. A recovery-time leaf
  (`infectiousness onset → recovery`) would gate each source's survival term.
- The **external / community hazard** (infection from outside the close-contact
  groups) is a further competing source with its own constant hazard, scored as
  an extra racing branch per susceptible.
- `transtat` fits an **accelerated failure time** regression with arbitrary
  covariates on the contact-interval scale; the household indicator here is the
  simplest such covariate and generalises to age, class, and spatial distance
  via the same per-pair scale.
- The **left-truncation** of pairs observed only after the source became
  infectious enters `transtat`'s likelihood as a conditioning survival; the
  closed-outbreak Hagelloch data sidestep it, but a real-time line list would
  need it.
- [`winning_probabilities`](@ref) integrates the cause-resolved split over a
  *shared* support floor, so for the anchored (per-source-clock) sources here we
  read the who-infected-whom split by Monte Carlo ``\arg\min`` instead. A
  composer that integrated each cause from its own support floor would give the
  derived split directly; that is a natural extension of the racing-hazard node
  for staggered onsets.
- The **contact-interval censoring** (infection dates recorded to the day) is
  left as a point evaluation here for clarity; wrapping each anchored leaf in a
  [`double_interval_censored`](@ref) primary/secondary window would carry the
  date uncertainty into the likelihood, the core CensoredDistributions surface.

These are the places where @seabbs's integration of the racing-hazard composer
with the contact-interval framework should refine this draft.
"""
