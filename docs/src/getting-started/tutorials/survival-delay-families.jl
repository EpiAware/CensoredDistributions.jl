# # [Delay families from SurvivalDistributions.jl](@id survival-delay-families)
#
# ## Introduction
#
# CensoredDistributions.jl treats any `UnivariateDistribution` as a delay leaf.
# That contract is satisfied by more than the families in Distributions.jl.
# [SurvivalDistributions.jl](https://github.com/JuliaSurv/SurvivalDistributions.jl)
# adds parametric delay families with hazard and accelerated-failure-time
# parameterisations (LogLogistic, GeneralizedGamma, PowerGeneralizedWeibull,
# ExponentiatedWeibull) and a piecewise-constant hazard distribution.
# They subtype `ContinuousUnivariateDistribution` and define
# `logpdf`/`cdf`/`rand`, and they already declare support on `[0, Inf)`, so they
# drop into the composed stack as leaves with no extra code.
#
# ### What are we going to do in this exercise
#
# 1. Use a SurvivalDistributions.jl family (LogLogistic) as a plain delay leaf.
# 2. Double-interval-censor that survival leaf.
# 3. Compose it into a record alongside a stock Distributions.jl delay.
# 4. Convolve it with another delay.
#
# ### What might I need to know before starting
#
# This tutorial builds on [Getting Started with
# CensoredDistributions.jl](@ref getting-started) and the composer reference,
# [Composing censored distributions](@ref composer-toolkit).

# ## Packages used
#
# We use SurvivalDistributions for the delay families and Distributions for the
# primary-event distribution and a stock leaf to compose against.
# SurvivalDistributions exports some names that clash with Distributions (e.g.
# `LogLogistic`), so it is imported qualified as `SD`.

using CensoredDistributions
using Distributions
import SurvivalDistributions as SD

# ## A survival family as a delay leaf
#
# A [LogLogistic](https://github.com/JuliaSurv/SurvivalDistributions.jl) delay
# is a plain leaf: it has a density and a cumulative distribution function over
# `[0, Inf)`, the support a delay needs.

delay = SD.LogLogistic(1.0, 2.0)

# The leaf reports its support, so the censoring and composition machinery can
# bound its quadrature windows.

minimum(delay), maximum(delay)

# ## Double-interval censoring a survival leaf
#
# A reporting delay is rarely observed exactly: the primary event falls in a
# window and the secondary event is recorded to an interval.
# [`double_interval_censored`](@ref) wraps the leaf with a primary-event
# distribution and a secondary interval, exactly as it does for a stock
# Distributions.jl delay.

censored = double_interval_censored(
    delay; primary_event = Uniform(0, 1), interval = 1.0)

# Censoring shifts the CDF relative to the bare survival leaf. At each day the
# censored CDF differs from the base `LogLogistic` CDF because the primary event
# is spread over its window and the secondary event is binned to the day.

xs = 0.0:5.0
[(x = x, base = round(cdf(delay, x); digits = 3),
     censored = round(cdf(censored, x); digits = 3)) for x in xs]

# ## A GeneralizedGamma leaf inside a composed record
#
# [`compose`](@ref) builds a record from per-event delays.
# A survival family composes the same way as a Distributions.jl family: here an
# onset-to-admission delay drawn from a GeneralizedGamma and an
# onset-to-notification delay from a Gamma, each double-interval-censored.

dic(x) = double_interval_censored(
    x; primary_event = Uniform(0, 1), interval = 1.0)

record = compose((
    onset_admit = dic(SD.GeneralizedGamma(1.0, 1.5, 2.0)),
    onset_notif = dic(Gamma(1.5, 1.0))))

# The composed object names its events like any other record.

event_names(record)

# It both scores an event record and simulates one.

rand(record)

# ## Convolving survival delays
#
# [`convolve_distributions`](@ref) forms the distribution of a sum of
# independent delays.
# A survival family convolves with another delay through the numeric quadrature
# path, which uses the leaves' `[0, Inf)` support.

total = convolve_distributions(
    SD.GeneralizedGamma(1.0, 1.5, 2.0), LogNormal(0.5, 0.4))

# The convolution has a monotone CDF over the combined support.

[round(cdf(total, x); digits = 3) for x in 0.0:2.0:10.0]

# ## Hazards from a composed tree
#
# The survival families exist for hazard modelling, and the package reads the
# hazard surface off ANY composed delay through the verbs (north-star tenet 5:
# the composed object is the input to downstream layers). The four accessors
# [`hazard`](@ref CensoredDistributions.hazard),
# [`loghazard`](@ref CensoredDistributions.loghazard),
# [`cumhazard`](@ref CensoredDistributions.cumhazard) and
# [`survival`](@ref CensoredDistributions.survival) are the standard
# survival-analysis identities (`h = f/S`, `H = -log S`), so they read a leaf,
# a censored wrapper, a [`modify`](@ref) hazard modification, or a composer.
#
# A SurvivalDistributions leaf reports its own hazard.

CensoredDistributions.hazard(SD.GeneralizedGamma(1.0, 1.5, 2.0), 2.0)

# The accessor matches SurvivalDistributions' own `hazard`, so the two
# interoperate: a tree of survival leaves and a survival leaf in a tree both
# read through either function.

CensoredDistributions.hazard(SD.GeneralizedGamma(1.0, 1.5, 2.0), 2.0) ==
SD.hazard(SD.GeneralizedGamma(1.0, 1.5, 2.0), 2.0)

# Getting the hazard from a TREE goes through the verbs. A
# [`sequential`](@ref) chain's total-time hazard is the hazard of its marginal
# convolution, reachable from either package's function name.

chain = sequential(SD.GeneralizedGamma(1.0, 1.5, 2.0), LogNormal(0.5, 0.4))
[round(CensoredDistributions.hazard(chain, x); digits = 4) for x in 1.0:2.0:9.0]

# A [`compete`](@ref) racing node's hazard is the total cause hazard (the rate
# of the first event), the natural input to a competing-risks layer.

race = compete(:recover => SD.GeneralizedGamma(1.0, 1.5, 2.0),
    :die => Weibull(1.4, 3.0))
[round(CensoredDistributions.hazard(race, x); digits = 4) for x in 1.0:1.0:5.0]

# A [`modify`](@ref) hazard modification reads back the hazard it constructed,
# `g⁻¹(g(h) + effect)`: a proportional-hazards (`log` link) doubling scales the
# base hazard by two.

base = SD.GeneralizedGamma(1.0, 1.5, 2.0)
doubled = modify(base, log(2.0); link = log)
[(t = t, base = round(CensoredDistributions.hazard(base, t); digits = 4),
     doubled = round(CensoredDistributions.hazard(doubled, t); digits = 4))
 for t in 1.0:2.0:5.0]

# ## When to reach for these families
#
# The survival families add delay shapes Distributions.jl does not cover: the
# LogLogistic and GeneralizedGamma give heavier or more flexible tails than a
# Gamma or LogNormal, useful for incubation, generation, and reporting delays.
# Because they meet the leaf contract, they compose, censor, and convolve with
# the rest of the toolkit with no new code.
#
# Two caveats hold for SurvivalDistributions v0.1.
# The LogLogistic and ExponentiatedWeibull `logpdf`s are not yet
# differentiable (an upstream `Float64` conversion), so a model fit by gradient
# methods should prefer GeneralizedGamma.
# It is the survival family the package routes through an AD-safe CDF, so its
# leaf and censored `logpdf`s are tested for gradient correctness against a
# ForwardDiff reference.
# The other survival families have no such AD-safe routing, so their gradients
# are unverified.
# The SurvivalDistributions.jl piecewise-constant hazard works as a bare leaf
# but its `logcdf` throws upstream, so it cannot yet route the numeric censoring
# quadrature.
# For a flexible hazard with an AD-safe survival, the package ships [`modify`](@ref),
# which modifies the hazard of a base delay through a link, shown in the
# [composer toolkit](@ref composer-toolkit).
