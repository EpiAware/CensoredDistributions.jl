# Light prototype: `primary_censored(::Vector, pe)` = sequential chain.
#
# Proves the unified-engine overload composes by delegating to the existing
# `SequentialDistribution` machinery, without removing any existing type.
# See `docs/design/unified-primary-censored.md` for the full design.
#
# Run from the package root:
#   julia --project=. docs/design/prototype_unified_primary_censored.jl
#
# This file is a design artefact, not part of the package; it defines the
# overload locally on the existing constructor so the engine is untouched.

using CensoredDistributions
using Distributions
using Random
using Test

# The proposed overload. In the real engine this would live in
# `src/censoring/PrimaryCensored.jl` and replace #317's parallel vector
# overload (the #317 flip). Here it is defined locally to keep the package
# unchanged: a `Vector` of delays with a primary event lowers to the existing
# `SequentialDistribution`, carrying the primary event as the origin censoring.
function prototype_sequential(delays::AbstractVector, primary_event; kwargs...)
    sequential_distribution(delays; primary_event = primary_event, kwargs...)
end

Random.seed!(20260605)

D1 = Gamma(2.0, 1.0)
D2 = LogNormal(0.5, 0.4)
D3 = Weibull(2.0, 1.5)

@testset "Vector overload delegates to SequentialDistribution" begin
    d = prototype_sequential([D1, D2, D3], Uniform(0.0, 1.0))
    @test d isa CensoredDistributions.SequentialDistribution
    @test length(d) == 3
    @test d.primary_event == Uniform(0.0, 1.0)
end

@testset "Observed intermediate conditions (independent factors)" begin
    # No primary censoring keeps the maths transparent: with every event
    # observed the joint is the product of single-delay factors.
    d = sequential_distribution([D1, D2, D3])
    obs = [0.0, 1.0, 3.0, 5.0]            # E0, E1, E2, E3 all observed
    expected = logpdf(D1, 1.0) + logpdf(D2, 2.0) + logpdf(D3, 2.0)
    @test logpdf(d, obs) ≈ expected
end

@testset "Missing intermediate marginalises by convolution" begin
    # E1 (index 2) is unobserved: the first observed gap E2 - E0 spans the
    # run D1, D2, marginalised as the convolution; the second gap is D3.
    d = sequential_distribution([D1, D2, D3])
    g1 = 3.0                              # E2 - E0
    g2 = 2.0                             # E3 - E2
    obs = [0.0, missing, g1, g1 + g2]
    run = convolve_distributions(D1, D2)
    expected = logpdf(run, g1) + logpdf(D3, g2)
    @test logpdf(d, obs) ≈ expected
    # Marginalising a continuous unobserved intermediate IS the convolution:
    # the marginalised run's term equals the convolution density at the gap.
    @test logpdf(d, obs) - logpdf(D3, g2) ≈ logpdf(run, g1)
end

@testset "rand produces the full path including unobserved intermediate" begin
    d = sequential_distribution([D1, D2, D3])
    path = rand(d)
    @test length(path) == 4              # E0..E3, the intermediate included
    @test path[1] == 0.0                 # origin at time zero
    @test issorted(path)                 # event times non-decreasing
end

println("prototype: all checks passed — Vector overload composes, ",
    "marginalise-by-convolution and rand-full-path hold.")
