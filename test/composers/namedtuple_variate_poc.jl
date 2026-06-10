# PROOF OF CONCEPT tests for the NamedTuple-valued composed variate
# (RFC / do-not-merge). These assert the variate BASICS that work: `rand`
# returns a NamedTuple, `logpdf` overlays by name, `mean` is a NamedTuple, and
# the hand-written array `rand` avoids the stack-overflow trap. The bijector /
# ForwardDiff / product_distribution consequences are recorded in the RFC PR
# body (they break), not asserted here.

@testitem "NamedTuple variate: rand / logpdf / mean basics" begin
    using Distributions, Random

    seq = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
    d = CensoredDistributions.ComposedNamedTuple(seq)
    rng = MersenneTwister(1)

    draw = rand(rng, d)
    @test draw isa NamedTuple
    @test keys(draw) == CensoredDistributions.tree_event_names(seq)

    # logpdf overlays by name and matches the vector path it overlays.
    ev = Vector{Union{Missing, Float64}}(collect(values(draw)))
    @test logpdf(d, draw) ≈ CensoredDistributions.event_logpdf(seq, ev)

    # Order-independence: a reordered NamedTuple scores identically.
    rev = (event_3 = draw.event_3, event_1 = draw.event_1,
        event_2 = draw.event_2)
    @test logpdf(d, rev) ≈ logpdf(d, draw)

    # mean is a NamedTuple keyed the same way as a draw.
    @test mean(d) isa NamedTuple
    @test keys(mean(d)) == keys(draw)
end

@testitem "NamedTuple variate: hand-written array rand" begin
    using Distributions, Random

    seq = Sequential(
        primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1)))
    d = CensoredDistributions.ComposedNamedTuple(seq)
    rng = MersenneTwister(2)

    # Distributions.jl has no array fallback for a NamedTuple variate; the
    # hand-written `Dims` and exact-`Int` methods must both work. The `Int`
    # method exists to win dispatch over Distributions' ambiguous varargs
    # `rand(rng, ::Sampleable, ::Int, ::Int...)` (the JuliaBUGS trap).
    a = rand(rng, d, 3)
    @test a isa Vector{<:NamedTuple}
    @test length(a) == 3

    m = rand(rng, d, (2, 2))
    @test size(m) == (2, 2)
    @test eltype(m) <: NamedTuple
end
