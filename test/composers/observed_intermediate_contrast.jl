# Contrast between the two lowerings of the SAME `Sequential` chain (#329/#333).
#
# 1. The general per-record event-vector scoring
#    (`logpdf(::Sequential, events)`, the path rows flow through via
#    `composed_distribution_model`): an event vector with one entry per event
#    `[E_0, E_1, ..., E_k]`. With every intermediate OBSERVED the chain
#    FACTORISES into one independent per-edge term at each observed gap.
#
# 2. The combine-then-censor scalar lowering (`observed_distribution`,
#    `test/composers/wrap.jl`): the chain collapses to the single `Convolved`
#    total time origin -> terminal event, with the intermediate MARGINALISED.
#
# These are genuinely different numbers. This test pins the semantic the
# maintainer was worried about: observed intermediates must factorise, not
# collapse into the marginal convolution.

@testitem "observed intermediate factorises, not the collapsed marginal" begin
    using Distributions

    seq = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    edge1 = Gamma(2.0, 1.0)
    edge2 = LogNormal(0.5, 0.4)

    e_j = 1.7   # observed first event time
    e_k = 4.3   # observed terminal event time

    # The event vector rows flow through is `Missing`-admitting (the element
    # type selects the per-record specialisation; a plain `Vector{Float64}` of
    # length 3 would instead hit the generic per-step `logpdf` and error). Here
    # E_0 is the origin (a plain delay chain has no primary, so the origin sits
    # at the observed 0.0), E_1 = e_j observed, E_2 = e_k observed.
    events = Vector{Union{Missing, Float64}}([0.0, e_j, e_k])
    scored = logpdf(seq, events)

    # (a) Equals the factorised per-edge sum: each observed intermediate
    # conditions on its own edge delay at the observed gap.
    factorised = logpdf(edge1, e_j - 0.0) + logpdf(edge2, e_k - e_j)
    @test scored ≈ factorised

    # (b) Does NOT equal the collapsed marginal: `observed_distribution(seq)`
    # lowers the chain to a single `Convolved` total and marginalises the
    # intermediate, scoring only the terminal time. That is a different number.
    obs = observed_distribution(seq)
    @test obs isa CensoredDistributions.Convolved
    collapsed = logpdf(obs, e_k)
    @test !isapprox(scored, collapsed)

    # The collapsed marginal genuinely integrates the intermediate out: it
    # equals the convolution density at the terminal time, independent of where
    # the intermediate event e_j fell.
    e_j_alt = 2.9
    events_alt = Vector{Union{Missing, Float64}}([0.0, e_j_alt, e_k])
    @test logpdf(obs, e_k) ≈ collapsed                  # marginal ignores e_j
    @test !isapprox(logpdf(seq, events_alt), scored)    # factorised does not
end
