# AD coverage for the consolidated composer API. The new public
# constructors (`sequential` / `parallel`) and the `update` node-replace edit are
# construction-time operations, so they must not disturb the differentiated
# scoring path: an object built through them and then scored differentiates w.r.t.
# its leaf parameters under both ForwardDiff and Mooncake reverse (the project's
# target backends). The edit verbs run outside the gradient tape; the gradient
# flows through the rebuilt object's `event_logpdf` exactly as for a directly
# constructed tree.

@testitem "composer-api built object gradient: ForwardDiff" tags=[
    :ad, :forwarddiff] begin
    using CensoredDistributions, Distributions
    using ADTypes: AutoForwardDiff
    using DifferentiationInterface: gradient
    using ForwardDiff: ForwardDiff

    evs = [Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0]),
        Vector{Union{Missing, Float64}}([0.5, missing, 7.0])]
    Ds = [8.0, 9.0]

    # θ = [inc μ, inc σ, delta μ, delta σ]. Build through `sequential` (a public
    # constructor) and then `update` (node-replace) the second step, so both new
    # API paths feed the differentiated `event_logpdf`.
    function f(θ)
        inc = primary_censored(LogNormal(θ[1], θ[2]), Uniform(0, 1))
        delta0 = primary_censored(LogNormal(0.0, 1.0), Uniform(0, 1))
        seq = sequential(:onset_mid => inc, :mid_obs => delta0)
        delta = primary_censored(LogNormal(θ[3], θ[4]), Uniform(0, 1))
        seq = update(seq, :mid_obs => delta)
        return sum(eachindex(evs)) do i
            CensoredDistributions.event_logpdf(seq, evs[i]; horizon = Ds[i])
        end
    end

    θ = [1.0, 0.5, 0.5, 0.4]
    @test isfinite(f(θ))
    g = gradient(f, AutoForwardDiff(), θ)
    @test g isa AbstractVector && length(g) == 4 && all(isfinite, g)
    @test all(!=(0), g)
end

@testitem "composer-api built object gradient: Mooncake reverse" tags=[
    :ad, :mooncake, :mooncake_reverse] begin
    using CensoredDistributions, Distributions
    using ADTypes: AutoMooncake, AutoForwardDiff
    using DifferentiationInterface: gradient
    using ForwardDiff: ForwardDiff
    using Mooncake: Mooncake

    # Fully-observed records: the missing-intermediate `event_logpdf` path hits
    # the string-based event-name derivation Mooncake reverse cannot trace (a
    # pre-existing limitation, unrelated to the construction-time edits), so the
    # Mooncake check uses observed intermediates.
    evs = [Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0]),
        Vector{Union{Missing, Float64}}([0.5, 3.0, 7.0])]
    Ds = [8.0, 9.0]

    function f(θ)
        inc = primary_censored(LogNormal(θ[1], θ[2]), Uniform(0, 1))
        delta0 = primary_censored(LogNormal(0.0, 1.0), Uniform(0, 1))
        seq = sequential(:onset_mid => inc, :mid_obs => delta0)
        delta = primary_censored(LogNormal(θ[3], θ[4]), Uniform(0, 1))
        seq = update(seq, :mid_obs => delta)
        return sum(eachindex(evs)) do i
            CensoredDistributions.event_logpdf(seq, evs[i]; horizon = Ds[i])
        end
    end

    θ = [1.0, 0.5, 0.5, 0.4]
    ref = gradient(f, AutoForwardDiff(), θ)
    g = gradient(f, AutoMooncake(; config = nothing), θ)
    @test g isa AbstractVector && length(g) == 4 && all(isfinite, g)
    @test isapprox(g, ref; rtol = 1e-6, atol = 1e-8)
end
