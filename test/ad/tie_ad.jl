# AD coverage for `tie(d, paths...; name)`. `tie` produces the SAME `Shared`
# artefact as a hand-written `shared(:tag, …)` build, so the gradient over the
# tied group must (a) flow, (b) be ONE parameter (the tied shape/scale drive
# both occurrences), and (c) equal the gradient of the hand-shared tree. The
# leaf is rebuilt from `θ` INSIDE the differentiated closure, so `tie` runs on
# every AD evaluation (it must be AD-transparent, exactly like `shared`).

@testitem "tie group gradient flows and equals hand-shared: ForwardDiff" tags=[
    :ad, :forwarddiff] begin
    using CensoredDistributions, Distributions
    using ADTypes: AutoForwardDiff
    using DifferentiationInterface: gradient
    using ForwardDiff: ForwardDiff

    # `:sourced` branch: a LogNormal `src` then the tied Gamma `inc`. The same
    # `inc` family also sits in the `:index` branch, so the tie makes one group.
    ev = [1.5, 2.5]

    # Build with `tie`, rebuilding leaves from θ each call.
    function f_tie(θ)
        base = selecting(
            :index => compose((inc = Gamma(θ[1], θ[2]),)),
            :sourced => compose((src = LogNormal(θ[3], θ[4]),
                inc = Gamma(θ[1], θ[2]))))
        tied = tie(base, (:index, :inc), (:sourced, :inc); name = :inc)
        return logpdf(tied, ev; kind = :sourced)
    end

    # The hand-written `shared(:inc, …)` build at the same leaves.
    function f_hand(θ)
        d = selecting(
            :index => compose((inc = shared(:inc, Gamma(θ[1], θ[2])),)),
            :sourced => compose((src = LogNormal(θ[3], θ[4]),
                inc = shared(:inc, Gamma(θ[1], θ[2])))))
        return logpdf(d, ev; kind = :sourced)
    end

    θ = [2.0, 1.0, 0.5, 0.4]
    @test f_tie(θ) ≈ f_hand(θ) atol=1e-12

    g = gradient(f_tie, AutoForwardDiff(), θ)
    gh = gradient(f_hand, AutoForwardDiff(), θ)
    @test g isa AbstractVector && length(g) == 4 && all(isfinite, g)
    # The tied `inc` (θ[1], θ[2]) drives the `:sourced` score (one parameter
    # group), so its gradient entries are non-zero.
    @test all(!=(0), g[1:2])
    # `tie`d == hand-`shared`d: identical gradient.
    @test isapprox(g, gh; rtol = 1e-6, atol = 1e-8)
end

@testitem "tie group gradient: Mooncake reverse equals hand-shared" tags=[
    :ad, :mooncake, :mooncake_reverse] begin
    using CensoredDistributions, Distributions
    using ADTypes: AutoForwardDiff, AutoMooncake
    using DifferentiationInterface: gradient
    using ForwardDiff: ForwardDiff
    using Mooncake: Mooncake

    ev = [1.5, 2.5]

    function f_tie(θ)
        base = selecting(
            :index => compose((inc = Gamma(θ[1], θ[2]),)),
            :sourced => compose((src = LogNormal(θ[3], θ[4]),
                inc = Gamma(θ[1], θ[2]))))
        tied = tie(base, (:index, :inc), (:sourced, :inc); name = :inc)
        return logpdf(tied, ev; kind = :sourced)
    end
    function f_hand(θ)
        d = selecting(
            :index => compose((inc = shared(:inc, Gamma(θ[1], θ[2])),)),
            :sourced => compose((src = LogNormal(θ[3], θ[4]),
                inc = shared(:inc, Gamma(θ[1], θ[2])))))
        return logpdf(d, ev; kind = :sourced)
    end

    θ = [2.0, 1.0, 0.5, 0.4]
    g_ref = gradient(f_tie, AutoForwardDiff(), θ)
    g = gradient(f_tie, AutoMooncake(config = nothing), θ)
    g_hand = gradient(f_hand, AutoMooncake(config = nothing), θ)
    @test g isa AbstractVector && length(g) == 4 && all(isfinite, g)
    @test isapprox(g, g_ref; rtol = 1e-4, atol = 1e-6)
    @test isapprox(g, g_hand; rtol = 1e-4, atol = 1e-6)
end
