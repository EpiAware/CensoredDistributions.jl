# AD coverage for nested-`Choose` tree routing. A record's nested Choose is
# resolved to its routed alternative (the selector is data, a Symbol), then the
# resolved tree scores through the numeric event-vector path. The gradient w.r.t.
# the delay params must flow under BOTH ForwardDiff and Mooncake reverse.
#
# The routed tree's by-name event vector derives its event names by string ops in
# `tree_events.jl` (`_split_edge_name` / `_is_positional_edge_name` /
# `_all_positional_event_names`). Those helpers used to match a compiled `Regex`
# (`r"^step_\d+$"` etc.); `Base.compile(::Regex)` uses a try/catch Mooncake
# reverse cannot differentiate, so when a helper was inlined into the traced
# scoring path Mooncake reverse failed to build a rule and the bdbv tutorial fell
# back to AutoForwardDiff (#409, gating Mooncake-everywhere in #438). The helpers
# now do a plain `startswith` + ASCII-digit scan with no `Regex`, so the Mooncake
# reverse gradient over the routed-tree logpdf compiles and matches ForwardDiff.

@testitem "nested Choose routing gradient: ForwardDiff" tags=[
    :ad, :forwarddiff] begin
    using CensoredDistributions, Distributions
    using ADTypes: AutoForwardDiff
    using DifferentiationInterface: gradient
    using ForwardDiff: ForwardDiff

    # Two records routing to DIFFERENT alternatives of a nested Choose, so the
    # log density depends on BOTH alternatives' params.
    rows = [(onset = 0.0, admit = 2.0, death = 5.0, kind = :a),
        (onset = 0.0, admit = 1.0, death = 7.0, kind = :b)]
    obs = [[0.0, 2.0, 5.0], [0.0, 1.0, 7.0]]

    # θ = [edge1 shape, edge1 scale, alt-a shape, alt-a scale,
    #      alt-b shape, alt-b scale].
    function f(θ)
        e1 = primary_censored(Gamma(θ[1], θ[2]), Uniform(0, 1))
        a = primary_censored(Gamma(θ[3], θ[4]), Uniform(0, 1))
        b = primary_censored(Gamma(θ[5], θ[6]), Uniform(0, 1))
        inner = choose(:a => a, :b => b)
        seq = Sequential((e1, inner), (:onset_admit, :admit_death))
        recs = CensoredDistributions.record_distributions(seq, rows)
        return sum(i -> logpdf(recs[i], obs[i]), eachindex(recs))
    end

    θ = [3.0, 1.0, 2.0, 1.0, 5.0, 1.0]
    g = gradient(f, AutoForwardDiff(), θ)
    @test g isa AbstractVector && length(g) == 6 && all(isfinite, g)
    # Both alternatives are routed across the two rows, so both contribute.
    @test all(!=(0), g)
end

# Mooncake reverse over the SAME routed-tree logpdf (#409 regression). Before the
# regex was removed from the `tree_events.jl` edge-name helpers, Mooncake reverse
# could not compile a rule here (the `Base.compile(::Regex)` try/catch on the
# scoring path); now the regex-free string parsing lets the reverse gradient
# compile, stay finite, and match the ForwardDiff reference.
@testitem "nested Choose routing gradient: Mooncake reverse" tags=[
    :ad, :mooncake, :mooncake_reverse] begin
    using CensoredDistributions, Distributions
    using ADTypes: AutoForwardDiff, AutoMooncake
    using DifferentiationInterface: gradient
    using ForwardDiff: ForwardDiff
    using Mooncake: Mooncake

    rows = [(onset = 0.0, admit = 2.0, death = 5.0, kind = :a),
        (onset = 0.0, admit = 1.0, death = 7.0, kind = :b)]
    obs = [[0.0, 2.0, 5.0], [0.0, 1.0, 7.0]]

    function f(θ)
        e1 = primary_censored(Gamma(θ[1], θ[2]), Uniform(0, 1))
        a = primary_censored(Gamma(θ[3], θ[4]), Uniform(0, 1))
        b = primary_censored(Gamma(θ[5], θ[6]), Uniform(0, 1))
        inner = choose(:a => a, :b => b)
        seq = Sequential((e1, inner), (:onset_admit, :admit_death))
        recs = CensoredDistributions.record_distributions(seq, rows)
        return sum(i -> logpdf(recs[i], obs[i]), eachindex(recs))
    end

    θ = [3.0, 1.0, 2.0, 1.0, 5.0, 1.0]
    g_ref = gradient(f, AutoForwardDiff(), θ)
    # The reverse gradient must COMPILE (no Regex try/catch on the path), be
    # finite, and match the ForwardDiff reference.
    g = gradient(f, AutoMooncake(config = nothing), θ)
    @test g isa AbstractVector && length(g) == 6 && all(isfinite, g)
    @test isapprox(g, g_ref; rtol = 1e-4, atol = 1e-6)
end
