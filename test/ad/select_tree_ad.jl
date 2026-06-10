# AD coverage for nested-`Select` tree routing. A record's nested Select is
# resolved to its routed alternative (the selector is data, a Symbol), then the
# resolved tree scores through the numeric event-vector path. The gradient w.r.t.
# the delay params must flow under ForwardDiff; under Mooncake reverse the routed
# tree's by-name event vector derives its event names by string ops
# (`_split_edge_name` in `tree_events.jl`) over a row that may carry `Missing`,
# which Mooncake reverse cannot trace, so the routed-tree gradient is kept on
# ForwardDiff and the block is flagged here (a package-level Mooncake fix is in
# progress on another branch).

@testitem "nested Select routing gradient: ForwardDiff" tags=[
    :ad, :forwarddiff] begin
    using CensoredDistributions, Distributions
    using ADTypes: AutoForwardDiff
    using DifferentiationInterface: gradient
    using ForwardDiff: ForwardDiff

    # Two records routing to DIFFERENT alternatives of a nested Select, so the
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
        inner = selecting(:a => a, :b => b)
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
