# Tests for the enriched `competing` composition (#466): the shared `competing`
# constructor + `AbstractCompeting` supertype, the no-event branch (Feature 1),
# and the racing-hazard `HazardCompeting` combinator (Feature 2) with its three
# consistent duals (rand argmin/min, logpdf marginal + cause-resolved, forward
# per-outcome sub-density stream).

@testitem "competing: shared constructor dispatches on payload shape" begin
    using Distributions

    # A `(delay, prob)` payload per outcome builds the fixed-probability mixture.
    mix = competing(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))
    @test mix isa CensoredDistributions.Competing
    @test mix isa CensoredDistributions.AbstractCompeting

    # Bare delays (no probabilities) build the racing-hazard node.
    haz = competing(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
    @test haz isa CensoredDistributions.HazardCompeting
    @test haz isa CensoredDistributions.AbstractCompeting

    # Mixing the two payload shapes is rejected (ambiguous node type).
    @test_throws ArgumentError competing(:a => (Gamma(1.0, 1.0), 0.5),
        :b => Gamma(2.0, 1.0))
end

@testitem "competing: NoEvent marker errors as a density" begin
    using Distributions

    @test_throws ArgumentError logpdf(NoEvent(), 1.0)
    @test_throws ArgumentError minimum(NoEvent())
    @test NoEvent() == NoEvent()
end

@testitem "competing: no-event mixture is a defective marginal" begin
    using Distributions

    ne = competing(:report => (Gamma(2.0, 1.0), 0.7), :none => (NoEvent(), 0.3))
    @test ne isa CensoredDistributions.Competing
    @test CensoredDistributions._has_no_event(ne)

    # A defective marginal has no scalar logpdf / mean / as_mixture.
    @test_throws ArgumentError logpdf(ne, 1.0)
    @test_throws ArgumentError mean(ne)
    @test_throws ArgumentError as_mixture(ne)

    # The occurrence probability is one minus the no-event mass.
    @test occurrence_probability(ne) ≈ 0.7
    @test winning_probabilities(ne) == (report = 0.7, none = 0.3)
end

@testitem "competing no-event: observed occurrence vs non-occurrence" begin
    using Distributions

    ρ = 0.7
    report_d = Gamma(2.0, 1.0)
    ne = competing(:report => (report_d, ρ), :none => (NoEvent(), 1 - ρ))
    # Nest off a censored origin so the event-vector path applies.
    onset = primary_censored(LogNormal(0.5, 0.4), Uniform(0, 1))
    d = compose((onset = onset, resolution = ne))
    enames = event_names(d)
    @test :report in enames
    @test :none in enames

    o = 0.3
    ireport = findfirst(==(:report), enames)
    inone = findfirst(==(:none), enames)
    iorigin = 1

    # OBSERVED occurrence: the report time is present; score log ρ + report delay
    # logpdf at the gap (the resolution conditions on the report branch).
    occ = Vector{Union{Missing, Float64}}(missing, length(enames))
    occ[iorigin] = o
    occ[ireport] = o + 2.5
    lp_occ = logpdf(d, occ)
    @test isfinite(lp_occ)

    # OBSERVED non-occurrence: the `none` slot is a present marker; its term is
    # `log(1 - ρ)` alone (no delay), distinct from the latent-missing case.
    nonocc = Vector{Union{Missing, Float64}}(missing, length(enames))
    nonocc[iorigin] = o
    nonocc[inone] = o  # presence marker (value not a time)
    lp_non = logpdf(d, nonocc)
    @test isfinite(lp_non)

    # LATENT non-occurrence: every resolution slot missing contributes no
    # competing term; only the origin (no extra factor) is scored.
    latent = Vector{Union{Missing, Float64}}(missing, length(enames))
    latent[iorigin] = o
    lp_lat = logpdf(d, latent)

    # The observed non-occurrence is the latent score plus the survival log q.
    @test lp_non ≈ lp_lat + log(1 - ρ)
end

@testitem "racing-hazard: three duals agree (acceptance test)" begin
    using Distributions, Random

    haz = competing(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))

    # Dual 1: derived winning probabilities (logpdf-consistent integral).
    wp = winning_probabilities(haz)
    @test sum(values(wp)) ≈ 1.0 atol = 1e-3

    # Dual 2: the marginal logpdf is the log-sum of cause-resolved sub-densities,
    # and equals the min-of-independent-delays density. Check survival ∏ S_k.
    t = 2.7
    S = ccdf(Gamma(2.0, 3.0), t) * ccdf(Gamma(3.0, 2.0), t)
    @test ccdf(haz, t) ≈ S
    fmix = pdf(Gamma(2.0, 3.0), t) * ccdf(Gamma(3.0, 2.0), t) +
           pdf(Gamma(3.0, 2.0), t) * ccdf(Gamma(2.0, 3.0), t)
    @test pdf(haz, t) ≈ fmix

    # Dual 3: forward per-outcome stream mass equals the derived winning prob
    # (sub-stochastic, NOT renormalised). Unit impulse, long horizon.
    series = zeros(80)
    series[1] = 1.0
    fwd = convolve_distributions(haz, series; events = (:death, :recover))
    @test sum(fwd.death) ≈ wp.death atol = 1e-3
    @test sum(fwd.recover) ≈ wp.recover atol = 1e-3

    # Dual 1 vs the MC frequency: argmin-cause frequencies match the derived
    # winning probabilities within Monte Carlo error.
    function mcfreq(node, N)
        rng = MersenneTwister(2024)
        nd = 0
        for _ in 1:N
            name, _ = CensoredDistributions.rand_outcome(rng, node)
            name === :death && (nd += 1)
        end
        return nd / N
    end
    fd = mcfreq(haz, 200_000)
    @test fd≈wp.death atol = 5e-3
end

@testitem "racing-hazard: nested cause-resolved scoring" begin
    using Distributions

    death_d, recover_d = Gamma(2.0, 3.0), Gamma(3.0, 2.0)
    haz = competing(:death => death_d, :recover => recover_d)
    onset = primary_censored(LogNormal(0.5, 0.4), Uniform(0, 1))
    d = compose((onset = onset, resolution = haz))
    enames = event_names(d)
    @test :death in enames
    @test :recover in enames

    # Build an event vector with the origin observed and death observed at a gap.
    o = 0.4
    gap = 3.0
    vals = map(enames) do n
        n === :death ? o + gap : (n === first(enames) ? o : missing)
    end
    ev = Vector{Union{Missing, Float64}}(collect(vals))
    direct = logpdf(d, ev)
    @test isfinite(direct)

    # The resolution term is the cause-resolved sub-density f_death(gap) *
    # ∏_{k≠death} S_k(gap) plus the origin's own term. Compare to the bare
    # hazard cause logpdf to confirm the racing term flows through.
    haz_term = CensoredDistributions._hazard_cause_logpdf(haz, 1, gap)
    @test haz_term ≈ logpdf(death_d, gap) + logccdf(recover_d, gap)
end

@testitem "racing-hazard: rand round-trips through logpdf" begin
    using Distributions, Random

    haz = competing(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
    onset = primary_censored(LogNormal(0.5, 0.4), Uniform(0, 1))
    d = compose((onset = onset, resolution = haz))

    rng = MersenneTwister(7)
    draw = rand(rng, d)
    @test draw isa NamedTuple
    # Exactly one of death/recover is non-missing (the argmin cause won).
    resolved = count(n -> getfield(draw, n) !== missing, (:death, :recover))
    @test resolved == 1
    @test isfinite(logpdf(d, draw))
end

@testitem "racing-hazard: AD through the cause-resolved logpdf" begin
    using Distributions
    using ForwardDiff: gradient

    # Differentiate the cause-resolved sub-density w.r.t. the racing delays'
    # parameters; the gradient must be finite and non-zero.
    function f(p)
        haz = competing(:death => Gamma(p[1], 3.0),
            :recover => Gamma(3.0, p[2]))
        return CensoredDistributions._hazard_cause_logpdf(haz, 1, 2.5)
    end
    g = gradient(f, [2.0, 2.0])
    @test all(isfinite, g)
    @test any(!=(0), g)
end
