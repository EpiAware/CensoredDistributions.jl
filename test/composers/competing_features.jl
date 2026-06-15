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

@testitem "racing-hazard: AD through a nested censored tree logpdf" begin
    using Distributions
    using ForwardDiff: gradient

    # The nested-tree event-vector scorer must keep the racing delays' Dual type
    # (regression: a `convert(T, ...)` to the data type stripped it). Differentiate
    # the censored-composer logpdf of a death-observed record w.r.t. the shapes.
    onset = primary_censored(LogNormal(0.5, 0.4), Uniform(0, 1))
    ev = Vector{Union{Missing, Float64}}([0.3, missing, 3.3, missing])
    function f(p)
        haz = competing(:death => Gamma(p[1], 3.0),
            :recover => Gamma(p[2], 2.0))
        d = compose((onset = onset, resolution = haz))
        return logpdf(d, ev)
    end
    g = gradient(f, [2.0, 3.0])
    @test all(isfinite, g)
    @test any(!=(0), g)
end

@testitem "no-event: forward stream drops the no-event mass" begin
    using Distributions

    # The forward per-outcome stream skips the no-event branch: only the real
    # outcome produces a count series, carrying its branch-probability mass.
    ρ = 0.6
    ne = competing(:report => (Gamma(2.0, 1.5), ρ), :none => (NoEvent(), 1 - ρ))
    series = zeros(60)
    series[1] = 1.0
    fwd = convolve_distributions(ne, series; events = (:report,))
    @test sum(fwd.report)≈ρ atol = 1e-6
    # The `none` branch is not a producible event (it has no series).
    @test_throws ArgumentError convolve_distributions(ne, series; events = (:none,))
end

@testitem "racing-hazard: SurvivalDistributions leaves race (#470)" begin
    using Distributions
    import SurvivalDistributions as SD

    # The racing-hazard node consumes the survival surface (`logccdf`/`logpdf`) of
    # its leaves, so a SurvivalDistributions family races alongside a stock leaf.
    haz = competing(:death => SD.LogLogistic(2.0, 1.5),
        :recover => Gamma(3.0, 2.0))
    t = 2.5
    # The marginal survival is the product of the two leaf survivals, and the
    # cause-resolved sub-density uses each leaf's logpdf + the other's logccdf.
    S = ccdf(SD.LogLogistic(2.0, 1.5), t) * ccdf(Gamma(3.0, 2.0), t)
    @test ccdf(haz, t) ≈ S
    @test isfinite(logpdf(haz, t))
    @test CensoredDistributions._hazard_cause_logpdf(haz, 1, t) ≈
          logpdf(SD.LogLogistic(2.0, 1.5), t) + logccdf(Gamma(3.0, 2.0), t)
end

@testitem "no-event: top-level DynamicPPL scores occurrence vs non-occurrence" begin
    using Distributions
    using DynamicPPL: @model, to_submodel, logjoint

    ρ = 0.7
    report_d = Gamma(2.0, 1.0)
    ne = competing(:report => (report_d, ρ), :none => (NoEvent(), 1 - ρ))
    @model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

    # OBSERVED occurrence: condition on the report branch (log ρ + delay logpdf).
    lj_occ = only(logjoint(demo(ne, (report = 2.5,)), (;)))
    @test lj_occ ≈ log(ρ) + logpdf(report_d, 2.5)

    # OBSERVED non-occurrence: the no-event slot present scores log(1 - ρ) alone.
    lj_non = only(logjoint(demo(ne, (none = 1.0,)), (;)))
    @test lj_non ≈ log(1 - ρ)
end

@testitem "non-terminal Competing: whole-tree event-name layout (#466 F3)" begin
    using Distributions

    # A composer-valued outcome (death => a Sequential subchain) spans its
    # SUBTREE's event slots; a leaf outcome (recover) spans one. The flat event
    # names interleave the subtree targets where the composer outcome sits.
    admit = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    burial = Gamma(1.5, 1.0)
    chain = Sequential((admit, burial), (:onset_admit, :admit_burial))
    recover = Gamma(3.0, 2.0)
    ne = competing(:death => (chain, 0.4), :recover => (recover, 0.6))

    @test ne isa CensoredDistributions.Competing
    @test CensoredDistributions._is_nonterminal(ne)
    # death outcome -> 2 subtree slots (admit, burial); recover -> 1 slot.
    @test CensoredDistributions._event_child_nleaves(ne) == 3

    d = compose((resolution = ne,))
    enames = event_names(d)
    # origin (event_1) + admit + burial (death subtree) + recover (leaf).
    @test enames == (:event_1, :admit, :burial, :recover)
end

@testitem "non-terminal Competing: composer-outcome slice scoring (#466 F3)" begin
    using Distributions

    admit = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    burial = Gamma(1.5, 1.0)
    chain = Sequential((admit, burial), (:onset_admit, :admit_burial))
    recover = Gamma(3.0, 2.0)
    pd, pr = 0.4, 0.6
    ne = competing(:death => (chain, pd), :recover => (recover, pr))
    d = compose((resolution = ne,))
    enames = event_names(d)
    i = Dict(n => k for (k, n) in enumerate(enames))

    o, t_admit, t_burial = 0.3, 2.5, 4.0

    # COMPOSER outcome observed (death subtree): score is
    # log p_death + subtree chain density (admit conditions on its declared
    # censoring at the gap from the shared origin, burial at its gap).
    evd = Vector{Union{Missing, Float64}}(missing, length(enames))
    evd[i[:event_1]] = o
    evd[i[:admit]] = o + t_admit
    evd[i[:burial]] = o + t_admit + t_burial
    expected_d = log(pd) + logpdf(admit, t_admit) + logpdf(burial, t_burial)
    @test logpdf(d, evd) ≈ expected_d

    # LEAF outcome observed (recover): the mixed leaf+composer slice arithmetic
    # still lands recover in its own slot; score is log p_recover + leaf density.
    evr = Vector{Union{Missing, Float64}}(missing, length(enames))
    evr[i[:event_1]] = o
    evr[i[:recover]] = o + 5.0
    @test logpdf(d, evr) ≈ log(pr) + logpdf(recover, 5.0)
end

@testitem "non-terminal Competing: rand round-trips through logpdf (#466 F3)" begin
    using Distributions, Random

    onset = primary_censored(LogNormal(0.5, 0.4), Uniform(0, 1))
    admit = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    burial = Gamma(1.5, 1.0)
    chain = Sequential((admit, burial), (:admit_death, :death_burial))
    recover = Gamma(3.0, 2.0)
    ne = competing(:death => (chain, 0.4), :recover => (recover, 0.6))
    d = compose((onset = onset, resolution = ne))

    rng = MersenneTwister(11)
    for _ in 1:25
        draw = rand(rng, d)
        @test draw isa NamedTuple
        # Exactly one outcome resolved: either the death subtree's slots are
        # populated (and recover missing) or recover is populated (subtree
        # missing). A simulated record always scores finite.
        death_obs = draw.death !== missing || draw.burial !== missing
        recover_obs = draw.recover !== missing
        @test death_obs ⊻ recover_obs
        @test isfinite(logpdf(d, draw))
    end
end

@testitem "non-terminal Competing: scalar marginal errors (#466 F3)" begin
    using Distributions

    admit = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    chain = Sequential((admit, Gamma(1.5, 1.0)), (:onset_admit, :admit_burial))
    ne = competing(:death => (chain, 0.4), :recover => (Gamma(3.0, 2.0), 0.6))

    # A non-terminal node is multivariate: no scalar logpdf / mean / as_mixture.
    @test_throws ArgumentError logpdf(ne, 1.0)
    @test_throws ArgumentError mean(ne)
    @test_throws ArgumentError as_mixture(ne)
    @test_throws ArgumentError cdf(ne, 1.0)

    # The same holds for a non-terminal racing-hazard node.
    haz = competing(:death => chain, :recover => Gamma(3.0, 2.0))
    @test haz isa CensoredDistributions.HazardCompeting
    @test CensoredDistributions._is_nonterminal(haz)
    @test_throws ArgumentError logpdf(haz, 1.0)
    @test_throws ArgumentError mean(haz)
    @test_throws ArgumentError var(haz)
    @test_throws ArgumentError winning_probabilities(haz)
end

@testitem "non-terminal Competing: forward stream recurses subtrees (#466 F3)" begin
    using Distributions

    # A composer-valued outcome fans its SUBTREE's events out, each carrying the
    # outcome's branch-probability thinning. The death subtree's burial endpoint
    # mass equals the outcome probability (a unit impulse, long horizon).
    pd, pr = 0.4, 0.6
    chain = Sequential((Gamma(2.0, 1.0), Gamma(1.5, 1.0)),
        (:onset_admit, :admit_burial))
    ne = competing(:death => (chain, pd), :recover => (Gamma(3.0, 2.0), pr))

    series = zeros(120)
    series[1] = 1.0
    fwd = convolve_distributions(ne, series; events = (:burial, :recover))
    # burial is the death subtree endpoint -> mass p_death; recover -> p_recover.
    @test sum(fwd.burial) ≈ pd atol = 1e-3
    @test sum(fwd.recover) ≈ pr atol = 1e-3
end

@testitem "non-terminal Competing: AD agrees across duals (#466 F3)" begin
    using Distributions
    using ForwardDiff: gradient

    # Differentiate the non-terminal tree logpdf w.r.t. the death subtree's leaf
    # params AND the branch probability; the gradient is finite and non-zero.
    ev = Vector{Union{Missing, Float64}}([0.3, 2.8, 6.5, missing])
    function f(p)
        admit = primary_censored(Gamma(p[1], 1.0), Uniform(0, 1))
        chain = Sequential((admit, Gamma(p[2], 1.0)),
            (:onset_admit, :admit_burial))
        ne = competing(:death => (chain, p[3]),
            :recover => (Gamma(3.0, 2.0), 1 - p[3]))
        return logpdf(compose((resolution = ne,)), ev)
    end
    g = gradient(f, [2.0, 1.5, 0.4])
    @test all(isfinite, g)
    @test any(!=(0), g)
    # d/dp[3] of log(p[3]) at 0.4 is 1/0.4 = 2.5 (the branch-prob weight).
    @test g[3] ≈ 2.5 atol = 1e-6
end

@testitem "non-terminal Competing: small Turing recovery (#466 F3)" tags=[:turing] begin
    using Distributions, Random
    using Turing
    using Statistics: mean

    # Simulate a populated death-subtree / leaf-recover dataset, then recover the
    # death branch probability by NUTS. The composer outcome's subtree slots are
    # populated on a death win, the recover slot on a recover win, so the model
    # scores the whole-tree non-terminal Competing per record.
    onset = primary_censored(LogNormal(0.5, 0.4), Uniform(0, 1))
    admit = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    burial = Gamma(1.5, 1.0)
    chain = Sequential((admit, burial), (:admit_death, :death_burial))
    recover = Gamma(3.0, 2.0)
    p_true = 0.35
    truth = compose((onset = onset,
        resolution = competing(:death => (chain, p_true),
            :recover => (recover, 1 - p_true))))

    rng = MersenneTwister(2027)
    rows = [rand(rng, truth) for _ in 1:400]

    @model function recover_cfr(rows)
        p ~ Beta(2, 2)
        node = compose((onset = onset,
            resolution = competing(:death => (chain, p),
                :recover => (recover, 1 - p))))
        for r in rows
            Turing.@addlogprob! logpdf(node, r)
        end
    end

    chn = sample(rng, recover_cfr(rows), NUTS(), 600; progress = false)
    p_hat = mean(chn[:p])
    # The posterior mean recovers the true CFR within a loose tolerance (N = 400).
    @test isapprox(p_hat, p_true; atol = 0.08)
end

@testitem "competing: introspection works for both node types" begin
    using Distributions

    haz = competing(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
    onset = primary_censored(LogNormal(0.5, 0.4), Uniform(0, 1))
    d = compose((onset = onset, severity = haz))

    # event_names / event_tree / event descend the racing node.
    @test :death in event_names(d)
    @test event(d, :severity) === haz
    @test event(d, :severity, :death) == Gamma(2.0, 3.0)

    # params_table lists the racing delays but NO branch_probs rows.
    pt = params_table(haz)
    @test !any(occursin("branch_probs", string(e)) for e in pt.edge)

    # update rebuilds the racing node from a new parameter set.
    haz2 = update(haz,
        (death = (shape = 4.0, scale = 3.0),
            recover = (shape = 5.0, scale = 2.0)))
    @test haz2 isa CensoredDistributions.HazardCompeting
    @test event(haz2, :death) == Gamma(4.0, 3.0)
end
