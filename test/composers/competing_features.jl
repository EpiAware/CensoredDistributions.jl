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

    # Mixing the two payload shapes is rejected (ambiguous node type) UNLESS it
    # is the residual form (every outcome but the last carries a probability).
    @test competing(:a => (Gamma(1.0, 1.0), 0.5), :b => Gamma(2.0, 1.0)) isa
          CensoredDistributions.Competing
    # A bare delay BEFORE the last outcome (more than one omitted) is ambiguous.
    @test_throws ArgumentError competing(:a => Gamma(1.0, 1.0),
        :b => (Gamma(2.0, 1.0), 0.5), :c => Gamma(2.0, 1.0))
end

@testitem "competing: residual last-outcome probability (#46)" begin
    using Distributions
    import ForwardDiff

    # Omitting the LAST outcome's probability makes it the residual `1 - sum(of
    # the others)`. The residual node is density-IDENTICAL to the all-explicit
    # form: same branch_probs, same logpdf, same winning_probabilities.
    cfr = 0.3
    explicit = competing(:death => (Gamma(1.5, 1.0), cfr),
        :disch => (Gamma(2.0, 1.5), 1 - cfr))
    resid = competing(:death => (Gamma(1.5, 1.0), cfr),
        :disch => Gamma(2.0, 1.5))
    @test resid isa CensoredDistributions.Competing
    @test resid.branch_probs == explicit.branch_probs
    @test logpdf(resid, 1.7) == logpdf(explicit, 1.7)
    @test winning_probabilities(resid) == winning_probabilities(explicit)
    @test mean(resid) == mean(explicit)

    # Three outcomes: the residual is `1 - (p_a + p_b)`.
    e3 = competing(:a => (Gamma(1.0, 1.0), 0.2), :b => (Gamma(1.0, 1.0), 0.3),
        :c => (Gamma(1.0, 1.0), 0.5))
    r3 = competing(:a => (Gamma(1.0, 1.0), 0.2), :b => (Gamma(1.0, 1.0), 0.3),
        :c => Gamma(1.0, 1.0))
    @test r3.branch_probs == e3.branch_probs
    @test logpdf(r3, 0.9) == logpdf(e3, 0.9)

    # Leading probabilities exceeding one (negative residual) errors clearly.
    @test_throws ArgumentError competing(:a => (Gamma(1.0, 1.0), 0.7),
        :b => (Gamma(1.0, 1.0), 0.5), :c => Gamma(1.0, 1.0))

    # The residual is a DIFFERENTIABLE function of the leading probabilities: a
    # `logistic(Xβ)`/sampled leading prob flows its `Dual` into the residual, so
    # the node differentiates w.r.t. the leading prob exactly as the explicit
    # form does (the residual carries the same partial, opposite sign).
    f_resid(p) = logpdf(
        competing(:death => (Gamma(1.5, 1.0), p[1]), :disch => Gamma(2.0, 1.5)),
        1.7)
    f_explicit(p) = logpdf(
        competing(:death => (Gamma(1.5, 1.0), p[1]),
            :disch => (Gamma(2.0, 1.5), 1 - p[1])),
        1.7)
    g_resid = ForwardDiff.gradient(f_resid, [cfr])
    g_explicit = ForwardDiff.gradient(f_explicit, [cfr])
    @test all(isfinite, g_resid)
    @test g_resid ≈ g_explicit

    # A residual NoEvent last outcome carries the residual no-event mass. The
    # residual is `1 - 0.7` exactly (the same float subtraction), so compare
    # against that rather than the literal `0.3` (a different bit pattern).
    ne = competing(:report => (Gamma(2.0, 1.0), 0.7), :none => NoEvent())
    @test CensoredDistributions._has_no_event(ne)
    @test ne.branch_probs == (0.7, 1 - 0.7)
    @test occurrence_probability(ne) ≈ 0.7
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

@testitem "non-terminal Competing: rand fills the winning subtree's slots" begin
    using Distributions, Random

    # The whole-tree `rand` of a NON-TERMINAL Competing must draw the resolved
    # outcome's WHOLE subtree into its slot slice (not a single time, not the
    # other outcome's slots) and at the right branch frequency. When the
    # composer-valued `death` outcome wins, BOTH of its subtree slots are filled
    # and time-ordered (origin < admit < burial); when `recover` wins, its
    # single slot is filled and the subtree slots stay missing. (#46 Task 2a
    # lock-in: the non-terminal whole-tree path samples into the right slots.)
    onset = primary_censored(LogNormal(0.5, 0.4), Uniform(0, 1))
    admit = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    burial = Gamma(1.5, 1.0)
    chain = Sequential((admit, burial), (:admit_death, :death_burial))
    recover = Gamma(3.0, 2.0)
    p_death = 0.4
    ne = competing(:death => (chain, p_death),
        :recover => (recover, 1 - p_death))
    d = compose((onset = onset, resolution = ne))
    @test CensoredDistributions._flat_event_names(d) ==
          (:event_1, :event_2, :death, :burial, :recover)

    # Count death wins inside a function so the loop counter is a clean local
    # (a bare top-level `for` in a `@testitem` hits soft-scope on the counter).
    function draw_and_check(rng, d, N)
        n_death = 0
        for _ in 1:N
            draw = rand(rng, d)
            if draw.death !== missing
                n_death += 1
                # Both subtree slots filled, recover missing, times ordered.
                @test draw.burial !== missing
                @test draw.recover === missing
                @test draw.event_1 < draw.death < draw.burial
            else
                # Recover wins: its slot filled, the death subtree missing.
                @test draw.recover !== missing
                @test draw.death === missing
                @test draw.burial === missing
            end
            @test isfinite(logpdf(d, draw))
        end
        return n_death
    end

    N = 4000
    n_death = draw_and_check(MersenneTwister(31), d, N)
    # The winning frequency matches the branch probability within MC error.
    @test n_death / N ≈ p_death atol = 0.03
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

@testitem "racing-hazard: var of a near-degenerate node is non-negative" begin
    using Distributions

    # Two near-degenerate racing nodes (large shape, tiny scale) concentrate
    # the survival drop into a narrow window. The two Gauss-Legendre
    # quadratures backing `E[T^2]` and `E[T]^2` can disagree at machine
    # precision and leave `e2 - m^2` a tiny negative; the clamp must keep the
    # reported variance non-negative.
    for (a, θ) in ((500.0, 0.002), (1000.0, 0.001), (2000.0, 0.0005))
        haz = competing(:a => Gamma(a, θ), :b => Gamma(a, θ))
        @test var(haz) >= 0
    end
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

@testitem "non-terminal HazardCompeting: cross-cause survival weighting (#479)" begin
    using Distributions

    # A NON-TERMINAL racing-hazard outcome (death => a chain) must weight its
    # subtree density by the survival of the OTHER (leaf) cause up to the
    # subtree's resolution time, mirroring the leaf formula f_j(t) ∏_{k≠j} S_k(t).
    admit = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    burial = Gamma(1.5, 1.0)
    chain = Sequential((admit, burial), (:onset_admit, :admit_burial))
    recover = Gamma(3.0, 2.0)
    haz = competing(:death => chain, :recover => recover)
    @test CensoredDistributions._is_nonterminal(haz)

    d = compose((resolution = haz,))
    enames = event_names(d)
    @test enames == (:event_1, :admit, :burial, :recover)
    i = Dict(n => k for (k, n) in enumerate(enames))

    o, t_admit, t_burial = 0.3, 2.5, 4.0
    t_res = t_admit + t_burial  # the death subtree's terminal (burial) gap.

    # COMPOSER cause wins (death subtree observed): the score is the subtree's own
    # density PLUS the cross-cause survival of recover at the resolution time.
    evd = Vector{Union{Missing, Float64}}(missing, length(enames))
    evd[i[:event_1]] = o
    evd[i[:admit]] = o + t_admit
    evd[i[:burial]] = o + t_admit + t_burial
    subtree = logpdf(admit, t_admit) + logpdf(burial, t_burial)
    expected_d = subtree + logccdf(recover, t_res)
    @test logpdf(d, evd) ≈ expected_d
    # The cross-cause factor is genuinely present (not a no-op): dropping it would
    # over-score by `-logccdf(recover, t_res) > 0`.
    @test logpdf(d, evd) < subtree

    # LEAF cause wins (recover observed): the cross-cause survival is now the
    # death SUBTREE's marginal resolution-time survival (the convolved chain).
    evr = Vector{Union{Missing, Float64}}(missing, length(enames))
    evr[i[:event_1]] = o
    evr[i[:recover]] = o + 5.0
    death_marginal = convolve_distributions((Gamma(2.0, 1.0), Gamma(1.5, 1.0)))
    expected_r = logpdf(recover, 5.0) + logccdf(death_marginal, 5.0)
    @test logpdf(d, evr) ≈ expected_r
end

@testitem "non-terminal HazardCompeting: three duals agree (#479)" begin
    using Distributions, Random

    # Plain (uncensored) leaves so the composer cause's marginal racing time is a
    # clean convolution; the three consistent duals must agree for the NON-TERMINAL
    # racing tree (death => a two-step chain, recover => a leaf).
    chain = Sequential((Gamma(2.0, 1.0), Gamma(1.5, 1.0)),
        (:onset_admit, :admit_burial))
    recover = Gamma(3.0, 2.0)
    haz = competing(:death => chain, :recover => recover)
    @test CensoredDistributions._is_nonterminal(haz)

    # The death subtree resolves at the SUM of its two delays (its marginal
    # racing-time distribution).
    death_T = convolve_distributions((Gamma(2.0, 1.0), Gamma(1.5, 1.0)))

    # Dual A (derived winning prob): P(j wins) = ∫ f_j(t) ∏_{k≠j} S_k(t) dt over a
    # fine grid, independent of any internal quadrature node set.
    ts = range(0.0, 60.0; length = 60_001)
    dt = step(ts)
    p_death = sum(pdf(death_T, t) * ccdf(recover, t) for t in ts) * dt
    p_recover = sum(pdf(recover, t) * ccdf(death_T, t) for t in ts) * dt
    @test p_death + p_recover ≈ 1.0 atol = 1e-3

    # Dual B (MC argmin frequency): draw each cause's racing time via the package
    # rand path (a composer cause draws its subtree marginal resolution time) and
    # count the death wins; matches the derived winning prob within MC error.
    function mcfreq(node, N)
        rng = MersenneTwister(2024)
        nd = 0
        for _ in 1:N
            td = CensoredDistributions._hazard_outcome_racing_time(
                rng, node.delays[1], Float64)
            tr = CensoredDistributions._hazard_outcome_racing_time(
                rng, node.delays[2], Float64)
            td < tr && (nd += 1)
        end
        return nd / N
    end
    fd_mc = mcfreq(haz, 400_000)
    @test fd_mc≈p_death atol = 5e-3

    # Dual C (scoring-path mass): the SCORED non-terminal record reproduces the
    # SAME cause-resolved sub-density A integrates. For a death-subtree win the
    # scored cross-cause factor is exactly S_recover(t_res) (the survival in A's
    # integrand), and the subtree density at its resolution is f_death's
    # contribution. Integrating exp(scored cross-cause + subtree marginal density)
    # over the resolution time recovers the death winning probability, tying the
    # scoring path (the #479 fix) to the derived winning prob and the MC frequency.
    cross(t) = CensoredDistributions._hazard_cross_cause_logsurvival(haz, 1, t)
    @test exp(cross(2.5)) ≈ ccdf(recover, 2.5)  # the scored S_recover factor.
    p_death_scored = sum(pdf(death_T, t) * exp(cross(t)) for t in ts) * dt
    @test p_death_scored ≈ p_death
    @test p_death_scored≈fd_mc atol = 5e-3
end

@testitem "non-terminal HazardCompeting: rand round-trips (#479)" begin
    using Distributions, Random

    onset = primary_censored(LogNormal(0.5, 0.4), Uniform(0, 1))
    admit = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    burial = Gamma(1.5, 1.0)
    chain = Sequential((admit, burial), (:admit_death, :death_burial))
    recover = Gamma(3.0, 2.0)
    haz = competing(:death => chain, :recover => recover)
    d = compose((onset = onset, resolution = haz))

    rng = MersenneTwister(11)
    draws = [rand(rng, d) for _ in 1:80]
    for draw in draws
        @test draw isa NamedTuple
        # Exactly one outcome resolved: either the death subtree slots populate
        # (recover missing) or recover populates (subtree missing). Both wins must
        # score a finite logpdf (composer-win uses the leaf cross-cause survival;
        # leaf-win uses the death subtree's marginal survival).
        death_obs = draw.death !== missing || draw.burial !== missing
        recover_obs = draw.recover !== missing
        @test death_obs ⊻ recover_obs
        @test isfinite(logpdf(d, draw))
    end
    # Both branches are exercised (the chosen seed wins each at least once).
    death_won(draw) = draw.death !== missing || draw.burial !== missing
    @test count(death_won, draws) > 0
    @test count(draw -> draw.recover !== missing, draws) > 0
end

@testitem "non-terminal HazardCompeting: rand fidelity (composer-win subtree)" begin
    using Distributions, Random, Statistics

    # DISTRIBUTIONAL fidelity (the #466 F3 rand/likelihood mismatch fix): for a
    # NON-TERMINAL racing outcome (death => a chain) drawn through the nested
    # event-record path, the RECORDED subtree must be the SAME realisation that
    # WON the race. Its resolution time then follows the CONDITIONAL distribution
    # `f_death(t | death wins) ∝ f_death(t) S_recover(t)` — stochastically SMALLER
    # than the unconditional marginal `f_death(t)`, because the winner won by
    # resolving first. The OLD code filled the recorded subtree with a SECOND
    # independent `_tree_rand!`, so the recorded resolution time followed the
    # UNCONDITIONAL marginal instead (its mean is `mean(death_T)`, well above the
    # conditional mean); this test FAILS there. A primary-censored onset gives the
    # tree a shared latent origin, so `rand` routes through `_tree_event_record`
    # (the path the fix touches). The racing chain hangs off that shared origin
    # (`event_1`), so the recorded resolution gap is the burial event minus
    # `event_1`; no secondary intervals are applied, so the gap stays continuous
    # for a tight distributional check.
    onset = primary_censored(LogNormal(0.5, 0.4), Uniform(0, 1))
    chain = Sequential((Gamma(2.0, 1.0), Gamma(1.5, 1.0)),
        (:onset_admit, :admit_burial))
    recover = Gamma(3.0, 2.0)
    haz = competing(:death => chain, :recover => recover)
    d = compose((onset = onset, resolution = haz))
    @test CensoredDistributions._is_nonterminal(haz)
    # Layout: (:event_1 shared origin, :event_2 onset, :admit, :burial, :recover).
    @test event_names(d) == (:event_1, :event_2, :admit, :burial, :recover)

    # The death subtree resolves at the SUM of its two (zero-origin) delays.
    death_T = convolve_distributions((Gamma(2.0, 1.0), Gamma(1.5, 1.0)))

    # Reference conditional resolution-time distribution on a fine grid:
    # f_death(t) * S_recover(t), normalised over the death-win mass.
    ts = range(1e-4, 60.0; length = 60_001)
    dt = step(ts)
    w = [pdf(death_T, t) * ccdf(recover, t) for t in ts]
    p_death = sum(w) * dt
    cond_mean = sum(t * wt for (t, wt) in zip(ts, w)) * dt / p_death
    uncond_mean = mean(death_T)  # what the OLD double-draw would record
    # The conditional mean is materially below the unconditional (the test's whole
    # point); guard the fixture so a degenerate setup cannot pass trivially.
    @test cond_mean < uncond_mean - 0.3

    # Simulate many records through the nested event-record path; collect the
    # recorded death-subtree resolution time (burial gap from the shared origin)
    # on every death win, and count the wins. Wrapped in a function so the loop
    # accumulators are locals (a testitem body runs at top-level soft scope).
    function simulate(dist, N)
        rng = MersenneTwister(20240617)
        res = Float64[]
        nd = 0
        nr = 0
        for _ in 1:N
            draw = rand(rng, dist)
            if draw.burial !== missing
                nd += 1
                push!(res, draw.burial - draw.event_1)
            elseif draw.recover !== missing
                nr += 1
            end
        end
        return res, nd, nr
    end
    res_times, n_death, n_recover = simulate(d, 60_000)

    # Win fraction matches the derived death-win probability (the race is honest).
    @test n_death / (n_death + n_recover)≈p_death atol = 5e-3
    # The recorded resolution times follow the CONDITIONAL distribution: their mean
    # tracks `cond_mean`, NOT the unconditional `uncond_mean`. Under the old double
    # draw this mean would be `uncond_mean`, failing the tight conditional band.
    smean = mean(res_times)
    @test smean≈cond_mean atol = 0.15
    @test smean < uncond_mean - 0.2
end

@testitem "non-terminal HazardCompeting: AD through the tree logpdf (#479)" begin
    using Distributions
    using ForwardDiff: gradient

    # Differentiate the NON-TERMINAL racing-tree logpdf w.r.t. the death subtree's
    # leaf shapes AND the racing recover leaf shape. The cross-cause survival of
    # recover must carry the recover Dual (so g[3] ≠ 0), and the subtree density
    # the chain Duals; all finite, none all-zero.
    onset = primary_censored(LogNormal(0.5, 0.4), Uniform(0, 1))
    # A death-subtree-observed record: origin, admit, burial present, recover off.
    ev = Vector{Union{Missing, Float64}}([0.3, 2.8, 6.5, missing])
    function f(p)
        admit = primary_censored(Gamma(p[1], 1.0), Uniform(0, 1))
        chain = Sequential((admit, Gamma(p[2], 1.0)),
            (:onset_admit, :admit_burial))
        recover = Gamma(p[3], 2.0)
        haz = competing(:death => chain, :recover => recover)
        return logpdf(compose((resolution = haz,)), ev)
    end
    g = gradient(f, [2.0, 1.5, 3.0])
    @test all(isfinite, g)
    @test any(!=(0), g)
    # The recover shape enters ONLY through the cross-cause survival logccdf(recover,
    # t_res); its gradient is non-zero, proving the cross-cause factor flows.
    @test g[3] != 0
end

@testitem "nested Competing inside a competing outcome: branch_probs override" begin
    using Distributions
    const CD = CensoredDistributions

    # S2 recursion: a mixture `Competing` nested INSIDE a (racing) competing
    # outcome's subtree must be FOUND by the per-record `branch_probs` override
    # path and overridden. Before the fix the override helpers stopped at the outer
    # `AbstractCompeting` (it shares the `UnivariateDistribution` supertype, so the
    # nested Competing hit the no-op leaf fallback): the node was silently
    # under-counted, bypassing the `n == 1` guard, and the override was never
    # applied. The fix RECURSES through `AbstractCompeting.delays`, so the override
    # reaches the single nested Competing.
    inner = competing(
        :burial => (Gamma(1.5, 1.0), 0.6), :cremation => (Gamma(2.0, 1.0), 0.4))
    death_chain = Sequential(
        (Gamma(2.0, 1.0), inner), (:onset_admit, :admit_outcome))
    recover = Gamma(3.0, 2.0)
    haz = competing(:death => death_chain, :recover => recover)
    d = compose((resolution = haz,))

    # The recursion finds exactly the one nested Competing.
    @test CD._count_competing(d) == 1

    # A death-win record observing the burial outcome, with a per-record override of
    # the nested Competing's branch probabilities.
    row = (event_1 = 0.0, admit = 2.0, burial = 5.0, cremation = missing,
        recover = missing, branch_probs = (burial = 0.3, cremation = 0.7))
    recs = CD.record_distributions(d, [row])
    lp = logpdf(recs[1], recs[1].events)

    # Reference: rebuild the nested Competing with the overridden probs by hand and
    # score the equivalent record directly. The override path must equal this.
    inner2 = competing(
        :burial => (Gamma(1.5, 1.0), 0.3), :cremation => (Gamma(2.0, 1.0), 0.7))
    death2 = Sequential(
        (Gamma(2.0, 1.0), inner2), (:onset_admit, :admit_outcome))
    d2 = compose((resolution = competing(:death => death2, :recover => recover),))
    ev = Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0, missing, missing])
    @test lp ≈ logpdf(d2, ev)

    # The override genuinely changed the score (the default stored probs differ).
    rec0 = CD.record_distributions(d,
        [(event_1 = 0.0, admit = 2.0, burial = 5.0, cremation = missing,
            recover = missing)])
    @test !(logpdf(rec0[1], rec0[1].events) ≈ lp)
end

@testitem "nested Select inside a competing outcome: per-record routing" begin
    using Distributions
    const CD = CensoredDistributions

    # S2 recursion: a `Select` nested INSIDE a (racing) competing outcome's subtree
    # must be FOUND and RESOLVED per record (and its selector field stripped before
    # event matching). Before the fix the Select-resolution helpers stopped at the
    # outer `AbstractCompeting` (a nested Select was never counted, so the
    # resolution rebuild was skipped and the Select silently scored its FIRST
    # alternative; `_select_fields` also missed the selector). The fix recurses
    # through `AbstractCompeting.delays`, so the routed alternative is scored.
    sel = selecting(:fast => Gamma(1.5, 1.0), :slow => Gamma(4.0, 1.0);
        selector = :speed)
    death_chain = Sequential((Gamma(2.0, 1.0), sel), (:onset_admit, :admit_outcome))
    recover = Gamma(3.0, 2.0)
    haz = competing(:death => death_chain, :recover => recover)
    d = compose((resolution = haz,))

    # The recursion finds exactly the one nested Select and its selector field.
    @test CD._count_selects(d) == 1
    @test CD._select_fields(d) == [:speed]

    ev = Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0, missing])
    # Routing to :slow scores the slow alternative; to :fast the fast one; the two
    # differ and each matches a hand-resolved (Select-free) reference tree.
    function routed_lp(speed)
        row = (event_1 = 0.0, admit = 2.0, outcome = 5.0, recover = missing,
            speed = speed)
        recs = CD.record_distributions(d, [row])
        return logpdf(recs[1], recs[1].events)
    end
    function ref_lp(alt)
        chain = Sequential((Gamma(2.0, 1.0), alt), (:onset_admit, :admit_outcome))
        dref = compose((resolution = competing(
            :death => chain, :recover => recover),))
        return logpdf(dref, ev)
    end
    @test routed_lp(:slow) ≈ ref_lp(Gamma(4.0, 1.0))
    @test routed_lp(:fast) ≈ ref_lp(Gamma(1.5, 1.0))
    @test !(routed_lp(:slow) ≈ routed_lp(:fast))

    # A record missing the selector field errors clearly (no silent first-alt).
    @test_throws ArgumentError CD.record_distributions(d,
        [(event_1 = 0.0, admit = 2.0, outcome = 5.0, recover = missing)])
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
