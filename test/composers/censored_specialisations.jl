# Reference tests for the censored specialisations of the generic composers
# (#329, PR3b). Each asserts the data-driven, dispatch-selected behaviour:
# single-edge == primary_censored, unobserved intermediate == convolution
# reference, observed intermediate == conditioning, Parallel == shared-origin
# Monte-Carlo, and the delay-adjusted CFR recovery via Resolve.

@testitem "Sequential single censored edge == primary_censored" begin
    using Distributions

    pc = primary_censored(LogNormal(1.5, 0.75), Uniform(0, 1))
    d = Sequential(pc)
    # Event vector [E0, E1]: origin primary censoring applied, scored at the gap.
    ev = Vector{Union{Missing, Float64}}([0.0, 2.5])
    @test logpdf(d, ev) ≈ logpdf(pc, 2.5)
    @test pdf(d, ev) ≈ pdf(pc, 2.5)

    # A non-zero origin shifts the gap, not the density.
    ev2 = Vector{Union{Missing, Float64}}([1.0, 3.5])
    @test logpdf(d, ev2) ≈ logpdf(pc, 2.5)
end

@testitem "Sequential unobserved intermediate == convolution reference" begin
    using Distributions

    d1 = LogNormal(1.2, 0.5)
    d2 = Gamma(2.0, 1.0)
    pe = Uniform(0, 1)
    chain = Sequential(primary_censored(d1, pe), primary_censored(d2, pe))

    # E1 unobserved: the origin segment spans both delays, so the convolution of
    # the continuous cores is primary-censored and scored at the observed gap;
    # the intermediate's own censoring is dropped (latent continuous time).
    ev = Vector{Union{Missing, Float64}}([0.0, missing, 4.0])
    ref = primary_censored(convolve_distributions(d1, d2), pe)
    @test logpdf(chain, ev) ≈ logpdf(ref, 4.0) rtol=1e-6
end

@testitem "Sequential observed intermediate conditions on the censored value" begin
    using Distributions

    pe = Uniform(0, 1)
    # An observed-bounded edge conditions on its OWN declared censoring (#329
    # "condition on its (censored) value"): each edge is scored through its own
    # censored logpdf at the day-observed gap, NOT the bare continuous core.
    d1 = primary_censored(LogNormal(1.2, 0.5), pe)
    # A day-resolution edge: double_interval_censored carries primary + interval
    # censoring, so a day-observed delay is interval-censored, not exact.
    d2 = double_interval_censored(Gamma(2.0, 1.0); primary_event = pe,
        interval = 1.0)
    chain = Sequential(d1, d2)

    ev = Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0])
    # Both observed-bounded edges score through their own censored logpdf at the
    # observed gaps (2.0 and 3.0), keeping the declared censoring.
    @test logpdf(chain, ev) ≈ logpdf(d1, 2.0) + logpdf(d2, 3.0) rtol=1e-6

    # A genuinely plain continuous edge (no declared censoring) is conditioned
    # on the continuous value for that edge.
    plain = Gamma(2.0, 1.0)
    chain2 = Sequential(d1, plain)
    @test logpdf(chain2, ev) ≈ logpdf(d1, 2.0) + logpdf(plain, 3.0) rtol=1e-6
end

@testitem "Sequential plain step path is unchanged (generic)" begin
    using Distributions

    # A concrete one-value-per-step vector still hits the generic PR3a path.
    s = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    @test logpdf(s, [1.5, 2.0]) ≈
          logpdf(Gamma(2.0, 1.0), 1.5) + logpdf(LogNormal(0.5, 0.4), 2.0)

    # The event-vector path needs one entry per event (length + 1).
    bad = Vector{Union{Missing, Float64}}([0.0, 1.0])
    @test_throws DimensionMismatch logpdf(s, bad)
end

@testitem "Parallel censored branches == shared-origin Monte-Carlo" begin
    using Distributions, Random
    Random.seed!(42)

    pe = Uniform(0.0, 1.0)
    b1 = Gamma(2.0, 1.0)
    b2 = LogNormal(1.0, 0.5)
    par = Parallel(primary_censored(b1, pe), primary_censored(b2, pe))

    y1, y2 = 2.3, 3.1
    # Missing origin: marginalise the shared origin over the present branches.
    ev = Vector{Union{Missing, Float64}}([missing, y1, y2])
    lp = logpdf(par, ev)

    # Monte-Carlo over the shared origin: E_O[ f_b1(y1-O) f_b2(y2-O) ].
    function mc_origin(pe, b1, b2, y1, y2, N)
        acc = 0.0
        for _ in 1:N
            o = rand(pe)
            acc += pdf(b1, y1 - o) * pdf(b2, y2 - o)
        end
        return log(acc / N)
    end
    @test lp ≈ mc_origin(pe, b1, b2, y1, y2, 4_000_000) rtol=3e-3

    # Present origin: conditions on it (prior + branch delays).
    o = 0.4
    evc = Vector{Union{Missing, Float64}}([o, y1, y2])
    @test logpdf(par, evc) ≈
          logpdf(pe, o) + logpdf(b1, y1 - o) + logpdf(b2, y2 - o)

    # A missing branch drops out: the joint reduces to the single-branch
    # primary-censored marginal.
    evd = Vector{Union{Missing, Float64}}([missing, y1, missing])
    @test logpdf(par, evd) ≈ logpdf(primary_censored(b1, pe), y1) rtol=1e-6
end

@testitem "Parallel branch of primary_censored(Sequential) flat path (#363)" begin
    using Distributions, Random, Statistics

    # `primary_censored(Sequential(...))` COLLAPSES the chain to a
    # `PrimaryCensored{Convolved}` leaf, so the Parallel sits on the FLAT
    # shared-origin path. The flat path strips each branch to its
    # `_marginal_core`; the regression is that the nested `Convolved` must stay
    # intact (not unwrap into its component VECTOR) so `rand`/`logpdf`/the
    # `_param_eltype` machinery see a distribution, not a vector (#363).
    pe = Uniform(0.0, 1.0)
    coreA = convolve_distributions(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    seqbranch = primary_censored(
        Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4)), pe)
    d = Parallel((seqbranch, Exponential(1.0)), (:a, :b))
    @test CensoredDistributions._nested_trait(d.components) isa
          CensoredDistributions._Flat
    # The collapsed branch's marginal core is the `Convolved`, kept intact.
    @test CensoredDistributions._marginal_core(seqbranch) isa
          CensoredDistributions.Convolved

    # Conditional logpdf == the hand factorisation over the shared origin.
    o, ya, yb = 0.5, 3.0, 1.2
    ev = Vector{Union{Missing, Float64}}([o, ya, yb])
    @test logpdf(d, ev) ≈
          logpdf(pe, o) +
          logpdf(coreA, ya - o) + logpdf(Exponential(1.0), yb - o)

    # Marginal (missing origin) == the 1-D integral over the shared origin.
    ev_m = Vector{Union{Missing, Float64}}([missing, ya, yb])
    ref_m = CensoredDistributions.gl_integrate(
        0.0, 1.0, CensoredDistributions._PRIMARY_GL) do u
        pdf(pe, u) * pdf(coreA, ya - u) * pdf(Exponential(1.0), yb - u)
    end
    @test logpdf(d, ev_m) ≈ log(ref_m)

    # rand draws the proper convolution sum on branch a (NOT a random component):
    # the recorded branch-a delay matches the `Convolved` core in mean and var.
    Random.seed!(202)
    N = 200_000
    origins = Vector{Float64}(undef, N)
    delaysA = Vector{Float64}(undef, N)
    for i in 1:N
        r = rand(d)
        origins[i] = r.event_1
        delaysA[i] = r.event_2 - r.event_1
    end
    @test mean(origins) ≈ mean(pe) rtol=2e-2
    @test mean(delaysA) ≈ mean(coreA) rtol=2e-2
    @test var(delaysA) ≈ var(coreA) rtol=4e-2
    # Every branch endpoint is after the shared origin.
    @test all(>=(0), delaysA)

    # `double_interval_censored(Sequential)` is the same collapse plus a
    # secondary interval; it also rides the flat path and scores finitely.
    dbranch = double_interval_censored(
        Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4));
        primary_event = pe, interval = 1.0)
    d2 = Parallel((dbranch, Exponential(1.0)), (:a, :b))
    @test isfinite(logpdf(d2, ev))
    Random.seed!(303)
    r2 = rand(d2)
    @test r2.event_2 >= r2.event_1            # endpoint after origin
    @test r2.event_2 == floor(r2.event_2)     # day-resolution interval applied
end

@testitem "Parallel primary_censored(Sequential) branch is ForwardDiff-safe" begin
    using Distributions, ForwardDiff

    # AD over the collapsed branch's leaf params must flow the `Dual` THROUGH the
    # nested `Convolved` core (the `_param_eltype`/`_flatten_params` promotion):
    # a vector core would drop the `Dual` and `params(::Vector)` would error.
    ev = Vector{Union{Missing, Float64}}([0.5, 3.0, 1.2])
    ev_m = Vector{Union{Missing, Float64}}([missing, 3.0, 1.2])
    function f(theta, events)
        seqbranch = primary_censored(
            Sequential(Gamma(theta[1], theta[2]),
                LogNormal(theta[3], theta[4])), Uniform(0, 1))
        d = Parallel((seqbranch, Exponential(theta[5])), (:a, :b))
        return logpdf(d, events)
    end
    theta0 = [2.0, 1.0, 0.5, 0.4, 1.0]
    g = ForwardDiff.gradient(t -> f(t, ev), theta0)
    @test all(isfinite, g)
    @test any(!iszero, g)
    g_m = ForwardDiff.gradient(t -> f(t, ev_m), theta0)
    @test all(isfinite, g_m)
    @test any(!iszero, g_m)
end

@testitem "Parallel plain branches condition on an observed origin" begin
    using Distributions

    b1 = Gamma(2.0, 1.0)
    b2 = LogNormal(1.0, 0.5)
    par = Parallel(b1, b2)
    o, y1, y2 = 0.4, 2.0, 3.0

    # Observed shared origin: each branch conditions on its gap, no primary
    # prior and no integral.
    ev = Vector{Union{Missing, Float64}}([o, y1, y2])
    @test logpdf(par, ev) ≈ logpdf(b1, y1 - o) + logpdf(b2, y2 - o)
    @test pdf(par, ev) ≈ exp(logpdf(par, ev))

    # A missing branch drops from the joint.
    evd = Vector{Union{Missing, Float64}}([o, y1, missing])
    @test logpdf(par, evd) ≈ logpdf(b1, y1 - o)

    # An out-of-support gap (origin after the observation) is -Inf.
    evx = Vector{Union{Missing, Float64}}([5.0, y1, y2])
    @test logpdf(par, evx) == -Inf
end

@testitem "Parallel plain branches need an observed origin" begin
    using Distributions

    par = Parallel(Gamma(2.0, 1.0), LogNormal(1.0, 0.5))
    ev = Vector{Union{Missing, Float64}}([missing, 2.0, 3.0])
    @test_throws ArgumentError logpdf(par, ev)
end

@testitem "Parallel plain-branch conditioning is type-stable" begin
    using Distributions

    par = Parallel(Gamma(2.0, 1.0), LogNormal(1.0, 0.5))
    ev = Vector{Union{Missing, Float64}}([0.4, 2.0, 3.0])
    @test (@inferred logpdf(par, ev)) isa Float64
end

@testitem "primary_censored(Parallel) wrapper == per-branch primary form" begin
    using Distributions, Random
    Random.seed!(7)

    pe = Uniform(0.0, 1.0)
    d1 = Gamma(2.0, 1.0)
    d2 = LogNormal(1.0, 0.5)

    # Notation invariance: declaring one shared primary by WRAPPING the Parallel
    # must give the SAME log density as the per-branch primary form (one shared
    # 1-D integral over the common primary, not independent per-branch integrals).
    perbranch = Parallel(primary_censored(d1, pe), primary_censored(d2, pe))
    wrapper = primary_censored(Parallel(d1, d2), pe)

    for ev in (Vector{Union{Missing, Float64}}([missing, 2.3, 3.1]),
        Vector{Union{Missing, Float64}}([0.4, 2.3, 3.1]),
        Vector{Union{Missing, Float64}}([missing, 2.3, missing]))
        @test logpdf(wrapper, ev) ≈ logpdf(perbranch, ev) rtol=1e-10
    end
end

@testitem "double_interval_censored(Parallel) shares one primary" begin
    using Distributions

    pe = Uniform(0.0, 1.0)
    d1 = Gamma(2.0, 1.0)
    d2 = LogNormal(1.0, 0.5)

    perbranch = Parallel(
        double_interval_censored(d1; primary_event = pe),
        double_interval_censored(d2; primary_event = pe))
    wrapper = double_interval_censored(Parallel(d1, d2); primary_event = pe)

    ev = Vector{Union{Missing, Float64}}([missing, 2.0, 3.0])
    @test logpdf(wrapper, ev) ≈ logpdf(perbranch, ev) rtol=1e-10
end

@testitem "Resolve recovers a delay-adjusted CFR" begin
    using Distributions, Random
    Random.seed!(7)

    true_cfr = 0.35
    death_delay = Gamma(2.0, 3.0)   # slow deaths
    disch_delay = Gamma(2.0, 1.0)   # fast discharges
    horizon = 8.0
    n = 300_000

    outcomes = Symbol[]
    times = Float64[]
    for _ in 1:n
        if rand() < true_cfr
            push!(outcomes, :death)
            push!(times, rand(death_delay))
        else
            push!(outcomes, :disch)
            push!(times, rand(disch_delay))
        end
    end
    resolved = times .<= horizon

    # Naive ratio among resolved cases is biased low (deaths resolve slower).
    naive = count(i -> resolved[i] && outcomes[i] == :death, 1:n) /
            count(resolved)
    @test naive < true_cfr - 0.02

    # The Resolve branch probability that maximises the right-truncated
    # mixture likelihood recovers the true CFR.
    obs_o = outcomes[resolved]
    obs_t = times[resolved]
    function negll(c)
        dd = truncated(death_delay; upper = horizon)
        rd = truncated(disch_delay; upper = horizon)
        pres = c * cdf(death_delay, horizon) +
               (1 - c) * cdf(disch_delay, horizon)
        ll = 0.0
        for (o, t) in zip(obs_o, obs_t)
            num = o == :death ?
                  c * cdf(death_delay, horizon) * pdf(dd, t) :
                  (1 - c) * cdf(disch_delay, horizon) * pdf(rd, t)
            ll += log(num / pres)
        end
        return -ll
    end
    grid = 0.01:0.005:0.99
    adj = grid[argmin([negll(c) for c in grid])]
    @test adj ≈ true_cfr atol=0.01

    # The Resolve node lowers to the CFR-weighted mixture (PR3a as_mixture).
    node = Resolve(:death => (death_delay, true_cfr),
        :disch => (disch_delay, 1 - true_cfr))
    @test pdf(node, 4.0) ≈
          true_cfr * pdf(death_delay, 4.0) +
          (1 - true_cfr) * pdf(disch_delay, 4.0)
end

@testitem "Sequential handles varying per-record missingness in a batch" begin
    using Distributions

    pe = Uniform(0, 1)
    d1 = LogNormal(1.2, 0.5)
    d2 = Gamma(2.0, 1.0)
    chain = Sequential(primary_censored(d1, pe), primary_censored(d2, pe))

    # A batch of records with different observed/missing patterns over the SAME
    # 2-step chain; per-record dispatch handles each from its own missingness,
    # not a global mode. Every record is one entry per event (length 3).
    records = [
        Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0]),       # all observed
        Vector{Union{Missing, Float64}}([0.0, missing, 4.0]),   # E1 unobserved
        Vector{Union{Missing, Float64}}([1.0, 3.0, 6.0])        # shifted origin
    ]
    lps = logpdf.(Ref(chain), records)
    @test all(isfinite, lps)
    # All observed: each edge scores through its own (censored) logpdf at its gap.
    @test lps[1] ≈
          logpdf(primary_censored(d1, pe), 2.0) +
          logpdf(primary_censored(d2, pe), 3.0) rtol=1e-6
    # E1 unobserved: marginalise by convolving the cores, origin primary applied.
    @test lps[2] ≈
          logpdf(primary_censored(convolve_distributions(d1, d2), pe), 4.0) rtol=1e-6
    # Shifted origin: gaps are E1-E0 = 2.0 and E2-E1 = 3.0, same as record 1.
    @test lps[3] ≈ lps[1] rtol=1e-6
end

@testitem "rand simulates the full event-time path of a censored chain" begin
    using Distributions, Random
    Random.seed!(11)

    pe = Uniform(0, 1)
    chain = Sequential(primary_censored(LogNormal(1.2, 0.5), pe),
        primary_censored(Gamma(2.0, 1.0), pe))

    # Full event-time path [E0, E1, E2] as a labelled NamedTuple: origin draw
    # plus cumulative delays, so every internal event time is returned.
    r = rand(chain)
    @test r isa NamedTuple
    @test length(r) == 3
    rv = collect(values(r))
    @test issorted(rv)             # monotone: each step adds a positive delay
    # The simulated all-observed labelled path scores finitely under the chain
    # (the NamedTuple is matched to the event vector by name internally).
    @test isfinite(logpdf(chain, r))
end

@testitem "rand simulates the shared origin and every branch of a Parallel" begin
    using Distributions, Random
    Random.seed!(12)

    pe = Uniform(0, 1)
    par = Parallel(primary_censored(Gamma(2.0, 1.0), pe),
        primary_censored(LogNormal(1.0, 0.5), pe))

    # Full event-time record [O, Y1, Y2] as a labelled NamedTuple: one shared
    # origin draw then each branch observation as origin plus a branch delay.
    r = rand(par)
    @test r isa NamedTuple
    @test length(r) == 3
    rv = collect(values(r))
    @test all(rv[2:end] .>= rv[1])  # each observation is after the shared origin
end

@testitem "rand on plain composers keeps the per-leaf realisation" begin
    using Distributions, Random
    Random.seed!(13)

    # No primary censoring: the generic per-leaf-value realisation (PR3a) is kept.
    @test length(rand(Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4)))) == 2
    @test length(rand(Parallel(Gamma(2.0, 1.0), LogNormal(1.0, 0.5)))) == 2
end

@testitem "double_interval_censored origin surfaces its primary event" begin
    using Distributions

    pe = Uniform(0, 1)
    # An origin edge wrapped as double_interval_censored still surfaces its
    # primary event through the censoring wrappers for the first segment.
    origin = double_interval_censored(LogNormal(1.0, 0.5);
        primary_event = pe, upper = 20.0, interval = 1.0)
    d2 = Gamma(2.0, 1.0)
    chain = Sequential(origin, primary_censored(d2, pe))

    # E1 unobserved: convolution of the continuous cores, primary-censored.
    ev = Vector{Union{Missing, Float64}}([0.0, missing, 5.0])
    ref = primary_censored(
        convolve_distributions(LogNormal(1.0, 0.5), d2), pe)
    @test logpdf(chain, ev) ≈ logpdf(ref, 5.0) rtol=1e-6
end

# --- Recursive nested-composer (irregular tree) scoring (#329, #333, #345) ---

@testitem "Nested tree: a Parallel step inside a Sequential recurses" begin
    using Distributions

    edge(mu,
        sigma) = double_interval_censored(LogNormal(mu, sigma);
        primary_event = Uniform(0, 1), interval = 1.0)
    e_oa = edge(1.4, 0.4)
    e_ad = edge(1.8, 0.5)
    e_ac = edge(2.0, 0.6)

    # onset -> admit -> {death, discharge}: death/discharge share the ADMIT
    # origin. One composed distribution scored against the 4-event row.
    d = Sequential(e_oa, Parallel(e_ad, e_ac))
    @test length(d) == 3            # 3 leaf events; event vector is length 4
    onset, admit, death, disch = 0.0, 4.0, 12.0, 11.0
    ev = Vector{Union{Missing, Float64}}([onset, admit, death, disch])

    # Per-pathway-chain reference: each edge's own censored logpdf, origin
    # reapplied (admit origin for the two branch edges).
    ref = logpdf(e_oa, admit - onset) +
          logpdf(e_ad, death - admit) +
          logpdf(e_ac, disch - admit)
    @test logpdf(d, ev) ≈ ref rtol=1e-10
    @test pdf(d, ev) ≈ exp(ref) rtol=1e-10
end

@testitem "Nested tree: a Sequential branch inside a Parallel recurses" begin
    using Distributions

    edge(mu,
        sigma) = double_interval_censored(LogNormal(mu, sigma);
        primary_event = Uniform(0, 1), interval = 1.0)
    e_oa = edge(1.4, 0.4)
    e_ad = edge(1.8, 0.5)
    e_on = edge(1.9, 0.5)

    # Parallel(Sequential(e_oa, e_ad), e_on): a chain branch and a leaf branch
    # off the same onset origin. Event layout [onset, admit, X, notif].
    d = Parallel(Sequential(e_oa, e_ad), e_on)
    @test length(d) == 3
    onset, admit, x, notif = 0.0, 4.0, 12.0, 9.0
    ev = Vector{Union{Missing, Float64}}([onset, admit, x, notif])
    ref = logpdf(e_oa, admit - onset) + logpdf(e_ad, x - admit) +
          logpdf(e_on, notif - onset)
    @test logpdf(d, ev) ≈ ref rtol=1e-10
end

@testitem "bdbv two-level tree == per-pathway-chain reference" begin
    using Distributions

    edge(mu,
        sigma) = double_interval_censored(LogNormal(mu, sigma);
        primary_event = Uniform(0, 1), interval = 1.0)
    e_oa = edge(1.4, 0.4)
    e_ad = edge(1.8, 0.5)
    e_ac = edge(2.0, 0.6)
    e_on = edge(1.9, 0.5)

    # Full bdbv tree: onset -> {admit -> {death, discharge}, notif}.
    # Parallel(Sequential(e_oa, Parallel(e_ad, e_ac)), e_on).
    d = Parallel(Sequential(e_oa, Parallel(e_ad, e_ac)), e_on)
    onset, admit, death, disch, notif = 0.0, 4.0, 12.0, 11.0, 9.0
    ev = Vector{Union{Missing, Float64}}([onset, admit, death, disch, notif])

    # Each edge's contribution equals its own censored logpdf, the ADMIT origin
    # reapplied to the death/discharge edges (not the onset origin).
    ref = logpdf(e_oa, admit - onset) +
          logpdf(e_ad, death - admit) +
          logpdf(e_ac, disch - admit) +
          logpdf(e_on, notif - onset)
    @test logpdf(d, ev) ≈ ref rtol=1e-10
end

@testitem "Three-level irregular tree A -> {B->F, C -> {D, E}} scores" begin
    using Distributions

    edge(mu,
        sigma) = double_interval_censored(LogNormal(mu, sigma);
        primary_event = Uniform(0, 1), interval = 1.0)
    e_ab = edge(1.0, 0.3)
    e_bf = edge(0.8, 0.4)
    e_ac = edge(1.2, 0.5)
    e_cd = edge(0.9, 0.3)
    e_ce = edge(1.1, 0.4)

    # Heterogeneous children under A: a chain (B -> F) and a branch (C -> {D, E}).
    d = Parallel(
        Sequential(e_ab, e_bf),
        Sequential(e_ac, Parallel(e_cd, e_ce)))
    @test length(d) == 5            # 5 leaf events; event vector length 6
    A, B, F, C, D, E = 0.0, 3.0, 5.5, 3.5, 6.0, 7.0
    ev = Vector{Union{Missing, Float64}}([A, B, F, C, D, E])
    ref = logpdf(e_ab, B - A) + logpdf(e_bf, F - B) +
          logpdf(e_ac, C - A) + logpdf(e_cd, D - C) + logpdf(e_ce, E - C)
    @test logpdf(d, ev) ≈ ref rtol=1e-10
end

@testitem "Nested tree logpdf is type-stable" begin
    using Distributions, Test

    edge(mu,
        sigma) = double_interval_censored(LogNormal(mu, sigma);
        primary_event = Uniform(0, 1), interval = 1.0)
    e_oa = edge(1.4, 0.4)
    e_ad = edge(1.8, 0.5)
    e_ac = edge(2.0, 0.6)
    e_on = edge(1.9, 0.5)

    d = Parallel(Sequential(e_oa, Parallel(e_ad, e_ac)), e_on)
    ev = Vector{Union{Missing, Float64}}([0.0, 4.0, 12.0, 11.0, 9.0])
    @test (@inferred logpdf(d, ev)) isa Float64
end

# Per-backend AD gradient correctness for the nested tree (ForwardDiff /
# ReverseDiff / Mooncake pass; Enzyme is the heterogeneous-edge gap #319) is
# covered by the dedicated AD fixture suite (`test/ADFixtures`, scenario
# "Nested tree censored observed logpdf") with the proper AD-backend deps.

# --- Nested Resolve self-dispatch (#333) ----------------------------------

@testitem "Nested Resolve: outcome event names anchor at the parent" begin
    using Distributions

    edge(mu,
        sigma) = double_interval_censored(LogNormal(mu, sigma);
        primary_event = Uniform(0, 1), interval = 1.0)
    cmp = Resolve(:death => (Gamma(2.0, 3.0), 0.3),
        :discharge => (Gamma(2.0, 1.0), 0.7))
    # bdbv: onset -> {admit -> Resolve(death, discharge), notif}.
    d = Parallel(Sequential(edge(1.4, 0.4), cmp), edge(1.9, 0.5))
    enames = event_names(d)
    # The Resolve contributes one EVENT slot per OUTCOME, not one opaque
    # resolution event: death/discharge appear, anchored after the admit event.
    @test :death in enames
    @test :discharge in enames
    # Event-slot count counts the outcomes (value-vector length() is unchanged).
    @test CensoredDistributions._event_nleaves(d.components) == 4
    @test length(d) == 3
end

@testitem "Nested Resolve: conditions on the observed outcome (#333)" begin
    using Distributions

    edge(mu,
        sigma) = double_interval_censored(LogNormal(mu, sigma);
        primary_event = Uniform(0, 1), interval = 1.0)
    e_oa, e_on = edge(1.4, 0.4), edge(1.9, 0.5)
    cfr = 0.3
    death_d, disch_d = Gamma(2.0, 3.0), Gamma(2.0, 1.0)
    cmp = Resolve(:death => (death_d, cfr), :discharge => (disch_d, 1 - cfr))
    d = Parallel(Sequential(e_oa, cmp), e_on)

    onset, admit, notif = 0.0, 4.0, 9.0
    # Event vector [onset, admit, death, discharge, notif]; death observed.
    ev = Vector{Union{Missing, Float64}}([onset, admit, 12.0, missing, notif])
    # Hand reference: condition the observed death branch
    # (log p_death + its delay logpdf at the gap from admit), each other edge on
    # its own declared censoring.
    ref = logpdf(e_oa, admit - onset) +
          log(cfr) + logpdf(death_d, 12.0 - admit) +
          logpdf(e_on, notif - onset)
    @test logpdf(d, ev) ≈ ref rtol=1e-10

    # Discharge observed instead selects the discharge branch.
    evd = Vector{Union{Missing, Float64}}([onset, admit, missing, 11.0, notif])
    refd = logpdf(e_oa, admit - onset) +
           log(1 - cfr) + logpdf(disch_d, 11.0 - admit) +
           logpdf(e_on, notif - onset)
    @test logpdf(d, evd) ≈ refd rtol=1e-10

    # No outcome observed -> the Resolve contributes no factor (the resolved-
    # but-unknown-outcome encoding for a nested node is deferred, #329).
    evm = Vector{Union{Missing, Float64}}(
        [onset, admit, missing, missing, notif])
    @test logpdf(d, evm) ≈
          logpdf(e_oa, admit - onset) + logpdf(e_on, notif - onset) rtol=1e-10

    # Two observed outcomes is an error (at most one resolves).
    evt = Vector{Union{Missing, Float64}}([onset, admit, 12.0, 11.0, notif])
    @test_throws ArgumentError logpdf(d, evt)
end

@testitem "Nested Resolve: type-stable scoring" begin
    using Distributions, Test

    edge(mu,
        sigma) = double_interval_censored(LogNormal(mu, sigma);
        primary_event = Uniform(0, 1), interval = 1.0)
    cmp = Resolve(:death => (Gamma(2.0, 3.0), 0.3),
        :discharge => (Gamma(2.0, 1.0), 0.7))
    d = Parallel(Sequential(edge(1.4, 0.4), cmp), edge(1.9, 0.5))
    ev = Vector{Union{Missing, Float64}}([0.0, 4.0, 12.0, missing, 9.0])
    @test (@inferred logpdf(d, ev)) isa Float64
end

@testitem "Nested Resolve: N-ary (three outcomes) self-dispatch (#333)" begin
    using Distributions

    edge(mu,
        sigma) = double_interval_censored(LogNormal(mu, sigma);
        primary_event = Uniform(0, 1), interval = 1.0)
    e_oa, e_on = edge(1.4, 0.4), edge(1.9, 0.5)
    death_d, disch_d, trans_d = Gamma(2.0, 3.0), Gamma(2.0, 1.0), Gamma(3.0, 1.0)
    cmp = Resolve(:death => (death_d, 0.2),
        :discharge => (disch_d, 0.5),
        :transfer => (trans_d, 0.3))
    d = Parallel(Sequential(e_oa, cmp), e_on)

    enames = event_names(d)
    @test :death in enames
    @test :discharge in enames
    @test :transfer in enames
    @test CensoredDistributions._event_nleaves(d.components) == 5

    onset, admit, notif = 0.0, 4.0, 9.0
    # Event vector [onset, admit, death, discharge, transfer, notif]; the THIRD
    # outcome (transfer) is the one observed.
    ev = Vector{Union{Missing, Float64}}(
        [onset, admit, missing, missing, 10.0, notif])
    ref = logpdf(e_oa, admit - onset) +
          log(0.3) + logpdf(trans_d, 10.0 - admit) +
          logpdf(e_on, notif - onset)
    @test logpdf(d, ev) ≈ ref rtol=1e-10

    # The first outcome (death) observed instead.
    evd = Vector{Union{Missing, Float64}}(
        [onset, admit, 8.0, missing, missing, notif])
    refd = logpdf(e_oa, admit - onset) +
           log(0.2) + logpdf(death_d, 8.0 - admit) +
           logpdf(e_on, notif - onset)
    @test logpdf(d, evd) ≈ refd rtol=1e-10
end

# --- Nested/Resolve-aware full-path rand ----------------------------------

@testitem "rand on a nested tree returns a named shared-origin path" begin
    using Distributions, Random
    Random.seed!(101)

    edge(mu,
        sigma) = double_interval_censored(LogNormal(mu, sigma);
        primary_event = Uniform(0, 1), interval = 1.0)
    e_oa = edge(1.4, 0.4)
    e_ad = edge(1.8, 0.5)
    e_on = edge(1.9, 0.5)

    # Parallel(Sequential(onset->admit), notif): a chain branch + a leaf branch
    # share one onset origin. Named edges so the record is semantically keyed.
    seq = Sequential((e_oa, e_ad), (:onset_admit, :admit_x))
    d = Parallel((seq, e_on), (:onset_x_seq, :onset_notif))
    enames = event_names(d)

    r = rand(d)
    @test r isa NamedTuple
    @test keys(r) == enames                     # named by the tree's events
    origin = r[enames[1]]
    @test all(r[k] >= origin for k in enames)   # every event after the origin
    # The simulated full path scores finitely under the same tree.
    ev = Vector{Union{Missing, Float64}}([r[k] for k in enames])
    @test isfinite(logpdf(d, ev))
end

@testitem "rand on a bdbv Resolve tree samples one outcome" begin
    using Distributions, Random

    edge(mu,
        sigma) = double_interval_censored(LogNormal(mu, sigma);
        primary_event = Uniform(0, 1), interval = 1.0)
    e_oa = edge(1.4, 0.4)
    e_on = edge(1.9, 0.5)
    cmp = Resolve(:death => (edge(2.0, 0.6), 0.3),
        :discharge => (edge(2.1, 0.6), 0.7))
    # bdbv: Parallel(Sequential(onset_admit, Resolve(death, discharge)), notif)
    seq = Sequential((e_oa, cmp), (:onset_admit, :admit_resolve))
    d = Parallel((seq, e_on), (:onset_notif_seq, :onset_notif))
    enames = event_names(d)
    @test enames == (:onset, :admit, :death, :discharge, :notif)

    r = rand(Xoshiro(7), d)
    @test keys(r) == enames
    @test r.onset !== missing && r.admit !== missing && r.notif !== missing
    # Exactly one Resolve outcome is resolved; the other is missing.
    outcomes = (r.death, r.discharge)
    @test count(!ismissing, outcomes) == 1
    # The resolved outcome hangs off the admit event.
    resolved = r.death === missing ? r.discharge : r.death
    @test resolved >= r.admit
    # The named record round-trips through logpdf.
    ev = Vector{Union{Missing, Float64}}([r[k] for k in enames])
    @test isfinite(logpdf(d, ev))
end

@testitem "rand Resolve outcome frequencies follow branch probs" begin
    using Distributions, Random

    edge(mu,
        sigma) = double_interval_censored(LogNormal(mu, sigma);
        primary_event = Uniform(0, 1), interval = 1.0)
    e_oa = edge(1.4, 0.4)
    e_on = edge(1.9, 0.5)
    cfr = 0.25
    cmp = Resolve(:death => (edge(2.0, 0.6), cfr),
        :discharge => (edge(2.1, 0.6), 1 - cfr))
    seq = Sequential((e_oa, cmp), (:onset_admit, :admit_resolve))
    d = Parallel((seq, e_on), (:onset_notif_seq, :onset_notif))

    rng = Xoshiro(42)
    n = 4000
    deaths = count(_ -> rand(rng, d).death !== missing, 1:n)
    @test isapprox(deaths / n, cfr; atol = 0.03)
end

@testitem "rand on a Choose top samples a branch's path" begin
    using Distributions, Random

    edge(mu,
        sigma) = double_interval_censored(LogNormal(mu, sigma);
        primary_event = Uniform(0, 1), interval = 1.0)
    # An `index` alternative (a flat chain) and a `sourced` alternative (a nested
    # tree that yields a named record), selected by the row's kind.
    idx = Sequential(edge(1.4, 0.4), edge(1.8, 0.5))
    src = Parallel(
        (Sequential((edge(1.2, 0.4), edge(1.6, 0.5)),
                (:onset_admit, :admit_x)),
            edge(1.9, 0.5)),
        (:onset_x_seq, :onset_notif))
    sel = choose(:index => idx, :sourced => src)

    # With a kind, the selected branch's own labelled draw (both the flat index
    # chain and the nested sourced tree now yield a NamedTuple event record).
    rs = rand(Xoshiro(3), sel; kind = :sourced)
    @test rs isa NamedTuple
    @test keys(rs) == event_names(src)
    ri = rand(Xoshiro(3), sel; kind = :index)
    @test ri isa NamedTuple && length(ri) == 3
    # Without a kind (forward simulation), a branch is sampled.
    r = rand(Xoshiro(5), sel)
    @test r isa NamedTuple
end

@testitem "rand simulates a labelled record over a nested tree" begin
    using Distributions, Random

    edge(mu,
        sigma) = double_interval_censored(LogNormal(mu, sigma);
        primary_event = Uniform(0, 1), interval = 1.0)
    cmp = Resolve(:death => (edge(2.0, 0.6), 0.3),
        :discharge => (edge(2.1, 0.6), 0.7))
    seq = Sequential((edge(1.4, 0.4), cmp), (:onset_admit, :admit_resolve))
    d = Parallel((seq, edge(1.9, 0.5)), (:onset_notif_seq, :onset_notif))

    # `rand(latent(d))` forward-simulates a full labelled event path; `rand(d)`
    # is the same labelled record (a composed tree already rands the full path).
    a = rand(Xoshiro(9), d)
    @test a isa NamedTuple
    @test keys(a) == event_names(d)
    # A batch of independent draws shares the same event labels.
    rng = Xoshiro(9)
    paths = [rand(rng, d) for _ in 1:5]
    @test length(paths) == 5
    @test all(p -> keys(p) == keys(a), paths)
end

@testitem "rand on a nested tree walk is type-stable" begin
    using Distributions, Random, Test

    edge(mu,
        sigma) = double_interval_censored(LogNormal(mu, sigma);
        primary_event = Uniform(0, 1), interval = 1.0)
    cmp = Resolve(:death => (edge(2.0, 0.6), 0.3),
        :discharge => (edge(2.1, 0.6), 0.7))
    seq = Sequential((edge(1.4, 0.4), cmp), (:onset_admit, :admit_resolve))
    d = Parallel((seq, edge(1.9, 0.5)), (:onset_notif_seq, :onset_notif))

    # The sampling walk returns a concrete typed event vector.
    v = @inferred CensoredDistributions._tree_event_vector(Xoshiro(1), d)
    @test v isa Vector{Union{Missing, Float64}}
end

@testitem "rand_outcome retains the resolved Resolve outcome" begin
    using Distributions, Random

    cmp = Resolve(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))
    name, t = CensoredDistributions.rand_outcome(Xoshiro(1), cmp)
    @test name in (:death, :disch)
    @test t > 0
    # Frequencies follow the branch probabilities.
    rng = Xoshiro(5)
    n = 4000
    deaths = count(_ -> CensoredDistributions.rand_outcome(rng, cmp)[1] ==
                        :death, 1:n)
    @test isapprox(deaths / n, 0.3; atol = 0.03)
end

@testitem "rand discretises event times to each leaf's interval" begin
    using Distributions, Random

    pe = Uniform(0, 1)
    edge(m) = double_interval_censored(Gamma(m, 1.0);
        primary_event = pe, interval = 1.0)

    # bdbv tree: onset -> {admit -> {death, discharge}, notif}.
    tree = Parallel(
        Sequential(edge(3.6), Parallel(edge(7.0), edge(7.5))), edge(14.0))
    r = rand(Xoshiro(1), tree)
    v = collect(values(r))
    # The origin and every leaf slot are floored to the day interval.
    @test all(x -> x === missing || x == floor(x), v)
end

@testitem "simulate-then-score gap means are unbiased (bdbv tree)" begin
    using Distributions, Random, Statistics

    # The scorer floors each gap to its day; the sim must apply the same interval
    # to the RECORDED times so E[gap] == E[Gamma], not E[Gamma] - 0.5. The origin
    # is floored to the origin edge's interval so first-level gaps are unbiased.
    pe = Uniform(0, 1)
    edge(m) = double_interval_censored(Gamma(m, 1.0);
        primary_event = pe, interval = 1.0)
    am, dm, nm = 3.6, 7.0, 14.0
    # onset -> {admit -> {death, discharge}, notif}; notif hangs off onset.
    tree = Parallel(
        Sequential(edge(am), Parallel(edge(dm), edge(7.5))), edge(nm))

    rng = Xoshiro(20)
    N = 200_000
    adm = Float64[];
    dth = Float64[];
    ntf = Float64[]
    for _ in 1:N
        v = collect(values(rand(rng, tree)))   # [onset, admit, death, disch, notif]
        push!(adm, v[2] - v[1])                # admit - onset
        push!(dth, v[3] - v[2])                # death - admit
        push!(ntf, v[5] - v[1])                # notif - onset
    end
    @test isapprox(mean(adm), am; atol = 0.05)
    @test isapprox(mean(dth), dm; atol = 0.05)
    @test isapprox(mean(ntf), nm; atol = 0.05)
end

@testitem "simulate-then-score gap is unbiased for a single censored leaf" begin
    using Distributions, Random, Statistics

    pe = Uniform(0, 1)
    m = 7.0
    chain = Sequential(
        double_interval_censored(LogNormal(1.0, 0.4);
            primary_event = pe, interval = 1.0),
        double_interval_censored(Gamma(m, 1.0);
            primary_event = pe, interval = 1.0))
    rng = Xoshiro(30)
    gaps = [(r = collect(rand(rng, chain)); r[3] - r[2])
            for _ in 1:200_000]
    # E[gap] == E[Gamma], not E[Gamma] - 0.5.
    @test isapprox(mean(gaps), m; atol = 0.05)
end

@testitem "rand discretises mixed leaf intervals correctly" begin
    using Distributions, Random

    pe = Uniform(0, 1)
    # Step 1 is day-resolution, step 2 two-day-resolution.
    mix = Sequential(
        double_interval_censored(Gamma(5.0, 1.0);
            primary_event = pe, interval = 1.0),
        double_interval_censored(Gamma(8.0, 1.0);
            primary_event = pe, interval = 2.0))
    rng = Xoshiro(40)
    for _ in 1:2000
        r = collect(rand(rng, mix))
        @test r[2] == floor(r[2])        # step 1: day floor
        @test r[3] % 2 == 0              # step 2: two-day floor
    end
end

@testitem "rand keeps primary-only leaves continuous" begin
    using Distributions, Random

    pe = Uniform(0, 1)
    # No secondary interval anywhere: every slot stays continuous.
    chain = Sequential(primary_censored(Gamma(4.0, 1.0), pe),
        primary_censored(Gamma(3.0, 1.0), pe))
    rng = Xoshiro(50)
    saw_noninteger = any(1:200) do _
        r = collect(rand(rng, chain))
        any(x -> x != floor(x), r)
    end
    @test saw_noninteger
end

@testitem "event_logpdf observed-intermediate whole-compose (#366)" begin
    using Distributions

    # Whole-compose TOTAL truncation of an OBSERVED-intermediate Sequential
    # record (#366): the factorised per-segment numerator over a single
    # conv-to-last-observed denominator. Equals the hand-rolled per-record
    # decomposition (the andv index-vs-sourced terms generalised to an observed
    # intermediate).
    pe = Uniform(0, 1)
    inc = primary_censored(LogNormal(1.2, 0.5), pe)
    delta = primary_censored(Gamma(2.0, 1.0), pe)
    seq = Sequential((inc, delta), (:onset_admit, :admit_death))

    D = 8.0
    o, a, dth = 0.0, 2.0, 5.0
    ev = Vector{Union{Missing, Float64}}([o, a, dth])

    # Numerator: each observed segment conditions on its own edge (unchanged
    # from the untruncated factorisation).
    numerator = logpdf(inc, a - o) + logpdf(delta, dth - a)
    @test logpdf(seq, ev) ≈ numerator

    # Denominator: conv of ALL components origin -> last observed event,
    # evaluated at window = D - origin.
    conv = CensoredDistributions._sequential_segment(seq.components, 1, 3, pe)
    expected = numerator - logcdf(conv, D - o)
    @test CensoredDistributions.event_logpdf(seq, ev; horizon = D) ≈ expected

    # The horizon adds exactly the conv-to-last-observed right-truncation term.
    @test CensoredDistributions.event_logpdf(seq, ev; horizon = D) ≈
          logpdf(seq, ev) - logcdf(conv, D - o)

    # A non-zero observed origin shifts the window to D - origin.
    o2 = 1.5
    ev2 = Vector{Union{Missing, Float64}}([o2, o2 + 2.0, o2 + 5.0])
    num2 = logpdf(inc, 2.0) + logpdf(delta, 3.0)
    @test CensoredDistributions.event_logpdf(seq, ev2; horizon = D) ≈
          num2 - logcdf(conv, D - o2)
end

@testitem "event_logpdf whole-compose reduces to endpoint-observed (#366)" begin
    using Distributions

    # With the intermediate UNOBSERVED the chain has a single observed segment,
    # so the conv-to-last-observed denominator IS that segment and the result
    # reduces to the endpoint-observed hanta total truncation.
    pe = Uniform(0, 1)
    inc = primary_censored(LogNormal(1.5, 0.5), pe)
    delta = primary_censored(Gamma(2.0, 1.0), pe)
    seq = Sequential((inc, delta), (:onset_mid, :mid_obs))
    D = 6.0

    ev = Vector{Union{Missing, Float64}}([0.0, missing, 5.0])
    seg = CensoredDistributions._sequential_segment(seq.components, 1, 3, pe)
    @test CensoredDistributions.event_logpdf(seq, ev; horizon = D) ≈
          logpdf(seg, 5.0) - logcdf(seg, D)
end

@testitem "event_logpdf whole-compose three-segment patterns (#366)" begin
    using Distributions

    # A three-step chain with the middle two intermediates: an observed middle
    # factorises into two segments, an unobserved middle convolves its run. The
    # whole-compose denominator is always conv-to-last-observed at D - origin.
    pe = Uniform(0, 1)
    e1 = primary_censored(LogNormal(1.0, 0.4), pe)
    e2 = primary_censored(Gamma(2.0, 0.8), pe)
    e3 = primary_censored(Gamma(1.5, 1.0), pe)
    seq = Sequential((e1, e2, e3), (:a_b, :b_c, :c_d))
    D = 12.0
    full = CensoredDistributions._sequential_segment(seq.components, 1, 4, pe)

    # All intermediates observed: three-way factorised numerator, full conv
    # denominator.
    ev = Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0, 7.0])
    num = logpdf(e1, 2.0) + logpdf(e2, 3.0) + logpdf(e3, 2.0)
    @test CensoredDistributions.event_logpdf(seq, ev; horizon = D) ≈
          num - logcdf(full, D)

    # Middle intermediate (b) unobserved: e1 e2 convolve, e3 conditions; the
    # denominator is still the full conv-to-last-observed (d).
    ev2 = Vector{Union{Missing, Float64}}([0.0, missing, 5.0, 7.0])
    s12 = CensoredDistributions._sequential_segment(seq.components, 1, 3, pe)
    num2 = logpdf(s12, 5.0) + logpdf(e3, 2.0)
    @test CensoredDistributions.event_logpdf(seq, ev2; horizon = D) ≈
          num2 - logcdf(full, D)

    # Last event (d) unobserved -> the last OBSERVED event is c: denominator
    # convolves only e1 e2 e3 up to c (origin->c), not to the missing d.
    ev3 = Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0, missing])
    to_c = CensoredDistributions._sequential_segment(seq.components, 1, 3, pe)
    num3 = logpdf(e1, 2.0) + logpdf(e2, 3.0)
    @test CensoredDistributions.event_logpdf(seq, ev3; horizon = D) ≈
          num3 - logcdf(to_c, D)
end

@testitem "event_logpdf whole-compose origin-missing (#366)" begin
    using Distributions
    using CensoredDistributions: record_distributions

    # When the latent origin E_0 is UNOBSERVED but a per-record horizon is
    # supplied, the conv-to-last-observed denominator is anchored at the first
    # OBSERVED event, which is an exact-time anchor, NOT the latent primary. The
    # origin primary is therefore reapplied only when the first observed event is
    # E_0 (`obs_idx[1] == 1`). The denominator delay must match
    # `_sequential_segment` with NO primary, and the flat `event_logpdf` path must
    # stay exactly equal to the vectorised `record_distributions` build (whose run
    # uses the same `a == 1 ? primary` rule).
    pe = Uniform(0, 1)
    e1 = primary_censored(LogNormal(1.0, 0.4), pe)
    e2 = primary_censored(Gamma(2.0, 0.8), pe)
    e3 = primary_censored(Gamma(1.5, 1.0), pe)
    seq = Sequential((e1, e2, e3), (:a_b, :b_c, :c_d))
    D = 12.0

    # Origin a missing; b, c, d observed. Last observed is d, so the denominator
    # spans the run (b -> d) WITHOUT the origin primary (b is observed exactly).
    ev = Vector{Union{Missing, Float64}}([missing, 2.0, 5.0, 7.0])
    last_seg = CensoredDistributions._sequential_segment(
        seq.components, 2, 4, nothing)
    num = logpdf(e2, 3.0) + logpdf(e3, 2.0)
    @test CensoredDistributions.event_logpdf(seq, ev; horizon = D) ≈
          num - logcdf(last_seg, D - 2.0)

    # The flat path equals the vectorised record build exactly.
    row = (a = missing, b = 2.0, c = 5.0, d = 7.0, obs_time = D)
    recs = record_distributions(seq, [row])
    @test logpdf(only(recs), [0.0, 2.0, 5.0, 7.0]) ≈
          CensoredDistributions.event_logpdf(seq, ev; horizon = D)
end
