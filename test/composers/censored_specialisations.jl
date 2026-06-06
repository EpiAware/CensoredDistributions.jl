# Reference tests for the censored specialisations of the generic composers
# (#329, PR3b). Each asserts the data-driven, dispatch-selected behaviour:
# single-edge == primary_censored, unobserved intermediate == convolution
# reference, observed intermediate == conditioning, Parallel == shared-origin
# Monte-Carlo, and the delay-adjusted CFR recovery via Competing.

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

@testitem "Sequential observed intermediate conditions on each gap" begin
    using Distributions

    d1 = LogNormal(1.2, 0.5)
    d2 = Gamma(2.0, 1.0)
    pe = Uniform(0, 1)
    chain = Sequential(primary_censored(d1, pe), primary_censored(d2, pe))

    # All events observed: origin segment is primary-censored d1 at the first
    # gap; the second (observed-bounded) edge conditions on the continuous core
    # d2 at the second gap, dropping its primary censoring.
    ev = Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0])
    @test logpdf(chain, ev) ≈
          logpdf(primary_censored(d1, pe), 2.0) + logpdf(d2, 3.0) rtol=1e-6
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

@testitem "Parallel plain branches reject shared-origin scoring" begin
    using Distributions

    par = Parallel(Gamma(2.0, 1.0), LogNormal(1.0, 0.5))
    ev = Vector{Union{Missing, Float64}}([missing, 2.0, 3.0])
    @test_throws ArgumentError logpdf(par, ev)
end

@testitem "Competing recovers a delay-adjusted CFR" begin
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

    # The Competing branch probability that maximises the right-truncated
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

    # The Competing node lowers to the CFR-weighted mixture (PR3a as_mixture).
    node = Competing(:death => (death_delay, true_cfr),
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

    # A batch of records with different observed/missing patterns; per-record
    # dispatch handles each from its own missingness, not a global mode.
    records = [
        Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0]),      # all observed
        Vector{Union{Missing, Float64}}([0.0, missing, 4.0]),   # E1 unobserved
        Vector{Union{Missing, Float64}}([1.0, 3.5])             # single edge
    ]
    lps = logpdf.(Ref(chain), records)
    @test all(isfinite, lps)
    @test lps[1] ≈ logpdf(primary_censored(d1, pe), 2.0) + logpdf(d2, 3.0) rtol=1e-6
    @test lps[2] ≈
          logpdf(primary_censored(convolve_distributions(d1, d2), pe), 4.0) rtol=1e-6
    @test lps[3] ≈ logpdf(primary_censored(d1, pe), 2.5) rtol=1e-6
end

@testitem "rand simulates the full event-time path of a censored chain" begin
    using Distributions, Random
    Random.seed!(11)

    pe = Uniform(0, 1)
    chain = Sequential(primary_censored(LogNormal(1.2, 0.5), pe),
        primary_censored(Gamma(2.0, 1.0), pe))

    # Full event-time path [E0, E1, E2]: origin draw plus cumulative delays, so
    # every internal event time is returned (not a summary).
    r = rand(chain)
    @test length(r) == 3
    @test issorted(r)              # monotone: each step adds a positive delay
    # The simulated all-observed path scores finitely under the chain.
    @test isfinite(logpdf(chain, convert(Vector{Union{Missing, Float64}}, r)))
end

@testitem "rand simulates the shared origin and every branch of a Parallel" begin
    using Distributions, Random
    Random.seed!(12)

    pe = Uniform(0, 1)
    par = Parallel(primary_censored(Gamma(2.0, 1.0), pe),
        primary_censored(LogNormal(1.0, 0.5), pe))

    # Full event-time vector [O, Y1, Y2]: one shared origin draw then each branch
    # observation as origin plus an independent branch delay.
    r = rand(par)
    @test length(r) == 3
    @test all(r[2:end] .>= r[1])   # each observation is after the shared origin
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
