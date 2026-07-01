# AD coverage for the whole-compose conv-to-last-observed right-truncation
# denominator. With an OBSERVED intermediate and a per-record horizon the
# Sequential record scores a factorised numerator over a single
# `-logcdf(conv-to-last-observed, window)` denominator. The gradient w.r.t. the
# delay params must flow: the denominator convolves the leaf cores and evaluates
# a CDF at the (constant) window, all on the differentiated param type.

@testitem "whole-compose truncation gradient: ForwardDiff" tags=[
    :ad, :forwarddiff] begin
    using CensoredDistributions, Distributions
    using ADTypes: AutoForwardDiff
    using DifferentiationInterface: gradient
    using ForwardDiff: ForwardDiff

    # One observed-intermediate record and one endpoint-observed record, both
    # right-truncated at their own horizon, so the gradient exercises both the
    # factorised (observed-intermediate) numerator and the conv-to-last-observed
    # denominator. LogNormal delays keep the truncation `logcdf` off the Gamma
    # shape-derivative path (`_gamma_inc`), matching the established AD-safe
    # right-truncation fixtures (`ADFixtures`).
    # events: [origin, mid, obs]; horizon per record.
    evs = [Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0]),
        Vector{Union{Missing, Float64}}([0.5, missing, 7.0])]
    Ds = [8.0, 9.0]

    # θ = [inc μ, inc σ, delta μ, delta σ].
    function f(θ)
        inc = primary_censored(LogNormal(θ[1], θ[2]), Uniform(0, 1))
        delta = primary_censored(LogNormal(θ[3], θ[4]), Uniform(0, 1))
        seq = Sequential((inc, delta), (:onset_mid, :mid_obs))
        return sum(eachindex(evs)) do i
            CensoredDistributions.event_logpdf(seq, evs[i]; horizon = Ds[i])
        end
    end

    θ = [1.0, 0.5, 0.5, 0.4]
    val = f(θ)
    @test isfinite(val)

    g = gradient(f, AutoForwardDiff(), θ)
    @test g isa AbstractVector && length(g) == 4 && all(isfinite, g)
    # Every parameter feeds both a numerator factor and the denominator, so each
    # partial is non-zero.
    @test all(!=(0), g)
end

@testitem "whole-compose truncation gradient: Enzyme reverse" tags=[
    :ad, :enzyme, :enzyme_reverse] begin
    using CensoredDistributions, Distributions
    using ADTypes: AutoEnzyme, AutoForwardDiff
    using DifferentiationInterface: gradient, Constant
    using Enzyme: Enzyme
    using ForwardDiff: ForwardDiff

    # The same conv-to-last-observed truncation path as the ForwardDiff item,
    # pinned on Enzyme reverse. This routes the `Vector{Union{Missing,
    # Float64}}` event handling through `_seq_event_logpdf_h` and builds a
    # freshly-allocated `Convolved` observed total for the denominator. The
    # missing-intermediate record travels as an inactive Constant context; the
    # per-record horizons are constants. Earlier registered broken on Enzyme
    # reverse, now a hard check.
    evs = [Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0]),
        Vector{Union{Missing, Float64}}([0.5, missing, 7.0])]

    function f(θ, evs)
        inc = primary_censored(LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0))
        delta = primary_censored(LogNormal(θ[3], θ[4]), Uniform(0.0, 1.0))
        seq = Sequential((inc, delta), (:onset_mid, :mid_obs))
        hs = (8.0, 9.0)
        total = CensoredDistributions.event_logpdf(
            seq, evs[1]; horizon = hs[1])
        for i in 2:length(evs)
            total += CensoredDistributions.event_logpdf(
                seq, evs[i]; horizon = hs[i])
        end
        return total
    end

    θ = [1.0, 0.5, 0.5, 0.4]
    ctx = Constant(evs)
    rev = AutoEnzyme(mode = Enzyme.set_runtime_activity(Enzyme.Reverse))

    ref = gradient(f, AutoForwardDiff(), θ, ctx)
    @test all(isfinite, ref)
    g = gradient(f, rev, θ, ctx)
    @test g isa AbstractVector && length(g) == 4 && all(isfinite, g)
    @test isapprox(g, ref; rtol = 5e-2, atol = 1e-6)
end

@testitem "δ-bounded truncation gradient: ForwardDiff" tags=[
    :ad, :forwarddiff] begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: WindowedHorizon
    using ADTypes: AutoForwardDiff
    using DifferentiationInterface: gradient
    using ForwardDiff: ForwardDiff

    # The δ-bounded right-truncation normaliser `cdf(upper) - cdf(lower)` must
    # differentiate w.r.t. the delay params through BOTH the leaf primitive and
    # the whole-compose chain denominator. LogNormal delays keep the truncation
    # `logcdf` off the Gamma shape-derivative path.
    evs = [Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0]),
        Vector{Union{Missing, Float64}}([0.5, missing, 7.0])]
    whs = [WindowedHorizon(8.0, 6.0), WindowedHorizon(9.0, 7.0)]

    function f(θ)
        inc = primary_censored(LogNormal(θ[1], θ[2]), Uniform(0, 1))
        delta = primary_censored(LogNormal(θ[3], θ[4]), Uniform(0, 1))
        seq = Sequential((inc, delta), (:onset_mid, :mid_obs))
        chain = sum(eachindex(evs)) do i
            CensoredDistributions.event_logpdf(seq, evs[i]; horizon = whs[i])
        end
        # The leaf δ-bounded primitive too.
        leaf = logpdf(
            truncated(LogNormal(θ[1], θ[2]); lower = 3.0, upper = 6.0), 4.0)
        return chain + leaf
    end

    θ = [1.0, 0.5, 0.5, 0.4]
    @test isfinite(f(θ))
    g = gradient(f, AutoForwardDiff(), θ)
    @test g isa AbstractVector && length(g) == 4 && all(isfinite, g)
    @test all(!=(0), g)
end

@testitem "δ-bounded truncation gradient: Mooncake reverse" tags=[
    :ad, :mooncake, :mooncake_reverse] begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: WindowedHorizon
    using ADTypes: AutoMooncake, AutoForwardDiff
    using DifferentiationInterface: gradient
    using ForwardDiff: ForwardDiff
    using Mooncake: Mooncake

    # Fully-observed records (the missing-intermediate `event_logpdf` path hits a
    # pre-existing Mooncake limitation in the string-based event-name derivation),
    # δ-bounded per record. The δ-bounded denominator routes through the chain
    # `logcdf` at both edges and must trace under Mooncake reverse.
    evs = [Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0]),
        Vector{Union{Missing, Float64}}([0.5, 3.0, 7.0])]
    whs = [WindowedHorizon(8.0, 6.0), WindowedHorizon(9.0, 7.0)]

    function f(θ)
        inc = primary_censored(LogNormal(θ[1], θ[2]), Uniform(0, 1))
        delta = primary_censored(LogNormal(θ[3], θ[4]), Uniform(0, 1))
        seq = Sequential((inc, delta), (:onset_mid, :mid_obs))
        chain = sum(eachindex(evs)) do i
            CensoredDistributions.event_logpdf(seq, evs[i]; horizon = whs[i])
        end
        leaf = logpdf(
            truncated(LogNormal(θ[1], θ[2]); lower = 3.0, upper = 6.0), 4.0)
        return chain + leaf
    end

    θ = [1.0, 0.5, 0.5, 0.4]
    ref = gradient(f, AutoForwardDiff(), θ)
    g = gradient(f, AutoMooncake(; config = nothing), θ)
    @test g isa AbstractVector && length(g) == 4 && all(isfinite, g)
    @test isapprox(g, ref; rtol = 1e-6, atol = 1e-8)
end

@testitem "compose-over wrappers gradient: ForwardDiff" tags=[
    :ad, :forwarddiff] begin
    using CensoredDistributions, Distributions
    using ADTypes: AutoForwardDiff
    using DifferentiationInterface: gradient
    using ForwardDiff: ForwardDiff

    # The wrap-over-composer paths must differentiate. A modifier over a composed
    # node now DISTRIBUTES into the leaf cores and keeps the tree shape (#655), so
    # a wrapped Sequential/Parallel is multivariate and scores an event/value
    # vector; the scalar combine-then-censor total is the explicit
    # `modifier(observed_distribution(node))` form. θ = leaf LogNormal log-means;
    # LogNormal keeps truncation `logcdf` off the Gamma shape-derivative path.
    function f(θ)
        a = LogNormal(θ[1], 0.5)
        b = LogNormal(θ[2], 0.4)

        # Sequential, node-level wrap (distributes into leaves, tree shape kept).
        # upper-only `truncated(seq; upper)` adds no origin primary, so the chain
        # scores a length-2 step-value vector; double_interval_censored adds the
        # origin primary, so it scores a length-3 event vector [E_0, E_1, E_2].
        seq = Sequential(a, b)
        s1 = logpdf(truncated(seq; upper = 10.0), [3.0, 5.0])
        s2 = logpdf(
            double_interval_censored(seq; primary_event = Uniform(0, 1),
                upper = 12.0, interval = 1.0),
            Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0]))
        # Sequential, scalar combine-then-censor total via observed_distribution.
        s2b = logpdf(
            double_interval_censored(observed_distribution(seq);
                primary_event = Uniform(0, 1), upper = 12.0, interval = 1.0),
            3.0)

        # Parallel: distributed truncation + interval censoring.
        par = Parallel(a, b)
        s3 = logpdf(truncated(par; upper = 10.0), [2.0, 3.0])
        s4 = logpdf(interval_censored(par, 1.0), [2.0, 3.0])

        # Choose: distributed primary censoring, scored per-alternative.
        sel = choose(:index => a, :sourced => b)
        s5 = logpdf(primary_censored(sel, Uniform(0, 1)), 2.0; kind = :index)
        s6 = logpdf(truncated(sel; upper = 10.0), 3.0; kind = :sourced)

        # The δ-bounded `truncated(node; lower, upper)` over a node distributes
        # into the leaf cores too (#711): the per-leaf finite-window `logcdf`
        # normaliser must differentiate w.r.t. the leaf params through the kept
        # tree shape.
        s7 = logpdf(truncated(seq; lower = 0.3, upper = 12.0), [3.0, 5.0])
        s8 = logpdf(truncated(par; lower = 0.5, upper = 12.0), [2.0, 3.0])

        return s1 + s2 + s2b + s3 + s4 + s5 + s6 + s7 + s8
    end

    θ = [1.0, 0.5]
    @test isfinite(f(θ))
    g = gradient(f, AutoForwardDiff(), θ)
    @test g isa AbstractVector && length(g) == 2 && all(isfinite, g)
    @test all(!=(0), g)
end
