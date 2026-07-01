# Tests for the `convolved(stack, series)` renewal method: push a
# timeseries through a composed delay stack to selected event count series.

@testsnippet ConvolveVectorRef begin
    using CensoredDistributions, Distributions

    # A hand-written EpiNow2-style discrete delay-convolution reference: the
    # series convolved with the discretised delay PMF (raw interval masses),
    # causal and truncated to the series window. Mirrors the method's contract
    # independently.
    function reference_convolution(delay, series; interval = 1.0)
        n = length(series)
        maxlag = n - 1
        grid = (0:maxlag) .* interval
        ic = interval_censored(delay, interval)
        pmf = [pdf(ic, g) for g in grid]
        out = zeros(Float64, n)
        for i in 1:n
            for k in 1:min(length(pmf), i)
                out[i] += pmf[k] * series[i - k + 1]
            end
        end
        return out
    end
end

@testitem "endpoint matches the discrete delay-convolution reference" setup=[
    ConvolveVectorRef] begin
    using CensoredDistributions, Distributions
    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]

    # Sequential stack: endpoint uses the total (collapsed) delay.
    seq = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    total = observed_distribution(seq)
    @test convolved(seq, series) ≈
          reference_convolution(total, series)

    # Single-leaf stack: the leaf is its own delay.
    leaf = Gamma(2.0, 1.0)
    @test convolved(leaf, series) ≈
          reference_convolution(leaf, series)
end

@testitem "interim event series matches the prefix convolution" setup=[
    ConvolveVectorRef] begin
    using CensoredDistributions, Distributions
    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]
    g = Gamma(2.0, 1.0)
    ln = LogNormal(0.5, 0.4)
    seq = Sequential((g, ln), (:onset_admit, :admit_death))

    # The `events=` selector keys on the FLAT event path of the chain: origin
    # :onset, targets :admit (prefix 1) and :death (prefix 2 = endpoint). For a
    # plain chain the public `event_names` follows the bare edge names, so the
    # flat path is taken from `_flat_event_names`.
    @test CensoredDistributions._flat_event_names(seq) ==
          (:onset, :admit, :death)

    # The interim :admit series uses ONLY the first leaf (prefix-1 delay).
    interim = convolved(seq, series; events = :admit)
    @test interim ≈ reference_convolution(g, series)

    # The :death series is the endpoint: the full (two-leaf) convolution.
    endpoint = convolved(seq, series; events = :death)
    @test endpoint ≈ convolved(seq, series)
    @test endpoint ≈
          reference_convolution(convolved(g, ln), series)
end

@testitem "events selector: single name vs tuple vs default" setup=[
    ConvolveVectorRef] begin
    using CensoredDistributions, Distributions
    series = [0.0, 2.0, 4.0, 7.0, 9.0, 6.0]
    seq = Sequential((Gamma(2.0, 1.0), LogNormal(0.5, 0.4)),
        (:onset_admit, :admit_death))

    # Default produces the endpoint as a bare vector.
    default = convolved(seq, series)
    @test default isa AbstractVector
    @test default ≈ convolved(seq, series; events = :death)

    # A single name produces just that bare series.
    one = convolved(seq, series; events = :admit)
    @test one isa AbstractVector

    # A tuple produces a NamedTuple keyed by the requested events only.
    nt = convolved(seq, series; events = (:admit, :death))
    @test nt isa NamedTuple
    @test keys(nt) == (:admit, :death)
    @test nt.admit ≈ one
    @test nt.death ≈ default

    # Requesting a single event in a tuple returns exactly that series.
    @test convolved(seq, series; events = (:death,)).death ≈
          default

    # An unknown event errors clearly.
    @test_throws ArgumentError convolved(
        seq, series; events = :nope)
end

@testitem "renewal method does not disturb distribution-args dispatch" begin
    using CensoredDistributions, Distributions

    # The numeric-vector second argument selects the renewal method.
    series = [0.0, 1.0, 2.0, 3.0]
    @test convolved(Gamma(2.0, 1.0), series) isa AbstractVector

    # The distribution-args forms still build a Convolved unambiguously.
    two = convolved(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    @test two isa CensoredDistributions.Convolved

    three = convolved(
        Gamma(2.0, 1.0), LogNormal(0.5, 0.4), Normal(0.0, 1.0))
    @test three isa CensoredDistributions.Convolved

    vec = convolved([Gamma(2.0, 1.0), LogNormal(0.5, 0.4)])
    @test vec isa CensoredDistributions.Convolved

    tup = convolved((Gamma(2.0, 1.0), LogNormal(0.5, 0.4)))
    @test tup isa CensoredDistributions.Convolved
end

@testitem "bare-leaf endpoint event is selectable by name" setup=[
    ConvolveVectorRef] begin
    using CensoredDistributions, Distributions
    series = [0.0, 1.0, 3.0, 6.0, 8.0]
    leaf = Gamma(2.0, 1.0)

    # A bare leaf has a single event, its endpoint, named :event_1.
    @test [s.name for s in CensoredDistributions._event_specs(leaf)] ==
          [:event_1]

    # Choosing that endpoint by name must match the default (endpoint) and the
    # reference, not error as if no events were available.
    @test convolved(leaf, series; events = :event_1) ≈
          convolved(leaf, series)
    @test convolved(leaf, series; events = :event_1) ≈
          reference_convolution(leaf, series)

    # A tuple choose the endpoint yields a NamedTuple keyed by it.
    nt = convolved(leaf, series; events = (:event_1,))
    @test nt isa NamedTuple
    @test keys(nt) == (:event_1,)
    @test nt.event_1 ≈ convolved(leaf, series)

    # An unknown name on a bare leaf still errors clearly.
    @test_throws ArgumentError convolved(
        leaf, series; events = :nope)
end

@testitem "interval != 1 is rejected to avoid grid/series-step conflation" begin
    using CensoredDistributions, Distributions
    series = [0.0, 1.0, 3.0, 6.0, 8.0]
    leaf = Gamma(2.0, 1.0)

    # The causal convolution shifts by integer SERIES steps (unit-spaced), so a
    # PMF grid step other than 1 conflates the discretisation width with the
    # series time-step. Reject it rather than silently mis-aligning.
    @test_throws ArgumentError convolved(leaf, series;
        interval = 0.5)
    @test_throws ArgumentError convolved(leaf, series;
        interval = 2.0)

    # The unit-spaced default is unchanged.
    @test convolved(leaf, series; interval = 1.0) ≈
          convolved(leaf, series)
end

@testitem "forward thin/cumulative apply on the convolved series" setup=[
    ConvolveVectorRef] begin
    using CensoredDistributions, Distributions
    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]
    g = Gamma(2.0, 1.0)

    # thin scales the endpoint series by the factor.
    plain = convolved(g, series)
    thinned = convolved(thin(g, 0.3), series)
    @test thinned ≈ 0.3 .* plain

    # cumulative accumulates the endpoint series.
    cumd = convolved(cumulative(g), series)
    @test cumd ≈ cumsum(plain)
end

@testitem "convolve over a Parallel branched stack returns per-branch series" setup=[
    ConvolveVectorRef] begin
    using CensoredDistributions, Distributions
    series = [0.0, 2.0, 4.0, 7.0, 9.0, 6.0, 3.0]
    dc = Gamma(1.8, 1.4)
    dd = Gamma(3.0, 4.0)

    # Two independent thinned streams sharing the renewal origin.
    p = compose((cases = thin(dc, 0.3), deaths = thin(dd, 0.012)))
    out = convolved(p, series; events = (:cases, :deaths))
    @test out isa NamedTuple
    @test keys(out) == (:cases, :deaths)
    @test out.cases ≈ 0.3 .* reference_convolution(dc, series)
    @test out.deaths ≈ 0.012 .* reference_convolution(dd, series)
end

@testitem "convolve branches on a shared incubation prefix" setup=[
    ConvolveVectorRef] begin
    using CensoredDistributions, Distributions
    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]
    incub = Gamma(2.5, 1.3)
    onset_report = Gamma(1.5, 1.2)
    onset_death = Gamma(3.0, 4.0)

    # Shared incubation, then branch: Sequential ending in a Parallel.
    stack = Sequential((incub,
        Parallel((thin(onset_report, 0.3), thin(onset_death, 0.012)),
            (:cases, :deaths))))
    out = convolved(stack, series; events = (:cases, :deaths))

    # Each branch's delay is incubation convolved with the branch tail.
    @test out.cases ≈
          0.3 .* reference_convolution(
        convolved(incub, onset_report), series)
    @test out.deaths ≈
          0.012 .* reference_convolution(
        convolved(incub, onset_death), series)
end

@testitem "Resolve reduces to per-outcome thinning under convolve" setup=[
    ConvolveVectorRef] begin
    using CensoredDistributions, Distributions
    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]
    d_death = Gamma(3.0, 4.0)
    d_disch = Gamma(2.0, 3.0)

    # A real partition: death vs discharge, branch probs sum to one.
    c = resolve(:death => (d_death, 0.7), :discharge => (d_disch, 0.3))
    out = convolved(c, series; events = (:death, :discharge))
    @test out.death ≈ 0.7 .* reference_convolution(d_death, series)
    @test out.discharge ≈ 0.3 .* reference_convolution(d_disch, series)

    # Partial observation: request only the observed endpoint.
    just_death = convolved(c, series; events = :death)
    @test just_death ≈ 0.7 .* reference_convolution(d_death, series)
end

# --- build-once DelayPMF for vector evaluation ----------------------------

@testitem "DelayPMF masses match the rebuild-every-time delay PMF" begin
    using CensoredDistributions, Distributions
    delay = convolved(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    maxlag = 12

    pmf = CensoredDistributions.discretise_pmf(delay, maxlag)
    @test pmf isa CensoredDistributions.DelayPMF
    @test length(pmf) == maxlag + 1

    # The masses are EXACTLY the private rebuild-every-time `_delay_pmf` values.
    rebuilt = CensoredDistributions._delay_pmf(delay, maxlag, 1.0)
    @test pmf.masses == rebuilt
end

@testitem "convolved(pmf, series) == rebuild-every-time path" setup=[
    ConvolveVectorRef] begin
    using CensoredDistributions, Distributions
    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]
    delay = convolved(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))

    # The PMF needs at least `length(series) - 1` lags to cover the window; the
    # rebuild path uses exactly `length(series) - 1`.
    pmf = CensoredDistributions.discretise_pmf(delay, length(series) - 1)

    built_once = convolved(pmf, series)
    rebuilt_each = convolved(delay, series)

    # Numerically IDENTICAL to the rebuild-every-time path, not just ≈.
    @test built_once == rebuilt_each
    @test built_once ≈ reference_convolution(delay, series)
end

@testitem "one DelayPMF reused across many series (nowcasting shape)" setup=[
    ConvolveVectorRef] begin
    using CensoredDistributions, Distributions
    delay = convolved(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))

    # A vector of reference-date series; build the PMF ONCE, reuse across all.
    reference_series = [
        [0.0, 1.0, 3.0, 6.0, 8.0],
        [0.0, 2.0, 4.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0]
    ]
    maxlag = maximum(length, reference_series) - 1
    pmf = CensoredDistributions.discretise_pmf(delay, maxlag)

    for s in reference_series
        # Each reuse equals rediscretising for that series (identical masses up
        # to the shared maxlag), so it matches the rebuild-every-time path.
        @test convolved(pmf, s) == convolved(delay, s)
    end
end

@testitem "pdf(pmf, lags) looks up the precomputed masses" begin
    using CensoredDistributions, Distributions
    delay = Gamma(2.0, 1.0)
    maxlag = 8
    pmf = CensoredDistributions.discretise_pmf(delay, maxlag)
    rebuilt = CensoredDistributions._delay_pmf(delay, maxlag, 1.0)

    # Scalar lookup returns the mass at that integer lag.
    for lag in 0:maxlag
        @test pdf(pmf, lag) == rebuilt[lag + 1]
    end

    # Out-of-range lags carry no mass.
    @test pdf(pmf, -1) == 0.0
    @test pdf(pmf, maxlag + 1) == 0.0

    # Vector lookup maps the scalar lookup, including out-of-range zeros.
    lags = [0, 3, 8, 20]
    @test pdf(pmf, lags) == [rebuilt[1], rebuilt[4], rebuilt[9], 0.0]
end

@testitem "DelayPMF rejects degenerate construction" begin
    using CensoredDistributions, Distributions
    delay = Gamma(2.0, 1.0)
    @test_throws ArgumentError CensoredDistributions.discretise_pmf(delay, -1)
    @test_throws ArgumentError CensoredDistributions.DelayPMF(Float64[], 1.0)
    @test_throws ArgumentError CensoredDistributions.DelayPMF([1.0], 0.0)

    # A non-unit PMF cannot drive the unit-step causal convolution.
    pmf = CensoredDistributions.discretise_pmf(delay, 5; interval = 0.5)
    @test_throws ArgumentError convolved(pmf, [0.0, 1.0, 2.0])
end
