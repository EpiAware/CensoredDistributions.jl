# Tests for the `convolve_distributions(stack, series)` renewal method: push a
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
    @test convolve_distributions(seq, series) ≈
          reference_convolution(total, series)

    # Single-leaf stack: the leaf is its own delay.
    leaf = Gamma(2.0, 1.0)
    @test convolve_distributions(leaf, series) ≈
          reference_convolution(leaf, series)
end

@testitem "interim event series matches the prefix convolution" setup=[
    ConvolveVectorRef] begin
    using CensoredDistributions, Distributions
    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]
    g = Gamma(2.0, 1.0)
    ln = LogNormal(0.5, 0.4)
    seq = Sequential((g, ln), (:onset_admit, :admit_death))

    # Events of the chain: origin :onset, targets :admit (prefix 1) and
    # :death (prefix 2 = endpoint).
    @test CensoredDistributions.tree_event_names(seq) ==
          (:onset, :admit, :death)

    # The interim :admit series uses ONLY the first leaf (prefix-1 delay).
    interim = convolve_distributions(seq, series; events = :admit)
    @test interim ≈ reference_convolution(g, series)

    # The :death series is the endpoint: the full (two-leaf) convolution.
    endpoint = convolve_distributions(seq, series; events = :death)
    @test endpoint ≈ convolve_distributions(seq, series)
    @test endpoint ≈ reference_convolution(convolve_distributions(g, ln), series)
end

@testitem "events selector: single name vs tuple vs default" setup=[
    ConvolveVectorRef] begin
    using CensoredDistributions, Distributions
    series = [0.0, 2.0, 4.0, 7.0, 9.0, 6.0]
    seq = Sequential((Gamma(2.0, 1.0), LogNormal(0.5, 0.4)),
        (:onset_admit, :admit_death))

    # Default produces the endpoint as a bare vector.
    default = convolve_distributions(seq, series)
    @test default isa AbstractVector
    @test default ≈ convolve_distributions(seq, series; events = :death)

    # A single name produces just that bare series.
    one = convolve_distributions(seq, series; events = :admit)
    @test one isa AbstractVector

    # A tuple produces a NamedTuple keyed by the requested events only.
    nt = convolve_distributions(seq, series; events = (:admit, :death))
    @test nt isa NamedTuple
    @test keys(nt) == (:admit, :death)
    @test nt.admit ≈ one
    @test nt.death ≈ default

    # Requesting a single event in a tuple returns exactly that series.
    @test convolve_distributions(seq, series; events = (:death,)).death ≈
          default

    # An unknown event errors clearly.
    @test_throws ArgumentError convolve_distributions(
        seq, series; events = :nope)
end

@testitem "renewal method does not disturb distribution-args dispatch" begin
    using CensoredDistributions, Distributions

    # The numeric-vector second argument selects the renewal method.
    series = [0.0, 1.0, 2.0, 3.0]
    @test convolve_distributions(Gamma(2.0, 1.0), series) isa AbstractVector

    # The distribution-args forms still build a Convolved unambiguously.
    two = convolve_distributions(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    @test two isa CensoredDistributions.Convolved

    three = convolve_distributions(
        Gamma(2.0, 1.0), LogNormal(0.5, 0.4), Normal(0.0, 1.0))
    @test three isa CensoredDistributions.Convolved

    vec = convolve_distributions([Gamma(2.0, 1.0), LogNormal(0.5, 0.4)])
    @test vec isa CensoredDistributions.Convolved

    tup = convolve_distributions((Gamma(2.0, 1.0), LogNormal(0.5, 0.4)))
    @test tup isa CensoredDistributions.Convolved
end

@testitem "bare-leaf endpoint event is selectable by name" setup=[
    ConvolveVectorRef] begin
    using CensoredDistributions, Distributions
    series = [0.0, 1.0, 3.0, 6.0, 8.0]
    leaf = Gamma(2.0, 1.0)

    # A bare leaf has a single target event, its endpoint, named :event_1.
    @test CensoredDistributions._stack_target_names(leaf) == (:event_1,)

    # Selecting that endpoint by name must match the default (endpoint) and the
    # reference, not error as if no events were available.
    @test convolve_distributions(leaf, series; events = :event_1) ≈
          convolve_distributions(leaf, series)
    @test convolve_distributions(leaf, series; events = :event_1) ≈
          reference_convolution(leaf, series)

    # A tuple selecting the endpoint yields a NamedTuple keyed by it.
    nt = convolve_distributions(leaf, series; events = (:event_1,))
    @test nt isa NamedTuple
    @test keys(nt) == (:event_1,)
    @test nt.event_1 ≈ convolve_distributions(leaf, series)

    # An unknown name on a bare leaf still errors clearly.
    @test_throws ArgumentError convolve_distributions(
        leaf, series; events = :nope)
end

@testitem "interval != 1 is rejected to avoid grid/series-step conflation" begin
    using CensoredDistributions, Distributions
    series = [0.0, 1.0, 3.0, 6.0, 8.0]
    leaf = Gamma(2.0, 1.0)

    # The causal convolution shifts by integer SERIES steps (unit-spaced), so a
    # PMF grid step other than 1 conflates the discretisation width with the
    # series time-step. Reject it rather than silently mis-aligning.
    @test_throws ArgumentError convolve_distributions(leaf, series;
        interval = 0.5)
    @test_throws ArgumentError convolve_distributions(leaf, series;
        interval = 2.0)

    # The unit-spaced default is unchanged.
    @test convolve_distributions(leaf, series; interval = 1.0) ≈
          convolve_distributions(leaf, series)
end
