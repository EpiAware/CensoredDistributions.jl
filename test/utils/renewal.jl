# Tests for the renewal recurrence `renewal(Rt, gi, I0; modulator)`: the
# composable forward scan `I[t] = R_t m(t) Σ_s g_s I[t-s]`. AD coverage is in
# the full-backend fixture harness (`test/ADFixtures` + `test/ad`); these are
# the value-level equivalence and interface checks.

@testsnippet RenewalRef begin
    using CensoredDistributions, Distributions
    # The renewal surface is public but unexported (pending the #611 scope
    # decision), so bring the names in explicitly.
    using CensoredDistributions: renewal, susceptibility_depletion,
                                 combine_modulators, NoModulation

    # The rt-renewal tutorial's hand-rolled loop, verbatim. The bare renewal
    # scan must reproduce it bit for bit.
    function handrolled_renewal(Rt, g, I0; seed_days = length(g))
        n = length(Rt)
        I = zeros(eltype(Rt), n)
        I[1:seed_days] .= I0
        for t in (seed_days + 1):n
            acc = zero(eltype(Rt))
            for s in 1:min(length(g), t - 1)
                acc += g[s] * I[t - s]
            end
            I[t] = Rt[t] * acc
        end
        return I
    end

    # An independent hand-written SIR renewal to check the
    # susceptibility-depletion modulator against.
    function handrolled_sir(Rt, g, I0, N; seed_days = length(g))
        n = length(Rt)
        I = zeros(n)
        I[1:seed_days] .= I0
        S = N - sum(I[1:seed_days])
        for t in (seed_days + 1):n
            acc = 0.0
            for s in 1:min(length(g), t - 1)
                acc += g[s] * I[t - s]
            end
            force = Rt[t] * acc
            I[t] = (S / N) * force
            S -= I[t]
        end
        return I
    end

    gi = pdf(
        interval_censored(
            truncated(Gamma(2.5, 1.3); lower = 1.0, upper = 12.0), 1.0), 1:12)
    Rt = vcat(fill(1.6, 18), fill(0.7, 18), fill(1.3, 18), fill(1.0, 16))
    I0 = 10.0
end

@testitem "bare renewal reproduces the hand-rolled loop exactly" setup=[
    RenewalRef] begin
    ref = handrolled_renewal(Rt, gi, I0)
    out = renewal(Rt, gi, I0)
    @test out == ref
end

@testitem "DelayPMF generation interval agrees with the PMF vector" setup=[
    RenewalRef] begin
    pmf = CensoredDistributions.discretise_pmf(
        interval_censored(
            truncated(Gamma(2.5, 1.3); lower = 1.0, upper = 12.0), 1.0), 12)
    ref = handrolled_renewal(Rt, gi, I0)
    # The DelayPMF carries a lag-0 mass; the renewal drops it, so the lag-s
    # weights match the lag-1-indexed PMF vector and the series agrees.
    @test renewal(Rt, pmf, I0) ≈ ref
end

@testitem "susceptibility depletion matches a hand SIR loop" setup=[
    RenewalRef] begin
    N = 1.0e5
    out = renewal(Rt, gi, I0; modulator = susceptibility_depletion(N))
    @test out ≈ handrolled_sir(Rt, gi, I0, N)
    # The depleting pool lowers the peak relative to the bare renewal.
    @test maximum(out) < maximum(renewal(Rt, gi, I0))
end

@testitem "susceptibility depletion collapses to bare renewal as N grows" setup=[
    RenewalRef] begin
    bare = renewal(Rt, gi, I0)
    big = renewal(Rt, gi, I0; modulator = susceptibility_depletion(1.0e12))
    @test maximum(abs.(big .- bare) ./ (bare .+ 1)) < 1.0e-3
end

@testitem "modulators compose: factors multiply" setup=[RenewalRef] begin
    # A composed identity-times-identity equals the bare renewal.
    both = combine_modulators(NoModulation(), NoModulation())
    @test renewal(Rt, gi, I0; modulator = both) == renewal(Rt, gi, I0)
    # Susceptibility * identity equals susceptibility alone.
    N = 5.0e4
    stacked = combine_modulators(susceptibility_depletion(N), NoModulation())
    @test renewal(Rt, gi, I0; modulator = stacked) ≈
          renewal(Rt, gi, I0; modulator = susceptibility_depletion(N))
end

@testitem "seed_days fixes the leading window" setup=[RenewalRef] begin
    out = renewal(Rt, gi, I0; seed_days = 5)
    @test all(out[1:5] .== I0)
    # A shorter seed than the generation interval still runs (history fills in).
    @test length(renewal(Rt, gi, I0; seed_days = 3)) == length(Rt)
end
