# Monte Carlo property tests for the PLAIN (uncensored) composers. The existing
# MC property tests live on the censored / hazard shapes (`wrap.jl`'s
# `interval_censored(Sequential)` total, `one_of_features.jl`'s racing-hazard
# winning probabilities). These pin the bare composer semantics directly:
#
#   - a plain `Sequential` scores each step's OWN gap, so the chain TOTAL (sum of
#     the per-step draws) is distributed as the CONVOLUTION of the steps. The
#     day-discretised PMF of the total must match a Monte Carlo histogram of the
#     summed draws AND the analytic `Convolved` of the same steps.
#   - a plain `Parallel` scores each branch independently, so each branch's draws
#     are an independent sample of that branch distribution and its empirical
#     CDF must match the branch's analytic CDF.

@testitem "plain Sequential total == MC convolution of the steps" begin
    using CensoredDistributions, Distributions, Random, Statistics

    Random.seed!(20240618)

    s1 = Gamma(2.0, 1.0)
    s2 = LogNormal(0.5, 0.4)
    seq = Sequential(s1, s2)

    # The chain total is the sum of the two step draws; its day-discretised PMF
    # over [k, k+1) is the reference. `rand(seq)` returns the per-step gaps
    # (a NamedTuple), so the total is the sum of its values.
    N = 2_000_000
    tot = Vector{Float64}(undef, N)
    for i in 1:N
        v = rand(seq)
        tot[i] = sum(values(v))
    end

    # Analytic convolution of the same steps: the total's density.
    conv = convolve_distributions(s1, s2)
    for k in (1.0, 2.0, 3.0, 4.0, 5.0)
        mc = mean(k .<= tot .< (k + 1))
        analytic = cdf(conv, k + 1) - cdf(conv, k)
        @test isapprox(mc, analytic; atol = 3e-3)
    end
end

@testitem "plain Parallel branches == MC of each branch" begin
    using CensoredDistributions, Distributions, Random, Statistics

    Random.seed!(20240619)

    b1 = Gamma(2.0, 1.0)
    b2 = LogNormal(1.0, 0.5)
    par = Parallel(b1, b2)

    # Each branch is scored and drawn independently; the empirical CDF of each
    # branch's draws must match the branch's analytic CDF.
    N = 2_000_000
    draws1 = Vector{Float64}(undef, N)
    draws2 = Vector{Float64}(undef, N)
    for i in 1:N
        v = rand(par)
        draws1[i] = v[1]
        draws2[i] = v[2]
    end

    for x in (0.5, 1.0, 2.0, 3.0, 5.0)
        @test isapprox(mean(draws1 .<= x), cdf(b1, x); atol = 3e-3)
        @test isapprox(mean(draws2 .<= x), cdf(b2, x); atol = 3e-3)
    end

    # The joint logpdf is the sum of the independent branch logpdfs (the
    # property the MC draws sample from).
    @test logpdf(par, [2.0, 3.0]) ≈ logpdf(b1, 2.0) + logpdf(b2, 3.0)
end
