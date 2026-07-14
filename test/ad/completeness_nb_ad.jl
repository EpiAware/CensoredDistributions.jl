# AD stability of the completeness-thinned negative-binomial offspring term.
#
# The andv joint model scores each source's offspring count with a
# negative binomial whose mean is a reproduction number thinned by the
# completeness of the convolved reporting delay (`delta + inc`). Scoring the
# thinned rate in linear space, `R_eff = R * cdf(conv, window)`, drives the mean
# to exactly zero when the completeness underflows (a recent source against a
# long incubation), collapsing the NB success probability `k / (k + R_eff)` to
# `1`: a degenerate point mass whose logpdf is `-Inf` and whose reverse-mode
# gradient is `NaN` for any positive count.
#
# The fix is to thin in log space with `log_thin_by_completeness` and build the
# NB success probability as `inv(1 + exp(log_mu - log_k))`, which stays strictly
# inside `(0, 1)` for any finite log mean and log dispersion. These items pin
# that the log-space term differentiates finitely on every backend at a window
# where the linear term's gradient is non-finite.
@testsnippet CompletenessNBCase begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: log_thin_by_completeness

    # A very short window against a long incubation: completeness underflows
    # toward zero, so the linear-space thinned rate collapses to (numerically)
    # zero and the negative-binomial success probability saturates to 1 (a
    # degenerate point mass at zero), which the log-space form avoids.
    const recent_window = 2.0
    const offspring_count = 3

    # Stable NB built from a log mean and log dispersion. Mirrors the andv
    # model's `nb_logmean`: success prob = inv(1 + exp(log_mu - log_k)) ∈ (0, 1).
    function stable_nb_logpdf(θ)
        mu_inc, log_sig_inc, mu_delta, log_sig_delta, log_k, log_R = θ
        inc = LogNormal(mu_inc, exp(log_sig_inc))
        delta = Normal(mu_delta, exp(log_sig_delta))
        conv = convolved(delta, inc)
        log_mu = log_thin_by_completeness(log_R, conv, recent_window)
        k = exp(log_k)
        p = inv(1 + exp(clamp(log_mu - log_k, -30.0, 30.0)))
        return logpdf(NegativeBinomial(k, p), offspring_count)
    end

    # Linear-space comparison: thin then build the NB directly. At this window
    # the completeness underflows, so this density is `-Inf` and ForwardDiff
    # returns a non-finite gradient, the failure the log-space form removes.
    function linear_nb_logpdf(θ)
        mu_inc, log_sig_inc, mu_delta, log_sig_delta, log_k, log_R = θ
        inc = LogNormal(mu_inc, exp(log_sig_inc))
        delta = Normal(mu_delta, exp(log_sig_delta))
        conv = convolved(delta, inc)
        R_eff = exp(log_R) * cdf(conv, recent_window)
        k = exp(log_k)
        return logpdf(NegativeBinomial(k, k / (k + R_eff)), offspring_count)
    end

    # A long incubation (log-mean 4.0) so that at the 2-day window the linear
    # completeness underflows; the rest match the andv posterior scale.
    const θ0 = [4.0, log(0.32), 0.17, log(0.62), log(0.5), log(2.3)]

    function check_stable(backend)
        g = DifferentiationInterface.gradient(stable_nb_logpdf, backend, θ0)
        @test g isa AbstractVector
        @test length(g) == length(θ0)
        @test all(isfinite, g)
        @test any(!iszero, g)
        ref = DifferentiationInterface.gradient(
            stable_nb_logpdf, AutoForwardDiff(), θ0)
        @test isapprox(g, ref; rtol = 5e-2, atol = 1e-6)
    end
end

@testitem "completeness-thinned NB linear form is unstable" setup=[
    CompletenessNBCase] tags=[:ad] begin
    # Document the failure the log-space form fixes: the linear thinning drives
    # the rate to (numerically) zero, so the negative-binomial success
    # probability saturates to 1, a degenerate point mass whose density at a
    # positive count is -Inf. The log-space form at the same point is finite.
    @test linear_nb_logpdf(θ0) == -Inf
    @test isfinite(stable_nb_logpdf(θ0))
end

@testitem "completeness-thinned NB: ForwardDiff" setup=[CompletenessNBCase] tags=[
    :ad, :forwarddiff] begin
    using ADTypes: AutoForwardDiff
    using DifferentiationInterface: DifferentiationInterface, gradient
    using ForwardDiff: ForwardDiff
    check_stable(AutoForwardDiff())
end

@testitem "completeness-thinned NB: ReverseDiff" setup=[CompletenessNBCase] tags=[
    :ad, :reversediff] begin
    using ADTypes: AutoForwardDiff, AutoReverseDiff
    using DifferentiationInterface: DifferentiationInterface, gradient
    using ForwardDiff: ForwardDiff
    using ReverseDiff: ReverseDiff
    check_stable(AutoReverseDiff(compile = false))
end

@testitem "completeness-thinned NB: Mooncake reverse" setup=[
    CompletenessNBCase] tags=[:ad, :mooncake, :mooncake_reverse] begin
    using ADTypes: AutoForwardDiff, AutoMooncake
    using DifferentiationInterface: DifferentiationInterface, gradient
    using ForwardDiff: ForwardDiff
    using Mooncake: Mooncake
    check_stable(AutoMooncake(config = nothing))
end
