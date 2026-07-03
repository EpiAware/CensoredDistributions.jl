# AD coverage for the latent event-time form, run in the AD environment
# (`test/ad/Project.toml`, which provides ForwardDiff). The latent scalar
# conditional gradient is also exercised across the full backend matrix by the
# `Latent PrimaryConditional LogNormal scalar logpdf` scenario in
# `test/ADFixtures`; this item adds the parameter-level marginal==latent
# equivalence proof that needs a trapezoidal integral over the primary window.

@testitem "latent joint logpdf is ForwardDiff-safe in the parameters" tags=[
    :ad, :forwarddiff] begin
    using CensoredDistributions, Distributions
    using ForwardDiff: gradient

    # The marginal == latent equivalence holds at the gradient level: the
    # parameter gradient of the marginal logpdf equals the gradient of the latent
    # joint integrated over the primary window.
    pe = Uniform(0.0, 1.0)
    y = 2.5

    marginal_logpdf(θ) = logpdf(primary_censored(LogNormal(θ[1], θ[2]), pe), y)
    function latent_int_logpdf(θ; n = 20_000)
        ld = latent(primary_censored(LogNormal(θ[1], θ[2]), pe))
        ps = range(0.0, 1.0; length = n)
        vals = map(p -> exp(logpdf(ld, [p, y])), ps)
        trap = sum((vals[1:(end - 1)] .+ vals[2:end]) ./ 2) * step(ps)
        return log(trap)
    end

    θ = [1.0, 0.5]
    gm = gradient(marginal_logpdf, θ)
    gl = gradient(latent_int_logpdf, θ)
    @test all(isfinite, gm)
    @test all(isfinite, gl)
    @test isapprox(gm, gl; rtol = 1e-3)
end

@testitem "interval-of-latent conditional gradient is finite at a zero delay" tags=[
    :ad, :forwarddiff] begin
    using CensoredDistributions, Distributions
    using ForwardDiff: gradient

    # A record whose observed delay floors to zero clamps the interval's lower
    # edge to the delay's support boundary, so the conditional evaluates the
    # delay cdf at that boundary. There `cdf(LogNormal, 0)` is exactly 0 but its
    # naive parameter derivative is `0 * Inf = NaN`; guarding the boundary keeps
    # the gradient finite so the latent double_interval_censored fit stays
    # AD-safe on real data (many records floor to a zero delay).
    function zero_delay_logpdf(θ; p = 0.5)
        ld = latent(double_interval_censored(LogNormal(θ[1], θ[2]);
            primary_event = Uniform(0, 3), upper = 12, interval = 1))
        return logpdf(ld, 0.0; primary = p)
    end

    θ = [1.5, 0.75]
    @test isfinite(zero_delay_logpdf(θ))
    g = gradient(zero_delay_logpdf, θ)
    @test all(isfinite, g)
end
