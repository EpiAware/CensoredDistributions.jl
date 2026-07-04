# AD coverage for the latent event-time form (ForwardDiff, in the AD env). Adds
# the parameter-level marginal==latent equivalence over a trapezoidal integral;
# the backend matrix covers the scalar conditional gradient in `test/ADFixtures`.

@testitem "latent joint logpdf is ForwardDiff-safe in the parameters" tags=[
    :ad, :forwarddiff] begin
    using CensoredDistributions, Distributions
    using ForwardDiff: gradient

    # The marginal == latent equivalence holds at the gradient level: the
    # parameter gradient of the marginal logpdf equals that of the integrated
    # latent joint.
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

    # A record flooring to zero evaluates the delay cdf at its support boundary,
    # where `cdf(LogNormal, 0)` is 0 but the naive derivative is `0 * Inf = NaN`;
    # guarding the boundary keeps the gradient finite for the fit.
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
