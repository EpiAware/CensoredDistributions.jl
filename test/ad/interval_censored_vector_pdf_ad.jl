# AD coverage for the vectorised `pdf`/`logpdf` of `IntervalCensored`.
#
# The batched path caches the interval CDFs in an array whose element type
# must follow the DISTRIBUTION's parameter type, not the evaluation points
# (#403). When the distribution carries `Dual`/tracked parameters but the
# grid and boundaries are plain `Float64`, a type pinned to the eval points
# strips the AD numbers and either errors (`Float64(::Dual)`) or returns a
# zero gradient. The gradient of a scalar summary of the batched PDF w.r.t.
# the distribution parameters must flow and match the scalar-broadcast path.

@testitem "IntervalCensored vectorised pdf gradient flows (ForwardDiff)" tags=[
    :ad, :forwarddiff] begin
    using CensoredDistributions, Distributions
    using ForwardDiff: gradient

    grid = collect(0.0:1.0:10.0)

    # Scalar-broadcast reference: maps the scalar `pdf` over the grid, which
    # keeps `Dual`s alive (the scalar path is unaffected by #403).
    function obj_scalar(theta)
        ic = interval_censored(Gamma(theta[1], theta[2]), 1.0)
        return sum(pdf.(Ref(ic), grid))
    end

    # Batched path under test: a single `pdf(ic, grid)` call.
    function obj_vec(theta)
        ic = interval_censored(Gamma(theta[1], theta[2]), 1.0)
        return sum(pdf(ic, grid))
    end

    function obj_vec_logpdf(theta)
        ic = interval_censored(Gamma(theta[1], theta[2]), 1.0)
        return sum(exp, logpdf(ic, grid))
    end

    theta = [2.0, 1.5]

    g_scalar = gradient(obj_scalar, theta)
    g_vec = gradient(obj_vec, theta)
    g_vec_log = gradient(obj_vec_logpdf, theta)

    @test all(isfinite, g_vec)
    @test any(!iszero, g_vec)
    @test isapprox(g_vec, g_scalar; rtol = 1e-8, atol = 1e-10)
    @test isapprox(g_vec_log, g_scalar; rtol = 1e-8, atol = 1e-10)
end

@testitem "IntervalCensored vectorised pdf gradient flows, arbitrary intervals (ForwardDiff)" tags=[
    :ad, :forwarddiff] begin
    using CensoredDistributions, Distributions
    using ForwardDiff: gradient

    boundaries = [0.0, 1.0, 2.5, 4.0, 6.0, 9.0]
    grid = [0.5, 1.5, 3.0, 5.0, 7.0, 8.5]

    function obj_scalar(theta)
        ic = interval_censored(Gamma(theta[1], theta[2]), boundaries)
        return sum(pdf.(Ref(ic), grid))
    end

    function obj_vec(theta)
        ic = interval_censored(Gamma(theta[1], theta[2]), boundaries)
        return sum(pdf(ic, grid))
    end

    theta = [2.0, 1.5]

    g_scalar = gradient(obj_scalar, theta)
    g_vec = gradient(obj_vec, theta)

    @test all(isfinite, g_vec)
    @test any(!iszero, g_vec)
    @test isapprox(g_vec, g_scalar; rtol = 1e-8, atol = 1e-10)
end

@testitem "IntervalCensored vectorised pdf gradient flows (ReverseDiff)" tags=[
    :ad, :reversediff] begin
    using CensoredDistributions, Distributions
    using ReverseDiff: gradient

    grid = collect(0.0:1.0:10.0)

    function obj_scalar(theta)
        ic = interval_censored(Gamma(theta[1], theta[2]), 1.0)
        return sum(pdf.(Ref(ic), grid))
    end

    function obj_vec(theta)
        ic = interval_censored(Gamma(theta[1], theta[2]), 1.0)
        return sum(pdf(ic, grid))
    end

    theta = [2.0, 1.5]

    g_scalar = gradient(obj_scalar, theta)
    g_vec = gradient(obj_vec, theta)

    @test all(isfinite, g_vec)
    @test any(!iszero, g_vec)
    @test isapprox(g_vec, g_scalar; rtol = 1e-8, atol = 1e-10)
end
