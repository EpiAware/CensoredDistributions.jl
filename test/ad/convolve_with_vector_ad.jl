# AD coverage for the `convolve_distributions(stack, series)` renewal method.
# The vector convolution is linear and the delay PMF depends differentiably on
# the delay parameters, so the gradient of a scalar summary of the output
# w.r.t. the delay params must flow under ForwardDiff and ReverseDiff and match
# a finite-difference reference.

@testitem "convolve_distributions(stack, series) gradient flows" tags=[
    :ad, :forwarddiff] begin
    using CensoredDistributions, Distributions
    using ForwardDiff
    using DifferentiationInterface: AutoForwardDiff, AutoFiniteDifferences,
                                    gradient
    using FiniteDifferences: central_fdm

    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]
    fd = AutoFiniteDifferences(; fdm = central_fdm(5, 1))

    # Scalar objective: sum of the endpoint count series of a two-step chain,
    # as a function of the (shape, scale) of each step delay.
    function objective(theta)
        seq = Sequential(
            Gamma(theta[1], theta[2]), LogNormal(theta[3], theta[4]))
        return sum(convolve_distributions(seq, series))
    end

    theta = [2.0, 1.0, 0.5, 0.4]
    g_fwd = gradient(objective, AutoForwardDiff(), theta)
    g_fd = gradient(objective, fd, theta)

    @test all(isfinite, g_fwd)
    @test isapprox(g_fwd, g_fd; rtol = 1e-4, atol = 1e-6)

    # The gradient is non-trivial (the params genuinely move the output).
    @test any(!iszero, g_fwd)
end

@testitem "convolve_distributions interim gradient flows (ReverseDiff)" tags=[
    :ad, :reversediff] begin
    using CensoredDistributions, Distributions
    using ReverseDiff
    using DifferentiationInterface: AutoReverseDiff, AutoFiniteDifferences,
                                    gradient
    using FiniteDifferences: central_fdm

    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]
    fd = AutoFiniteDifferences(; fdm = central_fdm(5, 1))

    # Objective on a named INTERIM event's series (prefix-1 delay only).
    function objective(theta)
        seq = Sequential(
            (Gamma(theta[1], theta[2]), LogNormal(theta[3], theta[4])),
            (:onset_admit, :admit_death))
        return sum(convolve_distributions(seq, series; events = :admit))
    end

    theta = [2.0, 1.0, 0.5, 0.4]
    g_rev = gradient(objective, AutoReverseDiff(), theta)
    g_fd = gradient(objective, fd, theta)

    @test all(isfinite, g_rev)
    @test isapprox(g_rev, g_fd; rtol = 1e-4, atol = 1e-6)

    # Only the first leaf feeds the :admit interim, so the LogNormal params
    # (entries 3, 4) have zero gradient.
    @test isapprox(g_rev[3], 0.0; atol = 1e-8)
    @test isapprox(g_rev[4], 0.0; atol = 1e-8)
    @test any(!iszero, g_rev[1:2])
end
