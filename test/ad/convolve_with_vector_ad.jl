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

@testitem "build-once DelayPMF gradient == rebuild-every-time path" tags=[
    :ad, :forwarddiff] begin
    # The build-once PMF object must differentiate cleanly w.r.t. the delay
    # parameters and give the SAME gradient as rediscretising on every call:
    # `discretise_pmf` builds the masses from the (possibly `Dual`) params, so
    # the discretisation is on the differentiated path and no stale cache hides
    # the parameter dependence.
    using CensoredDistributions, Distributions
    using ForwardDiff
    using DifferentiationInterface: AutoForwardDiff, AutoFiniteDifferences,
                                    gradient
    using FiniteDifferences: central_fdm

    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]
    fd = AutoFiniteDifferences(; fdm = central_fdm(5, 1))

    # Build the PMF ONCE inside the objective, then convolve the series.
    function objective_build_once(theta)
        delay = convolve_distributions(
            Gamma(theta[1], theta[2]), LogNormal(theta[3], theta[4]))
        pmf = CensoredDistributions.discretise_pmf(delay, length(series) - 1)
        return sum(convolve_distributions(pmf, series))
    end

    # The rebuild-every-time path for the same delay.
    function objective_rebuild(theta)
        delay = convolve_distributions(
            Gamma(theta[1], theta[2]), LogNormal(theta[3], theta[4]))
        return sum(convolve_distributions(delay, series))
    end

    theta = [2.0, 1.0, 0.5, 0.4]
    g_once = gradient(objective_build_once, AutoForwardDiff(), theta)
    g_rebuild = gradient(objective_rebuild, AutoForwardDiff(), theta)
    g_fd = gradient(objective_build_once, fd, theta)

    @test all(isfinite, g_once)
    @test any(!iszero, g_once)
    # Identical gradient to the rebuild-every-time path, and to finite diff.
    @test isapprox(g_once, g_rebuild; rtol = 1e-10, atol = 1e-12)
    @test isapprox(g_once, g_fd; rtol = 1e-4, atol = 1e-6)
end

@testitem "build-once DelayPMF gradient flows (ReverseDiff)" tags=[
    :ad, :reversediff] begin
    using CensoredDistributions, Distributions
    using ReverseDiff
    using DifferentiationInterface: AutoReverseDiff, AutoFiniteDifferences,
                                    gradient
    using FiniteDifferences: central_fdm

    series = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0]
    fd = AutoFiniteDifferences(; fdm = central_fdm(5, 1))

    function objective(theta)
        delay = convolve_distributions(
            Gamma(theta[1], theta[2]), LogNormal(theta[3], theta[4]))
        pmf = CensoredDistributions.discretise_pmf(delay, length(series) - 1)
        return sum(convolve_distributions(pmf, series))
    end

    theta = [2.0, 1.0, 0.5, 0.4]
    g_rev = gradient(objective, AutoReverseDiff(), theta)
    g_fd = gradient(objective, fd, theta)

    @test all(isfinite, g_rev)
    @test any(!iszero, g_rev)
    @test isapprox(g_rev, g_fd; rtol = 1e-4, atol = 1e-6)
end

@testitem "branched convolve gradient flows (Mooncake reverse)" tags=[
    :ad, :mooncake, :mooncake_reverse] begin
    # The Rt tutorial's shape: a shared incubation step that fans out into a
    # thinned `cases` branch and a thinned `deaths` branch, pushed through
    # `convolve_distributions(stack, infections; events = (:cases, :deaths))`.
    # Deriving the requested event names runs string ops Mooncake reverse
    # cannot trace; the zero-adjoint primitives in the Mooncake extension let
    # the gradient flow without a per-model overlay. The names are constant
    # w.r.t. the sampled parameters, so the result must match ForwardDiff.
    using CensoredDistributions, Distributions
    using ForwardDiff
    using ADTypes: AutoForwardDiff, AutoMooncake
    using DifferentiationInterface: gradient
    import Mooncake

    infections = [0.0, 2.0, 5.0, 9.0, 14.0, 11.0, 7.0, 4.0]

    # theta = (incub shape, incub scale, case shape, case scale, alpha,
    #          death shape, death scale, rho); the scales enter via `thin`.
    function objective(theta)
        incub = Gamma(theta[1], theta[2])
        stack = compose(incub;
            cases = thin(Gamma(theta[3], theta[4]), theta[5]),
            deaths = thin(Gamma(theta[6], theta[7]), theta[8]))
        streams = convolve_distributions(stack, infections;
            events = (:cases, :deaths))
        return sum(streams.cases) + sum(streams.deaths)
    end

    theta = [1.8, 1.4, 1.5, 1.2, 0.3, 3.0, 4.0, 0.012]
    g_fwd = gradient(objective, AutoForwardDiff(), theta)
    g_mnc = gradient(
        objective, AutoMooncake(; config = nothing), theta)

    @test all(isfinite, g_mnc)
    @test any(!iszero, g_mnc)
    @test isapprox(g_mnc, g_fwd; rtol = 1e-5, atol = 1e-8)
end
