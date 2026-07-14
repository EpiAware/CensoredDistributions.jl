# AD coverage for a `shared(:tag, ...)` censored leaf in a nested tree.
# The shared wrapper is transparent to scoring, so the
# origin-primary traversal now descends through it (the fix that makes a shared
# censored leaf at a tree origin score identically to the untagged leaf). The
# gradient w.r.t. the shared leaf's params must flow, and a shared leaf used in
# two positions must give the SAME value and gradient as two independent
# identical leaves.

# ForwardDiff over the LATENT-origin path (origin `missing`): the path the fix
# corrects (the origin's primary event is now recovered through the shared
# wrapper). Mooncake reverse is NOT run here: a `Missing`-bearing event vector
# trips Mooncake's `non-boolean (Missing) used in boolean context` on the
# nested-tree marginalisation path for the UNtagged tree too (a pre-existing
# limitation), so it is covered by the observed-origin item below instead.
@testitem "shared censored leaf latent-origin gradient: ForwardDiff" tags=[
    :ad, :forwarddiff] begin
    using CensoredDistributions, Distributions
    using ADTypes: AutoForwardDiff
    using DifferentiationInterface: gradient
    using ForwardDiff: ForwardDiff

    ev = Vector{Union{Missing, Float64}}([missing, 2.0, 5.0])

    # θ = [inc shape, inc scale, b shape, b scale]. `inc` is shared across the
    # origin edge and the routed alternative `:a`.
    function f_shared(θ)
        inc = shared(:inc, primary_censored(Gamma(θ[1], θ[2]), Uniform(0, 1)))
        b = primary_censored(Gamma(θ[3], θ[4]), Uniform(0, 1))
        seq = Sequential((inc, choose(:a => inc, :b => b)),
            (:onset_admit, :admit_death))
        return logpdf(seq, ev)
    end

    # Independent identical untagged leaves in the same positions.
    function f_indep(θ)
        inc1 = primary_censored(Gamma(θ[1], θ[2]), Uniform(0, 1))
        inc2 = primary_censored(Gamma(θ[1], θ[2]), Uniform(0, 1))
        b = primary_censored(Gamma(θ[3], θ[4]), Uniform(0, 1))
        seq = Sequential((inc1, choose(:a => inc2, :b => b)),
            (:onset_admit, :admit_death))
        return logpdf(seq, ev)
    end

    θ = [2.0, 1.0, 5.0, 1.0]
    @test f_shared(θ) ≈ f_indep(θ) atol=1e-12

    g = gradient(f_shared, AutoForwardDiff(), θ)
    gi = gradient(f_indep, AutoForwardDiff(), θ)
    @test g isa AbstractVector && length(g) == 4 && all(isfinite, g)
    # The shared-leaf params (θ[1], θ[2]) drive the gradient; the routed `:a`
    # (not `:b`) means θ[3], θ[4] do not contribute on this record.
    @test all(!=(0), g[1:2])
    @test isapprox(g, gi; rtol = 1e-6, atol = 1e-8)
end

# Mooncake reverse over an OBSERVED-origin shared-leaf tree (no `missing`, so the
# Mooncake/Missing limitation above does not apply). The shared leaf appears in
# two positions; the reverse gradient must compile, stay finite, and match the
# ForwardDiff reference, AND equal the untagged identical-leaf tree's gradient.
@testitem "shared leaf observed-origin gradient: Mooncake reverse" tags=[
    :ad, :mooncake, :mooncake_reverse] begin
    using CensoredDistributions, Distributions
    using ADTypes: AutoForwardDiff, AutoMooncake
    using DifferentiationInterface: gradient
    using ForwardDiff: ForwardDiff
    using Mooncake: Mooncake

    # Observed origin at 0.0; the shared `:inc` edge conditions on the gap and the
    # routed `:a` (also `inc`) conditions on its gap.
    ev = Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0])

    function f_shared(θ)
        inc = shared(:inc, primary_censored(Gamma(θ[1], θ[2]), Uniform(0, 1)))
        b = primary_censored(Gamma(θ[3], θ[4]), Uniform(0, 1))
        seq = Sequential((inc, choose(:a => inc, :b => b)),
            (:onset_admit, :admit_death))
        return logpdf(seq, ev)
    end
    function f_indep(θ)
        inc1 = primary_censored(Gamma(θ[1], θ[2]), Uniform(0, 1))
        inc2 = primary_censored(Gamma(θ[1], θ[2]), Uniform(0, 1))
        b = primary_censored(Gamma(θ[3], θ[4]), Uniform(0, 1))
        seq = Sequential((inc1, choose(:a => inc2, :b => b)),
            (:onset_admit, :admit_death))
        return logpdf(seq, ev)
    end

    θ = [2.0, 1.0, 5.0, 1.0]
    @test f_shared(θ) ≈ f_indep(θ) atol=1e-12

    g_ref = gradient(f_shared, AutoForwardDiff(), θ)
    g = gradient(f_shared, AutoMooncake(config = nothing), θ)
    g_indep = gradient(f_indep, AutoMooncake(config = nothing), θ)
    @test g isa AbstractVector && length(g) == 4 && all(isfinite, g)
    @test isapprox(g, g_ref; rtol = 1e-4, atol = 1e-6)
    @test isapprox(g, g_indep; rtol = 1e-4, atol = 1e-6)
end
