# AD coverage for the hazard-modified `modify`/`Modified` leaf.
#
# `modify(d, effect; link)` builds a `Modified` whose hazard is reshaped through
# the link, instantiated lazily so it nests as a leaf everywhere a
# `UnivariateDistribution` does. The gradient of `logpdf(modify(d, effect; link),
# x)` w.r.t. the base delay's params, the scalar effect and the hazard ratio must
# flow through each instantiation path:
#
# - continuous `link = log`: the analytic proportional-hazards survival `S^θ`
#   (`logpdf* = β + logpdf + (θ-1) logccdf`);
# - continuous `link = identity`: the analytic additive-hazards survival
#   `S e^{-βt}` (`logpdf* = log(h+β) + logS - βt`);
# - continuous general link (`:logit`): the numeric modified cumulative hazard
#   `H* = ∫ g⁻¹(g(h)+effect)` through the same Gauss-Legendre quadrature
#   primary-censoring uses;
# - discrete `link = :logit` on an interval-censored base: the exact per-bin PMF
#   reconstruction (`_apply_hazard_link`);
# - `modify` as a LEAF inside compose / primary_censored / interval_censored /
#   truncate_to_horizon: the composed gradient flows through the wrapper into the
#   modified leaf's params and effect.
#
# Each backend item compares against the ForwardDiff reference. The continuous
# paths and every leaf-in-composer path differentiate on all six backends. The
# DISCRETE per-bin path differentiates on ForwardDiff / ReverseDiff / Mooncake
# (both modes) but NOT on Enzyme (either mode): `_apply_hazard_link` builds the
# modified hazard with an in-place `Vector{T}(undef, n)` mutating loop and
# reconstructs the PMF, which Enzyme's type analysis cannot trace
# (`EnzymeMutabilityException` / `EnzymeNoDerivativeError`). That is an upstream
# Enzyme limitation on the mutating array-build reconstruction, not a `modify`
# bug, so the discrete Enzyme items are `@test_broken` (see the per-item notes).

# ============================================================================
# Shared scoring functions, defined once and reused per backend item.
# ============================================================================

@testsnippet ModifyAD begin
    using CensoredDistributions, Distributions

    const OBS = [0.5, 1.2, 2.5, 3.8, 5.1]
    const OBS_INT = [0.0, 1.0, 2.0, 3.0, 4.0]

    # Continuous analytic log link. θ = [μ, σ, β]: the gradient flows through the
    # base LogNormal params AND the scalar log-hazard effect β (θ = exp(β) the
    # hazard ratio).
    f_log(θ) = sum(
        x -> logpdf(modify(LogNormal(θ[1], θ[2]), θ[3]; link = log), x), OBS)

    # Continuous analytic identity link. θ = [μ, σ, β]: additive hazard h + β.
    f_identity(θ) = sum(
        x -> logpdf(modify(LogNormal(θ[1], θ[2]), θ[3]; link = identity), x),
        OBS)

    # Continuous general link via the Gauss-Legendre quadrature path: a logit
    # link on a continuous Gamma base routes through the numeric modified
    # cumulative hazard. θ = [k, scale, β].
    f_numeric(θ) = sum(
        x -> logpdf(modify(Gamma(θ[1], θ[2]), θ[3]; link = :logit), x), OBS)

    # Discrete per-bin logit path, gradient wrt the per-bin effect VECTOR on a
    # daily interval-censored base. The final-bin effect carries a zero gradient
    # (its hazard is pinned to one).
    f_discrete_eff(θ) = sum(
        x -> logpdf(
            modify(interval_censored(LogNormal(1.5, 0.5), 1.0), θ;
                link = :logit), x),
        OBS_INT)

    # Discrete per-bin logit path, gradient wrt the BASE delay params (effect
    # fixed). θ = [μ, σ].
    function f_discrete_base(θ)
        eff = fill(0.2, 5)
        return sum(
            x -> logpdf(
                modify(interval_censored(LogNormal(θ[1], θ[2]), 1.0), eff;
                    link = :logit), x),
            OBS_INT)
    end

    # `modify` as a leaf inside `compose`: a Parallel of a modified branch and a
    # plain branch, scored at the branch value vector. θ = [μ1, μ2, β].
    function f_compose(θ)
        tree = compose((
            a = modify(LogNormal(θ[1], 0.5), θ[3]; link = log),
            b = LogNormal(θ[2], 0.4)))
        return logpdf(tree, [2.0, 3.0])
    end

    # `modify` as the leaf of `primary_censored`. θ = [μ, σ, β].
    f_primary(θ) = sum(
        x -> logpdf(
            primary_censored(
                modify(LogNormal(θ[1], θ[2]), θ[3]; link = log),
                Uniform(0, 1)), x),
        OBS)

    # `modify` as the leaf of `interval_censored`. θ = [μ, σ, β].
    f_interval(θ) = sum(
        x -> logpdf(
            interval_censored(
                modify(LogNormal(θ[1], θ[2]), θ[3]; link = log), 1.0), x),
        OBS_INT)

    # `modify` as the leaf of `truncate_to_horizon`. θ = [μ, σ, β].
    f_truncate(θ) = sum(
        x -> logpdf(
            truncate_to_horizon(
                modify(LogNormal(θ[1], θ[2]), θ[3]; link = log), 8.0), x),
        OBS)

    # The continuous + leaf-in-composer paths every backend must differentiate,
    # paired with their parameter points.
    const CONTINUOUS_CASES = [
        ("log link", f_log, [1.5, 0.5, -0.4]),
        ("identity link", f_identity, [1.5, 0.5, 0.15]),
        ("numeric logit link", f_numeric, [2.0, 1.5, 0.3]),
        ("compose leaf", f_compose, [1.5, 0.5, -0.3]),
        ("primary_censored leaf", f_primary, [1.5, 0.5, -0.4]),
        ("interval_censored leaf", f_interval, [1.5, 0.5, -0.4]),
        ("truncate_to_horizon leaf", f_truncate, [1.5, 0.5, -0.4])
    ]

    # The discrete per-bin paths, broken out so the Enzyme items can mark them
    # `@test_broken` while the other backends test them hard.
    const DISCRETE_CASES = [
        ("discrete logit wrt effects", f_discrete_eff, fill(0.2, 5)),
        ("discrete logit wrt base", f_discrete_base, [1.5, 0.5])
    ]
end

# A working backend on a case: gradient matches the ForwardDiff reference,
# finite, and non-trivial.
@testsnippet ModifyADCheck begin
    using ADTypes: AutoForwardDiff
    using DifferentiationInterface: gradient
    using ForwardDiff: ForwardDiff

    function check_matches_reference(f, θ, backend)
        ref = gradient(f, AutoForwardDiff(), θ)
        g = gradient(f, backend, θ)
        @test g isa AbstractVector && length(g) == length(θ)
        @test all(isfinite, g)
        @test any(!iszero, g)
        @test isapprox(g, ref; rtol = 1e-4, atol = 1e-6)
    end
end

# ============================================================================
# ForwardDiff: the reference itself, so just assert finite/non-trivial.
# ============================================================================

@testitem "modify gradient: ForwardDiff" tags=[:ad, :forwarddiff] setup=[
    ModifyAD] begin
    using ADTypes: AutoForwardDiff
    using DifferentiationInterface: gradient
    using ForwardDiff: ForwardDiff

    for (name, f, θ) in vcat(CONTINUOUS_CASES, DISCRETE_CASES)
        g=gradient(f, AutoForwardDiff(), θ)
        @test g isa AbstractVector && length(g) == length(θ)
        @test all(isfinite, g)
        @test any(!iszero, g)
    end
end

# ============================================================================
# ReverseDiff: every path, hard against the reference.
# ============================================================================

@testitem "modify gradient: ReverseDiff" tags=[:ad, :reversediff] setup=[
    ModifyAD, ModifyADCheck] begin
    using ADTypes: AutoReverseDiff
    using ReverseDiff: ReverseDiff

    backend=AutoReverseDiff(compile = false)
    for (name, f, θ) in vcat(CONTINUOUS_CASES, DISCRETE_CASES)
        check_matches_reference(f, θ, backend)
    end
end

# ============================================================================
# Mooncake reverse: every path, hard against the reference.
# ============================================================================

@testitem "modify gradient: Mooncake reverse" tags=[
    :ad, :mooncake, :mooncake_reverse] setup=[ModifyAD, ModifyADCheck] begin
    using ADTypes: AutoMooncake
    using Mooncake: Mooncake

    backend=AutoMooncake(config = nothing)
    for (name, f, θ) in vcat(CONTINUOUS_CASES, DISCRETE_CASES)
        check_matches_reference(f, θ, backend)
    end
end

# ============================================================================
# Mooncake forward: every path, hard against the reference.
# ============================================================================

@testitem "modify gradient: Mooncake forward" tags=[
    :ad, :mooncake, :mooncake_forward] setup=[ModifyAD, ModifyADCheck] begin
    using ADTypes: AutoMooncakeForward
    using Mooncake: Mooncake

    backend=AutoMooncakeForward()
    for (name, f, θ) in vcat(CONTINUOUS_CASES, DISCRETE_CASES)
        check_matches_reference(f, θ, backend)
    end
end

# ============================================================================
# Enzyme reverse: continuous + leaf-in-composer paths pass; the discrete
# per-bin path is a known Enzyme limitation, marked `@test_broken`.
# ============================================================================

@testitem "modify gradient: Enzyme reverse" tags=[
    :ad, :enzyme, :enzyme_reverse] setup=[ModifyAD, ModifyADCheck] begin
    using ADTypes: AutoEnzyme
    using Enzyme: Enzyme

    backend=AutoEnzyme(mode = Enzyme.set_runtime_activity(Enzyme.Reverse))
    for (name, f, θ) in CONTINUOUS_CASES
        check_matches_reference(f, θ, backend)
    end

    # DISCRETE per-bin path: `_apply_hazard_link` builds the modified hazard in an
    # in-place `Vector{T}(undef, n)` mutating loop and reconstructs the PMF, which
    # Enzyme reverse cannot trace (`EnzymeMutabilityException` /
    # `EnzymeNoDerivativeError`). This is an upstream Enzyme limitation on the
    # mutating array-build reconstruction, not a `modify` bug: ForwardDiff,
    # ReverseDiff and Mooncake (both modes) all differentiate it correctly.
    using DifferentiationInterface: gradient
    for (name, f, θ) in DISCRETE_CASES
        ok=try
            g=gradient(f, backend, θ)
            g isa AbstractVector&&all(isfinite, g)
        catch
            false
        end
        @test_broken ok
    end
end

# ============================================================================
# Enzyme forward: continuous + leaf-in-composer paths pass; the discrete
# per-bin path is the same known Enzyme limitation, marked `@test_broken`.
# ============================================================================

@testitem "modify gradient: Enzyme forward" tags=[
    :ad, :enzyme, :enzyme_forward] setup=[ModifyAD, ModifyADCheck] begin
    using ADTypes: AutoEnzyme
    using Enzyme: Enzyme

    backend=AutoEnzyme(mode = Enzyme.set_runtime_activity(Enzyme.Forward))
    for (name, f, θ) in CONTINUOUS_CASES
        check_matches_reference(f, θ, backend)
    end

    # DISCRETE per-bin path: same upstream Enzyme limitation as the reverse item
    # above (the `_apply_hazard_link` mutating array-build reconstruction). Marked
    # `@test_broken`; the gradient correctness is covered by ForwardDiff /
    # ReverseDiff / Mooncake.
    using DifferentiationInterface: gradient
    for (name, f, θ) in DISCRETE_CASES
        ok=try
            g=gradient(f, backend, θ)
            g isa AbstractVector&&all(isfinite, g)
        catch
            false
        end
        @test_broken ok
    end
end
