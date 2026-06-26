# AD coverage for the `thin` resolve+NoEvent change and the distribution
# modifiers / combinators (`affine`, `weight`, `Convolved`, `Difference`) and
# their nested combinations.
#
# `thin(d, p)` now carries the probabilistic one-of `resolve(:event => (d, p),
# :none => (NoEvent(), 1 - p))` into `logpdf` / `rand`, so its scalar density is
# the DEFECTIVE `log(p) + logpdf(d, x)` (mass `p`) rather than the old
# logpdf-transparent forward scaler. That makes `logpdf(thin(d, p), x)`
# differentiable w.r.t. BOTH the reporting probability `p` (the `log p` term)
# and the base delay parameters (the `logpdf(d, x)` term). The gradient w.r.t.
# `p` is exactly `1/p` (per observation), which these items pin against the
# analytic value as well as the cross-backend ForwardDiff reference.
#
# The modifiers each touch a different part of the density: `affine` the
# change-of-variables `logpdf(inner, (y - shift)/scale) - log(scale)` (gradient
# through scale/shift and inner params), `weight` the `w * logpdf` scaling
# (gradient through `w` and inner params), and `Convolved` / `Difference` the
# AD-safe Gauss-Legendre quadrature (gradient through the component params).
# Nested combos check a modifier wrapping a composed node and a composed tree
# carrying these as leaves, so the gradient flows through the whole tree.
#
# Each case is checked on every target backend (ForwardDiff, ReverseDiff,
# Mooncake reverse+forward, Enzyme reverse+forward) against a ForwardDiff
# reference; one item per backend keeps the per-backend CI tags meaningful. As
# of this file every case differentiates correctly on every backend — no path
# is `@test_broken`. If a backend regresses on one case in future, split that
# case into its own item and mark it `@test_broken` with the
# backend × path × reason, rather than reding the whole backend.

# ----------------------------------------------------------------------------
# Shared case definitions (a snippet, re-included into each backend item so the
# file stays self-contained — no ADFixtures broken-registry entry is needed).
# Each case is `(name, f, θ)`: `f(θ)` is a scalar log density and `θ` the
# differentiated parameter vector. `p_pin`, when set, is `(index, n)` — the θ
# index of a `thin` reporting probability and the number of observations scored
# — whose analytic gradient `n/p` is additionally pinned.
# ----------------------------------------------------------------------------
@testsnippet ThinModifierCases begin
    using CensoredDistributions, Distributions

    # Shared observation grids. `obs` lands inside every continuous support
    # used below; `seq_ev` / `par_ev` are the value vectors the composed trees
    # score.
    const obs = [0.5, 1.2, 2.5, 3.8]
    const n_obs = length(obs)

    # `(name, f, θ, p_pin)` where `p_pin` is `nothing` or `(index, n)`.
    function thin_modifier_cases()
        return [
            # --- thin: resolve+NoEvent defective density ---------------------
            # logpdf = log(p) + logpdf(LogNormal, x). Gradient flows through the
            # base μ/σ (θ[1], θ[2]) AND the reporting probability p (θ[3]); the
            # p-gradient is n_obs / p exactly (n_obs observations scored).
            ("thin LogNormal logpdf wrt base+p",
                θ -> sum(x -> logpdf(thin(LogNormal(θ[1], θ[2]), θ[3]), x), obs),
                [1.0, 0.5, 0.3], (3, n_obs)),

            # thin OVER a convolution: the same defective `log p + logpdf(conv)`,
            # with the base term now the AD-safe numeric convolution density. p
            # (θ[3]) and both convolved component params differentiate.
            ("thin(Convolved Gamma+LogNormal) logpdf wrt base+p",
                θ -> sum(
                    x -> logpdf(
                        thin(
                            convolved(
                                Gamma(θ[1], θ[2]), LogNormal(0.5, 0.4)), θ[3]),
                        x), obs),
                [2.0, 1.0, 0.4], (3, n_obs)),

            # thin with p FIXED, gradient through the convolved base params only
            # (the `logpdf(d, x)` term): isolates the base-parameter path from
            # the `log p` term above.
            ("thin(Convolved) base params, p fixed",
                θ -> sum(
                    x -> logpdf(
                        thin(
                            convolved(
                                Gamma(θ[1], θ[2]), LogNormal(θ[3], θ[4])), 0.3),
                        x), obs),
                [2.0, 1.0, 0.5, 0.4], nothing),

            # --- affine: change-of-variables logpdf --------------------------
            # logpdf = logpdf(inner, (y - shift)/scale) - log(scale). Gradient
            # through inner μ/σ (θ[1], θ[2]), scale (θ[3]) and shift (θ[4]).
            ("affine LogNormal logpdf wrt inner+scale+shift",
                θ -> sum(
                    y -> logpdf(
                        affine(LogNormal(θ[1], θ[2]); scale = θ[3], shift = θ[4]),
                        y), [2.0, 3.5, 5.0, 7.0]),
                [1.0, 0.5, 2.0, 1.0], nothing),

            # --- weight: w * logpdf ------------------------------------------
            # The logpdf-side modifier. Gradient through inner μ/σ (θ[1], θ[2])
            # and the scalar weight w (θ[3]); the w-gradient is the unweighted
            # base logpdf sum.
            ("weight LogNormal logpdf wrt inner+w",
                θ -> sum(
                    i -> logpdf(weight(LogNormal(θ[1], θ[2]), θ[3]), obs[i]),
                    eachindex(obs)),
                [1.0, 0.5, 2.5], nothing),

            # --- Convolved / Difference of delays ----------------------------
            # Numeric convolution density: gradient through every component
            # parameter via the Gauss-Legendre quadrature.
            ("Convolved Gamma+LogNormal logpdf wrt components",
                θ -> sum(
                    x -> logpdf(
                        convolved(
                            Gamma(θ[1], θ[2]), LogNormal(θ[3], θ[4])), x), obs),
                [2.0, 1.0, 0.5, 0.4], nothing),
            # Numeric cross-correlation difference density: gradient through the
            # minuend AND subtrahend component parameters.
            ("Difference Gamma-LogNormal logpdf wrt components",
                θ -> sum(
                    z -> logpdf(
                        difference(
                            Gamma(θ[1], θ[2]), LogNormal(θ[3], θ[4])), z), obs),
                [3.0, 1.0, 0.5, 0.4], nothing),

            # --- nested combos -----------------------------------------------
            # A modifier (thin) wrapping a composed node (affine): the defective
            # `log p + logpdf(affine, x)`. Gradient through inner μ/σ (θ[1],
            # θ[2]), the affine scale (θ[3]) and the thin probability p (θ[4]).
            ("thin(affine LogNormal) nested wrt inner+scale+p",
                θ -> sum(
                    y -> logpdf(
                        thin(
                            affine(LogNormal(θ[1], θ[2]); scale = θ[3],
                                shift = 0.5), θ[4]), y), [2.0, 3.5, 5.0]),
                [1.0, 0.5, 2.0, 0.3], (4, 3)),
            # A composed tree (Sequential) whose first leaf is a Convolved node:
            # AD flows through the tree's slice recursion into the leaf
            # quadrature. Gradient through the convolved component params (θ[1],
            # θ[2]) and the second leaf's params (θ[3], θ[4]).
            ("Sequential[Convolved, LogNormal] tree logpdf",
                θ -> logpdf(
                    Sequential(
                        convolved(
                            Gamma(θ[1], θ[2]), LogNormal(0.5, 0.4)),
                        LogNormal(θ[3], θ[4])), [3.0, 2.0]),
                [2.0, 1.0, 0.5, 0.4], nothing),
            # A composed tree (Parallel) carrying a Difference leaf: AD through
            # the tree into the difference quadrature. Gradient through the
            # Difference's two Normal components (θ[1:4]).
            ("Parallel[Difference, LogNormal] tree logpdf",
                θ -> logpdf(
                    Parallel(
                        difference(Normal(θ[1], θ[2]), Normal(θ[3], θ[4])),
                        LogNormal(0.5, 0.4)), [1.0, 2.0]),
                [5.0, 1.0, 2.0, 1.0], nothing),
            # A composed tree of two modifier leaves (affine and a plain leaf):
            # both modifier change-of-variables logpdfs feed the tree score.
            ("Sequential[affine Gamma, LogNormal] tree logpdf",
                θ -> logpdf(
                    Sequential(
                        affine(Gamma(θ[1], θ[2]); scale = θ[3], shift = 0.0),
                        LogNormal(θ[4], θ[5])), [1.5, 2.0]),
                [2.0, 1.0, 1.5, 0.5, 0.4], nothing)
        ]
    end

    # Run every case on `backend` against a ForwardDiff reference: finite,
    # nonzero, and matching the reference. When a case carries a `thin`
    # probability index, also pin its analytic per-observation `n/p` gradient.
    function check_cases(backend)
        for (name, f, θ, p_pin) in thin_modifier_cases()
            ref = DifferentiationInterface.gradient(f, AutoForwardDiff(), θ)
            g = DifferentiationInterface.gradient(f, backend, θ)
            @test g isa AbstractVector
            @test length(g) == length(θ)
            @test all(isfinite, g)
            @test any(!iszero, g)
            @test isapprox(g, ref; rtol = 5e-2, atol = 1e-6)
            if p_pin !== nothing
                # The `log p` term contributes `1/p` per observation, so the
                # gradient w.r.t. the thin probability is exactly `n / p` for
                # the all-observed grids used here (`n` observations scored).
                p_index, n = p_pin
                @test isapprox(g[p_index], n / θ[p_index];
                    rtol = 1e-4, atol = 1e-6)
            end
        end
    end
end

@testitem "thin+modifier gradients: ForwardDiff" tags=[:ad, :forwarddiff] setup=[ThinModifierCases] begin
    using ADTypes: AutoForwardDiff
    using DifferentiationInterface: DifferentiationInterface, gradient
    using ForwardDiff: ForwardDiff
    check_cases(AutoForwardDiff())
end

@testitem "thin+modifier gradients: ReverseDiff" tags=[:ad, :reversediff] setup=[ThinModifierCases] begin
    using ADTypes: AutoForwardDiff, AutoReverseDiff
    using DifferentiationInterface: DifferentiationInterface, gradient
    using ForwardDiff: ForwardDiff
    using ReverseDiff: ReverseDiff
    check_cases(AutoReverseDiff(compile = false))
end

@testitem "thin+modifier gradients: Mooncake reverse" tags=[
    :ad, :mooncake, :mooncake_reverse] setup=[ThinModifierCases] begin
    using ADTypes: AutoForwardDiff, AutoMooncake
    using DifferentiationInterface: DifferentiationInterface, gradient
    using ForwardDiff: ForwardDiff
    using Mooncake: Mooncake
    check_cases(AutoMooncake(config = nothing))
end

@testitem "thin+modifier gradients: Mooncake forward" tags=[
    :ad, :mooncake, :mooncake_forward] setup=[ThinModifierCases] begin
    using ADTypes: AutoForwardDiff, AutoMooncakeForward
    using DifferentiationInterface: DifferentiationInterface, gradient
    using ForwardDiff: ForwardDiff
    using Mooncake: Mooncake
    check_cases(AutoMooncakeForward())
end

@testitem "thin+modifier gradients: Enzyme reverse" tags=[
    :ad, :enzyme, :enzyme_reverse] setup=[ThinModifierCases] begin
    using ADTypes: AutoForwardDiff, AutoEnzyme
    using DifferentiationInterface: DifferentiationInterface, gradient
    using ForwardDiff: ForwardDiff
    using Enzyme: Enzyme
    check_cases(AutoEnzyme(mode = Enzyme.set_runtime_activity(Enzyme.Reverse)))
end

@testitem "thin+modifier gradients: Enzyme forward" tags=[
    :ad, :enzyme, :enzyme_forward] setup=[ThinModifierCases] begin
    using ADTypes: AutoForwardDiff, AutoEnzyme
    using DifferentiationInterface: DifferentiationInterface, gradient
    using ForwardDiff: ForwardDiff
    using Enzyme: Enzyme
    check_cases(AutoEnzyme(mode = Enzyme.set_runtime_activity(Enzyme.Forward)))
end
