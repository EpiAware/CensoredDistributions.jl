# ============================================================================
# PROTOTYPE (issue #611): a composable renewal step
# ============================================================================
#
# This script demonstrates the renewal abstraction end to end. The renewal
# SCAN, the modulators and `combine_modulators` are production code in
# `src/utils/renewal.jl` (public, unexported, pending the scope decision); this
# script is the PROTOTYPE that exercises them, checks the equivalence to the
# rt-renewal tutorial's hand-rolled loop, and runs a small fit. It lives outside
# the test suite so it can be read top to bottom as the worked story.
#
# What it shows end to end:
#   - equivalence : the bare `renewal` scan reproduces the rt-renewal tutorial's
#                   hand-rolled loop EXACTLY (bit-for-bit);
#   - susceptibility : `susceptibility_depletion` gives the SIR-style renewal
#                   `I[t] = R_t (S[t]/N) Σ_s g_s I[t-s]`, `S[t] = S[t-1] - I[t]`,
#                   checked against an independent hand-written SIR loop and
#                   collapsing to the bare renewal as N -> ∞;
#   - composition : `combine_modulators` stacks a susceptibility term with a
#                   constant transmissibility term, the compose idiom carried
#                   from delays to renewal steps;
#   - fit         : recover the susceptible-pool size N from a Poisson-observed
#                   epidemic by maximum likelihood, the gradient flowing through
#                   the whole scan.
#
# Production-scope items the prototype DEFERS are listed at the bottom.

using CensoredDistributions
using Distributions
using Random

const CD = CensoredDistributions

# --- the generation interval (a leaf, as in the rt-renewal tutorial) --------

gi_max = 12
gen_dist = interval_censored(
    truncated(Gamma(2.5, 1.3); lower = 1.0, upper = Float64(gi_max)), 1.0)
g = pdf(gen_dist, 1:gi_max)

# --- 1. equivalence to the hand-rolled tutorial loop ------------------------
#
# The exact loop the rt-renewal tutorial hand-rolls. The new scan must match it
# bit for bit when the modulator is the identity (the default).

function handrolled_renewal(Rt, g, I0; seed_days = length(g))
    n = length(Rt)
    I = zeros(eltype(Rt), n)
    I[1:seed_days] .= I0
    for t in (seed_days + 1):n
        acc = zero(eltype(Rt))
        for s in 1:min(length(g), t - 1)
            acc += g[s] * I[t - s]
        end
        I[t] = Rt[t] * acc
    end
    return I
end

n_days = 70
true_Rt = vcat(fill(1.6, 18), fill(0.7, 18), fill(1.3, 18),
    fill(1.0, n_days - 54))
I0 = 10.0

bare = CD.renewal(true_Rt, g, I0)
ref = handrolled_renewal(true_Rt, g, I0)
@assert bare == ref
println("1. bare renewal == hand-rolled loop (bit-for-bit): ", bare == ref)

# The build-once `DelayPMF` form agrees too (lag-0-indexed masses, lag 0
# dropped), so the renewal consumes either a PMF vector or a `DelayPMF`.
pmf = CD.discretise_pmf(gen_dist, gi_max)
@assert CD.renewal(true_Rt, pmf, I0) ≈ ref
println("   DelayPMF generation interval agrees: ",
    CD.renewal(true_Rt, pmf, I0) ≈ ref)

# --- 2. susceptibility depletion (the SIR-style renewal) --------------------
#
# An independent hand-written SIR renewal to check the modulator against.

function handrolled_sir(Rt, g, I0, N; seed_days = length(g))
    n = length(Rt)
    I = zeros(n)
    I[1:seed_days] .= I0
    S = N - sum(I[1:seed_days])
    for t in (seed_days + 1):n
        acc = 0.0
        for s in 1:min(length(g), t - 1)
            acc += g[s] * I[t - s]
        end
        force = Rt[t] * acc
        I[t] = (S / N) * force
        S -= I[t]
    end
    return I
end

N = 1.0e5
sir = CD.renewal(true_Rt, g, I0; modulator = CD.susceptibility_depletion(N))
sir_ref = handrolled_sir(true_Rt, g, I0, N)
@assert sir ≈ sir_ref
println("2. susceptibility_depletion == hand SIR loop: ", sir ≈ sir_ref)

# As N -> ∞ the susceptible fraction stays ~1 and the SIR renewal collapses to
# the bare renewal: depletion is a strict generalisation.
sir_big = CD.renewal(true_Rt, g, I0;
    modulator = CD.susceptibility_depletion(1.0e12))
rel = maximum(abs.(sir_big .- ref) ./ (ref .+ 1))
println("   N -> inf collapses to bare renewal (max rel diff ",
    round(rel; sigdigits = 2), "): ", rel < 1.0e-3)

# Depletion bends the epidemic down: the SIR peak is below the bare-renewal peak
# because the susceptible pool runs down.
println("   bare peak ", round(maximum(bare); digits = 1),
    " vs SIR peak ", round(maximum(sir); digits = 1),
    " (depletion lowers the peak): ", maximum(sir) < maximum(bare))

# --- 3. composing modulators ------------------------------------------------
#
# `combine_modulators` stacks a susceptibility term with a constant
# transmissibility multiplier (here a seasonal-style 0.8 scaling). The compose
# idiom carried from delays to renewal steps: factors multiply, carry-states
# thread, and the result is itself a modulator so it nests.

struct ConstantFactor{T}
    c::T
end
(m::ConstantFactor)(state, t, force) = (m.c, state)
CD._modulator_init(::ConstantFactor) = nothing

stacked = CD.combine_modulators(
    CD.susceptibility_depletion(N), ConstantFactor(0.8))
composed = CD.renewal(true_Rt, g, I0; modulator = stacked)
println("3. composed (susceptibility * 0.8) runs, finite, below SIR: ",
    all(isfinite, composed) && maximum(composed) < maximum(sir))

# --- 4. a small fit: recover N by maximum likelihood ------------------------
#
# Simulate a Poisson-observed SIR epidemic, then recover log N by maximising the
# Poisson log-likelihood through the renewal scan. A dependency-free Nelder-Mead
# keeps the prototype self-contained; the production path would use the
# package's Turing / Optimization glue.

function _nelder_mead(f, x0; iters = 600, step = 0.4)
    n = length(x0)
    simplex = [copy(x0) for _ in 0:n]
    for i in 1:n
        simplex[i + 1][i] += step
    end
    fv = [f(x) for x in simplex]
    for _ in 1:iters
        ord = sortperm(fv)
        simplex, fv = simplex[ord], fv[ord]
        centroid = sum(simplex[1:n]) ./ n
        worst = simplex[end]
        refl = centroid .+ (centroid .- worst)
        fr = f(refl)
        if fr < fv[1]
            expd = centroid .+ 2 .* (centroid .- worst)
            fe = f(expd)
            simplex[end], fv[end] = fe < fr ? (expd, fe) : (refl, fr)
        elseif fr < fv[end - 1]
            simplex[end], fv[end] = refl, fr
        else
            contr = centroid .+ 0.5 .* (worst .- centroid)
            fc = f(contr)
            simplex[end], fv[end] = fc < fv[end] ? (contr, fc) :
                                    (simplex[1], fv[1])
        end
    end
    return simplex[argmin(fv)]
end

true_N = 5.0e4
rng = MersenneTwister(20260626)
expected = CD.renewal(true_Rt, g, I0;
    modulator = CD.susceptibility_depletion(true_N))
counts = rand.(rng, Poisson.(expected .+ 1.0e-6))

function nll(logN)
    inf = CD.renewal(true_Rt, g, I0;
        modulator = CD.susceptibility_depletion(exp(logN[1])))
    return -sum(c * log(i + 1.0e-6) - (i + 1.0e-6)
    for (c, i) in zip(counts, inf))
end

fit = _nelder_mead(nll, [log(2.0e4)])
recovered_N = exp(fit[1])
println("4. fit recovers N: truth ", round(true_N; sigdigits = 3),
    " vs estimate ", round(recovered_N; sigdigits = 3),
    " (within 25%): ", abs(recovered_N - true_N) / true_N < 0.25)

# ============================================================================
# Deferred PRODUCTION scope (decisions for the maintainer)
# ============================================================================
#
#  - Turing glue: a `composed_*_model`-style entry that fits Rt + modulator
#    params, reading priors back onto the renewal object (this prototype hand-
#    rolls Nelder-Mead). Ties into the priors front-door (#636).
#  - First-class modulator leaves: ship transmissibility / immunity-waning
#    modulators beside `susceptibility_depletion`, each a small struct + init.
#  - The observation bridge: the rt-renewal tutorial pairs the renewal with
#    `convolve_distributions(stack, infections)`; a documented end-to-end
#    `renewal |> convolve` pipeline (the #763 worked example).
#  - Generation-interval source: accept a `DelayPMF` / leaf directly and own the
#    lag-0 convention so callers do not hand-build the PMF vector.
#  - The convolve-LOOP dual (#759): the population forward view of the recurrent
#    multi-state model (#545); this step is the scalar-incidence special case.
