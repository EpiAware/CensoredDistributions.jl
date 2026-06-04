# Hanta (Andes virus) latent delay model — Turing integration scaffold.
#
# Turing is an optional dependency, so these test items are tagged
# `:turing` (matching `test/integration/Turing.jl`) and only run in the
# Turing-enabled test environment, not the general test suite.
#
# This file is a LIVING INTEGRATION TARGET. It reproduces the latent-time
# data-augmentation part of the Epuyén ANDV outbreak model from
# epiforecasts/andv-linelist-analysis (internal package
# `TransmissionLinelist`, `joint_model`), with the real-time / R(t)
# extension DELIBERATELY REMOVED. Only the incubation period and
# per-pair transmission-timing pieces are kept.
#
# Why it exists
# -------------
# The upstream model does not use CensoredDistributions.jl. The delay,
# convolution, and latent-time machinery here is therefore hand-rolled.
# Each hand-rolled piece is isolated into a small function so it can be
# swapped for a package feature as that feature lands, with an equivalence
# assertion against this baseline. The swap-in points are marked with
# `TODO(#NNN)` comments referencing:
#   - #295  Convolved distribution (sum of independent delays): the
#           generation interval GI = incubation + transmission-timing is a
#           convolution that `hanta_generation_interval` currently builds
#           by hand.
#   - #298  Tree-structured censored distribution: the per-pair
#           source -> secondary conjunction (and the zoonotic-index root)
#           is the disjunctive/conjunctive tree that
#           `latent_times_model` currently expresses with an explicit loop
#           and a `-Inf` reject.
#   - #299  Latent mode for primary-/double-interval-censored
#           distributions: the data augmentation over the continuous
#           latent infection and onset times (`T_inf`, `T_onset`) within
#           recorded windows is exactly what latent mode is meant to
#           provide; `hanta_window_prior` currently does this with raw
#           `Uniform` priors.
#
# Cross-representation validation of these swaps lives in #301; the
# walkthrough docs that build on this scaffold are #302. Part of #253.
#
# Data licence
# ------------
# The upstream line list (data/linelist.csv) is MIT-licensed and could be
# bundled. It is NOT copied here: it needs CSV/date parsing not present in
# the test environment, and it carries R(t) / offspring (`Z`) columns that
# are out of scope for the latent-only model. Instead we SIMULATE a tiny
# line list from the generative latent model with known parameters, which
# also gives ground truth for parameter recovery. A future PR may add the
# real data behind the equivalence checks once #299 lands.

@testitem "hanta latent: simulate -> recover (no Rt)" tags=[:turing] begin
    using Turing
    using DynamicPPL
    using Distributions
    using Random
    using FlexiChains
    using FlexiChains: Prefixed, niters

    # --- Isolated hand-rolled pieces (swap points) ----------------------

    # TODO(#299): latent mode. A case's continuous infection/onset times
    # are augmented within their recorded `[lo, hi)` windows. Replace these
    # raw Uniform window priors with a latent-mode censored distribution.
    hanta_window_prior(lo, hi) = Uniform(lo, hi)

    # TODO(#295): Convolved. The incubation period (T_onset - T_inf) is a
    # single delay here; the GI = incubation + transmission-timing is a
    # convolution assembled post hoc. Replace with `Convolved`.
    hanta_incubation(μ_inc, σ_inc) = LogNormal(μ_inc, σ_inc)

    # TODO(#295)/TODO(#298): the generation interval is the source's
    # incubation plus the transmission-timing offset. Currently derived
    # arithmetically from latents; this is the convolution `Convolved`
    # (#295) wired into the per-pair tree (#298).
    function hanta_generation_interval(T_inf_sec, T_inf_src)
        return T_inf_sec - T_inf_src
    end

    # --- Generative simulation -----------------------------------------
    # A tiny synthetic outbreak: one zoonotic index plus a star of
    # secondaries, then a second generation. Mirrors the upstream
    # structure (index with no human source, sourced cases with exposure
    # windows) but small enough for a short chain.

    # Ground-truth parameters.
    μ_inc_true = 3.0      # log-mean incubation (~20 d)
    σ_inc_true = 0.25     # log-SD incubation
    μ_δ_true = 0.0        # mean transmission timing rel. source onset (d)
    σ_δ_true = 0.6        # SD transmission timing (d)

    Random.seed!(20260508)
    inc_dist = LogNormal(μ_inc_true, σ_inc_true)
    δ_dist = Normal(μ_δ_true, σ_δ_true)

    # source_idx[i] == 0 marks the zoonotic index (no human source).
    # Build a small two-generation tree.
    source_idx = [0, 1, 1, 1, 1, 2, 2, 3]
    N = length(source_idx)
    win = 1.0  # one-day recorded windows, as in the upstream default

    T_inf = zeros(N)
    T_onset = zeros(N)

    # Index case: anchor infection at day 0.
    T_inf[1] = 0.0
    T_onset[1] = T_inf[1] + rand(inc_dist)

    for i in 2:N
        src = source_idx[i]
        # Secondary infected around the source's onset by δ; resample to
        # keep GI > 0 (matches the upstream positivity constraint).
        local t_inf
        while true
            t_inf = T_onset[src] + rand(δ_dist)
            t_inf > T_inf[src] && break
        end
        T_inf[i] = t_inf
        T_onset[i] = T_inf[i] + rand(inc_dist)
    end

    # Encode interval-censored recorded windows around the latent truth.
    onset_lo = floor.(T_onset)
    onset_hi = onset_lo .+ win
    exp_lo = floor.(T_inf)
    exp_hi = exp_lo .+ win

    # --- Latent-times model (Rt removed) -------------------------------
    # This is `joint_model` from the upstream package with the R(t) random
    # walk, the NegativeBinomial offspring term, and the `Zobs` data all
    # stripped out. Only incubation + transmission-timing remain.
    @model function latent_times_model(
            source_idx, onset_lo, onset_hi, exp_lo, exp_hi)
        N = length(source_idx)

        # Population parameters (upstream priors).
        μ_inc ~ Normal(3.0, 0.5)
        σ_inc ~ truncated(Normal(0.0, 0.5); lower = 0)
        μ_δ ~ Normal(0.0, 5.0)
        σ_δ ~ truncated(Normal(0.0, 1.0); lower = 0)

        T = typeof(μ_inc)
        inc_dist = hanta_incubation(μ_inc, σ_inc)

        # TODO(#299): latent onset times via data augmentation.
        T_onset = Vector{T}(undef, N)
        for i in 1:N
            T_onset[i] ~ hanta_window_prior(onset_lo[i], onset_hi[i])
        end

        # TODO(#298): per-pair source tree; TODO(#299) latent infection
        # times; TODO(#295) GI convolution.
        T_inf = Vector{T}(undef, N)
        for i in 1:N
            if source_idx[i] == 0
                # Zoonotic index: free latent infection time pre-onset.
                T_inf[i] ~ hanta_window_prior(
                    onset_lo[i] - 80.0, T_onset[i] - 1e-6)
                inc_i = T_onset[i] - T_inf[i]
                Turing.@addlogprob! logpdf(inc_dist, inc_i)
            else
                src = source_idx[i]
                T_inf[i] ~ hanta_window_prior(exp_lo[i], exp_hi[i])
                gi = hanta_generation_interval(T_inf[i], T_inf[src])
                if gi <= 0
                    # GI > 0 constraint (upstream -Inf reject).
                    Turing.@addlogprob! -Inf
                else
                    inc_i = T_onset[i] - T_inf[i]
                    δ_pair = T_inf[i] - T_onset[src]
                    Turing.@addlogprob! logpdf(inc_dist, inc_i)
                    Turing.@addlogprob! logpdf(Normal(μ_δ, σ_δ), δ_pair)
                end
            end
        end
    end

    model = latent_times_model(
        source_idx, onset_lo, onset_hi, exp_lo, exp_hi)

    # Short chain to keep CI viable. `:slow` would gate a longer run; this
    # baseline keeps it minimal but enough to recover the order of the
    # generating parameters.
    Random.seed!(20260508)
    # NUTS(n_adapts, δ): explicit adaptation steps and target acceptance.
    chain = sample(
        model, NUTS(150, 0.8), 300;
        chain_type = VNChain, progress = false
    )

    @test niters(chain) == 300

    μ_inc_post = chain[@varname(μ_inc)]
    σ_inc_post = chain[@varname(σ_inc)]
    μ_δ_post = chain[@varname(μ_δ)]
    σ_δ_post = chain[@varname(σ_δ)]

    @test all(isfinite.(μ_inc_post))
    @test all(σ_inc_post .> 0)
    @test all(σ_δ_post .> 0)

    # Parameter recovery within a loose tolerance (tiny data + short
    # chain). These check the scaffold recovers the right ballpark, not
    # tight calibration.
    using Statistics: mean
    @test isapprox(mean(μ_inc_post), μ_inc_true; atol = 0.4)
    @test isapprox(mean(μ_δ_post), μ_δ_true; atol = 2.0)
end

@testitem "hanta latent: swap-point functions are isolated" tags=[:turing] begin
    # A guard test asserting the hand-rolled swap points behave as the
    # package features they will be replaced by must behave. When #295 /
    # #298 / #299 land, these become equivalence checks against the
    # package implementations.
    using Distributions

    # Local copies of the swap-point helpers (kept in sync with the model
    # test above). Equivalence assertions against package features go here.
    hanta_window_prior(lo, hi) = Uniform(lo, hi)
    hanta_incubation(μ, σ) = LogNormal(μ, σ)
    hanta_gi(sec, src) = sec - src

    # TODO(#299): window prior is a Uniform over the recorded window.
    wp = hanta_window_prior(2.0, 5.0)
    @test wp isa Uniform
    @test minimum(wp) == 2.0 && maximum(wp) == 5.0

    # TODO(#295): incubation is the delay distribution to convolve.
    inc = hanta_incubation(3.0, 0.25)
    @test inc isa LogNormal
    @test params(inc) == (3.0, 0.25)

    # TODO(#295)/TODO(#298): GI is the (latent) infection-time gap.
    @test hanta_gi(10.0, 4.0) == 6.0
end
