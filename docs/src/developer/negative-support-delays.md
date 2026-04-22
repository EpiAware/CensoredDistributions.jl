# [Negative-support delays: scoping](@id negative-support-delays)

Scope for supporting delay distributions with support below zero (e.g. `Normal`, `Logistic`, `Cauchy`) in `CensoredDistributions.jl`, mirroring the work on the R side in [epinowcast/primarycensored#297](https://github.com/epinowcast/primarycensored/pull/297) (parent report: [primarycensored#267](https://github.com/epinowcast/primarycensored/issues/267)).

## Background

`primarycensored` (R) previously assumed delays were non-negative.
The primary-censored CDF short-circuited to `0` for `q <= 0` and the `pcd_*` family defaulted to a lower truncation of `L = 0`, so signed-support delays (`pnorm`, `plogis`, `pcauchy`) were unusable as delay families and produced incoherent subprobabilities.
The R PR removed the short-circuit, flipped the default truncation to `L = -Inf` (no truncation), and switched quantile inversion to a bracketing root-finder that extends the search on the left when needed.

## Current state in Julia

The Julia implementation is already largely compatible with signed-support delays because of design choices that differ from the original R API.
A file-by-file audit follows.

### `double_interval_censored` — default truncation

`double_interval_censored` already defaults `lower = nothing` (see `src/censoring/double_interval_censored.jl`), which is equivalent to the R PR's new `L = -Inf` default.
No breaking change to the default is required on the Julia side; callers who want `[0, D]` truncation already have to pass `lower = 0` explicitly.

### Numerical primary-censored CDF

`primarycensored_cdf(dist, primary_event, x, ::NumericSolver)` in `src/censoring/primarycensored_cdf.jl` already respects `minimum(dist)`:

- The lower edge-case check is `x <= minimum(dist)`, not `x <= 0`.
- The integration lower bound is `max(x - maximum(primary_event), minimum(dist))`, so it correctly extends to `-Inf` when the delay distribution has unbounded-below support.

This path is therefore signed-support correct as written.

### Analytical CDFs (Gamma, LogNormal, Weibull)

The three analytical specialisations only dispatch on `Gamma`, `LogNormal`, and `Weibull` delays with a `Uniform` primary event.
These families all have `minimum(dist) = 0`, so the existing `t <= 0` short-circuits are correct for them.
No changes needed; signed-support delays fall through to the numerical path, matching the R design.

### `IntervalCensored`

`IntervalCensored` computes masses via `cdf(get_dist(d), boundary)` and uses `minimum(get_dist(d))` / `maximum(get_dist(d))` for boundary handling.
It is signed-support safe already.

### Sampling

`Base.rand(rng, d::PrimaryCensored) = rand(rng, get_dist(d)) + rand(rng, d.primary_event)` draws directly from the delay and primary distributions and sums, so no rejection or clamping step could silently discard negatives.
The truncated-primary-censored sampler uses `_in_closed_interval` and will correctly accept samples in a signed-support window.

### Quantile inversion

`quantile(d::PrimaryCensored, p)` in `src/censoring/PrimaryCensored.jl` currently uses Nelder-Mead on `(cdf(d, q) - p)^2` via `_quantile_optimization` in `src/utils/quantile_optimization.jl`.
Two concerns for signed support:

1. The objective applies a large penalty with `1e10 + (q_val - minimum(d))^2` when `q` is outside the support; for distributions with `minimum(d) = -Inf` this term is `Inf` or `NaN`, which can stall optimisation.
2. The default initial guess `underlying_quantile + primary_mean` is still sensible for signed support, but Nelder-Mead without bracketing is less robust than the `Roots.find_zero` / bracket-extending approach the R PR adopted.

Recommended follow-up (non-blocking for core correctness): replace the penalty with a finite value when `minimum(d)` is `-Inf`, or switch to a bracketing root-finder (`Roots.find_zero` with `A42`/`Bisection` and an expanding bracket) for `PrimaryCensored` quantiles.

### PDF via numerical differentiation

`logpdf(d::PrimaryCensored, x)` uses a central difference on `logcdf` with a `h = 1e-8` step and `insupport` guard.
This is signed-support safe — `insupport` delegates to the underlying delay.

## Representing "no left truncation"

Julia already uses `lower::Union{Real, Nothing} = nothing` in `double_interval_censored` and `Distributions.truncated` accepts `lower = -Inf` natively.
No sentinel change is needed.
Document that `lower = nothing` (the default) means "no left truncation" and applies equally to signed- and positive-support delays.

## Checklist vs the parent issue

- [x] Audit non-negativity assumptions — only present inside analytical branches that dispatch on positive-support families; numerical path is signed-support safe.
- [x] Decide representation of "no left truncation" — already `lower = nothing` / `-Inf`; nothing to change.
- [x] Check analytical solutions — Gamma/LogNormal/Weibull-only dispatch means their `t <= 0` guards stay.
- [x] Check random sampling — direct `rand(delay) + rand(primary)`, no clamping.
- [ ] Check quantile inversion — Nelder-Mead with support penalty works in principle but is less robust than a bracketing root-finder for signed support; consider switching.
- [x] Default flip as breaking change — not needed; Julia default is already `lower = nothing`.
- [ ] Tests — add signed-support coverage (Normal, Logistic, Cauchy) for CDF monotonicity, tail limits (`->0` at `-Inf`, `->1` at `+Inf`), PMF-as-CDF-difference consistency, sampling within `[L, D)`, quantile round-trips.

## Proposed work

1. Add a test suite `test/censoring/negative_support.jl` covering signed-support delays with `Uniform` and non-`Uniform` primary events, exercising `primary_censored`, `interval_censored`, and `double_interval_censored`.
2. If (1) surfaces quantile failures, harden `_quantile_optimization` to handle `minimum(d) = -Inf` (guard the penalty term) or add a bracketing-root-finder path for `PrimaryCensored`.
3. Add a short user-facing note to the Getting Started docs that signed-support delays are supported and route through the numerical CDF.

Items (2) and (3) are scoped as follow-up PRs from this issue; this document exists to record the audit so the implementation PRs stay focused.
