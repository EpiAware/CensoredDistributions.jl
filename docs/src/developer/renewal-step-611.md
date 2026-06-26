# Composable renewal step (#611)

Status: DESIGN + PROTOTYPE for review.
The renewal scan, the modulators and `combine_modulators` are implemented in
`src/utils/renewal.jl` (public, unexported) with the full-backend AD harness and
value-level tests; the prototype in `prototypes/renewal_step_611.jl` runs the
worked story end to end.
The package-boundary scope decision (below) is open for the maintainer.

## The gap

The force-of-infection convolution `Σ_s g_s I_{t-s}` is already a single call:
`convolve_distributions(stack, series)` pushes a FIXED series through a delay
stack (the EpiNow2-style observation layer).
The renewal RECURRENCE is not, because the output feeds back as input:

```math
I[t] = R_t \, m(t) \, \sum_s g_s \, I[t-s].
```

`convolve_distributions` convolves a known series, so the recurrence cannot be a
single convolution call.
The rt-renewal tutorial hand-rolls the loop.
This is the small composable abstraction that lets it stop.

## Design

A forward scan, `renewal(Rt, gi, I0; modulator, seed_days)`, that reuses the
same causal generation-weighted sum the convolution kernel uses, applied ONE
output step at a time so the output can feed back:

- `_renewal_force(I, t, g)` is the per-step `Σ_s g_s I[t-s]`, the scalar
  analogue of `_causal_convolve`'s inner loop;
- the first `seed_days` steps are fixed to the seed `I0` to give the recurrence
  history (matching the tutorial loop);
- `gi` is either a PMF vector (`g[s]` the lag-`s` mass, lag-1-indexed, as
  `pdf(gen_dist, 1:gi_max)` returns) or a `DelayPMF` (lag-0-indexed masses, lag
  0 dropped), so the renewal consumes the generation interval the rest of the
  stack already builds.

This is NOT a parallel system: it reuses the existing discretisation
(`interval_censored` / `discretise_pmf`) and the same causal arithmetic, one
step at a time.

### The modulator interface

A modulator is the force-of-infection multiplier, a callable

```julia
m(state, t, force) -> (factor, state')
```

returning the multiplicative factor at time `t` and a carry-state for the next
step (`_modulator_init(m)` gives the initial carry).
This is the single extension point.

- `NoModulation()` — the identity (factor `1`, no carry); the default, giving
  the bare renewal.
- `susceptibility_depletion(N; S0 = N)` — the SIR-style depleting factor
  `S[t-1]/N`, carrying the susceptible pool and decrementing it by each step's
  infections, so `S[t] = S[t-1] - I[t]`.
- `combine_modulators(a, b)` — stacks two modulators: factors multiply,
  carry-states thread, and the result is itself a modulator so composition
  nests.
  This carries the `compose` idiom from delays to renewal steps:
  transmissibility / susceptibility / immunity terms STACK rather than living in
  one monolithic recurrence.

A new modulator is a small struct, a call method and a `_modulator_init`
method — the same low-surface extension contract as a new leaf.

### AD-friendliness

The scan is linear in the infection history and the modulator factors, and the
generation interval enters as a plain PMF (the AD-safe `interval_censored`
discretisation), so gradients flow w.r.t. `Rt`, the modulator parameters (`N`,
…) and — where the gi is built inside the differentiated function — the
generation-interval parameters.
The accumulator element type is promoted across `Rt`, the PMF, the seed and the
modulator state so `Dual` / tracked numbers propagate.

## What the prototype proves

`prototypes/renewal_step_611.jl` checks, end to end:

1. equivalence — the bare scan reproduces the rt-renewal tutorial's hand-rolled
   loop BIT FOR BIT (and the `DelayPMF` form agrees);
2. susceptibility — `susceptibility_depletion` matches an independent
   hand-written SIR loop, and collapses to the bare renewal as `N -> ∞`;
3. composition — `combine_modulators` stacks a susceptibility term with a
   constant transmissibility factor;
4. fit — a Poisson-observed SIR epidemic recovers `N` by maximum likelihood
   through the whole scan.

The AD-scored path (a Poisson log-likelihood through the scan and the
susceptibility modulator) runs through the full-backend ADFixtures harness:
ForwardDiff, ReverseDiff and Mooncake reverse match the reference; both Enzyme
modes are registered broken (the closure carries an active `Constant`-tuple
field Enzyme needs `function_annotation = Duplicated` to trace, the same class
as the existing vectorised scenarios).

## Relationship to the other renewal issues

- `#759` (convolve-loop): the population forward view of the recurrent
  multi-state model — convolving populations around a state cycle. This renewal
  step is the scalar-incidence special case; the convolve-loop is the
  multi-state generalisation.
- `#545` (recurrent multi-state): the per-individual dual. The renewal step is
  the population-renewal side of the individual-path ↔ population correspondence
  the `#763` worked example threads together, with susceptibility as the
  coupling.

## Open: package-boundary scope

This is renewal-PROCESS / transmission modelling, a layer above censored
distributions.
The issue lists three options:

1. minimal helper here — only the bare scan, leaving composition downstream;
2. composable renewal here — the modulator abstraction in this package
   (what this prototype builds), extending `compose` from delays to renewal
   steps;
3. downstream — keep renewal out of CensoredDistributions, exposing the
   convolution + composer primitives and building the process on top elsewhere.

The prototype demonstrates option 2 is small and reuses the existing kernel.
The decision the maintainer needs to settle is whether the composable renewal
step lives here (option 2) or downstream (option 3); the implementation is kept
public-but-unexported so it can be promoted or moved without a breaking change.

## Deferred production scope

- Turing glue: a `composed_*_model`-style fit entry reading priors back onto the
  renewal object (the prototype hand-rolls Nelder-Mead); ties into the priors
  front-door (#636).
- First-class transmissibility / immunity-waning modulators beside
  `susceptibility_depletion`.
- A documented `renewal |> convolve_distributions` observation pipeline (the
  #763 worked example).
- Accept a `DelayPMF` / leaf generation interval directly and own the lag-0
  convention so callers do not hand-build the PMF vector.
