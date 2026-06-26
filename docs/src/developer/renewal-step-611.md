# Composable renewal step (#611)

Status: implemented.
The maintainer chose to build the composable renewal step in
CensoredDistributions (extending the `compose` philosophy from delays to renewal
steps).
The scan, the modulators, the composition, the Turing glue and the observation
bridge live in `src/utils/renewal.jl` and the DynamicPPL extension, with the
full-backend AD harness, value-level tests, a fit test and a tutorial.
This page records the design and the relationships to the other renewal issues.

## The gap

The force-of-infection convolution `Σ_s g_s I_{t-s}` is already a single call:
`convolve_distributions(stack, series)` pushes a fixed series through a delay
stack (the EpiNow2-style observation layer).
The renewal recurrence is not, because the output feeds back as input:

```math
I[t] = R_t \, m(t) \, \sum_s g_s \, I[t-s].
```

`convolve_distributions` convolves a known series, so the recurrence cannot be a
single convolution call.
The rt-renewal tutorial hand-rolls the loop.
This is the small composable abstraction that lets it stop.

## Design

A forward scan, `renewal(Rt, gi, I0; modulator, seed_days)`, that reuses the
same causal generation-weighted sum the convolution kernel uses, applied one
output step at a time so the output can feed back:

- `_renewal_force(I, t, g)` is the per-step `Σ_s g_s I[t-s]`, the scalar
  analogue of `_causal_convolve`'s inner loop;
- the first `seed_days` steps are fixed to the seed `I0` to give the recurrence
  history (matching the tutorial loop);
- `gi` is either a PMF vector (`g[s]` the lag-`s` mass, lag-1-indexed, as
  `pdf(gen_dist, 1:gi_max)` returns) or a `DelayPMF` (lag-0-indexed masses, lag
  0 dropped), so the renewal consumes the generation interval the rest of the
  stack already builds.

It reuses the existing discretisation (`interval_censored` / `discretise_pmf`)
and the same causal arithmetic, one step at a time, rather than a parallel
system.

### The modulator interface

A modulator is the force-of-infection multiplier, a callable

```julia
m(state, t, force) -> (factor, state')
```

returning the multiplicative factor at step `t` and a carry-state for the next
step (`_modulator_init(m)` gives the initial carry).
This is the single extension point; a new modulator is a small struct, a call
method and a `_modulator_init` method.

- `NoModulation()` — the identity (factor `1`, no carry); the default, the bare
  renewal.
- `susceptibility_depletion(N; S0 = N)` — the SIR-style depleting factor
  `S[t-1]/N`, carrying the susceptible pool so `S[t] = S[t-1] - I[t]`.
- `transmissibility(beta)` — a per-step deterministic factor `β[t]` (scalar or
  vector); the seasonal-forcing / intervention channel. Stateless.
- `immunity_waning(N, omega; Z0 = 0)` — an immune pool that grows with
  infections and decays at rate `omega` (SIRS); the factor is the susceptible
  fraction `1 - Z[t-1]/N`. With `omega = 0` it matches permanent depletion.
- `combine_modulators(a, b)` — stacks two modulators: factors multiply,
  carry-states thread, the result is itself a modulator so composition nests.

Because the factors multiply and each modulator keeps its own carry-state, the
combined result does not depend on the pairing order: transmissibility,
susceptibility and immunity terms stack rather than living in one monolithic
recurrence (a test locks the order-invariance).

### The observation bridge

`observe_renewal(infections, delay; events)` pushes the renewal infection series
through an observation delay with the causal renewal convolution (it is
`convolve_distributions(delay, infections)` named for the pipeline). So
susceptibility-modulated incidence flows through to reported cases:
`renewal |> observe_renewal`.

### Turing glue

`renewal_model(gi, I0, Rt_prior; modulator_priors, make_modulator, seed_days)`
is the renewal analogue of `composed_parameters_model`: a DynamicPPL submodel
that samples the Rt path and the modulator parameters from priors, runs the
recurrence and returns the infections, which the user scores against observed
counts. The modulator parameters are sampled by name (prefixed submodels), so a
chain reads `N`, etc.; `make_modulator` maps the sampled `NamedTuple` to a
modulator. It lives in the DynamicPPL extension, keeping the core Turing-free.

### AD-friendliness

The scan is linear in the infection history and the modulator factors, and the
generation interval enters as a plain PMF (the AD-safe `interval_censored`
discretisation), so gradients flow w.r.t. `Rt`, the modulator parameters and —
where the gi is built inside the differentiated function — the
generation-interval parameters.
The accumulator element type is promoted across `Rt`, the PMF, the seed and the
modulator state so `Dual` / tracked numbers propagate.
The renewal scored path is covered in the full-backend ADFixtures harness, not a
ForwardDiff-only test.

## Regression checks

The prototype's equivalence checks are kept as regression tests in
`test/utils/renewal.jl`:

- the bare scan reproduces the rt-renewal tutorial's hand-rolled loop bit for
  bit (and the `DelayPMF` form agrees);
- `susceptibility_depletion` matches an independent hand-written SIR loop and
  collapses to the bare renewal as `N` grows;
- `transmissibility(c)` equals scaling `Rt` by `c`; `immunity_waning(N, 0)`
  equals permanent depletion;
- the modulators stack and the pairing order does not matter;
- `observe_renewal` equals the underlying convolution.

`test/integration/RenewalModel.jl` fits `renewal_model` and recovers `N` and the
`Rt` level; the `docs` renewal tutorial threads the full simulate -> fit story.

## Relationship to the other renewal issues

- `#759` (convolve-loop): the population forward view of the recurrent
  multi-state model — convolving populations around a state cycle. This renewal
  step is the scalar-incidence special case; the convolve-loop is the
  multi-state generalisation. Not built here.
- `#545` (recurrent multi-state): the per-individual dual. The renewal step is
  the population-renewal side of the individual-path ↔ population correspondence
  the `#763` worked example threads together, with susceptibility as the
  coupling. Not built here.
