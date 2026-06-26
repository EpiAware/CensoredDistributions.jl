"""
    ADFixtures

Shared AD gradient scenarios and backend metadata for CensoredDistributions.
Used by `test/ad/runtests.jl`, `benchmark/src/ad_gradients.jl`, and the
docs tutorial `docs/src/getting-started/tutorials/ad-backends.jl`.

The reference gradient is computed with `ForwardDiff`. `PrimaryCensored.logpdf`
internally finite-differences the CDF with a hardcoded step `h=1e-8`, which
defeats adaptive finite-difference references (e.g. `central_fdm(5, 1)`
disagrees with all AD backends by ~10% on Weibull analytical scenarios).
ForwardDiff's Dual-number propagation through that internal FD gives the
exact derivative of the package's own logpdf and matches every other working
backend (ReverseDiff, Mooncake reverse/forward) to ~1e-6.
"""
module ADFixtures

# `__precompile__(false)` skips the precompile cache so the Mooncake
# load chain (specifically `MooncakeAllocCheckExt.__init__` evaluating
# into the already-closed `AllocCheck` module) doesn't break the
# package build on CI. Negligible cost — this module is only loaded by
# the AD test, benchmark, and docs scripts, each of which already pays
# for Mooncake/Enzyme load time elsewhere.
__precompile__(false)

using CensoredDistributions
using Distributions: Distributions, Gamma, LogNormal, Weibull, Uniform, Normal,
                     truncated, pdf, logpdf, logccdf, cdf, mean, var
using ADTypes: ADTypes, AutoForwardDiff, AutoReverseDiff, AutoMooncake,
               AutoMooncakeForward, AutoEnzyme
using DifferentiationInterface: DifferentiationInterface, Constant
import ForwardDiff, ReverseDiff, Mooncake, Enzyme
import DifferentiationInterfaceTest as DIT
import SurvivalDistributions as SD

export scenarios, marginal_scenarios, latent_scenarios,
       backends, working_backends, broken_backends,
       broken_scenario_names, backend_broken_scenarios,
       backend_skip_scenarios

# Vectorised-records log density for the AD fixture: assemble the per-record
# distributions (sharing the segment construction) and sum each record's log
# density at its observed event vector. Equals the per-record loop and the
# product log density used by the Turing entry.
function _vectorised_records_logpdf(d, rows)
    recs = CensoredDistributions.record_distributions(d, rows)
    total = logpdf(recs[1],
        [CensoredDistributions._row_event_vector(d, rows[1])...])
    for i in 2:length(recs)
        total += logpdf(recs[i],
            [CensoredDistributions._row_event_vector(d, rows[i])...])
    end
    return total
end

# Vectorised log density of a nested-Resolve (bdbv) tree where each record's
# Resolve branch probability is a COVARIATE CFR carried in the row's reserved
# `branch_probs`. `ps` (one per record, derived from the differentiated params)
# is injected into the rows so the gradient flows through the per-record CFR.
function _vectorised_branch_probs_logpdf(d, rows, ps)
    full = [merge(rows[i], (branch_probs = ps[i],)) for i in eachindex(rows)]
    recs = CensoredDistributions.record_distributions(d, full)
    total = logpdf(recs[1],
        [CensoredDistributions._row_event_vector(d, rows[1])...])
    for i in 2:length(recs)
        total += logpdf(recs[i],
            [CensoredDistributions._row_event_vector(d, rows[i])...])
    end
    return total
end

# Vectorised log density of a Choose (hanta) top: each record selects its
# alternative by `:kind` and scores its single observed value.
function _vectorised_choose_logpdf(d, rows)
    recs = CensoredDistributions.record_distributions(d, rows)
    total = logpdf(recs[1], [Float64(rows[1].delay)])
    for i in 2:length(recs)
        total += logpdf(recs[i], [Float64(rows[i].delay)])
    end
    return total
end

# Horizon-aware whole-compose truncation log density: score each record's
# event vector at its own per-record horizon and sum. With an OBSERVED-intermediate
# record and an endpoint-observed record this exercises BOTH the factorised
# observed-intermediate numerator AND the conv-to-last-observed right-truncation
# denominator (a single `-logcdf(conv-to-last-observed, window)`), so the gradient
# flows through the denominator's convolution-CDF on the differentiated param type.
function _whole_compose_truncation_logpdf(seq, evs, horizons)
    total = CensoredDistributions.event_logpdf(seq, evs[1]; horizon = horizons[1])
    for i in 2:length(evs)
        total += CensoredDistributions.event_logpdf(
            seq, evs[i]; horizon = horizons[i])
    end
    return total
end

# `contexts` is a tuple of `Constant`-wrapped data (the observations),
# passed positionally to DI's `gradient` and to the differentiated
# function. See `scenarios` for why data travels as a context rather
# than a closure capture.
function _reference(f, θ, contexts)
    DifferentiationInterface.gradient(f, AutoForwardDiff(), θ, contexts...)
end

"""
    working_backends()

AD backends that compute correct gradients on at least one scenario.
"""
function working_backends()
    return [
        (name = "ForwardDiff", backend = AutoForwardDiff()),
        (name = "ReverseDiff (tape)",
            backend = AutoReverseDiff(compile = false)),
        (name = "Mooncake reverse",
            backend = AutoMooncake(config = nothing)),
        (name = "Mooncake forward", backend = AutoMooncakeForward()),
        # `set_runtime_activity` is the only user-facing Enzyme setting
        # needed; no `function_annotation = Duplicated`, because the
        # differentiated functions capture no data (see `scenarios`).
        (name = "Enzyme reverse",
            backend = AutoEnzyme(
                mode = Enzyme.set_runtime_activity(Enzyme.Reverse))),
        (name = "Enzyme forward",
            backend = AutoEnzyme(
                mode = Enzyme.set_runtime_activity(Enzyme.Forward)))
    ]
end

"""
    broken_backends()

AD backends that fail on at least some scenarios. `check_broken` in
`test/ad/setup.jl` runs each through plain
`DifferentiationInterface.gradient` and marks the scenarios that do
work as passing, so a partially-working backend is not forced to be
all-or-nothing. Empty today: every backend in [`backends`](@ref) is full
on all scenarios. Enzyme forward relies on `scenarios` constructing each
distribution as a literal rather than capturing a `Type` (see the comment
there).
"""
function broken_backends()
    return NamedTuple{(:name, :backend)}[]
end

"""
    backends()

Union of [`working_backends`](@ref) and [`broken_backends`](@ref).
"""
backends() = vcat(working_backends(), broken_backends())

"""
    broken_scenario_names()

Scenario names that fail for every backend (universal scenario-level
failures). Returns a `Vector{String}`.
"""
function broken_scenario_names()
    # No scenario fails on every backend. `IntervalCensored Gamma
    # arbitrary` previously did: it routed through stock
    # `Distributions.cdf(Gamma, x)` → `gamma_inc`, which no AD backend
    # covers. It now routes through the `_gamma_cdf` helper, so it works
    # everywhere except Mooncake forward (the shared gap, listed in
    # `backend_broken_scenarios`).
    return String[]
end

"""
    backend_broken_scenarios()

Per-backend scenario names that fail even though the backend works on
other scenarios. Returns a `Dict{String, Set{String}}` keyed on the
backend `name` from [`working_backends`](@ref).

"""
function backend_broken_scenarios()
    # NOTE: the nested-tree heterogeneous-edge family no longer fails
    # WHOLESALE on Enzyme. The recursion built a fresh `Vector{Union{Missing,
    # Float64}}` sub-event view per node (`_subevent_slice`); Enzyme's type
    # analysis could not prove the layout of that non-bits-union `Array`
    # allocation inside the differentiated walk (`EnzymeNoTypeError`). The slice
    # is pure constant-data shuffling (it copies observed event TIMES, which carry
    # no gradient -- only the leaf distribution PARAMS do), so it is now marked
    # `EnzymeRules.inactive` in `CensoredDistributionsEnzymeExt`. With that shield
    # the plain nested tree differentiates on BOTH Enzyme modes;
    # `double_interval_censored(Sequential)` now also differentiates on BOTH
    # modes (#506): collapsing the chain to a CONCRETE component tuple in
    # `wrap.jl`/`_sequential_segment` keeps the `Convolved`'s component types
    # concrete, so Enzyme's activity analysis no longer mixes the differentiable
    # leaf params with the constant quadrature nodes. The Resolve / hazard trees
    # differentiate on Enzyme FORWARD. The residual reverse-only and non-terminal
    # gaps below are SEPARATE, deeper Enzyme limitations (documented per
    # scenario). ForwardDiff / ReverseDiff / Mooncake differentiate every one of
    # these correctly.

    # The nested-Resolve tree and the nested racing-hazard tree
    # recurse through the heterogeneous censored-edge walk plus a one_of /
    # racing branch. With the `_subevent_slice` shield they now differentiate on
    # Enzyme FORWARD (verified against the ForwardDiff reference). Enzyme REVERSE
    # still fails with `EnzymeNoShadowError`: building the reverse shadow for the
    # `MixtureModel` / `Compete` branch struct nested inside the
    # `Parallel{Tuple{Sequential{...}, PrimaryCensored{...}}}` tree hits Enzyme's
    # mixed-activity shadow construction (the upstream struct-shadow gap),
    # which is upstream and not reachable from a value-level rule. Registered
    # broken for Enzyme REVERSE only.
    nested_comp = "Nested Resolve tree conditioned logpdf"
    nested_hazard = "Nested racing-hazard tree conditioned logpdf"
    # The external censoring wrapper over a `Sequential`. Now differentiates
    # on BOTH Enzyme modes (verified against the ForwardDiff reference). The
    # earlier reverse failure came from `observed_distribution(::Sequential)`
    # collapsing the chain through a `Vector{UnivariateDistribution}` (abstract
    # eltype): the `Convolved` then carried an abstractly-typed component tuple,
    # and Enzyme's activity analysis mixed the differentiable leaf params with
    # the constant quadrature nodes (`EnzymeRuntimeActivityError`, formerly seen
    # upstream as the `EnzymeNoShadowError` of #506). Flattening the chain to a
    # CONCRETE component tuple (`Tuple{Gamma, LogNormal}`) in `wrap.jl` fixes it
    # on both modes; no value-level rule needed. Fixes #506.
    # The whole-compose conv-to-last-observed right-truncation denominator builds
    # a freshly allocated `Convolved` observed total and routes the
    # `Vector{Union{Missing, Float64}}` event handling through
    # `_seq_event_logpdf_h`. With the concrete `Convolved` build and the
    # `_subevent_slice` shield it now differentiates on Enzyme REVERSE too,
    # matching the ForwardDiff reference (verified), so it is no longer
    # registered broken for any backend.
    # The non-terminal whole-tree Resolve scores a
    # composer-VALUED one_of outcome's subtree through the nested `_tree_score`,
    # AND carries a differentiated branch probability `θ[7]` whose complement
    # `1 - θ[7]` feeds the racing/one_of weighting. It still fails on BOTH
    # Enzyme modes with `IllegalTypeAnalysisException` -- a deeper upstream Enzyme
    # type-analysis gap on this combined composer-subtree-plus-active-branch-prob
    # path, distinct from the `_subevent_slice` allocation fixed earlier.
    # Registered broken for both Enzyme modes.
    nonterminal_comp = "Non-terminal Resolve whole-tree conditioned logpdf"
    # The vectorised path runs an AD-FREE pre-pass that collects the table rows
    # (`Tables.rows` iteration, vector building, validation `throw`s) before the
    # AD-traced build/evaluate. ForwardDiff and ReverseDiff trace straight through
    # this data-collection (it touches only the constant rows), but the COMPILED
    # backends build a rule for every reachable statement -- including the
    # validation branch and the row-collection loop -- and crash uncatchably on it
    # (the same reachable-branch class as the Parallel shared-origin / nested-tree
    # paths). The vectorised MATH itself is the all-continuous per-edge
    # conditioning the single-record scenario already differentiates on every
    # backend; only the data-collection wrapper trips the compiled backends, so it
    # is registered broken for them. Its gradient correctness is covered by
    # ForwardDiff and ReverseDiff.
    vectorised_seq = "Vectorised Sequential censored observed logpdf"
    # The vectorised bdbv (nested Resolve + per-record covariate CFR) and hanta
    # (Choose top) paths share the SAME AD-free data-collection pre-pass (row
    # iteration, vector building, validation) that the compiled backends crash on
    # for `vectorised_seq`; the bdbv path additionally walks the nested tree. Their
    # MATH matches the per-record loop and is verified on ForwardDiff / ReverseDiff;
    # the compiled backends are registered broken on the same pre-pass grounds.
    vectorised_bdbv = "Vectorised nested Resolve per-record branch_probs logpdf"
    vectorised_select = "Vectorised Choose per-record kind logpdf"
    compiled_broken = Set{String}(
        [vectorised_seq, vectorised_bdbv, vectorised_select])
    # The batched `pdf(::IntervalCensored, ::AbstractVector)` boundary
    # collection is now marked non-differentiable per backend (Mooncake
    # `@zero_derivative`, Enzyme `inactive`, ChainRules `@non_differentiable`),
    # so neither Mooncake nor Enzyme traces the `unique`/sort internals. The
    # boundaries are functions of the constant lags, not the AD parameters, so
    # the zero-tangent rule is exact; all backends now differentiate the path
    # (#699, #701).
    return Dict{String, Set{String}}(
        "ForwardDiff" => Set{String}(),
        "ReverseDiff (tape)" => Set{String}(),
        "Mooncake reverse" => copy(compiled_broken),
        "Mooncake forward" => copy(compiled_broken),
        # Enzyme REVERSE: the Resolve/hazard trees (reverse shadow construction)
        # and the non-terminal Resolve (`IllegalTypeAnalysisException`) remain
        # broken; `double_interval_censored(Sequential)` (#506), the plain nested
        # tree, and the whole-compose conv-to-last-observed truncation are fixed.
        "Enzyme reverse" => union(
            Set{String}([nested_comp, nested_hazard, nonterminal_comp]),
            compiled_broken),
        # Enzyme FORWARD: only the non-terminal Resolve remains broken; the
        # plain nested tree, the Resolve/hazard trees, and
        # `double_interval_censored(Sequential)` are now fixed on forward;
        # reverse stays broken, see above.
        "Enzyme forward" => union(
            Set{String}([nonterminal_comp]),
            compiled_broken)
    )
end

"""
    backend_skip_scenarios()

Per-backend scenario names that must be SKIPPED ENTIRELY (not even run through
`check_broken`). Some scenarios crash a compiled backend UNCATCHABLY (an abort /
`signal 6`) that a `try`/`catch` cannot recover, so they cannot be marked
`@test_broken` by running them; they are dropped from that backend's run and
their gradient correctness is covered by the analytic backends instead. Returns
a `Dict{String, Set{String}}` keyed on the backend `name`.
"""
function backend_skip_scenarios()
    # The vectorised bdbv scenario rebuilds the nested tree per record (the
    # per-record `branch_probs` override) inside the differentiated function;
    # Enzyme (both modes) aborts uncatchably on that reconstruction. Skip it for
    # Enzyme entirely; ForwardDiff / ReverseDiff / Mooncake verify its gradient.
    bdbv = "Vectorised nested Resolve per-record branch_probs logpdf"
    return Dict{String, Set{String}}(
        "Enzyme reverse" => Set{String}([bdbv]),
        "Enzyme forward" => Set{String}([bdbv])
    )
end

"""
    marginal_scenarios(; with_reference::Bool = false)

The marginal-density AD scenarios (PrimaryCensored, IntervalCensored,
DoubleIntervalCensored, Convolved, Difference, the composers, …). Equivalent
to `scenarios(; category = :marginal)`. This is what the marginal AD test
sweep consumes, so the marginal coverage is not conflated with the latent
path.
"""
function marginal_scenarios(; with_reference::Bool = false)
    return scenarios(; with_reference = with_reference, category = :marginal)
end

"""
    latent_scenarios(; with_reference::Bool = false)

The latent / augmented-primary AD scenarios (`Latent*`, `PrimaryConditional`),
whose gradients flow through the augmented latent primaries or the latent
conditional rather than the marginal density. Equivalent to
`scenarios(; category = :latent)`. Consumed by the latent AD test sweep,
alongside the vectorised `latent_*_ad.jl` test items.
"""
function latent_scenarios(; with_reference::Bool = false)
    return scenarios(; with_reference = with_reference, category = :latent)
end

"""
    scenarios(; with_reference::Bool = false, category::Symbol = :all)

Return a `Vector{DIT.Scenario{:gradient, :out}}`. When
`with_reference = true`, each scenario's `res1` is populated with a
ForwardDiff reference gradient (see module docstring for rationale).

`category` selects the scenario group:

  - `:marginal` — the marginal-density scenarios (PrimaryCensored,
    IntervalCensored, DoubleIntervalCensored, Convolved, composers, …).
  - `:latent` — the latent/augmented-primary scenarios (`Latent*`,
    `PrimaryConditional`), whose gradients flow through the augmented
    latent primaries rather than the marginal density.
  - `:all` (default) — both groups, in source order (the latent block sits
    among the marginal pushes, so the order is unchanged from before the
    split, only the set is the union). The same set and order the benchmark
    and docs surfaces consumed before; they group per-scenario by name, so the
    ordering is immaterial to them.

The test sweep ([`test/ad/setup.jl`](@ref)) runs `:marginal` and `:latent`
as separate test items so the marginal AD coverage is not conflated with
the latent path. See [`marginal_scenarios`](@ref) and
[`latent_scenarios`](@ref) for the per-group entry points.
"""
function scenarios(; with_reference::Bool = false, category::Symbol = :all)
    category in (:all, :marginal, :latent) ||
        throw(ArgumentError("category must be :all, :marginal or :latent"))
    obs = [0.5, 1.2, 2.5, 3.8, 5.1]
    obs_int = [0.0, 1.0, 2.0, 3.0, 4.0]
    boundaries = [0.0, 1.5, 3.0, 5.0, 10.0]
    obs_int_gamma = [0.5, 2.0, 4.0, 7.0]
    obs_double = [1.0, 2.0, 3.0, 4.0, 5.0]

    out = DIT.Scenario{:gradient, :out}[]
    skip_ref = Set(broken_scenario_names())

    function _push!(name, f, θ₀, contexts; cat::Symbol = :marginal)
        # Skip scenarios outside the requested category. `:all` keeps both
        # the marginal and the latent groups (the default the benchmark and
        # docs surfaces consume); `:marginal`/`:latent` keep only their own.
        (category == :all || category == cat) || return nothing
        # Globally-broken scenarios may break the reference backend
        # itself. Construct them without res1 so the test
        # runner can still mark them broken without erroring here.
        res1 = (with_reference && !(name in skip_ref)) ?
               _reference(f, θ₀, contexts) : nothing
        # Prepare at the real parameter point with the real data
        # contexts. DIT's defaults `zero(x)` and `zero_contexts` would
        # build e.g. `Gamma(0, 0)` and trip the `α > 0` domain assertion.
        prep_args = (; x = θ₀, contexts = contexts)
        push!(out,
            res1 === nothing ?
            DIT.Scenario{:gradient, :out}(
                f, θ₀, contexts...; prep_args = prep_args, name = name) :
            DIT.Scenario{:gradient, :out}(
                f, θ₀, contexts...;
                res1 = res1, prep_args = prep_args, name = name))
    end

    # Observation data is passed as a `Constant` DI context, not captured
    # in a closure, so the differentiated function holds no active fields.
    # Enzyme then needs no `function_annotation = Duplicated`, the call is
    # faster, and it is more portable across backends. Each function
    # references only its `θ` argument, the passed-in data, and
    # module-level constructors.
    #
    # Delay distributions are still written as literals rather than a
    # captured `ctor::Type`. Capturing a distribution `Type` in a function
    # that also makes a keyword call (`method = ...`) trips an upstream
    # Enzyme forward-mode "mixed activity for jl_new_struct" limitation,
    # because the keyword-call lowering builds a struct mixing the active
    # `Type` and `Vector` fields with the inactive solver-method argument.
    # Literal constructors avoid the captured-`Type` field, so Enzyme
    # forward differentiates every scenario; the analytical/numerical split
    # and math are unchanged.
    #
    # Numeric scenarios use the type-stable `method = NumericSolver()`
    # route: the deprecated `force_numeric` flag returns a `Union`-typed
    # solver here, which breaks Enzyme forward. `NumericSolver` is
    # qualified because these fixtures are also staged against the `main`
    # baseline during benchmarking; `benchmark/src/ad_gradients.jl` guards
    # scenario construction so that baseline (which lacks the `method`
    # keyword) skips the AD suite instead of aborting.
    _push!("PrimaryCensored Gamma+Uniform analytical",
        (θ,
            obs) -> sum(
            x -> logpdf(
                primary_censored(Gamma(θ[1], θ[2]), Uniform(0.0, 1.0)), x),
            obs),
        [2.0, 1.5], (Constant(obs),))
    _push!("PrimaryCensored Gamma+Uniform numerical",
        (θ,
            obs) -> sum(
            x -> logpdf(
                primary_censored(Gamma(θ[1], θ[2]), Uniform(0.0, 1.0);
                    method = CensoredDistributions.NumericSolver()), x),
            obs),
        [2.0, 1.5], (Constant(obs),))
    _push!("PrimaryCensored LogNormal+Uniform analytical",
        (θ,
            obs) -> sum(
            x -> logpdf(
                primary_censored(LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0)),
                x),
            obs),
        [1.0, 0.75], (Constant(obs),))
    _push!("PrimaryCensored LogNormal+Uniform numerical",
        (θ,
            obs) -> sum(
            x -> logpdf(
                primary_censored(LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0);
                    method = CensoredDistributions.NumericSolver()), x),
            obs),
        [1.0, 0.75], (Constant(obs),))
    _push!("PrimaryCensored Weibull+Uniform analytical",
        (θ,
            obs) -> sum(
            x -> logpdf(
                primary_censored(Weibull(θ[1], θ[2]), Uniform(0.0, 1.0)), x),
            obs),
        [2.0, 1.5], (Constant(obs),))
    _push!("PrimaryCensored Weibull+Uniform numerical",
        (θ,
            obs) -> sum(
            x -> logpdf(
                primary_censored(Weibull(θ[1], θ[2]), Uniform(0.0, 1.0);
                    method = CensoredDistributions.NumericSolver()), x),
            obs),
        [2.0, 1.5], (Constant(obs),))

    # === LATENT / augmented-primary scenarios (category = :latent) ===
    # These are scored on the augmented latent representation, NOT the marginal
    # density: their gradients flow through the augmented primaries / the
    # conditional `logpdf(delay, observed - primary)`. They are kept in their own
    # category so the marginal AD sweep is purely marginal. See
    # `latent_scenarios()` and the `latent_*_ad.jl` test items, which exercise the
    # vectorised `latent_observed_logpdf` path this single-record group complements.

    # Latent representation. Its logpdf is the primary prior plus the conditional
    # `logpdf(delay, observed - primary)`, so gradients flow through the delay
    # distribution's own logpdf. Event-time pairs are concrete [primary,
    # observed] vectors passed via a Constant context. Delay parameters varied.
    latent_obs = [[0.3, 1.2], [0.5, 2.6], [0.2, 3.8], [0.7, 5.1]]
    _push!("Latent PrimaryCensored LogNormal+Uniform",
        (θ,
            pys) -> sum(
            py -> logpdf(
                latent(primary_censored(LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0))),
                py),
            pys),
        [1.0, 0.75], (Constant(latent_obs),); cat = :latent)
    _push!("Latent PrimaryCensored Gamma+Uniform",
        (θ,
            pys) -> sum(
            py -> logpdf(
                latent(primary_censored(Gamma(θ[1], θ[2]), Uniform(0.0, 1.0))),
                py),
            pys),
        [2.0, 1.5], (Constant(latent_obs),); cat = :latent)

    # Latent gradient with respect to the sampled primary times themselves: the
    # varied vector IS the per-observation primary, delay parameters fixed. This
    # is the gradient a sampler takes over the augmented latent primaries.
    latent_y = [1.2, 2.6, 3.8, 5.1]
    _push!("Latent PrimaryCensored LogNormal+Uniform wrt primary",
        (θ,
            ys) -> sum(
            i -> logpdf(
                latent(primary_censored(LogNormal(1.0, 0.75), Uniform(0.0, 1.0))),
                [θ[i], ys[i]]),
            eachindex(ys)),
        [0.3, 0.5, 0.2, 0.7], (Constant(latent_y),); cat = :latent)
    # Latent Gamma wrt the sampled primary times (the Gamma delay analogue of the
    # LogNormal wrt-primary scenario above), so both delay families have augmented
    # latent-primary gradient coverage.
    _push!("Latent PrimaryCensored Gamma+Uniform wrt primary",
        (θ,
            ys) -> sum(
            i -> logpdf(
                latent(primary_censored(Gamma(2.0, 1.5), Uniform(0.0, 1.0))),
                [θ[i], ys[i]]),
            eachindex(ys)),
        [0.3, 0.5, 0.2, 0.7], (Constant(latent_y),); cat = :latent)

    # PrimaryConditional: the conditional scored via `~` in a model
    # (`y ~ PrimaryConditional(d, p)`). Differentiate with respect to the
    # realised primary times (the sampled latents), delay parameters fixed.
    _push!("PrimaryConditional LogNormal+Uniform wrt primary",
        (θ,
            ys) -> sum(
            i -> logpdf(
                PrimaryConditional(
                    primary_censored(LogNormal(1.0, 0.75), Uniform(0.0, 1.0)),
                    θ[i]),
                ys[i]),
            eachindex(ys)),
        [0.3, 0.5, 0.2, 0.7], (Constant(latent_y),); cat = :latent)
    # PrimaryConditional wrt the DELAY parameters (the complement of the
    # wrt-primary scenario): the realised primaries are fixed data and the
    # gradient flows through the delay distribution's params, the gradient a
    # parameter fit takes on the latent conditional.
    _push!("PrimaryConditional LogNormal+Uniform wrt params",
        (θ,
            ys) -> sum(
            i -> logpdf(
                PrimaryConditional(
                    primary_censored(LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0)),
                    0.3),
                ys[i]),
            eachindex(ys)),
        [1.0, 0.75], (Constant(latent_y),); cat = :latent)
    # === end latent scenarios ===

    # ExponentiallyTilted primary event — no analytical
    # `primarycensored_cdf(::Delay, ::ExponentiallyTilted, ...)` exists,
    # so the scalar `r` parameter of the prior is included in θ (as θ[3])
    # and the whole path runs through numeric integration. Exercises
    # gradient flow through both the delay distribution params and the
    # primary event's tilt parameter. Written as literal constructors
    # rather than a captured `ctor::Type` loop, for the reason above.
    _push!("PrimaryCensored Gamma+ExponentiallyTilted numerical",
        (θ,
            obs) -> sum(
            x -> logpdf(
                primary_censored(Gamma(θ[1], θ[2]),
                    ExponentiallyTilted(0.0, 1.0, θ[3])), x),
            obs),
        [2.0, 1.5, 0.5], (Constant(obs),))
    _push!("PrimaryCensored LogNormal+ExponentiallyTilted numerical",
        (θ,
            obs) -> sum(
            x -> logpdf(
                primary_censored(LogNormal(θ[1], θ[2]),
                    ExponentiallyTilted(0.0, 1.0, θ[3])), x),
            obs),
        [1.0, 0.75, 0.5], (Constant(obs),))
    _push!("PrimaryCensored Weibull+ExponentiallyTilted numerical",
        (θ,
            obs) -> sum(
            x -> logpdf(
                primary_censored(Weibull(θ[1], θ[2]),
                    ExponentiallyTilted(0.0, 1.0, θ[3])), x),
            obs),
        [2.0, 1.5, 0.5], (Constant(obs),))

    _push!("IntervalCensored LogNormal regular",
        (θ, obs) -> sum(
            x -> logpdf(interval_censored(LogNormal(θ[1], θ[2]), 1.0), x),
            obs),
        [1.0, 0.75], (Constant(obs_int),))

    # Two data contexts: the observations and the interval boundaries.
    _push!("IntervalCensored Gamma arbitrary",
        (θ, obs,
            bnd) -> sum(
            x -> logpdf(interval_censored(Gamma(θ[1], θ[2]), bnd), x),
            obs),
        [2.0, 1.5], (Constant(obs_int_gamma), Constant(boundaries)))

    _push!("DoubleIntervalCensored LogNormal",
        (θ,
            obs) -> sum(
            x -> logpdf(
                double_interval_censored(LogNormal(θ[1], θ[2]);
                    primary_event = Uniform(0.0, 1.0),
                    upper = 10.0, interval = 1.0), x),
            obs),
        [1.0, 0.75], (Constant(obs_double),))

    # Batched (vectorised) `pdf`/`logpdf` over an AbstractVector of lags.
    # These hit the `pdf(::IntervalCensored, ::AbstractVector)` path, which
    # evaluates each unique interval-boundary CDF once via the boundary cache
    # rather than the `2·(n+1)` overlapping evals a scalar `pdf`-per-lag loop
    # does. The old `Dict{Any,Any}` boundary cache forced a
    # `DynamicDerivedRule{Dict{Any,Any}}` and a bitcast Mooncake reverse-mode
    # refused to differentiate, so this whole batched path errored on
    # `prepare_gradient_cache` (#699). The scalar scenarios above never
    # exercised it. Passing the lag vector `obs` as the differentiated
    # argument's data (a `Constant` context) keeps the gradient w.r.t. the
    # delay params only. Marginal-density scenarios (`cat = :marginal`).
    obs_batch = collect(0.0:1.0:9.0)
    _push!("IntervalCensored LogNormal regular batched pdf",
        (θ, obs) -> sum(
            pdf(interval_censored(LogNormal(θ[1], θ[2]), 1.0), obs)),
        [1.0, 0.75], (Constant(obs_batch),))
    _push!("IntervalCensored LogNormal regular batched logpdf",
        (θ, obs) -> sum(
            logpdf(interval_censored(LogNormal(θ[1], θ[2]), 1.0), obs)),
        [1.0, 0.75], (Constant(obs_batch),))
    _push!("DoubleIntervalCensored LogNormal batched pdf",
        (θ,
            obs) -> sum(
            pdf(
            double_interval_censored(LogNormal(θ[1], θ[2]);
                primary_event = Uniform(0.0, 1.0),
                upper = 10.0, interval = 1.0), obs)),
        [1.0, 0.75], (Constant(obs_batch),))

    # Weighted scalar logpdf: a count/aggregated-data likelihood term
    # `n * logpdf(dist, x)`. The integer count is an inactive `Constant`
    # context; the gradient flows through the delay parameters only.
    counts = [3.0, 1.0, 4.0, 2.0, 5.0]
    _push!("Weighted LogNormal scalar logpdf",
        (θ, obs,
            cts) -> sum(
            i -> logpdf(weight(LogNormal(θ[1], θ[2]), cts[i]), obs[i]),
            eachindex(obs)),
        [1.0, 0.75], (Constant(obs), Constant(counts)))

    # Product{Weighted} vector logpdf via `weight(dist, counts::Vector)`,
    # which builds a `Product` of `Weighted` and routes the vector
    # observation through `_logpdf_product`. Counts are the (inactive)
    # constructor weights; the gradient is w.r.t. the shared delay params.
    _push!("Product{Weighted} LogNormal vector logpdf",
        (θ, obs,
            cts) -> logpdf(weight(LogNormal(θ[1], θ[2]), cts), obs),
        [1.0, 0.75], (Constant(obs), Constant(counts)))

    # PrimaryCensored with a NON-Uniform primary event: a truncated Normal
    # whose mean is a differentiable parameter (θ[3]). No analytical
    # `primarycensored_cdf(::Delay, ::Truncated, ...)` exists, so this runs
    # the numeric quadrature integrand with a differentiable primary param.
    _push!("PrimaryCensored LogNormal+truncNormal numerical",
        (θ,
            obs) -> sum(
            x -> logpdf(
                primary_censored(LogNormal(θ[1], θ[2]),
                    truncated(Normal(θ[3], 0.3), 0.0, 1.0)), x),
            obs),
        [1.0, 0.75, 0.5], (Constant(obs),))

    # DoubleIntervalCensored with a Gamma delay (only LogNormal covered).
    _push!("DoubleIntervalCensored Gamma",
        (θ,
            obs) -> sum(
            x -> logpdf(
                double_interval_censored(Gamma(θ[1], θ[2]);
                    primary_event = Uniform(0.0, 1.0),
                    upper = 10.0, interval = 1.0), x),
            obs),
        [2.0, 1.5], (Constant(obs_double),))

    # DoubleIntervalCensored with a Weibull delay.
    _push!("DoubleIntervalCensored Weibull",
        (θ,
            obs) -> sum(
            x -> logpdf(
                double_interval_censored(Weibull(θ[1], θ[2]);
                    primary_event = Uniform(0.0, 1.0),
                    upper = 10.0, interval = 1.0), x),
            obs),
        [2.0, 1.5], (Constant(obs_double),))

    # Survival path `logccdf` of PrimaryCensored (LogNormal). Interior
    # observations only: `logccdf` has constant-return branches at the
    # support boundaries (returns 0.0 / -Inf), which would zero the
    # gradient and break the reference comparison.
    obs_ccdf = [1.0, 2.0, 3.0, 4.0]
    _push!("PrimaryCensored LogNormal+Uniform logccdf",
        (θ,
            obs) -> sum(
            x -> logccdf(
                primary_censored(LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0)),
                x),
            obs),
        [1.0, 0.75], (Constant(obs_ccdf),))

    # Standalone ExponentiallyTilted logpdf w.r.t. its rate `r` (θ[1]).
    # Observations must lie inside [min, max] = [0, 1].
    obs_et = [0.2, 0.4, 0.6, 0.8]
    _push!("ExponentiallyTilted logpdf wrt r",
        (θ, obs) -> sum(
            x -> logpdf(ExponentiallyTilted(0.0, 1.0, θ[1]), x), obs),
        [0.5], (Constant(obs_et),))

    # Moment-parameterised (mean, shape) Gamma leaf: differentiate the log
    # density wrt the (mean, shape) pair (θ), so the gradient flows through the
    # derived scale `mean / shape` into the underlying Gamma density. This is
    # the scoring path a mean-coupled upstream prior uses (#710).
    _push!("MomentParams Gamma logpdf wrt mean+shape",
        (θ,
            obs) -> sum(
            x -> logpdf(from_moments(Gamma; mean = θ[1], shape = θ[2]), x),
            obs),
        [5.0, 2.0], (Constant(obs),))

    # Hazard-modified `modify`/`Modified` leaf. Four scenarios cover the
    # analytic log (proportional hazards) and identity (additive hazards)
    # paths, the numeric quadrature path (a logit link on a continuous base)
    # and the discrete per-bin path. The differentiated parameter is the
    # hazard effect; the gradient flows through the closed-form survival
    # expressions on the analytic paths, the Gauss-Legendre quadrature on the
    # numeric path (the same path primary-censoring differentiates) and the
    # per-bin logit reconstruction on the discrete path. Guarded on the verb
    # existing for the AirspeedVelocity baseline build, as above.
    if isdefined(CensoredDistributions, :modify)
        # Analytic proportional hazards: logpdf wrt the log-hazard effect β.
        _push!("Modified log link logpdf wrt effect",
            (θ,
                obs) -> sum(
                x -> logpdf(modify(LogNormal(1.5, 0.5), θ[1]; link = log), x),
                obs),
            [-0.4], (Constant(obs),))

        # Analytic additive hazards: logpdf wrt the additive effect β.
        _push!("Modified identity link logpdf wrt effect",
            (θ,
                obs) -> sum(
                x -> logpdf(
                    modify(LogNormal(1.5, 0.5), θ[1]; link = identity), x),
                obs),
            [0.15], (Constant(obs),))

        # Numeric quadrature path: a logit link on a continuous base routes
        # through the Gauss-Legendre solver; logpdf wrt the effect.
        _push!("Modified numeric logit logpdf wrt effect",
            (θ,
                obs) -> sum(
                x -> logpdf(modify(Gamma(2.0, 1.5), θ[1]; link = :logit), x),
                obs),
            [0.3], (Constant(obs),))

        # Discrete per-bin path: logpdf wrt the per-bin effect vector on a
        # daily interval-censored base. The final-bin effect carries a zero
        # gradient (its hazard is pinned to one).
        obs_disc = [0.0, 1.0, 2.0, 3.0, 4.0]
        _push!("Modified discrete logit logpdf wrt effects",
            (θ,
                obs) -> sum(
                x -> logpdf(
                    modify(interval_censored(LogNormal(1.5, 0.5), 1.0), θ;
                        link = :logit), x),
                obs),
            fill(0.2, 6), (Constant(obs_disc),))
    end

    # Affine transform. The change-of-variables logpdf is
    # `logpdf(inner, (y - shift) / scale) - log(scale)`, so the gradient flows
    # through the inner delay parameters (θ[1], θ[2]) AND the affine scale (θ[3])
    # and shift (θ[4]). Guarded on `affine` existing for the AirspeedVelocity
    # baseline build, as with the other PR-tree scenarios above.
    if isdefined(CensoredDistributions, :affine)
        obs_aff = [2.0, 3.5, 5.0, 7.0]
        _push!("Affine LogNormal scale+shift logpdf",
            (θ,
                obs) -> sum(
                x -> logpdf(
                    CensoredDistributions.affine(
                        LogNormal(θ[1], θ[2]); scale = θ[3], shift = θ[4]),
                    x),
                obs),
            [1.0, 0.5, 2.0, 1.0], (Constant(obs_aff),))
    end

    # IntervalCensored with regular intervals for Gamma and Weibull (only
    # LogNormal covered).
    _push!("IntervalCensored Gamma regular",
        (θ, obs) -> sum(
            x -> logpdf(interval_censored(Gamma(θ[1], θ[2]), 1.0), x),
            obs),
        [2.0, 1.5], (Constant(obs_int),))
    _push!("IntervalCensored Weibull regular",
        (θ, obs) -> sum(
            x -> logpdf(interval_censored(Weibull(θ[1], θ[2]), 1.0), x),
            obs),
        [2.0, 1.5], (Constant(obs_int),))

    # SurvivalDistributions.jl leaf. A GeneralizedGamma delay family
    # scored through its own `logpdf` — the gradient a sampler takes when fitting
    # one of these families. GeneralizedGamma's `logpdf` differentiates on every
    # backend (ForwardDiff / ReverseDiff / Mooncake reverse+forward / Enzyme
    # reverse+forward), so it is a full (non-broken) scenario. Three params
    # (shape σ, scale, power). Guarded on the AirspeedVelocity baseline: the
    # fixtures module is loaded when benchmarking the PR against `main`, and `SD`
    # is a fixtures dep there too, so the literal constructor is safe; the
    # scenario verifies leaf gradients on every backend.
    _push!("SurvivalDistributions GeneralizedGamma logpdf",
        (θ, obs) -> sum(
            x -> logpdf(SD.GeneralizedGamma(θ[1], θ[2], θ[3]), x), obs),
        [1.0, 1.5, 2.0], (Constant(obs),))

    # CENSORED GeneralizedGamma paths. The censoring integrands
    # query the leaf CDF/survival via `_cdf_ad_safe` / `_logccdf_ad_safe`. For a
    # GeneralizedGamma those route through the inner `Gamma(nu/gamma,
    # sigma^gamma)` at the transformed point `t^gamma` and into the package's
    # AD-safe `_gamma_cdf` helper (the `CensoredDistributionsSurvivalDistributions`
    # extension), instead of the stock `logccdf(::Gamma)` → `StatsFuns._gammalogccdf`
    # path, which has no Dual/Tracked/Mooncake method. Both interval- and
    # primary-censored variants differentiate on every backend (ForwardDiff /
    # ReverseDiff / Mooncake reverse+forward / Enzyme reverse+forward) and match
    # the ForwardDiff reference. Same three params as the leaf scenario.
    _push!("IntervalCensored GeneralizedGamma regular",
        (θ,
            obs) -> sum(
            x -> logpdf(
                interval_censored(
                    SD.GeneralizedGamma(θ[1], θ[2], θ[3]), 1.0),
                x),
            obs),
        [1.0, 1.5, 2.0], (Constant(obs_int),))
    _push!("PrimaryCensored GeneralizedGamma+Uniform numerical",
        (θ,
            obs) -> sum(
            x -> logpdf(
                primary_censored(
                    SD.GeneralizedGamma(θ[1], θ[2], θ[3]),
                    Uniform(0.0, 1.0)),
                x),
            obs),
        [1.0, 1.5, 2.0], (Constant(obs),))

    # Convolved (sum of independent delays). The analytic Normal+Normal
    # pair differentiates through `Distributions.convolve`; the
    # Gamma+LogNormal pair has no analytic convolution and exercises the
    # AD-safe numeric quadrature path (the same fixed-domain Gauss-Legendre
    # construction as PrimaryCensored). Literal constructors keep Enzyme
    # forward working.
    # Guarded on `convolve_distributions` existing: AirspeedVelocity benchmarks
    # the PR against the `main` baseline, building the baseline package
    # while still loading this (PR-tree) fixtures module. Referencing
    # `convolve_distributions` unconditionally would throw `UndefVarError` on the
    # baseline, where it does not yet exist. The guard lets the baseline
    # skip these scenarios and the PR include them.
    if isdefined(CensoredDistributions, :convolve_distributions)
        _push!("Convolved Normal+Normal analytical",
            (θ,
                obs) -> sum(
                x -> logpdf(
                    CensoredDistributions.convolve_distributions(
                        Normal(θ[1], θ[2]), Normal(0.0, 1.0)), x),
                obs),
            [1.0, 2.0], (Constant(obs),))
        _push!("Convolved Gamma+LogNormal numerical",
            (θ,
                obs) -> sum(
                x -> logpdf(
                    CensoredDistributions.convolve_distributions(
                        Gamma(θ[1], θ[2]), LogNormal(0.5, 0.4)), x),
                obs),
            [2.0, 1.0], (Constant(obs),))
        # Gamma as the INTEGRATION (last) component. The numeric
        # quadrature clamps the infinite window with a quantile of the last
        # component; a trailing `Gamma` would route that quantile through
        # `gamma_inc_inv`, which Enzyme cannot differentiate. The
        # `_finite_window` fix computes the window endpoint on AD-stripped
        # (primal) params, so the bound is a non-differentiated constant and
        # every backend — Enzyme included — differentiates the logpdf. The
        # differentiated parameters are on the trailing Gamma so the gradient
        # actually flows through the integration component, not just `rest`.
        _push!("Convolved LogNormal+Gamma numerical",
            (θ,
                obs) -> sum(
                x -> logpdf(
                    CensoredDistributions.convolve_distributions(
                        LogNormal(0.5, 0.4), Gamma(θ[1], θ[2])), x),
                obs),
            [2.0, 1.0], (Constant(obs),))
        # Convolved analytic moments: mean/var are the sums of the
        # component moments, so the gradient flows through each component's
        # closed-form `mean`/`var` w.r.t. its parameters. The `obs` context is
        # unused (the moments take no evaluation point) but keeps the scenario
        # shape uniform. Both `mean` and `var` are summed so the gradient
        # covers each moment path.
        _push!("Convolved Gamma+Normal mean+var moments",
            (θ,
                _obs) -> let d = CensoredDistributions.convolve_distributions(
                    Gamma(θ[1], θ[2]), Normal(θ[3], θ[4]))
                mean(d) + var(d)
            end,
            [2.0, 1.5, -0.5, 0.8], (Constant(obs),))
    end

    # Difference (Z = X - Y), the dual of Convolved. The analytic Normal-Normal
    # pair differentiates through the closed-form difference; the Gamma-LogNormal
    # pairs exercise the AD-safe numeric cross-correlation quadrature (the same
    # fixed-domain Gauss-Legendre construction as Convolved). Two pairs cover
    # gradients through the minuend X parameters and through the subtrahend Y
    # parameters: when Y is the unbounded-above integration factor the upper
    # quadrature window is a quantile of the differentiated component, so the
    # window-clamp must stay off the AD path (the `_window_quantile` zero-adjoint
    # rule) for Mooncake/Enzyme not to trace `gamma_inc_inv`. Literal
    # constructors keep Enzyme forward working. Guarded on `difference`
    # existing for the AirspeedVelocity baseline build, as for Convolved above.
    if isdefined(CensoredDistributions, :difference)
        _push!("Difference Normal-Normal analytical",
            (θ,
                obs) -> sum(
                z -> logpdf(
                    CensoredDistributions.difference(
                        Normal(θ[1], θ[2]), Normal(0.0, 1.0)), z),
                obs),
            [1.0, 2.0], (Constant(obs),))
        # Gradient through the minuend X (the f_X factor inside the integral).
        _push!("Difference Gamma-LogNormal numerical wrt X",
            (θ,
                obs) -> sum(
                z -> logpdf(
                    CensoredDistributions.difference(
                        Gamma(θ[1], θ[2]), LogNormal(0.5, 0.4)), z),
                obs),
            [3.0, 1.0], (Constant(obs),))
        # Gradient through the subtrahend Y (the f_Y integration factor and the
        # window-quantile bound). The differentiated Gamma is unbounded above,
        # so the upper window endpoint routes through `_window_quantile`; the
        # zero-adjoint rule keeps that bound a non-differentiated constant.
        _push!("Difference LogNormal-Gamma numerical wrt Y",
            (θ,
                obs) -> sum(
                z -> logpdf(
                    CensoredDistributions.difference(
                        LogNormal(0.5, 0.4), Gamma(θ[1], θ[2])), z),
                obs),
            [3.0, 1.0], (Constant(obs),))
        # Difference moments: mean is the difference of the means and var
        # the SUM of the variances, so the gradient flows through each
        # component's closed-form `mean`/`var`. The `obs` context is unused but
        # keeps the scenario shape uniform.
        _push!("Difference Gamma-Normal mean+var moments",
            (θ,
                _obs) -> let d = CensoredDistributions.difference(
                    Gamma(θ[1], θ[2]), Normal(θ[3], θ[4]))
                mean(d) + var(d)
            end,
            [3.0, 1.5, 2.0, 0.5], (Constant(obs),))
    end

    # Completeness thinning helpers. `thin_by_completeness(R, delay,
    # window) = R * cdf(delay, window)`, so the gradient flows through `R` and
    # the delay-distribution parameters via the CDF. The Convolved-chain form
    # routes the CDF through the AD-safe numeric convolution quadrature.
    # Guarded for the AirspeedVelocity baseline build, as above.
    if isdefined(CensoredDistributions, :thin_by_completeness)
        _push!("thin_by_completeness LogNormal delay",
            (θ,
                _obs) -> CensoredDistributions.thin_by_completeness(
                θ[1], LogNormal(θ[2], θ[3]), 7.0),
            [1.5, 1.5, 0.5], (Constant(obs),))
        if isdefined(CensoredDistributions, :convolve_distributions)
            _push!("thin_by_completeness Convolved chain",
                (θ,
                    _obs) -> CensoredDistributions.thin_by_completeness(
                    θ[1],
                    CensoredDistributions.convolve_distributions(
                        Gamma(θ[2], θ[3]), LogNormal(0.5, 0.4)),
                    14.0),
                [1.5, 2.0, 1.0], (Constant(obs),))
        end
    end

    # Right-truncation. The index single-delay term right-truncates a
    # LogNormal to the remaining window. This is the NaN-gradient regression:
    # an upper-only `truncated(dist; upper = window)` never differentiates
    # `logcdf(LogNormal, 0) = -Inf`, so the gradient stays finite. The chain
    # term right-truncates a Convolved (unobserved intermediate event), so the
    # denominator is the convolution CDF. Both guarded on the helper existing
    # for the AirspeedVelocity baseline (see the Convolved note above).
    if isdefined(CensoredDistributions, :truncate_to_horizon)
        _push!("Truncated LogNormal single-delay right-truncation",
            (θ,
                obs) -> sum(
                x -> logpdf(
                    truncate_to_horizon(LogNormal(θ[1], θ[2]), 6.0), x),
                obs),
            [1.0, 0.75], (Constant(obs),))
        # Component order matches the working "Convolved Gamma+LogNormal
        # numerical" scenario: the numeric convolution CDF replaces the
        # infinite upper endpoint with a quantile of the LAST component, so a
        # trailing LogNormal keeps Enzyme off `gamma_inc_inv_qsmall` (a known
        # Enzyme illegal-type-analysis failure on Gamma quantile inversion).
        _push!("Truncated Convolved chain right-truncation",
            (θ,
                obs) -> sum(
                x -> logpdf(
                    truncate_chain(
                        (Gamma(2.0, 1.0), LogNormal(θ[1], θ[2])),
                        (false,), 8.0), x),
                obs),
            [1.0, 0.75], (Constant(obs),))
    end

    # Pluggable integration path. The numeric primary-censored CDF
    # routes its quadrature through the package's default `GaussLegendre`
    # solver passed explicitly via the `solver` keyword. This is the cost
    # the integration refactor touches, so benchmarking it per backend
    # gives a clean before/after signal on the integration path; the test
    # suite runs it as a gradient-correctness check too. A 128-node rule
    # (twice the default) makes the quadrature cost the dominant term.
    # Guarded on `GaussLegendre` existing: on the `main` baseline the
    # solver type lived in Integrals.jl, not the package, so referencing
    # it unconditionally would throw `UndefVarError` when AirspeedVelocity
    # builds the baseline against this PR-tree fixtures module.
    if isdefined(CensoredDistributions, :GaussLegendre)
        _push!("PrimaryCensored Gamma+truncNormal numerical GaussLegendre solver",
            (θ,
                obs) -> sum(
                x -> logpdf(
                    primary_censored(Gamma(θ[1], θ[2]),
                        truncated(Normal(0.5, 0.3), 0.0, 1.0);
                        solver = CensoredDistributions.GaussLegendre(; n = 128)),
                    x),
                obs),
            [2.0, 1.5], (Constant(obs),))
    end

    # High-dimensional scenarios. Each observation carries its own delay
    # parameter, so the gradient is taken with respect to many inputs.
    # These give the reverse-mode backends (ReverseDiff, Enzyme reverse,
    # Mooncake reverse) a regime where they win: reverse mode costs one
    # pass regardless of the parameter count, while the forward modes
    # (ForwardDiff, Enzyme forward, Mooncake forward) pay per parameter.
    # Both an analytical and a numerical (quadrature) variant are included:
    # the numerical path does far more work per call, which is where the
    # compiled forward backends (Enzyme, Mooncake) can close the gap on
    # ForwardDiff's Dual-number propagation. Literal constructors keep
    # Enzyme forward working.
    n_hd = 32
    obs_hd = collect(range(0.5, 8.0; length = n_hd))
    _push!("PrimaryCensored LogNormal+Uniform analytical $(n_hd)d",
        (θ,
            obs_hd) -> sum(
            i -> logpdf(
                primary_censored(LogNormal(θ[i], 0.5), Uniform(0.0, 1.0)),
                obs_hd[i]),
            eachindex(obs_hd)),
        fill(1.0, n_hd), (Constant(obs_hd),))
    _push!("PrimaryCensored LogNormal+Uniform numerical $(n_hd)d",
        (θ,
            obs_hd) -> sum(
            i -> logpdf(
                primary_censored(LogNormal(θ[i], 0.5), Uniform(0.0, 1.0);
                    method = CensoredDistributions.NumericSolver()),
                obs_hd[i]),
            eachindex(obs_hd)),
        fill(1.0, n_hd), (Constant(obs_hd),))
    _push!("PrimaryCensored Gamma+Uniform analytical $(n_hd)d",
        (θ,
            obs_hd) -> sum(
            i -> logpdf(
                primary_censored(Gamma(θ[i], 1.5), Uniform(0.0, 1.0)),
                obs_hd[i]),
            eachindex(obs_hd)),
        fill(2.0, n_hd), (Constant(obs_hd),))
    _push!("PrimaryCensored Gamma+Uniform numerical $(n_hd)d",
        (θ,
            obs_hd) -> sum(
            i -> logpdf(
                primary_censored(Gamma(θ[i], 1.5), Uniform(0.0, 1.0);
                    method = CensoredDistributions.NumericSolver()),
                obs_hd[i]),
            eachindex(obs_hd)),
        fill(2.0, n_hd), (Constant(obs_hd),))

    # Generic composers. Differentiate the composer `logpdf` wrt the
    # leaf-distribution parameters, with the value vector passed as a Constant
    # context. The composers are plain (no censoring), so gradients flow only
    # through the leaf delay logpdfs the slice recursion sums. Literal
    # constructors (not a captured `ctor::Type`) keep Enzyme forward happy, as
    # above. `seq2` / `par2` are the two-leaf step/branch value vectors.
    seq2 = [1.5, 2.0]
    par2 = [2.0, 3.0]
    _push!("Sequential Gamma+LogNormal logpdf",
        (θ, x) -> logpdf(
            Sequential(Gamma(θ[1], θ[2]), LogNormal(θ[3], θ[4])), x),
        [2.0, 1.0, 0.5, 0.4], (Constant(seq2),))
    _push!("Parallel Gamma+LogNormal logpdf",
        (θ, x) -> logpdf(
            Parallel(Gamma(θ[1], θ[2]), LogNormal(θ[3], θ[4])), x),
        [2.0, 1.0, 1.0, 0.5], (Constant(par2),))
    # Nested stack: a Parallel of a leaf branch and a Sequential chain (the
    # stack the front-end builds). Constructed directly so the gradient flows
    # through the composer `logpdf` slice recursion, the AD-relevant path,
    # rather than through the `compose` NamedTuple builder. The nested Gamma
    # shape starts at 2.0, not the α = 1 boundary: at exactly α = 1 a Gamma
    # collapses to an Exponential and Mooncake's reverse rule drops the
    # `log(x)` term of the shape gradient (an upstream Distributions×Mooncake
    # edge case, not a composer issue).
    nest3 = [1.5, 2.0, 3.0]
    _push!("Composed nested Parallel-of-Sequential logpdf",
        (θ,
            x) -> logpdf(
            Parallel(Gamma(θ[1], θ[2]),
                Sequential(LogNormal(θ[3], θ[4]), Gamma(θ[5], θ[6]))), x),
        [2.0, 1.0, 0.5, 0.4, 2.0, 1.0], (Constant(nest3),))

    # Choose data-selected disjunction. The gradient flows through the
    # SELECTED alternative's own `logpdf` (the type-stable selection barriers
    # into the chosen concrete type), with the selection name fixed and the
    # value passed as a Constant context. Guarded on `choose` existing so
    # the AirspeedVelocity baseline build (which lacks `Choose`) skips it. Literal
    # constructors keep Enzyme forward happy.
    if isdefined(CensoredDistributions, :choose)
        sel_obs = [0.5, 1.2, 2.5, 3.8]
        _push!("Choose Gamma|LogNormal sourced logpdf",
            (θ,
                obs) -> sum(
                x -> logpdf(
                    CensoredDistributions.choose(
                        :index => Gamma(θ[1], θ[2]),
                        :sourced => LogNormal(θ[3], θ[4])),
                    x; kind = :sourced),
                obs),
            [2.0, 1.0, 0.5, 0.4], (Constant(sel_obs),))
    end

    # Censored composer specialisations. The event vector (carrying
    # `Missing`) travels as an inactive Constant context; the gradient flows
    # through the censored leaf delay parameters along the marginalise/condition
    # path selected by the missingness. Literal constructors keep Enzyme forward
    # happy. Guarded on `primary_censored` existing for the AirspeedVelocity
    # baseline build, as with the other PR-tree scenarios above.
    if isdefined(CensoredDistributions, :primary_censored) &&
       isdefined(CensoredDistributions, :Sequential)
        # The unobserved-intermediate (marginalise-by-convolution) path is not an
        # AD fixture: its gradient correctness is covered by the main-suite
        # reference tests (ForwardDiff/ReverseDiff) and by the existing
        # `Convolved` AD scenarios that exercise the same convolution arithmetic.
        # As an end-to-end composer scenario it routes the marginalising
        # convolution through the `Convolved` unbounded-tail clamp's
        # `quantile`/`gamma_inc_inv_qsmall`, the heterogeneous-edge gap that
        # hard-crashes the compiled backends (Enzyme/Mooncake) uncatchably, so it
        # is left out of the per-backend AD suite rather than worked around.
        #
        # Observed intermediate: every event observed, so each observed-bounded
        # edge conditions on its OWN declared censoring -- here each edge
        # is scored through its own `primary_censored` logpdf at the day gap.
        seq_ev_obs = Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0])
        _push!("Sequential censored observed-intermediate logpdf",
            (θ,
                ev) -> logpdf(
                Sequential(
                    primary_censored(
                        LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0)),
                    primary_censored(Gamma(θ[3], θ[4]), Uniform(0.0, 1.0))),
                ev),
            [1.2, 0.5, 2.0, 1.0], (Constant(seq_ev_obs),))
        # Vectorised / shared evaluation over MANY records: the per-record
        # distributions share the segment construction and the product log
        # density sums their observed-intermediate contributions. With every
        # record fully observed the path is the same all-continuous-arithmetic
        # per-edge conditioning as the single-record scenario above (no
        # `Convolved`/quadrature gap), so its gradient differentiates on every
        # backend and must match the per-record loop. The rows table is inactive
        # DATA carried as a `Constant`.
        batch_rows = [(onset = 0.0, admit = 2.0, death = 5.0),
            (onset = 0.0, admit = 3.0, death = 7.0),
            (onset = 0.0, admit = 1.0, death = 4.0)]
        _push!("Vectorised Sequential censored observed logpdf",
            (θ,
                rows) -> _vectorised_records_logpdf(
                Sequential(
                    primary_censored(
                        LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0)),
                    primary_censored(Gamma(θ[3], θ[4]), Uniform(0.0, 1.0))),
                rows),
            [1.2, 0.5, 2.0, 1.0], (Constant(batch_rows),))

        # Whole-compose conv-to-last-observed right-truncation denominator.
        # One observed-intermediate record and one endpoint-observed
        # record (missing intermediate), each right-truncated at its OWN
        # per-record horizon. The observed-intermediate record scores a
        # factorised numerator; both records share a single
        # `-logcdf(conv-to-last-observed, window)` denominator that convolves
        # the leaf cores and evaluates a CDF at the (constant) window on the
        # differentiated param type. The matching `whole_compose_truncation_ad.jl`
        # was ForwardDiff-only; this DIT scenario gives the reverse backends a
        # matching scenario (verified to match the ForwardDiff reference on
        # ReverseDiff and Mooncake reverse/forward). LogNormal delays keep the
        # truncation `logcdf` off the Gamma shape-derivative path, matching the
        # established AD-safe right-truncation fixtures. The per-record horizons
        # are constants baked into the closure; the `Missing`-bearing event
        # vectors travel as an inactive Constant context.
        wct_evs = [Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0]),
            Vector{Union{Missing, Float64}}([0.5, missing, 7.0])]
        _push!("Whole-compose conv-to-last-observed truncation logpdf",
            (θ,
                evs) -> _whole_compose_truncation_logpdf(
                Sequential(
                    (primary_censored(
                            LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0)),
                        primary_censored(
                            LogNormal(θ[3], θ[4]), Uniform(0.0, 1.0))),
                    (:onset_mid, :mid_obs)),
                evs, (8.0, 9.0)),
            [1.0, 0.5, 0.5, 0.4], (Constant(wct_evs),))

        # The Parallel shared-origin censored path is not an AD fixture: its
        # marginal routes through the 1-D origin quadrature and its conditional
        # through the parameter-type promotion, both of which the compiled
        # backends (Enzyme/Mooncake) crash on uncatchably. Its gradient is
        # verified on ForwardDiff and ReverseDiff by the main-suite reference
        # tests instead. Only the all-continuous-arithmetic Sequential
        # observed-intermediate scenario, which differentiates on every backend,
        # is kept here. The plain-branch (observed-origin) Parallel path shares
        # the same `logpdf(::Parallel, ::event vector)` entry, so it also reaches
        # the quadrature branch the compiled backends crash on; its gradient is
        # likewise verified on ForwardDiff and ReverseDiff in the main suite.

        # Nested-composer (irregular tree) fully-observed scoring: a
        # two-level tree onset -> {admit -> {death, discharge}, notif}, every
        # event observed. The recursive walk conditions each edge on its own
        # declared `primary_censored` censoring at the day gap, so the gradient
        # is all-continuous-arithmetic over the leaf delay params (the same form
        # as the single-level observed-intermediate scenario above) and
        # differentiates on the analytic backends. The marginalising
        # (missing-event) tree paths route through the `Convolved`/quadrature
        # gaps the compiled backends crash on, so they stay reference-only as
        # above. The event vector carries `Missing` as an inactive Constant.
        tree_ev = Vector{Union{Missing, Float64}}(
            [0.0, 4.0, 12.0, 11.0, 9.0])
        _push!("Nested tree censored observed logpdf",
            (θ,
                ev) -> logpdf(
                Parallel(
                    Sequential(
                        primary_censored(
                            LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0)),
                        Parallel(
                            primary_censored(
                                Gamma(θ[3], θ[4]), Uniform(0.0, 1.0)),
                            primary_censored(
                                Gamma(θ[5], θ[6]), Uniform(0.0, 1.0)))),
                    primary_censored(
                        LogNormal(θ[7], θ[8]), Uniform(0.0, 1.0))),
                ev),
            [1.4, 0.4, 2.0, 1.0, 2.0, 1.2, 1.9, 0.5], (Constant(tree_ev),))

        # Nested-Resolve tree: onset -> {admit -> Resolve(death,
        # discharge), notif}, the death outcome observed. The Resolve exposes
        # one event slot per outcome, so the event vector is
        # [onset, admit, death, discharge, notif] with discharge `Missing`
        # (inactive). The observed death conditions on its branch
        # (log p_death + logpdf(Gamma, gap)), so the gradient over the death
        # branch shape/scale + the surrounding edge params is all-continuous
        # arithmetic, differentiating on every analytic backend (Enzyme shares
        # the heterogeneous-edge gap, registered broken above).
        comp_ev = Vector{Union{Missing, Float64}}(
            [0.0, 4.0, 12.0, missing, 9.0])
        _push!("Nested Resolve tree conditioned logpdf",
            (θ,
                ev) -> logpdf(
                Parallel(
                    Sequential(
                        primary_censored(
                            LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0)),
                        Resolve(
                            :death => (Gamma(θ[3], θ[4]), 0.3),
                            :discharge => (Gamma(θ[5], θ[6]), 0.7))),
                    primary_censored(
                        LogNormal(θ[7], θ[8]), Uniform(0.0, 1.0))),
                ev),
            [1.4, 0.4, 2.0, 3.0, 2.0, 1.0, 1.9, 0.5], (Constant(comp_ev),))

        # Non-terminal whole-tree Resolve: onset ->
        # Resolve(death => admit_burial CHAIN, recover => leaf), the death
        # SUBTREE observed. A composer-valued outcome spans its subtree's event
        # slots, so the event vector is [onset, admit, burial, recover] with
        # recover `Missing` (inactive). The death branch scores
        # log p_death + (subtree chain density), so the gradient over the subtree
        # leaf params + the branch probability is all-continuous arithmetic,
        # differentiating on every analytic backend (Enzyme shares the
        # heterogeneous-edge gap, registered broken below).
        nt_comp_ev = Vector{Union{Missing, Float64}}(
            [0.0, 4.0, 12.0, missing])
        _push!("Non-terminal Resolve whole-tree conditioned logpdf",
            (θ,
                ev) -> logpdf(
                Parallel(
                    (Resolve(
                        :death => (
                            Sequential(
                                (
                                    primary_censored(
                                        Gamma(θ[1], θ[2]), Uniform(0.0, 1.0)),
                                    Gamma(θ[3], θ[4])),
                                (:onset_admit, :admit_burial)),
                            θ[7]),
                        :recover => (Gamma(θ[5], θ[6]), 1 - θ[7])),),
                    (:resolution,)),
                ev),
            [2.0, 1.0, 1.5, 1.2, 3.0, 2.0, 0.4], (Constant(nt_comp_ev),))

        # Vectorised nested-Resolve (bdbv) with a PER-RECORD covariate CFR:
        # each record's Resolve branch probability comes from a
        # differentiated covariate, injected as the reserved `branch_probs`. The
        # death outcome is observed (conditioned branch), so the gradient over the
        # branch shape/scale AND the per-record CFR is all-continuous arithmetic.
        # Matches the per-record loop; rows are inactive DATA carried as Constants.
        bdbv_rows = [
            (onset = 0.0, admit = 4.0, death = 12.0,
                discharge = missing, notif = 9.0),
            (onset = 0.5, admit = 5.0, death = 13.0,
                discharge = missing, notif = 10.0)]
        _push!("Vectorised nested Resolve per-record branch_probs logpdf",
            (θ,
                rows) -> _vectorised_branch_probs_logpdf(
                Parallel(
                    (
                        Sequential(
                            (primary_censored(
                                    LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0)),
                                Resolve(
                                    :death => (Gamma(θ[3], θ[4]), 0.3),
                                    :discharge => (Gamma(θ[5], θ[6]), 0.7))),
                            (:onset_admit, :admit_resolution)),
                        primary_censored(
                            LogNormal(θ[7], θ[8]), Uniform(0.0, 1.0))),
                    (:onset_admit, :onset_notif)),
                rows,
                # Per-record CFR from a logistic of the differentiated covariate.
                [(death = 1 / (1 + exp(-θ[9])),
                        discharge = 1 - 1 / (1 + exp(-θ[9]))),
                    (death = 1 / (1 + exp(-θ[9] - θ[10])),
                        discharge = 1 - 1 / (1 + exp(-θ[9] - θ[10])))]),
            [1.4, 0.4, 2.0, 3.0, 2.0, 1.0, 1.9, 0.5, 0.2, -0.3],
            (Constant(bdbv_rows),))

        # Nested racing-hazard tree: onset -> {Hazard(death, recover),
        # notif}, the death outcome observed. The Sequential's first step targets
        # `onset`, then the Compete exposes one event slot per outcome, so
        # the event vector is [origin, onset, death, recover, notif] with recover
        # `Missing` (inactive). The observed death conditions on the cause-resolved
        # sub-density f_death(gap) ∏_{k≠death} S_k(gap), so the gradient over the
        # racing shape/scale + the surrounding edge params flows through the
        # AD-safe Gamma logpdf/logccdf, differentiating on the analytic backends.
        haz_ev = Vector{Union{Missing, Float64}}(
            [0.0, 4.0, 12.0, missing, 9.0])
        _push!("Nested racing-hazard tree conditioned logpdf",
            (θ,
                ev) -> logpdf(
                Parallel(
                    Sequential(
                        primary_censored(
                            LogNormal(θ[1], θ[2]), Uniform(0.0, 1.0)),
                        Compete(
                            :death => Gamma(θ[3], θ[4]),
                            :recover => Gamma(θ[5], θ[6]))),
                    primary_censored(
                        LogNormal(θ[7], θ[8]), Uniform(0.0, 1.0))),
                ev),
            [1.4, 0.4, 2.0, 3.0, 2.0, 1.0, 1.9, 0.5], (Constant(haz_ev),))

        # Vectorised Choose (hanta) top: each record selects its alternative by
        # `:kind` and scores its single observed value, right-truncated at its
        # `obs_time`. The gradient over the selected alternative's params is the
        # leaf primary-censored path, differentiating on the analytic backends and
        # matching the per-record loop. Rows are inactive DATA Constants.
        hanta_rows = [(kind = :index, delay = 3.0, obs_time = 8.0),
            (kind = :sourced, delay = 5.0, obs_time = 12.0)]
        _push!("Vectorised Choose per-record kind logpdf",
            (θ,
                rows) -> _vectorised_choose_logpdf(
                choose(
                    :index => primary_censored(
                        Gamma(θ[1], θ[2]), Uniform(0.0, 1.0)),
                    :sourced => primary_censored(
                        Gamma(θ[3], θ[4]), Uniform(0.0, 1.0))),
                rows),
            [2.0, 1.0, 4.0, 1.5], (Constant(hanta_rows),))
    end

    # External censoring wrappers over composers. The SCALAR combine-then-censor
    # total is the explicit `observed_distribution(Sequential)` form: the chain
    # collapses to the convolution of its steps (its observed total) before
    # censoring, so the gradient flows through the numeric convolution and the
    # interval/primary CDF. A Parallel distributes the wrapper into each branch.
    # (The bare-node wrap now distributes into leaves and stays multivariate; the
    # AD fixture for that record path is exercised elsewhere.) Guarded on the
    # censoring wrappers existing so the AirspeedVelocity baseline (which lacks
    # the composer overloads) skips them. Literal constructors keep Enzyme forward
    # happy.
    if isdefined(CensoredDistributions, :Sequential)
        _push!("interval_censored(Sequential) over total",
            (θ,
                obs) -> sum(
                x -> logpdf(
                    interval_censored(
                        observed_distribution(
                            Sequential(Gamma(θ[1], θ[2]),
                            LogNormal(θ[3], θ[4]))),
                        1.0),
                    x),
                obs),
            [2.0, 1.0, 0.5, 0.4], (Constant(obs_int),))
        _push!("double_interval_censored(Sequential) over total",
            (θ,
                obs) -> sum(
                x -> logpdf(
                    double_interval_censored(
                        observed_distribution(
                            Sequential(Gamma(θ[1], θ[2]),
                            LogNormal(θ[3], θ[4])));
                        primary_event = Uniform(0.0, 1.0), interval = 1.0),
                    x),
                obs),
            [2.0, 1.0, 0.5, 0.4], (Constant(obs_int),))
        _push!("interval_censored(Parallel) distributed branches",
            (θ,
                x) -> logpdf(
                interval_censored(
                    Parallel(Gamma(θ[1], θ[2]), LogNormal(θ[3], θ[4])), 1.0),
                x),
            [2.0, 1.0, 1.0, 0.5], (Constant([2.0, 3.0]),))
    end

    return out
end

end # module
