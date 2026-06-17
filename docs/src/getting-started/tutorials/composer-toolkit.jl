# # [Composing censored distributions](@id composer-toolkit)
#
# ## Introduction
#
# CensoredDistributions.jl composes per-event delay distributions into one
# object that describes a whole record.
# The same object scores observed records and simulates new ones, so a model is
# built once and used in both directions.
# This page is the reference the case studies point to.
#
# ### What are we going to do in this exercise
#
# Each section is a small runnable example rather than a full analysis. We:
#
# 1. Compose a record from per-event delays with [`compose`](@ref), starting
#    from plain [Distributions.jl](https://juliastats.org/Distributions.jl)
#    leaves.
# 2. Build the four composers directly ([`Sequential`](@ref),
#    [`Parallel`](@ref), [`Competing`](@ref), [`Select`](@ref)) and see how they
#    nest.
# 3. Swap the plain leaves for censored ones
#    ([`double_interval_censored`](@ref), and the rarer
#    [`primary_censored`](@ref)) as drop-in replacements the same stack handles
#    transparently.
# 4. Score and simulate from one composed object.
# 5. Attach parameters and priors with [`params_table`](@ref) and
#    [`build_priors`](@ref).
#
# ### What might I need to know before starting
#
# This page is the composer reference the case studies point to. It builds on
# [Getting Started with CensoredDistributions.jl](@ref getting-started).

# ## Packages used
#
# We use Distributions for the delay distributions, DynamicPPL for the Turing
# entry points, and Random for reproducibility.
# The Turing-facing functions ([`composed_distribution_model`](@ref),
# [`composed_parameters_model`](@ref)) live in a package extension that loads
# only once DynamicPPL is available, so the core package stays Turing-free.

using CensoredDistributions
using Distributions
using DynamicPPL
using Random

# ## Composing a record
#
# A record is a set of events linked by delays.
# [`compose`](@ref) is the front-end: it takes a friendly description and lowers
# it to a nested stack of the four composers, without introducing a new tree
# type.
# The NamedTuple form names each branch.
# A bare distribution is a leaf branch, and a `Vector` value is a chain of steps
# (a [`Sequential`](@ref)).
#
# We start from plain Distributions leaves so the composing machinery is clear
# on its own. Censoring comes next as a drop-in swap (see
# [Censoring is a drop-in leaf swap](@ref censoring-drop-in)), and nothing in
# this section or the next two changes when the leaves become censored.

onset_admit = LogNormal(1.5, 0.4);

admit_death = Gamma(2.0, 1.0);

# Two branches off one onset: an onset-to-admission delay and an
# onset-to-notification delay.
parallel_stack = compose((onset_admit = onset_admit,
    onset_notif = Gamma(1.5, 1.0)));

event_names(parallel_stack)

# A chain step is a `Vector`: onset to admission, then admission to death.
chain = compose((path = [onset_admit, admit_death],));

event_names(chain)

# The same stack also lowers from a Tables.jl table or a matrix, so a
# column-oriented data source builds a composer without a hand-written
# NamedTuple.
# A table has `name` and `dist` columns, one row per branch.
# A `chain` column folds rows sharing a non-zero id into a [`Sequential`](@ref),
# and a `compete`/`prob` column pair folds rows into a [`Competing`](@ref) node
# whose `prob` entries are the branch probabilities.
# Here the death and discharge rows share a `compete` group, while the
# notification row stays a plain branch.

table = (name = [:death, :discharge, :onset_notif],
    dist = [Gamma(1.5, 1.0), Gamma(2.0, 1.5), Gamma(1.5, 1.0)],
    compete = [1, 1, 0],
    prob = [0.3, 0.7, missing]);

table_stack = compose(table);

event_names(table_stack)

# A matrix lowers the same way: rows are [`Parallel`](@ref) branches and the
# columns within a row are [`Sequential`](@ref) steps.
# `names` labels the rows and `step_names` labels the columns.
# A one-row matrix is a single chain, so this builds the onset-to-admission then
# admission-to-death sequence.

matrix_stack = compose([onset_admit admit_death];
    names = [:path], step_names = [:onset_admit, :admit_death]);

event_names(matrix_stack)

# ## The four composers
#
# Each front-end lowers to these composers, which can also be built directly.
# They differ in how the branches relate.
#
# [`Sequential`](@ref) is a conjunctive chain: each step adds an independent
# delay onto the previous event.
# The lowercase [`sequential`](@ref) verb is the public constructor (the
# uppercase struct is kept for internal use); name the steps with `name => dist`
# pairs.

seq = sequential(:onset_admit => onset_admit, :admit_death => admit_death);

# [`Parallel`](@ref) places independent branches off one shared origin, built
# with the [`parallel`](@ref) verb.

par = parallel(:onset_admit => onset_admit, :onset_notif => admit_death);

# [`Competing`](@ref) is a set of competing outcomes: exactly one occurs,
# governed by branch probabilities that sum to one.
# A death-versus-discharge competition makes the death probability the
# case-fatality ratio.

cfr = 0.3;

resolution = competing(:death => (Gamma(1.5, 1.0), cfr),
    :discharge => (Gamma(2.0, 1.5), 1 - cfr));

# The LAST outcome's probability may be omitted (a bare `name => delay`): it
# then takes the residual `1 - sum(of the others)`, so the discharge
# probability `1 - cfr` need not be written out.

resolution_residual = competing(:death => (Gamma(1.5, 1.0), cfr),
    :discharge => Gamma(2.0, 1.5));

# Its marginal is the time to resolution regardless of which outcome occurs.

mean(resolution)

# A `Competing` node carries its own time-to-resolution event slot alongside the
# named per-outcome slots, so its flat event layout pairs that resolution slot
# (defaulting to `:event_1`) with the `:death` and `:discharge` outcome names.

event_names(resolution)

# [`Select`](@ref) is a data-selected disjunction: the alternatives are
# independent sub-models with different origins, and a data field picks which
# one applies to a record.
# Neither `Parallel` (shared origin) nor `Competing` (shared origin) expresses
# this.

selector = selecting(:index => onset_admit, :sourced => admit_death);

# Scoring names the active alternative through the `kind` keyword.

logpdf(selector, 3.0; kind = :index)

# ## Nesting
#
# The composers nest, so trees of arbitrary depth are built by composing on
# composers.
# A `compose` result drops into another `compose` as a branch.

early = compose((onset_admit = onset_admit, onset_notif = admit_death));

nested = compose((early = early, late = chain));

event_names(nested)

# A `Select` can hold a `Select`, or a composed tree, as an alternative.

select_on_select = selecting(:a => selector, :b => onset_admit);

# A pre-built composer is a valid `Sequential` step, so a chain can carry a
# `Competing` resolution as its terminal step.
# Naming the chain steps gives the simulated record readable event names.

tree = compose((
    path = sequential(:onset_admit => onset_admit,
        :admit_resolve => resolution),
    onset_notif = admit_death));

# The flat event layout of a tree is derived from the edge names.

event_names(tree)

# ## [Censoring is a drop-in leaf swap](@id censoring-drop-in)
#
# Everything so far used plain Distributions leaves. Real data is rarely that
# clean: a delay is usually recorded to the day, so the primary event is
# censored to its day window and the observed delay is interval censored to a
# day. We handle this by swapping the plain leaf for a censored one. Nothing
# else about the stack changes.
#
# The plain stack first, for reference: two delays off a shared onset.

plain_stack = compose((
    onset_admit = LogNormal(1.5, 0.5),
    onset_death = Gamma(2.0, 1.0)));

# [`double_interval_censored`](@ref) layers primary-event censoring, optional
# truncation, and interval censoring onto a delay. It is the default leaf for
# line-list data, because day-resolution dates carry both a primary-event window
# and a day-wide observation interval. Swap each plain leaf for its censored
# counterpart and the NamedTuple keys, the tree shape, and the call site are all
# unchanged.

censored_stack = compose((
    onset_admit = double_interval_censored(LogNormal(1.5, 0.5); interval = 1,
        upper = 20),
    onset_death = double_interval_censored(Gamma(2.0, 1.0); interval = 1)));

# A composer holds any univariate distribution as a leaf and dispatches on the
# leaf type, so it treats a censored leaf exactly as it treated the plain one.
# The two stacks expose the same events.

event_names(plain_stack) == event_names(censored_stack)

# The package handles the swap transparently: the same `compose` stack scores
# and simulates either way, and the censoring is carried inside the leaf rather
# than bolted onto the tree. Mixing leaf kinds in one stack is fine, since each
# leaf is scored by its own type.

mixed_stack = compose((
    onset_admit = double_interval_censored(LogNormal(1.5, 0.5); interval = 1),
    onset_death = Gamma(2.0, 1.0)));

event_names(mixed_stack)

# [`primary_censored`](@ref) is the rarer drop-in: the primary event is censored
# but the observed delay is not interval censored (a continuously recorded
# observation time). It slots into the same stack the same way.

primary_only_stack = compose((
    onset_admit = primary_censored(LogNormal(1.5, 0.5), Uniform(0, 1)),
    onset_death = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))));

event_names(primary_only_stack)

# From here on we build with `double_interval_censored` leaves by default, since
# day-resolution line-list data is what we usually have.

# ## Truncating the whole chain
#
# The truncation above is per leaf: `double_interval_censored(...; upper = 20)`
# right-truncates a single delay before it enters the stack.
# An observation horizon is different. Real-time data is observed up to one
# cut-off, so the *whole* composed chain is right-truncated at that horizon, not
# each leaf in isolation.
# `event_logpdf(stack, events; horizon)` applies that single truncation across
# the assembled chain in one call: the factorised per-segment numerator is
# divided by one denominator, the CDF of the convolution from the origin to the
# last observed event evaluated at `horizon - origin`.
# The convolution itself is the sum-of-independent-delays primitive
# [`convolve_distributions`](@ref), which returns a [`Convolved`](@ref); a chain
# endpoint convolves its steps this way.
# The [An Rt renewal model with delay convolution](@ref rt-renewal-convolution)
# tutorial uses it directly.

obs_chain = sequential(
    :onset_admit => double_interval_censored(LogNormal(1.2, 0.5); interval = 1),
    :admit_death => double_interval_censored(Gamma(2.0, 1.0); interval = 1));

ev = Vector{Union{Missing, Float64}}([0.0, 2.0, 5.0]);

# Without a horizon the chain scores the full density. Passing `horizon` adds
# the whole-chain right-truncation correction in the same call.

full_lp = CensoredDistributions.event_logpdf(obs_chain, ev)

horizon_lp = CensoredDistributions.event_logpdf(obs_chain, ev; horizon = 8.0)

# The correction is one denominator over the whole chain, so it holds whatever
# the record observes. An endpoint-observed record with the intermediate
# admission `missing` truncates the same onset-to-death total.

ev_mid_missing = Vector{Union{Missing, Float64}}([0.0, missing, 5.0]);

CensoredDistributions.event_logpdf(obs_chain, ev_mid_missing; horizon = 8.0)

# !!! note "How a partly-observed chain is scored"
#     When only some events in a chain are seen, the package marginalises over
#     the unobserved intermediate events, integrating them out as the
#     convolution of their bare continuous delays. The day-interval (and
#     primary) censoring on an unobserved internal node is dropped, because no
#     date was recorded there to censor; only the observed events carry their
#     censoring, applied once over the convolved delay between them. So with the
#     admission `missing` above, the onset-to-admission and admission-to-death
#     delays convolve into one onset-to-death gap whose censoring lives on the
#     observed onset and death. This is automatic per record, driven by each
#     record's own missingness pattern.

# ### A bounded observation window
#
# The horizon above is upper-only: a record is kept if its last observed event
# fell at or before the cut-off, normalised by `cdf(delay, horizon)`.
# Some designs observe events only within a bounded window ending at the
# horizon, for example a fixed recall period or a study that enrols and follows
# cases for a set span.
# `truncate_to_window(d, horizon, δ)` adds a lower edge a width `δ` below the
# horizon, truncating to the finite window `[horizon - δ, horizon]` normalised
# by `cdf(d, horizon) - cdf(d, horizon - δ)`.
# Per record, a reserved `obs_window` row field carries `δ` alongside
# `obs_time`, so a row `(onset = 0, ..., obs_time = 8, obs_window = 3)` scores
# the record over the window `[5, 8]`.
# Leaving `obs_window` out (or `truncate_to_window(d, horizon, nothing)`)
# reproduces the upper-only horizon exactly.

window_dist = LogNormal(1.5, 0.5);

windowed = CensoredDistributions.truncate_to_window(window_dist, 6.0, 4.0)

window_lognorm = log(cdf(window_dist, 6.0) - cdf(window_dist, 2.0))

# ## Scoring and simulation from one object
#
# The composer is dual-purpose: it scores observed records and simulates new
# ones.
# We reuse the named two-step `obs_chain` over censored leaves to demonstrate
# both.
#
# [`composed_distribution_model`](@ref) scores one record passed as a NamedTuple
# keyed by event name.
# A `missing` field is integrated out; an observed field is conditioned on.
# This is the first use of Turing here, so the model entry comes from the
# DynamicPPL package extension loaded above.

@model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

row = (onset = 0.0, admit = 2.0, death = 5.0);

only(logjoint(demo(obs_chain, row), (;)))

# For many records, `record_distributions` assembles a vector of per-record
# distributions and `batched_event_logpdf` scores them at once.
# Each record bakes in its own missingness pattern, so one call handles a mix
# of missingness across the dataset: where the intermediate admission is
# observed the chain conditions on it, and where it is `missing` the same chain
# integrates it out (the two delays convolve into one onset-to-death gap).
# We use a small batch of records so the mix is visible: three observe the
# admission, three leave it `missing`, and one onset is even repeated to show
# that nothing special happens per record.

rows = [(onset = 0.0, admit = 2.0, death = 5.0),
    (onset = 0.0, admit = 3.0, death = 6.0),
    (onset = 1.0, admit = 2.5, death = 5.5),
    (onset = 0.0, admit = missing, death = 5.0),
    (onset = 1.0, admit = missing, death = 7.0),
    (onset = 2.0, admit = missing, death = 9.0),
    (onset = 1.0, admit = 3.0, death = 7.0)];

recs = CensoredDistributions.record_distributions(obs_chain, rows);

length(recs)

# Scoring the whole batch at once: the conditioned and integrated-out records
# contribute to one log density, with no per-record bookkeeping at the call
# site.
# `batched_event_logpdf` takes the records keyed by event name and reads each
# row's missingness directly, so a `missing` admission stays `missing` rather
# than being filled with a placeholder value.

CensoredDistributions.batched_event_logpdf(obs_chain, rows)

# The mix is genuine, not a relabelling: an observed-admission record scores the
# two segments separately, while a `missing`-admission record scores a single
# convolved onset-to-death gap.
# We can see the two regimes by scoring each record on its own.

per_record = [CensoredDistributions.batched_event_logpdf(obs_chain, [row])
              for row in rows];

(observed_admit = round.(per_record[1:3]; digits = 3),
    integrated_out = round.(per_record[4:6]; digits = 3))

# The same object simulates.
# A `rand` of a nested tree returns a full named event record: a shared origin
# draw, every event hung off it, and each `Competing` resolution sampled.
# Sampling a `Competing` node draws which outcome occurs from the branch
# probabilities and then draws that outcome's time, so exactly one outcome slot
# is filled and the competing outcomes that did not occur are left `missing`.
# The competing node is itself sampled; the `missing` slots are the outcomes
# that lost, not an un-sampled node.
#
# We build a simulation tree with the default `double_interval_censored` leaves:
# a death-versus-discharge resolution off the admission, alongside a
# notification branch. The named per-outcome event slots come from the competing
# outcome names.

sim_resolution = competing(
    :death => (double_interval_censored(Gamma(1.5, 1.0); interval = 1), cfr),
    :discharge => (double_interval_censored(Gamma(2.0, 1.5); interval = 1),
        1 - cfr));

sim_tree = compose((
    path = sequential(
        :onset_admit =>
            double_interval_censored(LogNormal(1.5, 0.4); interval = 1),
        :admit_resolve => sim_resolution),
    onset_notif = double_interval_censored(Gamma(2.0, 1.0); interval = 1)));

event_names(sim_tree)

# A draw fills the origin, the admission, exactly one of the competing
# resolution outcomes, and the notification. The other outcome is `missing`.

draw = rand(Xoshiro(7), sim_tree)

# Read the sampled resolution straight off the record: the one outcome with a
# time is the one that occurred, and its value is the sampled event time. With a
# case-fatality ratio below one half the death branch wins less often, so this
# draw usually resolves to discharge.

resolved = only(o for o in (:death, :discharge) if !ismissing(draw[o]));

(outcome = resolved, time = draw[resolved])

# Exactly one resolution outcome is sampled; the other competing slot is
# `missing`.

count(!ismissing, (draw.death, draw.discharge))

# Sampling the `Competing` node on its own makes the draw explicit:
# [`rand_outcome`](@ref) returns the drawn `(outcome, time)` pair directly,
# whereas the plain `rand` of a `Competing` returns only the marginal
# time-to-resolution and discards which outcome won.

CensoredDistributions.rand_outcome(Xoshiro(7), sim_resolution)

# ## [Marginal versus latent](@id marginal-versus-latent)
#
# Scoring and prediction are two directions on the same object.
# Scoring marginalises or conditions each event by its row missingness: an
# integrated-out event convolves into its neighbours, an observed event
# conditions.
# Prediction is the generative `rand`, which samples every internal event time.
#
# The marginal and latent forms are one family sharing the same parameters.
# The marginal form integrates the intermediate event out inside `logpdf`, so it
# adds no extra dimensions and is the cheap default.
# The latent form instead samples the intermediate event and scores each segment
# against it, the same object read in the other direction.
# [`latent`](@ref) wraps a node to select the latent representation.

leaf = double_interval_censored(LogNormal(1.2, 0.5); interval = 1);

ld = latent(leaf);

# The marginal form scores one number: the observed time with the primary event
# integrated out.

y = 3.0;

marginal_lp = logpdf(leaf, y)

# The latent form is multivariate over `[primary, observed]`: sample the
# intermediate primary event, then score the two segments against it.
# Its `logpdf` is the primary prior plus the observed-given-primary conditional.

path = rand(Xoshiro(11), ld)

# Scoring that drawn path is the joint over both segments.

latent_lp = logpdf(ld, path)

# The marginal scores one number; the latent scores the sampled event path. The
# latent integrates to the marginal over the primary, so a marginal fit and a
# latent fit recover the same parameters (worked through in the fitting
# tutorial).

(marginal = marginal_lp, latent_joint = latent_lp)

# Prediction samples every internal event time directly from the latent form via
# `rand(latent(d))`: a full `(primary, observed)` record with no model and no
# conditioning.

rand(Xoshiro(11), ld)

# Prefer the latent form when the marginal integral is impractical: very complex
# delay distributions where the convolution has no closed form and numeric
# integration is expensive, or small-count problems where the extra latent
# dimensions cost little and the sampler explores the joint more robustly than a
# stiff one-dimensional marginal.
# Prefer the marginal form by default: it carries no per-record latents, so it
# is cheaper and lower-dimensional at scale.
# Because both share the same parameters, a posterior fitted in the cheap
# marginal form drops straight into the latent form for event-based prediction.
# The [Fit marginal, sample event based](@ref) tutorial works this through end
# to end, and the [Real-time Andes virus delays from the Epuyén line
# list](@ref andv-linelist-analysis) case study fits both forms on the same data
# and compares the two fits directly.

# ## Parameters and priors
#
# A composed distribution carries a flat inventory of its free parameters.
# [`params_table`](@ref) lists one row per scalar parameter, keyed by the edge
# path and the parameter name, with the support a prior must respect.
# It prints as a table and is a Tables.jl source (a
# [`ParamsTable`](@ref CensoredDistributions.ParamsTable)), so
# `tbl.edge`/`tbl.param` read its columns and `DataFrame(tbl)` makes a
# DataFrame.

template = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)));

tbl = params_table(template)

# Its columns are accessed by name.

tbl.edge, tbl.param

# [`build_priors`](@ref) takes that table (any Tables.jl source with `edge`,
# `param`, `value`, `support` columns) and turns it into the nested prior
# NamedTuple the parameter model expects.
# It derives a default prior per row from that leaf's support: a positive scale
# parameter gets a positive-truncated prior, a location parameter an unbounded
# one, a `[0, 1]` probability a `Uniform(0, 1)`.
# So `build_priors(tbl)` alone yields a complete set, and a `default` function
# or per-parameter override replaces only the rows you care about, all defined
# against the table rather than by hand-matching the tree.

priors = build_priors(tbl);

priors.onset_admit.shape

# The nested prior NamedTuple feeds [`composed_parameters_model`](@ref): it
# samples those priors and rebuilds the same composer structure, which then
# drops into the record model ([`composed_distribution_model`](@ref)) for the
# likelihood.

@model function param_demo(t, p)
    d ~ to_submodel(composed_parameters_model(t, p))
    return d
end

reconstructed = param_demo(template, priors)();

event_names(reconstructed)

# [`update`](@ref) applies a set of parameter values back to a composed object,
# returning a distribution of the same structure.
# After fitting, `update(template, chain)` reads posterior means straight off a
# fitted chain (via [`chain_to_params`](@ref)), giving a ready-to-`rand` or
# ready-to-inspect distribution. The overall
# [`mean`](@ref CensoredDistributions.mean) gives the mean delay
# per branch endpoint (this `Parallel` template has two independent endpoints),
# and the per-event [`mean`](@ref CensoredDistributions.mean)`(latent(updated))`
# Vector reads every event mean off it in one call.

updated = update(template, (onset_admit = (shape = 3.0, scale = 1.5),
    admit_death = (mu = 0.7, sigma = 0.5)));

NamedTuple{keys(event_tree(updated))}(Tuple(mean(updated)))

# ## Editing a composed tree
#
# [`update`](@ref) also replaces whole nodes, not just their values.
# Passing `path => new_node` pairs swaps the node at each address for a new
# distribution, keeping the tree shape.
# The address is the same one [`event`](@ref) reads: a bare name, a dotted
# `Symbol`, or a tuple of edge names.

replaced = update(template, :admit_death => Gamma(3.0, 1.5));

event(replaced, :admit_death)

# Two edits that change the tree shape are kept separate.
# [`prune`](@ref) drops a branch (renormalising a [`Competing`](@ref) arm), and
# [`splice`](@ref) inserts a before/after step around a node.
# These are the two topology edits; `update` keeps the shape and replaces
# contents.

three_way = competing(:death => (Gamma(1.5, 1.0), 0.3),
    :discharge => (Gamma(2.0, 1.5), 0.4),
    :transfer => (Gamma(1.0, 1.0), 0.3));

resolution_tree = compose((resolution = three_way, onset = Gamma(1.0, 1.0)));

pruned = prune(resolution_tree, :resolution, :transfer);

# Pruning drops the `:transfer` outcome and renormalises the remaining arm, so
# the death and discharge probabilities scale up to sum to one again.
# Before pruning the three arms carry their original probabilities.

winning_probabilities(three_way)

# After pruning, the death and discharge arms scale up to sum to one.

winning_probabilities(event(pruned, :resolution))

# `splice` wraps a node in a chain, here adding a reporting delay after the
# notification branch.

spliced = splice(template, :admit_death;
    after = :death_report => Gamma(1.0, 2.0));

event_names(event(spliced, :admit_death))

# ## Syntax reference
#
# Every public form on one composed object, with whether it preserves the tree
# shape:
#
# | Syntax | What it does | Shape-preserving? |
# |---|---|---|
# | `compose((a = d1, b = d2))` | NamedTuple front-end; a `Vector` value is a chain | builds |
# | `compose(table)` | table front-end (a `name`/`dist` column source); an optional `chain` column folds rows into a `Sequential`, a `compete`/`prob` column pair into a `Competing` | builds |
# | `compose(matrix; names, step_names)` | matrix front-end (rows are `Parallel` branches, columns within a row `Sequential` steps) | builds |
# | `sequential(:a => d1, :b => d2)` | a [`Sequential`](@ref) chain (steps add up) | builds |
# | `parallel(:a => d1, :b => d2)` | a [`Parallel`](@ref) branch set (shared origin) | builds |
# | `competing(:a => (d1, p1), :b => (d2, p2))` | a [`Competing`](@ref) node (one outcome occurs); the last prob may be omitted as the residual `1 - sum(others)` | builds |
# | `selecting(:a => d1, :b => d2)` | a [`Select`](@ref) disjunction (data picks the branch) | builds |
# | `convolve_distributions(d1, d2)` | a [`Convolved`](@ref) sum `X + Y` (delays add) | builds |
# | `difference(d1, d2)` | a [`Difference`](@ref) `X - Y`, the dual of the sum; two-sided support, so an observation not a delay leaf | builds |
# | `primary_censored(d, pe)` | primary-event censoring leaf | leaf wrap |
# | `interval_censored(d; interval)` | interval-censoring leaf | leaf wrap |
# | `double_interval_censored(d; interval)` | primary + truncation + interval leaf | leaf wrap |
# | `truncate_to_horizon(d, h)` | right-truncate a delay at a horizon | leaf wrap |
# | `truncate_to_window(d, h, δ)` | δ-bounded right-truncation to `[h - δ, h]` | leaf wrap |
# | `shared(:tag, d)` | tag a leaf as a tied parameter group (leaf-local tie) | leaf wrap |
# | `tie(d, paths...; name)` | tie leaves at `paths` into one group (tree-level tie) | yes |
# | `update(d, (a = (shape = 3,),))` | replace free parameter values | yes |
# | `update(d, path => new_node)` | replace whole nodes | yes |
# | `prune(d, path)` | drop a branch (renormalise a `Competing` arm) | no (topology) |
# | `splice(d, path; before, after)` | insert a before/after step at a node | no (topology) |
# | `event(d, path)` | fetch a child or descend a name path | read |
# | `event_tree(d)` | the nested tree of event names | read |
# | `event_names(d)` | the flat per-event names | read |
# | `params_table(d)` | the flat free-parameter inventory | read |
# | `update(d, chain_to_params(d, chain))` | read fitted values back onto `d` | yes |
#
# The address `path` in `event` / `update` / `prune` / `splice` / `tie` is the
# same in all: a bare `Symbol`, a dotted `Symbol` (`:a.b`), or a tuple of edge
# names.
#
# `shared(:tag, d)` and `tie(d, paths...; name = :tag)` are two spellings of the
# SAME tie. `shared` tags a leaf where it is built (leaf-local); `tie` walks the
# tree to the named leaves and wraps each in the EXACT `shared(:tag, leaf)`
# artefact (tree-level). Both make the tagged occurrences one free parameter, so
# `params_table` / `build_priors` / `update` treat them identically.

# ## Summary
#
# - [`compose`](@ref) lowers a NamedTuple, table, or matrix to the same composer
#   stack.
# - [`Sequential`](@ref), [`Parallel`](@ref), [`Competing`](@ref), and
#   [`Select`](@ref) are conjunctive chains, shared-origin branches, competing
#   outcomes, and data-selected disjunctions.
# - The composers nest, including a composer as a chain step and a
#   `selecting` of a `selecting`.
# - Censoring is a drop-in leaf swap: plain Distributions leaves teach the
#   machinery, then [`double_interval_censored`](@ref) (the default for
#   day-resolution line lists) and the rarer [`primary_censored`](@ref) replace
#   them with no change to the stack, scored by leaf type.
# - One object scores records and simulates them; scoring marginalises by row
#   missingness (mixed across records), prediction is the generative `rand`,
#   which samples each `Competing` resolution (one outcome drawn, losers
#   `missing`).
# - `event_logpdf(stack, events; horizon)` right-truncates the *whole* composed
#   chain at an observation horizon in one call, distinct from per-leaf
#   truncation baked into a `double_interval_censored` leaf.
# - The marginal and latent forms are one family on the same parameters. The
#   marginal is the cheap default; the latent samples the intermediate event and
#   suits small counts or distributions whose marginal integral is impractical.
# - [`params_table`](@ref), [`build_priors`](@ref) (support-derived defaults),
#   [`composed_parameters_model`](@ref), and [`update`](@ref) attach parameters
#   and priors to the same object and feed the record model.
# - [`update`](@ref) edits the tree too: `path => new_node` replaces nodes
#   keeping the shape, while [`prune`](@ref) and [`splice`](@ref) are the two
#   topology edits. The syntax reference above lists every public form.
#
# ## Where next
#
# - To fit a composed distribution to data, see [Fitting CensoredDistributions.jl
#   modified distributions with Turing.jl](@ref), which takes the `params_table`
#   / `build_priors` / `composed_parameters_model` pieces shown here through a
#   full Turing fit and posterior summary. We do not repeat fitting on this page.
# - The [Fit marginal, sample event based](@ref) tutorial fits in the cheap
#   marginal form and then samples event paths from the latent form.
# - To write your own leaf or composer that plugs into these tools, see
#   [Extending the composer toolkit](@ref extending-composer): the interface a
#   custom distribution must satisfy, a worked example composed into a tree, and
#   the public conformance harness that checks it.
