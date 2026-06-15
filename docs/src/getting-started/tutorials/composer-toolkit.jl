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
# 1. Compose a record from per-event delays with [`compose`](@ref).
# 2. Build the four composers directly ([`Sequential`](@ref),
#    [`Parallel`](@ref), [`Competing`](@ref), [`Select`](@ref)) and see how they
#    nest.
# 3. Score and simulate from one composed object.
# 4. Attach parameters and priors with [`params_table`](@ref) and
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

onset_admit = primary_censored(LogNormal(1.5, 0.4), Uniform(0, 1));

admit_death = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1));

# Two branches off one onset: an onset-to-admission delay and an
# onset-to-notification delay.
parallel = compose((onset_admit = onset_admit,
    onset_notif = primary_censored(Gamma(1.5, 1.0), Uniform(0, 1))));

event_names(parallel)

# A chain step is a `Vector`: onset to admission, then admission to death.
chain = compose((path = [onset_admit, admit_death],));

event_names(chain)

# ## The four composers
#
# Each front-end lowers to these composers, which can also be built directly.
# They differ in how the branches relate.
#
# [`Sequential`](@ref) is a conjunctive chain: each step adds an independent
# delay onto the previous event.

seq = Sequential(onset_admit, admit_death);

# [`Parallel`](@ref) places independent branches off one shared origin.

par = Parallel(onset_admit, admit_death);

# [`Competing`](@ref) is a set of competing outcomes: exactly one occurs,
# governed by branch probabilities that sum to one.
# A death-versus-discharge competition makes the death probability the
# case-fatality ratio.

cfr = 0.3;

resolution = competing(:death => (Gamma(1.5, 1.0), cfr),
    :discharge => (Gamma(2.0, 1.5), 1 - cfr));

# Its marginal is the time to resolution regardless of which outcome occurs.

mean(resolution)

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
    path = Sequential((onset_admit, resolution),
        (:onset_admit, :admit_resolve)),
    onset_notif = admit_death));

# The flat event layout of a tree is derived from the edge names.

event_names(tree)

# ## Censoring is transparent
#
# A composer holds any univariate distribution as a leaf, so the censored
# building blocks go straight into the stack.
# [`double_interval_censored`](@ref) layers primary censoring, truncation, and
# interval censoring onto a delay, and the composer treats the result as an
# ordinary leaf.

censored_leaf = double_interval_censored(LogNormal(1.5, 0.5); interval = 1,
    upper = 20);

censored_stack = compose((
    onset_admit = censored_leaf,
    onset_death = double_interval_censored(Gamma(2.0, 1.0); interval = 1)));

event_names(censored_stack)

# ## Scoring and simulation from one object
#
# The composer is dual-purpose: it scores observed records and simulates new
# ones.
# We build a named two-step chain over censored leaves to demonstrate both.

obs_chain = Sequential(
    (primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1)),
        primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))),
    (:onset_admit, :admit_death));

# [`composed_distribution_model`](@ref) scores one record passed as a NamedTuple
# keyed by event name.
# A `missing` field is integrated out; an observed field is conditioned on.
# This is the first use of Turing here, so the model entry comes from the
# DynamicPPL package extension loaded above.

@model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

row = (onset = 0.0, admit = 2.0, death = 5.0);

only(logjoint(demo(obs_chain, row), (;)))

# For many records, `record_distributions` assembles a vector of per-record
# distributions and `product_distribution` scores them at once.
# Each record bakes in its OWN missingness pattern, so one call handles a mix of
# missingness across the dataset: where the intermediate admission is observed
# the chain conditions on it, and where it is `missing` the same chain integrates
# it out (the two delays convolve into one onset-to-death gap).
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

# The event matrix is one column per record in `[onset, admit, death]` layout.
# A `missing` admission keeps a placeholder slot (its value is ignored, since the
# record integrates that event out); we fill it with `0.0` to keep the matrix
# numeric.

events = reduce(hcat,
    [Float64[r.onset, coalesce(r.admit, 0.0), r.death] for r in rows]);

# Scoring the whole batch at once: the conditioned and integrated-out records
# contribute to one log density, with no per-record bookkeeping at the call site.

logpdf(product_distribution(recs), events)

# The mix is genuine, not a relabelling: an observed-admission record scores the
# two segments separately, while a `missing`-admission record scores a single
# convolved onset-to-death gap.
# We can see the two regimes by scoring each record on its own.

per_record = [logpdf(recs[i],
                  Float64[rows[i].onset, coalesce(rows[i].admit, 0.0),
                      rows[i].death])
              for i in eachindex(rows)];

(observed_admit = round.(per_record[1:3]; digits = 3),
    integrated_out = round.(per_record[4:6]; digits = 3))

# The same object simulates.
# A `rand` of a nested tree returns a full named event record: a shared origin
# draw, every event hung off it, and the unsampled `Competing` outcomes left
# `missing`.

draw = rand(Xoshiro(7), tree)

# Exactly one resolution outcome is sampled.

count(!ismissing, (draw.death, draw.discharge))

# ## Marginal versus latent
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

leaf = primary_censored(LogNormal(1.2, 0.5), Uniform(0, 1));

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

# Prefer the LATENT form when the marginal integral is impractical: very complex
# delay distributions where the convolution has no closed form and numeric
# integration is expensive, or small-count problems where the extra latent
# dimensions cost little and the sampler explores the joint more robustly than a
# stiff one-dimensional marginal.
# Prefer the MARGINAL form by default: it carries no per-record latents, so it
# is cheaper and lower-dimensional at scale.
# Because both share the same parameters, a posterior fitted in the cheap
# marginal form drops straight into the latent form for event-based prediction.
# The [Fit marginal, sample event based](@ref) tutorial works this through end
# to end, and [Fitting CensoredDistributions.jl modified distributions with
# Turing.jl](@ref) compares the two fits directly.

# ## Parameters and priors
#
# A composed distribution carries a flat inventory of its free parameters.
# [`params_table`](@ref) lists one row per scalar parameter, keyed by the edge
# path and the parameter name, with the support a prior must respect.
# It prints as a table and is a Tables.jl source (a
# [`ParamsTable`](@ref CensoredDistributions.ParamsTable)), so
# `tbl.edge`/`tbl.param` read its columns and `DataFrame(tbl)` makes a DataFrame.

template = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)));

tbl = params_table(template)

# Its columns are accessed by name.

tbl.edge, tbl.param

# [`build_priors`](@ref) takes that TABLE (any Tables.jl source with `edge`,
# `param`, `value`, `support` columns) and turns it into the nested prior
# NamedTuple the parameter model expects.
# It derives a default prior per row from that leaf's SUPPORT: a positive scale
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

# ## Summary
#
# - [`compose`](@ref) lowers a NamedTuple, table, or matrix to the same composer
#   stack.
# - [`Sequential`](@ref), [`Parallel`](@ref), [`Competing`](@ref), and
#   [`Select`](@ref) are conjunctive chains, shared-origin branches, competing
#   outcomes, and data-selected disjunctions.
# - The composers nest, including a composer as a chain step and a
#   `selecting` of a `selecting`.
# - One object scores records and simulates them; scoring marginalises by row
#   missingness (mixed across records), prediction is the generative `rand`.
# - The marginal and latent forms are one family on the same parameters. The
#   marginal is the cheap default; the latent samples the intermediate event and
#   suits small counts or distributions whose marginal integral is impractical.
# - [`params_table`](@ref), [`build_priors`](@ref) (support-derived defaults),
#   [`composed_parameters_model`](@ref), and [`update`](@ref) attach parameters
#   and priors to the same object and feed the record model.
#
# ## Where next
#
# - To FIT a composed distribution to data, see [Fitting CensoredDistributions.jl
#   modified distributions with Turing.jl](@ref), which takes the `params_table`
#   / `build_priors` / `composed_parameters_model` pieces shown here through a
#   full Turing fit and posterior summary. We do not repeat fitting on this page.
# - The [Fit marginal, sample event based](@ref) tutorial fits in the cheap
#   marginal form and then samples event paths from the latent form.
# - To write your OWN leaf or composer that plugs into these tools, see
#   [Extending the composer toolkit](@ref extending-composer): the interface a
#   custom distribution must satisfy, a worked example composed into a tree, and
#   the public conformance harness that checks it.
