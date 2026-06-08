# # [The composer toolkit](@id composer-toolkit)
#
# CensoredDistributions.jl composes per-event delay distributions into one
# object that describes a whole record.
# The same object scores observed records and simulates new ones, so a model is
# built once and used in both directions.
# This page is the reference the case studies point to: it demonstrates the
# composer front-ends, how they nest, how to score and simulate from one object,
# and how to attach parameters and priors.
# Each section is a small runnable example rather than a full analysis.

# ## Packages used
#
# We use Distributions for the delay distributions, DynamicPPL for the Turing
# entry points, and Random for reproducibility.

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

selector = select(:index => onset_admit, :sourced => admit_death);

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

select_on_select = select(:a => selector, :b => onset_admit);

# A pre-built composer is a valid `Sequential` step, so a chain can carry a
# `Competing` resolution as its terminal step.
# Naming the chain steps gives the simulated record readable event names.

tree = compose((
    path = Sequential((onset_admit, resolution),
        (:onset_admit, :admit_resolve)),
    onset_notif = admit_death));

# The flat event layout of a tree is derived from the edge names.

CensoredDistributions.tree_event_names(tree)

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

@model demo(d, r) = obs ~ to_submodel(composed_distribution_model(d, r))

row = (onset = 0.0, admit = 2.0, death = 5.0);

only(logjoint(demo(obs_chain, row), (;)))

# For many records, [`record_distributions`](@ref) assembles a vector of
# per-record distributions and `product_distribution` scores them at once.
# Each record bakes in its own missingness pattern, so the same call handles a
# mix of observed and unobserved intermediate events.

rows = [(onset = 0.0, admit = missing, death = 5.0),
    (onset = 1.0, admit = missing, death = 7.0)];

recs = CensoredDistributions.record_distributions(obs_chain, rows);

events = hcat([0.0, 0.0, 5.0], [1.0, 0.0, 7.0]);

logpdf(product_distribution(recs), events)

# The same object simulates.
# A `rand` of a nested tree returns a full named event record: a shared origin
# draw, every event hung off it, and the unsampled `Competing` outcomes left
# `missing`.

draw = rand(Xoshiro(7), tree)

# Exactly one resolution outcome is sampled.

count(!ismissing, (draw.death, draw.discharge))

# [`predict_events`](@ref) draws full event paths directly from a
# [`latent`](@ref) distribution, with no model and no conditioning.
# This is the forward-simulation path for fresh records.

ld = latent(primary_censored(LogNormal(1.4, 0.5), Uniform(0, 1)));

predict_events(ld; rng = MersenneTwister(1))

# ## Marginal versus latent
#
# Scoring and prediction are two directions on the same object.
# Scoring marginalises or conditions each event by its row missingness: an
# integrated-out event convolves into its neighbours, an observed event
# conditions.
# Prediction is the generative `rand`, which samples every internal event time.
# Because the marginal and latent forms are one family sharing the same
# parameters, a posterior fitted in the cheap marginal form drops straight into
# the latent form for event-based prediction.
# The [Fit marginal, sample event based](@ref) tutorial works this through end
# to end.

# ## Parameters and priors
#
# A composed distribution carries a flat inventory of its free parameters.
# [`params_table`](@ref) lists one row per scalar parameter, keyed by the edge
# path and the parameter name, with the support a prior must respect.

template = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)));

tbl = params_table(template);

tbl.edge, tbl.param

# [`build_priors`](@ref) turns that table into the nested prior NamedTuple the
# parameter model expects.
# A `default` function picks a prior per row, so priors are defined against the
# table rather than by hand-matching the tree.

priors = build_priors(tbl;
    default = row -> truncated(Normal(row.value, 1); lower = 0));

priors.onset_admit.shape

# [`composed_parameters_model`](@ref) samples those priors and rebuilds the same
# composer structure, ready to drop into a record model for the likelihood.

@model function param_demo(t, p)
    d ~ to_submodel(composed_parameters_model(t, p))
    return d
end

reconstructed = param_demo(template, priors)();

event_names(reconstructed)

# [`update`](@ref) applies a set of parameter values back to a composed object,
# returning a distribution of the same structure.
# Pair it with [`chain_to_params`](@ref) to read a posterior into the right
# NamedTuple after fitting.

updated = update(template, (onset_admit = (shape = 3.0, scale = 1.5),
    admit_death = (mu = 0.7, sigma = 0.5)));

get_event(updated, :onset_admit)

# ## Summary
#
# - [`compose`](@ref) lowers a NamedTuple, table, or matrix to the same composer
#   stack.
# - [`Sequential`](@ref), [`Parallel`](@ref), [`Competing`](@ref), and
#   [`Select`](@ref) are conjunctive chains, shared-origin branches, competing
#   outcomes, and data-selected disjunctions.
# - The composers nest, including a composer as a chain step and a `select` of a
#   `select`.
# - One object scores records and simulates them; scoring marginalises by row
#   missingness, prediction is the generative `rand`.
# - [`params_table`](@ref), [`build_priors`](@ref),
#   [`composed_parameters_model`](@ref), and [`update`](@ref) attach parameters
#   and priors to the same object.
