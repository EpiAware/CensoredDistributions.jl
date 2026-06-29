module CensoredDistributions

# Non-submodule imports
using DocStringExtensions: @template, DOCSTRING, EXPORTS, IMPORTS, TYPEDEF, TYPEDFIELDS,
                           TYPEDSIGNATURES
using Random: AbstractRNG, default_rng

# Explicit imports approach
# Import functions that we extend (for method extension)
import Distributions: params, insupport, pdf, logpdf, cdf, logcdf,
                      ccdf, logccdf, quantile, mean, var, std, median, sampler,
                      loglikelihood, probs
# Import from Base for functions we extend that are re-exported by Distributions
import Base: minimum, maximum
# Use explicit using for types, constructors, and utility functions (no method extension)
using Distributions: Distributions, UnivariateDistribution, Distribution,
                     Continuous, Multivariate, Univariate, VariateForm,
                     MixtureModel,
                     ValueSupport, Truncated, Product, Censored, truncated,
                     product_distribution, Exponential, Gamma, LogNormal, Uniform,
                     Weibull, Normal, shape, scale, meanlogx, stdlogx,
                     _in_closed_interval

using PrecompileTools: @setup_workload, @compile_workload

using LogExpFunctions: logsubexp, log1mexp

using SpecialFunctions: gamma, gamma_inc, loggamma, digamma

import Tables

import FastGaussQuadrature  # provides Gauss-Legendre nodes for the default solver

using Optimization: OptimizationFunction, OptimizationProblem, solve, ReturnCode

using OptimizationOptimJL: NelderMead

# Exported censoring functions.
export primary_censored, interval_censored, double_interval_censored

# Exported latent representation and its inverse
export latent, marginal, PrimaryConditional

# Exported marginal -> latent wrapper over a composed tree and its row deriver.
export latent_segments, latent_records

# Export underlying methods for user extension
export primarycensored_cdf, primarycensored_logcdf

# Exported solver methods for choosing the primary-censoring CDF backend
export AnalyticalSolver, NumericSolver

# Exported distributions
export ExponentiallyTilted

# Exported moment-parameterisation wrapper: parameterises any registered family
# by its moments / alternative parameters (e.g. a Gamma by `(mean, shape)`, the
# scale derived), so a prior on a derived quantity couples correctly through the
# prior front-door where the native parameterisation cannot. `from_moments` is
# the front-end; `register_moment_params` adds a family; `MomentParams` is the
# type (exported for dispatch / extension).
export MomentParams, from_moments, register_moment_params

# Exported hazard-modified distribution: modify the hazard of a base delay
# through a link, `h*(t) = g⁻¹(g(h(t)) + effect)`. `modify` is the verb;
# `Modified` the type. The named links `LogLink` (proportional hazards, the
# default), `IdentityLink` (additive hazards) and `LogitLink` (discrete-time
# reporting hazard) are exported markers; `hazard_link` wraps an arbitrary
# invertible callable.
export modify, Modified, LogLink, IdentityLink, LogitLink, hazard_link

# Exported difference constructor: the distribution of Z = X - Y for two
# independent components, the dual of the sum `convolved` builds.
# Z has two-sided (possibly negative) support, so it is a derived observation,
# not a non-negative delay leaf. `Difference` is the type; `difference` the
# friendly constructor.
export Difference, difference

# Exported affine transform: a deterministic shift+scale of a delay, nesting as
# a leaf. `Affine` is the type; `affine` the friendly constructor.
export Affine, affine

# Exported convolution constructor. A method on a numeric-vector second argument
# also provides the renewal layer (convolve a timeseries through a composed delay
# stack to selected event series), so it needs no separate export.
export convolved

# Exported renewal recurrence: the renewal scan and its composable force
# modulators. `renewal` runs `I[t] = R_t m(t) Σ_s g_s I[t-s]` as a forward scan;
# the modulators (`NoModulation`, `susceptibility_depletion`, `transmissibility`,
# `immunity_waning`) compose via `combine_modulators`; `observe_renewal` reports
# the infections through an observation delay.
export renewal, NoModulation, susceptibility_depletion, transmissibility,
       immunity_waning, combine_modulators, observe_renewal

# Exported generic composers and front-end constructors. `resolve(...)` builds
# the fixed-probability mixture `Resolve` (a branch probability per outcome);
# `compete(...)` builds the racing-hazard `Compete` (bare delays). `NoEvent`
# marks an absorbing no-event branch; `Distributions.probs` (extended, not
# re-exported) reads the per-outcome split of either node and
# `occurrence_probability` its sum (the any-event probability).
export Sequential, Parallel, Resolve, Compete, NoEvent,
       sequential, parallel, compete, resolve,
       compose, as_mixture, occurrence_probability

# Exported composed-distribution introspection: the flat prior table and
# name introspection. `event_names` is the flat per-event name tuple (the data-
# row key space); `event_tree` is the nested tree of event names; `event` fetches
# a child or descends a path. Nested name-keyed values come from the extended
# `Distributions.params`.
export params_table, event_names, event_tree, event, update, build_priors,
       param_priors, default_prior, inspect

# The PPL-neutral log-density layer — the flat-vector <-> nested-
# NamedTuple codec (`flatten` / `unflatten` / `flat_dimension`, ordered by
# `params_table`), the `as_logdensity` assembler and the `ComposedLogDensity`
# spec — is public but not exported: the generic `flatten` / `unflatten` names
# would otherwise occupy the top-level namespace and clash with the
# `Iterators.flatten` mental model, so it is reached by the qualified name
# (`CensoredDistributions.flatten`). The `public` declarations live in
# `public.jl` (guarded for Julia >= 1.11). The LogDensityProblems /
# DensityInterface / Bijectors glue lives in weakdep extensions; the codec and
# spec stay core and Turing-free.

# Exported structural edits on a composed tree. `update` (the `path => new_node`
# method, sharing the verb with the value-update NamedTuple method) replaces a
# named node, keeping the tree shape. `prune` drops a branch (renormalising a
# Resolve arm) and `splice` inserts a before/after step; these two are the
# topology edits (they change the shape).
export prune, splice

# Exported chain reader: read a fitted Turing chain into the
# nested NamedTuple `update` consumes. No method until DynamicPPL (or Turing) is
# loaded; the method lives in the package extension.
export chain_to_params

# Exported vectorised chain reader: read every draw of a fitted chain into a
# vector of parameter NamedTuples (one per draw), replacing per-draw
# `chain_to_params(...; draw = i)` loops. No method until both DynamicPPL and
# FlexiChains are loaded; the method lives in the package extension.
export param_draws

# Exported chain renamer: drop the outer submodel prefix from a fitted chain's
# parameter names (`d.onset_admit.shape` -> `onset_admit.shape`). No method until
# both DynamicPPL and FlexiChains are loaded; the method lives in the extension.
export strip_prefix

# Exported data-selected disjunction node (the case selector over independent
# alternatives). `Choose` is the type; `choose` the friendly constructor.
export Choose, choose

# Exported shared-parameter tag: tie a leaf across branches by name so the
# prior/params interface treats its occurrences as one free parameter. `Shared`
# is the type; `shared` the leaf-local constructor; `tie` the tree-level,
# path-based spelling of the same tie.
export Shared, shared, tie

# Exported composer-observed lowering used by the external censoring wrappers
export observed_distribution

# Exported utilities. `weight` is exported but DEPRECATED (issue #128): the
# weighting surface moves to the standalone ModifiedDistributions.jl package in
# a future breaking release. It stays functional and warns under `--depwarn`.
export weight, get_dist, get_dist_recursive, get_primary_event

# Exported discrete-time reporting-hazard helpers: the epinowcast hazard layer
# that reshapes a delay PMF by logit-scale reference + report effects and forms
# the per-(reference, report) expected-count matrix.
export delay_hazard, hazard_to_pmf, apply_hazard_effects,
       reference_report_matrix

# Exported forward-transform leaves: a deterministic op applied to the count
# series `convolved` produces (transparent to `logpdf`). `transform`
# is the generic verb; `thin` (scale by a probability) and `cumulative`
# (accumulate) are specialised constructors. `Transformed` is the type.
export Transformed, transform, thin, cumulative

# Exported DynamicPPL submodel constructors. These have no methods until
# DynamicPPL (or Turing) is loaded; the methods live in the package extension so
# the core stays Turing-free.
export primary_censored_model, interval_censored_model,
       double_interval_censored_model, composed_distribution_model,
       composed_parameters_model, renewal_model

# Exported recurrent / cyclic multi-state model. `recur` builds the renewal-
# over-states (semi-Markov) default; `ctmc` the memoryless generator-matrix fast
# path. `RecurrentStates` / `CTMCStates` are the types; `StatePath` the path
# record `rand` returns and `logpdf` scores. `recurrent_states_model` is the
# Turing glue stub (no method until DynamicPPL is loaded).
export recur, ctmc, RecurrentStates, CTMCStates, StatePath,
       recurrent_states_model

# Exported linear chain trick lowering: read the (rate, stages) Erlang-stage
# compartment structure off an Exponential/Erlang delay or Sequential chain, the
# distributions -> compartments bridge an ODE/compartment model consumes.
export compartment_stages

# Exported Catalyst reaction-network bridge: slot a composed delay onto a
# transition of a Catalyst reaction network (`linear_chain_reactions`). No
# methods until Catalyst is loaded; the methods live in the package extension so
# the core stays free of the SciML stack. Whole-model assembly (e.g. an SEIR or
# SIR built from composed delays) is application territory; the linear-chain
# tutorial works one through from this bridge.
export linear_chain_reactions

include("docstrings.jl")

# Abstract type hierarchy for the composer nodes and modifier leaves. Included
# first so every concrete type can subtype the abstracts (and `AbstractOneOf`
# can re-root under `AbstractComposedDistribution`).
include("interface.jl")

include("utils/gamma_ad.jl")

include("integration/integration.jl")

include("censoring/primarycensored_cdf.jl")
include("censoring/PrimaryCensored.jl")
include("censoring/PrimaryConditional.jl")
include("censoring/Latent.jl")
include("censoring/IntervalCensored.jl")
include("censoring/double_interval_censored.jl")

include("distributions/ExponentiallyTilted.jl")
# Moment-parameterisation leaf; reconstruction hooks in introspection.jl.
include("distributions/MomentParams.jl")
include("distributions/Convolved.jl")
# Difference (Z = X - Y); reuses Convolved's quadrature-window helpers.
include("distributions/Difference.jl")

# Right-truncation helpers; depend on Convolved.
include("censoring/truncation.jl")

include("composers/Sequential.jl")
include("composers/Parallel.jl")
include("composers/Resolve.jl")
# Racing-hazard one_of node; builds on Resolve helpers.
include("composers/hazard_one_of.jl")
include("composers/Choose.jl")
include("composers/nesting.jl")
include("composers/equality.jl")
include("composers/compose.jl")
include("composers/introspection.jl")
# Flat-vector <-> nested-NamedTuple codec and the ComposedLogDensity spec.
include("composers/logdensity.jl")
# Linear chain trick: lower an Exp/Erlang delay to its (rate, stages) structure.
include("composers/bridges/linear_chain.jl")
# Catalyst reaction-network bridge stubs (methods in the Catalyst ext).
include("composers/bridges/reaction_compartments.jl")
# Affine transform leaf.
include("distributions/Affine.jl")
# Monotone operational-time warp leaf.
include("distributions/TimeChange.jl")
# Structural edits on a composed tree (update / prune / splice).
include("composers/intervene.jl")
# Shared (name-tagged tied leaf).
include("composers/Shared.jl")
include("composers/tree_events.jl")
# Per-record `:field` modifier carrier type + resolver helpers.
include("composers/per_record_fields.jl")
include("composers/wrap.jl")

include("utils/Weighted.jl")

# Per-edge delay moments.
include("composers/composed_moments.jl")

include("utils/get_dist.jl")
include("utils/quantile_optimization.jl")
include("utils/thinning.jl")

# Forward-transform leaves (thin / cumulative).
include("utils/forward_transform.jl")

# Renewal layer: convolve a timeseries through a composed delay stack.
include("utils/convolve_with_vector.jl")

# Renewal recurrence: the renewal step as a composable forward scan.
include("utils/renewal.jl")

# Discrete-time reporting-hazard layer (epinowcast).
include("utils/reporting_hazard.jl")

# Hazard-modified distribution leaf (modify / Modified).
include("distributions/Modified.jl")

# Censored specialisations of the generic composers; included last. The shared
# helpers live in censored_specialisations.jl, so it is included first.
include("composers/censored_specialisations.jl")
# Interval/truncation-aware secondary conditional of PrimaryConditional.
include("censoring/secondary_conditional.jl")
include("composers/censored_scoring_tree.jl")
include("composers/censored_one_of.jl")
include("composers/censored_scoring_flat.jl")
include("composers/censored_rand.jl")

# Labelled NamedTuple outputs for multivariate composed distributions.
include("composers/named_outputs.jl")

# Per-record composed distributions for vectorised scoring + sampling.
include("composers/record_dists.jl")
# Vectorised latent scoring pair (stacked primary priors + conditional).
include("composers/record_latent.jl")
# The marginal -> latent wrapper over a composed tree (latent(tree)).
include("composers/latent_tree.jl")
# Grouped per-stratum assembly (record_distributions(...; group)).
include("composers/record_grouped.jl")
# Per-record `:field` modifier resolution + scoring.
include("composers/per_record_fields_scoring.jl")
# Batched record-aware `rand`: the forward-simulation dual to scoring.
include("composers/record_rand.jl")

# Hazard accessors from a composed tree (hazard / loghazard / cumhazard /
# survival). The SurvivalDistributions extension aligns these with its verbs.
include("utils/hazards.jl")

# Recurrent / cyclic multi-state: semi-Markov default and CTMC fast path.
include("composers/recurrent/RecurrentStates.jl")
include("composers/recurrent/path_io.jl")
include("composers/recurrent/CTMCStates.jl")
include("composers/recurrent/turing.jl")

# Turing-free `primary_censored_model` stub (methods in the extension).
include("turing_models.jl")

# Public interface-conformance harness (a public submodule).
include("TestUtils.jl")

# Public API - functions that are part of public interface but not exported
@static if VERSION >= v"1.11"
    include("public.jl")
else
    # Julia 1.10 compatibility - no public keyword, but structs are accessible
end

# Precompile workload covering the double_interval_censored pipeline for
# representative delay distributions, toggling the solver method to hit both
# the analytical and numeric primary-censored CDF paths in a single entry
# point. See https://github.com/EpiAware/CensoredDistributions.jl/issues/212.
@setup_workload begin
    delays = (
        Gamma(2.0, 1.5),
        LogNormal(1.5, 0.75),
        Weibull(2.0, 1.5),
        Exponential(1.5)
    )
    primary = Uniform(0.0, 1.0)
    x = 2.5

    @compile_workload begin
        for d in delays
            for method in (AnalyticalSolver(), NumericSolver())
                dic = double_interval_censored(
                    d; primary_event = primary, upper = 10.0,
                    interval = 1.0, method = method)
                cdf(dic, x)
                logcdf(dic, x)
                pdf(dic, x)
                logpdf(dic, x)
            end
        end
    end
end

end
