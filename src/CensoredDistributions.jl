module CensoredDistributions

# Non-submodule imports
using DocStringExtensions: @template, DOCSTRING, EXPORTS, IMPORTS, TYPEDEF, TYPEDFIELDS,
                           TYPEDSIGNATURES
using Random: AbstractRNG, default_rng

# Explicit imports approach
# Import functions that we extend (for method extension)
import Distributions: params, insupport, pdf, logpdf, cdf, logcdf,
                      ccdf, logccdf, quantile, mean, var, std, median, sampler,
                      loglikelihood
# Import from Base for functions we extend that are re-exported by Distributions
import Base: minimum, maximum
# Use explicit using for types, constructors, and utility functions (no method extension)
using Distributions: Distributions, UnivariateDistribution, Distribution,
                     Continuous, Multivariate, MixtureModel,
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

# Exported censoring functions. `double_censored` is a short, clear alias for
# `double_interval_censored` (NOT `dic`, which clashes with the Deviance
# Information Criterion).
export primary_censored, interval_censored, double_interval_censored,
       double_censored

# Exported latent representation and its inverse
export latent, marginal, PrimaryConditional, primary_conditional_logpdf

# Exported marginal -> latent wrapper over a composed tree and its row deriver.
export latent_segments, latent_records

# Export underlying methods for user extension
export primarycensored_cdf, primarycensored_logcdf

# Exported solver methods for choosing the primary-censoring CDF backend
export AnalyticalSolver, NumericSolver

# Exported distributions
export ExponentiallyTilted

# Exported nonparametric hazard leaf: a delay distribution on [0, ∞) defined by
# a piecewise-constant hazard on a grid. `PiecewiseHazard` is the type;
# `piecewise_hazard` the friendly constructor. The hazard values are the
# differentiable parameters.
export PiecewiseHazard, piecewise_hazard

# Exported difference constructor: the distribution of Z = X - Y for two
# independent components, the dual of the sum `convolve_distributions` builds.
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
export convolve_distributions

# Exported generic composers and front-end constructors. `resolve(...)` builds
# the fixed-probability mixture `Resolve` (a branch probability per outcome);
# `compete(...)` builds the racing-hazard `Compete` (bare delays). `NoEvent`
# marks an absorbing no-event branch; `winning_probabilities` /
# `occurrence_probability` read the per-cause winning / any-event probabilities
# of either node.
export Sequential, Parallel, Resolve, Compete, NoEvent,
       sequential, parallel, compete, resolve,
       compose, as_mixture, winning_probabilities, occurrence_probability

# Exported composed-distribution introspection: the flat prior table and
# name introspection. `event_names` is the FLAT per-event name tuple (the data-
# row key space); `event_tree` is the NESTED tree of event names; `event` fetches
# a child or descends a path. Nested name-keyed values come from the extended
# `Distributions.params`.
export params_table, event_names, event_tree, event, update, build_priors,
       param_priors, default_prior, inspect

# Exported structural edits on a composed tree. `update` (the `path => new_node`
# method, sharing the verb with the value-update NamedTuple method) replaces a
# named node, KEEPING the tree shape. `prune` drops a branch (renormalising a
# Resolve arm) and `splice` inserts a before/after step; these two are the
# TOPOLOGY edits (they change the shape). `intervene` / `swap_child` /
# `cut_branch` are deprecated aliases kept during the deprecation window.
export prune, splice, intervene, swap_child, cut_branch

# Per-event moments come through the standard `Distributions.mean`/`var`/`std`
# interface on the composed tree (a Multivariate distribution), returning a
# Vector in the same per-event layout as `rand(d)` (label with `event_names(d)`).
# `endpoint` collapses a chain to its terminal scalar (an alias for
# `observed_distribution`).
export endpoint

# Exported chain reader: read a fitted Turing chain into the
# nested NamedTuple `update` consumes. No method until DynamicPPL (or Turing) is
# loaded; the method lives in the package extension.
export chain_to_params

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

# Exported right-truncation helpers (single-delay vs convolved-chain), plus the
# δ-bounded variant truncating to a finite observation window.
export truncate_to_horizon, truncate_to_window, truncate_chain

# Exported utilities
export weight, get_dist, get_dist_recursive, get_primary_event

# Exported thinning helpers: completeness / ascertainment thinning,
# Turing-free and distributions-led.
export completeness_probability, thin_by_completeness

# Exported discrete-time reporting-hazard helpers: the epinowcast hazard layer
# that reshapes a delay PMF by logit-scale reference + report effects and forms
# the per-(reference, report) expected-count matrix.
export delay_hazard, hazard_to_pmf, apply_hazard_effects,
       reference_report_matrix

# Exported forward-transform leaves: a deterministic op applied to the count
# series `convolve_distributions` produces (transparent to `logpdf`). `transform`
# is the generic verb; `thin` (scale by a probability) and `cumulative`
# (accumulate) are specialised constructors. `Transformed` is the type.
export Transformed, transform, thin, cumulative

# Exported DynamicPPL submodel constructors. These have no methods until
# DynamicPPL (or Turing) is loaded; the methods live in the package extension so
# the core stays Turing-free.
export primary_censored_model, interval_censored_model,
       double_interval_censored_model, composed_distribution_model,
       composed_parameters_model

# Exported linear chain trick lowering: read the (rate, stages) Erlang-stage
# compartment structure off an Exponential/Erlang delay or Sequential chain, the
# distributions -> compartments bridge an ODE/compartment model consumes.
export linear_chain_stages

# Exported Catalyst reaction-network bridge: slot a composed delay onto a
# transition of a Catalyst reaction network (`linear_chain_reactions`). No
# methods until Catalyst is loaded; the methods live in the package extension so
# the core stays free of the SciML stack. Whole-model assembly (e.g. an SEIR or
# SIR built from composed delays) is application territory; the linear-chain
# tutorial works one through from this bridge.
export linear_chain_reactions

include("docstrings.jl")

include("utils/gamma_ad.jl")

include("integration/integration.jl")

include("censoring/primarycensored_cdf.jl")
include("censoring/PrimaryCensored.jl")
include("censoring/PrimaryConditional.jl")
include("censoring/Latent.jl")
include("censoring/IntervalCensored.jl")
include("censoring/double_interval_censored.jl")

include("distributions/ExponentiallyTilted.jl")
include("distributions/PiecewiseHazard.jl")
include("distributions/Convolved.jl")
# Difference (Z = X - Y), the dual of Convolved. After Convolved.jl since it
# reuses `_window_quantile` / `_CONVOLVED_TAIL` for the quadrature window clamp.
include("distributions/Difference.jl")

# Right-truncation helpers: depend on Convolved / convolve_distributions.
include("censoring/truncation.jl")

include("composers/Sequential.jl")
include("composers/Parallel.jl")
include("composers/Resolve.jl")
# Racing-hazard one_of node (the `min`-of-delays dual of convolve). Split out
# of Resolve.jl; after it since it builds on `AbstractOneOf`/`Resolve`
# helpers (`_n_branches`, `_is_no_event`, `_is_nonterminal`).
include("composers/hazard_one_of.jl")
include("composers/Choose.jl")
include("composers/nesting.jl")
include("composers/equality.jl")
include("composers/compose.jl")
include("composers/introspection.jl")
# Linear chain trick: lower an Exp/Erlang composed delay to its (rate, stages)
# compartment structure. After introspection so it reuses `free_leaf` to peel
# censoring; depends on `Sequential`.
include("composers/bridges/linear_chain.jl")
# Catalyst reaction-network bridge stubs (methods in the Catalyst extension).
# After linear_chain.jl since the extension methods reuse `linear_chain_stages`.
include("composers/bridges/reaction_compartments.jl")
# Affine transform leaf: after introspection so it can extend
# `free_leaf`/`rewrap_leaf` for transparent inner-delay introspection.
include("distributions/Affine.jl")
# Structural edits on a composed tree (`update` node replace / `prune` /
# `splice`): after introspection so it reuses `_rebuild`, `component_names`,
# `_split_edge` and the `update` value method.
include("composers/intervene.jl")
# Shared (name-tagged tied leaf): after introspection so it can extend
# `free_leaf`/`rewrap_leaf`, and before tree_events/wrap which traverse leaves.
include("composers/Shared.jl")
include("composers/tree_events.jl")
include("composers/wrap.jl")

include("utils/Weighted.jl")

# Per-edge delay moments: after Weighted (adds a `free_leaf(::Weighted)` method)
# and the composers (Sequential/Parallel/Resolve/Choose/Latent it walks).
include("composers/composed_moments.jl")

include("utils/get_dist.jl")
include("utils/quantile_optimization.jl")
include("utils/thinning.jl")

# Forward-transform leaves (thin / cumulative): after get_dist (extends it) and
# introspection (extends free_leaf/rewrap_leaf), before the convolve layer that
# applies them.
include("utils/forward_transform.jl")

# Renewal layer: convolve a timeseries through a composed delay stack. After the
# composers/wrap (uses `observed_distribution`, `_observed_leaves`) and
# tree_events (`_flat_event_names`).
include("utils/convolve_with_vector.jl")

# Discrete-time reporting-hazard layer (epinowcast): reshape a delay PMF by
# logit-scale reference + report effects and form the per-(reference, report)
# expected-count matrix. After convolve_with_vector, whose `_delay_pmf` it
# reuses as the baseline PMF.
include("utils/reporting_hazard.jl")

# Censored specialisations of the generic composers: included last
# as they depend on the composers, the censored types, `get_dist_recursive`
# (utils/get_dist.jl) and the integration helpers. Split across cohesive files;
# the shared recovery helpers and the `_Nested`/`_Flat` traits live in
# `censored_specialisations.jl` and so it is included FIRST, then the scoring
# and simulation files that use them (their order between each other is free,
# they only define methods over the already-defined helpers).
include("composers/censored_specialisations.jl")
include("composers/censored_scoring_tree.jl")
include("composers/censored_one_of.jl")
include("composers/censored_scoring_flat.jl")
include("composers/censored_rand.jl")

# Labelled NamedTuple OUTPUTS for multivariate composed distributions: an
# output/interface layer over the vector-valued scored representation. After the
# censored specialisations (`_composer_rand`, `_tree_primary_event`) and
# tree_events (`_row_event_vector`) it wraps, and the composers it names.
include("composers/named_outputs.jl")

# Per-record composed distributions for vectorised scoring + sampling: depends on
# the censored specialisations (`event_logpdf`, `_sequential_segment`,
# `_composer_rand`) and the row-parsing helpers in `tree_events.jl`.
include("composers/record_dists.jl")
# Vectorised LATENT scoring pair (stacked primary priors + vectorised observed
# conditional). Split out of record_dists.jl; after it since it reuses the row
# helpers (`_row_namedtuple`, `_row_event_vector`, `_weight_lp`) and the
# `_alternative_record` Choose build, and defines the shared `_narrow` helper.
include("composers/record_latent.jl")
# The marginal -> latent WRAPPER over a composed tree: `latent(tree)` lowers a
# composer to the per-segment latent `Choose`, and `latent_records` derives the
# per-segment rows the vectorised path scores. After record_latent.jl (it builds
# the `Choose` the vectorised path consumes) and the composer files (it reads the
# tree structure and event names).
include("composers/latent_tree.jl")
# Grouped per-stratum assembly (`record_distributions(ds, rows; group)` and
# `batched_event_logpdf`). Split out of record_dists.jl; after record_latent.jl
# since it reuses `_narrow`, and after record_dists.jl's single-`d`
# `record_distributions` which it dispatches each stratum through.
include("composers/record_grouped.jl")
# Batched record-aware `rand`: the forward-simulation dual to the scoring path.
# After record_dists.jl (the per-record `record_distributions` it draws from) and
# named_outputs.jl (the `_as_named` / `_output_names` labelling it reuses).
include("composers/record_rand.jl")

# Turing-free `primary_censored_model` function stub. Has no methods
# until DynamicPPL is loaded; the methods live in the package extension.
include("turing_models.jl")

# Public interface-conformance harness (a public submodule). Included last so it
# can reference the whole public surface; uses `Test` only inside its functions.
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
