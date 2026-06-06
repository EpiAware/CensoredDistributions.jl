module CensoredDistributionsDynamicPPLExt

# DynamicPPL weak-dependency extension providing `@model` submodels for the
# censored distributions. The base package stays Turing-free: these submodels
# load only when the user brings in `DynamicPPL` (or `Turing`, which re-exports
# it).
#
# The submodel constructors (`primary_censored_model`, `interval_censored_model`,
# `double_interval_censored_model`) are declared in the core package
# (`src/turing_models.jl`) with no methods and full docstrings; this extension
# imports them and adds the `@model` methods.
#
# Design (issue #88, locked):
#
# * The MARGINAL path needs no special machinery. `logpdf(d, y)` on a marginal
#   object already integrates the primary out internally, is AD-safe, and is
#   sampler-agnostic, so the marginal submodel simply scores it via a `~`
#   statement. Weighting goes through the `weight` distribution wrapper, which
#   is itself a distribution (`logpdf(weight(d, w), y) == w * logpdf(d, y)` and
#   `rand(weight(d, w))` draws from `d`), so it drops into a `~` natively with
#   no `@addlogprob!` hack.
#
# * The LATENT path is the only reason this extension exists. The latent
#   primary event is declared with its own `~` INSIDE the submodel, so it stays
#   in the submodel scope and never appears in the user's model. The submodel
#   then scores the conditional delay and returns the internal event times for
#   parent plumbing.
#
# * The PRIMARY owns the marginal-versus-latent decision. The pure struct
#   carries the resolved `mode` as a type parameter (#316 `_resolve_mode` /
#   `_can_marginalise` auto-fallback). These submodels DISPATCH on that resolved
#   type; they do not re-decide. A user switches a whole model between marginal
#   and latent by flipping ONLY the `mode` on `d` when building it — no other
#   code changes, because the right submodel method is selected automatically.

using CensoredDistributions: CensoredDistributions,
                             IntervalCensored, MarginalPrimaryCensored,
                             LatentPrimaryCensored, ParallelPrimaryCensored,
                             SequentialDistribution, EventTree, Competing,
                             as_mixture, get_dist,
                             get_primary_event, weight
# Import the constructor names so the `@model` definitions add methods to the
# core declarations rather than shadowing them.
import CensoredDistributions: primary_censored_model, interval_censored_model,
                              double_interval_censored_model
using DynamicPPL: DynamicPPL, @model, to_submodel, @addlogprob!
using Distributions: UnivariateDistribution, logpdf

# ============================================================================
# Marginal weighting helper
# ============================================================================

# Wrap `d` in the `weight` distribution when a multiplicity weight is supplied,
# so the marginal submodels score `w * logpdf(d, y)` through a native `~`. A
# `nothing` weight scores the unweighted distribution. Using `weight` (rather
# than `@addlogprob! w * logpdf`) keeps everything inside the `~` accumulation,
# which is the general extensible wrapper the package standardises on.
_weighted(d, ::Nothing) = d
_weighted(d, w::Real) = weight(d, w)

# ============================================================================
# primary_censored_model
# ============================================================================

# Marginal node: score the pure marginal logpdf through `~`, weighted via the
# `weight` wrapper when a multiplicity weight is supplied. The primary is
# integrated out inside `logpdf(d, y)`; no latent is exposed. See the core
# `primary_censored_model` docstring for the public documentation.
@model function primary_censored_model(
        d::MarginalPrimaryCensored, y; weight = nothing)
    y ~ _weighted(d, weight)
    return y
end

# Latent node: declare the latent primary INSIDE the submodel so it stays in
# submodel scope, then score the conditional delay. The observed delay satisfies
# `y = p + delay`, so the implied delay `y - p` is scored against the delay
# distribution. The latent `p` is owned by the sampler (drawn from the primary
# event prior) and never appears in the user's model. The internal event times
# are returned for parent plumbing (the return-value form is the locked
# open-question default for latent chains/trees).
#
# Scoring the implied delay `y - p` against the delay distribution (rather than
# `y` against a shifted delay) keeps the contribution a plain `logpdf` of the
# delay, matching the `Latent` joint
# `logpdf(primary_event, p) + logpdf(delay, y - p)` (see the pure
# `_latent_logpdf`). `weight` is ignored on the latent path: each record has its
# own latent, so multiplicities cannot be aggregated; latent models vectorise
# over records instead.
#
# Coupling escape-hatch (`origin`): when the caller supplies an external origin
# the submodel does NOT draw the dist's own primary-event prior. Instead it
# scores the conditional delay `y - origin` against the delay distribution,
# leaving the prior over `origin` to the caller's model. This is the deliberate
# coupled case (for example hanta: a source's onset feeds an offspring's
# infection, sampled in the user's loop): the user owns the coupled prior, the
# submodel owns only the conditional-delay mechanics. With `origin = nothing`
# (the default) behaviour is unchanged: the submodel draws `p` and owns the
# prior. The `origin` form does not double-count the prior, so a caller who has
# already scored the origin gets exactly one prior contribution.
@model function primary_censored_model(
        d::LatentPrimaryCensored, y; weight = nothing, origin = nothing)
    if origin === nothing
        p ~ get_primary_event(d)
        delay = y - p
        delay ~ get_dist(d)
        return (; p, y)
    else
        # Injected coupled origin: there is no latent to draw here (the caller
        # owns the prior over `origin`), only the conditional-delay likelihood
        # `logpdf(delay, y - origin)`. Score it with `@addlogprob!` rather than a
        # tilde: the implied delay is a deterministic function of the injected
        # `origin` (a kwarg, not a model variable), so a `delay ~` statement
        # would be parsed as a fresh latent to sample. `@addlogprob!` adds the
        # exact conditional log-density and nothing else, so the prior is never
        # double-counted.
        @addlogprob! logpdf(get_dist(d), y - origin)
        return (; p = origin, y)
    end
end

# ============================================================================
# interval_censored_model
# ============================================================================

# Interval censoring is a univariate marginal operation, so score `logpdf(d, y)`
# through `~`, weighted via the `weight` wrapper when supplied.
@model function interval_censored_model(
        d::IntervalCensored, y; weight = nothing)
    y ~ _weighted(d, weight)
    return y
end

# ============================================================================
# double_interval_censored_model
# ============================================================================

# Marginal composed pipeline: score the composed univariate logpdf via `~`. The
# composed object is an `IntervalCensored`/`Truncated`/`PrimaryCensored` stack,
# all univariate under the marginal default, so a single `~` covers the whole
# pipeline.
@model function double_interval_censored_model(
        d::UnivariateDistribution, y; weight = nothing)
    y ~ _weighted(d, weight)
    return y
end

# ============================================================================
# Composed-type submodels (the grammar engine types #317/#309/#318/#320)
# ============================================================================
#
# The composed engine types are data-driven multivariate distributions: the
# marginal-versus-latent decision for each interior node is carried by the
# OBSERVATION VECTOR, not by a type parameter. A `Missing` entry marks a node
# that is integrated out inside `logpdf` (the marginal mechanic); a concrete
# entry is conditioned on. This is the per-node mode living on the data the
# struct is evaluated against, the multivariate counterpart of the `mode` type
# parameter on the single `PrimaryCensored`. The submodels honour that pattern:
#
# * the MARGINAL mechanic scores the observation vector (with its `Missing`
#   entries) through the pure `logpdf`, so every latent node is integrated inside
#   that `logpdf` and nothing is exposed to the sampler;
# * the LATENT mechanic samples the latent (would-be-`Missing`) interior nodes
#   INSIDE the submodel, scores the fully-resolved joint, and returns the
#   internal node times for parent plumbing.
#
# Weighting: the composed types are multivariate, so the univariate `weight`
# wrapper does not apply. A multiplicity weight is therefore applied with
# `@addlogprob!` scaling the joint `logpdf` (the documented fallback for the
# non-univariate marginal path). The default is `nothing`: add the plain
# `logpdf`.

# Score a multivariate composed-type observation, with an optional multiplicity
# weight. The composed engine types implement `logpdf(d, obs)` (the per-record
# missingness dispatch) but not `loglikelihood`, and they are multivariate so the
# univariate `weight` wrapper does not apply; the contribution is therefore added
# with `@addlogprob!`. This is the documented multivariate counterpart of the
# univariate `~`/`weight` marginal path. A `nothing` weight adds the plain
# `logpdf`; a real weight scales it by the multiplicity.
@model function _score_multivariate(d, obs; weight = nothing)
    w = weight === nothing ? 1 : weight
    @addlogprob! w * logpdf(d, obs)
    return obs
end

# ----------------------------------------------------------------------------
# Sequential chain (`SequentialDistribution`, #309)
# ----------------------------------------------------------------------------

# `SequentialDistribution` is data-free and multivariate: its `logpdf` takes the
# per-event observation vector and, per record, marginalises the runs of delays
# spanning `Missing` (unobserved) events and conditions on the observed ones.
# Scoring its `logpdf` therefore reads each node's mode straight off the vector's
# missingness — observed nodes are conditioned, missing intermediate nodes are
# integrated inside `logpdf`. This is the marginal mechanic for the whole chain.
@model function primary_censored_model(
        d::SequentialDistribution, obs; weight = nothing)
    result ~ to_submodel(_score_multivariate(d, obs; weight = weight), false)
    return result
end

# ----------------------------------------------------------------------------
# Parallel shared-origin branches (`ParallelPrimaryCensored`, #317)
# ----------------------------------------------------------------------------

# `ParallelPrimaryCensored` is multivariate over `[primary, observed...]` with
# one shared latent origin. The observation vector's first entry is the primary:
# `missing` marginalises it (the marginal mechanic, one origin integral inside
# `logpdf`); a concrete value conditions on it.
#
# `observed` is the length-`n` branch-observation vector (entries may themselves
# be `Missing` to drop a branch).
#
# Marginal mechanic (`latent = false`, the default): score
# `[missing, observed...] ~ d`, integrating the shared origin inside `logpdf`.
#
# Latent mechanic (`latent = true`): sample the shared origin INSIDE the
# submodel, then score each present branch's implied delay `observed_i - p`
# against that branch's delay distribution. The branches are coupled through the
# single shared `p`, so they are scored against one sampled origin (not
# independent per-branch origins). The internal origin time is returned for
# parent plumbing. `weight` is ignored on this latent path (as for the single
# `LatentPrimaryCensored`): each record has its own latent origin, so
# multiplicities cannot be aggregated; latent models vectorise over records.
@model function primary_censored_model(
        d::ParallelPrimaryCensored, observed; weight = nothing,
        latent::Bool = false)
    if latent
        p ~ get_primary_event(d)
        branches = d.delays
        # Only the shared origin `p` is latent; each branch's implied delay
        # `observed[i] - p` is a DETERMINISTIC observation given `p` and the
        # data, so the branches are scored as likelihood contributions, not
        # resampled. A naive `delay ~ branches[i]` loop is wrong twice over:
        # the reused `delay` collapses every branch onto a single VarName, and
        # (because the loop LHS is a freshly bound local rather than model data)
        # DynamicPPL parses each `~` as a fresh latent to SAMPLE, so earlier
        # branches are overwritten and not scored. Adding each present branch's
        # conditional log-density with `@addlogprob!` scores every branch
        # exactly once and keeps `p` the single, cleanly named latent. The joint
        # log-density is therefore the origin prior plus the sum over present
        # branches, matching the pure `ParallelPrimaryCensored` conditional
        # `logpdf([p, observed...])`.
        for i in eachindex(branches)
            observed[i] === missing && continue
            @addlogprob! logpdf(branches[i], observed[i] - p)
        end
        return (; p, observed)
    else
        obs = vcat(missing, observed)
        result ~ to_submodel(
            _score_multivariate(d, obs; weight = weight), false)
        return result
    end
end

# ----------------------------------------------------------------------------
# Event tree (`EventTree`, #318) and competing nodes (`Competing`, #320)
# ----------------------------------------------------------------------------

# `EventTree` is multivariate and data-driven: its `logpdf` takes one time per
# event (keyed by `event_names(d)`), marginalising the `Missing` (latent)
# interior nodes by nested integrals — a node shared by several edges is
# integrated once — and conditioning on the observed ones. Competing
# (disjunctive) nodes lower to a `MixtureModel` via `as_mixture` and are
# marginalised over the branch by the tree `logpdf` (the realised branch is
# selected from the observed outcome). Scoring its `logpdf` therefore honours
# each node's mode straight off the observation's missingness and handles the
# competing branches, the recursion over edges, and the shared interior nodes
# inside the pure `logpdf`. This is the marginal mechanic for the whole tree.
@model function primary_censored_model(
        d::EventTree, obs; weight = nothing)
    result ~ to_submodel(_score_multivariate(d, obs; weight = weight), false)
    return result
end

# A `Competing` node on its own (outside a tree) lowers to its `MixtureModel`
# over the branch delays weighted by the branch probabilities. The marginal
# mechanic marginalises the branch: score the resolution gap against the
# mixture. The realised-branch (latent-category) mechanic is selected by the
# tree from the observed outcome, so a stand-alone competing submodel here only
# needs the marginalised mixture form.
@model function primary_censored_model(
        c::Competing, gap; weight = nothing)
    mixture = as_mixture(c)
    gap ~ _weighted(mixture, weight)
    return gap
end

end # module
