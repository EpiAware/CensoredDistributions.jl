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

using CensoredDistributions: CensoredDistributions, PrimaryCensored,
                             IntervalCensored, MarginalPrimaryCensored,
                             LatentPrimaryCensored, get_dist, get_primary_event,
                             weight
# Import the constructor names so the `@model` definitions add methods to the
# core declarations rather than shadowing them.
import CensoredDistributions: primary_censored_model, interval_censored_model,
                              double_interval_censored_model
using DynamicPPL: DynamicPPL, @model
using Distributions: UnivariateDistribution

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
@model function primary_censored_model(
        d::LatentPrimaryCensored, y; weight = nothing)
    p ~ get_primary_event(d)
    delay = y - p
    delay ~ get_dist(d)
    return (; p, y)
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

# ----------------------------------------------------------------------------
# STUB (slots in with the composed-grammar stack #309/#317/#318/#320)
# ----------------------------------------------------------------------------
# Latent composed types (sequential / parallel vector-of-delays / event tree /
# Competing) recurse into child submodels, each reading its own node's resolved
# mode and `to_submodel`-ing the appropriate child. Those engine types do not
# exist on this base branch yet; the methods land once the stack is rebased in.
# The single-primary latent path above is the load-bearing piece for now.

end # module
