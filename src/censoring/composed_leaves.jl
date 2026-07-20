# Make a censored delay a first-class leaf of a ComposedDistributions tree.
#
# The composer walks a tree's leaves to inventory their free parameters
# (`params_table`, `build_priors`) and to rebuild a leaf from new values
# (`update`). A censored leaf must report the parameters of the delay being
# estimated, not of the wrapper: the primary event, the solver method and the
# interval boundaries are fixed structure. These methods forward the composer's
# leaf protocol through the censoring wrappers.
#
# `Weighted` is deliberately absent: a likelihood weight is an observation-side
# wrapper, not a delay a composer estimates.

# Peel to the inner free delay. Recursing (rather than returning `d.dist`) peels
# a stacked wrapper -- the interval-censored view of a truncated,
# primary-censored delay that `double_interval_censored` builds -- in one call.
free_leaf(d::PrimaryCensored) = free_leaf(d.dist)
free_leaf(d::IntervalCensored) = free_leaf(d.dist)

# The inverse: rebuild this wrapper around `inner`, recursing so a stacked
# wrapper is reassembled in the order it was peeled. The primary event, solver
# method and interval boundaries carry over unchanged.
function rewrap_leaf(d::PrimaryCensored, inner)
    return PrimaryCensored(
        rewrap_leaf(d.dist, inner), d.primary_event, d.method)
end

function rewrap_leaf(d::IntervalCensored, inner)
    return IntervalCensored(rewrap_leaf(d.dist, inner), d.boundaries)
end

# A `shared` tag on the inner delay must stay visible through the censoring, or
# the composer inventories a censored copy of a tied delay as its own free
# parameter and estimates the shared delay twice.
_shared_tag(d::PrimaryCensored) = _shared_tag(d.dist)
_shared_tag(d::IntervalCensored) = _shared_tag(d.dist)

# A prior attached to the inner delay (`uncertain`) must likewise stay visible,
# so `params_table`'s `prior` column and `build_priors` pick it up instead of
# treating the parameter as fixed.
_uncertain_specs(d::PrimaryCensored) = _uncertain_specs(d.dist)
_uncertain_specs(d::IntervalCensored) = _uncertain_specs(d.dist)
