# CensoredDistributions x ComposedDistributions: make a censored delay a
# first-class leaf of a composed distribution.
#
# ComposedDistributions walks a tree's leaves to inventory their free
# parameters (`params_table`, `build_priors`) and to rebuild a leaf from new
# values (`update`). A leaf that is wrapped in censoring must report the
# parameters of the delay being *estimated*, not of the wrapper: the primary
# event, the solver method and the interval boundaries are fixed structure. The
# hooks below teach the walker to peel a CensoredDistributions wrapper down to
# that inner delay and to rebuild the same wrapper around a new one.
#
# No piracy: `free_leaf` / `rewrap_leaf` are owned by ComposedDistributions and
# every dispatched argument type here is owned by CensoredDistributions, so this
# extension sits exactly at the seam of the two packages that trigger it.
module CensoredDistributionsComposedDistributionsExt

using CensoredDistributions: PrimaryCensored, IntervalCensored

import ComposedDistributions: free_leaf, rewrap_leaf

# `_uncertain_specs` and `_leaf_detail_lines` are ComposedDistributions
# internals rather than part of its public surface, but a censored leaf has to
# recurse through them or a prior attached to the inner delay is lost and the
# leaf's `inspect` output is a raw struct dump. Upstream already ships the same
# recursion for `Truncated`; see the note in the PR about making these public.
import ComposedDistributions: _uncertain_specs, _leaf_detail_lines

# --- peel to the inner free delay ------------------------------------------

# The censoring wrappers are fixed structure, so a censored leaf's free delay is
# whatever its inner distribution peels to. Recursing (rather than returning
# `d.dist`) means a stacked wrapper -- e.g. the interval-censored view of a
# primary-censored delay that `double_interval_censored` builds -- peels all the
# way through in one call.
free_leaf(d::PrimaryCensored) = free_leaf(d.dist)
free_leaf(d::IntervalCensored) = free_leaf(d.dist)

# --- rebuild the same censoring around a new inner delay --------------------

# The inverse of `free_leaf`: rebuild this wrapper around `inner`, recursing so
# that a stacked wrapper is reassembled in the same order it was peeled. The
# primary event, solver method and interval boundaries are carried over
# unchanged -- only the estimated delay is replaced.
function rewrap_leaf(d::PrimaryCensored, inner)
    return PrimaryCensored(
        rewrap_leaf(d.dist, inner), d.primary_event, d.method)
end

function rewrap_leaf(d::IntervalCensored, inner)
    return IntervalCensored(rewrap_leaf(d.dist, inner), d.boundaries)
end

# --- carry the leaf protocols through the wrapper ---------------------------

# A prior attached to the inner delay (`uncertain`) must stay visible once that
# delay is censored, so `params_table`'s `prior` column and `build_priors` pick
# it up instead of treating the parameter as fixed.
_uncertain_specs(d::PrimaryCensored) = _uncertain_specs(d.dist)
_uncertain_specs(d::IntervalCensored) = _uncertain_specs(d.dist)

# `inspect` shows a leaf's detail lines; a censored leaf shows its own compact
# `show`, which already names the wrapper and the delay it censors.
_leaf_detail_lines(d::PrimaryCensored) = split(
    sprint(show, MIME"text/plain"(), d), '\n')
_leaf_detail_lines(d::IntervalCensored) = split(
    sprint(show, MIME"text/plain"(), d), '\n')

end
