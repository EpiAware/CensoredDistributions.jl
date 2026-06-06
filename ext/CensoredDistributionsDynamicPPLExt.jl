module CensoredDistributionsDynamicPPLExt

# DynamicPPL methods for the submodel constructors declared (and documented) in
# `src/turing_models.jl`. Loaded only when DynamicPPL is available, keeping the
# core Turing-free.

using CensoredDistributions: CensoredDistributions, PrimaryCensored,
                             IntervalCensored, get_dist, get_primary_event
import CensoredDistributions: primary_censored_model, interval_censored_model,
                              double_interval_censored_model
using DynamicPPL: @model, @addlogprob!
using Distributions: UnivariateDistribution, logpdf

# `CensoredDistributions.weight(d, w)` is called with the module qualifier
# because the `weight` keyword argument shadows the function name inside the
# `@model` bodies. `weight(d, nothing)` returns `d` unweighted, so one `~`
# statement covers both the weighted and unweighted cases.
const _weight = CensoredDistributions.weight

# Marginal: score the marginal logpdf via `~`, with the primary event integrated
# out inside `logpdf`. With `origin` supplied the caller owns the primary event
# time, so score the conditional delay `logpdf(get_dist(d), y - origin)`
# directly: the implied delay is fixed by `origin` (a caller value), so
# `@addlogprob!` adds exactly that conditional contribution without introducing a
# sampled variable.
@model function primary_censored_model(
        d::PrimaryCensored, y; weight = nothing, origin = nothing)
    if origin === nothing
        y ~ _weight(d, weight)
    else
        @addlogprob! logpdf(get_dist(d), y - origin)
    end
    return y
end

@model function interval_censored_model(
        d::IntervalCensored, y; weight = nothing)
    y ~ _weight(d, weight)
    return y
end

# `double_interval_censored` returns a composed univariate distribution (primary
# censoring, optional truncation, optional interval censoring), all marginal, so
# one `~` scores the whole pipeline.
@model function double_interval_censored_model(
        d::UnivariateDistribution, y; weight = nothing)
    y ~ _weight(d, weight)
    return y
end

end
