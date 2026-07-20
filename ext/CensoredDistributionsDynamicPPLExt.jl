module CensoredDistributionsDynamicPPLExt

# CensoredDistributions x ComposedDistributions x DynamicPPL: `as_turing`
# adapts a PPL-neutral `ComposedDistributions.ComposedLogDensity` spec (from
# `as_logdensity`) into a Turing/DynamicPPL model. Loaded only when
# `DynamicPPL` is available; `ComposedDistributions` is a hard dependency of
# this package (see the `sources` note in Project.toml), so it needs no
# separate trigger.
#
# Unlike a bespoke per-composer-type walk, the model samples generically off
# `params_table(prob.dist)`'s flat row set (`edge`, `param`, `prior`): every
# ESTIMATED row (`prior !== nothing`) becomes one named `tilde_assume!!` site
# `<prefix>.<edge>.<param>`. `params_table` already flattens any composer
# shape (`Sequential`, `Parallel`, `Resolve`, `Compete`, `Choose`, shared,
# pooled...) into that one row set, so `as_turing` needs no dispatch of its
# own on composer node type.

using CensoredDistributions: CensoredDistributions
using ComposedDistributions: ComposedDistributions, ComposedLogDensity,
                             as_logdensity, unflatten, update
using DynamicPPL: DynamicPPL, @model, VarName

# Every `as_turing` definition below is fully qualified
# (`CensoredDistributions.as_turing`), so no bare-name import is needed --
# `using CensoredDistributions: CensoredDistributions` above is enough.

# The estimated (spec'd) rows of `params_table(dist)`, in table order: the
# rows whose `prior` column is not `nothing`. A one-line reimplementation of
# ComposedDistributions' own internal `_estimated_rows` -- that helper is not
# part of the public introspection surface, and this filter is all
# `as_turing` needs from it.
_estimated_row_indices(table) = findall(!isnothing, table.prior)

# One flat parameter's VarName: `<prefix>.<edge>.<param>` as a single dotted
# Symbol (no nested optic). This is the same textual form DynamicPPL's own
# submodel `prefix`ing produces, and what
# `ComposedDistributions.chain_to_params` / `param_draws` read a fitted chain
# back at (see `ComposedDistributionsFlexiChainsExt._dotted`), so a chain
# sampled from this model reads straight onto the template.
function _row_varname(prefix::Symbol, edge::Symbol, param::Symbol)
    return VarName{Symbol(prefix, :., edge, :., param)}()
end

# A pooled parameter's row carries a `CentredPoolPrior` marker rather than an
# ordinary prior distribution (its population-dependent log-prior is scored
# separately from the fixed per-row term; see `logdensity.jl`). Sampling one
# through `tilde_assume!!` would hand DynamicPPL a non-`Distribution` object
# and corrupt the trace, so `as_turing` explicitly does not support a
# centred-pooled tree yet -- error clearly rather than silently scoring the
# wrong target. Tracked as a follow-up.
function _reject_centred_pool(table, idx)
    any(i -> table.prior[i] isa ComposedDistributions.CentredPoolPrior, idx) &&
        throw(ArgumentError(
            "as_turing does not yet support a tree with a centred-pooled " *
            "parameter (see ComposedDistributions.pool); its " *
            "population-dependent prior term is not a plain Distribution " *
            "and cannot be sampled through a single named tilde site"))
    return nothing
end

# Sample every estimated row of `table` as its own named site, threading the
# VarInfo through in table order. Returns the sampled values as a `Tuple`
# (fixed length, one concrete type per slot) rather than a `Vector{Any}`, so
# a reverse-mode AD backend tracing this loop sees a uniform per-slot layout
# -- the same reasoning behind CD's earlier head/tail submodel recursion for
# composer children (a heterogeneous `Vector{Any}` breaks Mooncake reverse).
function _sample_composed_rows(table, idx, prefix::Symbol, ctx, vi)
    vals = ntuple(length(idx)) do i
        row = idx[i]
        vn = _row_varname(prefix, table.edge[row], table.param[row])
        v, vi = DynamicPPL.tilde_assume!!(ctx, table.prior[row], vn, nothing, vi)
        v
    end
    return vals, vi
end

# The generic model body: sample every estimated row, reconstruct the tree,
# score the data. `as_turing` and its convenience forms below are the only
# public entry points; this is the one `@model` they all build.
@model function _composed_turing_model(prob::ComposedLogDensity; prefix = :d)
    table = ComposedDistributions.params_table(prob.dist)
    idx = _estimated_row_indices(table)
    _reject_centred_pool(table, idx)
    vals,
    __varinfo__ = _sample_composed_rows(
        table, idx, prefix, __model__.context, __varinfo__)
    d = update(prob.dist, unflatten(prob.dist, collect(vals)))
    DynamicPPL.@addlogprob! prob.loglik(d, prob.data)
    return d
end

# The headline adaptor: build the model straight from the assembled spec.
function CensoredDistributions.as_turing(prob::ComposedLogDensity; prefix = :d)
    return _composed_turing_model(prob; prefix = prefix)
end

# Convenience forms mirroring `as_logdensity`'s signatures: assemble the spec
# first (default priors = the tree's own `uncertain` specs, default `loglik`
# = summed per-record `logpdf`), then adapt it. Keywords other than `prefix`
# forward to `as_logdensity`, which owns those defaults.
function CensoredDistributions.as_turing(
        dist, priors, data; prefix = :d, kwargs...)
    return CensoredDistributions.as_turing(
        as_logdensity(dist, priors, data; kwargs...); prefix = prefix)
end

function CensoredDistributions.as_turing(dist, data; prefix = :d, kwargs...)
    return CensoredDistributions.as_turing(
        as_logdensity(dist, data; kwargs...); prefix = prefix)
end

end
