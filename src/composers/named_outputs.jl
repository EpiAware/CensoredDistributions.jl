# ============================================================================
# Labelled NamedTuple outputs for multivariate composed distributions
# ============================================================================
#
# The composed distributions score a vector-valued representation (the flat
# event vector consumed by `logpdf` / `product_distribution` / the AD paths),
# but their user-facing outputs are self-labelling: any multivariate composed
# output is presented as a `NamedTuple` keyed by the relevant names, while a
# univariate (collapsible) output stays a bare scalar. This is purely an
# output/interface layer over the unchanged vector-valued scored representation:
#
#   - `rand(d)` for a flat censored `Sequential`/`Parallel` -> a NamedTuple
#     keyed by the flat `event_names(d)` (the nested-tree `rand` already returns
#     one); `rand(latent(d))` follows the same rule.
#   - `mean(latent(d))` / `var(latent(d))` / `std(latent(d))` -> a NamedTuple
#     over the flat `event_names(d)`.
#   - `mean(parallel)` / `var(parallel)` / `std(parallel)` -> a NamedTuple over
#     the per-endpoint names (the multivariate marginal of a `Parallel`).
#
# A univariate-collapsible composer (`Sequential` chain, `Resolve`, censored
# leaf) keeps its scalar `mean`/`var`/`std`. The internal vector-valued moment /
# realisation builders are untouched; these helpers wrap their result by name.

# --- the wrap boundary ------------------------------------------------------

# Wrap a vector-valued output `v` into a `NamedTuple` keyed by `names`. The
# lengths must agree (an internal invariant; a mismatch is a bug in the name
# derivation). A result that is already a `NamedTuple` (the nested-tree `rand`)
# passes through unchanged.
function _as_named(names::Tuple, v::AbstractVector)
    length(names) == length(v) || throw(DimensionMismatch(
        "labelled output has $(length(v)) values but $(length(names)) names " *
        "$(collect(names))"))
    return NamedTuple{names}(Tuple(v))
end
_as_named(::Tuple, v::NamedTuple) = v

# Label a top-level composer realisation. The internal `_composer_rand` returns
# the flat event/value vector for a flat-censored or plain composer (the nested
# tree path already returns a NamedTuple); this wraps a vector result by the
# composer's output names so a user-facing `rand(d)` is self-labelling. A
# NamedTuple result passes straight through.
function _named_composer_rand(rng::AbstractRNG, d)
    _as_named(
        _output_names(d), _composer_rand(rng, d))
end

# The output names matching the per-event/per-value vector of `rand(d)` /
# `mean(latent(d))`, and the per-record key space the public
# [`event_names`](@ref) reports. A censored composer realises the flat event
# path (origin + one target per leaf edge), so its output names are the flat
# `_flat_event_names(d)`. A plain (uncensored) composer realises the per-value
# vector (one value per leaf, no latent origin), so its output names are the
# per-value leaf names. Calls `_flat_event_names` directly (not the public
# `event_names`, which now delegates back to this) to avoid a cycle.
function _output_names(d::AbstractMultiChild)
    _tree_primary_event(d) === nothing && return _value_names(d)
    return _flat_event_names(d)
end

# Per-value leaf names of a plain (uncensored) composer, in the same depth-first
# layout as `_value_moment_vector` / the generic `_composite_rand`: one name per
# leaf value, a nested composer recursing into its children, a leaf / `Resolve`
# named by its component name. Names from nested levels are joined into a single
# dotted-underscore path (`:r1_step_1`) so a positional default repeated across
# nesting levels (a default `:step_1` in two branches) still yields unique
# NamedTuple keys (a leaf at the top level keeps its bare name).
function _value_names(d::AbstractMultiChild)
    out = Symbol[]
    names = component_names(d)
    for i in eachindex(d.components)
        _append_value_names!(out, (names[i],), d.components[i])
    end
    return Tuple(out)
end

function _append_value_names!(out, path::Tuple,
        child::AbstractMultiChild)
    cnames = component_names(child)
    for i in eachindex(child.components)
        _append_value_names!(out, (path..., cnames[i]), child.components[i])
    end
    return out
end
function _append_value_names!(out, path::Tuple, ::Any)
    push!(out, _join_value_path(path))
    return out
end

# Join a value-name path into one `Symbol`: a single-level path keeps its bare
# name (`:a`); a nested path joins with `_` (`:r1_step_1`). This is the
# underscored ("_" separator) event/value namespace (output value names),
# distinct from the dotted ("." separator) parameter-path namespace
# (`_join_path` / `_split_edge` in introspection.jl).
function _join_value_path(path::Tuple)
    length(path) == 1 ? path[1] :
    Symbol(join(string.(path), "_"))
end

# --- per-endpoint names of a Parallel (the multivariate marginal) -----------
#
# `_endpoint_names(d)` is the tuple of names for the per-endpoint moment vector
# `mean(d::Parallel)` / `var` / `std` produces. It mirrors the
# `_endpoint_moment_vector` walk exactly: one name per collapsed branch
# endpoint, in branch order, a nested `Parallel` flattening its own endpoints
# in. A `Sequential` / `Resolve` / leaf branch collapses to its single
# endpoint and is named by its branch (component) name.

function _endpoint_names(d::Parallel)
    out = Symbol[]
    names = component_names(d)
    for i in eachindex(d.components)
        _append_endpoint_names!(out, names[i], d.components[i])
    end
    return Tuple(out)
end

# A nested `Parallel` branch flattens its own endpoint names in (under the
# nested branch names), mirroring `_append_endpoint_moments!`.
function _append_endpoint_names!(out, ::Symbol, branch::Parallel)
    bnames = component_names(branch)
    for i in eachindex(branch.components)
        _append_endpoint_names!(out, bnames[i], branch.components[i])
    end
    return out
end
# Every other branch (a `Sequential` / `Resolve` / leaf) collapses to one
# endpoint, named by its branch name.
function _append_endpoint_names!(out, name::Symbol, ::Any)
    push!(out, name)
    return out
end

# --- wrong-shape argument: a clear, actionable error ------------------------
#
# A composed `Sequential` / `Parallel` is multivariate: its `logpdf` / `pdf`
# scores a length-k vector (or the matching `NamedTuple` keyed by the composer's
# names), not a bare scalar. Passing a scalar where the per-event vector is
# expected would otherwise hit a confusing low-level `MethodError`; instead name
# the expected form (the length and the keys) so the right shape is discoverable
# at the REPL. A censored composer scores the flat event vector `[E_0, ..., E_k]`
# keyed by `event_names(d)`; a plain composer scores the per-value vector keyed
# by its value names.
function _wrong_shape_error(d::AbstractMultiChild, what, x)
    censored = _tree_primary_event(d) !== nothing
    names = censored ? event_names(d) : _value_names(d)
    kind = d isa Sequential ? "Sequential" : "Parallel"
    shape = censored ?
            "a length-$(length(names)) event vector [E_0, ..., E_k]" :
            "a length-$(length(d)) value vector"
    throw(ArgumentError(
        "$(what)(::$(kind)) expects $(shape), or a NamedTuple keyed by " *
        "$(collect(names)); got a scalar $(typeof(x)). A composed $(kind) is " *
        "multivariate: pass one value per event/step, e.g. " *
        "$(what)(d, rand(d))."))
end

function logpdf(d::AbstractMultiChild, x::Real)
    return _wrong_shape_error(d, "logpdf", x)
end
function pdf(d::AbstractMultiChild, x::Real)
    return _wrong_shape_error(d, "pdf", x)
end

# --- NamedTuple input to logpdf ---------------------------------------------
#
# `logpdf` scores the vector-valued representation; a labelled `NamedTuple` draw
# (as `rand(d)` now returns) is accepted and converted to the scored vector by
# name first, so a self-labelling draw round-trips straight back through
# `logpdf(d, rand(d))`. A censored composer scores the flat event vector
# (matched by name via `_row_event_vector`, a missing required event errors); a
# plain composer scores the per-value vector (matched by its value names). Field
# order does not matter; the names do.

function logpdf(d::AbstractMultiChild, x::NamedTuple)
    # A column table (a `NamedTuple` of vectors, `Tables.istable == true`) is a
    # multi-record source, not a single labelled draw; route it to the public
    # vectorised `logpdf(d, rows)` front-door. A single labelled draw
    # (`Tables.istable == false`) scores its one event/value vector below.
    Tables.istable(x) && return batched_event_logpdf(d, x)
    _tree_primary_event(d) === nothing &&
        return logpdf(d, _named_value_vector(d, x))
    return logpdf(d, _row_event_vector(d, x))
end

# The per-value vector of a plain composer from a labelled draw, matched to the
# `_value_names(d)` layout by name. A plain composer scores the per-value vector
# (no latent origin, no missing events), so this is a plain `Vector{Float64}`
# that routes to the plain `logpdf(::Sequential/Parallel, ::AbstractVector)`
# (not the Missing-admitting censored event scorer).
function _named_value_vector(d::AbstractMultiChild, x::NamedTuple)
    vnames = _value_names(d)
    for k in keys(x)
        k in vnames || throw(ArgumentError(
            "draw field $(repr(k)) is not a value of this composer; expected " *
            "$(collect(vnames)) (reordering is allowed; names are not)"))
    end
    out = Vector{Float64}(undef, length(vnames))
    for (i, name) in enumerate(vnames)
        haskey(x, name) || throw(ArgumentError(
            "draw is missing required value $(repr(name)); expected " *
            "$(collect(vnames))"))
        out[i] = Float64(x[name])
    end
    return out
end
