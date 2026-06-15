# ============================================================================
# Labelled NamedTuple OUTPUTS for multivariate composed distributions
# ============================================================================
#
# The composed distributions SCORE a vector-valued representation (the flat
# event vector consumed by `logpdf` / `product_distribution` / the AD paths),
# but their user-facing OUTPUTS are self-labelling: any MULTIVARIATE composed
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
#     the per-ENDPOINT names (the multivariate marginal of a `Parallel`).
#
# A univariate-collapsible composer (`Sequential` chain, `Competing`, censored
# leaf) keeps its scalar `mean`/`var`/`std`. The internal vector-valued moment /
# realisation builders are untouched; these helpers wrap their result by name.

# --- the wrap boundary ------------------------------------------------------

# Wrap a vector-valued OUTPUT `v` into a `NamedTuple` keyed by `names`. The
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
# the flat event/value VECTOR for a flat-censored or plain composer (the nested
# tree path already returns a NamedTuple); this wraps a vector result by the
# composer's output names so a user-facing `rand(d)` is self-labelling. A
# NamedTuple result passes straight through.
function _named_composer_rand(rng::AbstractRNG, d)
    _as_named(
        _output_names(d), _composer_rand(rng, d))
end

# The output names matching the per-event/per-value vector of `rand(d)` /
# `mean(latent(d))`. A CENSORED composer realises the flat EVENT path (origin +
# one target per leaf edge), so its output names are the flat `event_names(d)`.
# A PLAIN (uncensored) composer realises the per-VALUE vector (one value per
# leaf, no latent origin), so its output names are the per-value leaf names.
function _output_names(d::Union{Sequential, Parallel})
    _tree_primary_event(d) === nothing && return _value_names(d)
    return event_names(d)
end

# Per-VALUE leaf names of a plain (uncensored) composer, in the same depth-first
# layout as `_value_moment_vector` / the generic `_composite_rand`: one name per
# leaf value, a nested composer recursing into its children, a leaf / `Competing`
# named by its component name. Names from nested levels are JOINED into a single
# dotted-underscore path (`:r1_step_1`) so a positional default repeated across
# nesting levels (a default `:step_1` in two branches) still yields UNIQUE
# NamedTuple keys (a leaf at the top level keeps its bare name).
function _value_names(d::Union{Sequential, Parallel})
    out = Symbol[]
    names = component_names(d)
    for i in eachindex(d.components)
        _append_value_names!(out, (names[i],), d.components[i])
    end
    return Tuple(out)
end

function _append_value_names!(out, path::Tuple,
        child::Union{Sequential, Parallel})
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
# name (`:a`); a nested path joins with `_` (`:r1_step_1`).
function _join_value_path(path::Tuple)
    length(path) == 1 ? path[1] :
    Symbol(join(string.(path), "_"))
end

# --- per-ENDPOINT names of a Parallel (the multivariate marginal) -----------
#
# `_endpoint_names(d)` is the tuple of names for the per-endpoint moment vector
# `mean(d::Parallel)` / `var` / `std` produces. It mirrors the
# `_endpoint_moment_vector` walk exactly: one name per collapsed branch
# endpoint, in branch order, a nested `Parallel` flattening its own endpoints
# in. A `Sequential` / `Competing` / leaf branch collapses to its single
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
# Every other branch (a `Sequential` / `Competing` / leaf) collapses to ONE
# endpoint, named by its branch name.
function _append_endpoint_names!(out, name::Symbol, ::Any)
    push!(out, name)
    return out
end

# --- NamedTuple INPUT to logpdf ---------------------------------------------
#
# `logpdf` scores the vector-valued representation; a labelled `NamedTuple` draw
# (as `rand(d)` now returns) is accepted and converted to the scored vector BY
# NAME first, so a self-labelling draw round-trips straight back through
# `logpdf(d, rand(d))`. A CENSORED composer scores the flat EVENT vector
# (matched by name via `_row_event_vector`, a missing required event errors); a
# PLAIN composer scores the per-VALUE vector (matched by its value names). Field
# ORDER does not matter; the names do.

function logpdf(d::Union{Sequential, Parallel}, x::NamedTuple)
    _tree_primary_event(d) === nothing &&
        return logpdf(d, _named_value_vector(d, x))
    return logpdf(d, _row_event_vector(d, x))
end

# The per-VALUE vector of a plain composer from a labelled draw, matched to the
# `_value_names(d)` layout BY NAME. A plain composer scores the per-value vector
# (no latent origin, no missing events), so this is a plain `Vector{Float64}`
# that routes to the plain `logpdf(::Sequential/Parallel, ::AbstractVector)`
# (NOT the Missing-admitting censored event scorer).
function _named_value_vector(d::Union{Sequential, Parallel}, x::NamedTuple)
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
