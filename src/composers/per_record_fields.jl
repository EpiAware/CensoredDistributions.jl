# ============================================================================
# Per-record `:field` modifier parameters
# ============================================================================
#
# A modifier on a composed node usually bakes a constant bound/interval into the
# leaves at construction. Passing a `Symbol` instead names a per-record column:
# the value is read from that field of each row at scoring time, the same way
# the per-record observation horizon reads `:obs_time` / `:obs_window`. This
# generalises that one reserved horizon to any modifier parameter (the
# truncation `lower` / `upper`, the interval width, ...) without a parallel
# mechanism: a field-bound node defers the modifier, and `record_distributions`
# resolves each row's fields into a concrete modified node, reusing the
# per-record path.
#
# `_DeferredFields` carries the inner node, the modifier rebuilt per record, and
# the field names it reads (stripped from a row before event matching). It is
# not scored directly; `record_distributions` / `batched_event_logpdf` resolve
# it. The type and resolver helpers live here (before `wrap.jl`, which builds
# the carrier from `:field` modifier keywords); the assembly + scoring methods
# are in `per_record_fields_scoring.jl` (after the per-record path is defined).

# A composed node with one or more modifier parameters bound to per-record
# fields. `node` is the inner composed node; `build` takes a `NamedTuple` of
# the resolved field values (keyed by the field name) and returns the concrete
# modified node for that row; `fields` are the row columns read (so they are
# not matched as events).
struct _DeferredFields{D, B, F}
    node::D
    build::B
    fields::F
end

# A bound modifier parameter: either a constant value or a `:field` reference
# read per record.
const _MaybeField = Union{Real, Symbol, Nothing}

# Whether a bound value is a per-record `:field` reference.
_is_field(::Symbol) = true
_is_field(::Any) = false

# Resolve one bound value against a row's field NamedTuple: a `Symbol` reads the
# named column, anything else is the constant.
_resolve_field(v::Symbol, row::NamedTuple) = row[v]
_resolve_field(v, ::NamedTuple) = v

# The field names referenced by the bound values (the `Symbol`s), in order.
_field_names(vs...) = Tuple(v for v in vs if v isa Symbol)
