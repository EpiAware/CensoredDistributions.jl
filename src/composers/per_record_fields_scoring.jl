# ============================================================================
# Per-record `:field` modifier parameters: assembly + scoring
# ============================================================================
#
# Resolve a `_DeferredFields` node (see `per_record_fields.jl`) per record: read
# each row's bound fields, build that row's concrete modified node, then reuse
# the ordinary per-record path (with the bound fields stripped so they are not
# matched as events). One record per row, the same contract as the inner node,
# so the table front-door and the DynamicPPL model thread through unchanged.

# The flat event names are those of the inner node: a deferred modifier keeps
# the tree shape, so the row schema is unchanged (the bound fields are extra
# columns).
_flat_event_names(d::_DeferredFields) = _flat_event_names(d.node)
event_names(d::_DeferredFields) = event_names(d.node)
function _row_event_vector(d::_DeferredFields, row::NamedTuple)
    return _row_event_vector(d.node, row)
end

# Assemble per-record distributions for a field-bound node: each row resolves
# its bound fields into a concrete modified node, then builds that record
# through the ordinary per-record path. One record per row, the same contract
# as the inner node.
function record_distributions(d::_DeferredFields, rows)
    rowvec = collect(Tables.rows(rows))
    isempty(rowvec) && throw(ArgumentError(
        "record_distributions needs at least one record; got an empty table"))
    return map(rowvec) do row
        nt = _row_namedtuple(row)
        fieldvals = NamedTuple{d.fields}(map(f -> nt[f], d.fields))
        node = d.build(fieldvals)
        inner = _drop_named_fields(nt, d.fields)
        only(record_distributions(node, [inner]))
    end
end

# The internal batched entry and the public table front-door forward to the
# resolved per-record path, byte-identical to scoring each row's built node.
function batched_event_logpdf(d::_DeferredFields, rows)
    return _batched_records_logpdf(record_distributions(d, rows))
end
function logpdf(d::_DeferredFields, rows::AbstractVector{<:NamedTuple})
    return batched_event_logpdf(d, rows)
end
