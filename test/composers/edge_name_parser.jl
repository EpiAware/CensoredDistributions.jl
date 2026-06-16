# Direct unit tests for the edge-name parsers that the tree walks rely on.
# These were previously only exercised indirectly through `event_names` /
# `params_table`, yet the underscored event split has already caused a Mooncake
# regression (#409, the no-`Regex` rewrite). Two distinct namespaces are tested:
#
#   - the UNDERSCORED event/value split (`tree_events.jl`):
#     `:onset_admit` -> `(:onset, :admit)`, with the positional-default
#     (`:step_i` / `:branch_i`) fallback.
#   - the DOTTED parameter-path split (`introspection.jl`):
#     `_join_path` / `_split_edge`, which must round-trip.

@testitem "underscored edge-name split: dotted/positional cases" begin
    using CensoredDistributions: _split_edge_name, _has_positional_suffix,
                                 _is_positional_edge_name

    # A single internal underscore splits into origin/target event names.
    @test _split_edge_name(:onset_admit) == (:onset, :admit)
    @test _split_edge_name(:admit_death) == (:admit, :death)

    # No single split (zero or multiple internal underscores) -> nothing, so
    # the caller falls back to positional `:event_i` names.
    @test _split_edge_name(:onset) === nothing
    @test _split_edge_name(:onset_to_admit) === nothing
    # An empty half is not a valid split.
    @test _split_edge_name(Symbol("_admit")) === nothing
    @test _split_edge_name(Symbol("onset_")) === nothing

    # Positional-default edge names carry no real event names to split.
    @test _split_edge_name(:step_1) === nothing
    @test _split_edge_name(:branch_2) === nothing

    # The positional-suffix predicate: `prefix` then one-or-more ASCII digits.
    @test _has_positional_suffix("step_1", "step_")
    @test _has_positional_suffix("branch_42", "branch_")
    @test _has_positional_suffix("event_3", "event_")
    @test !_has_positional_suffix("step_", "step_")        # no digits
    @test !_has_positional_suffix("step_1a", "step_")      # trailing non-digit
    @test !_has_positional_suffix("onset_admit", "step_")  # wrong prefix

    # The positional edge-name predicate keys off `:step_`/`:branch_` only.
    @test _is_positional_edge_name(:step_1)
    @test _is_positional_edge_name(:branch_10)
    @test !_is_positional_edge_name(:onset_admit)
    @test !_is_positional_edge_name(:onset)
end

@testitem "dotted param-path join/split round-trips" begin
    using CensoredDistributions: _join_path, _split_edge

    # `_join_path` joins a path with '.'; a single-element path keeps its bare
    # name. `_split_edge` is its inverse.
    @test _join_path((:onset_admit,)) == :onset_admit
    @test _join_path((:onset_admit, :shape)) == Symbol("onset_admit.shape")
    @test _join_path((:r1, :onset_admit, :shape)) ==
          Symbol("r1.onset_admit.shape")

    @test _split_edge(:onset_admit) == (:onset_admit,)
    @test _split_edge(Symbol("onset_admit.shape")) == (:onset_admit, :shape)
    @test _split_edge(Symbol("r1.onset_admit.shape")) ==
          (:r1, :onset_admit, :shape)

    # Round-trip both directions on a range of paths.
    for path in ((:a,), (:a, :b), (:a, :b, :c), (:onset_admit, :meanlog))
        @test _split_edge(_join_path(path)) == path
    end
    for edge in (:bare, Symbol("a.b"), Symbol("a.b.c"))
        @test _join_path(_split_edge(edge)) == edge
    end
end

@testitem "flat event names reconstruct from edge-name splits" begin
    using CensoredDistributions, Distributions
    using CensoredDistributions: _flat_event_names

    # A named Sequential derives its flat event vector layout from the
    # underscored edge-name splits: origin from the first edge, one target per
    # edge, threaded forward.
    oa = primary_censored(LogNormal(1.5, 0.4), Uniform(0, 1))
    ad = primary_censored(Gamma(2.0, 1.0), Uniform(0, 1))
    seq = Sequential((oa, ad), (:onset_admit, :admit_death))
    @test _flat_event_names(seq) == (:onset, :admit, :death)

    # A Parallel hangs every branch off the shared origin from the first
    # edge's split.
    par = Parallel((oa, ad), (:onset_admit, :onset_notif))
    @test _flat_event_names(par) == (:onset, :admit, :notif)

    # An unnamed (positional-default) chain falls back to positional event
    # names `:event_i`.
    seq_pos = Sequential(LogNormal(1.0, 0.4), Gamma(2.0, 1.0))
    enames = _flat_event_names(seq_pos)
    @test all(n -> startswith(string(n), "event_"), enames)
    @test length(enames) == 3
end
