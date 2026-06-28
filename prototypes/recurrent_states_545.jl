# ============================================================================
# PROTOTYPE (issue #545): recurrent / cyclic multi-state transitions
# ============================================================================
#
# This is a design PROTOTYPE, not package code. It is deliberately self-
# contained (a single script, no new exports) and lives outside `src/` so it
# does not enter the public surface. It demonstrates the renewal-over-states
# approach the design proposal recommends: a cyclic multi-state path is an
# unfolded renewal where each step is a semi-Markov sojourn built from the
# existing `compete` racing-hazard verb.
#
# What it shows end to end:
#   - construct  a state graph with a back edge (a cycle the forward-only
#     composition grammar cannot represent today);
#   - simulate   a full path (`rand`) of (state, dwell, next-state) jumps;
#   - score      that path (`logpdf`) by summing the per-sojourn competing-risks
#     log-densities, which is exactly the term the existing `Compete` node
#     already computes;
#   - fit        the sojourn parameters on synthetic paths by maximum
#     likelihood, recovering the data-generating values.
#
# The single new idea over today's grammar is the WHILE loop: a path keeps
# taking sojourn steps until it reaches an absorbing state or a horizon. Every
# per-step quantity is the existing verb machinery. The production-scope items
# the prototype defers are listed at the bottom.

using CensoredDistributions
using Distributions
using Random

const CD = CensoredDistributions

# A tiny dependency-free Nelder-Mead so the prototype pulls in no optimiser
# package (the production path would use the package's Turing / Optimization
# glue). Minimises `f` from `x0`; returns the best point.
function _nelder_mead(f, x0; iters = 2000, step = 0.5)
    n = length(x0)
    simplex = [copy(x0) for _ in 0:n]
    for i in 1:n
        simplex[i + 1][i] += step
    end
    fv = [f(x) for x in simplex]
    for _ in 1:iters
        ord = sortperm(fv)
        simplex, fv = simplex[ord], fv[ord]
        centroid = sum(simplex[1:n]) ./ n
        worst = simplex[end]
        refl = centroid .+ (centroid .- worst)        # reflection
        fr = f(refl)
        if fr < fv[1]
            expd = centroid .+ 2 .* (centroid .- worst)  # expansion
            fe = f(expd)
            simplex[end], fv[end] = fe < fr ? (expd, fe) : (refl, fr)
        elseif fr < fv[end - 1]
            simplex[end], fv[end] = refl, fr
        else
            contr = centroid .+ 0.5 .* (worst .- centroid)  # contraction
            fc = f(contr)
            if fc < fv[end]
                simplex[end], fv[end] = contr, fc
            else
                for i in 2:(n + 1)                      # shrink
                    simplex[i] = simplex[1] .+ 0.5 .* (simplex[i] .- simplex[1])
                    fv[i] = f(simplex[i])
                end
            end
        end
    end
    return simplex[argmin(fv)]
end

# ----------------------------------------------------------------------------
# The model: a renewal over states
# ----------------------------------------------------------------------------
#
# A `StateGraph` maps each non-absorbing state to a `compete(...)` node over its
# OUTGOING edges (the cause-specific sojourn distributions). Reaching an
# absorbing state (no outgoing edges) ends the path. Cycles are allowed: an
# edge may point back to an already-visited state, which is the whole point of
# #545. The semi-Markov clock RESETS on entry to each state, so each sojourn is
# an independent draw from that state's competing-risks node -- exactly the
# clock-reset sojourn the multi-state framing tutorial already documents.

struct StateGraph
    # state name -> compete node over its outgoing edges (absorbing states are
    # simply absent from the dictionary).
    edges::Dict{Symbol, Any}
end

# Build the per-state `compete` nodes from `state => (next_state => dist, ...)`.
# Each inner pair is an outgoing edge: the destination state and the sojourn
# (dwell-time) distribution on that edge.
function state_graph(specs::Pair...)
    edges = Dict{Symbol, Any}()
    for (state, out) in specs
        # A single-edge state still needs a node; `compete` requires >= 2
        # outcomes, so a lone edge is stored as a plain `dest => dist` pair and
        # handled directly by the sampler/scorer.
        edges[state] = length(out) == 1 ? only(out) :
                       compete((first.(out) .=> last.(out))...)
    end
    return StateGraph(edges)
end

is_absorbing(g::StateGraph, s::Symbol) = !haskey(g.edges, s)

# ----------------------------------------------------------------------------
# Simulate a path (`rand`)
# ----------------------------------------------------------------------------
#
# Start in `start`, repeatedly draw the winning next-state and its dwell time
# from the current state's competing-risks node (a lone edge draws directly),
# accumulate calendar time, and stop at an absorbing state or when `horizon` is
# exceeded. Returns the jump chain as a vector of (from, to, dwell) tuples.

function rand_path(rng::AbstractRNG, g::StateGraph, start::Symbol;
        horizon::Real = Inf, max_jumps::Int = 1000)
    path = Tuple{Symbol, Symbol, Float64}[]
    state = start
    clock = 0.0
    for _ in 1:max_jumps
        is_absorbing(g, state) && break
        node = g.edges[state]
        if node isa Pair                      # a lone outgoing edge
            dest, dist = node
            dwell = rand(rng, dist)
        else                                  # a competing-risks node
            dest, dwell = CD.rand_outcome(rng, node)
        end
        clock + dwell > horizon && break
        push!(path, (state, dest, Float64(dwell)))
        clock += dwell
        state = dest
    end
    return path
end

function rand_path(g::StateGraph, start::Symbol; kwargs...)
    rand_path(Random.default_rng(), g, start; kwargs...)
end

# ----------------------------------------------------------------------------
# Score a path (`logpdf`)
# ----------------------------------------------------------------------------
#
# The log-density of an observed jump chain is the SUM of the per-sojourn
# competing-risks log sub-densities `log f_j(t) + sum_{k != j} log S_k(t)` --
# the cause-resolved term the `Compete` node already defines (here read through
# the hazard accessors so no internal name is needed). A lone-edge step scores
# its plain `logpdf`. This is the standard semi-Markov / msm jump-chain
# likelihood, and it factorises over steps because each sojourn clock resets.

# Cause-resolved log sub-density of taking edge `dest` after dwell `t` from a
# competing-risks `node`: `log f_dest(t) + sum_{k != dest} log S_k(t)`.
function _edge_logpdf(node, dest::Symbol, t::Real)
    names = CD.component_names(node)
    j = findfirst(==(dest), names)
    j === nothing && return -Inf
    total = zero(float(t))
    for (k, nm) in enumerate(names)
        d = node.delays[k]
        total += k == j ? logpdf(d, t) : logccdf(d, t)
    end
    return total
end

function logpdf_path(g::StateGraph, path::Vector{<:Tuple})
    total = 0.0
    for (from, to, dwell) in path
        node = g.edges[from]
        total += node isa Pair ? logpdf(node.second, dwell) :
                 _edge_logpdf(node, to, dwell)
    end
    return total
end

# ----------------------------------------------------------------------------
# End-to-end demonstration
# ----------------------------------------------------------------------------
#
# A waning-immunity / reinfection cycle, the motivating #545 use case:
#
#     susceptible --infect--> infected --recover--> recovered
#          ^                                            |
#          +-------------------- wane -------------------+
#
# `recovered` cycles back to `susceptible`, so a path is
# S -> I -> R -> S -> I -> ... an unbounded, random number of jumps: a genuine
# cycle the forward-only grammar cannot build. There is no absorbing state, so
# the path runs to the horizon.

function build_graph(; infect_scale = 4.0, recover_scale = 3.0,
        wane_scale = 30.0)
    return state_graph(
        # `infected` is the only competing state here (recovery only), but we
        # give `susceptible` and `recovered` lone edges to keep the cycle.
        :susceptible => (:infected => Gamma(2.0, infect_scale),),
        :infected => (:recovered => Gamma(2.0, recover_scale),),
        :recovered => (:susceptible => Gamma(2.0, wane_scale),))
end

# A two-cause variant so the competing-risks scoring is exercised: an infected
# case either recovers or dies (an absorbing state ending the path).
function build_graph_competing(; infect = 4.0, recover = 3.0, die = 8.0,
        wane = 30.0)
    return state_graph(
        :susceptible => (:infected => Gamma(2.0, infect),),
        :infected => (:recovered => Gamma(2.0, recover),
            :dead => Gamma(2.0, die)),
        :recovered => (:susceptible => Gamma(2.0, wane),))
end

function demo()
    rng = MersenneTwister(545)

    println("== construct ==")
    g = build_graph_competing()
    println("states with outgoing edges: ", sort(collect(keys(g.edges))))
    println("`dead` absorbing? ", is_absorbing(g, :dead))

    println("\n== simulate one cyclic path ==")
    path = rand_path(rng, g, :susceptible; horizon = 90.0)
    for (from, to, dwell) in path
        println("  ", rpad(string(from), 12), " -> ", rpad(string(to), 12),
            " after ", round(dwell; digits = 2), " days")
    end

    println("\n== score that path ==")
    println("  logpdf(path) = ", round(logpdf_path(g, path); digits = 4))

    println("\n== fit sojourn scales on synthetic paths ==")
    fit_demo(rng)
    return nothing
end

# ----------------------------------------------------------------------------
# Fit on synthetic data
# ----------------------------------------------------------------------------
#
# Simulate many cyclic paths from a known graph, then recover the three sojourn
# scales (infect, recover, wane) by maximising the summed jump-chain
# log-likelihood. Demonstrates that the renewal-over-states likelihood is a
# standard differentiable objective -- here optimised with Nelder-Mead, but the
# per-step terms are the same AD-safe `Compete` densities the package fits with
# Turing elsewhere.

function fit_demo(rng::AbstractRNG; n_paths = 800)
    truth = (infect = 4.0, recover = 3.0, wane = 30.0)
    g_true = build_graph(; infect_scale = truth.infect,
        recover_scale = truth.recover, wane_scale = truth.wane)
    # A long horizon so each path completes several full S -> I -> R -> S
    # cycles; the prototype scores only completed sojourns (the final, still-
    # running sojourn is dropped -- right-censoring is a deferred item).
    paths = [rand_path(rng, g_true, :susceptible; horizon = 600.0)
             for _ in 1:n_paths]

    # Negative log-likelihood as a function of log-scales (keeps them positive).
    function nll(logscales)
        s = exp.(logscales)
        g = build_graph(; infect_scale = s[1], recover_scale = s[2],
            wane_scale = s[3])
        return -sum(p -> logpdf_path(g, p), paths)
    end

    x0 = log.([2.0, 2.0, 20.0])           # deliberately off the truth
    fitted = exp.(_nelder_mead(nll, x0))
    println("  truth : infect=", truth.infect, " recover=", truth.recover,
        " wane=", truth.wane)
    println("  fitted: infect=", round(fitted[1]; digits = 2),
        " recover=", round(fitted[2]; digits = 2),
        " wane=", round(fitted[3]; digits = 2))
    return fitted
end

# ----------------------------------------------------------------------------
# Deferred production scope (NOT in this prototype)
# ----------------------------------------------------------------------------
#
#  - censored / interval-observed sojourns: wrap each edge dist in
#    `interval_censored` and score through the package's event-vector path
#    rather than the exact `logpdf` used here;
#  - panel observation (state seen at fixed visit times, transition times
#    unknown): needs marginalising over the hidden jump chain -- a transition-
#    probability-matrix recursion the prototype does not implement;
#  - calendar-time-varying intensities (piecewise-constant rates) on edges;
#  - a first-class verb (`recur` / `states(...)`) and event-record schema so a
#    cyclic model composes with `sequential` / `parallel` and reads back through
#    `event_names` / `params_table` / `update(d, chain)`;
#  - Turing glue (`composed_parameters_model` / priors front-door) for the
#    path likelihood;
#  - the CTMC (exponential-only, generator-matrix) alternative as a fast path
#    for the memoryless special case.

if abspath(PROGRAM_FILE) == @__FILE__
    demo()
end
