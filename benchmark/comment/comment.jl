# Build a legible PR benchmark comment from AirspeedVelocity result JSON.
#
# AirspeedVelocity's own action posts a flat table of every benchmark. With
# the AD-gradient suite added (`benchmark/src/ad_gradients.jl`), that table is
# an unreadable wall of hundreds of (scenario x backend) rows (#290). This
# script reads the same `results_<pkg>@<rev>.json` files the action writes and
# emits a comment with three parts:
#
#   1. A "most changed" summary (largest median-time ratio moves), so a reader
#      sees regressions/improvements at a glance across the whole suite,
#      including AD gradients.
#   2. A compact AD-gradients matrix (scenarios as rows, backends as columns,
#      cells are the median-time ratio) so the AD numbers are present but
#      legible rather than one row per pair.
#   3. The full non-AD table, folded behind <details>.
#
# Usage:
#   julia --project=benchmark/comment benchmark/comment/comment.jl \
#       <results-dir> <package> <base-rev> <head-rev> <out.md>
#
# Revs are matched leniently (prefix), because the action passes a full SHA
# while the on-disk filename may be truncated by benchpkg.

using JSON3

const AD_PREFIX = "AD gradients/"
const RATIO_THRESHOLD = 1.05   # |ratio - 1| >= 5% counts as "changed"
const TOP_N = 15               # rows shown in the "most changed" table

# ---- result loading --------------------------------------------------------

# Recursively flatten the BenchmarkTools/JSON3 group into `path => median_ns`.
# Leaf groups have a "times" vector (nanoseconds); inner groups have "data".
function flatten!(out::Dict{String, Float64}, node, prefix::String)
    (node isa AbstractDict || node isa JSON3.Object) || return out
    if haskey(node, "times")
        times = node["times"]
        if !isempty(times)
            out[prefix] = median_of(times)
        end
    elseif haskey(node, "data")
        for (k, v) in pairs(node["data"])
            key = String(k)
            next = isempty(prefix) ? key : prefix * "/" * key
            flatten!(out, v, next)
        end
    end
    return out
end

function median_of(times)
    xs = sort(collect(Float64, times))
    n = length(xs)
    n == 0 && return NaN
    isodd(n) ? xs[(n + 1) ÷ 2] : (xs[n ÷ 2] + xs[n ÷ 2 + 1]) / 2
end

# Locate `results_<pkg>@<rev>...json`, matching the rev as a prefix so a full
# SHA on the command line still finds a file benchpkg truncated.
function find_results(dir::AbstractString, pkg::AbstractString,
        rev::AbstractString)
    candidates = filter(readdir(dir; join = true)) do f
        endswith(f, ".json") && occursin("results_" * pkg * "@", basename(f))
    end
    isempty(candidates) &&
        error("no results json for $pkg in $dir")
    # Exact rev (or a prefix of it) embedded in the filename.
    for f in candidates
        tag = match(r"@(.+)\.json$", basename(f))
        tag === nothing && continue
        t = tag.captures[1]
        if startswith(rev, t) || startswith(t, rev) || rev == t
            return f
        end
    end
    # Single-file fallback: nothing to disambiguate.
    length(candidates) == 1 && return only(candidates)
    error("could not match rev $rev among $(basename.(candidates))")
end

function load_flat(dir, pkg, rev)
    file = find_results(dir, pkg, rev)
    data = open(file, "r") do io
        JSON3.read(read(io, String))
    end
    out = Dict{String, Float64}()
    flatten!(out, data, "")
    return out
end

# ---- formatting helpers ----------------------------------------------------

function fmt_time(ns::Float64)
    isnan(ns) && return "—"
    if ns < 1e3
        return string(round(ns; digits = 1), " ns")
    elseif ns < 1e6
        return string(round(ns / 1e3; digits = 2), " μs")
    elseif ns < 1e9
        return string(round(ns / 1e6; digits = 2), " ms")
    else
        return string(round(ns / 1e9; digits = 2), " s")
    end
end

fmt_ratio(r::Float64) = isnan(r) ? "—" : string(round(r; digits = 3))

# Mark notable moves so they catch the eye in markdown.
function ratio_cell(r::Float64)
    isnan(r) && return "—"
    s = fmt_ratio(r)
    if r >= 1.10
        return "🔴 " * s        # slower by >=10%
    elseif r <= 0.91
        return "🟢 " * s        # faster by >=10%
    else
        return s
    end
end

# Split a full AD key "AD gradients/<scenario>/<backend>" into its parts.
function ad_parts(key::AbstractString)
    rest = key[(length(AD_PREFIX) + 1):end]
    idx = findlast('/', rest)
    idx === nothing && return (rest, "")
    return (rest[1:(idx - 1)], rest[(idx + 1):end])
end

# ---- comment sections ------------------------------------------------------

function most_changed_section(io, base, head)
    rows = Tuple{String, Float64, Float64, Float64}[]
    for (k, h) in head
        haskey(base, k) || continue
        b = base[k]
        (b <= 0 || h <= 0) && continue
        push!(rows, (k, b, h, h / b))
    end
    changed = filter(r -> abs(r[4] - 1) >= (RATIO_THRESHOLD - 1), rows)
    println(io, "### Most changed (median time)\n")
    if isempty(changed)
        println(io,
            "No benchmark moved by more than ",
            round(Int, (RATIO_THRESHOLD - 1) * 100),
            "%. ", length(rows), " benchmarks compared.\n")
        return
    end
    sort!(changed; by = r -> abs(r[4] - 1), rev = true)
    println(io, "| Benchmark | base | PR | PR / base |")
    println(io, "|:--|--:|--:|--:|")
    for r in first(changed, TOP_N)
        println(io, "| `", r[1], "` | ", fmt_time(r[2]), " | ",
            fmt_time(r[3]), " | ", ratio_cell(r[4]), " |")
    end
    if length(changed) > TOP_N
        println(io, "\n", length(changed) - TOP_N,
            " further benchmarks changed (see full results below).")
    end
    println(io)
end

function ad_section(io, base, head)
    ad_keys = filter(k -> startswith(k, AD_PREFIX), keys(head))
    println(io, "### AD gradients (PR / base, median time)\n")
    if isempty(ad_keys)
        println(io, "No AD-gradient benchmarks in this run.\n")
        return
    end
    scenarios = String[]
    backends = String[]
    cell = Dict{Tuple{String, String}, Float64}()
    for k in ad_keys
        scen, back = ad_parts(k)
        haskey(base, k) || continue
        b = base[k]
        h = head[k]
        (b <= 0 || h <= 0) && continue
        scen in scenarios || push!(scenarios, scen)
        back in backends || push!(backends, back)
        cell[(scen, back)] = h / b
    end
    if isempty(cell)
        println(io,
            "AD benchmarks ran but had no comparable base counterpart.\n")
        return
    end
    sort!(scenarios)
    sort!(backends)
    print(io, "| Scenario |")
    for b in backends
        print(io, " ", b, " |")
    end
    println(io)
    print(io, "|:--|")
    for _ in backends
        print(io, "--:|")
    end
    println(io)
    for s in scenarios
        print(io, "| ", s, " |")
        for b in backends
            r = get(cell, (s, b), NaN)
            print(io, " ", ratio_cell(r), " |")
        end
        println(io)
    end
    println(io,
        "\nCells are PR median / base median. 🔴 ≥1.10 (slower), ",
        "🟢 ≤0.91 (faster). Blank = backend skipped on that scenario.\n")
end

function full_section(io, base, head)
    keys_all = sort(collect(keys(head)))
    non_ad = filter(k -> !startswith(k, AD_PREFIX), keys_all)
    ad = filter(k -> startswith(k, AD_PREFIX), keys_all)
    println(io, "<details><summary>Full results</summary>\n")
    for (title, ks) in (("Core benchmarks", non_ad),
        ("AD gradients (raw)", ad))
        isempty(ks) && continue
        println(io, "\n#### ", title, "\n")
        println(io, "| Benchmark | base | PR | PR / base |")
        println(io, "|:--|--:|--:|--:|")
        for k in ks
            h = get(head, k, NaN)
            b = get(base, k, NaN)
            r = (b > 0 && h > 0) ? h / b : NaN
            println(io, "| `", k, "` | ", fmt_time(b), " | ",
                fmt_time(h), " | ", fmt_ratio(r), " |")
        end
    end
    println(io, "\n</details>")
end

# ---- main ------------------------------------------------------------------

function main(args)
    length(args) == 5 || error(
        "usage: comment.jl <dir> <pkg> <base-rev> <head-rev> <out.md>")
    dir, pkg, base_rev, head_rev, out = args
    base = load_flat(dir, pkg, base_rev)
    head = load_flat(dir, pkg, head_rev)
    open(out, "w") do io
        println(io, "## Benchmark results\n")
        println(io,
            "Comparing PR head against the base branch ",
            "(median time; ratio = PR / base, <1 is faster).\n")
        most_changed_section(io, base, head)
        ad_section(io, base, head)
        full_section(io, base, head)
        println(io,
            "\n<sub>Generated from AirspeedVelocity results by ",
            "`benchmark/comment/comment.jl`.</sub>")
    end
    return nothing
end

main(ARGS)
