# Compare two benchmark result files and write a Markdown PR comment.
#
# Usage: julia --project=benchmark compare.jl <pr.json> <main.json> <out.md>
#
# The comment is:
#   1. A summary table: rows are Evaluation and each AD backend, columns
#      are time buckets (PR time as a percentage of main, lower = faster),
#      cells are how many benchmarks fall in each bucket.
#   2. Two collapsed tables — Evaluation and AD gradients — each listing
#      every benchmark sorted by how much its time moved.
using BenchmarkTools

const CHANGE_THRESHOLD = 0.05  # 5% time change = ±5% bucket edge
const COMMENT_MARKER = "<!-- benchmark-comparison -->"

# Time buckets for the summary table, keyed on PR time as a percentage of
# main (so < 100% is faster). Edges mirror around 100%: ±5 neutral, ±25,
# ±50. Each entry is (column label, upper edge in %). A benchmark lands in
# the first bucket whose edge it is below.
const BUCKETS = [
    ("🟢 <50%", 50.0),
    ("🟢 50–75%", 75.0),
    ("🟢 75–95%", 95.0),
    ("⚪ 95–105%", 105.0),
    ("🔴 105–125%", 125.0),
    ("🔴 125–150%", 150.0),
    ("🔴 >150%", Inf)
]

# Summary row order: Evaluation first, then AD backends in this order.
const BACKEND_ORDER = ["ForwardDiff", "ReverseDiff (tape)", "Mooncake reverse",
    "Mooncake forward", "Enzyme reverse", "Enzyme forward"]

pr_file, main_file, out_file = ARGS[1], ARGS[2], ARGS[3]

load_group(path) = BenchmarkTools.load(path)[1]

# Map each benchmark's key path (joined with " / ") to its minimum time
# (ns) and allocated memory (bytes). Minimum time is the stable estimator
# for a like-for-like ratio.
function index_results(group)
    out = Dict{String, NamedTuple{(:time, :memory), Tuple{Float64, Float64}}}()
    for (keypath, trial) in BenchmarkTools.leaves(group)
        name = join(string.(keypath), " / ")
        est = minimum(trial)
        out[name] = (time = Float64(time(est)), memory = Float64(memory(est)))
    end
    return out
end

is_ad(name) = startswith(name, "AD gradients")

function fmt_time(ns)
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

function fmt_mem(bytes)
    isnan(bytes) && return "—"
    if bytes < 1024
        return string(round(Int, bytes), " B")
    elseif bytes < 1024^2
        return string(round(bytes / 1024; digits = 2), " KiB")
    else
        return string(round(bytes / 1024^2; digits = 2), " MiB")
    end
end

# Ratio with a colour cue: red slower, green faster, white within noise.
function fmt_ratio(r)
    isnan(r) && return "—"
    marker = r > 1 + CHANGE_THRESHOLD ? "🔴" :
             r < 1 - CHANGE_THRESHOLD ? "🟢" : "⚪"
    return string(marker, " ", round(r; digits = 2), "×")
end

# One row per benchmark, carrying both revisions' numbers and the ratios.
struct Row
    name::String
    main_time::Float64
    pr_time::Float64
    main_mem::Float64
    pr_mem::Float64
    time_ratio::Float64   # NaN when the benchmark is new or removed
    status::Symbol        # :both, :new (PR only), :removed (main only)
end

function build_rows(pr, main)
    rows = Row[]
    for name in sort(collect(union(keys(pr), keys(main))))
        inpr, inmain = haskey(pr, name), haskey(main, name)
        pt = inpr ? pr[name].time : NaN
        mt = inmain ? main[name].time : NaN
        pm = inpr ? pr[name].memory : NaN
        mm = inmain ? main[name].memory : NaN
        ratio = (inpr && inmain) ? pt / mt : NaN
        status = inpr && inmain ? :both : inpr ? :new : :removed
        push!(rows, Row(name, mt, pt, mm, pm, ratio, status))
    end
    return rows
end

# Sort key: new/removed first (most noteworthy), then by distance from 1.
sort_key(r) = isnan(r.time_ratio) ? Inf : abs(r.time_ratio - 1)

function status_note(r)
    r.status === :new && return " *(new)*"
    r.status === :removed && return " *(removed)*"
    return ""
end

function render_table(rows)
    isempty(rows) && return "_none_\n"
    io = IOBuffer()
    println(io, "| Benchmark | main | PR | time | memory |")
    println(io, "|---|---|---|---|---|")
    for r in rows
        memratio = (isnan(r.pr_mem) || isnan(r.main_mem) || r.main_mem == 0) ?
                   NaN : r.pr_mem / r.main_mem
        println(io, "| ", r.name, status_note(r),
            " | ", fmt_time(r.main_time),
            " | ", fmt_time(r.pr_time),
            " | ", fmt_ratio(r.time_ratio),
            " | ", fmt_ratio(memratio), " |")
    end
    return String(take!(io))
end

# Summary group for a benchmark: "Evaluation" for non-AD rows, otherwise
# the AD backend (the last segment of the key path).
group_of(name) = is_ad(name) ? String(split(name, " / ")[end]) : "Evaluation"

# Index of the bucket a PR/main percentage falls in.
function bucket_index(pct)
    for (i, (_, edge)) in enumerate(BUCKETS)
        pct < edge && return i
    end
    return length(BUCKETS)
end

# Count benchmarks per (group, bucket) and render the summary table.
function summary_table(rows)
    counts = Dict{String, Vector{Int}}()
    for r in rows
        isnan(r.time_ratio) && continue
        v = get!(counts, group_of(r.name), zeros(Int, length(BUCKETS)))
        v[bucket_index(100 * r.time_ratio)] += 1
    end

    groups = String[]
    haskey(counts, "Evaluation") && push!(groups, "Evaluation")
    for b in BACKEND_ORDER
        haskey(counts, b) && push!(groups, b)
    end
    for g in sort(collect(keys(counts)))
        g in groups || push!(groups, g)
    end

    io = IOBuffer()
    println(io, "| Group | ", join(first.(BUCKETS), " | "), " |")
    println(io, "|---|", repeat("---|", length(BUCKETS)))
    for g in groups
        cells = [c == 0 ? "·" : string(c) for c in counts[g]]
        println(io, "| ", g, " | ", join(cells, " | "), " |")
    end
    return String(take!(io))
end

pr = index_results(load_group(pr_file))
main = index_results(load_group(main_file))
rows = build_rows(pr, main)

# Both detail tables are sorted by how much the TIME moved (biggest first).
all_sorted = sort(rows; by = sort_key, rev = true)
eval_rows = filter(r -> !is_ad(r.name), all_sorted)
ad_rows = filter(r -> is_ad(r.name), all_sorted)

io = IOBuffer()
println(io, COMMENT_MARKER)
println(io, "## Benchmark comparison vs `main`\n")
# Direction is fixed and stated up front: cells are PR time as a
# percentage of main, so below 100% means the PR is faster.
println(io,
    "Minimum time per call. Buckets are **PR time as a % of `main`, so ",
    "lower is faster** (🟢 faster, ⚪ within ", round(Int, 100CHANGE_THRESHOLD),
    "%, 🔴 slower). Counts of benchmarks per bucket:\n")
print(io, summary_table(rows))

println(io, "\n<details><summary><b>Evaluation</b> — ", length(eval_rows),
    " benchmarks (by time change)</summary>\n")
print(io, render_table(eval_rows))
println(io, "\n</details>")

println(io, "\n<details><summary><b>AD gradients</b> — ", length(ad_rows),
    " benchmarks (by time change)</summary>\n")
print(io, render_table(ad_rows))
println(io, "\n</details>")

write(out_file, String(take!(io)))
println("Wrote comparison comment to ", out_file)
