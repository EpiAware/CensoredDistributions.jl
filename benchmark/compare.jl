# Compare two benchmark result files and write a Markdown PR comment.
#
# Usage: julia --project=benchmark compare.jl <pr.json> <main.json> <out.md>
#
# The comment has two parts:
#   1. "Changed vs main" — benchmarks whose minimum time moved by more
#      than `CHANGE_THRESHOLD`, shown inline for visibility.
#   2. "All benchmarks" — every benchmark, in a collapsed section, sorted
#      by how far the time ratio is from 1.
# Each part is split into "Evaluation" and "AD gradients" so the gradient
# benchmarks are easy to find.
using BenchmarkTools
using Statistics: median

const CHANGE_THRESHOLD = 0.05  # 5% time change counts as "changed"
const COMMENT_MARKER = "<!-- benchmark-comparison -->"

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

# Render an "Evaluation" then "AD gradients" pair of tables for a row set.
function render_sections(rows)
    eval_rows = filter(r -> !is_ad(r.name), rows)
    ad_rows = filter(r -> is_ad(r.name), rows)
    io = IOBuffer()
    println(io, "#### Evaluation\n")
    print(io, render_table(eval_rows))
    println(io, "\n#### AD gradients\n")
    print(io, render_table(ad_rows))
    return String(take!(io))
end

pr = index_results(load_group(pr_file))
main = index_results(load_group(main_file))
rows = build_rows(pr, main)

changed = filter(rows) do r
    r.status !== :both || abs(r.time_ratio - 1) > CHANGE_THRESHOLD
end
sort!(changed; by = sort_key, rev = true)

all_sorted = sort(rows; by = sort_key, rev = true)

io = IOBuffer()
println(io, COMMENT_MARKER)
println(io, "## Benchmark comparison vs `main`\n")
println(io,
    "Minimum time and allocations per benchmark. Ratio is PR / main ",
    "(🔴 slower, 🟢 faster, ⚪ within ", round(Int, 100CHANGE_THRESHOLD),
    "%).\n")

if isempty(changed)
    println(io, "**No benchmarks changed by more than ",
        round(Int, 100CHANGE_THRESHOLD), "% vs `main`.**\n")
else
    println(io, "### Changed vs `main` (", length(changed), ")\n")
    print(io, render_sections(changed))
    println(io)
end

println(io, "<details><summary><b>All benchmarks</b> (",
    length(all_sorted), ", sorted by |ratio − 1|)</summary>\n")
print(io, render_sections(all_sorted))
println(io, "\n</details>")

write(out_file, String(take!(io)))
println("Wrote comparison comment to ", out_file)
