using TestItemRunner

# `:ad`-tagged items live under `test/ad/` with their own project
# (Enzyme, Mooncake, etc. are not deps of the main test env) and run in
# dedicated per-backend CI, so they are always excluded from the main
# suite. See `test/ad/runtests.jl`.

# `@run_package_tests` discovers test items by walking the package
# directory tree. Git worktrees are nested under `worktrees/` inside the
# repo (see the project's worktree conventions), so the walk would also
# pick up testitems from sibling worktrees. Those items rely on
# TestItemRunner's auto-injected `using CensoredDistributions`, which is
# not set up when they are globbed from outside their own package
# context, producing spurious `UndefVarError`/`Test Failed` results as
# well as huge duplicate runs. Restrict discovery to items living under
# *this* package's own `test/` directory.
# Trailing separator guards against sibling dirs that share a string
# prefix (e.g. a `test-extra/` next to `test/`).
const TEST_ROOT = normpath(@__DIR__) * "/"

in_this_package(ti) = startswith(normpath(ti.filename), TEST_ROOT)

# Filter tests based on command line arguments
if "skip_quality" in ARGS
    # Skip quality tests (JET, Aqua, formatting) used in CI for performance
    @run_package_tests filter = ti -> in_this_package(ti) &&
                                      !(:quality in ti.tags) &&
                                      !(:ad in ti.tags)
elseif "quality_only" in ARGS
    # Run only quality tests (Aqua, formatting, linting, doctests)
    @run_package_tests filter = ti -> in_this_package(ti) &&
                                      :quality in ti.tags
elseif "readme_only" in ARGS
    # Run only README tests
    @run_package_tests filter = ti -> in_this_package(ti) &&
                                      :readme in ti.tags
else
    # Run all tests (default for local development)
    @run_package_tests filter = ti -> in_this_package(ti) &&
                                      !(:ad in ti.tags)
end
