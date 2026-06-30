# Pluggable numerical integration.
#
# Every integral in the package routes through `integrate`, which
# dispatches on the solver type. The package ships one lightweight default
# solver, `GaussLegendre`, implemented directly with `FastGaussQuadrature`
# nodes (no heavy dependency). Loading the optional Integrals.jl extension
# adds an `integrate` method for any Integrals.jl algorithm
# (e.g. `QuadGKJL()`, `HCubatureJL()`), so a user can pass an Integrals.jl
# solver and have that integral routed through `IntegralProblem`/`solve`.

@doc "

Fixed-node Gauss-Legendre quadrature solver (the package default).

Integrates with a Gauss-Legendre rule of `n` nodes, evaluated as a bare
weighted dot product (see [`gl_integrate`](@ref)). The constant control
flow and the accumulator type being seeded from the integrand make this
the AD-safe default: every supported AD backend can differentiate through
it, unlike adaptive schemes whose node count depends on integrand values.

The reference nodes and weights are built once at construction and held on
the solver, so the differentiated hot path is a pure weighted sum with no
shared mutable state. This matters for trace-based reverse-mode backends
(Enzyme, ReverseDiff): resolving the rule through a mutated global cache
inside the integrand crashes Enzyme reverse, so the rule travels with the
solver instead.

`n = 64` is accurate to about `1e-13` on the smooth, density-weighted
integrands used in this package. This is the core default and needs no
heavy dependency. Load Integrals.jl and pass an Integrals.jl algorithm
(e.g. `QuadGKJL()`) when adaptive accuracy is wanted and AD is not.

# See also
- [`gl_integrate`](@ref): The underlying quadrature reduction.
- [`integrate`](@ref): The pluggable entry point dispatching on a solver.
"
struct GaussLegendre{R}
    "Number of Gauss-Legendre nodes."
    n::Int
    "The reference Gauss-Legendre rule (nodes/weights on `[-1, 1]`)."
    rule::R
end

# Fixed Gauss-Legendre rule carrying its own reference nodes/weights on
# `[-1, 1]`. Holding the nodes/weights directly (and calling the integrand
# inline in `_gl_reduce`) rather than going through an Integrals.jl
# `IntegralProblem`/`solve` boundary lets Julia specialise on the
# integrand's concrete return type, so `Dual`s and AD tangents propagate
# and the result type is inferred. This is the AD-safe pattern from
# epiforecasts/BVDOutbreakSize `src/integrate.jl`.
struct _GL{N, W}
    nodes::N
    weights::W
end

_GL(n::Int) = _GL(FastGaussQuadrature.gausslegendre(n)...)

# Number of Gauss-Legendre nodes for the convolution quadrature. The
# batched path integrates every point over one shared window, so a small
# point whose natural window is much tighter than the shared one is
# resolved by only the nodes that fall in its sub-range; a peaked
# component density (e.g. LogNormal) makes this the accuracy-limiting
# case. n = 192 brings the batched-vs-scalar gap on a typical batch to
# ~5e-4 (n = 64 left it at ~4e-3) and shrinks it ~15x on a deliberately
# wide batch. Cost is roughly linear in the node count and still small
# for these smooth, density-weighted integrands. The scalar path stays
# the accurate reference; raise this if a batch spans an extreme range.
const _CONVOLVED_NODES = 192

# Default rule for the convolution quadrature, built once at load.
const _CONVOLVED_GL = _GL(_CONVOLVED_NODES)

# Number of Gauss-Legendre nodes for the primary-censoring default solver.
const _PRIMARY_NODES = 64

# Default rule for the primary-censoring solver, built once at load.
const _PRIMARY_GL = _GL(_PRIMARY_NODES)

# Resolve the reference rule for `n` nodes. The two defaults (192 for the
# convolution quadrature, 64 for primary censoring) are precomputed
# constants; any other `n` builds its nodes/weights fresh. There is no
# shared mutable cache: the rule is built at solver-construction time and
# then held on the `GaussLegendre` solver, so the differentiated hot path
# never resolves a rule. A mutated global cache here would be written
# inside the traced region and crash Enzyme reverse. `gausslegendre`
# returns plain `Float64` nodes/weights independent of any AD inputs, so
# even building a fresh rule under tracing is differentiation-safe.
function _gl_rule(n::Int)
    n == _CONVOLVED_NODES && return _CONVOLVED_GL
    n == _PRIMARY_NODES && return _PRIMARY_GL
    return _GL(n)
end

# Build the solver with its reference rule resolved once, here at
# construction, so `integrate` only reads `solver.rule`.
GaussLegendre(; n::Int = _PRIMARY_NODES) = GaussLegendre(n, _gl_rule(n))

# Show a Gauss-Legendre solver as its type and node count, never its node and
# weight arrays, so a leaf holding a solver (and any composed tree containing it)
# prints compactly.
function Base.show(io::IO, gl::GaussLegendre)
    print(io, "GaussLegendre(", gl.n, ")")
    return nothing
end

# Reduce an integrand `g` over the reference domain `[-1, 1]` against the
# `rule`. Seeding `acc` with `weights[1] * g(nodes[1])` fixes the
# accumulator's element type from the integrand itself, so a component
# `Dual` flows into the result rather than being forced to `Float64`.
@inline function _gl_reduce(g::G, rule::_GL) where {G}
    n, w = rule.nodes, rule.weights
    @inbounds acc = w[1] * g(n[1])
    @inbounds for i in 2:length(n)
        acc += w[i] * g(n[i])
    end
    return acc
end

@doc "

Integrate a scalar function `f` over `[lo, hi]` by fixed-node
Gauss-Legendre quadrature.

The reference domain `[-1, 1]` is mapped onto `[lo, hi]` inside the
integrand and the result reduced as a weighted dot product, so the
accumulator's element type is taken from the integrand and AD `Dual`s and
tangents propagate. Returns a typed zero when `hi <= lo`.

# Arguments
- `f`: The scalar integrand.
- `lo`: Lower integration bound.
- `hi`: Upper integration bound.
- `rule`: Gauss-Legendre rule to use (default: the 192-node convolution
  rule).

# Examples
```@example
using CensoredDistributions

# Integrate x^2 over [0, 1] (exact value 1/3)
approx = CensoredDistributions.gl_integrate(x -> x^2, 0.0, 1.0)
```

# See also
- [`GaussLegendre`](@ref): The default solver wrapping this rule.
- [`integrate`](@ref): The pluggable entry point.
"
function gl_integrate(f::F, lo, hi, rule::_GL = _CONVOLVED_GL) where {F}
    hi <= lo && return zero(f(lo))
    h = (hi - lo) / 2
    m = (lo + hi) / 2
    return h * _gl_reduce(s -> f(m + h * s), rule)
end

@doc "

Integrate a scalar function `f` over `[lower, upper]` using `solver`.

This is the pluggable integration entry point: every integral in the
package calls `integrate`, dispatching on the solver type. The default
[`GaussLegendre`](@ref) solver routes to [`gl_integrate`](@ref). Loading
the optional Integrals.jl extension adds a method for Integrals.jl
algorithms (e.g. `QuadGKJL()`), which builds an `IntegralProblem` and
calls `solve`.

# Arguments
- `solver`: The integration backend. `GaussLegendre` by default; any
  Integrals.jl algorithm when that extension is loaded.
- `f`: The scalar integrand.
- `lower`: Lower integration bound.
- `upper`: Upper integration bound.

# Examples
```@example
using CensoredDistributions

solver = CensoredDistributions.GaussLegendre(; n = 64)
approx = CensoredDistributions.integrate(solver, x -> x^2, 0.0, 1.0)
```

# See also
- [`GaussLegendre`](@ref): The default solver.
- [`gl_integrate`](@ref): The default quadrature reduction.
"
function integrate(solver::GaussLegendre, f::F, lower, upper) where {F}
    return gl_integrate(f, lower, upper, solver.rule)
end
