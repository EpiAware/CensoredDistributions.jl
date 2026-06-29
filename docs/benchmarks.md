<!-- PACKAGE-OWNED — the prose hook spliced into the managed benchmark page. -->

Benchmarks for distribution operation performance, comparing analytical and
numerical methods where applicable.

## Benchmark structure

```
PrimaryCensored/
  Gamma+Uniform/
    analytical/  (cdf, pdf, logpdf, rand)
    numerical/   (cdf, pdf, logpdf, rand)
  LogNormal+Uniform/
    analytical/  (cdf, pdf, logpdf, rand)
    numerical/   (cdf, pdf, logpdf, rand)
  Weibull+Uniform/
    analytical/  (cdf, pdf, logpdf, rand)
    numerical/   (cdf, pdf, logpdf, rand)
  Exponential+Uniform/
    numerical/   (no analytical solution)
  Gamma+Exponential/
    numerical/   (non-Uniform primary)
  LogNormal+Exponential/
    numerical/   (non-Uniform primary)

IntervalCensored/
  Regular/     (cdf, pdf, logpdf, rand)
  Arbitrary/   (cdf, pdf, logpdf, rand)
  Exponential/ (cdf, pdf, logpdf, rand)

DoubleIntervalCensored/
  LogNormal+Uniform/  (cdf, pdf, logpdf, rand)
  Exponential+Uniform/ (cdf, pdf, logpdf, rand)
```

## Analytical versus numerical methods

The `PrimaryCensored` distribution supports analytical solutions for certain
distribution pairs with Uniform primary events.

- Gamma delay with Uniform primary
- LogNormal delay with Uniform primary
- Weibull delay with Uniform primary

All other combinations use numerical integration.
Pass `method = NumericSolver()` to force numerical integration.

## Continuous integration

Benchmarks run automatically on pull requests using the
[AirspeedVelocity GitHub Action](https://github.com/MilesCranmer/AirspeedVelocity.jl),
comparing the pull request head against the base branch.

The action's own flat table (every benchmark, including one row per AD
scenario x backend pair) is written to the job summary rather than posted, as
it is unreadable once the AD-gradient suite is included.
`benchmark/comment/comment.jl` reads the same result JSON and posts a single
pull request comment with a "most changed" summary, a compact AD scenario x
backend ratio matrix, and the full results folded behind a `<details>` block.
