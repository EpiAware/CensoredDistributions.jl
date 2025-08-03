# ExponentiallyTilted Distribution Implementation Plan

## Current Situation Analysis

**Problem**: PR #113 is failing comprehensively due to attempting too much at once (900+ lines across multiple files including implementation, tests, tutorials, documentation).

**Root Issues Identified**:
- Monolithic PR approach causing cascade failures
- Unclear file structure placement (utils vs censoring vs distributions)
- Missing dependencies properly declared
- Attempting to implement everything at once instead of incremental approach

## Key Architectural Decision: File Structure

**Question**: Where should `ExponentiallyTilted.jl` be located?

**Options Analysis**:
1. `src/utils/` - Current PR placement, but ExponentiallyTilted is a full distribution, not a utility
2. `src/censoring/` - Makes sense if it's only used for censoring, but it's a standalone distribution
3. `src/distributions/` - New folder for standalone distributions (doesn't exist yet)
4. `src/` - Direct in src for major distributions

**Recommendation**: Create `src/distributions/` folder
- ExponentiallyTilted is a complete distribution implementing Distributions.jl interface
- It's not just a utility (like Weighted.jl)
- It's not exclusively for censoring (though that's the main use case)
- Future distributions can follow same pattern
- Keeps src/ root clean

## File Structure Plan

```
src/
├── distributions/
│   └── ExponentiallyTilted.jl          # New location
├── censoring/
│   ├── PrimaryCensored.jl              # Existing
│   ├── IntervalCensored.jl             # Existing
│   └── DoubleIntervalCensored.jl       # Existing
├── utils/
│   ├── Weighted.jl                     # Existing
│   └── get_dist.jl                     # Existing
└── CensoredDistributions.jl            # Main module
```

## Phase 1: Minimal Viable Implementation

### Scope
- Basic ExponentiallyTilted struct and constructor
- Essential Distributions.jl interface methods: pdf, logpdf, cdf, logcdf, quantile, rand
- Parameter validation and error handling
- Numerical stability for edge cases (r ≈ 0)
- Basic test suite

### Files to Create/Modify
1. **NEW**: `src/distributions/ExponentiallyTilted.jl`
2. **MODIFY**: `src/CensoredDistributions.jl` (add include and export)
3. **NEW**: `test/distributions/ExponentiallyTilted.jl`

### Dependencies to Address
- LogExpFunctions.jl (used in logpdf implementation) - needs to be added to Project.toml
- Ensure all required Distributions.jl methods are imported

### Success Criteria
- ExponentiallyTilted passes basic Distributions.jl interface tests
- All basic mathematical operations work correctly
- Numerical stability verified for edge cases
- Tests pass locally before PR creation

## Phase 2: Integration Testing

### Scope
- Test integration with primary_censored function
- Verify compatibility with AD backends
- Edge case testing and error handling
- Performance baseline establishment

### Files to Create/Modify
1. **MODIFY**: `test/distributions/ExponentiallyTilted.jl` (expand tests)
2. **MODIFY**: `test/censoring/PrimaryCensored.jl` (add ExponentiallyTilted examples)

## Phase 3: Documentation and Examples

### Scope
- Complete docstrings with mathematical background
- Basic usage examples
- Integration examples with primary censoring
- README updates (minimal)

### Files to Create/Modify
1. **MODIFY**: `src/distributions/ExponentiallyTilted.jl` (enhanced docstrings)
2. **MODIFY**: `README.md` (brief mention of new distribution)

## Phase 4: Advanced Features (Future)

### Scope
- Analytical solutions for specific distribution combinations
- Advanced moments (mean, var, entropy) if needed
- Comprehensive tutorials
- Performance optimizations

## Immediate Action Plan

### Step 1: Clean Current State
- [ ] Remove existing `src/utils/ExponentiallyTilted.jl`
- [ ] Create `src/distributions/` directory
- [ ] Revert any changes to main module file

### Step 2: Dependency Management
- [ ] Check if LogExpFunctions.jl is in Project.toml
- [ ] Add if missing with proper version constraints

### Step 3: Phase 1 Implementation
- [ ] Create minimal `src/distributions/ExponentiallyTilted.jl`
- [ ] Update `src/CensoredDistributions.jl` with proper includes/exports
- [ ] Create basic test file
- [ ] Test locally to ensure basics work

### Step 4: PR Management
- [ ] Close existing PR #113 with explanation
- [ ] Create new PR for Phase 1 with clear, limited scope
- [ ] Link to this implementation plan in PR description

## Risk Mitigation

### Technical Risks
1. **LogExpFunctions dependency**: Verify it's available and version compatible
2. **Numerical stability**: Test edge cases thoroughly (r ≈ 0, extreme values)
3. **AD compatibility**: Test with ForwardDiff.jl at minimum

### Process Risks
1. **Scope creep**: Stick rigidly to Phase 1 scope, defer everything else
2. **Testing gaps**: Focus on mathematical correctness before optimization
3. **Integration issues**: Test with existing censoring functions early

## Questions to Resolve

1. **Dependencies**: Is LogExpFunctions.jl already available? What version?
2. **Testing framework**: Should we follow existing test patterns in the codebase?
3. **Export strategy**: Should ExponentiallyTilted be exported from main module?
4. **Naming**: Is `ExponentiallyTilted` the final name or should it be `ExponentiallyTiltedDistribution`?

## Success Metrics

### Phase 1 Success
- [ ] Basic distribution works with pdf, cdf, quantile, rand
- [ ] Integration with primary_censored works
- [ ] Tests pass on CI
- [ ] No breaking changes to existing functionality
- [ ] Code review feedback is implementable

### Long-term Success
- [ ] Performance competitive with other distributions
- [ ] Clear documentation enables real-world usage
- [ ] Community adoption for epidemic modelling use cases
- [ ] Foundation for advanced features in future phases
