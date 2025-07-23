# [Release Process](@id release-process)

CensoredDistributions.jl follows the standard Julia package ecosystem approach for releases, emphasising regular releases, semantic versioning, and automated tooling.

## Release Philosophy

- **Release frequently**: Ideally after every merged PR to main
- **Semantic versioning**: Follow [SemVer](https://semver.org/) strictly
- **GitHub releases as changelog**: Use release notes to document changes
- **Automated tooling**: Leverage standard Julia ecosystem tools

**When NOT to release**:
- Work-in-progress features
- Failing tests
- Incomplete documentation for new features
- During Julia ecosystem-wide changes (wait for stability)

## Release Workflow

### 1. Standard Julia Tooling

We use the same tools as major Julia packages (including the SciML ecosystem):

- **CompatHelper**: Automated dependency updates via daily PRs
- **@JuliaRegistrator**: Manual package registration to Julia General Registry
- **TagBot**: Automated GitHub release creation after registry registration

### 2. Release Steps

#### For Regular Releases (Patch/Minor)

1. **Ensure all tests pass** and pre-commit hooks are clean
2. **Update version** in `Project.toml`:
   ```toml
   version = "1.2.3"  # Follow semantic versioning
   ```
3. **Commit and merge** to main via PR
4. **Trigger registration** by commenting on the merge commit:
   ```
   @JuliaRegistrator register
   ```
5. **Automated flow**:
   - JuliaRegistrator creates PR to Julia General Registry
   - Once merged (~15 minutes), TagBot automatically creates GitHub release
   - Documentation is automatically deployed for new tagged versions

#### For Major Releases (Breaking Changes)

1. **Plan breaking changes** carefully with issue discussion
2. **Update version** with major version bump (e.g., `1.5.2` → `2.0.0`)
3. **Write comprehensive release notes** highlighting breaking changes
4. **Follow same registration process** as above
5. **Consider deprecation warnings** in the previous minor release

### 3. Semantic Versioning Guidelines

Following [SemVer](https://semver.org/):

- **Patch** (`1.0.0` → `1.0.1`): Bug fixes, performance improvements
- **Minor** (`1.0.1` → `1.1.0`): New features, additions to API (backwards compatible)
- **Major** (`1.1.0` → `2.0.0`): Breaking changes to public API

#### Examples:
- Adding new distribution types: **Minor**
- Fixing calculation bugs: **Patch**
- Changing function signatures: **Major**
- Adding optional parameters: **Minor**
- Removing deprecated functions: **Major**

### 4. Release Notes and Changelog

We use **GitHub releases as our official changelog**:

#### Good Release Notes Template:
```markdown
## What's Changed

### New Features
- Add support for CustomDistribution type by @contributor

### Bug Fixes
- Fix CDF calculation edge case for zero values
- Resolve type instability in rand() method

### Performance
- 2x speedup in censored_pmf calculations

### Documentation
- Add tutorial for advanced censoring scenarios
- Improve docstring examples

**Full Changelog**: https://github.com/EpiAware/CensoredDistributions.jl/compare/v1.2.0...v1.2.1
```

### 5. Automation Details

#### CompatHelper
- Runs daily via GitHub Actions (`CompatHelper.yaml`)
- Automatically updates `[compat]` entries in `Project.toml`
- Creates PRs for dependency version bumps
- Helps maintain up-to-date dependencies across Julia ecosystem

#### TagBot
- Triggered when package appears in Julia General Registry
- Automatically creates GitHub releases with generated changelog
- Uses SSH key (`DOCUMENTER_KEY`) for authenticated operations
- Supports custom release notes if provided

#### JuliaRegistrator
- Community bot for registering packages
- Only package collaborators can trigger registration
- Validates package before creating registry PR
- Usually completes registration within 15 minutes

### 6. Pre-Release Checklist

Before triggering `@JuliaRegistrator register`:

- [ ] All CI tests pass (including pre-commit hooks)
- [ ] Version number follows semantic versioning
- [ ] Breaking changes are documented
- [ ] New features have tests and documentation
- [ ] Dependencies are up to date (check CompatHelper PRs)
- [ ] Examples in docstrings work correctly

### 7. Emergency Releases

For critical bug fixes:

1. **Create hotfix branch** from latest release tag
2. **Apply minimal fix** with tests
3. **Bump patch version** immediately
4. **Fast-track PR review** and merge
5. **Register immediately** with `@JuliaRegistrator register`

### 8. Coordinating with Julia Ecosystem

#### Registry Compatibility
- Monitor Julia ecosystem releases that might affect dependencies
- Test against Julia LTS, stable, and pre-release versions in CI
- Follow Julia deprecation cycles

#### Documentation Releases
- Major releases trigger documentation rebuilds automatically
- Ensure tutorials work with new versions
- Update installation instructions if Julia version requirements change

### 9. Troubleshooting Releases

#### Common Issues:
- **Registry PR fails**: Check Project.toml syntax and version conflicts
- **TagBot doesn't trigger**: Verify `DOCUMENTER_KEY` secret is configured
- **Tests fail after release**: Emergency patch release procedure

#### Getting Help:
- Check [Julia Discourse](https://discourse.julialang.org/) for registry issues
- SciML community practices for guidance
- GitHub Issues for package-specific problems

---

This process ensures reliable, frequent releases while maintaining high quality and ecosystem compatibility.
