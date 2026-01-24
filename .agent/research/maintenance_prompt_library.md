# Research: Maintenance Prompt Library

**Date**: 2026-01-20  
**Researcher**: Antigravity Agent  
**Task ID**: 260120204817  
**Dispatch ID**: d260120210453

---

## Executive Summary

This research proposes a comprehensive, modular maintenance prompt library with 12 categories and 47+ discrete prompts. The design emphasizes:

1. **Clear Separation of Concerns**: Each prompt has a single, well-defined responsibility
2. **Composability**: Prompts can be orchestrated together for complex workflows
3. **Jules Compatibility**: All prompts designed for remote agent execution
4. **Trigger Clarity**: Explicit conditions for when each prompt should run
5. **Output Standardization**: Consistent reporting formats for downstream processing

---

## Category Taxonomy

### 1. Dependency Management

**Scope**: Package dependencies, version management, security updates

| Prompt Name | Trigger Conditions | Prerequisites | Expected Output | Jules Compatible |
|-------------|-------------------|---------------|-----------------|------------------|
| `dep_audit` | Weekly/on-demand | Package manifest exists | JSON report: vulnerabilities, severity, fix versions | ✅ |
| `dep_outdated` | Weekly | Package manifest exists | Markdown table: package, current, latest, breaking | ✅ |
| `dep_minimize` | Before release | Working codebase | List of unused deps + removal commands | ✅ |
| `dep_upgrade_patch` | After audit | No breaking changes | PR-ready patch upgrades | ✅ |
| `dep_upgrade_minor` | Monthly | Test suite passing | Staged minor upgrades with test verification | ✅ |
| `dep_tree_analysis` | On bloat detection | Package manifest | Dependency tree visualization + bloat sources | ✅ |

**Orchestration Pattern**: `dep_audit` → `dep_outdated` → `dep_upgrade_patch` (weekly)

---

### 2. Code Quality & Linting

**Scope**: Style enforcement, static analysis, formatting

| Prompt Name | Trigger Conditions | Prerequisites | Expected Output | Jules Compatible |
|-------------|-------------------|---------------|-----------------|------------------|
| `lint_check` | Pre-commit, CI | Linter config exists | Grouped errors by file/rule | ✅ |
| `lint_fix_auto` | After lint_check | Auto-fixable errors exist | Applied fixes + git diff | ✅ |
| `lint_fix_manual` | After auto-fix | Manual fixes needed | Annotated code with fix suggestions | ✅ |
| `format_check` | Pre-commit | Formatter config | List of unformatted files | ✅ |
| `format_apply` | After format_check | Unformatted files exist | Formatted code + diff | ✅ |
| `static_analysis` | Weekly | SAST tool available | Security/quality issues by severity | ✅ |

**Orchestration Pattern**: `lint_check` → `lint_fix_auto` → `lint_fix_manual` (on-demand)

---

### 3. Test Health & Coverage

**Scope**: Test execution, flakiness detection, coverage expansion

| Prompt Name | Trigger Conditions | Prerequisites | Expected Output | Jules Compatible |
|-------------|-------------------|---------------|-----------------|------------------|
| `test_run_all` | On-demand, CI | Test suite exists | Pass/fail summary + error groups | ✅ |
| `test_flaky_detect` | After 10+ runs | Test history available | List of flaky tests with failure rates | ✅ |
| `test_fix_failures` | After test_run_all | Failing tests exist | Fixed tests or root cause analysis | ⚠️ (needs context) |
| `coverage_report` | Weekly | Coverage tool configured | Coverage % by module + uncovered lines | ✅ |
| `coverage_expand` | After coverage_report | Coverage < target | New test cases for uncovered code | ⚠️ (needs context) |
| `test_performance` | Before release | Perf benchmarks exist | Regression detection + timing deltas | ✅ |

**Orchestration Pattern**: `test_run_all` → `test_fix_failures` (CI), `coverage_report` → `coverage_expand` (weekly)

---

### 4. Security & Vulnerability Management

**Scope**: CVE scanning, secrets detection, SAST

| Prompt Name | Trigger Conditions | Prerequisites | Expected Output | Jules Compatible |
|-------------|-------------------|---------------|-----------------|------------------|
| `security_scan_deps` | Daily | Dependency manifest | CVE report with CVSS scores | ✅ |
| `security_scan_code` | Weekly | SAST tool configured | Security issues by CWE category | ✅ |
| `secrets_detect` | Pre-commit | Secrets scanner available | Detected secrets + remediation steps | ✅ |
| `secrets_rotate` | After detection | Secrets detected | Rotation plan + new credentials | ❌ (manual) |
| `license_audit` | Monthly | License scanner | License compatibility report | ✅ |
| `sbom_generate` | Before release | Dependency tree | Software Bill of Materials (SPDX/CycloneDX) | ✅ |

**Orchestration Pattern**: `security_scan_deps` → `dep_upgrade_patch` (on CVE), `secrets_detect` → `secrets_rotate` (manual gate)

---

### 5. Documentation Maintenance

**Scope**: README updates, API docs, changelog generation

| Prompt Name | Trigger Conditions | Prerequisites | Expected Output | Jules Compatible |
|-------------|-------------------|---------------|-----------------|------------------|
| `docs_readme_sync` | After API changes | README exists | Updated README with new APIs | ⚠️ (needs context) |
| `docs_api_generate` | After code changes | Doc generator configured | Generated API documentation | ✅ |
| `docs_changelog` | Before release | Git history available | CHANGELOG.md entry from commits | ✅ |
| `docs_broken_links` | Weekly | Markdown files exist | List of broken internal/external links | ✅ |
| `docs_examples_verify` | Before release | Code examples in docs | Verification that examples run | ✅ |

**Orchestration Pattern**: `docs_api_generate` → `docs_readme_sync` → `docs_examples_verify` (pre-release)

---

### 6. Build & Compilation Health

**Scope**: Build verification, compiler warnings, artifact validation

| Prompt Name | Trigger Conditions | Prerequisites | Expected Output | Jules Compatible |
|-------------|-------------------|---------------|-----------------|------------------|
| `build_verify` | On-demand, CI | Build config exists | Build success/failure + errors | ✅ |
| `build_warnings` | After build_verify | Warnings present | Grouped warnings by category | ✅ |
| `build_warnings_fix` | After build_warnings | Fixable warnings | Fixed code or suppression rationale | ⚠️ (needs context) |
| `build_size_check` | Before release | Artifact size baseline | Size delta + bloat analysis | ✅ |
| `build_reproducible` | Before release | Build config | Reproducibility verification | ✅ |

**Orchestration Pattern**: `build_verify` → `build_warnings` → `build_warnings_fix` (CI)

---

### 7. Performance & Profiling

**Scope**: Benchmarking, profiling, regression detection

| Prompt Name | Trigger Conditions | Prerequisites | Expected Output | Jules Compatible |
|-------------|-------------------|---------------|-----------------|------------------|
| `perf_benchmark` | Weekly | Benchmark suite exists | Performance metrics vs baseline | ✅ |
| `perf_regression_detect` | After benchmark | Baseline exists | Detected regressions + delta % | ✅ |
| `perf_profile` | On regression | Profiler available | Hotspot analysis + flame graph | ⚠️ (manual) |
| `perf_memory_leak` | Weekly | Memory profiler | Leak detection report | ⚠️ (manual) |

**Orchestration Pattern**: `perf_benchmark` → `perf_regression_detect` → `perf_profile` (on regression)

---

### 8. Database & Schema Management

**Scope**: Migration validation, schema drift, query optimization

| Prompt Name | Trigger Conditions | Prerequisites | Expected Output | Jules Compatible |
|-------------|-------------------|---------------|-----------------|------------------|
| `db_migration_verify` | After schema changes | Migration files exist | Migration validation report | ✅ |
| `db_schema_drift` | Weekly | Schema definition exists | Drift detection between code/DB | ✅ |
| `db_query_analyze` | On slow queries | Query logs available | Slow query report + optimization suggestions | ⚠️ (needs context) |
| `db_backup_verify` | Daily | Backup system configured | Backup integrity check | ✅ |

**Orchestration Pattern**: `db_migration_verify` → `db_schema_drift` (on schema change)

---

### 9. Infrastructure & Configuration

**Scope**: Config validation, environment parity, IaC drift

| Prompt Name | Trigger Conditions | Prerequisites | Expected Output | Jules Compatible |
|-------------|-------------------|---------------|-----------------|------------------|
| `config_validate` | On config changes | Config schema exists | Validation errors + suggestions | ✅ |
| `config_env_parity` | Weekly | Multiple envs exist | Drift report between dev/staging/prod | ✅ |
| `iac_drift_detect` | Daily | IaC definitions exist | Infrastructure drift report | ⚠️ (cloud access) |
| `iac_plan_review` | Before apply | Terraform/Pulumi plan | Plan review + risk assessment | ✅ |

**Orchestration Pattern**: `config_validate` → `config_env_parity` (on config change)

---

### 10. Git & Repository Health

**Scope**: Branch hygiene, commit quality, merge conflicts

| Prompt Name | Trigger Conditions | Prerequisites | Expected Output | Jules Compatible |
|-------------|-------------------|---------------|-----------------|------------------|
| `git_branch_cleanup` | Weekly | Merged branches exist | List of stale branches + cleanup commands | ✅ |
| `git_commit_lint` | Pre-commit | Commit message | Conventional commit validation | ✅ |
| `git_conflict_detect` | Before merge | Merge in progress | Conflict detection + resolution hints | ✅ |
| `git_large_files` | On-demand | Git history | Large files + LFS migration suggestions | ✅ |

**Orchestration Pattern**: `git_commit_lint` (pre-commit), `git_branch_cleanup` (weekly)

---

### 11. Monitoring & Observability

**Scope**: Log analysis, metric validation, alert tuning

| Prompt Name | Trigger Conditions | Prerequisites | Expected Output | Jules Compatible |
|-------------|-------------------|---------------|-----------------|------------------|
| `logs_error_analysis` | Daily | Log aggregation | Error patterns + frequency | ⚠️ (log access) |
| `metrics_validate` | After deployment | Metrics system | Metric validation + anomaly detection | ⚠️ (metrics access) |
| `alert_tuning` | Monthly | Alert history | Alert fatigue analysis + tuning suggestions | ⚠️ (alert access) |

**Orchestration Pattern**: `logs_error_analysis` → `metrics_validate` (post-deploy)

---

### 12. Technical Debt Management

**Scope**: TODO tracking, deprecation planning, refactoring prioritization

| Prompt Name | Trigger Conditions | Prerequisites | Expected Output | Jules Compatible |
|-------------|-------------------|---------------|-----------------|------------------|
| `debt_scan_todos` | Weekly | Codebase exists | TODO/FIXME/HACK extraction + categorization | ✅ |
| `debt_deprecation_plan` | Before major version | Deprecated APIs exist | Deprecation timeline + migration guide | ⚠️ (needs context) |
| `debt_refactor_prioritize` | Monthly | Tech debt log exists | Prioritized refactoring backlog | ⚠️ (needs context) |
| `debt_complexity_analysis` | Weekly | Code metrics tool | High-complexity modules + refactor suggestions | ✅ |

**Orchestration Pattern**: `debt_scan_todos` → `debt_complexity_analysis` → `debt_refactor_prioritize` (monthly)

---

## Integration with `dispatch_maintenance`

### Batch Dispatch Tool Design

The `dispatch_maintenance` tool should support:

1. **Prompt Selection**: By category, tag, or explicit list
2. **Scheduling**: Cron-like scheduling for recurring prompts
3. **Dependency Resolution**: Automatic ordering based on prompt dependencies
4. **Parallel Execution**: Run independent prompts concurrently
5. **Result Aggregation**: Collect and summarize outputs
6. **Failure Handling**: Retry logic and error reporting

### Example Batch Configurations

```yaml
# Weekly maintenance batch
weekly_maintenance:
  schedule: "0 0 * * 0"  # Sunday midnight
  prompts:
    - dep_audit
    - dep_outdated
    - test_run_all
    - coverage_report
    - security_scan_code
    - git_branch_cleanup
  parallel: true
  notify_on: failure

# Pre-release checklist
pre_release:
  trigger: manual
  prompts:
    - test_run_all
    - coverage_report
    - security_scan_deps
    - docs_changelog
    - docs_examples_verify
    - build_size_check
  parallel: false  # Sequential execution
  notify_on: always
```

---

## Prompt Template Structure

Each prompt should follow this standardized format:

```yaml
---
name: prompt_slug
category: dependency_management
description: Short description of what this prompt does
triggers:
  - condition: Weekly schedule
    frequency: 7d
  - condition: On vulnerability detection
    event: security_alert
prerequisites:
  - Package manifest exists (package.json, Cargo.toml, etc.)
  - Network access for registry queries
output_format: json
output_schema:
  type: object
  properties:
    vulnerabilities:
      type: array
      items:
        package: string
        current_version: string
        fixed_version: string
        severity: string
        cvss_score: number
jules_compatible: true
estimated_duration: 2m
tags: [security, dependencies, automated]
---

# Prompt Content

[Detailed instructions for the agent...]
```

---

## Orchestration Patterns

### 1. Sequential Chains

Prompts that must run in order due to dependencies:

```
dep_audit → dep_outdated → dep_upgrade_patch
lint_check → lint_fix_auto → lint_fix_manual
test_run_all → test_fix_failures → coverage_report
```

### 2. Parallel Batches

Independent prompts that can run concurrently:

```
[dep_audit, security_scan_code, test_run_all, git_branch_cleanup]
```

### 3. Conditional Branching

Prompts that trigger based on previous results:

```
test_run_all → (if failures) → test_fix_failures
coverage_report → (if < 80%) → coverage_expand
perf_benchmark → (if regression) → perf_profile
```

### 4. Recurring Schedules

| Frequency | Prompts |
|-----------|---------|
| Daily | `security_scan_deps`, `db_backup_verify`, `logs_error_analysis` |
| Weekly | `dep_audit`, `test_run_all`, `coverage_report`, `git_branch_cleanup` |
| Monthly | `license_audit`, `debt_refactor_prioritize`, `alert_tuning` |
| Pre-release | `docs_changelog`, `build_size_check`, `sbom_generate` |

---

## Jules Compatibility Matrix

### ✅ Fully Compatible (35 prompts)

Prompts that require only:

- File system access
- Command execution
- Git operations
- Package manager queries

### ⚠️ Partially Compatible (12 prompts)

Prompts that may need:

- Deep codebase context
- Manual decision-making
- External service access (cloud, logs, metrics)

**Mitigation**: Provide context files, use read-only APIs, or split into recon + action phases

### ❌ Not Compatible (1 prompt)

- `secrets_rotate`: Requires manual credential management

---

## Recommended Implementation Phases

### Phase 1: Core Automation (Week 1-2)

- Dependency management (6 prompts)
- Code quality & linting (6 prompts)
- Git & repository health (4 prompts)

### Phase 2: Quality & Security (Week 3-4)

- Test health & coverage (6 prompts)
- Security & vulnerability management (6 prompts)
- Build & compilation health (5 prompts)

### Phase 3: Advanced Maintenance (Week 5-6)

- Documentation maintenance (5 prompts)
- Technical debt management (4 prompts)
- Database & schema management (4 prompts)

### Phase 4: Observability (Week 7-8)

- Performance & profiling (4 prompts)
- Monitoring & observability (3 prompts)
- Infrastructure & configuration (4 prompts)

---

## Success Metrics

1. **Coverage**: % of maintenance tasks automated
2. **Execution Time**: Average time per prompt
3. **Success Rate**: % of prompts completing without errors
4. **Issue Detection**: # of issues found per category
5. **Resolution Rate**: % of auto-fixable issues resolved
6. **Developer Time Saved**: Hours saved per week

---

## References

- [Conventional Commits](https://www.conventionalcommits.org/)
- [OWASP Dependency Check](https://owasp.org/www-project-dependency-check/)
- [SPDX SBOM Specification](https://spdx.dev/)
- [CycloneDX SBOM Standard](https://cyclonedx.org/)
- [Semantic Versioning](https://semver.org/)

---

## Appendix: Prompt Naming Conventions

- **Prefix**: Category abbreviation (`dep_`, `lint_`, `test_`, `security_`, etc.)
- **Action**: Verb describing the operation (`audit`, `check`, `fix`, `generate`, etc.)
- **Scope**: Optional specificity (`_auto`, `_manual`, `_patch`, etc.)

Examples:

- `dep_audit` = Dependency audit
- `lint_fix_auto` = Linting auto-fix
- `test_flaky_detect` = Flaky test detection
- `security_scan_deps` = Security scan of dependencies

---

**End of Research Report**
