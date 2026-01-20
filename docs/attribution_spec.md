# Attribution Specification

## Goal

Understand which data slices contribute to benchmark performance, enabling:
1. Informed data curation decisions
2. Debugging of unexpected benchmark results
3. Targeted data collection strategies

## Attribution Questions

### Primary Question
> "Which slices are correlated with improvements on Benchmark Y under this policy family?"

This is a **correlational** claim, not causal. We observe associations between slice composition and benchmark scores across multiple runs.

### Secondary Questions
1. Which slices hurt performance on specific benchmarks?
2. Are there slice interactions (e.g., edu + high_quality)?
3. How stable are attributions across random seeds?

## What We Do NOT Claim

- **Causation**: We cannot claim "adding slice X causes Y improvement" without controlled experiments
- **Generalization**: Attributions may not transfer to different model scales or architectures
- **Completeness**: Unmeasured confounders may exist

## Report Granularity

### Level 1: Slice-level (Default)
- Attribution per slice tag (e.g., `domain=edu`, `quality_bin=high`)
- Most interpretable, may miss interactions

### Level 2: Slice-group-level
- Group related slices (e.g., all domain slices)
- Reduces multiple testing burden

### Level 3: Feature-level (Advanced)
- Continuous features (e.g., exact quality score)
- Requires more sophisticated methods

## Output Format

### Attribution Report (JSON)
```json
{
  "method": "ridge_regression",
  "benchmark": "hellaswag",
  "coefficients": {
    "domain=edu": {"value": 0.15, "ci_low": 0.08, "ci_high": 0.22},
    "quality_bin=high": {"value": 0.23, "ci_low": 0.18, "ci_high": 0.28}
  },
  "r_squared": 0.45,
  "n_runs": 50,
  "sanity_checks": {
    "random_permutation_p": 0.02,
    "seed_stability": 0.89
  }
}
```

### Attribution Table (CSV)
| slice | benchmark | coefficient | ci_low | ci_high | significant |
|-------|-----------|-------------|--------|---------|-------------|
| domain=edu | hellaswag | 0.15 | 0.08 | 0.22 | true |
| domain=edu | mmlu | 0.08 | -0.02 | 0.18 | false |

## Methods

1. **Ridge Regression** (CG-1203): Fast, handles collinearity, provides coefficients
2. **Ablation** (CG-1204): More causal, expensive, limited to top slices
3. **Shapley** (CG-1205, optional): Handles interactions, very expensive

## Interpretation Guidelines

- Coefficients represent **marginal** contribution holding others constant
- Confidence intervals from bootstrap (1000 resamples)
- "Significant" = CI does not cross zero
- Always report R² to indicate explanatory power
- Compare to random baseline (permutation test)
