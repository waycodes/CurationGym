# Optimization Problem Formulation

## Objective

Maximize evaluation score subject to compute budget constraints:

```
maximize    score_total(policy)
subject to  compute_cost(policy) ≤ budget
            contamination_rate(policy) ≤ max_contamination
            dataset_tokens(policy) ≥ min_tokens
            diversity_score(policy) ≥ min_diversity
```

## Decision Variables

### Quality Filtering
- `min_words`: [10, 500] - Minimum word count
- `max_words`: [1000, 500000] - Maximum word count  
- `min_alpha_ratio`: [0.3, 0.9] - Minimum alphabetic character ratio
- `max_word_rep_ratio`: [0.05, 0.5] - Maximum word repetition ratio
- `enabled_rules`: subset of available quality rules

### Deduplication
- `dedup_method`: {exact, minhash, semantic}
- `dedup_scope`: {global, per_dump}
- `minhash_bands`: [5, 20] - Number of LSH bands
- `minhash_rows`: [4, 16] - Rows per band
- `keep_rule`: {first, longest, highest_quality}

### Decontamination
- `decontam_enabled`: {true, false}
- `decontam_mode`: {drop, redact, downweight, tag}
- `ngram_size`: [8, 20] - N-gram size for overlap detection
- `overlap_threshold`: [0.5, 0.95] - Contamination threshold

### Mixing
- `slice_weights`: dict[slice_tag, float] - Per-slice sampling weights
- `max_tokens`: [1B, 100B] - Total token budget
- `max_tokens_per_slice`: dict[slice_tag, int] - Per-slice caps

### Reproducibility
- `seed`: integer - Random seed

## Constraints

### Hard Constraints (must satisfy)
1. **Compute budget**: Total GPU-hours ≤ allocated budget
2. **Max contamination**: contamination_rate ≤ threshold (e.g., 0.01)

### Soft Constraints (penalized in objective)
1. **Min dataset size**: Penalize if tokens < target
2. **Min diversity**: Penalize if slice entropy < threshold

## Objective Function

```python
def objective(policy, eval_result, constraints):
    score = eval_result.aggregate_score
    
    # Penalize constraint violations
    if constraints.contamination_rate > max_contamination:
        return -inf  # Infeasible
    
    # Soft penalties
    if dataset_tokens < min_tokens:
        score *= (dataset_tokens / min_tokens)
    
    # Optional: incorporate compute cost
    if cost_aware:
        score = score / log(1 + compute_cost)
    
    return score
```

## Search Space Summary

| Variable | Type | Range | Default |
|----------|------|-------|---------|
| min_words | int | [10, 500] | 50 |
| max_words | int | [1000, 500000] | 100000 |
| min_alpha_ratio | float | [0.3, 0.9] | 0.6 |
| dedup_method | cat | {exact, minhash} | minhash |
| dedup_scope | cat | {global, per_dump} | per_dump |
| minhash_bands | int | [5, 20] | 14 |
| decontam_mode | cat | {drop, tag} | drop |
| overlap_threshold | float | [0.5, 0.95] | 0.8 |
| max_tokens | int | [1B, 100B] | 10B |
