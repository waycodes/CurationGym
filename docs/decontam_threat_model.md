# Decontamination Threat Model

## Goal

Prevent training data from containing evaluation benchmark content, which would artificially inflate benchmark scores and misrepresent model capabilities.

## Contamination Types

### 1. Exact Contamination
- Training data contains verbatim copies of eval examples
- Detection: Exact string matching or hash lookup
- Severity: High - directly inflates scores

### 2. Near-Exact Contamination  
- Training data contains slightly modified eval examples
- Detection: N-gram overlap (default: 13-gram, GPT-3 precedent)
- Severity: High - model may memorize patterns

### 3. Semantic Contamination
- Training data contains paraphrased or reformulated eval content
- Detection: Embedding similarity (optional, expensive)
- Severity: Medium - harder to detect, may still help model

## Detection Methods

### N-gram Overlap (Default)
- Build hash set of all n-grams from eval datasets
- For each training document, compute fraction of n-grams matching eval set
- Flag documents exceeding threshold (default: 80% overlap)
- GPT-3 used 13-gram overlap; we follow this precedent

### Exact Substring (Alternative)
- GPT-4 reportedly used ~50 character overlap signal
- Faster but less sensitive to reformatting

## Actions on Contaminated Documents

| Mode | Action | Use Case |
|------|--------|----------|
| drop | Remove document entirely | Conservative, safest |
| redact | Remove contaminated spans | Preserve non-contaminated content |
| downweight | Reduce sampling probability | Soft penalty |
| tag | Mark but keep | Audit-only, no filtering |

## Limitations

1. Cannot detect semantic contamination without embeddings
2. N-gram overlap may have false positives on common phrases
3. Eval set versions must be tracked - updates invalidate decontam

## Recommendations

1. Use 13-gram overlap with 80% threshold as baseline
2. Track eval dataset versions in manifest
3. Generate audit reports for manual review
4. Consider stricter thresholds for high-stakes benchmarks
