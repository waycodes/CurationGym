# CurationGym Project Charter

## Vision

A reproducible, compute-budget-aware system for searching over data curation policies using proxy-model training and standard eval suites, emitting versioned datasets with slice-to-benchmark attribution.

## MVP Scope

### In Scope
- **Text-only** policy search on small Common Crawl-derived sample (~1-10B tokens)
- Proxy LM training (50M-400M params)
- Evaluation on small benchmark subset (MMLU subset, HellaSwag, ARC-Easy, PIQA)
- Reproducible dataset artifacts with manifests
- Slice-to-benchmark attribution reports
- Single-machine execution (Ray/Slurm optional)
- Local artifact storage (HF/S3 export optional)

### Out of Scope (MVP)
- Full 240T-token scale processing
- Full 53-task DCLM evaluation suite
- Full DataComp-XL runs
- Multimodal (CLIP) pipelines (extension path preserved)
- Production deployment infrastructure

## Milestones

| Milestone | Description | Success Criteria |
|-----------|-------------|------------------|
| M1 | Repo + tooling | CI passes, CLI stubs work |
| M2 | Data pipeline | Execute policy → dataset artifact |
| M3 | Training loop | Proxy LM trains on curated data |
| M4 | Eval integration | Benchmark scores computed |
| M5 | Optimizer | Policy search finds better-than-baseline |
| M6 | Attribution | Slice→benchmark report generated |

## Constraints

- **Compute**: MVP runs on single workstation (1-4 GPUs)
- **Time**: Each proxy training ≤4 hours
- **Storage**: Intermediate artifacts ≤500GB
- **Dependencies**: Prefer existing tools (DataTrove, lm-eval-harness)

## Success Metrics

1. **Reproducibility**: Same policy + seed → identical dataset (bit-exact or documented variance)
2. **Budget awareness**: Optimizer respects token/compute caps
3. **Attribution signal**: Ridge coefficients show non-random slice→benchmark correlation
4. **Improvement**: Best policy outperforms random baseline on aggregate score

## Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Pipeline engine | DataTrove | Mature, supports local/Ray/Slurm |
| Text eval | lm-evaluation-harness | Standard, DCLM-compatible |
| Dedup default | MinHash (per-crawl) | FineWeb-validated |
| Decontam default | 13-gram overlap | GPT-3 precedent |
| Optimizer baseline | Random + multi-fidelity | Simple, debuggable |

## Risks

| Risk | Mitigation |
|------|------------|
| Eval instability | Pin eval code versions, record in manifest |
| Compute overrun | Hard budget caps, early stopping |
| Attribution noise | Sanity checks, multiple seeds |
