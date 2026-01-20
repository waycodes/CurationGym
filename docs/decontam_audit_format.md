# Decontamination Audit Format

Audit reports are generated at `artifacts/decontam/` or `runs/<id>/decontam/`.

## Report Structure (decontam_report.json)

```json
{
  "summary": {
    "docs_checked": 100000,
    "docs_contaminated": 150,
    "contamination_rate": 0.0015,
    "docs_dropped": 150,
    "docs_redacted": 0,
    "docs_downweighted": 0,
    "docs_tagged": 0,
    "by_eval_source": {
      "hellaswag": 45,
      "mmlu": 80,
      "arc_easy": 25
    }
  },
  "flagged_documents": [
    {
      "doc_id": "abc123",
      "eval_source": "mmlu",
      "overlap_score": 0.92,
      "matched_ngrams": [
        "the capital of france is paris which",
        "paris which is located in the northern"
      ],
      "action_taken": "dropped",
      "doc_preview": "The capital of France is Paris, which is located..."
    }
  ]
}
```

## Fields

### Summary
- `docs_checked`: Total documents processed
- `docs_contaminated`: Documents flagged as contaminated
- `contamination_rate`: Fraction contaminated
- `docs_dropped/redacted/downweighted/tagged`: Count by action
- `by_eval_source`: Contamination count per benchmark

### Flagged Documents
- `doc_id`: Document identifier
- `eval_source`: Primary benchmark source of contamination
- `overlap_score`: Fraction of doc n-grams matching eval set
- `matched_ngrams`: Sample of matching n-grams (max 10)
- `action_taken`: What was done (dropped, redacted, etc.)
- `doc_preview`: First 200 chars for manual review

## Additional Files

- `flagged_doc_ids.txt`: Plain list of contaminated doc IDs
- `decontam_report.parquet`: Full report in Parquet (optional)
