# Run Artifact Layout

Each run produces artifacts in `runs/<run_id>/` with the following structure:

```
runs/<run_id>/
├── config.json          # Full run configuration
├── events.jsonl         # Structured event log
├── metrics.json         # Final aggregated metrics
├── train/
│   ├── checkpoints/     # Model checkpoints
│   └── metrics.json     # Training metrics
├── eval/
│   ├── results.json     # Per-task eval results
│   └── aggregate.json   # Aggregated scores
├── curate/
│   ├── manifest.json    # Dataset manifest
│   └── stats.json       # Curation statistics
└── attribution/
    └── report.json      # Slice attribution results
```

## Event Log Format (events.jsonl)

Each line is a JSON object:
```json
{"timestamp": "2024-01-01T00:00:00Z", "run_id": "...", "event": "config", "config": {...}}
{"timestamp": "2024-01-01T00:01:00Z", "run_id": "...", "event": "metric", "name": "loss", "value": 2.5, "step": 100}
{"timestamp": "2024-01-01T00:02:00Z", "run_id": "...", "event": "artifact", "name": "checkpoint", "path": "..."}
```
