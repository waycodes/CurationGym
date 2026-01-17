# Document Metadata Schema

CurationGym uses an extended metadata schema for tracking document provenance, quality signals, and curation state.

## Core Fields

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Document content |
| `id` | string | Unique identifier (content hash or source ID) |

## Metadata Fields

### Provenance
| Field | Type | Description |
|-------|------|-------------|
| `source` | string | Data source identifier |
| `dump` | string | Common Crawl dump (e.g., `CC-MAIN-2024-10`) |
| `url` | string | Original URL |

### Language
| Field | Type | Description |
|-------|------|-------------|
| `language` | string | ISO language code |
| `language_score` | float | Language detection confidence (0-1) |

### Size
| Field | Type | Description |
|-------|------|-------------|
| `token_count` | int | Token count (tokenizer-specific) |

### Quality
| Field | Type | Description |
|-------|------|-------------|
| `quality_scores` | dict | Quality signal scores |
| `quality_scores.gopher_rep` | float | Gopher repetition score |
| `quality_scores.c4_filter` | bool | Passes C4 filters |
| `quality_scores.perplexity` | float | Model perplexity |

### Deduplication
| Field | Type | Description |
|-------|------|-------------|
| `dedup_cluster_id` | string | Cluster ID for near-duplicates |
| `dedup_method` | string | Method used (exact, minhash, semantic) |

### Decontamination
| Field | Type | Description |
|-------|------|-------------|
| `contamination_flags` | dict | Per-benchmark contamination info |
| `contamination_flags.<benchmark>` | dict | `{overlap_score, action, matched_ngrams}` |

### Slices
| Field | Type | Description |
|-------|------|-------------|
| `slice_tags` | list[string] | Assigned slice identifiers |

## Example

```json
{
  "text": "Document content here...",
  "id": "abc123def456",
  "metadata": {
    "source": "commoncrawl",
    "dump": "CC-MAIN-2024-10",
    "url": "https://example.com/page",
    "language": "en",
    "language_score": 0.98,
    "token_count": 512,
    "quality_scores": {
      "gopher_rep": 0.02,
      "c4_filter": true
    },
    "dedup_cluster_id": "cluster_789",
    "contamination_flags": {},
    "slice_tags": ["domain=edu", "quality_bin=high", "dump=CC-MAIN-2024-10"]
  }
}
```
