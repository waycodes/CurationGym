"""Policy execution - the core curation loop."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from curationgym.core.document import Document
from curationgym.core.manifest import DatasetManifest
from curationgym.policy.schema import Policy
from curationgym.policy.hash import compute_policy_hash, get_code_version
from curationgym.operators import LanguageFilter, TokenCounter, HeuristicQualityFilter, QualityConfig, PIIMasker
from curationgym.operators.dedup import ScopedDedup, DedupScope, MinHashConfig
from curationgym.operators.decontam import NgramDecontaminator, DecontamMode
from curationgym.slices import assign_and_store, SliceStatsCollector
from curationgym.mixing import SliceSampler, SamplingConfig


def execute_policy(
    policy: Policy,
    input_docs: Iterator[Document],
    output_dir: str | Path,
    input_signature: str = "unknown",
) -> DatasetManifest:
    """Execute a curation policy on input documents.

    Args:
        policy: Policy configuration
        input_docs: Iterator of input documents
        output_dir: Directory for output shards and manifest
        input_signature: Signature of input data for reproducibility

    Returns:
        DatasetManifest describing the output dataset
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize operators
    lang_filter = LanguageFilter(target_lang=policy.language, min_score=policy.min_language_score)
    token_counter = TokenCounter()
    quality_filter = HeuristicQualityFilter(QualityConfig(
        min_words=policy.quality.min_words,
        max_words=policy.quality.max_words,
        min_avg_word_length=policy.quality.min_avg_word_length,
        max_word_rep_ratio=policy.quality.max_word_rep_ratio,
        min_alpha_ratio=policy.quality.min_alpha_ratio,
        enabled_rules=policy.quality.enabled_rules,
    ))
    pii_masker = PIIMasker()

    # Dedup
    dedup = ScopedDedup(
        scope=DedupScope.PER_DUMP if policy.dedup.scope == "per_dump" else DedupScope.GLOBAL,
        method=policy.dedup.method,
        minhash_config=MinHashConfig(
            num_bands=policy.dedup.num_bands,
            rows_per_band=policy.dedup.rows_per_band,
            ngram_size=policy.dedup.ngram_size,
        ),
    )

    # Decontam
    decontam = NgramDecontaminator(
        ngram_size=policy.decontam.ngram_size,
        overlap_threshold=policy.decontam.overlap_threshold,
        mode=DecontamMode(policy.decontam.mode),
    ) if policy.decontam.enabled else None

    # Sampler
    sampler = SliceSampler(SamplingConfig(
        weights=policy.slice_weights or None,
        max_tokens_per_slice=policy.max_tokens_per_slice or None,
        seed=policy.seed,
    ))

    # Stats collector
    stats_collector = SliceStatsCollector()

    # Process documents
    total_tokens = 0
    shard_idx = 0
    shard_docs: list[Document] = []
    shard_size = 10000  # Docs per shard

    def write_shard():
        nonlocal shard_idx, shard_docs
        if not shard_docs:
            return None
        shard_path = output_dir / f"shard_{shard_idx:05d}.jsonl"
        with open(shard_path, "w") as f:
            for doc in shard_docs:
                f.write(json.dumps(doc.to_dict()) + "\n")
        shard_idx += 1
        result = {"path": str(shard_path.name), "doc_count": len(shard_docs)}
        shard_docs = []
        return result

    shards = []

    for doc in input_docs:
        # Language filter
        doc = lang_filter.annotate(doc)
        if doc.language != policy.language or (doc.language_score or 0) < policy.min_language_score:
            continue

        # Token count
        doc = token_counter(doc)

        # Quality filter
        result = quality_filter.filter(doc)
        if result is None:
            continue
        doc = result

        # PII masking
        doc = pii_masker.mask(doc)

        # Slice assignment
        doc = assign_and_store(doc)

        # Dedup
        deduped = list(dedup.process(iter([doc])))
        if not deduped:
            stats_collector.add_document(doc, kept=False)
            continue
        doc = deduped[0]

        # Decontam
        if decontam:
            processed, _ = decontam.process(doc)
            if processed is None:
                stats_collector.add_document(doc, kept=False)
                continue
            doc = processed

        # Check token budget
        doc_tokens = doc.metadata.get("token_count", 0)
        if total_tokens + doc_tokens > policy.max_tokens:
            break

        # Add to sampler (handles per-slice caps)
        if not sampler.add_document(doc):
            continue

        # Collect stats and write
        stats_collector.add_document(doc, kept=True)
        total_tokens += doc_tokens
        shard_docs.append(doc)

        if len(shard_docs) >= shard_size:
            shard_info = write_shard()
            if shard_info:
                shards.append(shard_info)

    # Write final shard
    shard_info = write_shard()
    if shard_info:
        shards.append(shard_info)

    # Save slice stats
    stats_collector.save(output_dir / "slice_stats.json")

    # Create manifest
    manifest = DatasetManifest(
        dataset_id=f"{policy.name}_{compute_policy_hash(policy)}",
        policy_config=policy.to_dict(),
        random_seed=policy.seed,
    )
    manifest.compute_policy_hash()
    manifest.code_commit = get_code_version()
    manifest.input_sources = [{"signature": input_signature}]
    manifest.shards = [{"path": s["path"], "checksum": "", "doc_count": s["doc_count"]} for s in shards]
    manifest.stats = {
        "total_docs": stats_collector.total_stats.doc_count,
        "total_tokens": stats_collector.total_stats.token_count,
    }
    manifest.save(output_dir / "manifest.json")

    return manifest
