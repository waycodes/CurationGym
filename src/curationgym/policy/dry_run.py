"""Policy dry run for quick validation."""

from collections.abc import Iterator
from dataclasses import dataclass

from curationgym.core.document import Document
from curationgym.policy.schema import Policy
from curationgym.operators import LanguageFilter, TokenCounter, HeuristicQualityFilter, QualityConfig


@dataclass
class DryRunReport:
    """Results from policy dry run."""

    docs_sampled: int
    docs_passed_lang: int
    docs_passed_quality: int
    estimated_retention_rate: float
    estimated_tokens_kept: int
    avg_tokens_per_doc: float


def dry_run_policy(
    policy: Policy,
    input_docs: Iterator[Document],
    sample_size: int = 1000,
) -> DryRunReport:
    """Run policy on sample to estimate retention rates.

    Args:
        policy: Policy to test
        input_docs: Input documents
        sample_size: Number of docs to sample

    Returns:
        DryRunReport with estimated metrics
    """
    lang_filter = LanguageFilter(target_lang=policy.language, min_score=policy.min_language_score)
    token_counter = TokenCounter()
    quality_filter = HeuristicQualityFilter(QualityConfig(
        min_words=policy.quality.min_words,
        max_words=policy.quality.max_words,
        enabled_rules=policy.quality.enabled_rules,
    ))

    docs_sampled = 0
    docs_passed_lang = 0
    docs_passed_quality = 0
    total_tokens = 0

    for doc in input_docs:
        if docs_sampled >= sample_size:
            break

        docs_sampled += 1

        # Language check
        doc = lang_filter.annotate(doc)
        if doc.language != policy.language or (doc.language_score or 0) < policy.min_language_score:
            continue
        docs_passed_lang += 1

        # Token count
        doc = token_counter(doc)

        # Quality check
        if quality_filter.filter(doc) is None:
            continue
        docs_passed_quality += 1
        total_tokens += doc.metadata.get("token_count", 0)

    retention_rate = docs_passed_quality / docs_sampled if docs_sampled > 0 else 0
    avg_tokens = total_tokens / docs_passed_quality if docs_passed_quality > 0 else 0

    return DryRunReport(
        docs_sampled=docs_sampled,
        docs_passed_lang=docs_passed_lang,
        docs_passed_quality=docs_passed_quality,
        estimated_retention_rate=retention_rate,
        estimated_tokens_kept=int(retention_rate * policy.max_tokens / avg_tokens) if avg_tokens > 0 else 0,
        avg_tokens_per_doc=avg_tokens,
    )
