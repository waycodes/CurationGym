"""Heuristic quality filtering pack (Gopher, C4, FineWeb-style)."""

import re
from collections import Counter
from dataclasses import dataclass, field

from curationgym.core.document import Document


@dataclass
class QualityConfig:
    """Configuration for quality filters."""

    # Gopher-style repetition filters
    max_word_rep_ratio: float = 0.2
    max_line_rep_ratio: float = 0.3
    max_paragraph_rep_ratio: float = 0.3
    max_char_rep_ratio: float = 0.2

    # Length filters
    min_words: int = 50
    max_words: int = 100000
    min_avg_word_length: float = 3.0
    max_avg_word_length: float = 10.0

    # C4-style filters
    min_sentence_end_ratio: float = 0.1  # Sentences ending with punctuation
    max_ellipsis_ratio: float = 0.1
    max_bullet_ratio: float = 0.9

    # FineWeb custom filters
    max_curly_brace_ratio: float = 0.05  # Code/template detection
    max_digit_ratio: float = 0.3
    min_alpha_ratio: float = 0.6

    # Enabled rules
    enabled_rules: list[str] = field(default_factory=lambda: [
        "word_rep", "line_rep", "char_rep",
        "min_words", "max_words", "avg_word_length",
        "sentence_end", "ellipsis", "bullet",
        "curly_brace", "digit_ratio", "alpha_ratio",
    ])


class HeuristicQualityFilter:
    """Apply heuristic quality filters based on Gopher, C4, and FineWeb."""

    def __init__(self, config: QualityConfig | None = None):
        self.config = config or QualityConfig()
        self._rules = {
            "word_rep": self._check_word_repetition,
            "line_rep": self._check_line_repetition,
            "char_rep": self._check_char_repetition,
            "min_words": self._check_min_words,
            "max_words": self._check_max_words,
            "avg_word_length": self._check_avg_word_length,
            "sentence_end": self._check_sentence_endings,
            "ellipsis": self._check_ellipsis,
            "bullet": self._check_bullets,
            "curly_brace": self._check_curly_braces,
            "digit_ratio": self._check_digit_ratio,
            "alpha_ratio": self._check_alpha_ratio,
        }

    def __call__(self, doc: Document) -> tuple[bool, dict[str, float], list[str]]:
        """Check document quality.

        Returns:
            (passes, scores, failed_rules)
        """
        text = doc.text
        scores: dict[str, float] = {}
        failed: list[str] = []

        for rule_name in self.config.enabled_rules:
            if rule_name in self._rules:
                passes, score = self._rules[rule_name](text)
                scores[rule_name] = score
                if not passes:
                    failed.append(rule_name)

        return len(failed) == 0, scores, failed

    def filter(self, doc: Document) -> Document | None:
        """Filter document, returning None if rejected."""
        passes, scores, failed = self(doc)

        doc.metadata["quality_scores"] = scores
        if not passes:
            doc.metadata["quality_filter_rejected"] = True
            doc.metadata["quality_filter_failed_rules"] = failed
            return None

        return doc

    def _get_words(self, text: str) -> list[str]:
        return text.split()

    def _get_lines(self, text: str) -> list[str]:
        return [l for l in text.split("\n") if l.strip()]

    # Gopher-style repetition checks
    def _check_word_repetition(self, text: str) -> tuple[bool, float]:
        words = self._get_words(text)
        if len(words) < 10:
            return True, 0.0
        counts = Counter(words)
        most_common_ratio = counts.most_common(1)[0][1] / len(words) if words else 0
        return most_common_ratio <= self.config.max_word_rep_ratio, most_common_ratio

    def _check_line_repetition(self, text: str) -> tuple[bool, float]:
        lines = self._get_lines(text)
        if len(lines) < 3:
            return True, 0.0
        counts = Counter(lines)
        dup_lines = sum(c - 1 for c in counts.values() if c > 1)
        ratio = dup_lines / len(lines) if lines else 0
        return ratio <= self.config.max_line_rep_ratio, ratio

    def _check_char_repetition(self, text: str) -> tuple[bool, float]:
        if len(text) < 100:
            return True, 0.0
        # Check for repeated character sequences
        pattern = r"(.)\1{9,}"  # 10+ repeated chars
        matches = re.findall(pattern, text)
        ratio = len("".join(matches)) * 10 / len(text) if text else 0
        return ratio <= self.config.max_char_rep_ratio, ratio

    # Length checks
    def _check_min_words(self, text: str) -> tuple[bool, float]:
        count = len(self._get_words(text))
        return count >= self.config.min_words, float(count)

    def _check_max_words(self, text: str) -> tuple[bool, float]:
        count = len(self._get_words(text))
        return count <= self.config.max_words, float(count)

    def _check_avg_word_length(self, text: str) -> tuple[bool, float]:
        words = self._get_words(text)
        if not words:
            return True, 0.0
        avg = sum(len(w) for w in words) / len(words)
        passes = self.config.min_avg_word_length <= avg <= self.config.max_avg_word_length
        return passes, avg

    # C4-style checks
    def _check_sentence_endings(self, text: str) -> tuple[bool, float]:
        lines = self._get_lines(text)
        if not lines:
            return True, 0.0
        ending_punct = sum(1 for l in lines if l.rstrip()[-1:] in ".!?")
        ratio = ending_punct / len(lines)
        return ratio >= self.config.min_sentence_end_ratio, ratio

    def _check_ellipsis(self, text: str) -> tuple[bool, float]:
        lines = self._get_lines(text)
        if not lines:
            return True, 0.0
        ellipsis_lines = sum(1 for l in lines if "..." in l or "…" in l)
        ratio = ellipsis_lines / len(lines)
        return ratio <= self.config.max_ellipsis_ratio, ratio

    def _check_bullets(self, text: str) -> tuple[bool, float]:
        lines = self._get_lines(text)
        if not lines:
            return True, 0.0
        bullet_pattern = r"^[\s]*[-*•◦▪▸►]"
        bullet_lines = sum(1 for l in lines if re.match(bullet_pattern, l))
        ratio = bullet_lines / len(lines)
        return ratio <= self.config.max_bullet_ratio, ratio

    # FineWeb custom checks
    def _check_curly_braces(self, text: str) -> tuple[bool, float]:
        if not text:
            return True, 0.0
        ratio = (text.count("{") + text.count("}")) / len(text)
        return ratio <= self.config.max_curly_brace_ratio, ratio

    def _check_digit_ratio(self, text: str) -> tuple[bool, float]:
        if not text:
            return True, 0.0
        digits = sum(1 for c in text if c.isdigit())
        ratio = digits / len(text)
        return ratio <= self.config.max_digit_ratio, ratio

    def _check_alpha_ratio(self, text: str) -> tuple[bool, float]:
        if not text:
            return True, 0.0
        alpha = sum(1 for c in text if c.isalpha())
        ratio = alpha / len(text)
        return ratio >= self.config.min_alpha_ratio, ratio
