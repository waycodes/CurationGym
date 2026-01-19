"""Language detection and filtering using fastText."""

from pathlib import Path

from curationgym.core.document import Document


class LanguageFilter:
    """Detect and filter documents by language using fastText."""

    def __init__(
        self,
        target_lang: str = "en",
        min_score: float = 0.65,
        model_path: str | Path | None = None,
    ):
        self.target_lang = target_lang
        self.min_score = min_score
        self.model_path = model_path
        self._model = None

    def _load_model(self):
        """Lazy load fastText model."""
        if self._model is not None:
            return

        try:
            import fasttext
        except ImportError:
            raise ImportError("fasttext required: pip install fasttext")

        if self.model_path and Path(self.model_path).exists():
            self._model = fasttext.load_model(str(self.model_path))
        else:
            # Download lid.176.bin if not available
            import urllib.request
            model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
            cache_dir = Path.home() / ".cache" / "curationgym"
            cache_dir.mkdir(parents=True, exist_ok=True)
            model_file = cache_dir / "lid.176.bin"

            if not model_file.exists():
                urllib.request.urlretrieve(model_url, model_file)

            self._model = fasttext.load_model(str(model_file))

    def detect(self, text: str) -> tuple[str, float]:
        """Detect language of text.

        Returns:
            (language_code, confidence_score)
        """
        self._load_model()

        # Clean text for prediction (single line, no newlines)
        clean_text = " ".join(text.split())[:1000]  # Limit length
        if not clean_text:
            return "unknown", 0.0

        predictions = self._model.predict(clean_text, k=1)
        label = predictions[0][0].replace("__label__", "")
        score = float(predictions[1][0])

        return label, score

    def __call__(self, doc: Document) -> Document | None:
        """Filter document by language.

        Returns document with language metadata if passes, None if rejected.
        """
        lang, score = self.detect(doc.text)

        # Update metadata
        doc.metadata["language"] = lang
        doc.metadata["language_score"] = round(score, 4)

        # Check if passes filter
        if lang == self.target_lang and score >= self.min_score:
            return doc

        doc.metadata["lang_filter_rejected"] = True
        doc.metadata["lang_filter_reason"] = (
            f"lang={lang}" if lang != self.target_lang else f"score={score:.3f}<{self.min_score}"
        )
        return None

    def annotate(self, doc: Document) -> Document:
        """Add language metadata without filtering."""
        lang, score = self.detect(doc.text)
        doc.metadata["language"] = lang
        doc.metadata["language_score"] = round(score, 4)
        return doc


def detect_language(text: str, model_path: str | None = None) -> tuple[str, float]:
    """Convenience function for language detection."""
    filter = LanguageFilter(model_path=model_path)
    return filter.detect(text)
