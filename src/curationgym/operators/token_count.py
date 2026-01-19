"""Token counting annotation operator."""

from curationgym.core.document import Document


class TokenCounter:
    """Count tokens in documents using a specified tokenizer."""

    def __init__(self, tokenizer_name: str = "gpt2"):
        self.tokenizer_name = tokenizer_name
        self._tokenizer = None

    def _load_tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is not None:
            return

        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("transformers required: pip install transformers")

        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

    def count(self, text: str) -> int:
        """Count tokens in text."""
        self._load_tokenizer()
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def __call__(self, doc: Document) -> Document:
        """Add token_count to document metadata."""
        token_count = self.count(doc.text)
        doc.metadata["token_count"] = token_count
        doc.metadata["tokenizer"] = self.tokenizer_name
        return doc


def count_tokens(text: str, tokenizer_name: str = "gpt2") -> int:
    """Convenience function for token counting."""
    counter = TokenCounter(tokenizer_name)
    return counter.count(text)
