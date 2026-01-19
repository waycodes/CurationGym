"""Operators for document processing in CurationGym."""

from curationgym.operators.extract_text import TextExtractor, extract_text
from curationgym.operators.url_filter import URLFilter, create_url_filter
from curationgym.operators.lang_filter import LanguageFilter, detect_language

__all__ = [
    "TextExtractor", "extract_text",
    "URLFilter", "create_url_filter",
    "LanguageFilter", "detect_language",
]