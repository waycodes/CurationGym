"""Operators for document processing in CurationGym."""

from curationgym.operators.extract_text import TextExtractor, extract_text
from curationgym.operators.url_filter import URLFilter, create_url_filter

__all__ = ["TextExtractor", "extract_text", "URLFilter", "create_url_filter"]