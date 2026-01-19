"""PII masking operator for emails and IP addresses."""

import re
from dataclasses import dataclass

from curationgym.core.document import Document


@dataclass
class PIIMaskStats:
    """Statistics from PII masking."""

    emails_masked: int = 0
    ips_masked: int = 0


class PIIMasker:
    """Mask PII (emails and public IPs) in documents."""

    # Email pattern
    EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    )

    # IPv4 pattern (public ranges only - excludes private/local)
    IPV4_PATTERN = re.compile(
        r"\b(?!10\.)(?!127\.)(?!172\.(?:1[6-9]|2[0-9]|3[01])\.)(?!192\.168\.)"
        r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    )

    def __init__(
        self,
        email_replacement: str = "{{EMAIL}}",
        ip_replacement: str = "{{IP}}",
        mask_emails: bool = True,
        mask_ips: bool = True,
    ):
        self.email_replacement = email_replacement
        self.ip_replacement = ip_replacement
        self.mask_emails = mask_emails
        self.mask_ips = mask_ips

    def __call__(self, doc: Document) -> tuple[Document, PIIMaskStats]:
        """Mask PII in document.

        Returns:
            (masked_document, stats)
        """
        text = doc.text
        stats = PIIMaskStats()

        if self.mask_emails:
            text, count = self.EMAIL_PATTERN.subn(self.email_replacement, text)
            stats.emails_masked = count

        if self.mask_ips:
            text, count = self.IPV4_PATTERN.subn(self.ip_replacement, text)
            stats.ips_masked = count

        new_metadata = {
            **doc.metadata,
            "pii_emails_masked": stats.emails_masked,
            "pii_ips_masked": stats.ips_masked,
        }

        return Document(text=text, id=doc.id, metadata=new_metadata), stats

    def mask(self, doc: Document) -> Document:
        """Mask PII and return document (discards stats)."""
        masked_doc, _ = self(doc)
        return masked_doc


def mask_pii(text: str, **kwargs) -> tuple[str, PIIMaskStats]:
    """Convenience function for PII masking."""
    doc = Document(text=text, id="temp")
    masker = PIIMasker(**kwargs)
    masked_doc, stats = masker(doc)
    return masked_doc.text, stats
