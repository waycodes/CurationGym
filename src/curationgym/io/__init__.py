"""I/O readers for CurationGym."""

from curationgym.io.hf_reader import HFDatasetReader
from curationgym.io.commoncrawl_reader import CommonCrawlReader

__all__ = ["HFDatasetReader", "CommonCrawlReader"]