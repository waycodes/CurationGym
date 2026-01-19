"""Hugging Face dataset reader for CurationGym."""

from collections.abc import Iterator
from typing import Any

from curationgym.core.document import Document


class HFDatasetReader:
    """Stream documents from Hugging Face datasets."""

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        text_field: str = "text",
        id_field: str | None = None,
        streaming: bool = True,
        **kwargs: Any,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.text_field = text_field
        self.id_field = id_field
        self.streaming = streaming
        self.kwargs = kwargs

    def __iter__(self) -> Iterator[Document]:
        from datasets import load_dataset

        ds = load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=self.streaming,
            **self.kwargs,
        )

        for idx, item in enumerate(ds):
            text = item.get(self.text_field, "")
            doc_id = str(item.get(self.id_field, idx)) if self.id_field else str(idx)

            metadata = {
                "source": f"hf:{self.dataset_name}",
                **{k: v for k, v in item.items() if k not in (self.text_field, self.id_field)},
            }

            yield Document(text=text, id=doc_id, metadata=metadata)

    def read(self, limit: int | None = None) -> Iterator[Document]:
        """Read documents with optional limit."""
        for i, doc in enumerate(self):
            if limit and i >= limit:
                break
            yield doc
