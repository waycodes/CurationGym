"""DataTrove adapter for CurationGym pipeline execution.

Bridges CurationGym's Document model with DataTrove's pipeline blocks.
"""

from collections.abc import Iterator
from typing import Any, Callable

from curationgym.core.document import Document


# Type alias for pipeline blocks
PipelineBlock = Callable[[Iterator[Document]], Iterator[Document]]


def document_to_datatrove(doc: Document) -> dict[str, Any]:
    """Convert CurationGym Document to DataTrove format."""
    return {"text": doc.text, "id": doc.id, "metadata": doc.metadata}


def datatrove_to_document(data: dict[str, Any]) -> Document:
    """Convert DataTrove format to CurationGym Document."""
    return Document(
        text=data.get("text", ""),
        id=data.get("id", ""),
        metadata=data.get("metadata", {}),
    )


class DataTroveAdapter:
    """Adapter wrapping DataTrove pipeline functionality.

    Provides a unified interface for building and executing pipelines
    using DataTrove's block-based architecture.
    """

    def __init__(self) -> None:
        self._blocks: list[PipelineBlock] = []

    def add_block(self, block: PipelineBlock) -> "DataTroveAdapter":
        """Add a processing block to the pipeline."""
        self._blocks.append(block)
        return self

    def add_reader(self, reader_fn: Callable[[], Iterator[Document]]) -> "DataTroveAdapter":
        """Add a reader as the first block."""
        def reader_block(_: Iterator[Document]) -> Iterator[Document]:
            yield from reader_fn()
        self._blocks.insert(0, reader_block)
        return self

    def add_filter(
        self, predicate: Callable[[Document], bool], name: str = "filter"
    ) -> "DataTroveAdapter":
        """Add a filter block."""
        def filter_block(docs: Iterator[Document]) -> Iterator[Document]:
            for doc in docs:
                if predicate(doc):
                    yield doc
        self._blocks.append(filter_block)
        return self

    def add_mapper(
        self, mapper: Callable[[Document], Document], name: str = "mapper"
    ) -> "DataTroveAdapter":
        """Add a mapper block that transforms documents."""
        def mapper_block(docs: Iterator[Document]) -> Iterator[Document]:
            for doc in docs:
                yield mapper(doc)
        self._blocks.append(mapper_block)
        return self

    def add_stats_collector(
        self, collector: Callable[[Document], None]
    ) -> "DataTroveAdapter":
        """Add a stats collection block (pass-through with side effects)."""
        def stats_block(docs: Iterator[Document]) -> Iterator[Document]:
            for doc in docs:
                collector(doc)
                yield doc
        self._blocks.append(stats_block)
        return self

    def run(self, input_docs: Iterator[Document] | None = None) -> Iterator[Document]:
        """Execute the pipeline on input documents."""
        if not self._blocks:
            if input_docs:
                yield from input_docs
            return

        # Chain blocks together
        current: Iterator[Document] = input_docs or iter([])
        for block in self._blocks:
            current = block(current)

        yield from current

    def run_to_list(self, input_docs: Iterator[Document] | None = None) -> list[Document]:
        """Execute pipeline and collect results to list."""
        return list(self.run(input_docs))

    def clear(self) -> None:
        """Clear all blocks."""
        self._blocks.clear()


def create_filter_block(predicate: Callable[[Document], bool]) -> PipelineBlock:
    """Create a filter block from a predicate function."""
    def block(docs: Iterator[Document]) -> Iterator[Document]:
        for doc in docs:
            if predicate(doc):
                yield doc
    return block


def create_mapper_block(mapper: Callable[[Document], Document]) -> PipelineBlock:
    """Create a mapper block from a transform function."""
    def block(docs: Iterator[Document]) -> Iterator[Document]:
        for doc in docs:
            yield mapper(doc)
    return block
