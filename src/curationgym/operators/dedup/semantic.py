"""Semantic deduplication using embeddings and ANN indexing (SemDeDup-style)."""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from curationgym.core.document import Document


@dataclass
class SemanticDedupConfig:
    """Configuration for semantic deduplication."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.9
    batch_size: int = 32
    use_faiss: bool = True
    nlist: int = 100  # FAISS IVF clusters


class SemanticDedup:
    """Embedding-based semantic deduplication using FAISS ANN."""

    def __init__(self, config: SemanticDedupConfig | None = None):
        self.config = config or SemanticDedupConfig()
        self._model = None
        self._index = None
        self._doc_ids: list[str] = []
        self._embeddings: list[np.ndarray] = []

    def _load_model(self):
        """Lazy load embedding model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers required: pip install sentence-transformers")

        self._model = SentenceTransformer(self.config.model_name)

    def _init_index(self, dim: int):
        """Initialize FAISS index."""
        if not self.config.use_faiss:
            return

        try:
            import faiss
        except ImportError:
            self.config.use_faiss = False
            return

        # Use IVF index for scalability
        quantizer = faiss.IndexFlatIP(dim)
        self._index = faiss.IndexIVFFlat(quantizer, dim, self.config.nlist)

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Compute embeddings for texts."""
        self._load_model()
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings, dtype=np.float32)

    def _find_similar(self, embedding: np.ndarray) -> tuple[bool, str | None]:
        """Find if similar document exists.

        Returns:
            (is_duplicate, similar_doc_id)
        """
        if not self._embeddings:
            return False, None

        if self.config.use_faiss and self._index is not None and self._index.is_trained:
            # Use FAISS for search
            D, I = self._index.search(embedding.reshape(1, -1), 1)
            if D[0][0] >= self.config.similarity_threshold:
                return True, self._doc_ids[I[0][0]]
        else:
            # Brute force search
            embeddings_matrix = np.vstack(self._embeddings)
            similarities = np.dot(embeddings_matrix, embedding)
            max_idx = np.argmax(similarities)
            if similarities[max_idx] >= self.config.similarity_threshold:
                return True, self._doc_ids[max_idx]

        return False, None

    def add_document(self, doc: Document) -> tuple[bool, str | None]:
        """Add document and check for semantic duplicates.

        Returns:
            (is_unique, similar_doc_id)
        """
        embedding = self._embed([doc.text])[0]

        # Initialize index on first embedding
        if self._index is None and self.config.use_faiss:
            self._init_index(len(embedding))

        is_dup, similar_id = self._find_similar(embedding)

        if not is_dup:
            self._doc_ids.append(doc.id)
            self._embeddings.append(embedding)

            # Train and add to FAISS index
            if self.config.use_faiss and self._index is not None:
                if not self._index.is_trained and len(self._embeddings) >= self.config.nlist:
                    train_data = np.vstack(self._embeddings)
                    self._index.train(train_data)
                    self._index.add(train_data)
                elif self._index.is_trained:
                    self._index.add(embedding.reshape(1, -1))

        return not is_dup, similar_id

    def process(self, docs: Iterator[Document]) -> Iterator[Document]:
        """Process documents, yielding only semantically unique ones."""
        for doc in docs:
            is_unique, similar_id = self.add_document(doc)
            doc.metadata["dedup_method"] = "semantic"

            if is_unique:
                doc.metadata["dedup_cluster_id"] = doc.id[:16]
                yield doc
            else:
                doc.metadata["dedup_dropped"] = True
                doc.metadata["dedup_similar_to"] = similar_id
                doc.metadata["dedup_cluster_id"] = similar_id[:16] if similar_id else None

    def reset(self) -> None:
        """Clear index and embeddings."""
        self._index = None
        self._doc_ids.clear()
        self._embeddings.clear()

    @property
    def stats(self) -> dict[str, int]:
        """Return dedup statistics."""
        return {"unique_docs": len(self._doc_ids)}
