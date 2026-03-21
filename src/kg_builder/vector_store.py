import logging
from pathlib import Path
from typing import Dict, List, Tuple

import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import LinearCombinationReranker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        index_path: Path,
        embedding_model_name: str,
        reranker_model_name: str,
        chunk_size: int,
        chunk_overlap: int,
        separators: List[str],
        vector_weight: float = 0.5,
        fresh_start: bool = False,
    ):
        self.index_path = index_path
        self.table_name = "chunks"
        self.vector_weight = vector_weight

        self.index_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.index_path))

        logger.info(
            f"Loading LanceDB embedding function with model: {embedding_model_name}..."
        )
        self.embed_func = (
            get_registry()
            .get("sentence-transformers")
            .create(name=embedding_model_name, normalize=False)
        )

        class ChunkModel(LanceModel):
            text: str = self.embed_func.SourceField()
            vector: Vector(self.embed_func.ndims()) = self.embed_func.VectorField()
            file_path: str
            file_hash: str

        self.chunk_model = ChunkModel

        if fresh_start:
            logger.info(
                "Fresh start requested. Dropping existing vector index database..."
            )
            if self.table_name in self.db.table_names():
                self.db.drop_table(self.table_name)

        if self.table_name not in self.db.table_names():
            logger.info(
                f"Creating new LanceDB table '{self.table_name}' at: {self.index_path}"
            )
            self.table = self.db.create_table(self.table_name, schema=self.chunk_model)
        else:
            logger.info(
                f"Opening existing LanceDB table '{self.table_name}' at: {self.index_path}"
            )
            self.table = self.db.open_table(self.table_name)

        # Ensure FTS index exists if FTS search is enabled (vector_weight < 1.0)
        if self.vector_weight < 1.0:
            self._ensure_fts_index()

        logger.info(f"Loading reranker model: {reranker_model_name}...")
        self.reranker = CrossEncoder(reranker_model_name)
        logger.info("Reranker model loaded successfully.")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators,
        )

    def _ensure_fts_index(self):
        try:
            logger.info("Ensuring FTS index on 'text' field...")
            self.table.create_fts_index("text", replace=False)
        except Exception as e:
            logger.warning(f"Could not create FTS index: {e}")

    def save(self):
        logger.debug("LanceDB save() called; no explicit flush required.")

    def get_and_update_file_chunks(
        self, file_path: str, content: str, file_hash: str
    ) -> List[str]:
        existing = (
            self.table.search()
            .where(f"file_path = '{file_path}'")
            .select(["text", "file_hash"])
            .limit(1)
            .to_list()
        )

        if existing and existing[0]["file_hash"] == file_hash:
            logger.debug(f"Cache hit for {file_path}. Retrieving existing chunks.")
            all_chunks = (
                self.table.search()
                .where(f"file_path = '{file_path}'")
                .select(["text"])
                .to_list()
            )
            return [c["text"] for c in all_chunks]

        if existing:
            logger.debug(f"Hash mismatch for {file_path}. Updating index.")
            self.remove_document(file_path)

        return self.add_document(file_path, content, file_hash)

    def add_document(
        self, file_path: str, content: str, file_hash: str = ""
    ) -> List[str]:
        if not content:
            return []

        chunks = self.text_splitter.split_text(content)
        if not chunks:
            return []

        rows = [
            {"text": chunk, "file_path": str(file_path), "file_hash": file_hash}
            for chunk in chunks
        ]
        self.table.add(rows)
        logger.debug(f"Added {len(rows)} chunk vectors for document {file_path}")

        if self.vector_weight < 1.0:
            self._ensure_fts_index()

        return chunks

    def remove_document(self, file_path: str) -> int:
        before = self.total_vectors
        self.table.delete(f"file_path = '{file_path}'")
        after = self.total_vectors

        removed = max(0, before - after)
        if removed > 0:
            logger.debug(
                f"Removed {removed} vectors from LanceDB for document {file_path}"
            )
        return removed

    def search(self, query_text: str, k: int) -> List[Dict]:
        if self.total_vectors == 0:
            return []

        num_to_search = min(k + 1, self.total_vectors)

        if 0.0 < self.vector_weight < 1.0:
            logger.debug(
                f"Performing hybrid search (Vector weight: {self.vector_weight})"
            )
            # Use reranker for hybrid search
            reranker = LinearCombinationReranker(weight=self.vector_weight)
            results = (
                self.table.search(query_text, query_type="hybrid")
                .limit(num_to_search)
                .rerank(reranker=reranker)
                .select(["text", "file_path"])
                .to_list()
            )
        elif self.vector_weight <= 0.0:
            logger.debug("Performing FTS search")
            results = (
                self.table.search(query_text, query_type="fts")
                .select(["text", "file_path"])
                .limit(num_to_search)
                .to_list()
            )
        else:
            logger.debug("Performing vector search")
            results = (
                self.table.search(query_text, query_type="vector")
                .select(["text", "file_path"])
                .limit(num_to_search)
                .to_list()
            )

        return results

    def rerank(
        self, query: str, candidates: List[str], top_k: int = 5, threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        if not candidates:
            return []

        pairs = [[query, doc] for doc in candidates]
        scores = self.reranker.predict(pairs, show_progress_bar=False)

        results = list(zip(candidates, scores))
        results.sort(key=lambda x: x[1], reverse=True)
        filtered_results = [
            (text, score) for text, score in results if score >= threshold
        ]

        return filtered_results[:top_k]

    @property
    def total_vectors(self) -> int:
        return int(self.table.count_rows()) if self.table is not None else 0
