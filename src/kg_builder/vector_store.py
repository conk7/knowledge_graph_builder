import gc
import logging
from pathlib import Path
from typing import List, Optional

import lancedb
import torch
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import LinearCombinationReranker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

from .models import RerankResult, SearchResult

logger = logging.getLogger(__name__)


class SentenceWindowSplitter:
    """Splits text into overlapping sentence-window chunks using spaCy."""

    def __init__(self, nlp, window_before: int = 1, window_after: int = 1):
        self.nlp = nlp
        self.window_before = window_before
        self.window_after = window_after

    def split_text(self, text: str) -> List[str]:
        doc = self.nlp(text)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        chunks = []
        for i in range(len(sents)):
            start = max(0, i - self.window_before)
            end = min(len(sents), i + self.window_after + 1)
            chunk = " ".join(sents[start:end])
            if chunk:
                chunks.append(chunk)
        return chunks


class VectorStore:
    @staticmethod
    def _escape_filter_value(value: str) -> str:
        return value.replace("'", "''")

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
        lang: str = "ru",
        splitter_type: str = "recursive",
        sentence_window_before: int = 1,
        sentence_window_after: int = 1,
    ):
        self.lang = lang
        self.index_path = index_path
        self.table_name = "chunks"
        self.vector_weight = vector_weight
        self.reranker_model_name = reranker_model_name
        self.reranker = None
        self._nlp = None
        self._splitter_type = splitter_type
        self._sentence_window_before = sentence_window_before
        self._sentence_window_after = sentence_window_after
        self._sentence_splitter: Optional[SentenceWindowSplitter] = None

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
            text_lemmatized: str
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

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators,
        )

    def _ensure_fts_index(self):
        try:
            logger.info("Ensuring FTS index on 'text_lemmatized' field...")
            self.table.create_fts_index("text_lemmatized", replace=True)
        except Exception as e:
            logger.warning(f"Could not create FTS index: {e}")

    def rebuild_fts_index(self):
        self._ensure_fts_index()

    def save(self):
        try:
            from datetime import timedelta

            self.table.cleanup_old_versions(older_than=timedelta(days=0))
            self.table.compact_files()
            logger.debug("LanceDB compacted and old versions cleaned up.")
        except Exception as e:
            logger.warning(f"Failed to cleanup/compact LanceDB: {e}")

    def load_reranker(self):
        if self.reranker is None:
            logger.info(f"Loading reranker model: {self.reranker_model_name}...")
            self.reranker = CrossEncoder(self.reranker_model_name)
            if hasattr(self.reranker, "show_progress_bar"):
                self.reranker.show_progress_bar = False
            logger.info("Reranker model loaded successfully.")

    def unload_reranker(self):
        if self.reranker is not None:
            logger.info(f"Unloading reranker model: {self.reranker_model_name}...")
            del self.reranker
            self.reranker = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Reranker model unloaded.")

    def _get_nlp(self):
        if self._nlp is None:
            model_name = "ru_core_news_sm" if self.lang == "ru" else "en_core_web_sm"
            try:
                import spacy

                logger.info(
                    f"Lazy loading spacy model '{model_name}' at instance level..."
                )
                self._nlp = spacy.load(model_name)
            except Exception as e:
                logger.error(f"Failed to load spacy model {model_name}: {e}")
                raise
        return self._nlp

    def _get_splitter(self):
        if self._splitter_type == "sentence_window":
            if self._sentence_splitter is None:
                self._sentence_splitter = SentenceWindowSplitter(
                    nlp=self._get_nlp(),
                    window_before=self._sentence_window_before,
                    window_after=self._sentence_window_after,
                )
            return self._sentence_splitter
        return self.text_splitter

    def _get_lemmatized_text(self, text: str) -> str:
        nlp = self._get_nlp()
        doc = nlp(text)
        return " ".join([w.lemma_.lower() for w in doc if w.is_alpha or w.like_num])

    def get_and_update_file_chunks(
        self, file_path: str, content: str, file_hash: str
    ) -> List[str]:
        existing = (
            self.table.search()
            .where(f"file_path = '{self._escape_filter_value(file_path)}'")
            .select(["text", "file_hash"])
            .limit(1)
            .to_list()
        )

        if existing and existing[0]["file_hash"] == file_hash:
            logger.debug(f"Cache hit for {file_path}. Retrieving existing chunks.")
            all_chunks = (
                self.table.search()
                .where(f"file_path = '{self._escape_filter_value(file_path)}'")
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

        chunks = self._get_splitter().split_text(content)
        if not chunks:
            return []

        rows = []
        for chunk in chunks:
            lemmatized = self._get_lemmatized_text(chunk)
            rows.append(
                {
                    "text": chunk,
                    "text_lemmatized": lemmatized,
                    "file_path": str(file_path),
                    "file_hash": file_hash,
                }
            )

        self.table.add(rows)
        logger.debug(f"Added {len(rows)} chunk vectors for document {file_path}")

        return chunks

    def remove_document(self, file_path: str) -> int:
        before = self.total_vectors
        self.table.delete(f"file_path = '{self._escape_filter_value(file_path)}'")
        after = self.total_vectors

        removed = max(0, before - after)
        if removed > 0:
            logger.debug(
                f"Removed {removed} vectors from LanceDB for document {file_path}"
            )
        return removed

    def search(self, query_text: str, k: int) -> List[SearchResult]:
        if self.total_vectors == 0:
            return []

        num_to_search = min(k + 1, self.total_vectors)

        if 0.0 < self.vector_weight < 1.0:
            logger.debug(
                f"Performing hybrid search (Vector weight: {self.vector_weight})"
            )
            lemmatized_query = self._get_lemmatized_text(query_text)
            reranker = LinearCombinationReranker(weight=self.vector_weight)
            results = (
                self.table.search(lemmatized_query, query_type="hybrid")
                .limit(num_to_search)
                .rerank(reranker=reranker)
                .to_list()
            )
        elif self.vector_weight <= 0.0:
            logger.debug("Performing FTS search")
            lemmatized_query = self._get_lemmatized_text(query_text)
            results = (
                self.table.search(lemmatized_query, query_type="fts")
                .limit(num_to_search)
                .to_list()
            )
        else:
            logger.debug("Performing vector search")
            results = (
                self.table.search(query_text, query_type="vector")
                .limit(num_to_search)
                .to_list()
            )

        return [
            SearchResult(
                text=r["text"],
                file_path=r["file_path"],
                distance=float(r.get("_distance", r.get("_score", 0.0))),
            )
            for r in results
        ]

    def rerank(
        self, query: str, candidates: List[str], top_k: int = 5, threshold: float = 0.0
    ) -> List[RerankResult]:
        if not candidates:
            return []

        if self.reranker is None:
            logger.warning("Reranker not loaded. Initializing it lazily.")
            self.load_reranker()

        pairs = [[query, doc] for doc in candidates]
        scores = self.reranker.predict(pairs, show_progress_bar=False)

        results = list(zip(candidates, scores))
        results.sort(key=lambda x: x[1], reverse=True)
        filtered_results = [
            RerankResult(text=text, score=float(score))
            for text, score in results
            if score >= threshold
        ]

        return filtered_results[:top_k]

    @property
    def total_vectors(self) -> int:
        return int(self.table.count_rows()) if self.table is not None else 0

    def get_document_summary_with_lemmas(self, file_path: str) -> tuple[str, str]:
        if self.table is None:
            return "", ""
        results = (
            self.table.search()
            .where(f"file_path = '{self._escape_filter_value(file_path)}'")
            .select(["text", "text_lemmatized"])
            .limit(1)
            .to_list()
        )
        if results:
            return results[0]["text"], results[0]["text_lemmatized"]
        return "", ""

    def get_document_summary(self, file_path: str) -> str:
        if self.table is None:
            return ""
        results = (
            self.table.search()
            .where(f"file_path = '{self._escape_filter_value(file_path)}'")
            .select(["text"])
            .limit(1)
            .to_list()
        )
        if results:
            return results[0]["text"]
        return ""
