import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging


class EmbeddingService:
    def __init__(
        self,
        embedding_model_name: str,
        reranker_model_name: str,
        chunk_size: int,
        chunk_overlap: int,
        separators: List[str],
    ):
        logging.info(f"Loading embedding model: {embedding_model_name}...")
        self.model = SentenceTransformer(embedding_model_name)
        logging.info("Embedding model loaded successfully.")

        logging.info(f"Loading reranker model: {reranker_model_name}...")
        self.reranker = CrossEncoder(reranker_model_name)
        logging.info("Reranker model loaded successfully.")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators,
        )

    def chunk_text(self, text: str) -> List[str]:
        if not text:
            return []
        return self.text_splitter.split_text(text)

    def get_embeddings(self, chunks: List[str]) -> np.ndarray:
        if not chunks:
            return np.array([])

        embeddings = self.model.encode(
            chunks,
            show_progress_bar=False,
        )
        return embeddings

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
