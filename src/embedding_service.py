import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging


class EmbeddingService:
    def __init__(
        self,
        model_name: str,
        chunk_size: int,
        chunk_overlap: int,
        separators: List[str],
    ):
        logging.info(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        logging.info("Embedding model loaded successfully.")

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
