import faiss
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple


class VectorStore:
    def __init__(self, index_path: Path, dimension: int):
        self.index_path = index_path
        self.dimension = dimension

        try:
            if self.index_path.exists():
                logging.info(f"Loading existing FAISS index from: {self.index_path}")
                self.index = faiss.read_index(str(self.index_path))

                if self.index.d != self.dimension:
                    logging.error(
                        f"Index dimension mismatch! Expected {self.dimension}, got {self.index.d}. Re-creating index."
                    )
                    self._create_new_index()
            else:
                logging.info("No FAISS index found. Creating a new one.")
                self._create_new_index()
        except Exception as e:
            logging.error(
                f"Failed to load or create FAISS index: {e}. Re-creating index."
            )
            self._create_new_index()

    def _create_new_index(self):
        base_index = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIDMap(base_index)

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        logging.info(f"Saving FAISS index to: {self.index_path}")
        faiss.write_index(self.index, str(self.index_path))

    def add(self, vectors: np.ndarray, ids: List[int]):
        if vectors.size == 0:
            return

        if len(vectors) != len(ids):
            raise ValueError("The number of vectors and IDs must be the same.")

        id_array = np.array(ids, dtype=np.int64)
        self.index.add_with_ids(vectors, id_array)
        logging.debug(f"Added {len(vectors)} vectors to the index.")

    def remove(self, ids_to_remove: List[int]) -> int:
        if not ids_to_remove:
            return 0

        selector = faiss.IDSelectorBatch(np.array(ids_to_remove, dtype=np.int64))
        num_removed = self.index.remove_ids(selector)

        if num_removed > 0:
            logging.debug(f"Removed {num_removed} vectors from the index.")
        return num_removed

    def search(self, query_vector: np.ndarray, k: int) -> Tuple[List[float], List[int]]:
        if self.index.ntotal == 0:
            return [], []

        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        num_to_search = min(k + 1, self.index.ntotal)

        distances, ids = self.index.search(query_vector, num_to_search)

        return distances[0].tolist(), ids[0].tolist()

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal if self.index else 0
