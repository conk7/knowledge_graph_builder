from dataclasses import dataclass, field

from src.kg_builder.config import (
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME,
    VECTOR_SEARCH_WEIGHT,
)

#
# GraphRAG pipeline defaults
#
DEFAULT_MAX_HOPS = 2
DEFAULT_BEAM_WIDTH = 4
DEFAULT_SCORE_THRESHOLD = 0.2
DEFAULT_TOP_K_SEED = 5
DEFAULT_TOP_K_CONTEXT = 7
DEFAULT_NER_BOOST_FACTOR = 1.5


@dataclass
class GraphRAGConfig:
    max_hops: int = DEFAULT_MAX_HOPS
    beam_width: int = DEFAULT_BEAM_WIDTH
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    top_k_seed: int = DEFAULT_TOP_K_SEED
    top_k_context: int = DEFAULT_TOP_K_CONTEXT
    ner_boost_factor: float = DEFAULT_NER_BOOST_FACTOR


@dataclass
class VaultConfig:
    lang: str = "en"
    embedding_model: str = EMBEDDING_MODEL_NAME
    reranker_model: str = RERANKER_MODEL_NAME
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    separators: list[str] = field(default_factory=lambda: list(CHUNK_SEPARATORS))
    vector_weight: float = VECTOR_SEARCH_WEIGHT

    @classmethod
    def from_dict(cls, cfg: dict) -> "VaultConfig":
        models = cfg.get("models", {})
        chunking = cfg.get("chunking", {})
        retrieval = cfg.get("retrieval", {})
        return cls(
            lang=cfg.get("lang", "en"),
            embedding_model=models.get("embedding", {}).get(
                "model_name", EMBEDDING_MODEL_NAME
            ),
            reranker_model=models.get("reranker", {}).get(
                "model_name", RERANKER_MODEL_NAME
            ),
            chunk_size=chunking.get("chunk_size", CHUNK_SIZE),
            chunk_overlap=chunking.get("chunk_overlap", CHUNK_OVERLAP),
            separators=chunking.get("separators", list(CHUNK_SEPARATORS)),
            vector_weight=retrieval.get("vector_search_weight", VECTOR_SEARCH_WEIGHT),
        )
