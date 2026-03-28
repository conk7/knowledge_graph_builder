import logging
import os
from pathlib import Path

#
# DIRS
#
META_DIR_NAME = ".kg_builder"
METADATA_FILE_NAME = "metadata.json"
LINKS_CONFIG_FILE_NAME = "config.json"
OUTPUT_DIR = ".out"
OUTPUT_LINKS_FILE_NAME = "links.json"
CANDIDATES_FILE_NAME = "candidates.json"
RERANKED_CANDIDATES_FILE_NAME = "reranked_candidates.json"
LOG_FILE_NAME = "obsidian_ai_linker.log"
RUN_STATE_FILE_NAME = "run_state.json"

#
# EMBEDDING MODEL
#
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSION = 384


#
# RERANKER MODEL
#
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANKER_TOP_K = 5
RERANKER_THRESHOLD = 0.7


#
# LLM
#
LLM_MODEL_PATH = Path(
    os.environ.get("LLM_MODEL_PATH", "")
)
LLM_TEMPERATURE = 0.0
LLM_TOP_P = 0.1
LLM_N_GPU_LAYERS = -1
LLM_N_CTX = 8192
LLM_N_BATCH = 512
LLM_CONCURRENCY = 1
LLM_BACKEND = "vulkan"
MAX_RETRIES = 10


#
# KNOWLEDGE GRAPH
#
CHUNK_SIZE = 300
CHUNK_OVERLAP = 100
CHUNK_SEPARATORS = ["\n# ", "\n## ", "\n### ", "\n\n", ". ", "? ", "! ", "\n", " ", ""]

SPLITTER_TYPE = "recursive"  # "recursive" | "sentence_window"
SENTENCE_WINDOW_BEFORE = 1
SENTENCE_WINDOW_AFTER = 1

INITIAL_RETRIEVAL_K = 15
VECTOR_SEARCH_WEIGHT = 0.5
BROAD_QUERY_MODE_DEFAULT = "title_summary"


#
# LINKS
#
DEFAULT_LINK_TEMPLATE = "- {relation_type}:: [[{target_file_name}]]"
LINK_HEADER = "\n\n## Related Connections\n\n"
DEFAULT_LINK_TYPES = [
    "Is a",
    "Part of",
    "Uses",
    "Solves",
    "Originates from",
    "Precedes",
    "Extends",
    "Influences",
    "Supports",
    "Contradicts",
    "Compared with",
    "Mentions",
]
DEFAULT_VAULT_LANG = "en"


#
# LOGGING DEFAULTS
#
DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(vault_path: Path, log_level: str = DEFAULT_LOG_LEVEL):
    dir = vault_path / META_DIR_NAME / OUTPUT_DIR
    dir.mkdir(parents=True, exist_ok=True)
    log_file = dir / LOG_FILE_NAME

    _resolved_level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=_resolved_level,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        force=True,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8", mode="w"),
        ],
    )

    logging.getLogger(__name__).info(f"Logging initialized. File: {log_file}")
