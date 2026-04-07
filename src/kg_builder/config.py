import logging
import os
from enum import Enum
from pathlib import Path


class BroadQueryMode(str, Enum):
    CHUNK = "chunk"
    TITLE_SUMMARY = "title_summary"


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
# RETRIEVAL
#
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

CHUNK_SIZE = 3000
CHUNK_OVERLAP = 300
CHUNK_SEPARATORS = ["\n# ", "\n## ", "\n### ", "\n\n", ". ", "? ", "! ", "\n", " ", ""]

SPLITTER_TYPE = "recursive"  # "recursive" | "sentence_window"
SENTENCE_WINDOW_BEFORE = 1
SENTENCE_WINDOW_AFTER = 1

INITIAL_RETRIEVAL_K = 15
VECTOR_SEARCH_WEIGHT = 1
BROAD_QUERY_MODE_DEFAULT = BroadQueryMode.CHUNK


#
# RERANKER MODEL
#
RERANKER_MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"
RERANKER_TOP_K = 5
RERANKER_THRESHOLD = 0.5
RERANKER_BATCH_SIZE = 16


#
# LLM
#
LLM_MODEL_PATH = Path(os.environ.get("LLM_MODEL_PATH", ""))
LLM_TEMPERATURE = 0.0
LLM_TOP_P = 0.1
LLM_N_GPU_LAYERS = -1
LLM_N_CTX = 8192
LLM_N_BATCH = 512
LLM_CONCURRENCY = 1
LLM_BACKEND = "vulkan"
MAX_RETRIES = 10

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
    "Influences",
    "Contradicts",
    "Compared with",
    "Mentions",
]
DEFAULT_VAULT_LANG = "en"


#
# LOGGING
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
