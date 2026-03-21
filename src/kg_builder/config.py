import logging
from pathlib import Path

#
# DIRS
#
META_DIR_NAME = ".ai_meta"
METADATA_FILE_NAME = "metadata.json"
LINKS_FILE_NAME = "links.json"
LOG_FILE_NAME = "obsidian_ai_linker.log"

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
    "/home/conk/.lmstudio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
)
LLM_TEMPERATURE = 0.0
LLM_TOP_P = 0.1
LLM_N_GPU_LAYERS = -1
LLM_N_CTX = 8192
LLM_N_BATCH = 512
LLM_CONCURRENCY = 10
LLM_BACKEND = "vulkan"


#
# KNOWLEDGE GRAPH
#
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300
CHUNK_SEPARATORS = ["\n# ", "\n## ", "\n### ", "\n\n", ". ", "? ", "! ", "\n", " ", ""]

INITIAL_RETRIEVAL_K = 25
VECTOR_SEARCH_WEIGHT = 0.0


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


#
# LOGGING DEFAULTS
#
DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(vault_path: Path, log_level: str = DEFAULT_LOG_LEVEL):
    meta_dir = vault_path / ".ai_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    log_file = meta_dir / LOG_FILE_NAME

    _resolved_level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=_resolved_level,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        force=True,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

    logging.getLogger(__name__).info(f"Logging initialized. File: {log_file}")
