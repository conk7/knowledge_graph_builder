# config.py
from pathlib import Path
import logging

#
# OBSIDIAN
#
VAULT_PATH = Path("/home/conk/Files/Diploma/testfield/1. CS/")

INBOX_DIR = VAULT_PATH / "00_Inbox"
NOTES_DIR = VAULT_PATH / "10_Notes"
TAGS_DIR = VAULT_PATH / "20_Tags"
META_DIR = VAULT_PATH / ".ai_meta"

INDEX_PATH = META_DIR / "index.faiss"
METADATA_PATH = META_DIR / "metadata.json"

IGNORED_DIRS = [
    VAULT_PATH / "90_Fleeting",
    META_DIR,
]

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
RERANKER_THRESHOLD = 0.35


#
# LLM
#
LLM_MODEL_PATH = Path(
    "/home/conk/.lmstudio/models/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_S.gguf"
)
LLM_TEMPERATURE = 0.0
LLM_N_GPU_LAYERS = -1
LLM_N_CTX = 8192
LLM_N_BATCH = 4
LLM_BACKEND = "vulkan"


#
# KNOWLEDGE GRAPH
#
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 150
CHUNK_SEPARATORS = ["\n## ", "\n### ", "\n\n", ". ", "? ", "! ", "\n", " ", ""]

INITIAL_RETRIEVAL_K = 5
SIMILARITY_DISTANCE_THRESHOLD = 5.0

TAG_CONFIDENCE_THRESHOLD = 0.75
FOLDER_CREATION_THRESHOLD = 3


#
# MISC
#
DEFAULT_LINK_TEMPLATE = "{relation_type}:: [[{target_file_name}]]"

DEFAULT_LINK_TYPES = [
    "uses_concept",
    "uses_method",
    "discovered",
    "contradicts",
    "exemplifies",
]

DEFAULT_LINK_EN2RU_TRANSLATION = {
    "uses_concept": "Использует понятие",
    "uses_method": "Использует метод из",
    "discovered": "Открыло",
    "contradicts": "Противоречит",
    "exemplifies": "Является примером",
}


# DEFAULT_LINK_TYPES = [
#     "is_example_of",
#     "explains_concept",
#     "contradicts",
#     "uses_method",
#     "is_part_of",
#     "references_source",
#     "similar",
#     "mentions",
# ]

# DEFAULT_LINK_EN2RU_TRANSLATION = {
#     "exemplifies": "Является примером для",
#     "is_example_of": "Является примером для",
#     "explains_concept": "Объясняет концепцию из",
#     "contradicts": "Противоречит",
#     "uses_method": "Использует метод из",
#     "is_part_of": "Является частью",
#     "references_source": "Ссылается на источник",
#     "similar": "Похож на",
#     "mentions": "Упоминает",
# }

LINK_HEADER = "\n\n### Связи\n\n"

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = VAULT_PATH / "obsidian_ai_linker.log"
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
