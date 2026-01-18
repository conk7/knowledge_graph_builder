# config.py
from pathlib import Path

#
# OBSIDIAN
#
VAULT_PATH = Path("/home/conk/Files/Diploma/notes")

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
# LLM
#
LLM_MODEL_PATH = Path(
    "/home/conk/.lmstudio/models/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_S.gguf"
)
LLM_TEMPERATURE = 0.0
LLM_N_GPU_LAYERS = -1
LLM_N_CTX = 8192
LLM_BACKEND = "vulkan"


#
# KNOWLEDGE GRAPH
#
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 150
CHUNK_SEPARATORS = ["\n## ", "\n### ", "\n\n", ". ", "? ", "! ", "\n", " ", ""]

SIMILARITY_SEARCH_K = 5
SIMILARITY_DISTANCE_THRESHOLD = 10.0

TAG_CONFIDENCE_THRESHOLD = 0.75
FOLDER_CREATION_THRESHOLD = 3


#
# MISC
#
RELATIONSHIP_TEMPLATE = "* {relation_type}: [[{target_file_name}]]"

LLM_RELATION_TYPES = [
    "is_example_of",
    "explains_concept",
    "contradicts",
    "uses_method",
    "is_part_of",
    "references_source",
    "similar",
    "mentions",
]

RELATION_DISPLAY_MAP = {
    "exemplifies": "Является примером для",
    "is_example_of": "Является примером для",
    "explains_concept": "Объясняет концепцию из",
    "contradicts": "Противоречит",
    "uses_method": "Использует метод из",
    "is_part_of": "Является частью",
    "references_source": "Ссылается на источник",
    "similar": "Похож на",
    "mentions": "Упоминает",
}

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = VAULT_PATH / "obsidian_ai_linker.log"
