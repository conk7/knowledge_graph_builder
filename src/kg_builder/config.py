import logging
from pathlib import Path

#
# OBSIDIAN
#
VAULT_PATH = Path("/home/conk/Files/Diploma/testfield/2. Math")

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
RERANKER_THRESHOLD = 0.25


#
# LLM
#
LLM_MODEL_PATH = Path(
    "/home/conk/.lmstudio/models/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_S.gguf"
)
LLM_TEMPERATURE = 0.0
LLM_N_GPU_LAYERS = -1
LLM_N_CTX = 8192
LLM_N_BATCH = 512
LLM_CONCURRENCY = 1
LLM_BACKEND = "vulkan"


#
# KNOWLEDGE GRAPH
#
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300
CHUNK_SEPARATORS = ["\n## ", "\n### ", "\n\n", ". ", "? ", "! ", "\n", " ", ""]

INITIAL_RETRIEVAL_K = 15
# SIMILARITY_DISTANCE_THRESHOLD = 5.0

TAG_CONFIDENCE_THRESHOLD = 0.75
FOLDER_CREATION_THRESHOLD = 3


#
# LINKS
#
DEFAULT_LINK_TEMPLATE = "{relation_type}:: [[{target_file_name}]]"

# DEFAULT_LINK_TYPES = [
#     "uses_concept",
#     "developed_by",
#     "precedes",
#     "follows_from",
#     "part_of",
#     "contradicts",
#     "defined_by",
#     "analogous_to",
#     "solves_problem",
#     "mathematical_relation",
#     "influenced_by",
#     "named_after",
#     "exemplifies",
#     "inverse_of",
#     "related_to",
# ]

# DEFAULT_LINK_EN2RU_TRANSLATION = {
#     "uses_concept": "Использует понятие",
#     "developed_by": "Разработано",
#     "precedes": "Предшествует",
#     "follows_from": "Следует из",
#     "part_of": "Является частью",
#     "contradicts": "Противоречит",
#     "defined_by": "Определяется через",
#     "analogous_to": "Аналогично",
#     "solves_problem": "Решает проблему",
#     "mathematical_relation": "Математически связано с",
#     "influenced_by": "Попало под влияние",
#     "named_after": "Названо в честь",
#     "exemplifies": "Является примером",
#     "inverse_of": "Является обратным для",
#     "related_to": "Относится к",
# }

DEFAULT_LINK_TYPES = ["uses_concept", "uses_method", "discovered", "contradicts"]

DEFAULT_LINK_EN2RU_TRANSLATION = {
    "uses_concept": "Использует понятие",
    "uses_method": "Использует метод из",
    "discovered": "Открыло",
    "contradicts": "Противоречит",
}


LINK_HEADER = "\n\n### Связи\n\n"


#
# LOGGING
#
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = VAULT_PATH / "obsidian_ai_linker.log"
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
