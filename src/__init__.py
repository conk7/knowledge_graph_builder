import logging
import os

for logger_name in ("httpx", "httpcore", "openai", "urllib3", "botocore", "boto3"):
    logging.getLogger(logger_name).setLevel(logging.WARNING)

for logger_name in ("langchain", "llama_index"):
    logging.getLogger(logger_name).setLevel(logging.WARNING)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
try:
    import transformers

    transformers.logging.set_verbosity_error()
except ImportError:
    pass

logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
