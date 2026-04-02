import logging
import os

_BLOCKED_LOG_MESSAGES = ("AFC is enabled",)

try:
    from tqdm import tqdm as _tqdm

    class _TqdmHandler(logging.StreamHandler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                if any(p in record.getMessage() for p in _BLOCKED_LOG_MESSAGES):
                    return
                _tqdm.write(self.format(record))
            except Exception:
                self.handleError(record)

    _handler: logging.Handler = _TqdmHandler()
except ImportError:
    _handler = logging.StreamHandler()

    class _BlockFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return not any(p in record.getMessage() for p in _BLOCKED_LOG_MESSAGES)

    _handler.addFilter(_BlockFilter())

_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
_root = logging.getLogger()
_root.setLevel(logging.INFO)
_root.handlers.clear()
_root.addHandler(_handler)

for _name in (
    "httpx",
    "httpcore",
    "openai",
    "urllib3",
    "botocore",
    "boto3",
    "google",
    "google.genai",
    "tenacity",
    "langchain",
    "llama_index",
    "sentence_transformers",
):
    logging.getLogger(_name).setLevel(logging.WARNING)

try:
    import absl.logging as _absl_log

    _absl_log.set_verbosity(_absl_log.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)
except ImportError:
    pass

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
try:
    import transformers

    transformers.logging.set_verbosity_error()
except ImportError:
    pass
