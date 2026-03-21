from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from .config import (
    DEFAULT_LINK_TEMPLATE,
    DEFAULT_LINK_TYPES,
)

logger = logging.getLogger(__name__)


@dataclass
class FileMetadata:
    file_path: str
    hash: str
    tags: List[Any] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FileMetadata:
        return cls(
            file_path=data["file_path"],
            hash=data["hash"],
            tags=data.get("tags", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "hash": self.hash,
            "tags": self.tags,
        }


@dataclass
class VaultMetadata:
    files: Dict[str, FileMetadata] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VaultMetadata:
        return cls(
            files={
                path: FileMetadata.from_dict(meta)
                for path, meta in data.get("files", {}).items()
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"files": {path: meta.to_dict() for path, meta in self.files.items()}}

    def get_file(self, file_path_str: str) -> Optional[FileMetadata]:
        return self.files.get(file_path_str)

    def add_or_update_file(self, file_meta: FileMetadata):
        self.files[file_meta.file_path] = file_meta

    def remove_file(self, file_path_str: str):
        if file_path_str in self.files:
            del self.files[file_path_str]


@dataclass
class LinkConfig:
    link_template: str
    llm_link_types: List[str]

    @classmethod
    def get_defaults(cls) -> LinkConfig:
        return cls(
            link_template=DEFAULT_LINK_TEMPLATE,
            llm_link_types=DEFAULT_LINK_TYPES,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "link_template": self.link_template,
            "llm_link_types": self.llm_link_types,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LinkConfig:
        defaults = cls.get_defaults()
        return cls(
            link_template=data.get("link_template", defaults.link_template),
            llm_link_types=data.get("llm_link_types", defaults.llm_link_types),
        )


def _pydantic_validate(model_cls, data: Any):
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    return model_cls.parse_obj(data)


def _pydantic_dump(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json", exclude_none=True)
    return json.loads(model.json(exclude_none=True))


class RunStage(str, Enum):
    INIT = "init"
    COLLECTING_CANDIDATES = "collecting_candidates"
    PENDING_LLM_PAIRS = "pending_llm_pairs"
    CLASSIFYING_LLM_PAIRS = "classifying_llm_pairs"
    RESUMING_LLM_CLASSIFICATION = "resuming_llm_classification"
    NO_CANDIDATES = "no_candidates"
    FAILED = "failed"
    COMPLETED = "completed"


class ChunkingSnapshot(BaseModel):
    chunk_size: int
    chunk_overlap: int
    separators: List[str]


class RetrievalSnapshot(BaseModel):
    initial_retrieval_k: int
    vector_search_weight: float


class EmbeddingSnapshot(BaseModel):
    model_name: str
    dimension: int


class RerankerSnapshot(BaseModel):
    model_name: str
    top_k: int
    threshold: float


class LlmSnapshot(BaseModel):
    use_api: bool
    model_path: str
    backend: str
    temperature: float
    top_p: float
    n_gpu_layers: int
    n_ctx: int
    n_batch: int
    concurrency: int


class ModelsSnapshot(BaseModel):
    embedding: EmbeddingSnapshot
    reranker: RerankerSnapshot
    llm: LlmSnapshot


class RuntimeSnapshot(BaseModel):
    chunking: ChunkingSnapshot
    retrieval: RetrievalSnapshot
    models: ModelsSnapshot
    link_header: str


class CandidatePair(BaseModel):
    text_a: str
    text_b: str
    path_a: str
    path_b: str

    vector_distance: Optional[float] = None
    reranker_score: Optional[float] = None


class RunState(BaseModel):
    stage: Optional[RunStage] = None
    stage_details: Dict[str, Any] = Field(default_factory=dict)
    runtime_snapshot: Optional[RuntimeSnapshot] = None
    candidates_file: Optional[str] = None
    partial_links_file: Optional[str] = None
    partial_predictions_file: Optional[str] = None
    pending_pairs_count: int = 0
    llm_offset: int = 0
    llm_total: int = 0


class MetadataManager:
    def __init__(self, metadata_path: Path):
        self.metadata_path = metadata_path
        self.config_path = self.metadata_path.parent / "links_config.json"
        self.run_state_path = self.metadata_path.parent / "run_state.json"
        self.candidates_path = self.metadata_path.parent / "candidates.json"
        self.partial_links_path = self.metadata_path.parent / "links_partial.json"
        self.partial_predictions_path = (
            self.metadata_path.parent / "predictions_partial.json"
        )
        self.vault = VaultMetadata()
        self._is_fresh_start = False
        self.run_state: RunState = RunState()
        self.pending_pairs: List[CandidatePair] = []
        self._pending_pairs_dirty = False

        self._load_metadata()
        self._load_config()
        self._load_run_state()

    def _load_metadata(self):
        logger.info(f"Attempting to load metadata from {self.metadata_path}")
        if not self.metadata_path.exists():
            logger.info("Metadata file not found. Starting with a fresh state.")
            self._is_fresh_start = True
            return
        try:
            with self.metadata_path.open("r", encoding="utf-8") as f:
                raw_data = json.load(f)
            self.vault = VaultMetadata.from_dict(raw_data)
            logger.info(
                f"Metadata loaded successfully. Tracking {len(self.vault.files)} files."
            )
        except (json.JSONDecodeError, IOError, KeyError) as e:
            logger.error(
                f"Error loading or parsing metadata file: {e}. Starting with a fresh state."
            )
            self.vault = VaultMetadata()

    def _load_config(self):
        logger.info(f"Attempting to load config from {self.config_path}")
        if not self.config_path.exists():
            logger.info("Config file not found. Using defaults")
            self.config = LinkConfig.get_defaults()
            return

        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                raw_config = json.load(f)
            self.config = LinkConfig.from_dict(raw_config)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading config file: {e}. Using internal defaults.")
            self.config = LinkConfig.get_defaults()

        logger.info("link configuration loaded successfully.")

    def _load_run_state(self):
        logger.info(f"Attempting to load run state from {self.run_state_path}")
        if not self.run_state_path.exists():
            self.run_state = RunState()
            self.pending_pairs = []
            return

        try:
            with self.run_state_path.open("r", encoding="utf-8") as f:
                raw = json.load(f) or {}

            self.run_state = _pydantic_validate(RunState, raw)

            candidates_file = self.run_state.candidates_file
            if not candidates_file:
                self.pending_pairs = []
                return

            candidates_path = self.run_state_path.parent / str(candidates_file)
            if not candidates_path.exists():
                self.pending_pairs = []
                return

            with candidates_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, list):
                self.pending_pairs = []
                return

            pairs: List[CandidatePair] = []
            for item in loaded:
                try:
                    pairs.append(_pydantic_validate(CandidatePair, item))
                except ValidationError:
                    continue
            self.pending_pairs = pairs
            self.run_state.pending_pairs_count = len(self.pending_pairs)

            logger.info(
                f"Run state loaded successfully. Pending pairs: {len(self.pending_pairs)}."
            )
        except (json.JSONDecodeError, IOError, ValidationError) as e:
            logger.error(f"Error loading run state file: {e}. Ignoring run state.")
            self.run_state = RunState()
            self.pending_pairs = []

    def is_fresh_start(self) -> bool:
        return self._is_fresh_start

    def save(self):
        try:
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create metadata folder: {e}")

        try:
            with self.metadata_path.open("w", encoding="utf-8") as f:
                json.dump(self.vault.to_dict(), f, indent=4, ensure_ascii=False)
            logger.info(f"Metadata saved successfully to {self.metadata_path}")
        except IOError as e:
            logger.error(f"Failed to save metadata: {e}")

        try:
            with self.config_path.open("w", encoding="utf-8") as f:
                json.dump(self.config.to_dict(), f, indent=4, ensure_ascii=False)
            logger.info(f"Configuration saved successfully to {self.config_path}")
        except IOError as e:
            logger.error(f"Failed to save configuration: {e}")

        self._save_run_state()

    def save_run_state_only(self):
        self._save_run_state()

    def _save_run_state(self):
        try:
            self.run_state_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create run state folder: {e}")
            return

        try:
            if self.pending_pairs:
                if self._pending_pairs_dirty or not self.candidates_path.exists():
                    with self.candidates_path.open("w", encoding="utf-8") as f:
                        json.dump(
                            [_pydantic_dump(p) for p in self.pending_pairs],
                            f,
                            indent=2,
                            ensure_ascii=False,
                        )
                    self._pending_pairs_dirty = False
                self.run_state.candidates_file = self.candidates_path.name
                self.run_state.pending_pairs_count = len(self.pending_pairs)
            else:
                self.run_state.candidates_file = None
                self.run_state.pending_pairs_count = 0
                if self.candidates_path.exists():
                    self.candidates_path.unlink()
        except (IOError, OSError) as e:
            logger.error(f"Failed to save candidates file: {e}")

        try:
            with self.run_state_path.open("w", encoding="utf-8") as f:
                json.dump(
                    _pydantic_dump(self.run_state), f, indent=2, ensure_ascii=False
                )
            logger.info(f"Run state saved successfully to {self.run_state_path}")
        except IOError as e:
            logger.error(f"Failed to save run state: {e}")

    def set_llm_progress(self, offset: int, total: int):
        self.run_state.llm_offset = max(0, int(offset))
        self.run_state.llm_total = max(0, int(total))

    def reset_llm_progress(self, total: int):
        self.set_llm_progress(offset=0, total=total)

    def get_llm_progress(self) -> tuple[int, int]:
        return self.run_state.llm_offset, self.run_state.llm_total

    def save_partial_links(self, links: Dict[str, List[str]]):
        try:
            with self.partial_links_path.open("w", encoding="utf-8") as f:
                json.dump(links, f, ensure_ascii=False, indent=2)
            self.run_state.partial_links_file = self.partial_links_path.name
        except (IOError, OSError) as e:
            logger.error(f"Failed to save partial links: {e}")

    def load_partial_links(self) -> Dict[str, List[str]]:
        partial_file = self.run_state.partial_links_file
        if not partial_file:
            return {}
        path = self.run_state_path.parent / partial_file
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load partial links file: {e}")
        return {}

    def clear_partial_links(self):
        self.run_state.partial_links_file = None
        if self.partial_links_path.exists():
            try:
                self.partial_links_path.unlink()
            except OSError as e:
                logger.error(f"Failed to delete partial links file: {e}")

    def save_partial_predictions(self, predictions: Dict[str, List[Dict[str, Any]]]):
        try:
            with self.partial_predictions_path.open("w", encoding="utf-8") as f:
                json.dump(predictions, f, ensure_ascii=False, indent=2)
            self.run_state.partial_predictions_file = self.partial_predictions_path.name
        except (IOError, OSError) as e:
            logger.error(f"Failed to save partial predictions: {e}")

    def load_partial_predictions(self) -> Dict[str, List[Dict[str, Any]]]:
        partial_file = self.run_state.partial_predictions_file
        if not partial_file:
            return {}
        path = self.run_state_path.parent / partial_file
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                normalized: Dict[str, List[Dict[str, Any]]] = {}
                for key, value in data.items():
                    if isinstance(value, list):
                        normalized[key] = [v for v in value if isinstance(v, dict)]
                return normalized
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load partial predictions file: {e}")
        return {}

    def clear_partial_predictions(self):
        self.run_state.partial_predictions_file = None
        if self.partial_predictions_path.exists():
            try:
                self.partial_predictions_path.unlink()
            except OSError as e:
                logger.error(f"Failed to delete partial predictions file: {e}")

    def set_runtime_snapshot(self, snapshot: RuntimeSnapshot | Dict[str, Any]):
        self.run_state.runtime_snapshot = (
            snapshot
            if isinstance(snapshot, RuntimeSnapshot)
            else _pydantic_validate(RuntimeSnapshot, snapshot)
        )

    def set_stage(self, stage: RunStage, details: Optional[Dict[str, Any]] = None):
        self.run_state.stage = stage
        if details:
            self.run_state.stage_details.update(details)

    def set_pending_pairs(self, pairs: List[CandidatePair] | List[Dict[str, Any]]):
        if not pairs:
            self.pending_pairs = []
            self.run_state.pending_pairs_count = 0
            self.reset_llm_progress(total=0)
            self.clear_partial_links()
            self.clear_partial_predictions()
            return

        normalized: List[CandidatePair] = []
        for p in pairs:
            if isinstance(p, CandidatePair):
                normalized.append(p)
                continue
            try:
                normalized.append(_pydantic_validate(CandidatePair, p))
            except ValidationError:
                continue

        self.pending_pairs = normalized
        self._pending_pairs_dirty = True
        self.run_state.pending_pairs_count = len(self.pending_pairs)
        self.reset_llm_progress(total=len(self.pending_pairs))
        self.clear_partial_links()
        self.clear_partial_predictions()
        if self.pending_pairs:
            self.set_stage(RunStage.PENDING_LLM_PAIRS)

    def has_pending_pairs(self) -> bool:
        return bool(self.pending_pairs)

    def get_pending_pairs_as_llm_inputs(
        self,
    ) -> tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
        texts: List[Dict[str, str]] = []
        meta: List[Dict[str, Any]] = []

        for p in self.pending_pairs:
            texts.append({"text_a": p.text_a, "text_b": p.text_b})
            meta.append(
                {
                    "path_a": Path(p.path_a),
                    "path_b": Path(p.path_b),
                    "vector_distance": p.vector_distance,
                    "reranker_score": p.reranker_score,
                }
            )

        return texts, meta

    def clear_run_state(self, keep_snapshot: bool = True):
        self.pending_pairs = []
        if self.candidates_path.exists():
            try:
                self.candidates_path.unlink()
            except OSError as e:
                logger.error(f"Failed to delete candidates file: {e}")
        self.clear_partial_links()
        self.clear_partial_predictions()

        if keep_snapshot:
            snapshot = self.run_state.runtime_snapshot
            self.run_state = RunState(runtime_snapshot=snapshot, pending_pairs_count=0)
        else:
            self.run_state = RunState(pending_pairs_count=0)

        self._save_run_state()

    def clear_metadata(self):
        logger.warning("Clearing all metadata and resetting knowledge vault state...")

        self.vault = VaultMetadata()
        self._is_fresh_start = True
        self.clear_run_state(keep_snapshot=False)

        if self.metadata_path.exists():
            try:
                self.metadata_path.unlink()
                logger.info(f"Deleted metadata file: {self.metadata_path}")
            except OSError as e:
                logger.error(f"Error deleting metadata file: {e}")
        else:
            logger.info("No metadata file found on disk to delete.")

    def get_file_record(self, file_path_str: str) -> Optional[FileMetadata]:
        return self.vault.get_file(file_path_str)

    def add_or_update_file_record(self, file_meta: FileMetadata):
        self.vault.add_or_update_file(file_meta)
        logger.debug(f"Added/updated record for file: {file_meta.file_path}")

    def remove_file_record(self, file_path_str: str):
        self.vault.remove_file(file_path_str)
        logger.debug(f"Removed record for file: {file_path_str}")
