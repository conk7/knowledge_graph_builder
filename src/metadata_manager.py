from __future__ import annotations
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from config import (
    DEFAULT_LINK_TYPES,
    DEFAULT_LINK_EN2RU_TRANSLATION,
    DEFAULT_LINK_TEMPLATE,
)


@dataclass
class ChunkMetadata:
    faiss_id: int
    chunk_index: int
    text_preview: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChunkMetadata:
        return cls(
            faiss_id=data["faiss_id"],
            chunk_index=data["chunk_index"],
            text_preview=data["text_preview"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "faiss_id": self.faiss_id,
            "chunk_index": self.chunk_index,
            "text_preview": self.text_preview,
        }


@dataclass
class FileMetadata:
    file_path: str
    hash: str
    chunks: List[ChunkMetadata] = field(default_factory=list)
    tags: List[Any] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FileMetadata:
        return cls(
            file_path=data["file_path"],
            hash=data["hash"],
            chunks=[ChunkMetadata.from_dict(c) for c in data.get("chunks", [])],
            tags=data.get("tags", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "hash": self.hash,
            "chunks": [c.to_dict() for c in self.chunks],
            "tags": self.tags,
        }

    def get_faiss_ids(self) -> List[int]:
        return [chunk.faiss_id for chunk in self.chunks]


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
    link_en2ru_translation: Dict[str, str]

    @classmethod
    def get_defaults(cls) -> LinkConfig:
        return cls(
            link_template=DEFAULT_LINK_TEMPLATE,
            llm_link_types=DEFAULT_LINK_TYPES,
            link_en2ru_translation=DEFAULT_LINK_EN2RU_TRANSLATION,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "link_template": self.link_template,
            "llm_link_types": self.llm_link_types,
            "link_en2ru_translation": self.link_en2ru_translation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LinkConfig:
        defaults = cls.get_defaults()
        return cls(
            link_template=data.get("link_template", defaults.link_template),
            llm_link_types=data.get("llm_link_types", defaults.llm_link_types),
            link_en2ru_translation=data.get(
                "link_en2ru_translation", defaults.link_en2ru_translation
            ),
        )


class MetadataManager:
    def __init__(self, metadata_path: Path):
        self.metadata_path = metadata_path
        self.config_path = self.metadata_path.parent / "links_config.json"
        self.vault = VaultMetadata()
        self.faiss_id_to_chunk_info: Dict[int, Dict[str, Any]] = {}
        self.__next_faiss_id = 1
        self._is_fresh_start = False

        self._load_metadata()
        self._load_config()

    def _load_metadata(self):
        logging.info(f"Attempting to load metadata from {self.metadata_path}")
        if not self.metadata_path.exists():
            logging.info("Metadata file not found. Starting with a fresh state.")
            self._is_fresh_start = True
            return
        try:
            with self.metadata_path.open("r", encoding="utf-8") as f:
                raw_data = json.load(f)
            self.vault = VaultMetadata.from_dict(raw_data)
            self._build_reverse_map_and_set_next_id()
            logging.info(
                f"Metadata loaded successfully. Tracking {len(self.vault.files)} files."
            )
        except (json.JSONDecodeError, IOError, KeyError) as e:
            logging.error(
                f"Error loading or parsing metadata file: {e}. Starting with a fresh state."
            )
            self.vault = VaultMetadata()

    def _load_config(self):
        logging.info(f"Attempting to load config from {self.config_path}")
        if not self.config_path.exists():
            logging.info("Config file not found. Using defaults")
            self.config = LinkConfig.get_defaults()
            return

        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                raw_config = json.load(f)
            self.config = LinkConfig.from_dict(raw_config)
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Error loading config file: {e}. Using internal defaults.")
            self.config = LinkConfig.get_defaults()

        logging.info("link configuration loaded successfully.")

    def is_fresh_start(self) -> bool:
        return self._is_fresh_start

    def save(self):
        try:
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.error(f"Failed to create metadata folder: {e}")

        try:
            with self.metadata_path.open("w", encoding="utf-8") as f:
                json.dump(self.vault.to_dict(), f, indent=4, ensure_ascii=False)
            logging.info(f"Metadata saved successfully to {self.metadata_path}")
        except IOError as e:
            logging.error(f"Failed to save metadata: {e}")

        try:
            with self.config_path.open("w", encoding="utf-8") as f:
                json.dump(self.config.to_dict(), f, indent=4, ensure_ascii=False)
            logging.info(f"Configuration saved successfully to {self.config_path}")
        except IOError as e:
            logging.error(f"Failed to save configuration: {e}")

    def clear_metadata(self):
        logging.warning("Clearing all metadata and resetting knowledge vault state...")

        self.vault = VaultMetadata()
        self.faiss_id_to_chunk_info = {}
        self.__next_faiss_id = 1
        self._is_fresh_start = True

        if self.metadata_path.exists():
            try:
                self.metadata_path.unlink()
                logging.info(f"Deleted metadata file: {self.metadata_path}")
            except OSError as e:
                logging.error(f"Error deleting metadata file: {e}")
        else:
            logging.info("No metadata file found on disk to delete.")

    def _build_reverse_map_and_set_next_id(self):
        self.faiss_id_to_chunk_info = {}
        max_id = 0
        for file_meta in self.vault.files.values():
            for chunk_meta in file_meta.chunks:
                faiss_id = chunk_meta.faiss_id
                self.faiss_id_to_chunk_info[faiss_id] = {
                    "file_path": file_meta.file_path,
                    "chunk_index": chunk_meta.chunk_index,
                }
                if faiss_id > max_id:
                    max_id = faiss_id

        self.__next_faiss_id = max_id + 1

    def generate_new_faiss_id(self) -> int:
        new_id = self.__next_faiss_id
        self.__next_faiss_id += 1
        return new_id

    def get_file_record(self, file_path_str: str) -> Optional[FileMetadata]:
        return self.vault.get_file(file_path_str)

    def add_or_update_file_record(self, file_meta: FileMetadata):
        self.vault.add_or_update_file(file_meta)
        for chunk in file_meta.chunks:
            self.faiss_id_to_chunk_info[chunk.faiss_id] = {
                "file_path": file_meta.file_path,
                "chunk_index": chunk.chunk_index,
            }
        logging.debug(f"Added/updated record for file: {file_meta.file_path}")

    def remove_file_record(self, file_path_str: str) -> List[int]:
        file_to_remove = self.get_file_record(file_path_str)
        if not file_to_remove:
            return []

        ids_to_remove = file_to_remove.get_faiss_ids()
        self.vault.remove_file(file_path_str)

        for faiss_id in ids_to_remove:
            if faiss_id in self.faiss_id_to_chunk_info:
                del self.faiss_id_to_chunk_info[faiss_id]

        logging.debug(f"Removed record for file: {file_path_str}")
        return ids_to_remove

    def get_chunk_info_by_faiss_id(self, faiss_id: int) -> Optional[Dict[str, Any]]:
        lookup_info = self.faiss_id_to_chunk_info.get(faiss_id)
        if not lookup_info:
            return None

        file_path = lookup_info["file_path"]
        chunk_index = lookup_info["chunk_index"]

        file_meta = self.get_file_record(file_path)
        if file_meta and len(file_meta.chunks) > chunk_index:
            chunk_data = file_meta.chunks[chunk_index].to_dict()
            chunk_data["file_path"] = file_path
            return chunk_data
        return None
