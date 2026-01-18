from __future__ import annotations
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


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
class KnowledgeVault:
    files: Dict[str, FileMetadata] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KnowledgeVault:
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


class MetadataManager:
    def __init__(self, metadata_path: Path):
        self.metadata_path = metadata_path
        self.vault = KnowledgeVault()
        self.faiss_id_to_chunk_info: Dict[int, Dict[str, Any]] = {}
        self.__next_faiss_id = 1
        self.__is_fresh_start = False

        logging.info(f"Attempting to load metadata from {self.metadata_path}")
        if not self.metadata_path.exists():
            logging.info("Metadata file not found. Starting with a fresh state.")
            self.__is_fresh_start = True
            return
        try:
            with self.metadata_path.open("r", encoding="utf-8") as f:
                raw_data = json.load(f)
            self.vault = KnowledgeVault.from_dict(raw_data)
            self._build_reverse_map_and_set_next_id()
        except (json.JSONDecodeError, IOError, KeyError) as e:
            logging.error(
                f"Error loading or parsing metadata file: {e}. Starting with a fresh state."
            )
            self.vault = KnowledgeVault()

        logging.info(
            f"Metadata loaded successfully. Tracking {len(self.vault.files)} files."
        )

    def is_fresh_start(self) -> bool:
        return self.__is_fresh_start

    def save(self):
        try:
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with self.metadata_path.open("w", encoding="utf-8") as f:
                json.dump(self.vault.to_dict(), f, indent=4, ensure_ascii=False)
            logging.info(f"Metadata saved successfully to {self.metadata_path}")
        except IOError as e:
            logging.error(f"Failed to save metadata: {e}")

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
