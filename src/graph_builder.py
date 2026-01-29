import logging
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple
from pydantic import BaseModel

from vault_manager import VaultManager
from metadata_manager import MetadataManager, FileMetadata, ChunkMetadata
from embedding_service import EmbeddingService
from vector_store import VectorStore
from llm_service import LLMService

from config import (
    VAULT_PATH,
    IGNORED_DIRS,
    METADATA_PATH,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    INDEX_PATH,
    EMBEDDING_DIMENSION,
    LLM_MODEL_PATH,
    LLM_N_GPU_LAYERS,
    LLM_N_CTX,
    LLM_TEMPERATURE,
    SIMILARITY_SEARCH_K,
    SIMILARITY_DISTANCE_THRESHOLD,
)


class NewlyAddedChunk(BaseModel):
    faiss_id: int
    content: str
    vector: np.ndarray
    file_path: str  # Relative path

    class Config:
        arbitrary_types_allowed = True


class KnowledgeGraphBuilder:
    def __init__(
        self,
        fresh_start: bool = False,
        use_google_api: bool = False,
    ):
        self.vault_manager = VaultManager(
            vault_path=VAULT_PATH, ignored_dirs=IGNORED_DIRS
        )
        self.metadata_manager = MetadataManager(METADATA_PATH)
        self.embedding_service = EmbeddingService(
            model_name=EMBEDDING_MODEL_NAME,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=CHUNK_SEPARATORS,
        )
        self.vector_store = VectorStore(
            index_path=INDEX_PATH, dimension=EMBEDDING_DIMENSION
        )
        self.llm_service = LLMService(
            model_path=LLM_MODEL_PATH,
            n_gpu_layers=LLM_N_GPU_LAYERS,
            n_ctx=LLM_N_CTX,
            temperature=LLM_TEMPERATURE,
            use_google_api=use_google_api,
        )

        if self.metadata_manager.is_fresh_start() or fresh_start:
            self.metadata_manager.clear_metadata()
            all_files = self.vault_manager.scan_markdown_files()
            self.vault_manager.clear_all_ai_links(all_files)

        logging.info("Successfully inited graph builder.")

    def run_update(self):
        try:
            files_to_add, files_to_update, files_to_remove = self._determine_changes()

            self._process_removals(files_to_update + files_to_remove)

            new_chunks_data = self._process_additions_and_updates(
                files_to_add + files_to_update
            )

            if new_chunks_data:
                self._find_and_save_new_links(new_chunks_data)
            else:
                logging.info("No new files for linking.")

        finally:
            self._save_state()
            logging.info("Updated finished.")

    def _determine_changes(self) -> Tuple[List[Path], List[Path], List[Path]]:
        logging.info("Looking for changes in vault..")
        files_paths = self.vault_manager.scan_markdown_files()
        files_rel_paths = {p.relative_to(VAULT_PATH) for p in files_paths}

        tracked_files_rel = set(map(Path, self.metadata_manager.vault.files.keys()))

        added_files = [VAULT_PATH / p for p in (files_rel_paths - tracked_files_rel)]
        removed_files = [VAULT_PATH / p for p in (tracked_files_rel - files_rel_paths)]

        potential_updates = files_rel_paths.intersection(tracked_files_rel)
        updated_files = []
        for file_rel in potential_updates:
            file_abs = VAULT_PATH / file_rel
            if not file_abs.exists():
                continue
            old_hash = self.metadata_manager.get_file_record(str(file_rel)).hash
            new_hash = self.vault_manager.calculate_file_hash(file_abs)
            if old_hash != new_hash:
                updated_files.append(file_abs)

        logging.info(
            f"Found: {len(added_files)} new, {len(updated_files)} updated, {len(removed_files)} deleted files."
        )
        return added_files, updated_files, removed_files

    def _process_removals(self, files_to_process: List[Path]):
        if not files_to_process:
            return
        logging.info(f"Processing {len(files_to_process)} updated/deleted files...")

        all_ids_to_remove = []
        for file_path in files_to_process:
            rel_path_str = str(file_path.relative_to(VAULT_PATH))
            ids = self.metadata_manager.remove_file_record(rel_path_str)
            all_ids_to_remove.extend(ids)

        if all_ids_to_remove:
            self.vector_store.remove(all_ids_to_remove)
            logging.info(
                f"Deleted {len(all_ids_to_remove)} vectors from index storage."
            )

    def _process_additions_and_updates(
        self, files_to_process: List[Path]
    ) -> List[Dict]:
        if not files_to_process:
            return []
        logging.info(f"Indexing {len(files_to_process)} new/updated files...")

        newly_added_chunks = []
        for file_path in files_to_process:
            rel_path_str = str(file_path.relative_to(VAULT_PATH))
            content = self.vault_manager.get_file_content(file_path)
            if not content:
                continue

            chunks = self.embedding_service.chunk_text(content)
            if not chunks:
                continue

            embeddings = self.embedding_service.get_embeddings(chunks)
            file_hash = self.vault_manager.calculate_file_hash(file_path)

            chunk_records: List[ChunkMetadata] = []
            ids_to_add = []

            for i, chunk_text in enumerate(chunks):
                new_faiss_id = self.metadata_manager.generate_new_faiss_id()
                chunk_meta = ChunkMetadata(
                    faiss_id=new_faiss_id,
                    chunk_index=i,
                    text_preview=chunk_text[:100].replace("\n", " ") + "...",
                )
                chunk_records.append(chunk_meta)
                ids_to_add.append(new_faiss_id)

                newly_added_chunks.append(
                    NewlyAddedChunk(
                        faiss_id=new_faiss_id,
                        content=chunk_text,
                        vector=embeddings[i],
                        file_path=rel_path_str,
                    )
                )

            self.vector_store.add(embeddings, ids_to_add)

            file_meta = FileMetadata(
                file_path=rel_path_str, hash=file_hash, chunks=chunk_records
            )
            self.metadata_manager.add_or_update_file_record(file_meta)
            logging.info(
                f"Successfully indexed '{file_path.name}' ({len(chunks)} chunks)."
            )

        return newly_added_chunks

    def _find_and_save_new_links(self, new_chunks_data: List[NewlyAddedChunk]):
        logging.info(f"Searching for links in {len(new_chunks_data)} new chunks...")

        links_to_write = defaultdict(set)

        for i, curr_chunk in enumerate(new_chunks_data, start=1):
            logging.info(
                f"Searching in chunk {i}/{len(new_chunks_data)} from '{curr_chunk.file_path}'..."
            )

            distances, other_chunk_ids = self.vector_store.search(
                curr_chunk.vector, SIMILARITY_SEARCH_K
            )

            relevance_batch_input = []
            candidate_metadata = []

            for distance, other_chunk_id in zip(distances, other_chunk_ids):
                if other_chunk_id == curr_chunk.faiss_id:
                    continue

                other_chunk = self.metadata_manager.get_chunk_info_by_faiss_id(
                    other_chunk_id
                )
                if not other_chunk or other_chunk["file_path"] == curr_chunk.file_path:
                    continue

                logging.debug(
                    f"distance = {distance:.2f}, {curr_chunk.file_path} vs {other_chunk['file_path']}"
                )
                if distance > SIMILARITY_DISTANCE_THRESHOLD:
                    continue

                other_file_path = VAULT_PATH / other_chunk["file_path"]
                other_full_content = self.vault_manager.get_file_content(
                    other_file_path
                )
                other_chunks = self.embedding_service.chunk_text(other_full_content)
                other_chunk_index = other_chunk["chunk_index"]

                if other_chunk_index >= len(other_chunks):
                    logging.warning(
                        f" Inxed of {other_chunk_index} in out of range of {other_file_path}."
                    )
                    continue

                relevance_batch_input.append(
                    {
                        "text_a": curr_chunk.content,
                        "text_b": other_chunks[other_chunk_index],
                    }
                )
                candidate_metadata.append(
                    {
                        "curr_chunk_file_path": curr_chunk.file_path,
                        "other_chunk_file_path": other_chunk["file_path"],
                        "other_chunk_content": other_chunks[other_chunk_index],
                    }
                )

                if not relevance_batch_input:
                    continue

                logging.info(
                    f"Checking relevance for {len(relevance_batch_input)} potential links..."
                )
                relevance_results = self.llm_service.batch_check_relevance(
                    relevance_batch_input
                )

                classification_batch_input = []
                classification_metadata = []

                for is_relevant, meta in zip(relevance_results, candidate_metadata):
                    if is_relevant:
                        classification_batch_input.append(
                            {
                                "text_a": curr_chunk.content,
                                "text_b": meta["other_chunk_content"],
                            }
                        )
                        classification_metadata.append(meta)
                    else:
                        logging.info(
                            f"Skipping irrelevant link between {meta['curr_chunk_file_path']} and {meta['other_chunk_file_path']}..."
                        )

                if not classification_batch_input:
                    continue

                logging.info(
                    f"Classifying {len(classification_batch_input)} relevant links..."
                )
                relation_results = self.llm_service.batch_classify_link(
                    classification_batch_input,
                    relation_types=self.metadata_manager.config.llm_link_types,
                )

                for relation, meta in zip(relation_results, classification_metadata):
                    if not relation:
                        continue

                    target_file_name = (VAULT_PATH / meta["other_chunk_file_path"]).stem
                    relation_text = (
                        self.metadata_manager.config.link_en2ru_translation.get(
                            relation, relation
                        )
                    )
                    link_str = self.metadata_manager.config.link_template.format(
                        relation_type=relation_text,
                        target_file_name=target_file_name,
                    )
                    links_to_write[meta["curr_chunk_file_path"]].add(link_str)
                    logging.info(
                        f"Made a '{relation_text}' link between {meta['curr_chunk_file_path']} and {meta['other_chunk_file_path']}"
                    )

        if links_to_write:
            logging.info("Saving new links...")
            for rel_path_str, links in links_to_write.items():
                file_abs_path = VAULT_PATH / rel_path_str
                self.vault_manager.append_links_to_file(file_abs_path, links)

    def _save_state(self):
        logging.info("Saving vault state...")
        self.metadata_manager.save()
        self.vector_store.save()

    def close(self):
        self.llm_service.close()
