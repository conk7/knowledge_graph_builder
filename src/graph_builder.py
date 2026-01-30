import logging
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Set, Tuple
from pydantic import BaseModel
from tqdm import tqdm

from vault_manager import VaultManager
from metadata_manager import MetadataManager, FileMetadata, ChunkMetadata
from embedding_service import EmbeddingService
from vector_store import VectorStore
from llm_service import LLMService

from config import (
    RERANKER_THRESHOLD,
    RERANKER_TOP_K,
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
    INITIAL_RETRIEVAL_K,
    RERANKER_MODEL_NAME,
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
            embedding_model_name=EMBEDDING_MODEL_NAME,
            reranker_model_name=RERANKER_MODEL_NAME,
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
                batch_inputs, batch_metadata = self._collect_reranked_candidates(
                    new_chunks_data
                )
                if batch_inputs:
                    links_to_write = self._classify_and_format_links(
                        batch_inputs, batch_metadata
                    )

                    if links_to_write:
                        self._save_new_links(links_to_write)
                    else:
                        logging.info("LLM classified all candidates as irrelevant.")
                else:
                    logging.info("No candidates passed the reranker threshold.")
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

    def _collect_reranked_candidates(
        self, new_chunks_data: List[NewlyAddedChunk]
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        logging.info(f"Finding candidates for {len(new_chunks_data)} new chunks...")

        all_text_inputs = []
        all_metadata = []
        pbar = tqdm(
            enumerate(new_chunks_data, start=1), total=len(new_chunks_data), leave=False
        )
        for i, curr_chunk in pbar:
            pbar.set_description(f"Processing chunk {i} from '{curr_chunk.file_path}'")

            distances, other_chunk_ids = self.vector_store.search(
                curr_chunk.vector, INITIAL_RETRIEVAL_K
            )

            candidates_map = {}
            candidate_texts = []

            for distance, other_chunk_id in zip(distances, other_chunk_ids):
                if other_chunk_id == curr_chunk.faiss_id:
                    continue

                other_chunk_info = self.metadata_manager.get_chunk_info_by_faiss_id(
                    other_chunk_id
                )
                if (
                    not other_chunk_info
                    or other_chunk_info["file_path"] == curr_chunk.file_path
                ):
                    continue

                other_file_path = VAULT_PATH / other_chunk_info["file_path"]
                other_full_content = self.vault_manager.get_file_content(
                    other_file_path
                )
                other_chunks = self.embedding_service.chunk_text(other_full_content)
                other_chunk_index = other_chunk_info["chunk_index"]

                if other_chunk_index >= len(other_chunks):
                    continue

                other_text = other_chunks[other_chunk_index]

                candidate_texts.append(other_text)
                candidates_map[other_text] = {
                    "file_path": other_chunk_info["file_path"],
                }

            if not candidate_texts:
                continue

            reranked_results = self.embedding_service.rerank(
                query=curr_chunk.content,
                candidates=candidate_texts,
                top_k=RERANKER_TOP_K,
                threshold=RERANKER_THRESHOLD,
            )

            for text, _ in reranked_results:
                meta = candidates_map[text]

                all_text_inputs.append(
                    {
                        "text_a": curr_chunk.content,
                        "text_b": text,
                    }
                )

                all_metadata.append(
                    {
                        "curr_chunk_file_path": curr_chunk.file_path,
                        "other_chunk_file_path": meta["file_path"],
                    }
                )

        logging.info(f"Total candidates passed to LLM: {len(all_text_inputs)}")
        return all_text_inputs, all_metadata

    def _classify_and_format_links(
        self, text_inputs: List[Dict[str, str]], text_metadata: List[Dict[str, str]]
    ) -> Dict[str, Set[str]]:
        logging.info(f"Classifying {len(text_inputs)} pairs using LLM...")

        relation_results = self.llm_service.batch_classify_link(
            text_inputs,
            relation_types=self.metadata_manager.config.llm_link_types,
        )

        links_to_write = defaultdict(set)
        valid_link_count = 0

        for relation, meta in zip(relation_results, text_metadata):
            if not relation:
                continue

            target_file_name = (VAULT_PATH / meta["other_chunk_file_path"]).stem
            relation_text = self.metadata_manager.config.link_en2ru_translation.get(
                relation, relation
            )

            link_str = self.metadata_manager.config.link_template.format(
                relation_type=relation_text,
                target_file_name=target_file_name,
            )

            links_to_write[meta["curr_chunk_file_path"]].add(link_str)
            valid_link_count += 1

        logging.info(f"LLM identified {valid_link_count} valid semantic links.")
        return links_to_write

    def _save_new_links(self, links_to_write: Dict[str, Set[str]]):
        logging.info("Saving new links to files...")
        for rel_path_str, links in links_to_write.items():
            file_abs_path = VAULT_PATH / rel_path_str
            self.vault_manager.append_links_to_file(file_abs_path, links)
            logging.info(f"Appended {len(links)} links to {rel_path_str}")

    def _save_state(self):
        logging.info("Saving vault state...")
        self.metadata_manager.save()
        self.vector_store.save()

    def close(self):
        self.llm_service.close()
