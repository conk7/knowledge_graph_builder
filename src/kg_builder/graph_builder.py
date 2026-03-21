import json
import logging
import shutil
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from pydantic import BaseModel
from tqdm import tqdm

from src.shared.llm_service import LLMService

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    CHUNK_SIZE,
    DEFAULT_LINK_TYPES,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    INITIAL_RETRIEVAL_K,
    LINK_HEADER,
    LINKS_FILE_NAME,
    LLM_BACKEND,
    LLM_CONCURRENCY,
    LLM_MODEL_PATH,
    LLM_N_BATCH,
    LLM_N_CTX,
    LLM_N_GPU_LAYERS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LOG_FILE_NAME,
    META_DIR_NAME,
    METADATA_FILE_NAME,
    RERANKER_MODEL_NAME,
    RERANKER_THRESHOLD,
    RERANKER_TOP_K,
    VECTOR_SEARCH_WEIGHT,
)
from .metadata_manager import (
    ChunkingSnapshot,
    EmbeddingSnapshot,
    FileMetadata,
    LlmSnapshot,
    MetadataManager,
    ModelsSnapshot,
    RerankerSnapshot,
    RetrievalSnapshot,
    RunStage,
    RuntimeSnapshot,
)
from .vault_manager import VaultManager
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class SaveMode(Enum):
    INPLACE = "inplace"
    JSON = "json"
    EXPORT = "export"


class NewlyAddedChunk(BaseModel):
    content: str
    file_path: str

    class Config:
        arbitrary_types_allowed = True


class KnowledgeGraphBuilder:
    def __init__(
        self,
        vault_path: Path,
        ignored_dirs: List[Path] = None,
        fresh_start: bool = False,
        ignore_local_config: bool = False,
        use_api: bool = False,
        save_mode: SaveMode = SaveMode.INPLACE,
        export_path: Optional[Path] = None,
        output_json_path: Optional[Path] = None,
    ):
        self.vault_path = vault_path
        self.meta_dir = self.vault_path / META_DIR_NAME
        self.index_path = self.meta_dir
        self.metadata_path = self.meta_dir / METADATA_FILE_NAME

        self.save_mode = save_mode
        self.export_path = export_path or (
            self.vault_path.parent / (self.vault_path.name + "_enriched")
        )
        self.output_json_path = output_json_path or (self.meta_dir / LINKS_FILE_NAME)

        self.use_api = use_api
        self.metadata_manager = MetadataManager(self.metadata_path)

        current_config = self._resolve_runtime_config(fresh_start, ignore_local_config)
        self.runtime_config: RuntimeSnapshot = current_config

        ignored = ignored_dirs or []
        if self.meta_dir not in ignored:
            ignored.append(self.meta_dir)

        self.vault_manager = VaultManager(
            vault_path=self.vault_path,
            ignored_dirs=ignored,
            link_header=current_config.link_header,
        )

        self.llm_service = None

        if fresh_start:
            logger.info("Fresh start.")
            self.metadata_manager.clear_metadata()
            self._purge_meta_dir_files()
            all_files = self.vault_manager.scan_markdown_files()
            self.vault_manager.clear_all_ai_links(all_files)

        self.vector_store = VectorStore(
            index_path=self.index_path,
            embedding_model_name=current_config.models.embedding.model_name,
            reranker_model_name=current_config.models.reranker.model_name,
            chunk_size=current_config.chunking.chunk_size,
            chunk_overlap=current_config.chunking.chunk_overlap,
            separators=current_config.chunking.separators,
            vector_weight=current_config.retrieval.vector_search_weight,
            fresh_start=fresh_start,
        )

        self.metadata_manager.set_runtime_snapshot(current_config)

        logger.info(f"Successfully inited graph builder for: {self.vault_path}")

    def _resolve_runtime_config(
        self, fresh_start: bool, ignore_local_config: bool
    ) -> RuntimeSnapshot:
        local_config = RuntimeSnapshot(
            chunking=ChunkingSnapshot(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=CHUNK_SEPARATORS,
            ),
            retrieval=RetrievalSnapshot(
                initial_retrieval_k=INITIAL_RETRIEVAL_K,
                vector_search_weight=VECTOR_SEARCH_WEIGHT,
            ),
            models=ModelsSnapshot(
                embedding=EmbeddingSnapshot(
                    model_name=EMBEDDING_MODEL_NAME,
                    dimension=EMBEDDING_DIMENSION,
                ),
                reranker=RerankerSnapshot(
                    model_name=RERANKER_MODEL_NAME,
                    top_k=RERANKER_TOP_K,
                    threshold=RERANKER_THRESHOLD,
                ),
                llm=LlmSnapshot(
                    use_api=self.use_api,
                    model_path=str(LLM_MODEL_PATH),
                    backend=LLM_BACKEND,
                    temperature=LLM_TEMPERATURE,
                    top_p=LLM_TOP_P,
                    n_gpu_layers=LLM_N_GPU_LAYERS,
                    n_ctx=LLM_N_CTX,
                    n_batch=LLM_N_BATCH,
                    concurrency=LLM_CONCURRENCY,
                ),
            ),
            link_header=LINK_HEADER,
        )

        if fresh_start or ignore_local_config:
            logger.info("Using hyperparameters from config.py.")
            return local_config

        saved_snapshot = self.metadata_manager.run_state.runtime_snapshot
        if not saved_snapshot:
            logger.info("No saved hyperparameters found. Using config.py.")
            return local_config

        logger.info("Restoring hyperparameters from saved metadata (run_state.json).")
        return saved_snapshot

    def _purge_meta_dir_files(self):
        if not self.meta_dir.exists():
            return

        try:
            for p in sorted(self.meta_dir.rglob("*"), reverse=True):
                if p.is_file() or p.is_symlink():
                    if p.name == LOG_FILE_NAME:
                        continue
                    try:
                        p.unlink()
                    except OSError as e:
                        logger.warning(f"Could not delete meta file {p}: {e}")

            for p in sorted(self.meta_dir.rglob("*"), reverse=True):
                if p.is_dir():
                    try:
                        p.rmdir()
                    except OSError:
                        pass
        except Exception as e:
            logger.warning(f"Failed to purge meta dir {self.meta_dir}: {e}")

    def _init_llm_service(self):
        if self.llm_service is None:
            logger.info("Loading LLM Service...")
            llm_cfg = self.runtime_config.models.llm
            self.llm_service = LLMService(
                model_path=Path(llm_cfg.model_path),
                n_gpu_layers=llm_cfg.n_gpu_layers,
                n_ctx=llm_cfg.n_ctx,
                n_batch=llm_cfg.n_batch,
                temperature=llm_cfg.temperature,
                top_p=llm_cfg.top_p,
                use_api=llm_cfg.use_api,
                backend=llm_cfg.backend,
                concurrency=llm_cfg.concurrency,
                default_link_types=DEFAULT_LINK_TYPES,
            )

    def _unload_llm_service(self):
        if self.llm_service:
            logger.info("Unloading LLM Service...")
            del self.llm_service
            self.llm_service = None
            torch.cuda.empty_cache()

    def run_update(self):
        self.initialize_vault()

        if self.metadata_manager.has_pending_pairs():
            logger.warning(
                "Found pending candidate pairs from a previous run. Resuming LLM classification..."
            )
            resumed_ok = False
            try:
                texts, meta = self.metadata_manager.get_pending_pairs_as_llm_inputs()
                if not texts:
                    raise ValueError(
                        "Pending pairs exist, but could not be parsed into LLM inputs."
                    )

                self.metadata_manager.set_stage(
                    RunStage.RESUMING_LLM_CLASSIFICATION, {"pairs": len(texts)}
                )
                self.metadata_manager.save_run_state_only()

                self._init_llm_service()
                self._classify_pairs_with_checkpoints(
                    texts,
                    meta,
                    stage=RunStage.RESUMING_LLM_CLASSIFICATION,
                )
                self._unload_llm_service()
                resumed_ok = True
            except Exception as e:
                self.metadata_manager.set_stage(
                    RunStage.FAILED,
                    {"during": "resume_llm_classification", "error": str(e)},
                )
                raise
            finally:
                if resumed_ok:
                    self.metadata_manager.set_stage(RunStage.COMPLETED)
                    self.metadata_manager.clear_run_state(keep_snapshot=True)
                self._save_state()
            return

        try:
            files_to_add, files_to_update, files_to_remove = self._determine_changes()

            self._process_removals(files_to_update + files_to_remove)

            new_chunks_data = self._process_additions_and_updates(
                files_to_add + files_to_update
            )
            if not new_chunks_data:
                logger.info("No new files for linking.")
                return
            self.metadata_manager.set_stage(
                RunStage.COLLECTING_CANDIDATES, {"new_chunks": len(new_chunks_data)}
            )
            self.metadata_manager.save()

            texts, meta = self._collect_candidates(new_chunks_data)
            if self.metadata_manager.has_pending_pairs():
                self.metadata_manager.save()
            else:
                self.metadata_manager.set_stage(RunStage.NO_CANDIDATES)
                self.metadata_manager.save()

            if texts:
                self.metadata_manager.set_stage(
                    RunStage.CLASSIFYING_LLM_PAIRS, {"pairs": len(texts)}
                )
                self.metadata_manager.save_run_state_only()
                self._init_llm_service()

                try:
                    self._classify_pairs_with_checkpoints(
                        texts,
                        meta,
                        stage=RunStage.CLASSIFYING_LLM_PAIRS,
                    )
                except Exception as e:
                    self.metadata_manager.set_stage(
                        RunStage.FAILED,
                        {"during": "llm_classification", "error": str(e)},
                    )
                    self.metadata_manager.save_run_state_only()
                    raise

                self._unload_llm_service()
                self.metadata_manager.set_stage(RunStage.COMPLETED)
                self.metadata_manager.clear_run_state(keep_snapshot=True)
            else:
                logger.info("No candidates passed the reranker.")
        finally:
            self._save_state()
            logger.info("Updated finished.")

    def _determine_changes(self) -> Tuple[List[Path], List[Path], List[Path]]:
        logger.info("Looking for changes in vault..")
        files_paths = self.vault_manager.scan_markdown_files()
        files_rel_paths = {p.relative_to(self.vault_path) for p in files_paths}

        tracked_files_rel = set(map(Path, self.metadata_manager.vault.files.keys()))

        added_files = [
            self.vault_path / p for p in (files_rel_paths - tracked_files_rel)
        ]
        removed_files = [
            self.vault_path / p for p in (tracked_files_rel - files_rel_paths)
        ]

        potential_updates = files_rel_paths.intersection(tracked_files_rel)
        updated_files = []
        for file_rel in potential_updates:
            file_abs = self.vault_path / file_rel
            if not file_abs.exists():
                continue
            old_hash = self.metadata_manager.get_file_record(str(file_rel)).hash
            new_hash = self.vault_manager.calculate_file_hash(file_abs)
            if old_hash != new_hash:
                updated_files.append(file_abs)

        logger.info(
            f"Found: {len(added_files)} new, {len(updated_files)} updated, {len(removed_files)} deleted files."
        )
        return added_files, updated_files, removed_files

    def _process_removals(self, files_to_process: List[Path]):
        if not files_to_process:
            return
        logger.info(f"Processing {len(files_to_process)} updated/deleted files...")

        for file_path in files_to_process:
            rel_path_str = str(file_path.relative_to(self.vault_path))
            self.metadata_manager.remove_file_record(rel_path_str)
            self.vector_store.remove_document(rel_path_str)

    def _process_additions_and_updates(
        self, files_to_process: List[Path]
    ) -> List[NewlyAddedChunk]:
        if not files_to_process:
            return []
        logger.info(f"Indexing/Retrieving {len(files_to_process)} files...")

        newly_added_chunks = []
        for file_path in files_to_process:
            rel_path_str = str(file_path.relative_to(self.vault_path))
            content = self.vault_manager.get_file_content(file_path)
            if not content:
                continue

            file_hash = self.vault_manager.calculate_file_hash(file_path)
            chunk_texts = self.vector_store.get_and_update_file_chunks(
                rel_path_str, content, file_hash
            )

            if not chunk_texts:
                continue

            for chunk_text in chunk_texts:
                newly_added_chunks.append(
                    NewlyAddedChunk(content=chunk_text, file_path=rel_path_str)
                )

            file_meta = FileMetadata(file_path=rel_path_str, hash=file_hash)
            self.metadata_manager.add_or_update_file_record(file_meta)
            logger.info(f"Processed '{file_path.name}' ({len(chunk_texts)} chunks).")

        return newly_added_chunks

    def initialize_vault(self):
        logger.info(f"Initializing AI metadata for vault: {self.vault_path}")
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_manager.save()

    def _collect_candidates(
        self, new_chunks_data: List[NewlyAddedChunk]
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        logger.info(f"Finding candidates for {len(new_chunks_data)} new chunks...")

        all_text_inputs = []
        all_metadata = []
        pending_pairs: List[Dict[str, Any]] = []
        pbar = tqdm(new_chunks_data, total=len(new_chunks_data), leave=False)

        retrieval_k = self.runtime_config.retrieval.initial_retrieval_k
        reranker_top_k = self.runtime_config.models.reranker.top_k
        reranker_threshold = self.runtime_config.models.reranker.threshold

        for curr_chunk in pbar:
            pbar.set_description(f"Processing '{curr_chunk.file_path}'")

            search_results = self.vector_store.search(curr_chunk.content, retrieval_k)

            candidates_map = {}
            candidate_texts = []

            for result in search_results:
                other_text = result["text"]
                other_file_path = result["file_path"]
                distance = result.get("_distance", 0.0)

                if other_file_path == curr_chunk.file_path:
                    continue

                if other_text not in candidates_map:
                    candidate_texts.append(other_text)
                    candidates_map[other_text] = {
                        "file_path": other_file_path,
                        "vector_distance": distance,
                    }

            if not candidate_texts:
                continue

            reranked_results = self.vector_store.rerank(
                query=curr_chunk.content,
                candidates=candidate_texts,
                top_k=reranker_top_k,
                threshold=reranker_threshold,
            )

            for text, score in reranked_results:
                meta = candidates_map[text]

                all_text_inputs.append(
                    {
                        "text_a": curr_chunk.content,
                        "text_b": text,
                    }
                )

                all_metadata.append(
                    {
                        "path_a": Path(curr_chunk.file_path),
                        "path_b": Path(meta["file_path"]),
                        "vector_distance": meta.get("vector_distance"),
                        "reranker_score": float(score),
                    }
                )

                pending_pairs.append(
                    {
                        "text_a": curr_chunk.content,
                        "text_b": text,
                        "path_a": curr_chunk.file_path,
                        "path_b": meta["file_path"],
                        "vector_distance": meta.get("vector_distance"),
                        "reranker_score": float(score),
                    }
                )

        logger.info(f"Total candidates passed to LLM: {len(all_text_inputs)}")
        self.metadata_manager.set_pending_pairs(pending_pairs)
        return all_text_inputs, all_metadata

    def _classify_and_format_links(
        self, texts: List[Dict[str, str]], text_meta: List[Dict[str, Any]]
    ) -> Dict[Path, Set[str]]:
        logger.info(f"Classifying {len(texts)} pairs using LLM...")

        relation_results = self.llm_service.batch_classify_link(
            texts,
            text_meta,
            relation_types=self.metadata_manager.config.llm_link_types,
        )

        links_to_write = self._resolve_and_format_links_for_batch(
            batch_texts=texts,
            batch_meta=text_meta,
            relation_results=relation_results,
        )
        valid_link_count = sum(len(v) for v in links_to_write.values())

        logger.info(f"LLM identified {valid_link_count} valid semantic links.")
        return links_to_write

    def _resolve_and_format_links_for_batch(
        self,
        batch_texts: List[Dict[str, str]],
        batch_meta: List[Dict[str, Any]],
        relation_results: List[Optional[str]],
    ) -> Dict[Path, Set[str]]:
        pair_predictions = self._collect_pair_predictions_from_batch(
            batch_texts=batch_texts,
            batch_meta=batch_meta,
            relation_results=relation_results,
        )
        return self._resolve_pair_predictions(pair_predictions)

    def _collect_pair_predictions_from_batch(
        self,
        batch_texts: List[Dict[str, str]],
        batch_meta: List[Dict[str, Any]],
        relation_results: List[Optional[str]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        pair_predictions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for relation, text_item, meta_item in zip(
            relation_results, batch_texts, batch_meta
        ):
            if not relation:
                continue

            path_a = Path(meta_item["path_a"])
            path_b = Path(meta_item["path_b"])
            pair_key = f"{path_a}||{path_b}"
            pair_predictions[pair_key].append(
                {
                    "relation_type": relation,
                    "text_a": text_item["text_a"],
                    "text_b": text_item["text_b"],
                    "reranker_score": meta_item.get("reranker_score"),
                    "vector_distance": meta_item.get("vector_distance"),
                }
            )
        return dict(pair_predictions)

    def _merge_pair_predictions(
        self,
        base: Dict[str, List[Dict[str, Any]]],
        incoming: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        for pair_key, rows in incoming.items():
            if not rows:
                continue
            base.setdefault(pair_key, []).extend(rows)
        return base

    def _resolve_pair_predictions(
        self,
        pair_predictions: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[Path, Set[str]]:
        normalized_predictions: Dict[Tuple[str, str], List[Dict[str, Any]]] = (
            defaultdict(list)
        )
        for pair_key, rows in pair_predictions.items():
            if "||" not in pair_key:
                continue
            path_a_str, path_b_str = pair_key.split("||", 1)
            normalized_predictions[(path_a_str, path_b_str)].extend(rows)

        final_relations: Dict[Tuple[str, str], str] = {}
        conflict_inputs: List[Dict[str, Any]] = []
        conflict_keys: List[Tuple[str, str]] = []
        for pair_key, evidence_rows in normalized_predictions.items():
            candidate_counts: Dict[str, int] = defaultdict(int)
            for row in evidence_rows:
                candidate_counts[row["relation_type"]] += 1

            if len(candidate_counts) == 1:
                final_relations[pair_key] = next(iter(candidate_counts.keys()))
                continue

            path_a_str, path_b_str = pair_key
            conflict_inputs.append(
                {
                    "filename_a": (self.vault_path / path_a_str).stem,
                    "filename_b": (self.vault_path / path_b_str).stem,
                    "candidate_counts": dict(candidate_counts),
                    "evidence": evidence_rows,
                }
            )
            conflict_keys.append(pair_key)

        if conflict_inputs:
            logger.info(
                f"Resolving {len(conflict_inputs)} link-type conflicts via LLM..."
            )
            resolved_relations = self.llm_service.batch_resolve_link_conflicts(
                conflict_inputs,
                relation_types=self.metadata_manager.config.llm_link_types,
                show_progress=False,
            )
            for pair_key, resolved_relation in zip(conflict_keys, resolved_relations):
                if resolved_relation:
                    final_relations[pair_key] = resolved_relation

        links_to_write: Dict[Path, Set[str]] = defaultdict(set)
        for (path_a_str, path_b_str), relation in final_relations.items():
            source_rel_path = Path(path_a_str)
            target_rel_path = Path(path_b_str)
            link_str = self._build_link_string(relation, target_rel_path)
            links_to_write[source_rel_path].add(link_str)

        return links_to_write

    def _build_link_string(self, relation: str, target_rel_path: Path) -> str:
        target_file_name = (self.vault_path / target_rel_path).stem
        relation_text = relation
        return self.metadata_manager.config.link_template.format(
            relation_type=relation_text,
            target_file_name=target_file_name,
        )

    def _classify_pairs_with_checkpoints(
        self,
        texts: List[Dict[str, str]],
        text_meta: List[Dict[str, Any]],
        stage: RunStage,
    ):
        total = len(texts)
        if total == 0:
            return

        offset, saved_total = self.metadata_manager.get_llm_progress()
        if saved_total != total:
            offset = min(offset, total)
            self.metadata_manager.set_llm_progress(offset=offset, total=total)
            self.metadata_manager.save_run_state_only()

        pair_predictions_accum = self.metadata_manager.load_partial_predictions()

        llm_concurrency = self.llm_service.concurrency if self.use_api else 1
        llm_concurrency = max(1, int(llm_concurrency))
        batch_size = max(llm_concurrency, 10 * llm_concurrency)

        with tqdm(
            total=total,
            initial=offset,
            desc="Classifying Semantic Links",
            unit="pair",
            leave=False,
        ) as pbar:
            for start in range(offset, total, batch_size):
                end = min(start + batch_size, total)
                batch_texts = texts[start:end]
                batch_meta = text_meta[start:end]

                relation_results = self.llm_service.batch_classify_link(
                    batch_texts,
                    batch_meta,
                    relation_types=self.metadata_manager.config.llm_link_types,
                    pbar=pbar,
                    show_progress=False,
                )

                batch_pair_predictions = self._collect_pair_predictions_from_batch(
                    batch_texts=batch_texts,
                    batch_meta=batch_meta,
                    relation_results=relation_results,
                )
                pair_predictions_accum = self._merge_pair_predictions(
                    pair_predictions_accum, batch_pair_predictions
                )

                self.metadata_manager.set_stage(
                    stage,
                    {
                        "offset": end,
                        "total": total,
                        "batch_size": batch_size,
                        "llm_concurrency": llm_concurrency,
                    },
                )
                self.metadata_manager.set_llm_progress(offset=end, total=total)
                self.metadata_manager.save_partial_predictions(pair_predictions_accum)
                self.metadata_manager.save_run_state_only()

        links_to_write = self._resolve_pair_predictions(pair_predictions_accum)
        if links_to_write:
            self._save_new_links(links_to_write)

    def _save_new_links(self, links_to_write: Dict[Path, Set[str]]):
        logger.info(f"Saving new links (Mode: {self.save_mode.value})...")

        if self.save_mode == SaveMode.INPLACE:
            for rel_path_str, links in links_to_write.items():
                file_abs_path = self.vault_path / rel_path_str
                self.vault_manager.append_links_to_file(file_abs_path, links)
                logger.info(f"Appended {len(links)} links to {rel_path_str}")
        elif self.save_mode == SaveMode.JSON:
            self._save_links_to_json(links_to_write)
        elif self.save_mode == SaveMode.EXPORT:
            self._export_enriched_vault(links_to_write)

    def _save_links_to_json(self, links_to_write: Dict[str, Set[str]]):
        logger.info(f"Saving new links to JSON: {self.output_json_path}")
        serializable_links = {
            str(path): list(links) for path, links in links_to_write.items()
        }
        self.output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_json_path, "w", encoding="utf-8") as f:
            json.dump(serializable_links, f, ensure_ascii=False, indent=4)

    def _export_enriched_vault(self, links_to_write: Dict[str, Set[str]]):
        logger.info(f"Exporting enriched vault to: {self.export_path}")

        if self.export_path.exists():
            logger.info(f"Removing existing export directory: {self.export_path}")
            shutil.rmtree(self.export_path)

        shutil.copytree(
            self.vault_path,
            self.export_path,
            ignore=shutil.ignore_patterns(".ai_meta", ".obsidian", "*.log"),
        )

        for rel_path_str, links in links_to_write.items():
            exported_file_path = self.export_path / rel_path_str
            if exported_file_path.exists():
                self.vault_manager.append_links_to_file(exported_file_path, links)
                logger.info(f"Appended {len(links)} links to EXPORTED {rel_path_str}")

    def _save_state(self):
        logger.info("Saving vault state...")
        self.metadata_manager.save()
        self.vector_store.save()

    def close(self):
        if self.llm_service:
            self.llm_service.close()
