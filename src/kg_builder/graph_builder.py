import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.shared.llm_service import LLMService

from .config import (
    CANDIDATES_FILE_NAME,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    CHUNK_SIZE,
    SENTENCE_WINDOW_AFTER,
    SENTENCE_WINDOW_BEFORE,
    SPLITTER_TYPE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    INITIAL_RETRIEVAL_K,
    LINK_HEADER,
    LLM_BACKEND,
    LLM_CONCURRENCY,
    LLM_MODEL_PATH,
    LLM_N_BATCH,
    LLM_N_CTX,
    LLM_N_GPU_LAYERS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    META_DIR_NAME,
    OUTPUT_DIR,
    OUTPUT_LINKS_FILE_NAME,
    RERANKED_CANDIDATES_FILE_NAME,
    RERANKER_MODEL_NAME,
    RERANKER_THRESHOLD,
    RERANKER_TOP_K,
    VECTOR_SEARCH_WEIGHT,
)
from .export_service import ExportService, SaveMode
from .metadata_manager import (
    ChunkingSnapshot,
    EmbeddingSnapshot,
    LlmSnapshot,
    MetadataManager,
    ModelsSnapshot,
    RerankerSnapshot,
    RetrievalSnapshot,
    RunStage,
    RuntimeSnapshot,
)
from .models import CandidatePair, NewlyAddedChunk
from .retrieval import (
    BroadQueryMode,
    CombinedRetrievalStrategy,
    RetrievalStrategyMode,
    StrictRetrievalStrategy,
    VectorSearchRerankStrategy,
)
from .vault_manager import VaultManager
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


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
        retrieval_strategy_name: RetrievalStrategyMode = RetrievalStrategyMode.STRICT,
        broad_query_mode: BroadQueryMode = BroadQueryMode.TITLE_SUMMARY,
        lang: Optional[str] = None,
        splitter_type: str = SPLITTER_TYPE,
        sentence_window_before: int = SENTENCE_WINDOW_BEFORE,
        sentence_window_after: int = SENTENCE_WINDOW_AFTER,
    ):
        self.vault_path = vault_path
        self.meta_dir = self.vault_path / META_DIR_NAME
        self.output_dir = self.meta_dir / OUTPUT_DIR
        self.index_path = self.output_dir
        self.metadata_path = self.meta_dir
        self.candidates_path = self.output_dir / CANDIDATES_FILE_NAME
        self.reranked_candidates_path = self.output_dir / RERANKED_CANDIDATES_FILE_NAME

        self.save_mode = save_mode
        self.export_path = export_path or (
            self.vault_path.parent / (self.vault_path.name + "_enriched")
        )
        self.output_json_path = output_json_path or (
            self.meta_dir / OUTPUT_LINKS_FILE_NAME
        )

        self.use_api = use_api
        self.retrieval_strategy_name = retrieval_strategy_name
        self.broad_query_mode = broad_query_mode
        self.metadata_manager = MetadataManager(self.metadata_path)

        self.lang = lang
        self.splitter_type = splitter_type
        self.sentence_window_before = sentence_window_before
        self.sentence_window_after = sentence_window_after

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

        self.llm_service = LLMService(
            vault_path=self.vault_path,
            metadata_manager=self.metadata_manager,
            reranked_candidates_path=self.reranked_candidates_path,
        )

        self.export_service = ExportService(
            vault_path=self.vault_path,
            vault_manager=self.vault_manager,
            export_path=self.export_path,
            output_json_path=self.output_json_path,
        )

        if fresh_start:
            logger.info("Fresh start.")
            self.metadata_manager.clear_metadata()
            self.metadata_manager.purge_meta_dir_files()
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
            lang=self.lang,
            splitter_type=current_config.chunking.splitter_type,
            sentence_window_before=current_config.chunking.sentence_window_before,
            sentence_window_after=current_config.chunking.sentence_window_after,
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
                splitter_type=self.splitter_type,
                sentence_window_before=self.sentence_window_before,
                sentence_window_after=self.sentence_window_after,
            ),
            retrieval=RetrievalSnapshot(
                initial_retrieval_k=INITIAL_RETRIEVAL_K,
                vector_search_weight=VECTOR_SEARCH_WEIGHT,
                broad_query_mode=self.broad_query_mode.value,
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

        if self.metadata_manager.is_fresh_start():
            logger.info("No saved hyperparameters found. Using config.py.")
            return local_config

        saved_snapshot = self.metadata_manager.config.to_runtime_snapshot()
        logger.info("Restoring hyperparameters from saved config (config.json).")
        if saved_snapshot.retrieval.broad_query_mode != self.broad_query_mode.value:
            logger.info(
                "Overriding broad query mode from CLI: %s",
                self.broad_query_mode.value,
            )
            saved_snapshot.retrieval.broad_query_mode = self.broad_query_mode.value
        return saved_snapshot

    def run_update(self):
        self.initialize_vault()

        if self.metadata_manager.has_pending_pairs():
            logger.warning(
                "Found pending candidate pairs from a previous run. Resuming LLM classification..."
            )
            resumed_ok = False
            try:
                candidate_pairs = self.metadata_manager.pending_pairs
                if not candidate_pairs:
                    raise ValueError(
                        "Pending pairs exist, but could not be parsed into LLM inputs."
                    )

                self.metadata_manager.set_stage(
                    RunStage.RESUMING_LLM_CLASSIFICATION,
                    {"pairs": len(candidate_pairs)},
                )
                self.metadata_manager.save_run_state_only()

                self.llm_service.load()
                try:
                    links_to_write = self.llm_service.classify_pairs_with_checkpoints(
                        candidate_pairs,
                        stage=RunStage.RESUMING_LLM_CLASSIFICATION,
                        strategy=self.retrieval_strategy_name,
                    )
                    if links_to_write:
                        self.export_service.save_new_links(links_to_write, self.save_mode)
                finally:
                    self.llm_service.unload()
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
            files_to_add, files_to_update, files_to_remove = (
                self.vault_manager.determine_changes(self.metadata_manager)
            )

            self.vault_manager.process_removals(
                files_to_update + files_to_remove,
                self.metadata_manager,
                self.vector_store,
            )

            new_chunks_data = self.vault_manager.process_additions_and_updates(
                files_to_add + files_to_update, self.metadata_manager, self.vector_store
            )
            if not new_chunks_data:
                logger.info("No new files for linking.")
                return
            self.metadata_manager.set_stage(
                RunStage.COLLECTING_CANDIDATES, {"new_chunks": len(new_chunks_data)}
            )
            self.metadata_manager.save()

            candidate_pairs = self._collect_candidates(new_chunks_data)
            if self.metadata_manager.has_pending_pairs():
                self.metadata_manager.save()
            else:
                self.metadata_manager.set_stage(RunStage.NO_CANDIDATES)
                self.metadata_manager.save()

            if candidate_pairs:
                self.metadata_manager.set_stage(
                    RunStage.CLASSIFYING_LLM_PAIRS, {"pairs": len(candidate_pairs)}
                )
                self.metadata_manager.save_run_state_only()
                self.llm_service.load()
                try:
                    links_to_write = self.llm_service.classify_pairs_with_checkpoints(
                        candidate_pairs,
                        stage=RunStage.CLASSIFYING_LLM_PAIRS,
                        strategy=self.retrieval_strategy_name,
                    )
                    if links_to_write:
                        self.export_service.save_new_links(
                            links_to_write, self.save_mode
                        )
                except Exception as e:
                    self.metadata_manager.set_stage(
                        RunStage.FAILED,
                        {"during": "llm_classification", "error": str(e)},
                    )
                    self.metadata_manager.save_run_state_only()
                    raise
                finally:
                    self.llm_service.unload()
                self.metadata_manager.set_stage(RunStage.COMPLETED)
                self.metadata_manager.clear_run_state(keep_snapshot=True)
            else:
                logger.info("No candidates passed the reranker.")
        finally:
            self._save_state()
            logger.info("Updated finished.")

    def initialize_vault(self):
        logger.info(f"Initializing AI metadata for vault: {self.vault_path}")
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_manager.save()

    def _collect_candidates(
        self, new_chunks_data: List[NewlyAddedChunk]
    ) -> List[CandidatePair]:
        logger.info(f"Finding candidates for {len(new_chunks_data)} new chunks...")

        all_candidate_pairs: List[CandidatePair] = []

        all_initial_candidates_to_save: List[Dict[str, Any]] = []
        all_reranked_candidates_to_save: List[Dict[str, Any]] = []

        retrieval_k = self.runtime_config.retrieval.initial_retrieval_k
        reranker_top_k = self.runtime_config.models.reranker.top_k
        reranker_threshold = self.runtime_config.models.reranker.threshold

        global_entity_dict = None
        if self.retrieval_strategy_name in (
            RetrievalStrategyMode.STRICT,
            RetrievalStrategyMode.COMBINED,
        ):
            global_entity_dict = self.vault_manager.build_global_entity_dict()

        if self.retrieval_strategy_name == RetrievalStrategyMode.BROAD:
            retrieval_strategy = VectorSearchRerankStrategy(
                vector_store=self.vector_store,
                retrieval_k=retrieval_k,
                reranker_top_k=reranker_top_k,
                reranker_threshold=reranker_threshold,
                broad_query_mode=self.broad_query_mode,
            )
        elif self.retrieval_strategy_name == RetrievalStrategyMode.STRICT:
            retrieval_strategy = StrictRetrievalStrategy(
                vector_store=self.vector_store,
                global_entity_dict=global_entity_dict,
            )
        elif self.retrieval_strategy_name == RetrievalStrategyMode.COMBINED:
            broad_strat = VectorSearchRerankStrategy(
                vector_store=self.vector_store,
                retrieval_k=retrieval_k,
                reranker_top_k=reranker_top_k,
                reranker_threshold=reranker_threshold,
                broad_query_mode=self.broad_query_mode,
            )
            strict_strat = StrictRetrievalStrategy(
                vector_store=self.vector_store,
                global_entity_dict=global_entity_dict,
            )
            retrieval_strategy = CombinedRetrievalStrategy(strict_strat, broad_strat)
        else:
            raise ValueError(
                f"Unknown retrieval strategy: {self.retrieval_strategy_name}"
            )

        needs_full_docs = self.retrieval_strategy_name != RetrievalStrategyMode.BROAD or self.broad_query_mode == BroadQueryMode.TITLE_SUMMARY
        full_docs = {}
        if needs_full_docs:
            for chunk in new_chunks_data:
                if chunk.file_path not in full_docs:
                    abs_path = self.vault_path / chunk.file_path
                    full_docs[chunk.file_path] = self.vault_manager.get_file_content(
                        abs_path
                    )

        self.vector_store.load_reranker()
        try:
            final_candidates, initial_candidates_meta = retrieval_strategy.retrieve(
                chunks=new_chunks_data, full_docs=full_docs
            )
        finally:
            self.vector_store.unload_reranker()

        all_initial_candidates_to_save.extend(initial_candidates_meta)

        all_candidate_pairs.extend(final_candidates)
        for cand in final_candidates:
            all_reranked_candidates_to_save.append(
                {
                    "source_path": str(cand.source_path),
                    "source_content": cand.source_content,
                    "target_path": str(cand.target_path),
                    "target_content": cand.target_content,
                    "vector_distance": cand.vector_distance,
                    "reranker_score": cand.reranker_score,
                }
            )

        if all_initial_candidates_to_save:
            logger.info(
                f"Saving {len(all_initial_candidates_to_save)} initial candidates to {self.candidates_path}"
            )
            with open(self.candidates_path, "w", encoding="utf-8") as f:
                json.dump(
                    all_initial_candidates_to_save, f, ensure_ascii=False, indent=4
                )

        if all_reranked_candidates_to_save:
            logger.info(
                f"Saving {len(all_reranked_candidates_to_save)} reranked candidates to {self.reranked_candidates_path}"
            )
            with open(self.reranked_candidates_path, "w", encoding="utf-8") as f:
                json.dump(
                    all_reranked_candidates_to_save, f, ensure_ascii=False, indent=4
                )

        logger.info(f"Total candidates passed to LLM: {len(all_candidate_pairs)}")
        self.metadata_manager.set_pending_pairs(all_candidate_pairs)
        return all_candidate_pairs

    def _save_state(self):
        logger.info("Saving vault state...")
        self.metadata_manager.save()
        self.vector_store.save()

    def close(self):
        if self.llm_service and self.llm_service.llm:
            self.llm_service.close()
