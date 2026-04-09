import asyncio
import gc
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import json_repair
import torch
from langchain_cerebras import ChatCerebras
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from src.kg_builder.config import MAX_RETRIES
from src.kg_builder.metadata_manager import RunStage
from src.kg_builder.models import (
    CandidatePair,
    ContextSnippet,
    GroupedCandidatePair,
)

from .templates import (
    HUMAN_PROMPT_TEMPLATE_GROUPED,
    SYSTEM_PROMPT_TEMPLATE_GROUPED,
)

logger = logging.getLogger(__name__)

_LLM_TIMEOUT_SEC: int = 240
_BATCH_MAX_RETRIES: int = 3
_DEFAULT_TOP_N = 5
_DEFAULT_BATCH_SIZE = 5
_DEFAULT_CONCURRENT_REQUESTS = 5


class _RelationPrediction(BaseModel):
    id: int = Field(description="0-based index of the item within the current batch")
    reasoning: str = Field(description="One-sentence explanation")
    predicted_type: str = Field(
        description="Exactly one allowed relation type, or 'no link'"
    )


class _BatchRelationResponse(BaseModel):
    results: list[_RelationPrediction]


class LLMService:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        n_gpu_layers: int = -1,
        n_ctx: int = 8192,
        n_batch: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.1,
        use_api: bool = False,
        backend: str = "vulkan",
        concurrency: int = 1,
        default_link_types: Optional[List[str]] = None,
        vault_path: Optional[Path] = None,
        metadata_manager: Optional[Any] = None,
        reranked_candidates_path: Optional[Path] = None,
    ):
        self.backend = backend
        self.concurrency = concurrency
        self.default_link_types = default_link_types or []
        self.model_name = str(model_path) if model_path else ""
        self.temperature = temperature
        self.top_p = top_p
        self.use_api = use_api
        self.provider = None
        self.llm = None
        self.vault_path = vault_path
        self.metadata_manager = metadata_manager
        self.reranked_candidates_path = reranked_candidates_path

        if model_path is not None:
            self._init_model(model_path, n_gpu_layers, n_ctx, n_batch)

    def _init_model(
        self, model_path: Path, n_gpu_layers: int, n_ctx: int, n_batch: int
    ):
        if self.use_api:
            from dotenv import load_dotenv

            load_dotenv()
            self.provider = os.environ.get("LLM_PROVIDER", None).lower()
            model = os.environ.get("MODEL", "")
            logger.info(f"Initialising LLM: provider={self.provider!r}, model={model!r}")
            if self.provider == "openai":
                self.llm = self._init_openai()
            elif self.provider == "google":
                self.llm = self._init_google()
            elif self.provider == "cerebras":
                self.llm = self._init_cerebras()
            elif self.provider == "groq":
                self.llm = self._init_groq()
            else:
                raise ValueError("unsupported api provider")
        else:
            logger.info(f"Initialising local LLM: model_path={model_path}")
            self.llm = self._init_local(model_path, n_gpu_layers, n_ctx, n_batch)

        logger.info("LLM loaded successfully.")

    def _init_openai(self) -> ChatOpenAI:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1")
        model = os.environ.get("MODEL")

        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=self.temperature,
            model_kwargs={"top_p": self.top_p},
            streaming=False,
            max_retries=MAX_RETRIES,
            timeout=_LLM_TIMEOUT_SEC,
        )

        try:
            llm = llm.bind(response_format={"type": "json_object"})
        except Exception as e:
            logger.debug(f"Could not bind json_object response format: {e}")

        return llm

    def _init_google(self) -> ChatGoogleGenerativeAI:
        api_key = os.environ.get("GOOGLE_API_KEY")
        model = os.environ.get("MODEL")

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=self.temperature,
            top_p=self.top_p,
            max_retries=MAX_RETRIES,
        )

    def _init_cerebras(self) -> ChatCerebras:
        api_key = os.environ.get("CEREBRAS_API_KEY")
        model = os.environ.get("MODEL", "llama3.1-8b")

        llm = ChatCerebras(
            model=model,
            api_key=api_key,
            temperature=self.temperature,
            top_p=self.top_p,
            max_retries=MAX_RETRIES,
            timeout=_LLM_TIMEOUT_SEC,
        )

        try:
            llm = llm.bind(response_format={"type": "json_object"})
        except Exception as e:
            logger.debug(f"Could not bind json_object response format: {e}")

        return llm

    def _init_groq(self):
        from langchain_groq import ChatGroq

        api_key = os.environ.get("GROQ_API_KEY")
        model = os.environ.get("MODEL", "")
        return ChatGroq(
            model=model,
            api_key=api_key,
            temperature=self.temperature,
            max_retries=MAX_RETRIES,
            request_timeout=_LLM_TIMEOUT_SEC,
        )

    def _init_local(
        self, model_path: Path, n_gpu_layers: int, n_ctx: int, n_batch: int
    ) -> ChatLlamaCpp:
        return ChatLlamaCpp(
            model_path=str(model_path),
            temperature=self.temperature,
            top_p=self.top_p,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=n_batch,
            verbose=False,
            backend=self.backend,
            streaming=False,
        )

    def close(self):
        if hasattr(self.llm, "client") and hasattr(self.llm.client, "close"):
            self.llm.client.close()
        del self.llm
        gc.collect()
        if not self.use_api:
            torch.cuda.empty_cache()
        logger.info("LLM client resources released.")

    def _group_candidate_pairs(
        self,
        candidate_pairs: List[CandidatePair],
        top_n: int = _DEFAULT_TOP_N,
    ) -> List[GroupedCandidatePair]:
        """Group CandidatePairs by (source_path, target_path) and keep top-N snippets."""
        groups: Dict[Tuple[Path, Path], List[ContextSnippet]] = defaultdict(list)
        for pair in candidate_pairs:
            key = (pair.source_path, pair.target_path)
            groups[key].append(
                ContextSnippet(
                    source_content=pair.source_content,
                    target_content=pair.target_content,
                    reranker_score=pair.reranker_score,
                )
            )
        result: List[GroupedCandidatePair] = []
        for (source_path, target_path), snippets in groups.items():
            snippets.sort(key=lambda s: s.reranker_score, reverse=True)
            result.append(
                GroupedCandidatePair(
                    source_path=source_path,
                    target_path=target_path,
                    contexts=snippets[:top_n],
                )
            )
        return result

    def _format_grouped_batch_item(
        self,
        idx: int,
        source_path: Path,
        target_path: Path,
        contexts: List[ContextSnippet],
    ) -> str:
        source_clean = source_path.stem.replace("_", " ")
        target_clean = target_path.stem.replace("_", " ")
        snippets_formatted = "\n".join(
            f"    {i + 1}. [Source] {s.source_content.strip()}\n"
            f"       [Target] {s.target_content.strip()}"
            for i, s in enumerate(contexts)
        )
        return (
            f'[{idx}] Source: "{source_clean}"  →  Target: "{target_clean}"\n'
            f"  Context snippets:\n{snippets_formatted}"
        )

    def _build_grouped_messages(
        self,
        batch: List[GroupedCandidatePair],
        relation_types: List[str],
    ) -> list:
        relation_types_str = ", ".join(relation_types)
        system_content = SYSTEM_PROMPT_TEMPLATE_GROUPED.format(
            relation_types=relation_types_str
        )
        items_str = "\n\n".join(
            self._format_grouped_batch_item(i, g.source_path, g.target_path, g.contexts)
            for i, g in enumerate(batch)
        )
        human_content = HUMAN_PROMPT_TEMPLATE_GROUPED.format(
            count=len(batch), items=items_str
        )

        if self.provider == "google":
            return [HumanMessage(content=system_content), HumanMessage(content=human_content)]
        return [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content),
        ]

    def _extract_raw_text(self, response: Any) -> str:
        content = response.content if hasattr(response, "content") else response
        if isinstance(content, list):
            return "".join(
                c.get("text") or c.get("content") or str(c)
                if isinstance(c, dict)
                else (c.text if hasattr(c, "text") else str(c))
                for c in content
            )
        return str(content)

    def _parse_grouped_response(
        self, raw_text: str, n_items: int
    ) -> List[Optional[dict]]:
        if not raw_text or not raw_text.strip():
            logger.warning("LLM returned an empty response")
            return [None] * n_items
        parsed = json_repair.loads(raw_text)
        if not isinstance(parsed, (dict, list)):
            logger.warning(f"LLM returned unparseable response: {raw_text[:100]!r}")
            return [None] * n_items
        if isinstance(parsed, list):
            result_dict = None
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "thinking":
                    continue
                if "results" in item:
                    result_dict = item
                    break
            if result_dict is None:
                result_dict = {
                    "results": [
                        i
                        for i in parsed
                        if isinstance(i, dict) and i.get("type") != "thinking"
                    ]
                }
            parsed = result_dict
        batch_response = _BatchRelationResponse(**parsed)
        out: List[Optional[dict]] = [None] * n_items
        for pred in batch_response.results:
            if 0 <= pred.id < n_items:
                out[pred.id] = {
                    "predicted_type": pred.predicted_type.lower(),
                    "reasoning": pred.reasoning,
                }
        return out

    async def _ainvoke_grouped_batch(
        self,
        batch: List[GroupedCandidatePair],
        relation_types: List[str],
        semaphore: asyncio.Semaphore,
    ) -> Tuple[List[GroupedCandidatePair], List[Optional[dict]]]:
        async with semaphore:
            messages = self._build_grouped_messages(batch, relation_types)
            response = await self.llm.ainvoke(messages)
            return batch, self._parse_grouped_response(
                self._extract_raw_text(response), len(batch)
            )

    async def _run_async_grouped_classification(
        self,
        to_process: List[GroupedCandidatePair],
        batch_size: int,
        concurrent_requests: int,
        relation_types: List[str],
        completed_predictions: Dict[str, Any],
        total: int,
    ) -> None:
        batches = [
            to_process[s : s + batch_size]
            for s in range(0, len(to_process), batch_size)
        ]
        semaphore = asyncio.Semaphore(concurrent_requests)
        write_lock = asyncio.Lock()

        async def process_batch(batch: List[GroupedCandidatePair]) -> None:
            items, preds = batch, [None] * len(batch)
            for attempt in range(1, _BATCH_MAX_RETRIES + 1):
                try:
                    items, preds = await self._ainvoke_grouped_batch(
                        batch, relation_types, semaphore
                    )
                    break
                except Exception as e:
                    if attempt < _BATCH_MAX_RETRIES:
                        logger.warning(
                            f"Batch attempt {attempt}/{_BATCH_MAX_RETRIES} failed "
                            f"(src={batch[0].source_path.stem}…): {e} — retrying"
                        )
                        await asyncio.sleep(2**attempt)
                    else:
                        logger.error(
                            f"Batch failed after {_BATCH_MAX_RETRIES} attempts "
                            f"(src={batch[0].source_path.stem}…): {e}"
                        )

            async with write_lock:
                for item, pred in zip(items, preds):
                    if pred is None:
                        logger.warning(
                            f"No prediction for pair (src={item.source_path}, "
                            f"tgt={item.target_path}) — will retry on restart"
                        )
                        continue
                    str_key = f"{item.source_path}||{item.target_path}"
                    completed_predictions[str_key] = [
                        {
                            "relation_type": pred["predicted_type"],
                            "reasoning": pred["reasoning"],
                        }
                    ]
                # Checkpoint after every batch
                self.metadata_manager.save_partial_predictions(completed_predictions)
                done = len(completed_predictions)
                self.metadata_manager.set_llm_progress(offset=done, total=total)
                self.metadata_manager.save_run_state_only()

        tasks = [asyncio.create_task(process_batch(b)) for b in batches]
        with tqdm(total=len(batches), desc="Classifying Links", unit="batch") as pbar:
            for coro in asyncio.as_completed(tasks):
                await coro
                pbar.update(1)

    def _build_links_from_predictions(
        self,
        completed_predictions: Dict[str, Any],
    ) -> Dict[Path, Set[str]]:
        links_to_write: Dict[Path, Set[str]] = defaultdict(set)
        for str_key, pred_list in completed_predictions.items():
            if not pred_list or "||" not in str_key:
                continue
            path_a_str, path_b_str = str_key.split("||", 1)
            pred = pred_list[0]
            relation = pred.get("relation_type", "").lower()
            if not relation or relation == "no link":
                continue
            link_str = self._build_link_string(relation, Path(path_b_str))
            links_to_write[Path(path_a_str)].add(link_str)
        return dict(links_to_write)

    def _update_reranked_candidates_from_predictions(
        self,
        completed_predictions: Dict[str, Any],
    ) -> None:
        if (
            not self.reranked_candidates_path
            or not self.reranked_candidates_path.exists()
        ):
            return

        logger.info(f"Updating {self.reranked_candidates_path} with LLM results...")
        try:
            with open(self.reranked_candidates_path, "r", encoding="utf-8") as f:
                reranked_candidates = json.load(f)

            lookup: Dict[Tuple[str, str], dict] = {}
            for str_key, pred_list in completed_predictions.items():
                if "||" not in str_key or not pred_list:
                    continue
                path_a, path_b = str_key.split("||", 1)
                lookup[(path_a, path_b)] = pred_list[0]

            updated_count = 0
            for candidate in reranked_candidates:
                key = (candidate["source_path"], candidate["target_path"])
                if key in lookup:
                    pred = lookup[key]
                    candidate["llm_type"] = pred.get("relation_type", "no link")
                    candidate["llm_reasoning"] = pred.get("reasoning", "")
                    updated_count += 1
                else:
                    candidate["llm_type"] = "no link"
                    candidate["llm_reasoning"] = ""

            with open(self.reranked_candidates_path, "w", encoding="utf-8") as f:
                json.dump(reranked_candidates, f, ensure_ascii=False, indent=4)

            logger.info(
                f"Updated {updated_count} entries in {self.reranked_candidates_path} "
                "with LLM results."
            )
        except Exception as e:
            logger.warning(
                f"Failed to update reranked_candidates.json with LLM results: {e}"
            )

    def _build_link_string(self, relation: str, target_rel_path: Path) -> str:
        target_file_name = target_rel_path.stem
        return self.metadata_manager.config.link_template.format(
            relation_type=relation,
            target_file_name=target_file_name,
        )

    def load(self):
        if self.llm is None and self.metadata_manager:
            logger.info("Loading LLM Service...")
            llm_cfg = self.metadata_manager.config.models.llm
            self.use_api = llm_cfg.use_api
            self.backend = llm_cfg.backend
            self.temperature = llm_cfg.temperature
            self.top_p = llm_cfg.top_p
            self.concurrency = llm_cfg.concurrency
            self.model_name = llm_cfg.model_path

            self._init_model(
                Path(llm_cfg.model_path),
                llm_cfg.n_gpu_layers,
                llm_cfg.n_ctx,
                llm_cfg.n_batch,
            )

    def unload(self):
        if self.llm:
            logger.info("Unloading LLM Service...")
            self.close()
            self.llm = None
            self._context_link_chain = None

    def classify_pairs_with_checkpoints(
        self,
        candidate_pairs: List[CandidatePair],
        stage: RunStage,
        strategy: str = "broad",
    ) -> Dict[Path, Set[str]]:
        if not candidate_pairs:
            return {}

        # Group all candidate pairs by (source_path, target_path) and rank contexts
        grouped = self._group_candidate_pairs(candidate_pairs, top_n=_DEFAULT_TOP_N)
        total = len(grouped)

        # Load checkpoint: keys already processed from a previous (interrupted) run
        completed_predictions = self.metadata_manager.load_partial_predictions()

        to_process = [
            g
            for g in grouped
            if f"{g.source_path}||{g.target_path}" not in completed_predictions
        ]

        if to_process:
            logger.info(
                f"Grouped pairs: {total}, "
                f"already done: {total - len(to_process)}, "
                f"remaining: {len(to_process)}"
            )

            concurrent_requests = max(1, int(self.concurrency)) if self.use_api else 1
            batch_size = _DEFAULT_BATCH_SIZE
            relation_types = self.metadata_manager.config.llm_link_types

            self.metadata_manager.set_llm_progress(
                offset=total - len(to_process), total=total
            )
            self.metadata_manager.set_stage(
                stage,
                {
                    "offset": total - len(to_process),
                    "total": total,
                    "batch_size": batch_size,
                    "llm_concurrency": concurrent_requests,
                },
            )
            self.metadata_manager.save_run_state_only()

            try:
                asyncio.run(
                    self._run_async_grouped_classification(
                        to_process=to_process,
                        batch_size=batch_size,
                        concurrent_requests=concurrent_requests,
                        relation_types=relation_types,
                        completed_predictions=completed_predictions,
                        total=total,
                    )
                )
            except KeyboardInterrupt:
                tqdm.write(
                    "\nInterrupted — saving progress and computing partial results."
                )
        else:
            logger.info(
                "All grouped pairs already processed — building links from checkpoint."
            )

        links_to_write = self._build_links_from_predictions(completed_predictions)
        self._update_reranked_candidates_from_predictions(completed_predictions)
        return links_to_write
