import gc
import json
import logging
import os
import re
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import torch
from langchain_cerebras import ChatCerebras
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
)
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from src.kg_builder.config import MAX_RETRIES
from src.kg_builder.metadata_manager import RunStage
from src.kg_builder.models import CandidatePair, LinkConflict, LinkPrediction

from .templates import (
    HUMAN_PROMPT_TEMPLATE_BROAD,
    HUMAN_PROMPT_TEMPLATE_FOR_CONTEXT_LINKING,
    HUMAN_PROMPT_TEMPLATE_FOR_LINK_CONFLICT_RESOLUTION,
    HUMAN_PROMPT_TEMPLATE_STRICT,
    SYSTEM_PROMPT_TEMPLATE_BROAD,
    SYSTEM_PROMPT_TEMPLATE_FOR_CONTEXT_LINKING,
    SYSTEM_PROMPT_TEMPLATE_FOR_LINK_CONFLICT_RESOLUTION,
    SYSTEM_PROMPT_TEMPLATE_STRICT,
)

logger = logging.getLogger(__name__)


class Link(BaseModel):
    reasoning: str = Field(description="A brief explanation for the chosen link.")
    relation_type: str = Field(description="The single most appropriate link type.")


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
        self._link_chain = None
        self._link_strict_chain = None
        self._link_conflict_chain = None
        self._context_link_chain = None

        self.vault_path = vault_path
        self.metadata_manager = metadata_manager
        self.reranked_candidates_path = reranked_candidates_path

        if model_path is not None:
            self._init_model(model_path, n_gpu_layers, n_ctx, n_batch)

    def _init_model(
        self, model_path: Path, n_gpu_layers: int, n_ctx: int, n_batch: int
    ):
        logger.info(f"Loading model (API: {self.use_api})...")
        if self.use_api:
            from dotenv import load_dotenv

            load_dotenv()
            self.provider = os.environ.get("LLM_PROVIDER", None).lower()
            if self.provider == "openai":
                self.llm = self._init_openai()
            elif self.provider == "google":
                self.llm = self._init_google()
            elif self.provider == "cerebras":
                self.llm = self._init_cerebras()
            else:
                raise ValueError("unsupported api provider")
        else:
            self.llm = self._init_local(model_path, n_gpu_layers, n_ctx, n_batch)

        logger.info("Model loaded successfully.")
        self._link_chain = self._create_link_chain(
            SYSTEM_PROMPT_TEMPLATE_BROAD, HUMAN_PROMPT_TEMPLATE_BROAD
        )
        self._link_strict_chain = self._create_link_chain(
            SYSTEM_PROMPT_TEMPLATE_STRICT, HUMAN_PROMPT_TEMPLATE_STRICT
        )
        self._link_conflict_chain = self._create_link_conflict_chain()
        self._context_link_chain = self._create_context_link_chain()

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
        )

        try:
            llm = llm.bind(response_format={"type": "json_object"})
        except Exception as e:
            logger.debug(f"Could not bind json_object response format: {e}")

        return llm

    def _init_google(self) -> ChatGoogleGenerativeAI:
        api_key = os.environ.get("GOOGLE_API_KEY")
        model = os.environ.get("MODEL")

        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=self.temperature,
            top_p=self.top_p,
            max_retries=MAX_RETRIES,
        )

        return llm

    def _init_cerebras(self) -> ChatCerebras:
        api_key = os.environ.get("CEREBRAS_API_KEY")
        model = os.environ.get("MODEL", "llama3.1-8b")

        llm = ChatCerebras(
            model=model,
            api_key=api_key,
            temperature=self.temperature,
            top_p=self.top_p,
            max_retries=MAX_RETRIES,
        )

        try:
            llm = llm.bind(response_format={"type": "json_object"})
        except Exception as e:
            logger.debug(f"Could not bind json_object response format: {e}")

        return llm

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

    def _create_structured_chain(
        self,
        system_prompt: str,
        human_prompt: str,
        schema: Type[BaseModel],
        partial_variables: Optional[Dict[str, Any]] = None,
    ) -> Any:
        parser = PydanticOutputParser(pydantic_object=schema)

        if self.provider == "google":
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("human", system_prompt + "\n\n" + human_prompt),
                ]
            )
        else:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", human_prompt),
                ]
            )
        prompt = prompt.partial(
            format_instructions=parser.get_format_instructions(),
            **partial_variables if partial_variables is not None else {},
        )

        if self.provider != "google":
            try:
                if hasattr(self.llm, "with_structured_output"):
                    structured_llm = self.llm.with_structured_output(schema)
                    return prompt | structured_llm
            except (NotImplementedError, AttributeError):
                pass

        return prompt | self.llm | parser

    def _create_link_chain(
        self,
        system_prompt: str = SYSTEM_PROMPT_TEMPLATE_BROAD,
        human_prompt: str = HUMAN_PROMPT_TEMPLATE_BROAD,
    ):
        return self._create_structured_chain(
            system_prompt,
            human_prompt,
            schema=Link,
        )

    def _create_link_conflict_chain(
        self,
        system_prompt: str = SYSTEM_PROMPT_TEMPLATE_FOR_LINK_CONFLICT_RESOLUTION,
        human_prompt: str = HUMAN_PROMPT_TEMPLATE_FOR_LINK_CONFLICT_RESOLUTION,
    ):
        return self._create_structured_chain(
            system_prompt,
            human_prompt,
            schema=Link,
        )

    def batch_classify_link(
        self,
        candidate_pairs: List[CandidatePair],
        relation_types: List[str],
        strategy: str = "broad",
        pbar: Optional[Any] = None,
        show_progress: bool = True,
    ) -> List[Optional[Dict[str, str]]]:
        if not candidate_pairs:
            return []

        relation_types_str = ", ".join(relation_types)
        relation_types_normalized = [t.lower() for t in relation_types]
        texts = []
        for pair in candidate_pairs:
            if strategy == "strict":
                raw_summary = self._extract_document_summary(pair.target_path)
                target_summary = self._sanitize_text_input(raw_summary)
                texts.append(
                    {
                        "target_name": pair.target_path.stem,
                        "target_summary": target_summary,
                        "source_filename": pair.source_path.stem,
                        "extracted_context": self._sanitize_text_input(
                            pair.target_content
                        ),
                        "relation_types": relation_types_str,
                    }
                )
            else:
                texts.append(
                    {
                        "text_a": self._sanitize_text_input(pair.source_content),
                        "text_b": self._sanitize_text_input(pair.target_content),
                        "filename_a": pair.source_path.stem,
                        "filename_b": pair.target_path.stem,
                        "relation_types": relation_types_str,
                    }
                )
        logger.debug(f"Classifying links for a batch of {len(texts)} items.")

        results = []
        concurrency = 1 if not self.use_api else self.concurrency
        config = RunnableConfig(max_concurrency=concurrency)

        created_pbar = False
        if pbar is None and show_progress:
            pbar = tqdm(
                total=len(texts),
                desc="Classifying Semantic Links",
                unit="pair",
                leave=False,
            )
            created_pbar = True

        try:
            for i in range(0, len(texts), concurrency):
                mini_batch = texts[i : i + concurrency]
                try:
                    chain = (
                        self._link_strict_chain
                        if strategy == "strict"
                        else self._link_chain
                    )
                    responses = chain.batch(
                        mini_batch, config=config, return_exceptions=True
                    )
                except Exception as e:
                    logger.error(f"Critical error in batch {i // concurrency}: {e}")
                    responses = [None] * len(mini_batch)

                for j, response in enumerate(responses):
                    if isinstance(response, Exception):
                        logger.debug(
                            f"Item failed during LLM execution in batch_classify_link: {response}\nTraceback:\n"
                            f"{''.join(traceback.format_exception(type(response), response, response.__traceback__))}"
                        )
                        results.append(None)
                        continue

                    try:
                        link_type = response.relation_type.lower()
                        sanitized_link = self._sanitize_model_response(link_type)

                        src_filename = mini_batch[j].get(
                            "filename_a",
                            mini_batch[j].get("source_filename", "Unknown"),
                        )
                        tgt_filename = mini_batch[j].get(
                            "filename_b", mini_batch[j].get("target_name", "Unknown")
                        )

                        logger.info(
                            f"LLM classified link between '{src_filename}' and "
                            f"'{tgt_filename}' as '{sanitized_link}'"
                        )
                        logger.info(f"    Raw response: {response}")

                        if sanitized_link == "no link":
                            results.append(None)
                        elif sanitized_link in relation_types_normalized:
                            results.append(
                                {
                                    "type": sanitized_link,
                                    "reasoning": response.reasoning,
                                }
                            )
                        else:
                            logger.warning(
                                f"Invalid relation type returned: '{sanitized_link}'. "
                                f"Expected one of: {relation_types_normalized} or 'no link'"
                            )
                            results.append(None)

                    except Exception as e:
                        logger.error(f"Failed to parse LLM response: {e}")
                        results.append(None)

                if pbar is not None:
                    pbar.update(len(mini_batch))
        finally:
            if created_pbar and pbar is not None:
                pbar.close()

        return results

    def _create_context_link_chain(
        self,
        system_prompt: str = SYSTEM_PROMPT_TEMPLATE_FOR_CONTEXT_LINKING,
        human_prompt: str = HUMAN_PROMPT_TEMPLATE_FOR_CONTEXT_LINKING,
    ):
        return self._create_structured_chain(
            system_prompt,
            human_prompt,
            schema=Link,
        )

    def batch_classify_context_link(
        self,
        contexts_data: List[Dict[str, Any]],
        relation_types: List[str],
        max_concurrency: Optional[int] = None,
    ) -> List[Optional[str]]:
        if not contexts_data:
            return []

        relation_types_str = ", ".join(relation_types)
        relation_types_normalized = [t.lower() for t in relation_types]
        formatted_inputs = []
        for item in contexts_data:
            raw_contexts = item.get("contexts", [])
            if not raw_contexts:
                formatted_contexts = "No context provided."
            else:
                formatted_contexts = "\n".join(
                    [
                        f"- {self._sanitize_text_input(c).replace('\n', ' ')}"
                        for c in raw_contexts
                    ]
                )

            formatted_inputs.append(
                {
                    "source_title": self._sanitize_text_input(item["source"]),
                    "target_title": self._sanitize_text_input(item["target"]),
                    "contexts": formatted_contexts,
                    "relation_types": relation_types_str,
                }
            )

        logger.debug(
            f"Classifying context links for a batch of {len(contexts_data)} items."
        )

        effective_concurrency = (
            1 if not self.use_api else (max_concurrency or self.concurrency)
        )
        config = RunnableConfig(max_concurrency=effective_concurrency)

        results = []
        with tqdm(
            total=len(formatted_inputs),
            desc="Classifying Context Links",
            unit="link",
            leave=False,
        ) as pbar:
            for i in range(0, len(formatted_inputs), effective_concurrency):
                mini_batch = formatted_inputs[i : i + effective_concurrency]
                try:
                    responses = self._context_link_chain.batch(
                        mini_batch, config=config, return_exceptions=True
                    )
                except Exception as e:
                    logger.error(
                        f"Critical error in batch {i // effective_concurrency}: {e}"
                    )
                    responses = [None] * len(mini_batch)

                for j, response in enumerate(responses):
                    if isinstance(response, Exception):
                        logger.debug(
                            f"Item failed during LLM execution: {response}\n"
                            f"{''.join(traceback.format_exception(type(response), response, response.__traceback__))}"
                        )
                        results.append(None)
                        continue

                    try:
                        link_type = response.relation_type.lower()
                        sanitized_link = self._sanitize_model_response(link_type)

                        if sanitized_link == "no link":
                            results.append(None)
                        elif sanitized_link in relation_types_normalized:
                            results.append(sanitized_link)
                        else:
                            logger.warning(
                                f"Invalid context relation type returned: '{sanitized_link}'. "
                                f"Expected one of: {relation_types_normalized} or 'no link'"
                            )
                            results.append(None)
                    except Exception as e:
                        logger.warning(f"Failed to parse LLM response: {e}")
                        results.append(None)

                pbar.update(len(mini_batch))

        return results

    def _sanitize_model_response(self, text: str) -> str:
        text = re.sub(r"[^a-zA-Zа-яА-ЯёЁ ]", "", text)
        return re.sub(r"\s+", " ", text).strip().lower()

    def _sanitize_text_input(self, text: str) -> str:
        return re.sub(r"[^\w\s.,;:()\-\"\'\[\]@#=+/&]", "", text)

    def _extract_document_summary(self, target_path: Path) -> str:
        if not self.vault_path:
            return ""

        abs_path = (
            target_path if target_path.is_absolute() else self.vault_path / target_path
        )
        if not abs_path.exists():
            return ""

        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()
            content = re.sub(
                r"^---\r?\n.*?\r?\n---\r?\n", "", content, flags=re.DOTALL
            ).lstrip()
            summary = content.split("\n\n")[0].strip()
            return summary
        except Exception as e:
            logger.warning(f"Failed to extract summary from {abs_path}: {e}")
            return ""

    def batch_resolve_link_conflicts(
        self,
        conflicts: List[LinkConflict],
        relation_types: List[str],
        pbar: Optional[Any] = None,
        show_progress: bool = True,
    ) -> List[Optional[str]]:
        if not conflicts:
            return []

        relation_types_str = ", ".join(relation_types)
        relation_types_normalized = [t.lower() for t in relation_types]

        prepared_conflicts = []
        for conflict in conflicts:
            candidate_predictions = self._format_candidate_predictions(
                conflict.candidate_counts
            )
            evidence = self._format_conflict_evidence(conflict.evidence)
            prepared_conflicts.append(
                {
                    "filename_a": self._sanitize_text_input(conflict.filename_a),
                    "filename_b": self._sanitize_text_input(conflict.filename_b),
                    "relation_types": relation_types_str,
                    "candidate_predictions": candidate_predictions,
                    "evidence": evidence,
                }
            )

        results = []
        concurrency = 1 if not self.use_api else self.concurrency
        config = RunnableConfig(max_concurrency=concurrency)

        created_pbar = False
        if pbar is None and show_progress:
            pbar = tqdm(
                total=len(prepared_conflicts),
                desc="Resolving Link Type Conflicts",
                unit="pair",
                leave=False,
            )
            created_pbar = True

        try:
            for i in range(0, len(prepared_conflicts), concurrency):
                mini_batch = prepared_conflicts[i : i + concurrency]
                try:
                    responses = self._link_conflict_chain.batch(
                        mini_batch, config=config, return_exceptions=True
                    )
                except Exception as e:
                    logger.error(
                        f"Critical error in conflict-resolution batch {i // concurrency}: {e}"
                    )
                    responses = [None] * len(mini_batch)

                for response in responses:
                    if isinstance(response, Exception):
                        logger.debug(
                            "Item failed during conflict-resolution LLM execution: "
                            f"{response}\nTraceback:\n"
                            f"{''.join(traceback.format_exception(type(response), response, response.__traceback__))}"
                        )
                        results.append(None)
                        continue

                    try:
                        link_type = response.relation_type.lower()
                        sanitized_link = self._sanitize_model_response(link_type)
                        if sanitized_link == "no link":
                            results.append(None)
                        elif sanitized_link in relation_types_normalized:
                            results.append(sanitized_link)
                        else:
                            logger.warning(
                                f"Invalid resolved relation type: '{sanitized_link}'. "
                                f"Expected one of: {relation_types_normalized} or 'no link'"
                            )
                            results.append(None)
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse conflict-resolution LLM response: {e}"
                        )
                        results.append(None)

                if pbar is not None:
                    pbar.update(len(mini_batch))
        finally:
            if created_pbar and pbar is not None:
                pbar.close()

        return results

    def _format_candidate_predictions(self, candidate_counts: Dict[str, int]) -> str:
        if not candidate_counts:
            return "- No valid candidates."

        lines = []
        for relation, count in sorted(
            candidate_counts.items(), key=lambda item: (-item[1], item[0])
        ):
            lines.append(f"- {relation}: {count}")
        return "\n".join(lines)

    def _format_conflict_evidence(self, evidence_rows: List[LinkPrediction]) -> str:
        if not evidence_rows:
            return "- No evidence rows provided."

        rows = []
        MAX_EVIDENCE_ROWS = 12  # limit to keep conflict resolution prompt within context window
        for idx, row in enumerate(evidence_rows[:MAX_EVIDENCE_ROWS], start=1):
            relation = row.relation_type
            source_chunk = self._sanitize_text_input(row.text_a)[:400]
            target_chunk = self._sanitize_text_input(row.text_b)[:400]

            reranker_score = row.reranker_score
            vector_distance = row.vector_distance

            rows.append(
                (
                    f"- Evidence #{idx}\n"
                    f"  predicted_relation: {relation}\n"
                    f"  reranker_score: {reranker_score}\n"
                    f"  vector_distance: {vector_distance}\n"
                    f'  source_chunk: "{source_chunk}"\n'
                    f'  target_chunk: "{target_chunk}"'
                )
            )

        return "\n".join(rows)

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
            self._link_chain = None
            self._link_strict_chain = None
            self._link_conflict_chain = None
            self._context_link_chain = None

    def classify_pairs_with_checkpoints(
        self,
        candidate_pairs: List[CandidatePair],
        stage: RunStage,
        strategy: str = "broad",
    ) -> Dict[Path, Set[str]]:
        total = len(candidate_pairs)
        if total == 0:
            return {}

        offset, saved_total = self.metadata_manager.get_llm_progress()
        if saved_total != total:
            offset = min(offset, total)
            self.metadata_manager.set_llm_progress(offset=offset, total=total)
            self.metadata_manager.save_run_state_only()

        pair_predictions_accum = self.metadata_manager.load_partial_predictions()

        llm_concurrency = self.concurrency if self.use_api else 1
        llm_concurrency = max(1, int(llm_concurrency))
        batch_size = 10 * llm_concurrency

        with tqdm(
            total=total,
            initial=offset,
            desc="Classifying Semantic Links",
            unit="pair",
            leave=False,
        ) as pbar:
            for start in range(offset, total, batch_size):
                end = min(start + batch_size, total)
                batch_pairs = candidate_pairs[start:end]

                relation_results = self.batch_classify_link(
                    batch_pairs,
                    relation_types=self.metadata_manager.config.llm_link_types,
                    strategy=strategy,
                    pbar=pbar,
                    show_progress=False,
                )

                batch_pair_predictions = self._collect_pair_predictions_from_batch(
                    candidate_pairs=batch_pairs,
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

        self._update_reranked_candidates_with_llm_results(pair_predictions_accum)

        return links_to_write

    def _collect_pair_predictions_from_batch(
        self,
        candidate_pairs: List[CandidatePair],
        relation_results: List[Optional[Dict[str, str]]],
    ) -> Dict[Tuple[str, str], List[LinkPrediction]]:
        pair_predictions: Dict[Tuple[str, str], List[LinkPrediction]] = defaultdict(
            list
        )
        for relation_data, pair in zip(relation_results, candidate_pairs):
            if not relation_data:
                continue

            pair_key = (str(pair.source_path), str(pair.target_path))
            pair_predictions[pair_key].append(
                LinkPrediction(
                    relation_type=relation_data["type"],
                    reasoning=relation_data.get("reasoning", ""),
                    text_a=pair.source_content,
                    text_b=pair.target_content,
                    reranker_score=pair.reranker_score,
                    vector_distance=pair.vector_distance,
                )
            )
        return dict(pair_predictions)

    def _merge_pair_predictions(
        self,
        base: Dict[str, List[Dict[str, Any]]],
        incoming: Dict[Tuple[str, str], List[LinkPrediction]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        for pair_key, rows in incoming.items():
            if not rows:
                continue
            str_key = f"{pair_key[0]}||{pair_key[1]}"

            base.setdefault(str_key, []).extend([r.model_dump() for r in rows])
        return base

    def _resolve_pair_predictions(
        self,
        pair_predictions: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[Path, Set[str]]:
        normalized_predictions: Dict[Tuple[str, str], List[LinkPrediction]] = (
            defaultdict(list)
        )
        for pair_key, rows in pair_predictions.items():
            if "||" not in pair_key:
                continue
            path_a_str, path_b_str = pair_key.split("||", 1)
            for r in rows:
                normalized_predictions[(path_a_str, path_b_str)].append(
                    LinkPrediction(**r)
                )

        final_relations: Dict[Tuple[str, str], str] = {}
        conflict_inputs: List[LinkConflict] = []
        conflict_keys: List[Tuple[str, str]] = []
        for pair_key, evidence_rows in normalized_predictions.items():
            candidate_counts: Dict[str, int] = defaultdict(int)
            for row in evidence_rows:
                candidate_counts[row.relation_type] += 1

            if len(candidate_counts) == 1:
                final_relations[pair_key] = next(iter(candidate_counts.keys()))
                continue

            path_a_str, path_b_str = pair_key

            conflict_inputs.append(
                LinkConflict(
                    filename_a=Path(path_a_str).stem,
                    filename_b=Path(path_b_str).stem,
                    candidate_counts=dict(candidate_counts),
                    evidence=evidence_rows,
                )
            )
            conflict_keys.append(pair_key)

        if conflict_inputs:
            logger.info(
                f"Resolving {len(conflict_inputs)} link-type conflicts via LLM..."
            )
            resolved_relations = self.batch_resolve_link_conflicts(
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
        target_file_name = target_rel_path.stem
        relation_text = relation
        return self.metadata_manager.config.link_template.format(
            relation_type=relation_text,
            target_file_name=target_file_name,
        )

    def _update_reranked_candidates_with_llm_results(
        self, pair_predictions_accum: Dict[str, List[Dict[str, Any]]]
    ):
        if (
            not self.reranked_candidates_path
            or not self.reranked_candidates_path.exists()
        ):
            return

        logger.info(f"Updating {self.reranked_candidates_path} with LLM results...")
        try:
            with open(self.reranked_candidates_path, "r", encoding="utf-8") as f:
                reranked_candidates = json.load(f)

            lookup = {}
            for pair_key, predictions in pair_predictions_accum.items():
                if "||" not in pair_key:
                    continue
                path_a, path_b = pair_key.split("||", 1)
                for pred in predictions:
                    lookup[(path_a, path_b, pred["text_a"], pred["text_b"])] = pred

            updated_count = 0
            for candidate in reranked_candidates:
                key = (
                    candidate["source_path"],
                    candidate["target_path"],
                    candidate["source_content"],
                    candidate["target_content"],
                )
                if key in lookup:
                    pred = lookup[key]
                    candidate["llm_type"] = pred["relation_type"]
                    candidate["llm_reasoning"] = pred["reasoning"]
                    updated_count += 1
                else:
                    candidate["llm_type"] = "no link"
                    candidate["llm_reasoning"] = ""

            with open(self.reranked_candidates_path, "w", encoding="utf-8") as f:
                json.dump(reranked_candidates, f, ensure_ascii=False, indent=4)

            logger.info(
                f"Updated {updated_count} entries in {self.reranked_candidates_path} with LLM results."
            )
        except Exception as e:
            logger.warning(
                f"Failed to update reranked_candidates.json with LLM results: {e}"
            )
