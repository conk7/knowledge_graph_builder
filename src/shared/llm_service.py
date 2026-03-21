import gc
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import torch
from dotenv import load_dotenv
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

from .templates import (
    HUMAN_PROMPT_TEMPLATE_FOR_CONTEXT_LINKING,
    HUMAN_PROMPT_TEMPLATE_FOR_LINK_CONFLICT_RESOLUTION,
    HUMAN_PROMPT_TEMPLATE_FOR_LINKING,
    SYSTEM_PROMPT_TEMPLATE_FOR_CONTEXT_LINKING,
    SYSTEM_PROMPT_TEMPLATE_FOR_LINK_CONFLICT_RESOLUTION,
    SYSTEM_PROMPT_TEMPLATE_FOR_LINKING,
)

logger = logging.getLogger(__name__)


class Link(BaseModel):
    reasoning: str = Field(description="A brief explanation for the chosen link.")
    relation_type: str = Field(description="The single most appropriate link type.")


class LLMService:
    def __init__(
        self,
        model_path: Path,
        n_gpu_layers: int = -1,
        n_ctx: int = 8192,
        n_batch: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.1,
        use_api: bool = False,
        backend: str = "vulkan",
        concurrency: int = 1,
        default_link_types: Optional[List[str]] = None,
    ):
        self.backend = backend
        self.concurrency = concurrency
        self.default_link_types = default_link_types or []
        self.model_name = str(model_path)
        self.temperature = temperature
        self.top_p = top_p
        self.use_api = use_api
        self.provider = None

        logger.info(f"Loading model (API: {use_api})...")
        if use_api:
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

        self._link_chain = self._create_link_chain()
        self._link_conflict_chain = self._create_link_conflict_chain()

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
        system_prompt: str = SYSTEM_PROMPT_TEMPLATE_FOR_LINKING,
        human_prompt: str = HUMAN_PROMPT_TEMPLATE_FOR_LINKING,
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
        texts: List[Dict[str, str]],
        text_meta: List[Dict[str, Path]],
        relation_types: List[str],
        pbar: Optional[Any] = None,
        show_progress: bool = True,
    ) -> List[Optional[str]]:
        if not texts:
            return []

        relation_types_str = ", ".join(relation_types)
        relation_types = [t.lower() for t in relation_types]
        texts = [
            {
                "text_a": self._sanitize_text_input(text["text_a"]),
                "text_b": self._sanitize_text_input(text["text_b"]),
                "filename_a": meta["path_a"].stem,
                "filename_b": meta["path_b"].stem,
                "relation_types": relation_types_str,
            }
            for text, meta in zip(texts, text_meta)
        ]
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
                    responses = self._link_chain.batch(
                        mini_batch, config=config, return_exceptions=True
                    )
                except Exception as e:
                    logger.error(f"Critical error in batch {i // concurrency}: {e}")
                    responses = [None] * len(mini_batch)

                import traceback

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

                        logger.info(
                            f"LLM classified link between '{mini_batch[j]['filename_a']}' and "
                            f"'{mini_batch[j]['filename_b']}' as '{sanitized_link}'"
                        )
                        logger.info(f"    Raw response: {response}")

                        if sanitized_link == "no link":
                            results.append(None)
                        elif sanitized_link in relation_types:
                            results.append(sanitized_link)
                        else:
                            logger.warning(
                                f"Invalid relation type returned: '{sanitized_link}'. "
                                f"Expected one of: {relation_types} or 'no link'"
                            )
                            results.append(None)

                    except Exception as e:
                        logger.warning(f"Failed to parse LLM response: {e}")
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

        max_concurrency = max_concurrency or self.concurrency
        effective_concurrency = 1 if not self.use_api else max_concurrency
        config = RunnableConfig(max_concurrency=effective_concurrency)

        if not hasattr(self, "_context_link_chain"):
            self._context_link_chain = self._create_context_link_chain()

        results = []
        with tqdm(
            total=len(formatted_inputs),
            desc="Classifying Context Links",
            unit="link",
            leave=False,
        ) as pbar:
            for i in range(0, len(formatted_inputs), max_concurrency):
                mini_batch = formatted_inputs[i : i + max_concurrency]
                try:
                    responses = self._context_link_chain.batch(
                        mini_batch, config=config, return_exceptions=True
                    )
                except Exception as e:
                    logger.error(f"Critical error in batch {i // max_concurrency}: {e}")
                    responses = [None] * len(mini_batch)

                import traceback

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
                                f"Expected one of: {relation_types} or 'no link'"
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
        return re.sub(r"[^\w\s]", "", text)

    def batch_resolve_link_conflicts(
        self,
        conflicts: List[Dict[str, Any]],
        relation_types: List[str],
        pbar: Optional[Any] = None,
        show_progress: bool = True,
    ) -> List[Optional[str]]:
        if not conflicts:
            return []

        relation_types_str = ", ".join(relation_types)
        relation_types = [t.lower() for t in relation_types]

        prepared_conflicts = []
        for conflict in conflicts:
            candidate_predictions = self._format_candidate_predictions(
                conflict.get("candidate_counts", {})
            )
            evidence = self._format_conflict_evidence(conflict.get("evidence", []))
            prepared_conflicts.append(
                {
                    "filename_a": self._sanitize_text_input(
                        conflict.get("filename_a", "")
                    ),
                    "filename_b": self._sanitize_text_input(
                        conflict.get("filename_b", "")
                    ),
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

                import traceback

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
                        elif sanitized_link in relation_types:
                            results.append(sanitized_link)
                        else:
                            logger.warning(
                                f"Invalid resolved relation type: '{sanitized_link}'. "
                                f"Expected one of: {relation_types} or 'no link'"
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

    def _format_conflict_evidence(self, evidence_rows: List[Dict[str, Any]]) -> str:
        if not evidence_rows:
            return "- No evidence rows provided."

        rows = []
        for idx, row in enumerate(evidence_rows[:12], start=1):
            relation = row.get("relation_type", "unknown")
            source_chunk = self._sanitize_text_input(str(row.get("text_a", "")))[:400]
            target_chunk = self._sanitize_text_input(str(row.get("text_b", "")))[:400]

            source_chunk_index = row.get("source_chunk_index")
            target_chunk_index = row.get("target_chunk_index")
            reranker_score = row.get("reranker_score")
            vector_distance = row.get("vector_distance")

            rows.append(
                (
                    f"- Evidence #{idx}\n"
                    f"  predicted_relation: {relation}\n"
                    f"  source_chunk_index: {source_chunk_index}\n"
                    f"  target_chunk_index: {target_chunk_index}\n"
                    f"  reranker_score: {reranker_score}\n"
                    f"  vector_distance: {vector_distance}\n"
                    f'  source_chunk: "{source_chunk}"\n'
                    f'  target_chunk: "{target_chunk}"'
                )
            )

        return "\n".join(rows)
