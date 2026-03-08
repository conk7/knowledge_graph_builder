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
    HUMAN_TEMPLATE_FOR_LINKING,
    HUMAN_TEMPLATE_FOR_RELEVANCE_CHECK,
    SYSTEM_TEMPLATE_FOR_LINKING,
    SYSTEM_TEMPLATE_FOR_RELEVANCE_CHECK,
    SYSTEM_TEMPLATE_FOR_CONTEXT_LINKING,
    HUMAN_TEMPLATE_FOR_CONTEXT_LINKING,
)


class Link(BaseModel):
    reasoning: str = Field(description="A brief explanation for the chosen link.")
    relation_type: str = Field(description="The single most appropriate link type.")


class Relevance(BaseModel):
    reasoning: str = Field(description="A brief explanation for the chosen link.")
    is_relevant: str = Field(description="Are texts relevant?")


class LLMService:
    def __init__(
        self,
        model_path: Path,
        n_gpu_layers: int = -1,
        n_ctx: int = 8192,
        n_batch: int = 512,
        temperature: float = 0.0,
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
        self.use_api = use_api

        logging.info(f"Loading model (API: {use_api})...")
        if use_api:
            load_dotenv()
            provider = os.environ.get("LLM_PROVIDER", None).lower()
            if provider == "openai":
                self.llm = self._init_openai()
            elif provider == "google":
                self.llm = self._init_google()
            elif provider == "cerebras":
                self.llm = self._init_cerebras()
            else:
                raise ValueError("unsupported api provider")
        else:
            self.llm = self._init_local(model_path, n_gpu_layers, n_ctx, n_batch)

        logging.info("Model loaded successfully.")

        self._link_chain = self._create_link_chain()
        self._relevance_chain = self._create_relevance_chain()

    def _init_openai(self) -> ChatOpenAI:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1")
        model = os.environ.get("MODEL")

        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=self.temperature,
            streaming=False,
        )

        try:
            llm = llm.bind(response_format={"type": "json_object"})
        except Exception as e:
            logging.debug(f"Could not bind json_object response format: {e}")

        return llm

    def _init_google(self) -> ChatGoogleGenerativeAI:
        api_key = os.environ.get("GOOGLE_API_KEY")
        model = os.environ.get("MODEL")

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=self.temperature,
        )

    def _init_cerebras(self) -> ChatCerebras:
        api_key = os.environ.get("CEREBRAS_API_KEY")
        model = os.environ.get("MODEL", "llama3.1-8b")

        llm = ChatCerebras(
            model=model,
            api_key=api_key,
        )

        try:
            llm = llm.bind(response_format={"type": "json_object"})
        except Exception as e:
            logging.debug(f"Could not bind json_object response format: {e}")

        return llm

    def _init_local(
        self, model_path: Path, n_gpu_layers: int, n_ctx: int, n_batch: int
    ) -> ChatLlamaCpp:
        # stop_tokens = ["</s>", "<|endoftext|>"]

        return ChatLlamaCpp(
            model_path=str(model_path),
            temperature=self.temperature,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=n_batch,
            verbose=False,
            backend=self.backend,
            streaming=False,
            # stop=stop_tokens,
        )

    def close(self):
        if hasattr(self.llm, "client") and hasattr(self.llm.client, "close"):
            self.llm.client.close()
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()
        logging.info("LLM client resources released.")

    def _create_structured_chain(
        self,
        system_template: str,
        human_template: str,
        schema: Type[BaseModel],
        partial_variables: Optional[Dict[str, Any]] = None,
    ) -> Any:
        parser = PydanticOutputParser(pydantic_object=schema)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_template),
                ("human", human_template),
            ]
        )
        prompt = prompt.partial(
            format_instructions=parser.get_format_instructions(),
            **partial_variables if partial_variables is not None else {},
        )

        try:
            if hasattr(self.llm, "with_structured_output"):
                structured_llm = self.llm.with_structured_output(schema)
                return prompt | structured_llm
        except (NotImplementedError, AttributeError):
            pass

        return prompt | self.llm | parser

    def _create_relevance_chain(self):
        return self._create_structured_chain(
            system_template=SYSTEM_TEMPLATE_FOR_RELEVANCE_CHECK,
            human_template=HUMAN_TEMPLATE_FOR_RELEVANCE_CHECK,
            schema=Relevance,
            partial_variables={"link_types": self.default_link_types},
        )

    def batch_check_relevance(
        self, inputs: List[Dict[str, str]], max_concurrency: Optional[int] = None
    ) -> List[bool]:
        if not inputs:
            return []

        sanitized_inputs = [
            {
                "text_a": self._sanitize_text_input(item["text_a"]),
                "text_b": self._sanitize_text_input(item["text_b"]),
            }
            for item in inputs
        ]

        logging.debug(
            f"Checking link relevance for a batch of {len(sanitized_inputs)} items."
        )
        results = []

        max_concurrency = max_concurrency or self.concurrency
        max_concurrency = max_concurrency or self.concurrency
        effective_concurrency = 1 if not self.use_api else max_concurrency
        config = RunnableConfig(max_concurrency=effective_concurrency)
        try:
            responses = self._relevance_chain.batch(
                sanitized_inputs, config=config, return_exceptions=True
            )
        except Exception as e:
            logging.error(f"An error occurred during batch relevance check: {e}")
            return [False] * len(inputs)

        import traceback

        for response in responses:
            if isinstance(response, Exception):
                logging.error(
                    f"Item failed during LLM execution in batch_check_relevance: {response}\nTraceback:\n"
                    f"{''.join(traceback.format_exception(type(response), response, response.__traceback__))}"
                )
                results.append(False)
                continue

            try:
                is_relevant_str = response.is_relevant.lower()
                results.append("yes" in self._sanitize_model_response(is_relevant_str))
            except Exception as e:
                logging.error(
                    f"Could not validate or parse a response in batch: {response} - Error: {e}"
                )
                results.append(False)

        return results

    def _create_link_chain(self):
        return self._create_structured_chain(
            system_template=SYSTEM_TEMPLATE_FOR_LINKING,
            human_template=HUMAN_TEMPLATE_FOR_LINKING,
            schema=Link,
        )

    def batch_classify_link(
        self,
        texts: List[Dict[str, str]],
        text_meta: List[Dict[str, Path]],
        relation_types: List[str],
        max_concurrency: Optional[int] = None,
    ) -> List[Optional[str]]:
        if not texts:
            return []

        relation_types_str = ", ".join(relation_types)
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
        logging.debug(f"Classifying links for a batch of {len(texts)} items.")

        results = []
        max_concurrency = max_concurrency or self.concurrency
        max_concurrency = max_concurrency or self.concurrency
        effective_concurrency = 1 if not self.use_api else max_concurrency
        config = RunnableConfig(max_concurrency=effective_concurrency)

        with tqdm(
            total=len(texts),
            desc="Classifying Semantic Links",
            unit="pair",
            leave=False,
        ) as pbar:
            for i in range(0, len(texts), max_concurrency):
                mini_batch = texts[i : i + max_concurrency]
                try:
                    responses = self._link_chain.batch(
                        mini_batch, config=config, return_exceptions=True
                    )
                except Exception as e:
                    logging.error(
                        f"Critical error in batch {i // max_concurrency}: {e}"
                    )
                    responses = [None] * len(mini_batch)

                import traceback

                for i, response in enumerate(responses):
                    if isinstance(response, Exception):
                        logging.debug(
                            f"Item failed during LLM execution in batch_classify_link: {response}\nTraceback:\n"
                            f"{''.join(traceback.format_exception(type(response), response, response.__traceback__))}"
                        )
                        results.append(None)
                        continue

                    try:
                        link_type = response.relation_type.lower()
                        sanitized_link = self._sanitize_model_response(link_type)

                        logging.info(
                            f"LLM classified link between '{mini_batch[i]['filename_a']}' and "
                            f"'{mini_batch[i]['filename_b']}' as '{sanitized_link}'"
                        )
                        logging.info(f"    Raw response: {response}")

                        if sanitized_link in relation_types:
                            results.append(sanitized_link)
                        else:
                            logging.warning(
                                f"Invalid relation type returned: {sanitized_link}"
                            )
                            results.append(None)

                    except Exception as e:
                        logging.warning(f"Failed to parse LLM response: {e}")
                        results.append(None)

                pbar.update(len(mini_batch))

        return results

    
    def _create_context_link_chain(self):
        return self._create_structured_chain(
            system_template=SYSTEM_TEMPLATE_FOR_CONTEXT_LINKING,
            human_template=HUMAN_TEMPLATE_FOR_CONTEXT_LINKING,
            schema=Link,
        )

    def batch_classify_context_link(
        self,
        contexts: List[Dict[str, str]],
        relation_types: List[str],
        max_concurrency: Optional[int] = None,
    ) -> List[Optional[str]]:
        if not contexts:
            return []

        relation_types_str = ", ".join(relation_types)
        formatted_inputs = [
            {
                "source_title": self._sanitize_text_input(item["source"]),
                "target_title": self._sanitize_text_input(item["target"]),
                "context": self._sanitize_text_input(item["context"]),
                "relation_types": relation_types_str,
            }
            for item in contexts
        ]
        logging.debug(f"Classifying context links for a batch of {len(contexts)} items.")

        max_concurrency = max_concurrency or self.concurrency
        effective_concurrency = 1 if not self.use_api else max_concurrency
        config = RunnableConfig(max_concurrency=effective_concurrency)
        
        if not hasattr(self, '_context_link_chain'):
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
                    logging.error(f"Critical error in batch {i // max_concurrency}: {e}")
                    responses = [None] * len(mini_batch)

                import traceback

                for j, response in enumerate(responses):
                    if isinstance(response, Exception):
                        logging.debug(
                            f"Item failed during LLM execution: {response}\n"
                            f"{''.join(traceback.format_exception(type(response), response, response.__traceback__))}"
                        )
                        results.append(None)
                        continue

                    try:
                        link_type = response.relation_type.lower()
                        sanitized_link = self._sanitize_model_response(link_type)

                        if sanitized_link in relation_types:
                            results.append(sanitized_link)
                        else:
                            logging.warning(f"Invalid relation type returned: {sanitized_link}")
                            results.append(None)
                    except Exception as e:
                        logging.warning(f"Failed to parse LLM response: {e}")
                        results.append(None)

                pbar.update(len(mini_batch))

        return results

    def _sanitize_model_response(self, text: str) -> str:
        return re.sub(r"[^\w_]", "", text)

    def _sanitize_text_input(self, text: str) -> str:
        return re.sub(r"[^\w\s]", "", text)
