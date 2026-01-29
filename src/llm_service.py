from langchain_google_genai import ChatGoogleGenerativeAI
from config import (
    LLM_BACKEND,
    LLM_N_GPU_LAYERS,
    LLM_N_CTX,
    LLM_TEMPERATURE,
    DEFAULT_LINK_TYPES,
    LLM_N_BATCH,
)
from templates import PROMPT_TEMPLATE_FOR_LINKING, PROMPT_TEMPLATE_FOR_RELEVANCE_CHECK
from typing import Dict, List, Optional
import logging
import re
import gc
import os

import torch
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableConfig
from pydantic.v1 import BaseModel, Field


class Link(BaseModel):
    reasoning: str = Field(description="A brief explanation for the chosen link.")
    relation_type: str = Field(description="The single most appropriate link type.")


class Relevance(BaseModel):
    reasoning: str = Field(description="A brief explanation for the chosen link.")
    is_revelant: str = Field(description="Are texts relevant?")


class LLMService:
    def __init__(
        self,
        model_path: Path,
        n_gpu_layers: int = LLM_N_GPU_LAYERS,
        n_ctx: int = LLM_N_CTX,
        n_batch: int = LLM_N_BATCH,
        temperature: float = LLM_TEMPERATURE,
        use_google_api: bool = False,
    ):
        self.model_name = str(model_path)

        logging.info(f"Loading GGUF model via LlamaCpp from: {model_path}...")
        if use_google_api:
            load_dotenv()
            api_key = os.environ.get("GOOGLE_API_KEY")
            model = os.environ.get("MODEL")
            self.llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)
        else:
            self.llm = LlamaCpp(
                model_path=str(model_path),
                temperature=temperature,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                n_batch=n_batch,
                verbose=False,
                backend=LLM_BACKEND,
            )
        logging.info("GGUF model loaded successfully.")

        self._link_chain = self._create_link_chain()
        self._relevance_chain = self._create_relevance_chain()

    def close(self):
        self.llm.client.close()
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()
        logging.info("LlamaCpp client resources released.")

    def _create_relevance_chain(self):
        parser = JsonOutputParser(pydantic_object=Relevance)

        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE_FOR_RELEVANCE_CHECK,
            input_variables=["text_a", "text_b"],
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
                "link_types": DEFAULT_LINK_TYPES,
            },
        )

        return prompt | self.llm | parser

    def check_relevance(self, text_a: str, text_b: str) -> bool:
        text_a = self._sanitize_text_input(text_a)
        text_b = self._sanitize_text_input(text_b)

        logging.debug(
            f"Checking link relevance of\ntext_a:\n{text_a}\n\ntext_b:\n{text_b}\n"
        )
        try:
            response = self._relevance_chain.invoke(
                {"text_a": text_a, "text_b": text_b}
            )
            logging.debug(f"Models response: \n\n{response}\n")
        except Exception as e:
            logging.debug(f"Could not invoke relevance chain: {e}")
            return False

        try:
            is_revelant = Relevance.validate(response).is_revelant.lower()
        except Exception as e:
            logging.error(f"Could not validate model response: {e}")
            return False

        return "yes" in self._sanitize_model_response(is_revelant)

    def batch_check_relevance(
        self, inputs: List[Dict[str, str]], max_concurrency: int = LLM_N_BATCH
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

        config = RunnableConfig(max_concurrency=max_concurrency)
        try:
            responses = self._relevance_chain.batch(sanitized_inputs, config=config)
        except Exception as e:
            logging.error(f"An error occurred during batch relevance check: {e}")
            return [False] * len(inputs)

        for response in responses:
            try:
                is_relevant_str = Relevance.validate(response).is_revelant.lower()
                results.append("yes" in self._sanitize_model_response(is_relevant_str))
            except Exception as e:
                logging.error(
                    f"Could not validate or parse a response in batch: {response} - Error: {e}"
                )
                results.append(False)

        return results

    def _create_link_chain(self):
        parser = JsonOutputParser(pydantic_object=Link)

        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE_FOR_LINKING,
            input_variables=["text_a", "text_b", "relation_types"],
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
            },
        )

        return prompt | self.llm | parser

    def classify_link(
        self, text_a: str, text_b: str, relation_types: List[str]
    ) -> Optional[str]:
        text_a = self._sanitize_text_input(text_a)
        text_b = self._sanitize_text_input(text_b)
        try:
            response = self._link_chain.invoke(
                {
                    "text_a": text_a,
                    "text_b": text_b,
                    "relation_types": ", ".join(relation_types),
                }
            )
        except Exception as e:
            logging.error(f"An error occurred during classification of a link: {e}")
        try:
            link = Link.validate(response).relation_type.lower()
        except Exception as e:
            logging.error(f"Could not parse model response: {e}")
            return None

        link_parsed = self._sanitize_model_response(link)

        if link_parsed in relation_types:
            return link_parsed
        else:
            logging.warning(
                f"LLM returned an invalid link type: '{link}' (parsed: '{link_parsed}')."
            )
            return None

    def batch_classify_link(
        self,
        inputs: List[Dict[str, str]],
        relation_types: List[str],
        max_concurrency: int = LLM_N_BATCH,
    ) -> List[Optional[str]]:
        if not inputs:
            return []

        relation_types_str = ", ".join(relation_types)
        batch_inputs = [
            {
                "text_a": self._sanitize_text_input(item["text_a"]),
                "text_b": self._sanitize_text_input(item["text_b"]),
                "relation_types": relation_types_str,
            }
            for item in inputs
        ]

        logging.debug(f"Classifying links for a batch of {len(batch_inputs)} items.")
        results = []

        config = RunnableConfig(max_concurrency=max_concurrency)
        try:
            responses = self._link_chain.batch(batch_inputs, config=config)
        except Exception as e:
            logging.error(f"An error occurred during batch link classification: {e}")
            return [None] * len(inputs)

        for response in responses:
            try:
                link = Link.validate(response).relation_type.lower()
            except Exception as e:
                logging.error(
                    f"Could not parse a model response in batch: {response} - Error: {e}"
                )
                results.append(None)

            sanitized_link = self._sanitize_model_response(link)
            if sanitized_link in relation_types:
                results.append(sanitized_link)
            else:
                logging.warning(
                    f"LLM returned an invalid link type in batch: '{link}' (sanitized: '{sanitized_link}')."
                )
                results.append(None)

        return results

    def _sanitize_model_response(self, text: str) -> str:
        return re.sub(r"[^\w_]", "", text)

    def _sanitize_text_input(self, text: str) -> str:
        return re.sub(r"[^\w\s]", "", text)
