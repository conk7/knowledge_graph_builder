from config import (
    LLM_BACKEND,
    LLM_N_GPU_LAYERS,
    LLM_N_CTX,
    LLM_TEMPERATURE,
    LLM_RELATION_TYPES,
)
from templates import RELATIONSHIP_PROMPT_TEMPLATE, RELEVANCE_PROMPT_TEMPLATE
from typing import List, Optional
import logging
import re
import gc

import torch
from pathlib import Path
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic.v1 import BaseModel, Field


class Relationship(BaseModel):
    reasoning: str = Field(
        description="A brief explanation for the chosen relationship."
    )
    relation_type: str = Field(
        description="The single most appropriate relationship type."
    )


class Relevance(BaseModel):
    reasoning: str = Field(
        description="A brief explanation for the chosen relationship."
    )
    is_revelant: str = Field(description="Are texts relevant?")


class LLMService:
    def __init__(
        self,
        model_path: Path,
        n_gpu_layers: int = LLM_N_GPU_LAYERS,
        n_ctx: int = LLM_N_CTX,
        temperature: float = LLM_TEMPERATURE,
    ):
        self.model_name = str(model_path)

        logging.info(f"Loading GGUF model via LlamaCpp from: {model_path}...")
        self.llm = LlamaCpp(
            model_path=str(model_path),
            temperature=temperature,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
            backend=LLM_BACKEND,
            # max_tokens=1024,
        )
        logging.info("GGUF model loaded successfully.")

        self._relationship_chain = self._create_relationship_chain()
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
            template=RELEVANCE_PROMPT_TEMPLATE,
            input_variables=["text_a", "text_b"],
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
                "relationship_types": LLM_RELATION_TYPES,
            },
        )

        return prompt | self.llm | parser

    def check_relevance(self, text_a: str, text_b: str) -> bool:
        text_a = self._process_text_input(text_a)
        text_b = self._process_text_input(text_b)

        logging.debug(
            f"Checking link relevance of\ntext_a:\n{text_a}\n\ntext_b:\n{text_b}\n"
        )
        response = self._relevance_chain.invoke({"text_a": text_a, "text_b": text_b})
        logging.debug(f"Models response: \n\n{response}\n")

        try:
            is_revelant = Relevance.validate(response).is_revelant.lower()
        except Exception as e:
            logging.error(f"Could not validate model response: {e}")
            return False

        return "yes" in self._process_model_response(is_revelant)

    def _create_relationship_chain(self):
        parser = JsonOutputParser(pydantic_object=Relationship)

        prompt = PromptTemplate(
            template=RELATIONSHIP_PROMPT_TEMPLATE,
            input_variables=["text_a", "text_b", "relation_types"],
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
            },
        )

        return prompt | self.llm | parser

    def classify_relationship(
        self, text_a: str, text_b: str, relation_types: List[str]
    ) -> Optional[str]:
        text_a = self._process_text_input(text_a)
        text_b = self._process_text_input(text_b)
        try:
            response = self._relationship_chain.invoke(
                {
                    "text_a": text_a,
                    "text_b": text_b,
                    "relation_types": ", ".join(relation_types),
                }
            )
        except Exception as e:
            logging.error(
                f"An error occurred during classification of relationship: {e}"
            )
        try:
            relationship = Relationship.validate(response).relation_type.lower()
        except Exception as e:
            logging.error(f"Could not parse model response: {e}")
            return None

        relationship_parsed = self._process_model_response(relationship)

        if relationship_parsed in relation_types:
            return relationship_parsed
        else:
            logging.warning(
                f"LLM returned an invalid relationship type: '{relationship}' (parsed: '{relationship_parsed}')."
            )
            return None

    def _process_model_response(self, text: str) -> str:
        return re.sub(r"[^\w_]", "", text)

    def _process_text_input(self, text: str) -> str:
        return re.sub(r"[^\w\s]", "", text)
