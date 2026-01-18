from config import (
    LLM_BACKEND,
    LLM_N_GPU_LAYERS,
    LLM_N_CTX,
    LLM_TEMPERATURE,
    LLM_RELATION_TYPES,
)
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


# class Tag(BaseModel):
#     """A model to represent a single document tag with confidence."""

#     tag: str = Field(description="The topic or tag.")
#     confidence: float = Field(description="The confidence score, from 0.0 to 1.0.")


# class TagList(BaseModel):
#     """A model representing a list of tags for a document."""

#     tags: List[Tag] = Field(description="A list of tag objects.")


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
        # self._tagging_chain = self._create_tagging_chain()

    def close(self):
        self.llm.client.close()
        self.llm = None
        gc.collect()
        torch.cuda.empty_cache()
        logging.info("LlamaCpp client resources released.")

    def _create_relevance_chain(self):
        parser = JsonOutputParser(pydantic_object=Relevance)

        prompt = PromptTemplate(
            template="""You are a relevance analysis expert. Your task is to determine if a meaningful, 
non-trivial semantic relationship exists between Document A and Document B.
A trivial relationship is one of just "similarity". A meaningful relationship could be {relationship_types}, etc.
Think step-by-step and then conclude your answer with a single word: "Yes" or "No".

{format_instructions}

Document A:
---
{text_a}
---

Document B:
---
{text_b}
---
Step-by-step thought process:
1. What is the main topic of Document A?
2. What is the main topic of Document B?
3. Do they discuss the same concepts from different perspectives, or are they just vaguely related?
4. Is there a clear, definable link (like example, explanation, contradiction)?

Conclusion (Yes or No):""",
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

        try:
            logging.debug(
                f"Checking link relevance of\ntext_a:\n{text_a}\n\ntext_b:\n{text_b}\n"
            )
            response = self._relevance_chain.invoke(
                {"text_a": text_a, "text_b": text_b}
            )
            logging.debug(f"Models response: \n\n{response}\n")
        except Exception as e:
            logging.error(f"An error occurred during relevance check: {e}")
            return False

        try:
            response = Relevance.validate(response).is_revelant.lower()
        except Exception as e:
            logging.error(f"Could not parse model response: {e}")
            return False

        if "yes" in self._process_model_response(response):
            return True
        return False

    def _create_relationship_chain(self):
        parser = JsonOutputParser(pydantic_object=Relationship)

        prompt = PromptTemplate(
            template="""You are a highly specialized API endpoint that only returns JSON.
Your task is to analyze the semantic relationship from Document A to Document B.
Your entire response MUST be a single, valid JSON object and nothing else.

Follow these formatting instructions precisely:
{format_instructions}

The possible relationship types are: {relation_types}

---
Document A: {text_a}
---
Document B: {text_b}
---

Now, perform the analysis and return the JSON object.
        """,
            input_variables=["text_a", "text_b", "relation_types"],
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
            },
        )

        return prompt | self.llm | parser

    def classify_relationship(
        self, text_a: str, text_b: str, relation_types: List[str]
    ) -> Optional[str]:
        if not self._relationship_chain:
            logging.error("Cannot classify relationship: chain is not available.")
            return None

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

    #     def _create_tagging_chain(self):
    #         if not self.llm:
    #             return None

    #         parser = JsonOutputParser(pydantic_object=TagList)

    #         prompt = PromptTemplate(
    #             template="""You are an expert librarian. Analyze the document and identify the {max_tags} most relevant topics as tags.
    # Your response MUST be ONLY a valid JSON object.

    # {format_instructions}

    # Document:
    # ---
    # {text}
    # ---
    # JSON response:""",
    #             input_variables=["text", "max_tags"],
    #             partial_variables={"format_instructions": parser.get_format_instructions()},
    #         )

    #         return prompt | self.llm | parser

    # def classify_tags(self, text: str, max_tags: int = 3) -> List[Dict[str, Any]]:
    #     if not self._tagging_chain:
    #         logging.error("Cannot classify tags: chain is not available.")
    #         return []

    #     response_data = self._tagging_chain.invoke({"text": text, "max_tags": max_tags})
    #     return response_data.get("tags", [])
