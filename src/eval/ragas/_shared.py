from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import (
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

logger = logging.getLogger(__name__)

_DEFAULT_TEMPERATURE: float = 0.0
_DEFAULT_TOP_P: float = 0.1
_LLM_MAX_RETRIES: int = 10
_LLM_TIMEOUT_SEC: int = 240


def _load_llm(
    prefix: str = "",
    temperature: float | None = None,
    top_p: float | None = None,
    max_retries: int = _LLM_MAX_RETRIES,
    timeout: int = _LLM_TIMEOUT_SEC,
) -> Any:
    load_dotenv()

    def _get(key: str, default: str = "") -> str:
        return os.environ.get(f"{prefix}{key}") or os.environ.get(key, default)

    provider = _get("LLM_PROVIDER").lower()
    model = _get("MODEL")

    if temperature is None:
        temperature = float(_get("LLM_TEMPERATURE", str(_DEFAULT_TEMPERATURE)))
    if top_p is None:
        top_p = float(_get("LLM_TOP_P", str(_DEFAULT_TOP_P)))

    role = "eval LLM" if prefix else "pipeline LLM"
    logger.info(
        "Loading %s: provider=%r model=%r temperature=%s top_p=%s",
        role,
        provider,
        model,
        temperature,
        top_p,
    )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=_get("OPENAI_API_KEY"),
            base_url=_get("OPENAI_BASE_URL", "http://localhost:1234/v1"),
            temperature=temperature,
            model_kwargs={"top_p": top_p},
            max_retries=max_retries,
            timeout=timeout,
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=_get("GOOGLE_API_KEY"),
            temperature=temperature,
            top_p=top_p,
            max_retries=max_retries,
        )
    elif provider == "cerebras":
        from langchain_cerebras import ChatCerebras

        return ChatCerebras(
            model=model or "llama3.1-8b",
            api_key=_get("CEREBRAS_API_KEY"),
            temperature=temperature,
            top_p=top_p,
            max_retries=max_retries,
            timeout=timeout,
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model,
            api_key=_get("GROQ_API_KEY"),
            temperature=temperature,
            max_retries=max_retries,
            request_timeout=timeout,
        )
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER ({role}): {provider!r}. "
            "Set LLM_PROVIDER to one of: openai, google, cerebras, groq."
        )


def _load_embeddings(model_name: str) -> LangchainEmbeddingsWrapper:
    from langchain_huggingface import HuggingFaceEmbeddings

    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=model_name))


def _build_metrics(
    ragas_llm: LangchainLLMWrapper,
    ragas_embeddings: LangchainEmbeddingsWrapper,
) -> list:
    return [
        ContextRecall(llm=ragas_llm),
        ContextPrecision(llm=ragas_llm),
        Faithfulness(llm=ragas_llm),
        AnswerCorrectness(llm=ragas_llm, embeddings=ragas_embeddings),
    ]


def _print_results(scores: dict, title: str = "RAGAS Evaluation Results") -> None:
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    for metric, value in scores.items():
        if isinstance(value, float):
            print(f"  {metric:<25} {value:.4f}")
    print("=" * 50)
