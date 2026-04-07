import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import (
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from ragas.run_config import RunConfig

from src.graphrag.config import (
    DEFAULT_BEAM_WIDTH,
    DEFAULT_MAX_HOPS,
    DEFAULT_NER_BOOST_FACTOR,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TOP_K_CONTEXT,
    DEFAULT_TOP_K_SEED,
    GraphRAGConfig,
)
from src.graphrag.pipeline import GraphRAGPipeline, _load_vault_config
from src.kg_builder.vault_manager import VaultManager

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


_LLM_MAX_RETRIES: int = 10
_LLM_TIMEOUT_SEC: int = 240


def _load_llm() -> Any:
    load_dotenv()
    provider = os.environ.get("LLM_PROVIDER", "").lower()
    model = os.environ.get("MODEL", "")
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
    top_p = float(os.environ.get("LLM_TOP_P", "0.1"))

    logger.info(f"Loading LLM: provider={provider!r} model={model!r}")

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1"),
            temperature=temperature,
            model_kwargs={"top_p": top_p},
            max_retries=_LLM_MAX_RETRIES,
            timeout=_LLM_TIMEOUT_SEC,
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            temperature=temperature,
            top_p=top_p,
            max_retries=_LLM_MAX_RETRIES,
        )
    elif provider == "cerebras":
        from langchain_cerebras import ChatCerebras

        return ChatCerebras(
            model=model or "llama3.1-8b",
            api_key=os.environ.get("CEREBRAS_API_KEY"),
            temperature=temperature,
            top_p=top_p,
            max_retries=_LLM_MAX_RETRIES,
            timeout=_LLM_TIMEOUT_SEC,
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model,
            api_key=os.environ.get("GROQ_API_KEY"),
            temperature=temperature,
            max_retries=_LLM_MAX_RETRIES,
            request_timeout=_LLM_TIMEOUT_SEC,
        )
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: {provider!r}. "
            "Set LLM_PROVIDER to one of: openai, google, cerebras, groq."
        )


def _load_embeddings(model_name: str) -> LangchainEmbeddingsWrapper:
    from langchain_huggingface import HuggingFaceEmbeddings

    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=model_name))


def _load_reference_contexts(
    relevant_docs: list[str],
    vault_dir: Path,
    vm: VaultManager,
) -> list[str]:
    contexts: list[str] = []
    for stem in relevant_docs:
        path = vault_dir / f"{stem}.md"
        if not path.exists():
            logger.warning(f"relevant_doc not found: {path}")
            continue
        content = vm.get_file_content(path)
        body, _ = vm._split_content_and_links(content)
        if body.strip():
            contexts.append(body.strip())
    return contexts


_PIPELINE_MAX_RETRIES = 3


async def _run_single_item(
    i: int,
    total: int,
    item: dict,
    pipeline: GraphRAGPipeline,
    vault_dir: Path,
    vm: VaultManager,
    semaphore: asyncio.Semaphore,
) -> tuple[int, SingleTurnSample, dict]:
    question = item["question"]
    reference = item.get("answer", "")
    relevant_docs = item.get("relevant_docs", [])

    async with semaphore:
        for attempt in range(_PIPELINE_MAX_RETRIES):
            try:
                contexts, response = await asyncio.to_thread(pipeline.run, question)
                break
            except Exception as exc:
                if attempt == _PIPELINE_MAX_RETRIES - 1:
                    raise
                logger.warning(
                    f"[{i + 1}/{total}] attempt {attempt + 1} failed: {exc!r}, retrying..."
                )

    logger.info(
        f"[{i + 1}/{total}] {question[:80]}\n"
        f"  contexts retrieved: {len(contexts)}, response length: {len(response)}"
    )

    ref_contexts = _load_reference_contexts(relevant_docs, vault_dir, vm)

    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=contexts,
        response=response,
        reference=reference,
        reference_contexts=ref_contexts if ref_contexts else None,
    )
    raw = {
        "question": question,
        "answer": reference,
        "relevant_docs": relevant_docs,
        "retrieved_contexts": contexts,
        "response": response,
    }
    return i, sample, raw


def run_evaluation(
    vault_dir: Path,
    qa_items: list[dict],
    pipeline: GraphRAGPipeline,
    vm: VaultManager,
    pipeline_workers: int = 4,
) -> tuple[EvaluationDataset, list[dict]]:
    async def _run() -> tuple[EvaluationDataset, list[dict]]:
        semaphore = asyncio.Semaphore(pipeline_workers)
        tasks = [
            _run_single_item(i, len(qa_items), item, pipeline, vault_dir, vm, semaphore)
            for i, item in enumerate(qa_items)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        ordered: list[tuple[int, SingleTurnSample, dict]] = []
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Item failed permanently: {res!r}")
            else:
                ordered.append(res)

        ordered.sort(key=lambda x: x[0])
        samples = [s for _, s, _ in ordered]
        raw_results = [r for _, _, r in ordered]
        return EvaluationDataset(samples=samples), raw_results

    return asyncio.run(_run())


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


def _print_results(scores: dict) -> None:
    print("\n" + "=" * 50)
    print("RAGAS GraphRAG Evaluation Results")
    print("=" * 50)
    for metric, value in scores.items():
        if isinstance(value, float):
            print(f"  {metric:<25} {value:.4f}")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAGAS E2E evaluation of the GraphRAG pipeline."
    )
    parser.add_argument(
        "--vault",
        required=True,
        help="Path to vault directory (e.g. data/test_vaults/gold/sequence).",
    )
    parser.add_argument(
        "--qa-file",
        required=True,
        help="Path to QA JSON dataset (list of {question, answer, relevant_docs}).",
    )
    parser.add_argument(
        "--output",
        default="results/ragas_graphrag.json",
        help="Output path for detailed per-sample results JSON.",
    )
    parser.add_argument(
        "--ignore-local-config",
        action="store_true",
        help="Ignore saved vault config and use defaults from config.py instead.",
    )
    parser.add_argument("--max-hops", type=int, default=DEFAULT_MAX_HOPS)
    parser.add_argument("--beam-width", type=int, default=DEFAULT_BEAM_WIDTH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    parser.add_argument("--top-k-seed", type=int, default=DEFAULT_TOP_K_SEED)
    parser.add_argument("--top-k-context", type=int, default=DEFAULT_TOP_K_CONTEXT)
    parser.add_argument(
        "--ner-boost-factor", type=float, default=DEFAULT_NER_BOOST_FACTOR
    )
    parser.add_argument(
        "--pipeline-workers",
        type=int,
        default=4,
        help="Max parallel pipeline.run() calls during QA collection (default 4).",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Max concurrent LLM calls during RAGAS evaluation via RunConfig.max_workers (lower = fewer 429s).",
    )
    parser.add_argument(
        "--eval-timeout",
        type=int,
        default=6000,
        help="Timeout in seconds for a single RAGAS LLM call (default 6000).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=100,
        help="Max retries per RAGAS LLM call (default 100).",
    )
    args = parser.parse_args()

    vault_dir = Path(args.vault)
    qa_path = Path(args.qa_file)
    output_path = Path(args.output)

    if not vault_dir.is_dir():
        raise FileNotFoundError(f"Vault not found: {vault_dir}")
    if not qa_path.exists():
        raise FileNotFoundError(f"QA file not found: {qa_path}")

    with qa_path.open("r", encoding="utf-8") as f:
        qa_items: list[dict] = json.load(f)
    logger.info(f"Loaded {len(qa_items)} QA items from {qa_path}")

    llm = _load_llm()
    ragas_llm = LangchainLLMWrapper(llm)

    raw_vault_cfg = {} if args.ignore_local_config else _load_vault_config(vault_dir)
    embedding_model = (
        raw_vault_cfg.get("models", {})
        .get("embedding", {})
        .get(
            "model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    )
    ragas_embeddings = _load_embeddings(embedding_model)

    logger.info("Indexing vault and building GraphRAG pipeline...")
    pipeline_config = GraphRAGConfig(
        max_hops=args.max_hops,
        beam_width=args.beam_width,
        score_threshold=args.threshold,
        top_k_seed=args.top_k_seed,
        top_k_context=args.top_k_context,
        ner_boost_factor=args.ner_boost_factor,
    )
    with GraphRAGPipeline.from_vault(
        vault_dir=vault_dir,
        llm=llm,
        config=pipeline_config,
        ignore_local_config=args.ignore_local_config,
    ) as pipeline:
        from src.kg_builder.config import META_DIR_NAME
        from src.kg_builder.vault_manager import VaultManager as VM

        vm = VM(vault_path=vault_dir, ignored_dirs=[vault_dir / META_DIR_NAME])

        dataset, raw_results = run_evaluation(
            vault_dir=vault_dir,
            qa_items=qa_items,
            pipeline=pipeline,
            vm=vm,
            pipeline_workers=args.pipeline_workers,
        )

    logger.info("Running RAGAS evaluation...")
    metrics = _build_metrics(ragas_llm, ragas_embeddings)
    run_config = RunConfig(
        timeout=args.eval_timeout,
        max_retries=args.max_retries,
        max_workers=args.max_concurrent,
    )
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=run_config,
    )
    df = result.to_pandas()
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if not nan_cols.empty:
        logger.warning(
            f"NaN values detected (failed samples per metric):\n{nan_cols.to_string()}"
        )
    scores = df.mean(numeric_only=True).to_dict()
    scores["_evaluated_samples"] = int(
        df.shape[0] - nan_cols.max() if not nan_cols.empty else df.shape[0]
    )
    _print_results(scores)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "vault": str(vault_dir),
        "qa_file": str(qa_path),
        "config": {
            "max_hops": args.max_hops,
            "beam_width": args.beam_width,
            "threshold": args.threshold,
            "top_k_seed": args.top_k_seed,
            "top_k_context": args.top_k_context,
        },
        "scores": scores,
        "samples": raw_results,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
