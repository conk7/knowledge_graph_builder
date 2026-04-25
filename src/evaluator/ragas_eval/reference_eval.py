"""
RAGAS end-to-end evaluation of the ReferenceGraphRAGPipeline
(ObsidianRAG re-implementation: hybrid search → CrossEncoder rerank →
single-hop wikilink expansion → LLM generation).
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms.base import LangchainLLMWrapper
from ragas.run_config import RunConfig

from src.graphrag.config import (
    DEFAULT_TOP_K_CONTEXT,
    DEFAULT_TOP_K_SEED,
    EMBEDDING_MODEL_NAME,
    GraphRAGConfig,
)
from src.graphrag.reference import ReferenceGraphRAGPipeline, _load_vault_config
from src.kg_builder.config import META_DIR_NAME
from src.kg_builder.vault_manager import VaultManager

from ._shared import (
    _DEFAULT_TEMPERATURE,
    _DEFAULT_TOP_P,
    _LLM_MAX_RETRIES,
    _LLM_TIMEOUT_SEC,
    _build_metrics,
    _load_embeddings,
    _load_llm,
    _print_results,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_PIPELINE_MAX_RETRIES = 3


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


async def _run_single_item(
    i: int,
    total: int,
    item: dict,
    pipeline: ReferenceGraphRAGPipeline,
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
    pipeline: ReferenceGraphRAGPipeline,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "RAGAS E2E evaluation of the reference GraphRAG pipeline "
            "(ObsidianRAG: hybrid search → CrossEncoder rerank → "
            "single-hop wikilink expansion)."
        )
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
        default="results/ragas_reference_graphrag.json",
        help="Output path for detailed per-sample results JSON.",
    )
    parser.add_argument(
        "--ignore-local-config",
        action="store_true",
        help="Ignore saved vault config and use defaults from config.py instead.",
    )
    parser.add_argument(
        "--top-k-seed",
        type=int,
        default=DEFAULT_TOP_K_SEED,
        help="Passed to GraphRAGConfig (not used by reference pipeline directly, "
             "kept for config parity).",
    )
    parser.add_argument(
        "--top-k-context",
        type=int,
        default=DEFAULT_TOP_K_CONTEXT,
        help="Passed to GraphRAGConfig (not used by reference pipeline directly).",
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
        help="Max concurrent LLM calls during RAGAS evaluation via RunConfig.max_workers.",
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
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=f"Sampling temperature for both pipeline and eval LLMs "
             f"(default: {_DEFAULT_TEMPERATURE}; falls back to LLM_TEMPERATURE env).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help=f"Nucleus sampling probability "
             f"(default: {_DEFAULT_TOP_P}; falls back to LLM_TOP_P env).",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=_LLM_MAX_RETRIES,
        help=f"Max retries per LLM API call (default: {_LLM_MAX_RETRIES}).",
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=_LLM_TIMEOUT_SEC,
        help=f"Timeout in seconds per LLM API call (default: {_LLM_TIMEOUT_SEC}).",
    )
    parser.add_argument(
        "--eval-provider",
        default=None,
        help="LLM provider for RAGAS evaluation (overrides EVAL_LLM_PROVIDER / LLM_PROVIDER).",
    )
    parser.add_argument(
        "--eval-model",
        default=None,
        help="Model name for RAGAS evaluation (overrides EVAL_MODEL / MODEL).",
    )
    args = parser.parse_args()

    if args.eval_provider:
        os.environ["EVAL_LLM_PROVIDER"] = args.eval_provider
    if args.eval_model:
        os.environ["EVAL_MODEL"] = args.eval_model

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

    pipeline_llm = _load_llm(
        temperature=args.temperature,
        top_p=args.top_p,
        max_retries=args.llm_max_retries,
        timeout=args.llm_timeout,
    )
    eval_llm = _load_llm(
        prefix="EVAL_",
        temperature=args.temperature,
        top_p=args.top_p,
        max_retries=args.llm_max_retries,
        timeout=args.llm_timeout,
    )
    ragas_llm = LangchainLLMWrapper(eval_llm)

    raw_vault_cfg = {} if args.ignore_local_config else _load_vault_config(vault_dir)
    embedding_model = (
        raw_vault_cfg.get("models", {})
        .get("embedding", {})
        .get("model_name", EMBEDDING_MODEL_NAME)
    )
    ragas_embeddings = _load_embeddings(embedding_model)

    logger.info("Indexing vault and building reference GraphRAG pipeline...")
    pipeline_config = GraphRAGConfig(
        top_k_seed=args.top_k_seed,
        top_k_context=args.top_k_context,
    )
    with ReferenceGraphRAGPipeline.from_vault(
        vault_dir=vault_dir,
        llm=pipeline_llm,
        config=pipeline_config,
        ignore_local_config=args.ignore_local_config,
    ) as pipeline:
        vm = VaultManager(vault_path=vault_dir, ignored_dirs=[vault_dir / META_DIR_NAME])

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
    _print_results(scores, title="RAGAS Reference GraphRAG (ObsidianRAG) Evaluation Results")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "vault": str(vault_dir),
        "qa_file": str(qa_path),
        "config": {
            "initial_k": 12,
            "rerank_top_k": 6,
            "rerank_threshold": 0.3,
            "link_expand_k": 5,
            "link_score_mult": 0.9,
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
