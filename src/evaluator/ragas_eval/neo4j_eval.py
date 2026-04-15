from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms.base import LangchainLLMWrapper
from ragas.run_config import RunConfig

from src.kg_builder.config import EMBEDDING_MODEL_NAME, META_DIR_NAME

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


DEFAULT_MAX_HOPS = 3
DEFAULT_TOP_K_SEED = 10
DEFAULT_TOP_K_CONTEXT = 10
DEFAULT_MIN_SCORE = 0.0
FULLTEXT_INDEX_NAME = "neo4j_entity_fulltext_idx"

_PIPELINE_MAX_RETRIES = 3
_PIPELINE_BACKOFF_BASE = 2.0

_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]:\s*['\"](\d+(?:\.\d+)?)s['\"]")
_TRANSIENT_MARKERS = ("RESOURCE_EXHAUSTED", "429", "UNAVAILABLE", "503")


def _is_transient_error(exc: Exception) -> bool:
    msg = str(exc)
    return any(marker in msg for marker in _TRANSIENT_MARKERS)


def _parse_retry_delay(exc: Exception, default: float = 30.0) -> float:
    m = _RETRY_DELAY_RE.search(str(exc))
    return float(m.group(1)) + 1.0 if m else default


_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """You are an expert analytical assistant. Answer the user's question based strictly on the provided knowledge graph context.

CRITICAL INSTRUCTIONS:
1. Language Mirroring (ABSOLUTE): Write your <answer> in the EXACT SAME LANGUAGE as the user's Question.
2. Logic First: Inside your <reasoning> tag, state the language of the Question, then plan your answer.
3. Style and Tone (CRUCIAL FOR METRICS): Write your <answer> as a cohesive, flowing paragraph. Do NOT use markdown bullet points or numbered lists unless absolutely unavoidable. Synthesize the facts naturally.
4. Concise Accuracy: Directly answer the core question. Include necessary specific names and numbers from the context, but do NOT add extra historical background or broad summaries that weren't directly requested.
5. Output Format: Use <reasoning> for your internal logic and <answer> for the final response.

Context:\n{context}\n\nQuestion: {question}""",
        ),
    ]
)


class Neo4jGraphRAGPipeline:
    def __init__(
        self,
        graph: Neo4jGraph,
        llm: Any,
        vault: str | None = None,
        max_hops: int = DEFAULT_MAX_HOPS,
        top_k_seed: int = DEFAULT_TOP_K_SEED,
        top_k_context: int = DEFAULT_TOP_K_CONTEXT,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> None:
        self.graph = graph
        self.llm = llm
        self.vault = vault
        self.max_hops = max_hops
        self.top_k_seed = top_k_seed
        self.top_k_context = top_k_context
        self.min_score = min_score
        self._vault_label = (
            "".join(part.capitalize() for part in re.split(r"[_\-\s]+", vault) if part)
            if vault
            else ""
        )
        self._fulltext_index = (
            f"{self._vault_label}_text_index"
            if self._vault_label
            else FULLTEXT_INDEX_NAME
        )
        self._doc_fulltext_index = "document_text_index"
        self._chain = _ANSWER_PROMPT | llm
        self._ensure_fulltext_index()

    def _ensure_fulltext_index(self) -> None:
        label = self._vault_label or "__Entity__"
        try:
            self.graph.query(
                f"""
                CREATE FULLTEXT INDEX {self._fulltext_index} IF NOT EXISTS
                FOR (n:{label}) ON EACH [n.display_name]
                """
            )
            logger.info("Full-text index '%s' ready.", self._fulltext_index)
        except Exception as exc:
            logger.warning("Could not create entity full-text index: %s", exc)

        try:
            self.graph.query(
                f"""
                CREATE FULLTEXT INDEX {self._doc_fulltext_index} IF NOT EXISTS
                FOR (d:Document) ON EACH [d.text]
                """
            )
            logger.info(
                "Document full-text index '%s' ready.", self._doc_fulltext_index
            )
        except Exception as exc:
            logger.warning("Could not create document full-text index: %s", exc)

    def run(self, query: str) -> tuple[list[str], str]:
        seed_ids = self._stage1_seed(query)
        all_entity_ids = self._stage2_traverse(seed_ids)
        contexts = self._stage3_collect_docs(all_entity_ids, query)
        response = self._stage4_generate(query, contexts)
        return contexts, response

    def _stage1_seed(self, query: str) -> list[str]:
        lucene_query = self._build_lucene_query(query)
        if not lucene_query:
            return []

        vault_filter = "AND node.vault = $vault" if self.vault else ""
        params: dict[str, Any] = {
            "q": lucene_query,
            "top_k_internal": self.top_k_seed * 10,
            "top_k": self.top_k_seed,
            "min_score": self.min_score,
        }
        if self.vault:
            params["vault"] = self.vault

        try:
            rows = self.graph.query(
                f"""
                CALL db.index.fulltext.queryNodes('{self._fulltext_index}', $q,
                    {{limit: $top_k_internal}})
                YIELD node, score
                WHERE score >= $min_score
                  AND node:__Entity__
                  {vault_filter}
                RETURN node.id AS id, score
                ORDER BY score DESC
                LIMIT $top_k
                """,
                params=params,
            )
        except Exception as exc:
            logger.warning("Full-text seed search failed: %s", exc)
            return []

        ids = [row["id"] for row in rows if row.get("id")]
        logger.debug("Stage 1: %d seed entities for query %r", len(ids), query[:60])
        return ids

    _STOPWORDS: frozenset[str] = frozenset(
        {
            "the",
            "and",
            "for",
            "are",
            "was",
            "were",
            "has",
            "had",
            "have",
            "his",
            "her",
            "its",
            "our",
            "their",
            "this",
            "that",
            "these",
            "those",
            "with",
            "from",
            "into",
            "upon",
            "what",
            "when",
            "who",
            "how",
            "did",
            "not",
            "but",
            "all",
            "can",
            "also",
            "than",
            "как",
            "что",
            "это",
            "был",
            "была",
            "были",
            "его",
            "её",
            "они",
            "все",
            "при",
            "для",
            "над",
            "под",
            "или",
            "уже",
            "еще",
            "ещё",
            "вот",
            "так",
            "где",
            "там",
            "тем",
            "тот",
            "та",
            "те",
        }
    )

    @classmethod
    def _build_lucene_query(cls, text: str) -> str:
        raw_tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9]{4,}", text)
        seen: set[str] = set()
        parts: list[str] = []
        for tok in raw_tokens:
            low = tok.lower()
            if low in cls._STOPWORDS or low in seen:
                continue
            seen.add(low)
            if len(tok) > 5:
                parts.append(f"{tok} OR {tok}~1")
            else:
                parts.append(tok)
        if not parts:
            return ""
        return " OR ".join(parts)

    def _stage2_traverse(self, seed_ids: list[str]) -> list[str]:
        if not seed_ids:
            return []

        vault_filter = "AND node.vault = $vault" if self.vault else ""
        params: dict[str, Any] = {
            "seeds": seed_ids,
            "hops": self.max_hops,
        }
        if self.vault:
            params["vault"] = self.vault

        try:
            rows = self.graph.query(
                f"""
                MATCH (seed:__Entity__)
                WHERE seed.id IN $seeds
                CALL apoc.path.subgraphNodes(seed, {{
                    maxLevel: $hops,
                    labelFilter: '+__Entity__'
                }}) YIELD node
                WHERE 1=1 {vault_filter}
                RETURN DISTINCT node.id AS id
                """,
                params=params,
            )
            ids = [row["id"] for row in rows if row.get("id")]
        except Exception:
            ids = self._traverse_plain_cypher(seed_ids)

        all_ids = list({*seed_ids, *ids})
        logger.debug(
            "Stage 2: %d total entities after %d-hop expansion",
            len(all_ids),
            self.max_hops,
        )
        return all_ids

    def _traverse_plain_cypher(self, seed_ids: list[str]) -> list[str]:
        """Variable-depth traversal without APOC."""
        vault_filter = "AND neighbor.vault = $vault" if self.vault else ""
        params: dict[str, Any] = {"seeds": seed_ids, "hops": self.max_hops}
        if self.vault:
            params["vault"] = self.vault

        try:
            rows = self.graph.query(
                f"""
                MATCH path = (seed:__Entity__)-[*1..$hops]-(neighbor:__Entity__)
                WHERE seed.id IN $seeds {vault_filter}
                AND NONE(r IN relationships(path) WHERE type(r) IN ['MENTIONS', 'NEXT_CHUNK'])
                RETURN DISTINCT neighbor.id AS id
                """,
                params=params,
            )
            return [row["id"] for row in rows if row.get("id")]
        except Exception as exc:
            logger.warning("Traversal fallback failed: %s", exc)
            return []

    def _stage3_collect_docs(self, entity_ids: list[str], query: str) -> list[str]:
        vault_filter = "AND d.vault = $vault" if self.vault else ""

        entity_texts: list[tuple[str, float]] = []
        if entity_ids:
            params: dict[str, Any] = {
                "ids": entity_ids,
                "limit": self.top_k_context * 2,
            }
            if self.vault:
                params["vault"] = self.vault
            try:
                rows = self.graph.query(
                    f"""
                    MATCH (d:Document)-[:MENTIONS]->(e:__Entity__)
                    WHERE e.id IN $ids {vault_filter}
                    WITH d, count(e) AS entity_match_count
                    ORDER BY entity_match_count DESC
                    LIMIT $limit
                    OPTIONAL MATCH (d)-[:NEXT_CHUNK]->(next:Document)
                    RETURN d.text + COALESCE('\n' + next.text, '') AS text,
                           entity_match_count AS score
                    """,
                    params=params,
                )
                entity_texts = [
                    (row["text"], float(row["score"]))
                    for row in rows
                    if row.get("text")
                ]
            except Exception as exc:
                logger.warning("Document retrieval (entity branch) failed: %s", exc)

        direct_texts = self._stage3_direct_doc_search(query)

        def _normalise(items: list[tuple[str, float]]) -> list[tuple[str, float]]:
            if not items:
                return []
            max_s = max(s for _, s in items) or 1.0
            return [(t, s / max_s) for t, s in items]

        entity_norm = _normalise(entity_texts)
        direct_norm = _normalise(direct_texts)

        combined: dict[str, float] = {}
        for text, score in entity_norm:
            key = text[:120]
            combined[key] = combined.get(key, 0.0) + score
        for text, score in direct_norm:
            key = text[:120]
            combined[key] = combined.get(key, 0.0) + score

        seen: set[str] = set()
        ordered: list[tuple[str, float]] = []
        all_texts = {t[:120]: t for t, _ in entity_norm + direct_norm}
        for key, score in sorted(combined.items(), key=lambda x: -x[1]):
            if key not in seen:
                seen.add(key)
                ordered.append((all_texts[key], score))

        texts = [t for t, _ in ordered[: self.top_k_context]]
        logger.debug(
            "Stage 3: %d hybrid contexts (%d entity, %d direct)",
            len(texts),
            len(entity_texts),
            len(direct_texts),
        )
        return texts

    def _stage3_direct_doc_search(self, query: str) -> list[tuple[str, float]]:
        lucene_query = self._build_lucene_query(query)
        if not lucene_query:
            return []

        vault_filter = "AND d.vault = $vault" if self.vault else ""
        params: dict[str, Any] = {
            "q": lucene_query,
            "limit": self.top_k_context * 2,
            "min_score": self.min_score,
        }
        if self.vault:
            params["vault"] = self.vault

        try:
            rows = self.graph.query(
                f"""
                CALL db.index.fulltext.queryNodes('{self._doc_fulltext_index}', $q,
                    {{limit: $limit}})
                YIELD node AS d, score
                WHERE score >= $min_score {vault_filter}
                OPTIONAL MATCH (d)-[:NEXT_CHUNK]->(next:Document)
                RETURN d.text + COALESCE('\n' + next.text, '') AS text, score
                ORDER BY score DESC
                LIMIT $limit
                """,
                params=params,
            )
            return [(row["text"], row["score"]) for row in rows if row.get("text")]
        except Exception as exc:
            logger.warning("Document fulltext search failed: %s", exc)
            return []

    def _stage4_generate(self, query: str, contexts: list[str]) -> str:
        if contexts:
            context_str = "\n\n---\n\n".join(contexts)
        else:
            context_str = "(no relevant context found in the knowledge graph)"

        result = self._chain.invoke({"context": context_str, "question": query})
        content = result.content if hasattr(result, "content") else result
        if isinstance(content, list):
            raw_output = "".join(
                c.get("text") or c.get("content") or str(c)
                if isinstance(c, dict)
                else (c.text if hasattr(c, "text") else str(c))
                for c in content
            )
        else:
            raw_output = str(content)

        reasoning_match = re.search(
            r"<reasoning>(.*?)</reasoning>", raw_output, re.DOTALL | re.IGNORECASE
        )
        if reasoning_match:
            logger.debug("LLM Reasoning: %s", reasoning_match.group(1).strip())

        if "<answer>" in raw_output.lower():
            answer = raw_output.split("<answer>")[-1].replace("</answer>", "").strip()
            return answer or raw_output

        logger.warning("LLM did not use <answer> tags. Returning raw output.")
        return re.sub(
            r"<reasoning>.*?</reasoning>", "", raw_output, flags=re.DOTALL
        ).strip()


def _load_reference_contexts(relevant_docs: list[str], vault_dir: Path) -> list[str]:
    contexts: list[str] = []
    for stem in relevant_docs:
        path = vault_dir / f"{stem}.md"
        if not path.exists():
            logger.warning("relevant_doc not found: %s", path)
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        text = re.sub(r"\A---\n.*?\n---\n?", "", text, flags=re.DOTALL).strip()
        text = re.sub(
            r"^\s*#{1,3}\s*(Related Connections|Connections|Links)\s*$.*\Z",
            "",
            text,
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        ).strip()
        if text:
            contexts.append(text)
    return contexts


async def _run_single_item(
    i: int,
    total: int,
    item: dict,
    pipeline: Neo4jGraphRAGPipeline,
    vault_dir: Path,
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
                if _is_transient_error(exc):
                    delay = _parse_retry_delay(
                        exc, default=_PIPELINE_BACKOFF_BASE**attempt
                    )
                    logger.warning(
                        "[%d/%d] attempt %d rate-limited, retrying in %.1fs",
                        i + 1,
                        total,
                        attempt + 1,
                        delay,
                    )
                else:
                    delay = _PIPELINE_BACKOFF_BASE**attempt
                    logger.warning(
                        "[%d/%d] attempt %d failed: %r, retrying in %.1fs",
                        i + 1,
                        total,
                        attempt + 1,
                        exc,
                        delay,
                    )
                await asyncio.sleep(delay)

    logger.info(
        "[%d/%d] %s\n  contexts: %d, response length: %d",
        i + 1,
        total,
        question[:80],
        len(contexts),
        len(response),
    )

    ref_contexts = _load_reference_contexts(relevant_docs, vault_dir)

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
    pipeline: Neo4jGraphRAGPipeline,
    pipeline_workers: int = 4,
) -> tuple[EvaluationDataset, list[dict]]:
    async def _run() -> tuple[EvaluationDataset, list[dict]]:
        semaphore = asyncio.Semaphore(pipeline_workers)
        tasks = [
            _run_single_item(i, len(qa_items), item, pipeline, vault_dir, semaphore)
            for i, item in enumerate(qa_items)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        ordered: list[tuple[int, SingleTurnSample, dict]] = []
        for res in results:
            if isinstance(res, Exception):
                logger.error("Item failed permanently: %r", res)
            else:
                ordered.append(res)

        ordered.sort(key=lambda x: x[0])
        samples = [s for _, s, _ in ordered]
        raw_results = [r for _, _, r in ordered]
        return EvaluationDataset(samples=samples), raw_results

    return asyncio.run(_run())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAGAS E2E evaluation of a Neo4j-backed GraphRAG pipeline."
    )

    parser.add_argument(
        "--vault",
        required=True,
        help="Path to vault directory (used for loading reference contexts).",
    )
    parser.add_argument(
        "--qa-file",
        required=True,
        help="Path to QA JSON dataset (list of {question, answer, relevant_docs}).",
    )
    parser.add_argument(
        "--output",
        default="results/ragas_neo4j.json",
        help="Output path for detailed per-sample results JSON.",
    )
    parser.add_argument(
        "--neo4j-url",
        default="bolt://localhost:7687",
        help="Neo4j Bolt URL (default: bolt://localhost:7687; overrides NEO4J_URL env).",
    )
    parser.add_argument(
        "--neo4j-user",
        default="neo4j",
        help="Neo4j username (default: neo4j; overrides NEO4J_USER env).",
    )

    parser.add_argument("--max-hops", type=int, default=DEFAULT_MAX_HOPS)
    parser.add_argument("--top-k-seed", type=int, default=DEFAULT_TOP_K_SEED)
    parser.add_argument("--top-k-context", type=int, default=DEFAULT_TOP_K_CONTEXT)
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="Minimum Lucene score for seed entity results (default: 0.0).",
    )
    parser.add_argument(
        "--pipeline-workers",
        type=int,
        default=4,
        help="Max parallel pipeline.run() calls during QA collection.",
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
    parser.add_argument(
        "--embedding-model",
        default=EMBEDDING_MODEL_NAME,
        help="HuggingFace embedding model for RAGAS AnswerCorrectness.",
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Max concurrent LLM calls during RAGAS evaluation (lower = fewer 429s).",
    )
    parser.add_argument(
        "--eval-timeout",
        type=int,
        default=6000,
        help="Timeout in seconds for a single RAGAS LLM call.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=100,
        help="Max retries per RAGAS LLM call.",
    )

    args = parser.parse_args()

    load_dotenv()

    if args.eval_provider:
        os.environ["EVAL_LLM_PROVIDER"] = args.eval_provider
    if args.eval_model:
        os.environ["EVAL_MODEL"] = args.eval_model

    vault_dir = Path(args.vault)
    qa_path = Path(args.qa_file)
    output_path = Path(args.output)

    if not vault_dir.is_dir():
        raise FileNotFoundError(f"Vault not found: {vault_dir}")
    if not (vault_dir / META_DIR_NAME).is_dir():
        logger.warning(
            "Vault '%s' has no .kg_builder directory — it may not be a valid vault",
            vault_dir,
        )
    if not qa_path.exists():
        raise FileNotFoundError(f"QA file not found: {qa_path}")

    with qa_path.open("r", encoding="utf-8") as f:
        qa_items: list[dict] = json.load(f)
    logger.info("Loaded %d QA items from %s", len(qa_items), qa_path)

    neo4j_url = args.neo4j_url or os.environ.get("NEO4J_URL", "bolt://localhost:7687")
    neo4j_user = args.neo4j_user or os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")
    if not neo4j_password:
        raise EnvironmentError(
            "NEO4J_PASSWORD is required (set in .env or environment)"
        )

    graph = Neo4jGraph(url=neo4j_url, username=neo4j_user, password=neo4j_password)
    logger.info("Connected to Neo4j at %s", neo4j_url)

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
    ragas_embeddings = _load_embeddings(args.embedding_model)

    pipeline = Neo4jGraphRAGPipeline(
        graph=graph,
        llm=pipeline_llm,
        vault=vault_dir.name,
        max_hops=args.max_hops,
        top_k_seed=args.top_k_seed,
        top_k_context=args.top_k_context,
        min_score=args.min_score,
    )
    logger.info(
        "Neo4jGraphRAGPipeline ready (vault=%s, max_hops=%d, top_k_seed=%d, top_k_context=%d, min_score=%s)",
        vault_dir.name,
        args.max_hops,
        args.top_k_seed,
        args.top_k_context,
        args.min_score,
    )

    dataset, raw_results = run_evaluation(
        vault_dir=vault_dir,
        qa_items=qa_items,
        pipeline=pipeline,
        pipeline_workers=args.pipeline_workers,
    )

    logger.info("Running RAGAS evaluation...")
    metrics = _build_metrics(ragas_llm, ragas_embeddings)
    run_config = RunConfig(
        timeout=args.eval_timeout,
        max_retries=args.max_retries,
        max_workers=args.max_concurrent,
    )
    result = evaluate(dataset=dataset, metrics=metrics, run_config=run_config)
    df = result.to_pandas()
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if not nan_cols.empty:
        logger.warning(
            "NaN values detected (failed samples per metric):\n%s",
            nan_cols.to_string(),
        )
    scores = df.mean(numeric_only=True).to_dict()
    scores["_evaluated_samples"] = int(
        df.shape[0] - nan_cols.max() if not nan_cols.empty else df.shape[0]
    )
    _print_results(scores, title="RAGAS Neo4j GraphRAG Evaluation Results")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "vault": str(vault_dir),
        "qa_file": str(qa_path),
        "config": {
            "max_hops": args.max_hops,
            "top_k_seed": args.top_k_seed,
            "top_k_context": args.top_k_context,
            "min_score": args.min_score,
        },
        "scores": scores,
        "samples": raw_results,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
