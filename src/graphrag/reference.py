"""
ReferenceGraphRAGPipeline — faithful re-implementation of ObsidianRAG
(https://github.com/Vasallo94/ObsidianRAG) built on top of the project's
own VectorStore / VaultManager infrastructure.

Pipeline (4 stages):
  1. Hybrid search (BM25 + vector) → top-INITIAL_K chunks
  2. CrossEncoder rerank → top-RERANK_TOP_K with score ≥ RERANK_THRESHOLD
  3. Single-hop [[wikilink]] expansion → up to LINK_EXPAND_K linked notes
     (linked-note chunks receive a LINK_SCORE_MULT score penalty)
  4. Combine & sort all chunks by score → LLM generation
"""

import json
import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from src.graphrag.config import GraphRAGConfig, VaultConfig
from src.kg_builder.config import LINKS_CONFIG_FILE_NAME, META_DIR_NAME, SPLITTER_TYPE
from src.kg_builder.vault_manager import VaultManager
from src.kg_builder.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Matches [[Note]] and [[Note|Alias]] — plain Obsidian wikilinks.
# Also captures typed links of the form `- rel:: [[Target]]` because the
# target `[[Target]]` sub-pattern is present inside those strings too.
_WIKILINK_RE = re.compile(r"\[\[\s*([^|\]\n]+?)(?:\|[^\]]+)?\s*\]\]")

# _ANSWER_PROMPT = ChatPromptTemplate.from_messages(
#     [
#         (
#             "human",
#             """You are an expert analytical assistant. Answer the user's question based strictly on the provided knowledge graph context.

# CRITICAL INSTRUCTIONS:
# 1. Language Mirroring (ABSOLUTE): Write your <answer> in the EXACT SAME LANGUAGE as the user's Question.
# 2. Logic First: Inside your <reasoning> tag, state the language of the Question, then plan your answer.
# 3. Style and Tone (CRUCIAL FOR METRICS): Write your <answer> as a cohesive, flowing paragraph. Do NOT use markdown bullet points or numbered lists unless absolutely unavoidable. Synthesize the facts naturally.
# 4. Concise Accuracy: Directly answer the core question. Include necessary specific names and numbers from the context, but do NOT add extra historical background or broad summaries that weren't directly requested.
# 5. Output Format: Use <reasoning> for your internal logic and <answer> for the final response.

# === EXAMPLES OF YOUR EXPECTED BEHAVIOR ===

# Example 1:
# Context:
# - Заметка 'План': Заменить масло в машине. Сделать это до поездки к бабушке.
# - Заметка 'Покупки': Купить моторное масло в Автомаге.
# - Заметка 'Расписание': Поездка к бабушке в 14:00. В 10:00 заехать за кофе.

# Question: Каков хронологический порядок задач, связанных с машиной?

# Response:
# <reasoning>
# 1. Language: Russian.
# 2. Logic: Buy oil -> Change oil -> Do it before 14:00.
# </reasoning>
# <answer>
# Для подготовки машины необходимо сначала купить моторное масло в Автомаге, а затем произвести его замену. Обе эти задачи должны быть выполнены строго до 14:00, так как на это время запланирована поездка к бабушке.
# </answer>
# ===========================================
# Context:\n{context}\n\nQuestion: {question}""",
#         ),
#     ]
# )

_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "You are a helpful assistant that answers questions "
            "based strictly on the provided knowledge graph context.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        ),
    ]
)

# ── ObsidianRAG hyper-parameters ──────────────────────────────────────────────
_RERANK_THRESHOLD = 0.3  # minimum CrossEncoder score to keep a chunk
_LINK_EXPAND_K = 5  # maximum number of linked notes to expand (one hop)
_LINK_SCORE_MULT = 0.9  # score multiplier applied to linked-note chunks
# ─────────────────────────────────────────────────────────────────────────────


class ReferenceGraphRAGPipeline:
    """
    Single-hop GraphRAG that closely follows the ObsidianRAG reference design.

    Key differences from BaseGraphRAGPipeline / TypedGraphRAGPipeline:
      • CrossEncoder reranking is applied *before* graph expansion (not after).
      • Graph expansion is limited to a *single hop* over plain [[wikilinks]].
      • Relation types are *ignored* — only wikilink targets are followed.
      • Expanded notes are scored with a penalty (× LINK_SCORE_MULT) and
        merged with the reranked seed chunks; no second reranking pass.
    """

    def __init__(
        self,
        vault_dir: Path,
        vector_store: VectorStore,
        vault_manager: VaultManager,
        llm: Any,
        config: GraphRAGConfig | None = None,
        _tmp_dir: str | None = None,
    ) -> None:
        self.vault_dir = vault_dir
        self.vs = vector_store
        self.vm = vault_manager
        self.cfg = config or GraphRAGConfig()
        self._tmp_dir = _tmp_dir
        self._chain = _ANSWER_PROMPT | llm

        self._name_to_path: dict[str, Path] = {}
        for md_file in vault_manager.scan_markdown_files():
            stem = md_file.stem
            if stem in self._name_to_path:
                logger.warning(
                    f"Duplicate note name {stem!r}: "
                    f"{self._name_to_path[stem]} vs {md_file} (keeping first)"
                )
            else:
                self._name_to_path[stem] = md_file

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_vault(
        cls,
        vault_dir: Path,
        llm: Any,
        config: GraphRAGConfig | None = None,
        ignore_local_config: bool = False,
    ) -> "ReferenceGraphRAGPipeline":
        cfg = config or GraphRAGConfig()
        raw_vault_cfg = _load_vault_config(vault_dir)
        if ignore_local_config:
            raw_vault_cfg = {"lang": raw_vault_cfg.get("lang", "en")}
        vault_cfg = VaultConfig.from_dict(raw_vault_cfg)

        tmp_dir = tempfile.mkdtemp()
        vs = VectorStore(
            index_path=Path(tmp_dir),
            embedding_model_name=vault_cfg.embedding_model,
            reranker_model_name=vault_cfg.reranker_model,
            chunk_size=vault_cfg.chunk_size,
            chunk_overlap=vault_cfg.chunk_overlap,
            separators=vault_cfg.separators,
            vector_weight=vault_cfg.vector_weight,
            fresh_start=True,
            lang=vault_cfg.lang,
            splitter_type=SPLITTER_TYPE,
        )

        vm = VaultManager(
            vault_path=vault_dir,
            ignored_dirs=[vault_dir / META_DIR_NAME],
        )

        for md_file in vm.scan_markdown_files():
            content = vm.get_file_content(md_file)
            body, _ = vm._split_content_and_links(content)
            if body.strip():
                vs.add_document(str(md_file), body)

        vs.rebuild_fts_index()
        vs.load_reranker()

        logger.info(
            f"ReferenceGraphRAGPipeline ready: "
            f"{vs.total_vectors} vectors indexed from {vault_dir.name}"
        )

        return cls(
            vault_dir=vault_dir,
            vector_store=vs,
            vault_manager=vm,
            llm=llm,
            config=cfg,
            _tmp_dir=tmp_dir,
        )

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "ReferenceGraphRAGPipeline":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self.vs.unload_reranker()
        if self._tmp_dir:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_query: str) -> tuple[list[str], str]:
        reranked = self._seed_and_rerank(user_query)
        expanded = self._expand_links(reranked)
        context_texts = self._collect(reranked, expanded)
        response = self._generate(user_query, context_texts)
        return context_texts, response

    # ------------------------------------------------------------------
    # Stage 1+2: hybrid search → CrossEncoder rerank
    # ------------------------------------------------------------------

    def _seed_and_rerank(self, query: str) -> list[tuple[str, float, Path]]:
        """
        Returns up to RERANK_TOP_K tuples of (chunk_text, score, file_path)
        after CrossEncoder reranking of the initial hybrid-search results.
        Mirrors ObsidianRAG's EnsembleRetriever + CrossEncoder stage.
        """
        search_results = self.vs.search(query, k=self.cfg.top_k_seed)
        if not search_results:
            logger.debug("Stage 1 (seed): no results from hybrid search")
            return []

        # Build a text → file_path map so we can recover paths after reranking.
        # If two chunks happen to share the same text, the later one wins —
        # acceptable because we only need a valid path per unique text.
        text_to_path: dict[str, Path] = {
            r.text: Path(r.file_path) for r in search_results
        }
        chunks = [r.text for r in search_results]

        reranked = self.vs.rerank(
            query, chunks, top_k=self.cfg.top_k_context, threshold=_RERANK_THRESHOLD
        )

        result = [(r.text, r.score, text_to_path[r.text]) for r in reranked]
        logger.debug(
            f"Stage 1+2 (seed+rerank): {len(search_results)} retrieved, "
            f"{len(result)} kept after reranking (threshold={_RERANK_THRESHOLD})"
        )
        return result

    # ------------------------------------------------------------------
    # Stage 3: single-hop [[wikilink]] expansion
    # ------------------------------------------------------------------

    def _expand_links(
        self, reranked: list[tuple[str, float, Path]]
    ) -> list[tuple[str, float]]:
        """
        For each unique file in the reranked set, extract all [[wikilinks]]
        from the file's full content and fetch the first chunk of each
        linked note (up to LINK_EXPAND_K total).

        Linked-note chunks receive a score penalty (× LINK_SCORE_MULT)
        to reflect lower confidence, mirroring ObsidianRAG's 0.9× multiplier.
        """
        if not reranked:
            return []

        # Files already in the result set — do not re-add them.
        seen_paths: set[Path] = {fp for _, _, fp in reranked}

        expanded: list[tuple[str, float]] = []

        for _, seed_score, file_path in reranked:
            if len(expanded) >= _LINK_EXPAND_K:
                break

            content = self.vm.get_file_content(file_path)
            for m in _WIKILINK_RE.finditer(content):
                if len(expanded) >= _LINK_EXPAND_K:
                    break

                target_name = m.group(1).strip()
                target_path = self._name_to_path.get(target_name)
                if target_path is None or target_path in seen_paths:
                    continue

                summary = self.vs.get_document_summary(str(target_path))
                if not summary:
                    continue

                seen_paths.add(target_path)
                linked_score = seed_score * _LINK_SCORE_MULT
                expanded.append((summary, linked_score))
                logger.debug(
                    f"  → expanded link: {target_name!r} (score={linked_score:.3f})"
                )

        logger.debug(f"Stage 3 (expand): {len(expanded)} linked notes added")
        return expanded

    # ------------------------------------------------------------------
    # Stage 4: merge + sort → build context string list
    # ------------------------------------------------------------------

    def _collect(
        self,
        reranked: list[tuple[str, float, Path]],
        expanded: list[tuple[str, float]],
    ) -> list[str]:
        """
        Merge reranked seed chunks and linked-note chunks, sort by score
        (descending), and return ordered text list for the LLM.
        """
        all_scored: list[tuple[str, float]] = [
            (text, score) for text, score, _ in reranked
        ] + expanded
        all_scored.sort(key=lambda x: x[1], reverse=True)

        texts = [text for text, _ in all_scored]
        logger.debug(
            f"Stage 4 (collect): {len(reranked)} seed chunks + "
            f"{len(expanded)} linked chunks = {len(texts)} total"
        )
        return texts

    # ------------------------------------------------------------------
    # Stage 5: LLM answer generation
    # ------------------------------------------------------------------

    def _generate(self, query: str, context_texts: list[str]) -> str:
        if not context_texts:
            context_str = "(no relevant context found in the knowledge graph)"
        else:
            context_str = "\n\n---\n\n".join(context_texts)

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
            logger.debug(f"LLM Reasoning: {reasoning_match.group(1).strip()}")

        if "<answer>" in raw_output.lower():
            final_answer = (
                raw_output.split("<answer>")[-1].replace("</answer>", "").strip()
            )
            if not final_answer:
                logger.warning("Empty <answer> block from LLM.")
            return final_answer

        logger.warning("LLM did not use <answer> tags. Returning raw output.")
        return re.sub(
            r"<reasoning>.*?</reasoning>", "", raw_output, flags=re.DOTALL
        ).strip()


def _load_vault_config(vault_dir: Path) -> dict:
    config_path = vault_dir / META_DIR_NAME / LINKS_CONFIG_FILE_NAME
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read vault config: {e}")
    return {}
