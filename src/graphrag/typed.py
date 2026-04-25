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
from src.kg_builder.models import RerankResult
from src.kg_builder.vault_manager import VaultManager
from src.kg_builder.vector_store import VectorStore

logger = logging.getLogger(__name__)

_LINK_RE = re.compile(r"-?\s*(.+?)\s*::\s*\[\[\s*(.+?)\s*\]\]")

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

=== EXAMPLES OF YOUR EXPECTED BEHAVIOR ===

Example 1:
Context: 
- Заметка 'План': Заменить масло в машине. Сделать это до поездки к бабушке.
- Заметка 'Покупки': Купить моторное масло в Автомаге.
- Заметка 'Расписание': Поездка к бабушке в 14:00. В 10:00 заехать за кофе.

Question: Каков хронологический порядок задач, связанных с машиной?

Response:
<reasoning>
1. Language: Russian.
2. Logic: Buy oil -> Change oil -> Do it before 14:00.
</reasoning>
<answer>
Для подготовки машины необходимо сначала купить моторное масло в Автомаге, а затем произвести его замену. Обе эти задачи должны быть выполнены строго до 14:00, так как на это время запланирована поездка к бабушке.
</answer>
===========================================
Context:\n{context}\n\nQuestion: {question}""",
        ),
    ]
)


class TypedGraphRAGPipeline:
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
                    f"Duplicate note name {stem!r}: {self._name_to_path[stem]} vs {md_file} (keeping first)"
                )
            else:
                self._name_to_path[stem] = md_file

    @classmethod
    def from_vault(
        cls,
        vault_dir: Path,
        llm: Any,
        config: GraphRAGConfig | None = None,
        ignore_local_config: bool = False,
    ) -> "TypedGraphRAGPipeline":
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
            f"TypedGraphRAGPipeline ready: {vs.total_vectors} vectors indexed from {vault_dir.name}"
        )

        return cls(
            vault_dir=vault_dir,
            vector_store=vs,
            vault_manager=vm,
            llm=llm,
            config=cfg,
            _tmp_dir=tmp_dir,
        )

    def __enter__(self) -> "TypedGraphRAGPipeline":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self.vs.unload_reranker()
        if self._tmp_dir:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None

    def run(self, user_query: str) -> tuple[list[str], str]:
        visited = self._seed(user_query)
        self._traverse(user_query, visited)
        context_texts = self._collect(user_query, visited)
        response = self._generate(user_query, context_texts)
        return context_texts, response

    # ------------------------------------------------------------------
    # Stage 1: seed nodes via hybrid search (identical to BaseGraphRAGPipeline)
    # ------------------------------------------------------------------

    def _seed(self, query: str) -> dict[Path, str]:
        seed_results = self.vs.search(query, k=self.cfg.top_k_seed)
        visited: dict[Path, str] = {}
        for r in seed_results:
            fp = Path(r.file_path)
            if fp not in visited:
                visited[fp] = self.vs.get_document_summary(r.file_path) or ""

        logger.debug(f"Stage 1 (seed): {len(visited)} seed nodes")
        return visited

    # ------------------------------------------------------------------
    # Stage 2: typed scored BFS — cross-encoder uses relation type,
    #           all above threshold kept
    # ------------------------------------------------------------------

    def _traverse(self, query: str, visited: dict[Path, str]) -> None:
        active_beams = list(visited.keys())

        for hop in range(self.cfg.max_hops):
            if not active_beams:
                break

            candidates: list[tuple[str, str, Path, str]] = []
            for beam_path in active_beams:
                content = self.vm.get_file_content(beam_path)
                _, links_section = self.vm._split_content_and_links(content)
                for m in _LINK_RE.finditer(links_section):
                    relation = m.group(1).strip()
                    target = m.group(2).strip()
                    target_path = self._name_to_path.get(target)
                    if target_path is None or target_path in visited:
                        continue

                    summary = self.vs.get_document_summary(str(target_path)) or target
                    candidates.append((relation, target, target_path, summary))

            if not candidates:
                break

            pair_texts = [f"{rel}: {tgt}. {sm}" for rel, tgt, tp, sm in candidates]
            with self.vs._reranker_lock:
                raw_scores: list[float] = self.vs.reranker.predict(
                    [(query, t) for t in pair_texts],
                    show_progress_bar=False,
                ).tolist()

            active_beams = []
            for score, (_, _, tp, sm) in zip(raw_scores, candidates):
                if score >= self.cfg.score_threshold:
                    visited[tp] = sm
                    active_beams.append(tp)

            logger.debug(
                f"Stage 2 (typed BFS) hop {hop + 1}: {len(candidates)} candidates, "
                f"{len(active_beams)} above threshold"
            )

    # ------------------------------------------------------------------
    # Stage 3: context pruning (identical to BaseGraphRAGPipeline)
    # ------------------------------------------------------------------

    def _collect(self, query: str, visited: dict[Path, str]) -> list[str]:
        all_chunks: list[str] = []
        for node_path in visited:
            content = self.vm.get_file_content(node_path)
            body, _ = self.vm._split_content_and_links(content)
            all_chunks.extend(self.vs._get_splitter().split_text(body))

        if not all_chunks:
            return []

        top: list[RerankResult] = self.vs.rerank(
            query, all_chunks, top_k=self.cfg.top_k_context, threshold=0.0
        )
        logger.debug(
            f"Stage 3 (collect): {len(all_chunks)} total chunks, {len(top)} kept"
        )
        return [r.text for r in top]

    # ------------------------------------------------------------------
    # Stage 4: LLM answer generation (identical to BaseGraphRAGPipeline)
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
