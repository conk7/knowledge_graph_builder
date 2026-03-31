import json
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from src.kg_builder.config import (
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    LINKS_CONFIG_FILE_NAME,
    META_DIR_NAME,
    RERANKER_MODEL_NAME,
    VECTOR_SEARCH_WEIGHT,
)
from src.kg_builder.models import RerankResult
from src.kg_builder.vault_manager import VaultManager
from src.kg_builder.vector_store import VectorStore

logger = logging.getLogger(__name__)

_LINK_RE = re.compile(r"-?\s*(.+?)\s*::\s*\[\[\s*(.+?)\s*\]\]")

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


@dataclass
class GraphRAGConfig:
    max_hops: int = 2
    beam_width: int = 3
    score_threshold: float = 0.2
    top_k_seed: int = 3
    top_k_context: int = 5
    ner_boost_factor: float = 1.5


@dataclass
class VaultConfig:
    lang: str = "en"
    embedding_model: str = EMBEDDING_MODEL_NAME
    reranker_model: str = RERANKER_MODEL_NAME
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    separators: list[str] = field(default_factory=lambda: list(CHUNK_SEPARATORS))
    vector_weight: float = VECTOR_SEARCH_WEIGHT

    @classmethod
    def from_dict(cls, cfg: dict) -> "VaultConfig":
        models = cfg.get("models", {})
        chunking = cfg.get("chunking", {})
        retrieval = cfg.get("retrieval", {})
        return cls(
            lang=cfg.get("lang", "en"),
            embedding_model=models.get("embedding", {}).get(
                "model_name", EMBEDDING_MODEL_NAME
            ),
            reranker_model=models.get("reranker", {}).get(
                "model_name", RERANKER_MODEL_NAME
            ),
            chunk_size=chunking.get("chunk_size", CHUNK_SIZE),
            chunk_overlap=chunking.get("chunk_overlap", CHUNK_OVERLAP),
            separators=chunking.get("separators", list(CHUNK_SEPARATORS)),
            vector_weight=retrieval.get("vector_search_weight", VECTOR_SEARCH_WEIGHT),
        )


class GraphRAGPipeline:
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
                    "Duplicate note name %r: %s vs %s (keeping first)",
                    stem,
                    self._name_to_path[stem],
                    md_file,
                )
            else:
                self._name_to_path[stem] = md_file

    @classmethod
    def from_vault(
        cls,
        vault_dir: Path,
        llm: Any,
        config: GraphRAGConfig | None = None,
    ) -> "GraphRAGPipeline":
        cfg = config or GraphRAGConfig()
        vault_cfg = VaultConfig.from_dict(_load_vault_config(vault_dir))

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
            splitter_type="sentence_window",
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
            "GraphRAGPipeline ready: %d vectors indexed from %s",
            vs.total_vectors,
            vault_dir.name,
        )

        return cls(
            vault_dir=vault_dir,
            vector_store=vs,
            vault_manager=vm,
            llm=llm,
            config=cfg,
            _tmp_dir=tmp_dir,
        )

    def __enter__(self) -> "GraphRAGPipeline":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self.vs.unload_reranker()
        if self._tmp_dir:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None

    def run(self, user_query: str) -> tuple[list[str], str]:
        visited, query_lemmas, query_entities = self._stage1_seed(user_query)
        self._stage2_traverse(user_query, visited, query_lemmas, query_entities)
        context_texts = self._stage3_prune(user_query, visited)
        response = self._stage4_generate(user_query, context_texts)
        return context_texts, response

    # ------------------------------------------------------------------
    # Stage 1: seed nodes via hybrid search
    # ------------------------------------------------------------------

    def _stage1_seed(self, query: str) -> tuple[dict[Path, str], set[str], set[str]]:
        nlp = self.vs._get_nlp()
        q_doc = nlp(query)
        query_lemmas = {
            t.lemma_.lower()
            for t in q_doc
            if t.is_alpha and not t.is_stop and not t.is_punct
        }
        query_entities = {ent.text for ent in q_doc.ents}

        seed_results = self.vs.search(query, k=self.cfg.top_k_seed)
        visited: dict[Path, str] = {}
        for r in seed_results:
            fp = Path(r.file_path)
            if fp not in visited:
                visited[fp] = self.vs.get_document_summary(r.file_path) or ""

        logger.debug(
            "Stage 1: %d seed nodes, lemmas=%s, entities=%s",
            len(visited),
            query_lemmas,
            query_entities,
        )
        return visited, query_lemmas, query_entities

    # ------------------------------------------------------------------
    # Stage 2: dynamic beam traversal
    # ------------------------------------------------------------------

    def _stage2_traverse(
        self,
        query: str,
        visited: dict[Path, str],
        query_lemmas: set[str],
        query_entities: set[str],
    ) -> None:
        """Follow outgoing KG links up to max_hops, pruning by relevance."""
        active_beams = list(visited.keys())

        for hop in range(self.cfg.max_hops):
            if not active_beams:
                break

            # 2.1 Collect all unvisited outgoing link targets.
            # _name_to_path supports nested vaults: [[Target]] resolves by stem,
            # not by assumed flat vault_dir/Target.md path.
            # candidates: (relation, target_stem, target_path, summary, lemmas)
            candidates: list[tuple[str, str, Path, str, str]] = []
            for beam_path in active_beams:
                content = self.vm.get_file_content(beam_path)
                _, links_section = self.vm._split_content_and_links(content)
                for m in _LINK_RE.finditer(links_section):
                    relation = m.group(1).strip()
                    target = m.group(2).strip()
                    target_path = self._name_to_path.get(target)
                    if target_path is None or target_path in visited:
                        continue

                    summary, summary_lemmas = self.vs.get_document_summary_with_lemmas(
                        str(target_path)
                    )
                    if not summary:
                        summary = target
                        summary_lemmas = target.lower()
                    candidates.append(
                        (relation, target, target_path, summary, summary_lemmas)
                    )

            if not candidates:
                break

            # 2.2 CPU pre-filter: keep candidates sharing at least one lemma with query.
            # Uses pre-fetched text_lemmatized from LanceDB — no spaCy call needed.
            filtered = [
                (rel, tgt, tp, sm, sl)
                for rel, tgt, tp, sm, sl in candidates
                if query_lemmas & set(sl.split())
            ]
            if not filtered:
                # Fallback: don't drop all candidates when lemma overlap is zero
                # (e.g. cross-language queries or very short summaries).
                filtered = candidates

            # 2.3-2.4 Cross-Encoder scoring — raw scores needed before NER boost
            pair_texts = [f"{rel}: {tgt}. {sm}" for rel, tgt, tp, sm, sl in filtered]
            raw_scores: list[float] = self.vs.reranker.predict(
                [(query, t) for t in pair_texts],
                show_progress_bar=False,
            ).tolist()

            # 2.5 NER boost: multiply score by ner_boost_factor if target name
            # matches a named entity extracted from the query.
            for i, (_, tgt, _, _, _) in enumerate(filtered):
                if any(ent.lower() in tgt.lower() for ent in query_entities):
                    raw_scores[i] *= self.cfg.ner_boost_factor

            # 2.6 Prune: keep top-beam_width above threshold, update active beams
            ranked = sorted(zip(raw_scores, filtered), key=lambda x: x[0], reverse=True)
            active_beams = []
            for score, (_, _, tp, sm, _) in ranked[: self.cfg.beam_width]:
                if score >= self.cfg.score_threshold:
                    visited[tp] = sm
                    active_beams.append(tp)

            logger.debug(
                "Stage 2 hop %d: %d candidates → %d beams",
                hop + 1,
                len(candidates),
                len(active_beams),
            )

    # ------------------------------------------------------------------
    # Stage 3: context pruning — chunk full content, rerank, keep top-N
    # ------------------------------------------------------------------

    def _stage3_prune(self, query: str, visited: dict[Path, str]) -> list[str]:
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
        logger.debug("Stage 3: %d total chunks → %d kept", len(all_chunks), len(top))
        return [r.text for r in top]

    # ------------------------------------------------------------------
    # Stage 4: LLM answer generation
    # ------------------------------------------------------------------

    def _stage4_generate(self, query: str, context_texts: list[str]) -> str:
        if not context_texts:
            context_str = "(no relevant context found in the knowledge graph)"
        else:
            context_str = "\n\n---\n\n".join(context_texts)

        result = self._chain.invoke({"context": context_str, "question": query})
        return result.content if hasattr(result, "content") else str(result)


def _load_vault_config(vault_dir: Path) -> dict:
    config_path = vault_dir / META_DIR_NAME / LINKS_CONFIG_FILE_NAME
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read vault config: %s", e)
    return {}
