import hashlib
import re
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple

from spacy.matcher import PhraseMatcher
from tqdm import tqdm

from .config import BROAD_QUERY_MODE_DEFAULT, BroadQueryMode
from .models import CandidatePair, DocumentEntity, NewlyAddedChunk
from .vector_store import VectorStore


class RetrievalStrategyMode(str, Enum):
    BROAD = "broad"
    STRICT = "strict"
    COMBINED = "combined"


class CandidateRetrievalStrategy(ABC):
    @abstractmethod
    def retrieve(
        self, chunks: List[NewlyAddedChunk], full_docs: Dict[str, str]
    ) -> Tuple[List[CandidatePair], List[Dict[str, Any]]]:
        pass


class VectorSearchRerankStrategy(CandidateRetrievalStrategy):
    def __init__(
        self,
        vector_store: VectorStore,
        retrieval_k: int,
        reranker_top_k: int,
        reranker_threshold: float,
        broad_query_mode: BroadQueryMode = BROAD_QUERY_MODE_DEFAULT,
    ):
        self.vector_store = vector_store
        self.retrieval_k = retrieval_k
        self.reranker_top_k = reranker_top_k
        self.reranker_threshold = reranker_threshold
        self.broad_query_mode = broad_query_mode

    def retrieve(
        self,
        chunks: List[NewlyAddedChunk],
        full_docs: Dict[str, str],
        show_progress: bool = True,
    ) -> Tuple[List[CandidatePair], List[Dict[str, Any]]]:
        all_final_candidates = []
        all_initial_meta = []

        if self.broad_query_mode == BroadQueryMode.TITLE_SUMMARY:
            source_items = list(
                {
                    chunk.file_path: self._build_title_summary_query(
                        full_docs.get(chunk.file_path, ""), chunk.file_path
                    )
                    for chunk in chunks
                }.items()
            )
            pbar = (
                tqdm(
                    source_items,
                    desc="Vector Retrieval & Reranking (title+summary)",
                    leave=False,
                )
                if show_progress
                else source_items
            )
            for file_path, query_text in pbar:
                cands, meta = self._retrieve_chunk(query_text, file_path)
                all_final_candidates.extend(cands)
                all_initial_meta.extend(meta)
            return all_final_candidates, all_initial_meta

        pbar = (
            tqdm(chunks, desc="Vector Retrieval & Reranking (chunks)", leave=False)
            if show_progress
            else chunks
        )
        for chunk in pbar:
            cands, meta = self._retrieve_chunk(chunk.content, chunk.file_path)
            all_final_candidates.extend(cands)
            all_initial_meta.extend(meta)
        return all_final_candidates, all_initial_meta

    def _build_title_summary_query(self, doc_content: str, file_path: str) -> str:
        title = Path(file_path).stem.replace("_", " ").strip()
        text = (doc_content or "").strip()
        if not text:
            return title

        first_block = text.split("\n\n", 1)[0].strip()
        if not first_block:
            return title

        if first_block.lstrip().startswith("#"):
            return first_block
        return f"{title}\n{first_block}" if title else first_block

    def _retrieve_chunk(
        self, chunk_content: str, chunk_file_path: str
    ) -> Tuple[List[CandidatePair], List[Dict[str, Any]]]:
        search_results = self.vector_store.search(chunk_content, self.retrieval_k)

        candidates_map = {}
        candidate_texts = []
        initial_candidates_meta = []

        for result in search_results:
            other_text = result.text
            other_file_path = result.file_path
            distance = result.distance

            if other_file_path == chunk_file_path:
                continue

            if other_text not in candidates_map:
                candidate_texts.append(other_text)
                candidates_map[other_text] = {
                    "file_path": other_file_path,
                    "vector_distance": distance,
                }

                initial_candidates_meta.append(
                    {
                        "source_path": chunk_file_path,
                        "source_content": chunk_content,
                        "target_path": other_file_path,
                        "target_content": other_text,
                        "vector_distance": distance,
                    }
                )

        if not candidate_texts:
            return [], initial_candidates_meta

        reranked_results = self.vector_store.rerank(
            query=chunk_content,
            candidates=candidate_texts,
            top_k=self.reranker_top_k,
            threshold=self.reranker_threshold,
        )

        final_candidates_map = {}
        for res in reranked_results:
            meta = candidates_map[res.text]
            fpath = meta["file_path"]
            if fpath not in final_candidates_map:
                final_candidates_map[fpath] = {
                    "chunks": [],
                    "best_vector_distance": meta.get("vector_distance", 0.0),
                    "best_rerank_score": float(res.score),
                }
            final_candidates_map[fpath]["chunks"].append(res.text)

        final_candidates = []
        initial_candidates_meta_grouped = []
        for fpath, data in final_candidates_map.items():
            combined_text = "\n...\n".join(data["chunks"])
            final_candidates.append(
                CandidatePair(
                    source_path=Path(chunk_file_path),
                    source_content=chunk_content,
                    target_path=Path(fpath),
                    target_content=combined_text,
                    vector_distance=data["best_vector_distance"],
                    reranker_score=data["best_rerank_score"],
                )
            )
            initial_candidates_meta_grouped.append(
                {
                    "source_path": chunk_file_path,
                    "source_content": chunk_content,
                    "target_path": fpath,
                    "target_content": combined_text,
                    "vector_distance": data["best_vector_distance"],
                }
            )

        return final_candidates, initial_candidates_meta_grouped


class StrictRetrievalStrategy(CandidateRetrievalStrategy):
    def __init__(
        self,
        vector_store: VectorStore,
        global_entity_dict: Dict[str, DocumentEntity],
        context_sents_before: int = 1,
        context_sents_after: int = 1,
    ):
        self.vector_store = vector_store
        self.global_entity_dict = global_entity_dict
        self.context_sents_before = context_sents_before
        self.context_sents_after = context_sents_after

        self.nlp = self.vector_store._get_nlp()

        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LEMMA")
        self._build_matcher_patterns()

    def _build_matcher_patterns(self):
        alias_to_paths = {}
        all_cleaned_aliases = []

        for target_path, entity in self.global_entity_dict.items():
            raw_aliases = entity.all_names or [Path(target_path).stem]
            for a in raw_aliases:
                if not a:
                    continue

                cleaned = re.sub(r"[_\\-]", " ", a).strip().lower()
                if cleaned:
                    if cleaned not in alias_to_paths:
                        alias_to_paths[cleaned] = []
                        all_cleaned_aliases.append(cleaned)
                    alias_to_paths[cleaned].append(target_path)

        docs = list(self.nlp.pipe(all_cleaned_aliases, batch_size=1000))

        patterns_by_path = {}
        for text, doc in zip(all_cleaned_aliases, docs):
            for target_path in alias_to_paths[text]:
                if target_path not in patterns_by_path:
                    patterns_by_path[target_path] = []
                patterns_by_path[target_path].append(doc)

        for target_path, patterns in patterns_by_path.items():
            self.matcher.add(target_path, patterns)

    def retrieve(
        self,
        chunks: List[NewlyAddedChunk],
        full_docs: Dict[str, str],
        show_progress: bool = True,
    ) -> Tuple[List[CandidatePair], List[Dict[str, Any]]]:
        all_final_candidates = []
        all_initial_meta = []

        pbar = (
            tqdm(full_docs.items(), desc="Strict Linking (NLP)", leave=False)
            if show_progress
            else full_docs.items()
        )
        for doc_path, doc_content in pbar:
            if not doc_content:
                continue
            cands, meta = self._retrieve_doc(doc_content, doc_path)
            all_final_candidates.extend(cands)
            all_initial_meta.extend(meta)
        return all_final_candidates, all_initial_meta

    def _retrieve_doc(
        self, content: str, file_path: str
    ) -> Tuple[List[CandidatePair], List[Dict[str, Any]]]:
        final_candidates = []
        initial_candidates_meta = []

        doc = self.nlp(content)
        matches = self.matcher(doc)

        if not matches:
            return [], []

        sents = list(doc.sents)
        seen_targets = set()
        for match_id, start, end in matches:
            target_path = self.nlp.vocab.strings[match_id]

            if target_path == file_path or target_path in seen_targets:
                continue

            seen_targets.add(target_path)

            matched_span = doc[start:end]
            target_sent = matched_span.sent

            try:
                sent_idx = sents.index(target_sent)
                start_idx = max(0, sent_idx - self.context_sents_before)
                end_idx = min(len(sents), sent_idx + self.context_sents_after + 1)
                window_sents = sents[start_idx:end_idx]
                combined_context = " ".join(s.text.strip() for s in window_sents)
            except ValueError:
                combined_context = target_sent.text.strip()

            final_candidates.append(
                CandidatePair(
                    source_path=Path(file_path),
                    source_content=content,
                    target_path=Path(target_path),
                    target_content=combined_context,
                    vector_distance=0.0,
                    reranker_score=1.0,
                )
            )

            initial_candidates_meta.append(
                {
                    "source_path": file_path,
                    "source_content": content,
                    "target_path": target_path,
                    "target_content": combined_context,
                    "vector_distance": 0.0,
                }
            )

        return final_candidates, initial_candidates_meta


class CombinedRetrievalStrategy(CandidateRetrievalStrategy):
    def __init__(
        self,
        strict_strat: StrictRetrievalStrategy,
        broad_strat: VectorSearchRerankStrategy,
    ):
        self.strict_strat = strict_strat
        self.broad_strat = broad_strat

    def retrieve(
        self,
        chunks: List[NewlyAddedChunk],
        full_docs: Dict[str, str],
        show_progress: bool = True,
    ) -> Tuple[List[CandidatePair], List[Dict[str, Any]]]:
        strict_cands, strict_meta = self.strict_strat.retrieve(
            chunks, full_docs, show_progress=show_progress
        )
        broad_cands, broad_meta = self.broad_strat.retrieve(
            chunks, full_docs, show_progress=show_progress
        )

        def _pair_hash(c: CandidatePair) -> str:
            key = f"{c.source_path}|{c.target_path}|{c.target_content}"
            return hashlib.md5(key.encode("utf-8")).hexdigest()

        merged_cands = list(strict_cands)
        merged_meta = list(strict_meta)
        seen_hashes = {_pair_hash(c) for c in strict_cands}

        for i, c in enumerate(broad_cands):
            h = _pair_hash(c)
            if h not in seen_hashes:
                merged_cands.append(c)
                merged_meta.append(broad_meta[i])
                seen_hashes.add(h)

        return merged_cands, merged_meta
