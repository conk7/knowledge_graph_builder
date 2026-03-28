import argparse
import hashlib
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from src.kg_builder.config import (
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    INITIAL_RETRIEVAL_K,
    LINK_HEADER,
    LINKS_CONFIG_FILE_NAME,
    META_DIR_NAME,
    RERANKER_MODEL_NAME,
    RERANKER_THRESHOLD,
    RERANKER_TOP_K,
    VECTOR_SEARCH_WEIGHT,
)
from src.kg_builder.models import CandidatePair, DocumentEntity, NewlyAddedChunk
from src.kg_builder.retrieval import (
    StrictRetrievalStrategy,
    VectorSearchRerankStrategy,
)
from src.kg_builder.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _content_hash(text: str) -> str:
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def merge_combined_results(
    strict_result: Tuple[List[CandidatePair], List[Dict[str, Any]]],
    broad_result: Tuple[List[CandidatePair], List[Dict[str, Any]]],
) -> Tuple[List[CandidatePair], List[Dict[str, Any]]]:
    strict_cands, strict_meta = strict_result
    broad_cands, broad_meta = broad_result

    merged_cands = list(strict_cands)
    merged_meta = list(strict_meta)

    def _pair_hash(c: CandidatePair) -> str:
        return _content_hash(f"{c.source_path}|{c.target_path}|{c.target_content}")

    seen_hashes = {_pair_hash(c) for c in strict_cands}

    for i, cand in enumerate(broad_cands):
        h = _pair_hash(cand)
        if h not in seen_hashes:
            merged_cands.append(cand)
            merged_meta.append(broad_meta[i])
            seen_hashes.add(h)

    return merged_cands, merged_meta


class Metrics:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0


_nlp_cache: Dict[str, Any] = {}

FUZZY_MATCH_THRESHOLD = 0.5


def get_sample_lang(sample_dir: Path) -> str:
    """Read lang from .kg_builder/config.json, or return default."""
    config_path = sample_dir / META_DIR_NAME / LINKS_CONFIG_FILE_NAME
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("lang")
        except (json.JSONDecodeError, IOError):
            pass


def _get_nlp(lang: str):
    if lang not in _nlp_cache:
        import spacy

        model = "ru_core_news_sm" if lang == "ru" else "en_core_web_sm"
        _nlp_cache[lang] = spacy.load(model, disable=["parser", "ner"])
    return _nlp_cache[lang]


def _lemma_set(text: str, nlp) -> set:
    return {
        token.lemma_.lower()
        for token in nlp(text)
        if not token.is_stop and not token.is_punct and token.is_alpha
    }


def is_fuzzy_match(
    gold_sent: str, pred_content: str, nlp, threshold: float = FUZZY_MATCH_THRESHOLD
) -> bool:
    gold_lemmas = _lemma_set(gold_sent, nlp)
    pred_lemmas = _lemma_set(pred_content, nlp)
    if not gold_lemmas:
        return gold_sent.strip() in pred_content
    intersection = gold_lemmas & pred_lemmas
    return len(intersection) / len(gold_lemmas) >= threshold


def get_entities_from_dir(directory: Path) -> Dict[str, DocumentEntity]:
    entities = {}
    for md_file in directory.glob("*.md"):
        rel_path = str(md_file)
        title = md_file.stem
        entities[rel_path] = DocumentEntity(rel_path=rel_path, title=title, aliases=[])
    return entities


def evaluate_sample(
    sample_dir: Path,
    gold_data: List[Dict[str, Any]],
    lang: str,
    show_inner_progress: bool = False,
) -> Dict[str, Metrics]:
    entities = get_entities_from_dir(sample_dir)
    if not entities:
        return {}

    full_docs = {}
    for file_path_str in entities.keys():
        file_path = Path(file_path_str)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        if LINK_HEADER in content:
            content = content[: content.index(LINK_HEADER)]
        full_docs[file_path_str] = content

    temp_dir = tempfile.mkdtemp()
    try:
        vs = VectorStore(
            index_path=Path(temp_dir),
            embedding_model_name=EMBEDDING_MODEL_NAME,
            reranker_model_name=RERANKER_MODEL_NAME,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            splitter_type="sentence_window",
            separators=CHUNK_SEPARATORS,
            vector_weight=VECTOR_SEARCH_WEIGHT,
            fresh_start=True,
            lang=lang,
        )

        all_chunks = []
        for fp, content in full_docs.items():
            chunks = vs.add_document(fp, content)
            for c in chunks:
                all_chunks.append(NewlyAddedChunk(file_path=fp, content=c))

        vs.rebuild_fts_index()

        strict_strat = StrictRetrievalStrategy(
            vector_store=vs,
            global_entity_dict=entities,
            context_sents_before=0,
            context_sents_after=0,
        )
        broad_strat = VectorSearchRerankStrategy(
            vector_store=vs,
            retrieval_k=INITIAL_RETRIEVAL_K,
            reranker_top_k=RERANKER_TOP_K,
            reranker_threshold=RERANKER_THRESHOLD,
        )
        strategies = {
            "Strict": strict_strat,
            "Broad": broad_strat,
        }

        vs.load_reranker()
        try:
            results = {}
            retrieval_cache = {}
            for name, strat in strategies.items():
                logger.info(f"Evaluating {name}")
                retrieval_cache[name] = strat.retrieve(
                    all_chunks, full_docs, show_progress=show_inner_progress
                )

            logger.info("Evaluating Combined")
            retrieval_cache["Combined"] = merge_combined_results(
                retrieval_cache["Strict"], retrieval_cache["Broad"]
            )

            nlp = _get_nlp(lang)

            for name in ("Strict", "Broad", "Combined"):
                metrics = Metrics()
                pred_candidates, _ = retrieval_cache[name]

                pred_map = {}
                for cand in pred_candidates:
                    key = (cand.source_path.name, cand.target_path.stem)
                    if key not in pred_map:
                        pred_map[key] = []
                    pred_map[key].append(cand.target_content)

                for gold_item in gold_data:
                    gold_key = (gold_item["source_file"], gold_item["entity"])
                    gold_sent = gold_item["sentence"].strip()

                    if gold_key in pred_map:
                        found = any(
                            is_fuzzy_match(gold_sent, pred_content, nlp)
                            for pred_content in pred_map[gold_key]
                        )
                        if found:
                            metrics.tp += 1
                        else:
                            metrics.fn += 1
                    else:
                        metrics.fn += 1

                for key, contents in pred_map.items():
                    for content in contents:
                        match_found = any(
                            key == (gold_item["source_file"], gold_item["entity"])
                            and is_fuzzy_match(
                                gold_item["sentence"].strip(), content, nlp
                            )
                            for gold_item in gold_data
                        )
                        if not match_found:
                            metrics.fp += 1

                results[name] = metrics

            return results, retrieval_cache
        finally:
            vs.unload_reranker()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Validate retrieval strategies performance."
    )
    parser.add_argument(
        "--vault",
        required=True,
        help="Path to the directory containing samples (subdirs).",
    )
    parser.add_argument(
        "--gold-dir",
        required=True,
        help="Path to the directory containing gold JSON files.",
    )
    parser.add_argument(
        "--show-inner-progress",
        action="store_true",
        help="Show nested progress bars for retrieval strategies.",
    )
    parser.add_argument(
        "--output-dir",
        help="Path to save evaluation results JSONs.",
    )
    args = parser.parse_args()

    vault_path = Path(args.vault)
    gold_dir = Path(args.gold_dir)

    if not vault_path.is_dir():
        logger.error(f"Vault directory not found: {vault_path}")
        return
    if not gold_dir.is_dir():
        logger.error(f"Gold directory not found: {gold_dir}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("Strict", "Broad", "Combined"):
            (output_dir / name).mkdir(parents=True, exist_ok=True)

    subdirs = [
        d for d in vault_path.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]
    if not subdirs:
        logger.warning(
            f"No subdirectories found in {vault_path}. Treating as single sample."
        )
        subdirs = [vault_path]

    total_metrics = {
        "Strict": Metrics(),
        "Broad": Metrics(),
        "Combined": Metrics(),
    }

    print(f"Starting evaluation of {len(subdirs)} samples...")
    for subdir in tqdm(sorted(subdirs), desc="Evaluating samples"):
        gold_file = gold_dir / f"{subdir.name}.json"
        if not gold_file.exists():
            continue

        with open(gold_file, "r", encoding="utf-8") as f:
            gold_data = json.load(f)

        sample_lang = get_sample_lang(subdir)
        sample_results, retrieval_data = evaluate_sample(
            subdir,
            gold_data,
            sample_lang,
            show_inner_progress=args.show_inner_progress,
        )

        if output_dir:
            for name, (pred_candidates, _) in retrieval_data.items():
                output_items = []
                for cand in pred_candidates:
                    output_items.append(
                        {
                            "entity": cand.target_path.stem,
                            "source_file": cand.source_path.name,
                            "sentence": cand.target_content,
                        }
                    )

                output_file = output_dir / name / f"{subdir.name}.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(output_items, f, ensure_ascii=False, indent=2)

        for name, metrics in sample_results.items():
            total_metrics[name].tp += metrics.tp
            total_metrics[name].fp += metrics.fp
            total_metrics[name].fn += metrics.fn

    print("\n" + "=" * 65)
    print(
        f"{'Strategy':<15} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}"
    )
    print("-" * 65)
    for name, m in total_metrics.items():
        print(
            f"{name:<15} | {m.tp:<5} | {m.fp:<5} | {m.fn:<5} | {m.precision:10.4f} | {m.recall:10.4f} | {m.f1:10.4f}"
        )
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
